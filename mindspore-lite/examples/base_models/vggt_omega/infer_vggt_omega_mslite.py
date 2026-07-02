# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MindSpore Lite (Ascend) inference for VGGT-Omega.

Loads the converted MindIR, preprocesses images to the fixed export shape with
pure numpy/PIL (no torch), runs a single feed-forward pass on Ascend and decodes
the camera pose and dense depth. ``--compare-onnx`` runs the ONNX model on the
same input with ONNX Runtime and reports cosine similarity (must be > 0.99).

This script deliberately does NOT import torch.
"""

import argparse
import glob
import os
import time

import numpy as np
from PIL import Image

try:
    import mindspore_lite as msl
except ImportError as exc:  # pragma: no cover
    raise SystemExit("mindspore_lite is required: pip install mindspore-lite") from exc

_DEFAULT_SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
_PATCH_SIZE = 16


def _collect_image_paths(inputs):
    """Collect image paths from a list of files/directories/globs."""
    if not inputs:
        files = sorted(glob.glob(os.path.join(_DEFAULT_SAMPLES, "*.jpg")))
        if not files:
            raise FileNotFoundError(f"No images found in {_DEFAULT_SAMPLES}")
        return files
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.jpg"))))
            paths.extend(sorted(glob.glob(os.path.join(item, "*.png"))))
        elif os.path.isfile(item):
            paths.append(item)
        else:
            paths.extend(sorted(glob.glob(item)))
    if not paths:
        raise FileNotFoundError(f"No images found for {inputs}")
    return paths


def preprocess(image_paths, num_frames, img_h, img_w):
    """Load images and resize to the fixed ``[num_frames, 3, img_h, img_w]`` shape.

    Images are kept in the ``[0, 1]`` range; the model applies ResNet mean/std
    normalization internally. If fewer images than ``num_frames`` are provided,
    the last image is repeated to fill the sequence.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if img_h % _PATCH_SIZE or img_w % _PATCH_SIZE:
        raise ValueError(f"Image size must be a multiple of {_PATCH_SIZE}")

    frames = []
    for path in image_paths[:num_frames]:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((img_w, img_h), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            frames.append(arr.transpose(2, 0, 1))
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


def quat_to_mat(quaternions):
    """Convert (..., 4) scalar-last quaternions to (..., 3, 3) rotation matrices."""
    i, j, k, r = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def encoding_to_camera(pose_enc, img_h, img_w):
    """Decode VGGT-Omega 9D pose encoding into extrinsics and intrinsics."""
    t = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    rot = quat_to_mat(quat)
    extrinsics = np.concatenate([rot, t[..., None]], axis=-1)
    fy = (img_h / 2.0) / np.tan(fov_h / 2.0)
    fx = (img_w / 2.0) / np.tan(fov_w / 2.0)
    intrinsics = np.zeros(pose_enc.shape[:-1] + (3, 3), dtype=np.float32)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = img_w / 2.0
    intrinsics[..., 1, 2] = img_h / 2.0
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def _build_mslite_inputs(model, feed_dict):
    """Build the model input list by matching tensor names, then by order."""
    inputs = model.get_inputs()
    matched = [None] * len(inputs)
    for idx, msl_in in enumerate(inputs):
        arr = feed_dict.get(msl_in.name)
        if arr is None:
            continue
        msl_in.set_data_from_numpy(arr.astype(np.float32))
        matched[idx] = True
    if all(m is None for m in matched):
        ordered = list(feed_dict.values())
        for idx, msl_in in enumerate(inputs):
            msl_in.set_data_from_numpy(ordered[idx].astype(np.float32))
            matched[idx] = True
    if any(m is None for m in matched):
        raise ValueError(f"Could not match all model inputs: {[i.name for i in inputs]}")
    return inputs


def _cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_mslite_model(mindir_dir, device, device_id):
    """Build a MindSpore Lite model from the graph + variables directory."""
    graph_path = os.path.join(mindir_dir, "vggt_omega_mindir_graph.mindir")
    if not os.path.exists(graph_path):
        alt = os.path.join(mindir_dir, "vggt_omega_mindir.mindir")
        if os.path.exists(alt):
            graph_path = alt
    context = msl.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = device_id
    model = msl.Model()
    model.build_from_file(graph_path, msl.ModelType.MINDIR, context)
    return model


def run_mslite(model, images):
    """Run a single MindSpore Lite inference and return numpy outputs + time."""
    feed = images[None, ...].astype(np.float32)
    inputs = _build_mslite_inputs(model, {"images": feed})
    t0 = time.time()
    outputs = model.predict(inputs)
    elapsed = time.time() - t0
    return [o.get_data_to_numpy() for o in outputs], elapsed


def run_onnx_reference(onnx_dir, images):
    """Run the ONNX model with ONNX Runtime as a precision reference."""
    import onnxruntime as ort  # noqa: WPS433

    onnx_path = os.path.join(onnx_dir, "vggt_omega.onnx")
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 16
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    return sess.run(None, {"images": images[None, ...].astype(np.float32)})


def main():
    """Parse arguments and run MindSpore Lite inference (optionally vs ONNX)."""
    parser = argparse.ArgumentParser(description="VGGT-Omega MindSpore Lite inference")
    parser.add_argument("--mindir-dir", default="./outputs")
    parser.add_argument("--input", nargs="*", default=None)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--device", default="ascend")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--compare-onnx", action="store_true")
    parser.add_argument("--onnx-dir", default="./outputs")
    args = parser.parse_args()

    image_paths = _collect_image_paths(args.input)
    print(f"[mslite] images: {image_paths}")

    t0 = time.time()
    images = preprocess(image_paths, args.num_frames, args.img_h, args.img_w)
    pre_ms = (time.time() - t0) * 1000

    model = load_mslite_model(args.mindir_dir, args.device, args.device_id)
    # Warmup (first call compiles/allocates on Ascend).
    run_mslite(model, images)
    outputs, infer_s = run_mslite(model, images)
    pose_enc, depth, depth_conf = outputs

    total_ms = pre_ms + infer_s * 1000
    extrinsics, intrinsics = encoding_to_camera(pose_enc, args.img_h, args.img_w)
    print(f"[mslite] preprocess={pre_ms:.1f}ms inference={infer_s * 1000:.1f}ms "
          f"total={total_ms:.1f}ms")
    print(f"[mslite] pose_enc {pose_enc.shape} depth {depth.shape} "
          f"depth_conf {depth_conf.shape}")
    print(f"[mslite] extrinsics[0,0]=\n{extrinsics[0, 0]}")
    print(f"[mslite] intrinsics[0,0]=\n{intrinsics[0, 0]}")
    print(f"[mslite] depth mean={depth.mean():.4f} min={depth.min():.4f} "
          f"max={depth.max():.4f}")
    print(f"[mslite] depth_conf mean={depth_conf.mean():.4f}")

    if args.compare_onnx:
        onnx_outputs = run_onnx_reference(args.onnx_dir, images)
        names = ["pose_enc", "depth", "depth_conf"]
        print("[mslite] mslite vs onnx cosine similarity:")
        for name, ref, out in zip(names, onnx_outputs, outputs):
            print(f"  {name}: {_cosine(ref, out):.6f} max_abs={np.abs(ref - out).max():.6e}")


if __name__ == "__main__":
    main()
