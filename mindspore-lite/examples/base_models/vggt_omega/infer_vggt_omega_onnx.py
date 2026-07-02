# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ONNX Runtime inference and torch-precision alignment for VGGT-Omega.

Loads the exported ``vggt_omega.onnx``, preprocesses a set of images to the fixed
export shape, runs a single feed-forward pass and decodes the camera pose and
dense depth. ``--compare-torch`` runs the original PyTorch model on the same
input and reports cosine similarity (must be > 0.99).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

_DEFAULT_UPSTREAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt-omega")
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


def _cosine(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_onnx(onnx_path, images, provider):
    """Run a single ONNX inference and return outputs + elapsed time."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 16
    sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=[provider])
    feed = images[None, ...].astype(np.float32)
    t0 = time.time()
    outputs = sess.run(None, {"images": feed})
    return outputs, time.time() - t0


def run_torch_reference(checkpoint, upstream_dir, images):
    """Run the original PyTorch model (submodules, no autocast) on the same input."""
    if upstream_dir not in sys.path:
        sys.path.insert(0, upstream_dir)
    import torch  # noqa: WPS433
    from vggt_omega.models import VGGTOmega  # noqa: WPS433

    model = VGGTOmega().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    imgs = torch.from_numpy(images).float().unsqueeze(0)
    with torch.inference_mode():
        agg, patch_token_start = model.aggregator(imgs)
        pose = model.camera_head(agg, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(agg, images=imgs, patch_token_start=patch_token_start)
    return [pose.numpy(), depth.numpy(), depth_conf.numpy()]


def main():
    """Parse arguments and run ONNX inference (optionally vs torch)."""
    parser = argparse.ArgumentParser(description="VGGT-Omega ONNX Runtime inference")
    parser.add_argument("--onnx-dir", default="./outputs")
    parser.add_argument("--checkpoint", default="/VGGT-omega/model/vggt_omega_1b_512.pt")
    parser.add_argument("--upstream-dir", default=_DEFAULT_UPSTREAM)
    parser.add_argument("--input", nargs="*", default=None)
    parser.add_argument("--num-frames", type=int, default=2)
    parser.add_argument("--img-h", type=int, default=512)
    parser.add_argument("--img-w", type=int, default=512)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--compare-torch", action="store_true")
    args = parser.parse_args()

    onnx_path = os.path.join(args.onnx_dir, "vggt_omega.onnx")
    image_paths = _collect_image_paths(args.input)
    print(f"[onnx] images: {image_paths}")

    t0 = time.time()
    images = preprocess(image_paths, args.num_frames, args.img_h, args.img_w)
    pre_ms = (time.time() - t0) * 1000

    outputs, infer_s = run_onnx(onnx_path, images, args.provider)
    pose_enc, depth, depth_conf = outputs

    extrinsics, intrinsics = encoding_to_camera(pose_enc, args.img_h, args.img_w)
    print(f"[onnx] preprocess={pre_ms:.1f}ms inference={infer_s * 1000:.1f}ms")
    print(f"[onnx] pose_enc {pose_enc.shape} depth {depth.shape} depth_conf {depth_conf.shape}")
    print(f"[onnx] extrinsics[0,0]=\n{extrinsics[0, 0]}")
    print(f"[onnx] intrinsics[0,0]=\n{intrinsics[0, 0]}")
    print(f"[onnx] depth mean={depth.mean():.4f} min={depth.min():.4f} max={depth.max():.4f}")
    print(f"[onnx] depth_conf mean={depth_conf.mean():.4f}")

    if args.compare_torch:
        torch_outputs = run_torch_reference(args.checkpoint, args.upstream_dir, images)
        names = ["pose_enc", "depth", "depth_conf"]
        print("[onnx] torch vs onnx cosine similarity:")
        for name, ref, out in zip(names, torch_outputs, outputs):
            print(f"  {name}: {_cosine(ref, out):.6f} max_abs={np.abs(ref - out).max():.6e}")


if __name__ == "__main__":
    main()
