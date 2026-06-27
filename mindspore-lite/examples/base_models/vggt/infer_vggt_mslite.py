#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""VGGT MindSpore Lite inference script.

Loads the converted MindIR model and runs full-pipeline inference on input
images, producing camera poses, depth maps, and 3D point maps.

This script does NOT depend on torch — all computation uses numpy/PIL.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite not found. Install MindSpore Lite.")
    sys.exit(1)


OUTPUT_NAMES = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]


def load_and_preprocess_images(image_paths, target_size=518):
    """Load and preprocess images for VGGT inference.

    Images are center-padded to square and resized to target_size x target_size,
    matching the original VGGT preprocessing pipeline.

    Args:
        image_paths: List of image file paths.
        target_size: Target image size (square).

    Returns:
        numpy array of shape [1, S, 3, target_size, target_size] in [0, 1].
    """
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        arr = np.asarray(square_img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)
        images.append(arr)

    images = np.stack(images, axis=0)
    images = images[np.newaxis, ...]
    return np.ascontiguousarray(images, dtype=np.float32)


def generate_random_images(num_frames, img_size=518, seed=42):
    """Generate random images for testing.

    Args:
        num_frames: Number of frames to generate.
        img_size: Image size (square).
        seed: Random seed for reproducibility.

    Returns:
        numpy array of shape [1, num_frames, 3, img_size, img_size] in [0, 1].
    """
    rng = np.random.RandomState(seed)
    images = rng.rand(1, num_frames, 3, img_size, img_size).astype(np.float32)
    return np.ascontiguousarray(images)


def quat_to_mat_np(quaternions):
    """Convert quaternions (scalar-last, xyzw) to rotation matrices.

    Args:
        quaternions: numpy array of shape (..., 4).

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    i, j, k, r = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)

    o = np.stack([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ], axis=-1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def pose_encoding_to_extri_intri_np(pose_enc, image_size_hw):
    """Decode pose encoding to extrinsics and intrinsics.

    Args:
        pose_enc: numpy array of shape [B, S, 9].
        image_size_hw: Tuple of (height, width).

    Returns:
        Tuple of (extrinsics [B, S, 3, 4], intrinsics [B, S, 3, 3]).
    """
    t_vec = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]

    rot_mat = quat_to_mat_np(quat)
    extrinsics = np.concatenate([rot_mat, t_vec[..., None]], axis=-1)

    h, w = image_size_hw
    fy = (h / 2.0) / np.tan(fov_h / 2.0)
    fx = (w / 2.0) / np.tan(fov_w / 2.0)

    shape = pose_enc.shape[:2] + (3, 3)
    intrinsics = np.zeros(shape, dtype=np.float32)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = w / 2
    intrinsics[..., 1, 2] = h / 2
    intrinsics[..., 2, 2] = 1.0
    return extrinsics.astype(np.float32), intrinsics


class VGGTMSLiteInfer:
    """MindSpore Lite inference wrapper for VGGT model."""

    def __init__(self, model_path, device_id=0):
        """Initialize the MindSpore Lite inference session.

        Args:
            model_path: Path to the MindIR graph file (e.g. xxx_graph.mindir).
            device_id: Ascend device ID.
        """
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = device_id

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
        self.inputs = self.model.get_inputs()
        self.input_name = self.inputs[0].name

    def infer(self, images):
        """Run inference on input images.

        Args:
            images: Input images array [1, S, 3, H, W] in [0, 1].

        Returns:
            Dict mapping output names to numpy arrays.
        """
        inp = self.inputs[0]
        inp.set_data_from_numpy(images.astype(np.float32))
        outputs = self.model.predict([inp])
        return {name: o.get_data_to_numpy() for name, o in zip(OUTPUT_NAMES, outputs)}


def print_results(results, extrinsics=None, intrinsics=None):
    """Print inference output shapes and summary statistics.

    Args:
        results: Dict mapping output names to numpy arrays.
        extrinsics: Optional decoded extrinsics array.
        intrinsics: Optional decoded intrinsics array.
    """
    print("--- Output Shapes ---")
    for name in OUTPUT_NAMES:
        if name in results:
            arr = results[name]
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"    min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")

    if extrinsics is not None:
        print(f"  extrinsics: shape={extrinsics.shape}")
        print(f"    translation range: [{extrinsics[..., :3, 3].min():.4f}, "
              f"{extrinsics[..., :3, 3].max():.4f}]")
    if intrinsics is not None:
        print(f"  intrinsics: shape={intrinsics.shape}")
        print(f"    fx range: [{intrinsics[..., 0, 0].min():.2f}, "
              f"{intrinsics[..., 0, 0].max():.2f}]")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VGGT MindSpore Lite inference")
    parser.add_argument("--model", type=str, default="models/vggt_1b_graph.mindir",
                        help="Path to MindIR graph file")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated list of image paths")
    parser.add_argument("--num-frames", type=int, default=2,
                        help="Number of random frames (when --images not provided)")
    parser.add_argument("--img-size", type=int, default=518,
                        help="Image size (square)")
    parser.add_argument("--device-id", type=int, default=0,
                        help="Ascend device ID")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of timed runs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic input")
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("=== VGGT MindSpore Lite Inference ===")
    print(f"  Model: {args.model}")
    print(f"  Device: Ascend (device_id={args.device_id})")

    t0 = time.time()
    inferencer = VGGTMSLiteInfer(args.model, args.device_id)
    print(f"  Model load time: {time.time() - t0:.2f}s")

    preprocess_time = 0.0
    if args.images:
        image_paths = [p.strip() for p in args.images.split(",")]
        t0 = time.time()
        images = load_and_preprocess_images(image_paths, args.img_size)
        preprocess_time = time.time() - t0
        print(f"  Input: {len(image_paths)} real images")
    else:
        images = generate_random_images(args.num_frames, args.img_size, args.seed)
        print(f"  Input: {args.num_frames} random frames (seed={args.seed})")

    print(f"  Input shape: {images.shape}")
    print(f"  Preprocess time: {preprocess_time:.2f}s")
    print()

    # Warmup
    for _ in range(args.warmup):
        inferencer.infer(images)

    # Timed runs
    print(f"--- Performance ({args.runs} runs, warmup={args.warmup}) ---")
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        results = inferencer.infer(images)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"  Mean latency: {np.mean(times):.2f} ms")
    print(f"  Min latency:  {np.min(times):.2f} ms")
    print(f"  Max latency:  {np.max(times):.2f} ms")
    print(f"  Throughput:   {1000.0 / np.mean(times):.2f} fps")
    print()

    # Final inference with post-processing
    results = inferencer.infer(images)

    # Decode pose encoding
    pose_enc = results["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri_np(pose_enc, (args.img_size, args.img_size))

    print_results(results, extrinsics, intrinsics)


if __name__ == "__main__":
    main()
