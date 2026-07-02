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
"""VGGT ONNX Runtime inference script.

Loads the VGGT ONNX model and runs inference on input images,
producing camera poses, depth maps, and 3D point maps.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)


RESNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3, 1, 1))
RESNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3, 1, 1))

OUTPUT_NAMES = ["pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"]


def load_and_preprocess_images(image_paths, target_size=518):
    """Load and preprocess images for VGGT inference.

    Images are center-padded to square and resized to target_size x target_size.

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


class VGGTONNXInfer:
    """ONNX Runtime inference wrapper for VGGT model."""

    def __init__(self, model_path, device="cpu"):
        """Initialize the ONNX Runtime inference session.

        Args:
            model_path: Path to the ONNX model file.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def infer(self, images):
        """Run inference on input images.

        Args:
            images: Input images array [1, S, 3, H, W] in [0, 1].

        Returns:
            Dict mapping output names to numpy arrays.
        """
        results = self.session.run(self.output_names, {self.input_name: images})
        return dict(zip(self.output_names, results))


def print_results(results):
    """Print inference output shapes and summary statistics.

    Args:
        results: Dict mapping output names to numpy arrays.
    """
    print("--- Output Shapes ---")
    for name in OUTPUT_NAMES:
        if name in results:
            arr = results[name]
            print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
            print(f"    min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="VGGT ONNX Runtime inference")
    parser.add_argument("--model", type=str, default="models/vggt_1b.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--images", type=str, default=None,
                        help="Comma-separated list of image paths")
    parser.add_argument("--num-frames", type=int, default=2,
                        help="Number of random frames (used when --images not provided)")
    parser.add_argument("--img-size", type=int, default=518,
                        help="Image size (square)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("=== VGGT ONNX Runtime Inference ===")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")

    inferencer = VGGTONNXInfer(args.model, args.device)

    if args.images:
        image_paths = [p.strip() for p in args.images.split(",")]
        images = load_and_preprocess_images(image_paths, args.img_size)
        print(f"  Input: {len(image_paths)} real images")
    else:
        images = generate_random_images(args.num_frames, args.img_size, args.seed)
        print(f"  Input: {args.num_frames} random frames (seed={args.seed})")

    print(f"  Input shape: {images.shape}")
    print()

    for _ in range(args.warmup):
        inferencer.infer(images)

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
    print()

    results = inferencer.infer(images)
    print_results(results)


if __name__ == "__main__":
    main()
