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
"""BEVDet All-in-One MindSpore Lite inference script (MindIR model).

This script runs inference on the full BEVDet model that includes:
    - Image Backbone (ResNet-50)
    - Image Neck (CustomFPN)
    - LSS View Transformer (depth_net + BEVPool)
    - BEV Encoder (CustomResNet + FPN_LSS)
    - Detection Head (CenterHead)

Input: 6 camera images with shape (batch, 6, 3, 256, 704)
Output: Detection results (reg, height, dim, rot, vel, heatmap)
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite package not found.")
    print("Please install: pip install mindspore-lite")
    sys.exit(1)


SEED = 1024

DETECTION_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
]

CAM_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

# Camera order used in BEVDet export script
BEVDET_CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

INPUT_SIZE = (256, 704)
SRC_SIZE = (900, 1600)


def generate_random_images(
    batch: int, num_cams: int, channels: int, height: int, width: int
) -> np.ndarray:
    """Generate random image inputs for testing."""
    rng = np.random.RandomState(SEED)
    imgs = rng.rand(batch, num_cams, channels, height, width).astype(np.float32)
    return np.ascontiguousarray(imgs)


def load_nuscenes_sample(ann_file: str, sample_idx: int = 0) -> dict:
    """Load a sample from NuScenes validation set."""
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        infos = data.get('infos', [])
    elif isinstance(data, list):
        infos = data
    else:
        raise ValueError(f"Unexpected annotation file format: {type(data)}")

    if sample_idx >= len(infos):
        raise ValueError(f"Sample index {sample_idx} out of range (total: {len(infos)})")

    return infos[sample_idx]


def preprocess_image(img: np.ndarray, input_size: tuple, src_size: tuple) -> np.ndarray:
    """Preprocess a single image for BEVDet inference."""
    h, w = img.shape[:2]

    resize = np.array(input_size) / np.array(src_size)
    resize_h, resize_w = int(h * resize[0]), int(w * resize[1])
    img = cv2.resize(img, (resize_w, resize_h))

    crop_h, crop_w = input_size
    img = img[(resize_h - crop_h) // 2:(resize_h - crop_h) // 2 + crop_h,
              (resize_w - crop_w) // 2:(resize_w - crop_w) // 2 + crop_w]

    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img


def load_sample_images(info: dict, data_root: str) -> np.ndarray:
    """Load and preprocess 6 camera images from a NuScenes sample."""
    imgs = []
    for cam_name in BEVDET_CAM_ORDER:
        cam_info = info['cams'][cam_name]
        cam_path = cam_info['data_path']
        if not Path(cam_path).is_absolute():
            cam_path = str(Path(data_root) / cam_path)

        img = cv2.imread(cam_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {cam_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_processed = preprocess_image(img, INPUT_SIZE, SRC_SIZE)
        imgs.append(img_processed)

    imgs_array = np.stack(imgs, axis=0)
    imgs_array = np.expand_dims(imgs_array, axis=0)
    return np.ascontiguousarray(imgs_array)


class BEVDetAllInOneMsLite:
    """BEVDet All-in-One MindSpore Lite inference wrapper."""

    def __init__(self, model_path: str, device: str = "cpu", device_id: int = 0):
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)
        self.device = device

    def infer(self, imgs: np.ndarray) -> dict:
        """Run inference on image inputs."""
        if imgs.dtype != np.float32:
            imgs = imgs.astype(np.float32)

        inputs_mslite = [mslite.Tensor(np.ascontiguousarray(imgs))]
        outputs = self.model.predict(inputs_mslite)

        output_names = ["reg", "height", "dim", "rot", "vel", "heatmap"]
        results = {}
        for name, output in zip(output_names, outputs):
            results[name] = output.get_data_to_numpy()

        return results


def parse_args() -> argparse.Namespace:
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to MindIR model")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument("--imH", type=int, default=256)
    parser.add_argument("--imW", type=int, default=704)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--ann-file", type=str, default=None,
                        help="Path to NuScenes annotation file (e.g., bevdetv3-nuscenes_infos_val.pkl)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Path to NuScenes data root directory (e.g., data/nuscenes/)")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Index of the sample to use from the annotation file")
    return parser.parse_args()


def print_detection_results(results: dict):
    """Print detection output shapes and information."""
    print("--- Detection Output Shapes ---")
    for name, tensor in results.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    print("\n--- Output Interpretation ---")
    batch_size = results["heatmap"].shape[0]
    num_classes = results["heatmap"].shape[1]
    bev_h = results["heatmap"].shape[2]
    bev_w = results["heatmap"].shape[3]

    print(f"  Batch size: {batch_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  BEV grid size: {bev_h} x {bev_w}")
    print(f"  Detection classes: {', '.join(DETECTION_CLASSES)}")


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("=== BEVDet All-in-One MindIR Inference ===")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    model = BEVDetAllInOneMsLite(args.model, args.device, args.device_id)

    if args.ann_file and Path(args.ann_file).exists():
        print("Loading real NuScenes validation data...")
        print(f"  Annotation file: {args.ann_file}")
        print(f"  Data root: {args.data_root}")
        print(f"  Sample index: {args.sample_idx}")

        info = load_nuscenes_sample(args.ann_file, args.sample_idx)
        imgs = load_sample_images(info, args.data_root)
        print(f"  Input shape: {tuple(imgs.shape)}")
        print(f"  Cameras: {', '.join(CAM_NAMES)}")
    else:
        if args.ann_file:
            print(f"Warning: Annotation file not found: {args.ann_file}")
            print("Using random input instead.")

        print("Using random input for testing")
        print(f"Input shape: ({args.batch}, {args.num_cams}, 3, {args.imH}, {args.imW})")
        print(f"Seed: {SEED}")

        imgs = generate_random_images(args.batch, args.num_cams, 3, args.imH, args.imW)

    print()

    for _ in range(args.warmup):
        model.infer(imgs)

    print(f"--- Performance Benchmark ({args.runs} runs, warmup={args.warmup}) ---")
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        _ = model.infer(imgs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"  Mean latency: {np.mean(times):.2f} ms")
    print(f"  Min latency:  {np.min(times):.2f} ms")
    print(f"  Max latency:  {np.max(times):.2f} ms")
    print()

    results = model.infer(imgs)
    print_detection_results(results)


if __name__ == "__main__":
    main()
