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
"""YOLOv10 MindSpore Lite inference script using MindIR model."""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite package not found.")
    print("Please install: pip install mindspore-lite")
    mslite = None

try:
    from torchvision.ops import nms
except ImportError:
    nms = None


def _read_proc_status_kb(key: str) -> Optional[int]:
    """_read_proc_status_kb"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _memory_snapshot() -> dict:
    return {
        "vmrss_kb": _read_proc_status_kb("VmRSS"),
        "vmhwm_kb": _read_proc_status_kb("VmHWM"),
    }


def _non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> List[np.ndarray]:
    """
    Perform Non-Maximum Suppression on YOLO detection outputs.
    """
    output = []
    for pred in prediction:
        conf_mask = pred[:, 4] > conf_thres
        pred = pred[conf_mask]

        if pred.shape[0] == 0:
            output.append(np.zeros((0, 6), dtype=np.float32))
            continue

        x1 = pred[:, 0]
        y1 = pred[:, 1]
        x2 = pred[:, 2]
        y2 = pred[:, 3]
        scores = pred[:, 4]

        boxes = np.column_stack([x1, y1, x2, y2]).astype(np.float32)
        order = scores.argsort()[::-1][:max_det]

        if nms is not None:
            keep = nms(torch.from_numpy(boxes[order]), torch.from_numpy(scores[order]), iou_thres)
            keep = order[keep.numpy()]
        else:
            keep = []
            while order.shape[0] > 0:
                i = order[0]
                keep.append(i)
                if order.shape[0] == 1:
                    break
                iou = _box_iou(boxes[i:i+1], boxes[order[1:]])
                mask = iou.squeeze() < iou_thres
                order = order[1:][mask]
            keep = np.array(keep, dtype=np.int32)

        result = pred[keep][:, [0, 1, 2, 3, 4, 5]]
        output.append(result)

    return output


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculate IoU between two sets of boxes."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    iou = inter_area / (area1[:, None] + area2 - inter_area + 1e-9)
    return iou


@dataclass
class YoloOutput:
    boxes: np.ndarray
    orig_shape: Tuple[int, int]


class YoloV10MsLiteInferencer:
    """YOLOv10 MindSpore Lite inference class."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        device_id: int = 0,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        img_size: int = 640,
    ):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")

        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def preprocess(self, image_paths: List[str]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Preprocess images for YOLO input."""
        images = []
        orig_shapes = []

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            orig_shapes.append(img.size[::-1])

            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = img_array.transpose(2, 0, 1)
            images.append(img_array)

        images = np.stack(images, axis=0)
        return images.astype(np.float32), orig_shapes

    def forward(self, images: np.ndarray) -> np.ndarray:
        """Run MindIR model inference."""
        inputs = [mslite.Tensor(images)]
        outputs = self.model.predict(inputs)
        return outputs[0].get_data_to_numpy()

    def postprocess(
        self, outputs: np.ndarray, orig_shapes: List[Tuple[int, int]]
    ) -> List[YoloOutput]:
        """Postprocess YOLO outputs with NMS."""
        detections = _non_max_suppression(
            outputs, conf_thres=self.conf_thres, iou_thres=self.iou_thres
        )

        results = []
        for det, orig_shape in zip(detections, orig_shapes):
            if det.shape[0] > 0:
                det[:, [0, 2]] /= self.img_size
                det[:, [1, 3]] /= self.img_size
                det[:, [0, 2]] *= orig_shape[1]
                det[:, [1, 3]] *= orig_shape[0]
            results.append(YoloOutput(boxes=det, orig_shape=orig_shape))

        return results

    def infer(self, image_paths: List[str]) -> List[YoloOutput]:
        """Run full inference pipeline."""
        images, orig_shapes = self.preprocess(image_paths)
        outputs = self.forward(images)
        return self.postprocess(outputs, orig_shapes)


def _parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="YOLOv10 MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv10 MindIR model")
    parser.add_argument("--image", type=str, required=True, help="Image path, or comma-separated paths")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main():
    args = _parse_args()

    image_paths = [s.strip() for s in args.image.split(",") if s.strip()]
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

    inferencer = YoloV10MsLiteInferencer(
        model_path=args.model,
        device=args.device,
        device_id=args.device_id,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
    )

    images, _ = inferencer.preprocess(image_paths)

    mem_before = _memory_snapshot()
    for _ in range(args.warmup):
        _ = inferencer.forward(images)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        _ = inferencer.forward(images)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)
    mem_after = _memory_snapshot()

    results = inferencer.infer(image_paths)

    print("Detection Results:")
    for i, result in enumerate(results):
        print(f"\nImage: {image_paths[i]}")
        if result.boxes.shape[0] == 0:
            print("  No detections")
        else:
            for j, box in enumerate(result.boxes):
                x1, y1, x2, y2, conf, cls = box
                print(f"  Det {j}: cls={int(cls)}, conf={conf:.4f}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    lat_np = np.array(lat, dtype=np.float32)
    print("\nPerformance:")
    print(f"  batch_size: {images.shape[0]}")
    print(f"  warmup: {args.warmup}, runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print("\nMemory:")
    print(f"  VmRSS: {mem_after['vmrss_kb']} KB (before: {mem_before['vmrss_kb']} KB)")
    print(f"  VmHWM: {mem_after['vmhwm_kb']} KB (before: {mem_before['vmhwm_kb']} KB)")


if __name__ == "__main__":
    main()
