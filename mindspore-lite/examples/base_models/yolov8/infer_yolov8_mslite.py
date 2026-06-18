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
"""YOLOv8 MindSpore Lite inference script using MindIR model (numpy-only, no torch)."""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    mslite = None


SEED = 1024
IMG_SIZE = 640

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _read_proc_status_kb(key: str) -> Optional[int]:
    """read a field (KB) from /proc/self/status."""
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
    """snapshot process memory (VmRSS / VmHWM in KB)."""
    return {"vmrss_kb": _read_proc_status_kb("VmRSS"),
            "vmhwm_kb": _read_proc_status_kb("VmHWM")}


def _letterbox(img: np.ndarray, new_shape: int = IMG_SIZE,
               color: Tuple[int, int, int] = (114, 114, 114)):
    """letterbox resize keeping aspect ratio; returns (img, ratio, (dw, dh))."""
    h, w = img.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (new_shape - nh) // 2, (new_shape - nw) // 2
    top, bottom = dh, new_shape - nh - dh
    left, right = dw, new_shape - nw - dw
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one box (4,) against many boxes (N,4)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """numpy NMS, returns kept indices."""
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        iou = _box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][iou < iou_thres]
    return np.array(keep, dtype=np.int64)


def _postprocess(output: np.ndarray, ratio: float, pad: Tuple[int, int],
                 orig_shape: Tuple[int, int], conf_thres: float,
                 iou_thres: float) -> np.ndarray:
    """convert raw YOLOv8 output to [N,6] (x1,y1,x2,y2,conf,cls) in orig coords."""
    pred = output[0].T  # [8400, 84]
    boxes_xywh = pred[:, :4]
    scores = pred[:, 4:]
    class_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)
    mask = confs > conf_thres
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]
    if confs.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = _nms(boxes, confs, iou_thres)
    boxes = boxes[keep]
    confs = confs[keep]
    class_ids = class_ids[keep]

    boxes[:, 0::2] = (boxes[:, 0::2] - pad[0]) / ratio
    boxes[:, 1::2] = (boxes[:, 1::2] - pad[1]) / ratio
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, orig_shape[1])
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, orig_shape[0])
    return np.concatenate([boxes, confs[:, None],
                           class_ids[:, None].astype(np.float32)], axis=1)


def _load_image(path: str, img_size: int):
    """load and letterbox an image; returns (input[1,3,H,W], ratio, pad, orig_shape)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    orig_shape = img.shape[:2]
    img, ratio, pad = _letterbox(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img[None, ...], ratio, pad, orig_shape


def _random_input(img_size: int):
    """fixed-seed random input for perf/alignment testing."""
    rng = np.random.RandomState(SEED)
    arr = rng.rand(1, 3, img_size, img_size).astype(np.float32)
    return np.ascontiguousarray(arr)


class YoloV8MsLiteInferencer:
    """YOLOv8 MindSpore Lite inference class."""

    def __init__(self, model_path: str, device: str = "cpu", device_id: int = 0):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        if device not in ("cpu", "ascend"):
            raise ValueError("device must be cpu or ascend")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """run MindIR model forward."""
        inputs = [mslite.Tensor(np.ascontiguousarray(images))]
        outputs = self.model.predict(inputs)
        return outputs[0].get_data_to_numpy()


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="YOLOv8 MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv8 MindIR model")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path; if omitted, fixed-seed random input is used")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def _print_detections(dets: np.ndarray, image_path: Optional[str]):
    """print detection boxes."""
    print("Detection Results:")
    if image_path:
        print(f"Image: {image_path}")
    if dets.shape[0] == 0:
        print("  No detections")
        return
    for j, box in enumerate(dets):
        x1, y1, x2, y2, conf, cls = box
        name = COCO_NAMES[int(cls)] if 0 <= int(cls) < len(COCO_NAMES) else str(int(cls))
        print(f"  Det {j}: cls={int(cls)} ({name}), conf={conf:.4f}, "
              f"bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


def main():
    """main entry."""
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if mslite is None:
        print("Error: mindspore_lite not installed.")
        sys.exit(1)

    inferencer = YoloV8MsLiteInferencer(args.model, args.device, args.device_id)

    random_mode = args.image is None
    ratio, pad, orig_shape = None, None, None
    if random_mode:
        images = _random_input(args.img_size)
        print(f"Using random input, shape={images.shape}, seed={SEED}")
    else:
        if not Path(args.image).exists():
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        images, ratio, pad, orig_shape = _load_image(args.image, args.img_size)
        print(f"Image: {args.image}, input shape={images.shape}")

    mem_before = _memory_snapshot()
    for _ in range(args.warmup):
        inferencer.forward(images)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        inferencer.forward(images)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    out = inferencer.forward(images)
    print(f"Raw output shape: {out.shape}")
    if not random_mode:
        dets = _postprocess(out, ratio, pad, orig_shape, args.conf_thres, args.iou_thres)
        _print_detections(dets, args.image)
    else:
        print("(random input: postprocessing skipped)")

    lat_np = np.array(lat, dtype=np.float32)
    print("\nPerformance:")
    print(f"  batch_size: 1, warmup: {args.warmup}, runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  latency_ms_p99:  {float(np.percentile(lat_np, 99)):.3f}")
    print("\nMemory:")
    print(f"  VmRSS: {mem_after['vmrss_kb']} KB (before: {mem_before['vmrss_kb']} KB)")
    print(f"  VmHWM: {mem_after['vmhwm_kb']} KB (before: {mem_before['vmhwm_kb']} KB)")


if __name__ == "__main__":
    main()
