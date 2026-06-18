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
"""Export YOLOv8 to ONNX model via ultralytics."""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _check_deps():
    """check ultralytics dependency."""
    if YOLO is None:
        print("Error: ultralytics not found. Please install: pip install ultralytics")
        sys.exit(1)


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX.")
    parser.add_argument("--model-variant", type=str, default="yolov8n",
                        help="YOLOv8 variant: yolov8n / yolov8s / yolov8m / yolov8l / yolov8x")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="local .pt path (overrides --model-variant; e.g. ModelScope weights)")
    parser.add_argument("--output-dir", type=str, default=".", help="output directory")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="export with dynamic batch size")
    parser.add_argument("--img-size", type=int, default=640, help="input image size")
    return parser.parse_args()


def main():
    """main entry: export yolov8 to onnx."""
    _check_deps()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.checkpoint if args.checkpoint else (args.model_variant + ".pt"))
    model.export(format="onnx", imgsz=args.img_size, opset=args.opset, dynamic=args.dynamic)

    exported = list(Path(".").glob("*.onnx"))
    if exported:
        print("Export complete.")
        print(f"ONNX saved to: {exported[0].resolve()}")
    else:
        print(f"Error: Export failed. Check output directory: {output_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
