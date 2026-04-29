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
"""Export YOLOv10-X to ONNX model."""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _check_deps():
    if YOLO is None:
        print("Error: ultralytics not found. Please install: pip install ultralytics")
        sys.exit(1)


def main():
    _check_deps()

    parser = argparse.ArgumentParser(description="Export YOLOv10-X to ONNX.")
    parser.add_argument("--model-variant", type=str, default="yolov10x", help="YOLOv10 variant")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic batch size.",
    )
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model_variant)

    if args.dynamic:
        dynamic_axes = {0: "batch"}
    else:
        dynamic_axes = None

    model.export(format="onnx", imgsz=args.img_size, opset=args.opset, dynamic=dynamic_axes)

    exported_files = list(output_dir.glob("*.onnx"))
    if exported_files:
        print("Export complete.")
        print(f"ONNX saved to: {output_dir}")
    else:
        print(f"Error: Export failed. Check output directory: {output_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
