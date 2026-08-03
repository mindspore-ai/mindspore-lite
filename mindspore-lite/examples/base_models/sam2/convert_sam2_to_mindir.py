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
"""Convert SAM2 ONNX models to MindSpore Lite MindIR (Ascend-oriented).

Converts `sam2_encoder.onnx` and `sam2_decoder.onnx` into `.mindir` files
using the MindSpore Lite converter with Ascend-oriented optimization. The
output MindIR models are loaded by `infer_sam2_mslite.py` for Ascend inference.

Usage:
    python convert_sam2_to_mindir.py \
        --onnx-dir ./onnx --output-dir ./mindir
"""

import argparse
import os

from mindspore_lite import Converter, FmkType, ModelType


def convert_one(onnx_path, output_path_no_ext, weight_fp16=False):
    """Convert a single ONNX file to MindIR with ascend_oriented optimization.

    If weight_fp16=True, const tensors (weights) of float32 are saved as
    float16, reducing model size and cast nodes during Ascend inference.
    """
    converter = Converter()
    converter.optimize = "ascend_oriented"
    converter.save_type = ModelType.MINDIR
    converter.weight_fp16 = weight_fp16
    converter.convert(FmkType.ONNX, onnx_path, output_path_no_ext)
    mindir_path = output_path_no_ext + ".mindir"
    size = os.path.getsize(mindir_path) / (1024 * 1024)
    print(f"  {os.path.basename(mindir_path)}: {size:.1f} MB "
          f"(weight_fp16={weight_fp16})")
    return mindir_path


def main():
    """Convert both SAM2 ONNX models to MindIR."""
    parser = argparse.ArgumentParser(description="Convert SAM2 ONNX to MindIR")
    parser.add_argument("--onnx-dir", default="./onnx", help="ONNX input dir")
    parser.add_argument("--output-dir", default="./mindir", help="MindIR output dir")
    parser.add_argument("--weight-fp16", action="store_true",
                        help="Save float32 const tensors as float16")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pairs = [
        ("sam2_encoder.onnx", "sam2_encoder"),
        ("sam2_decoder.onnx", "sam2_decoder"),
    ]
    mode = "weight_fp16" if args.weight_fp16 else "fp32_weights"
    print(f"=== Converting SAM2 ONNX -> MindIR (ascend_oriented, {mode}) ===")
    for onnx_name, out_name in pairs:
        onnx_path = os.path.join(args.onnx_dir, onnx_name)
        if not os.path.exists(onnx_path):
            print(f"  [skip] {onnx_path} not found")
            continue
        print(f"\nConverting {onnx_name} ...")
        out_path = os.path.join(args.output_dir, out_name)
        convert_one(onnx_path, out_path, weight_fp16=args.weight_fp16)
    print("\n=== Conversion complete ===")


if __name__ == "__main__":
    main()
