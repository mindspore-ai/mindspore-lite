# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Export ViT-Base model from timm to ONNX format.

This script exports a pretrained Vision Transformer (ViT) model from the
timm library to ONNX format for deployment with MindSpore Lite.
"""

import argparse

import timm
import torch


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing batch size and output path.
    """
    parser = argparse.ArgumentParser(
        description="Export ViT model to ONNX format"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for the exported model (default: 256)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default="vit_base_b256.onnx",
        help='Output path for the ONNX model (default: vit_base_b256.onnx)'
    )
    return parser.parse_args()


def export_vit_to_onnx(batch_size: int, output_path: str) -> None:
    """
    Export ViT-Base model to ONNX format.

    Args:
        batch_size: Batch size for the exported model.
        output_path: Path where the ONNX model will be saved.
    """
    print("[INFO] Loading pretrained ViT-Base model from timm...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True).eval()

    # Create standard NCHW input tensor (Batch, Channel, Height, Width)
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    print(f"[INFO] Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['output'],
        opset_version=13,
        do_constant_folding=True
    )
    print("[INFO] ONNX export completed successfully")


def main() -> None:
    """Main function for exporting ViT model to ONNX."""
    args = parse_args()
    export_vit_to_onnx(args.batch_size, args.output)


if __name__ == '__main__':
    main()

