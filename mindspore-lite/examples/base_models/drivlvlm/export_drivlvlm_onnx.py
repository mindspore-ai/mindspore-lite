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
"""Export DriveVLM vision encoder to ONNX.

DriveVLM = Vision encoder + LLM (7B+). This scaffold exports the vision encoder
(image -> image_embeds). The LLM part (prefill + decode) reuses the qwen2_7b
pipeline pattern; see README for the LLM stage-2 plan.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel

try:
    from torch.onnx import OperatorExportTypes
except ImportError:
    OperatorExportTypes = None  # noqa


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export DriveVLM vision encoder to ONNX.")
    parser.add_argument("--model-id", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--output", type=str, default="drivlvlm_onnx/drivlvlm_vision.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-h", type=int, default=336)
    parser.add_argument("--img-w", type=int, default=336)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class VisionEncoder(nn.Module):
    """DriveVLM vision encoder wrapper exposing image_embeds."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        """pixel_values [1,3,H,W] -> image_embeds [1,N,D]."""
        out = self.model(pixel_values)
        if isinstance(out, (tuple, list)):
            return out[0]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, dict):
            return out.get("image_embeds", list(out.values())[0])
        return out


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    full = AutoModel.from_pretrained(args.model_id, trust_remote_code=True,
                                     torch_dtype=torch.float32)
    vision = full.vision_model if hasattr(full, "vision_model") else full.model.vision_tower
    vision.to(args.device).eval()

    wrapper = VisionEncoder(vision).to(args.device).eval()
    dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device, dtype=torch.float32)
    print(f"Exporting DriveVLM vision encoder ONNX, input shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            dynamic_axes={"pixel_values": {0: "batch"},
                          "image_embeds": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
