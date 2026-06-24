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
"""Export LaneATT (anchor-based lane detection) to ONNX.

LaneATT is a pure-pytorch model (lucastabelini/LaneAtt). This script loads the
upstream checkpoint and wraps the backbone+head to export raw lane proposals.
"""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export LaneATT to ONNX.")
    parser.add_argument("--model-module", type=str, default="models.lanenet",
                        help="upstream module exposing the model class")
    parser.add_argument("--model-class", type=str, default="LaneNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="laneatt_onnx/laneatt.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-h", type=int, default=288)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class LaneATTWrapper(nn.Module):
    """LaneATT wrapper exposing backbone+head raw lane proposal outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        """img -> features -> head outputs (existence + lane offsets)."""
        feats = self.model.extract_features(img) if hasattr(self.model, "extract_features") \
            else self.model.backbone(img)
        out = self.model.head(feats)
        # LaneATT head returns tuple (existence, offsets) or dict.
        if isinstance(out, (tuple, list)):
            return out[0], out[-1]
        if isinstance(out, dict):
            return list(out.values())[0], list(out.values())[-1]
        return out, out


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    module = importlib.import_module(args.model_module)
    ModelCls = getattr(module, args.model_class)
    model = ModelCls()
    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    wrapper = LaneATTWrapper(model).to(args.device).eval()
    dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device, dtype=torch.float32)
    print(f"Exporting LaneATT ONNX, input shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["img"],
            output_names=["lane_existence", "lane_offsets"],
            dynamic_axes={"img": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
