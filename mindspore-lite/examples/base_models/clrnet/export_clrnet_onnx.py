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
"""Export CLRNet (lane detection) to ONNX via mmcv/mmdet."""

import argparse
from pathlib import Path

import torch
from torch import nn

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models.builder import build_detector


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export CLRNet to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="clrnet_onnx/clrnet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-h", type=int, default=288)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class CLRNetWrapper(nn.Module):
    """CLRNet wrapper exposing backbone+neck+head raw outputs (lane priors)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        """img -> features -> lane head outputs (existence + coords)."""
        feats = self.model.backbone(img)
        feats = self.model.neck(feats)
        out = self.model.head(feats)
        # CLRHead.forward returns dict with existence logits and lane coordinates.
        if isinstance(out, dict):
            logits = out.get("logits", list(out.values())[0])
            coords = out.get("lane_coords", list(out.values())[-1])
            return logits, coords
        return out[0], out[-1]


def build_clrnet(cfg_path, ckpt_path, device):
    """build CLRNet detector from mmdet config."""
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = build_clrnet(args.config, args.checkpoint, args.device)
    wrapper = CLRNetWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device, dtype=torch.float32)
    print(f"Exporting CLRNet ONNX, input shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["img"],
            output_names=["lane_logits", "lane_coords"],
            dynamic_axes={"img": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
