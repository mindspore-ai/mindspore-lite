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
"""Export CenterPoint (voxel) detection head to ONNX via mmdetection3d.

CenterPoint = voxelization + spconv backbone + CenterHead. Same spconv blocker
as SECOND. This scaffold exports the dense 2D path (backbone+neck+CenterHead)
from a pre-built BEV feature map.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet3d.models import build_model
except ImportError:
    from mmdet3d.models.builder import build_model  # noqa


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export CenterPoint head to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="centerpoint_voxel_onnx/centerpoint_voxel.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--in-channels", type=int, default=256)
    parser.add_argument("--feat-h", type=int, default=180)
    parser.add_argument("--feat-w", type=int, default=180)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class CenterPointWrapper(nn.Module):
    """CenterPoint 2D head wrapper: dense BEV -> backbone -> neck -> CenterHead outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev_feat):
        """bev_feat [1,C,H,W] -> CenterHead branches (reg/height/dim/rot/vel/heatmap)."""
        backbone = self.model.pts_backbone if hasattr(self.model, 'pts_backbone') else self.model.backbone
        neck = self.model.pts_neck if hasattr(self.model, 'pts_neck') else self.model.neck
        head = self.model.pts_bbox_head if hasattr(self.model, 'pts_bbox_head') else self.model.bbox_head
        feats = backbone(bev_feat)
        feats = neck(feats)
        outs = head(feats)
        flat = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                if key in out[0]:
                    flat.append(out[0][key])
        return flat


def build_centerpoint(cfg_path, ckpt_path, device):
    """build CenterPoint model from mmdet3d config."""
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = build_centerpoint(args.config, args.checkpoint, args.device)
    wrapper = CenterPointWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, args.in_channels, args.feat_h, args.feat_w, device=args.device,
                        dtype=torch.float32)
    print(f"Exporting CenterPoint head ONNX, dense BEV shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["bev_feat"],
            output_names=["reg", "height", "dim", "rot", "vel", "heatmap"],
            dynamic_axes={"bev_feat": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
