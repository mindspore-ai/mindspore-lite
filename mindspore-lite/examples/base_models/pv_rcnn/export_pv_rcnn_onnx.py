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
"""Export PV-RCNN detection head to ONNX via mmdetection3d.

PV-RCNN = voxelization + spconv + RoI-grid PointNet pooling + Anchor3DHead.
Stage-2 blockers: spconv (sparse conv) + RoI grid pooling. This scaffold
exports the dense 2D head path from a pre-built BEV feature map.
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
    parser = argparse.ArgumentParser(description="Export PV-RCNN head to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="pv_rcnn_onnx/pv_rcnn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--in-channels", type=int, default=256)
    parser.add_argument("--feat-h", type=int, default=200)
    parser.add_argument("--feat-w", type=int, default=176)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class PVRcnnWrapper(nn.Module):
    """PV-RCNN 2D head wrapper: dense BEV -> backbone -> neck -> AnchorHead outputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bev_feat):
        """bev_feat [1,C,H,W] -> (cls, box, dir) deepest FPN level."""
        feats = self.model.backbone(bev_feat)
        feats = self.model.neck(feats)
        outs = self.model.bbox_head(feats)
        cls_scores, bbox_preds = outs[0], outs[1]
        dir_cls = outs[2] if len(outs) >= 3 else outs[1]
        return cls_scores[-1], bbox_preds[-1], dir_cls[-1]


def build_pvrccn(cfg_path, ckpt_path, device):
    """build PV-RCNN model from mmdet3d config."""
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

    model = build_pvrccn(args.config, args.checkpoint, args.device)
    wrapper = PVRcnnWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, args.in_channels, args.feat_h, args.feat_w, device=args.device,
                        dtype=torch.float32)
    print(f"Exporting PV-RCNN head ONNX, dense BEV shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["bev_feat"],
            output_names=["cls_scores", "bbox_preds", "dir_cls"],
            dynamic_axes={"bev_feat": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
