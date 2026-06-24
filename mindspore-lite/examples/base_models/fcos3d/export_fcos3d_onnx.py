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
"""Export FCOS3D (FCOSMono3D, monocular 3D detection) to ONNX via mmdetection3d."""

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
    parser = argparse.ArgumentParser(description="Export FCOS3D to ONNX.")
    parser.add_argument("--config", type=str, required=True,
                        help="mmdet3d fcos3d config path")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint path")
    parser.add_argument("--output", type=str, default="fcos3d_onnx/fcos3d.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-h", type=int, default=320, help="export input height")
    parser.add_argument("--img-w", type=int, default=800, help="export input width")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class FCOS3DWrapper(nn.Module):
    """Wrapper exposing backbone+neck+bbox_head raw outputs (deepest FPN level)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        """forward: img -> feature -> bbox_head raw outputs."""
        feats = self.model.extract_feat(img)
        outs = self.model.bbox_head(feats)
        # mmdet3d FCOSMono3D bbox_head.forward returns 4-tuple:
        # (cls_scores, bbox_preds, centernesses, dir_cls_scores), each multi-level.
        cls_scores, bbox_preds, centernesses = outs[0], outs[1], outs[2]
        dir_cls = outs[3] if len(outs) >= 4 else outs[2]
        return cls_scores[-1], bbox_preds[-1], centernesses[-1], dir_cls[-1]


def build_model_from_config(cfg_path, ckpt_path, device):
    """build FCOSMono3D model from mmdet3d config."""
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

    model = build_model_from_config(args.config, args.checkpoint, args.device)
    wrapper = FCOS3DWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device, dtype=torch.float32)
    print(f"Exporting FCOS3D ONNX, input shape={tuple(dummy.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output,
            opset_version=args.opset,
            input_names=["img"],
            output_names=["cls_score", "bbox_pred", "centerness", "dir_cls"],
            dynamic_axes={"img": {0: "batch"}},
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
