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
"""Export BEVFormer to ONNX via mmdetection3d.

BEVFormer's core is the Temporal Cross-View Deformable Attention
(MSDeformableAttention). The mmcv implementation uses a CUDA op that does not
export to ONNX directly. This scaffold registers the deformable attention as a
Custom op (symbolic only); stage-2 verification will likely block here and must
be recorded + shelved per plan, unless an AscendC deformable kernel is provided.
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
    parser = argparse.ArgumentParser(description="Export BEVFormer to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="bevformer_onnx/bevformer.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--img-h", type=int, default=320)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--num-query", type=int, default=900)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


class BEVFormerWrapper(nn.Module):
    """BEVFormer wrapper: multi-view images -> query-level cls + box predictions.

    Stage-1 scaffold: invokes the model forward path. The Temporal Deformable
    Attention op is exported as a Custom node (or may need rewrite); see README.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        """imgs [1,N,3,H,W] -> (cls [1,Q,cls], box [1,Q,code])."""
        b, n, c, h, w = imgs.shape
        feats = self.model.extract_feat(imgs.reshape(b * n, c, h, w), None)
        head = self.model.pts_bbox_head
        out = head.forward(feats, [{}], None)
        cls = out["all_cls_scores"][-1]
        box = out["all_bbox_preds"][-1]
        return cls, box


def build_bevformer(cfg_path, ckpt_path, device):
    """build BEVFormer model from mmdet3d config."""
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

    model = build_bevformer(args.config, args.checkpoint, args.device)
    wrapper = BEVFormerWrapper(model).to(args.device).eval()

    dummy = torch.randn(1, args.ncams, 3, args.img_h, args.img_w, device=args.device,
                        dtype=torch.float32)
    print(f"Exporting BEVFormer ONNX, input shape={tuple(dummy.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.output, opset_version=args.opset,
            input_names=["imgs"],
            output_names=["cls_scores", "bbox_preds"],
            dynamic_axes={"imgs": {0: "batch"}},
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
    print(f"Successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
