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
# See the License for the license governing permissions and
# limitations under the License.
# ============================================================================
"""Export BEVDepth to ONNX.

BEVDepth is an BEVDet-family model that adds explicit depth supervision. It reuses
the same Custom BEVPoolV3 op pattern as the bevdet example. The exported ONNX
contains a Custom node (BEVPoolV3), so it must be converted to MindIR and run on
Ascend (ONNX Runtime cannot execute it). Stage-2 verification may block on the
BEVPool custom op if no AscendC kernel is available; record and shelve per plan.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
    from mmdet3d.models import build_model
except ImportError:
    from mmdet3d.utils import compat_cfg


_bev_pool_ctx = {}


class CustomBEVPoolV3(torch.autograd.Function):
    """Custom BEVPoolV3 op (symbolic registers Custom::BEVPoolV3 in ONNX)."""

    @staticmethod
    def forward(ctx, depth, feat, ranks_bev, with_depth, b, d, h, w, c):
        """forward placeholder (real pooling done by AscendC kernel at runtime)."""
        del ctx, with_depth, c
        out = torch.zeros(b, feat.shape[1], d, h, w, device=depth.device, dtype=depth.dtype)
        return out.contiguous()

    @staticmethod
    def symbolic(g, depth, feat, ranks_bev, with_depth, b, d, h, w, c):
        return g.op("Custom", depth, feat, ranks_bev,
                    with_depth_s=with_depth, b_i=b, d_i=d, h_i=h, w_i=w, c_i=c,
                    input_names_s=["depth", "feat", "ranks_depth", "ranks_feat", "ranks_bev"],
                    optional_input_names_s=["depth", "ranks_depth", "ranks_feat"],
                    type_s="BEVPoolV3",
                    input_index_i=[0, 1, 4],
                    output_names_s=["out"])


class BEVDepthWrapper(nn.Module):
    """BEVDepth all-in-one wrapper (backbone + view transformer + BEV encoder + head)."""

    def __init__(self, model, bev_pool_meta):
        super().__init__()
        self.model = model
        ranks_bev = bev_pool_meta[0]
        self.register_buffer('ranks_bev_buf', ranks_bev)
        _bev_pool_ctx.update({
            'ranks_depth': bev_pool_meta[1],
            'ranks_feat': bev_pool_meta[2],
            'interval_starts': bev_pool_meta[3],
            'interval_lengths': bev_pool_meta[4],
        })
        vt = model.img_view_transformer
        self.bev_h, self.bev_w = 128, 128
        self.bev_c = int(vt.out_channels)
        self.bev_z = int(vt.grid_size[2].item()) if hasattr(vt, "grid_size") else 8

    def forward(self, img):
        """img [B,N,3,H,W] -> head outputs (reg/height/dim/rot/vel/heatmap)."""
        b, n, c, h, w = img.shape
        x = self.model.img_backbone(img.reshape(b * n, c, h, w))
        x = self.model.img_neck(x)
        _, c_new, hf, wf = x.shape
        x = x.view(b, n, c_new, hf, wf)
        bev_feat = self.model.img_view_transformer.depth_net(
            x.reshape(b * n, c_new, hf, wf))
        bev_feat = bev_feat.permute(0, 2, 3, 1).contiguous()
        depth_2d = bev_feat[:, :, :, :1].reshape(b * n * hf * wf, 1)
        feat_2d = bev_feat[:, :, :, 1:].reshape(b * n * hf * wf, -1)
        ranks_bev = self.ranks_bev_buf
        pooled = CustomBEVPoolV3.apply(
            depth_2d, feat_2d, ranks_bev, "false",
            int(b), self.bev_z, self.bev_h, self.bev_w, self.bev_c)
        pooled = pooled.view(b, -1, self.bev_h, self.bev_w)
        outs = self.model.pts_bbox_head([self.model.bev_encoder(pooled)])
        flat = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                flat.append(out[0][key])
        return flat


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="Export BEVDepth to ONNX.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="bevdepth_onnx/bevdepth.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=704)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    """main entry."""
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + "TRT"
    cfg.model.img_backbone.with_cp = False
    cfg = compat_cfg(cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(args.device).eval()

    imgs, bev_pool_inputs = build_dummy_inputs(args.device, args.ncams, args.img_h, args.img_w)
    with torch.no_grad():
        metas = model.get_bev_pool_input(list(bev_pool_inputs))
    if metas[0] is None:
        raise RuntimeError("bev_pool meta is None; dummy extrinsics may be invalid")

    wrapper = BEVDepthWrapper(model, metas).to(args.device).eval()
    print(f"Exporting BEVDepth ONNX, input shape={tuple(imgs.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, imgs.float().contiguous(), args.output, opset_version=args.opset,
            input_names=["img"],
            output_names=["reg", "height", "dim", "rot", "vel", "heatmap"],
            dynamic_axes={"img": {0: "batch"}},
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        )
    print(f"Successfully exported to: {args.output}")


def build_dummy_inputs(device, ncams, img_h, img_w):
    """build dummy camera inputs and bev pool inputs."""
    b = 1
    imgs = torch.randn(b, ncams, 3, img_h, img_w, device=device, dtype=torch.float32)
    eye4 = torch.eye(4, device=device, dtype=torch.float32)
    eye3 = torch.eye(3, device=device, dtype=torch.float32)
    sensor2egos = eye4.view(1, 1, 4, 4).expand(b, ncams, 4, 4).contiguous()
    ego2globals = sensor2egos.clone()
    intrins = eye3.view(1, 1, 3, 3).expand(b, ncams, 3, 3).contiguous()
    post_rots = eye3.view(1, 1, 3, 3).expand(b, ncams, 3, 3).contiguous()
    post_trans = torch.zeros(b, ncams, 3, device=device, dtype=torch.float32)
    bda = eye4.view(1, 4, 4).expand(b, 4, 4)
    return imgs, [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]


if __name__ == "__main__":
    main()
