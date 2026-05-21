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
"""Export bevdet to ONNX model."""
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


def parse_args() -> argparse.Namespace:
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="BEVDet/configs/bevdet/bevdet-r50.py",
        help="model config file path",
    )
    parser.add_argument(
        "--checkpoint",
        default="bevdet-dev2.1/bevdet-r50.pth",
        help="checkpoint file path",
    )
    parser.add_argument(
        "--output",
        default="bevdet_onnx/bevdet_r50_all.onnx",
        help="output onnx file path",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=704)
    return parser.parse_args()


def build_bevdet_trt_model(cfg_path: str, checkpoint_path: str, device: str) -> torch.nn.Module:
    """build bevdet trt model"""
    cfg = Config.fromfile(cfg_path)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + "TRT"
    cfg.model.img_backbone.with_cp = False
    cfg = compat_cfg(cfg)
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def build_dummy_inputs(device: str, ncams: int, img_h: int, img_w: int):
    """build dummy inputs"""
    b = 1
    imgs = torch.randn(b, ncams, 3, img_h, img_w, device=device, dtype=torch.float32)
    sensor2egos = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4)
    sensor2egos = sensor2egos.expand(b, ncams, 4, 4).contiguous()
    ego2globals = torch.eye(4, device=device, dtype=torch.float32).view(1, 1, 4, 4)
    ego2globals = ego2globals.expand(b, ncams, 4, 4).contiguous()
    intrins = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3)
    intrins = intrins.expand(b, ncams, 3, 3).contiguous()
    post_rots = torch.eye(3, device=device, dtype=torch.float32).view(1, 1, 3, 3)
    post_rots = post_rots.expand(b, ncams, 3, 3).contiguous()
    post_trans = torch.zeros(b, ncams, 3, device=device, dtype=torch.float32)
    bda = torch.eye(4, device=device, dtype=torch.float32).view(1, 4, 4).expand(b, 4, 4)

    return imgs, [imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda]


_bev_pool_ctx = {}


class CustomBEVPoolV3(torch.autograd.Function):
    """CustomBEVPoolV3"""

    @staticmethod
    def forward(ctx, depth, feat, ranks_bev, with_depth, b, d, h, w, c):
        """
        Args:
            depth: [N_RANKS, D_depth] 2维深度张量
            feat: [N_RANKS, C] 2维特征张量
            ranks_bev: [N_RANKS] BEV索引
        """
        del ctx, with_depth, c
        C = feat.shape[1]
        B = b
        D_z = d

        out = torch.zeros(B, C, D_z, h, w, device=depth.device, dtype=depth.dtype)

        interval_starts = _bev_pool_ctx['interval_starts']
        interval_lengths = _bev_pool_ctx['interval_lengths']

        start = int(interval_starts[0].item())
        end = int(interval_starts[-1].item()) + int(interval_lengths[-1].item())

        idx = ranks_bev[start:end].long()
        bev_size = h * w

        for i in range(end - start):
            bi = idx[i].item()

            b_out = bi // (D_z * bev_size)
            bev_offset = bi % (D_z * bev_size)
            d_out = bev_offset // bev_size
            hw_out = bev_offset % bev_size

            if b_out < B and d_out < D_z:
                depth_val = depth[start + i]  # [D_depth]
                feat_val = feat[start + i]    # [C]
                weighted = feat_val * depth_val.mean()  # 简化处理
                out[b_out, :, d_out, hw_out // w, hw_out % w] += weighted

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


class BEVDetAllInOneWrapper(nn.Module):
    """BEVDetAllInOneWrapper"""

    def __init__(self, model: nn.Module, bev_pool_meta: Tuple):
        super().__init__()
        self.model = model
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = bev_pool_meta

        self.register_buffer('ranks_bev_buf', ranks_bev)

        global _bev_pool_ctx
        _bev_pool_ctx = {}
        _bev_pool_ctx['ranks_depth'] = ranks_depth
        _bev_pool_ctx['ranks_feat'] = ranks_feat
        _bev_pool_ctx['interval_starts'] = interval_starts
        _bev_pool_ctx['interval_lengths'] = interval_lengths

        grid_size = getattr(model.img_view_transformer, "grid_size", None)
        if grid_size is None:
            self.bev_h = 128
            self.bev_w = 128
        else:
            self.bev_w = int(grid_size[0].item())
            self.bev_h = int(grid_size[1].item())
        self.bev_c = int(model.img_view_transformer.out_channels)
        self.bev_z = int(grid_size[2].item()) if grid_size is not None else 8

    def result_serialize(self, outs):
        """result_serialize"""
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def forward(self, img):
        """forward"""
        B, N, C, H, W = img.shape
        x = img.view(B * N, C, H, W)
        x = self.model.img_backbone(x)
        x = self.model.img_neck(x)

        _, C_new, H_feat, W_feat = x.shape
        x = x.view(B, N, C_new, H_feat, W_feat)

        x_4d = x.view(B * N, C_new, H_feat, W_feat)
        print("x_4d.shape : ", x_4d.shape)
        x_4d = self.model.img_view_transformer.depth_net(x_4d)

        depth_5d = x_4d[:, :self.model.img_view_transformer.D] \
                   .softmax(dim=1) \
                   .view(B, N, self.model.img_view_transformer.D, H_feat, W_feat)
        tran_feat_5d = x_4d[:, self.model.img_view_transformer.D:(
            self.model.img_view_transformer.D +
            self.model.img_view_transformer.out_channels)] \
            .view(B, N, self.model.img_view_transformer.out_channels, H_feat, W_feat)
        tran_feat_5d = tran_feat_5d.permute(0, 1, 3, 4, 2).contiguous()

        ranks_depth = _bev_pool_ctx['ranks_depth']
        ranks_feat = _bev_pool_ctx['ranks_feat']

        D = self.model.img_view_transformer.D
        C = self.model.img_view_transformer.out_channels

        depth_flat = depth_5d.reshape(-1)
        base_idx = (ranks_depth.long() // D) * D
        depth_2d = torch.stack([torch.gather(depth_flat, 0, base_idx + d) for d in range(D)], dim=1)

        feat_flat = tran_feat_5d.reshape(-1, C)
        feat_2d = torch.gather(feat_flat, 0, ranks_feat.long().unsqueeze(-1).expand(-1, C))

        x = CustomBEVPoolV3.apply(
            depth_2d, feat_2d,
            self.ranks_bev_buf,
            "false",
            int(B), self.bev_z, self.bev_h, self.bev_w, self.bev_c
        )

        B_out, C_out, D_out, H_out, W_out = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B_out, C_out * D_out, H_out, W_out)

        bev_feat = self.model.bev_encoder(x)
        outs = self.model.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_bevdet_trt_model(args.config, args.checkpoint, args.device)
    img, bev_pool_inputs = build_dummy_inputs(args.device, args.ncams, args.img_h, args.img_w)

    with torch.no_grad():
        metas = model.get_bev_pool_input(list(bev_pool_inputs))

    if metas[0] is None:
        raise RuntimeError("bev_pool meta is None; dummy extrinsics may be invalid")

    ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = metas

    wrapper = BEVDetAllInOneWrapper(model, (ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths))
    wrapper.to(args.device)
    wrapper.eval()

    print("Exporting BEVDet all-in-one ONNX model...")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Input image shape: ({args.ncams}, 3, {args.img_h}, {args.img_w})")
    print(f"  BEV pool intervals: {interval_starts.shape[0]}")

    dynamic_axes = {
        "img": {0: "batch"},
        "reg": {0: "batch"},
        "height": {0: "batch"},
        "dim": {0: "batch"},
        "rot": {0: "batch"},
        "vel": {0: "batch"},
        "heatmap": {0: "batch"},
    }
    print("img.shape : ", img.shape)
    torch.onnx.export(
        wrapper,
        img.float().contiguous(),
        str(output_path),
        opset_version=args.opset,
        input_names=["img"],
        output_names=["reg", "height", "dim", "rot", "vel", "heatmap"],
        dynamic_axes=dynamic_axes,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )

    print(f"Successfully exported to: {str(output_path)}")


if __name__ == "__main__":
    main()
