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

import torch
from torch import nn
from mmcv import Config
from mmcv.runner import load_checkpoint



try:
    from mmdet.utils import compat_cfg
    from mmdet3d.models import build_model
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.datasets import build_dataloader, build_dataset


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
        default="bevdet_onnx/bevdet_r50.onnx",
        help="output onnx file path",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def build_bevdet_trt_model(cfg_path: str, checkpoint_path: str, device: str):
    """build bevdet trt model and return (model, cfg)"""
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
    return model, cfg


def load_first_sample(cfg, device: str):
    """Load the first real sample from the nuScenes test set, mirroring
    BEVDet/tools/analysis_tools/benchmark_trt.py.

    Returns:
        img:         [1, N, 3, H, W] tensor on `device` (kept with batch dim
                     so the wrapper's forward sees B,N,C,H,W).
        img_inputs:  list of the 7 img_inputs tensors (img + extrinsics/
                     intrinsics/post-*/bda), each moved to `device`. This is
                     the structure expected by model.get_bev_pool_input().
    """
    assert cfg.data.test.test_mode
    test_dataloader_default_args = {
        "samples_per_gpu": 1, "workers_per_gpu": 0,
        "dist": False, "shuffle": False}
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    data = next(iter(data_loader))
    img_inputs = [t.to(device) for t in data['img_inputs'][0]]
    img = img_inputs[0]
    return img, img_inputs

class CustomBEVPoolV3(torch.autograd.Function):
    """CustomBEVPoolV3"""

    @staticmethod
    def forward(_ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev, _with_depth, b, d, h, w, c):
        """Pure-torch equivalent of CUDA bev_pool_v2 / QuickCumsumCuda."""
        B, Z, Y, X, C = b, d, h, w, c
        depth_flat = depth.reshape(-1)
        feat_flat = feat.reshape(-1, C)
        contrib = depth_flat[ranks_depth.long()].unsqueeze(-1) * \
            feat_flat[ranks_feat.long()]

        sorted_idx = torch.argsort(ranks_bev)
        sorted_ranks = ranks_bev[sorted_idx].long()
        sorted_contrib = contrib[sorted_idx]

        N = sorted_contrib.shape[0]
        cumsum = sorted_contrib.cumsum(dim=0)
        cumsum_padded = torch.cat(
            [torch.zeros(1, C, dtype=cumsum.dtype), cumsum], dim=0
        )

        changes = sorted_ranks[1:] != sorted_ranks[:-1]
        starts = torch.cat([torch.zeros(1, dtype=torch.long),
                            torch.nonzero(changes).squeeze(-1) + 1])
        ends = torch.cat([starts[1:], torch.tensor([N])])

        seg_sums = cumsum_padded[ends] - cumsum_padded[starts]
        unique_ranks = sorted_ranks[starts]

        out = torch.zeros(B * Z * Y * X, C, device=depth.device, dtype=depth.dtype)
        out.scatter_(0, unique_ranks.unsqueeze(-1).expand(-1, C), seg_sums)
        return out.view(B, Z, Y, X, C).contiguous()

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev, with_depth, b, d, h, w, c):
        return g.op("Custom", depth, feat, ranks_depth, ranks_feat, ranks_bev,
                    with_depth_s=with_depth, b_i=b, d_i=d, h_i=h, w_i=w, c_i=c,
                    input_names_s=["depth", "feat", "ranks_depth", "ranks_feat", "ranks_bev"],
                    optional_input_names_s=["depth", "ranks_depth", "ranks_feat"],
                    type_s="BEVPoolV3",
                    input_index_i=[0, 1, 2, 3, 4],
                    output_names_s=["out"])


class BEVDetAllInOneWrapper(nn.Module):
    """BEVDetAllInOneWrapper"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

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

    def forward(self, img, ranks_depth, ranks_feat, ranks_bev):
        """forward — Option A: ranks are dynamic inputs."""
        B, N, C_in, H, W = img.shape
        x = img.view(B * N, C_in, H, W)
        x = self.model.img_backbone(x)
        x = self.model.img_neck(x)

        _, _, H_feat, W_feat = x.shape
        x_4d = self.model.img_view_transformer.depth_net(x)

        D = self.model.img_view_transformer.D
        C_out = self.model.img_view_transformer.out_channels
        depth_5d = x_4d[:, :D].softmax(dim=1).view(B, N, D, H_feat, W_feat)
        tran_feat_5d = x_4d[:, D:D + C_out].view(B, N, C_out, H_feat, W_feat)
        tran_feat_5d = tran_feat_5d.permute(0, 1, 3, 4, 2).contiguous()
        x = CustomBEVPoolV3.apply(
            depth_5d, tran_feat_5d,
            ranks_depth, ranks_feat, ranks_bev,
            "true",
            int(B), self.bev_z, self.bev_h, self.bev_w, self.bev_c
        )

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = torch.cat(x.unbind(dim=2), 1)

        bev_feat = self.model.bev_encoder(x)
        outs = self.model.pts_bbox_head([bev_feat])
        outs = self.result_serialize(outs)
        return outs


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = build_bevdet_trt_model(args.config, args.checkpoint, args.device)
    img, bev_pool_inputs = load_first_sample(cfg, args.device)

    # Compute example ranks for trace. Values are irrelevant post-export
    # (ranks become dynamic inputs); only shape matters during trace.
    with torch.no_grad():
        metas = model.get_bev_pool_input(list(bev_pool_inputs))

    if metas[0] is None:
        raise RuntimeError("bev_pool meta is None; first sample may have invalid extrinsics")

    ranks_bev, ranks_depth, ranks_feat, _, _ = metas

    wrapper = BEVDetAllInOneWrapper(model)
    wrapper.to(args.device)
    wrapper.eval()

    print("Exporting BEVDet all-in-one ONNX model...")
    print(f"  Config: {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {args.device}")
    print(f"  Input image shape: {tuple(img.shape)}")
    print(f"  Example ranks N_Points (for trace): {ranks_bev.shape[0]}")

    dynamic_axes = {
        "ranks_depth": {0: "n_points"},
        "ranks_feat": {0: "n_points"},
        "ranks_bev": {0: "n_points"},
    }

    torch.onnx.export(
        wrapper,
        (img.float().contiguous(),
         ranks_depth.contiguous(),
         ranks_feat.contiguous(),
         ranks_bev.contiguous()),
        str(output_path),
        opset_version=args.opset,
        input_names=["img", "ranks_depth", "ranks_feat", "ranks_bev"],
        output_names=["reg", "height", "dim", "rot", "vel", "heatmap"],
        dynamic_axes=dynamic_axes,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )

    print(f"Successfully exported to: {str(output_path)}")


if __name__ == "__main__":
    main()
