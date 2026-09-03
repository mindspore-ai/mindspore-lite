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
"""Export BEVDet ONNX (torch) with UnsortedSegmentSum BEV pool.

The BEV pool replaces bev_pool_v2 (cuda) with Gather+Gather+Mul + ONE Custom
(UnsortedSegmentSum)node. UnsortedSegmentSum is a built-in CANN op (NOT BEVPoolV3,
NOT the broken ScatterElements(add)) computing y[segment_ids[i]] += x[i] -- exactly
the BEV pool accumulation. ~30x faster BEV pool pure-torch, no accuracy loss, and no
converter downstream-miscompilation issue (built-in op adapter is complete).

forward runs pure-torch scatter_add (correct in eager); symbolic emits the
Custom UnsortedSegmentSum node for ONNX/MindIR.

Run from examples/base_models/bevdet/:
    python export_bevdet_onnx.py \
        --config BEVDet/configs/bevdet/bevdet-r50.py \
        --checkpoint bevdet-r50.pth \
        --output output \
        --prefix bevdet_r50
"""
import argparse
import os

import torch
from torch import nn
from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet3d.models import build_model


class BEVPoolSegmentSum(torch.autograd.Function):
    """y[segment_ids[i]] += x[i] via CANN UnsortedSegmentSum."""

    @staticmethod
    def forward(ctx, x, segment_ids, num_segments):
        """Forward pass using scatter_add for eager-mode correctness."""
        del ctx
        _, C = x.shape
        ns = int(num_segments.item())
        out = torch.zeros(ns, C, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, segment_ids.long().unsqueeze(-1).expand(-1, C), x)
        return out

    @staticmethod
    def symbolic(g, x, segment_ids, num_segments):
        """ONNX symbolic: emit Custom UnsortedSegmentSum node."""
        return g.op("Custom", x, segment_ids, num_segments,
                    type_s="UnsortedSegmentSum",
                    input_names_s=["x", "segment_ids", "num_segments"],
                    optional_input_names_s=[],
                    output_names_s=["y"],
                    output_num_i=1,
                    input_index_i=[0, 1, 2])


class BEVDetSegSumWrapper(nn.Module):
    """Full BEVDet: backbone/neck/depth_net + UnsortedSegmentSum BEV pool
    + bev_encoder + head."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        ivt = model.img_view_transformer
        grid_size = getattr(ivt, "grid_size", None)
        if grid_size is None:
            self.bev_h, self.bev_w, self.bev_z = 128, 128, 1
        else:
            self.bev_w = int(grid_size[0].item())
            self.bev_h = int(grid_size[1].item())
            self.bev_z = int(grid_size[2].item())
        self.bev_c = int(ivt.out_channels)

    def result_serialize(self, outs):
        """Flatten detection head outputs into a flat tensor list."""
        outs_ = []
        for out in outs:
            for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                outs_.append(out[0][key])
        return outs_

    def forward(self, img, ranks_depth, ranks_feat, ranks_bev):
        """Run full BEVDet inference with UnsortedSegmentSum BEV pool."""
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

        # BEV pool via UnsortedSegmentSum: gather+mul (standard) + Custom segsum
        depth_flat = depth_5d.reshape(-1)
        feat_flat = tran_feat_5d.reshape(-1, C_out)
        contrib = depth_flat[ranks_depth.long()].unsqueeze(-1) * \
            feat_flat[ranks_feat.long()]
        num_seg = torch.tensor([B * self.bev_z * self.bev_h * self.bev_w],
                               dtype=torch.int32, device=contrib.device)
        x = BEVPoolSegmentSum.apply(contrib, ranks_bev, num_seg)
        x = x.view(B, self.bev_z, self.bev_h, self.bev_w, C_out)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(B, C_out * self.bev_z, self.bev_h, self.bev_w)

        bev_feat = self.model.bev_encoder(x)
        outs = self.model.pts_bbox_head([bev_feat])
        return self.result_serialize(outs)


def parse_args():
    """Parse command-line arguments for ONNX export."""
    p = argparse.ArgumentParser(
        description='Export BEVDet ONNX (UnsortedSegmentSum BEV pool)')
    p.add_argument('--config', help='model config',
                   default='BEVDet/configs/bevdet/bevdet-r50.py')
    p.add_argument('--checkpoint', help='checkpoint file',
                   default='bevdet-r50.pth')
    p.add_argument('--output', help='output directory', default='output')
    p.add_argument('--prefix', help='prefix of output file', default='bevdet_r50')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--num-points', type=int, default=179832,
                   help='number of points for random ranks')
    p.add_argument('--device', default='cpu')
    return p.parse_args()


def main():
    """Export BEVDet model to ONNX with UnsortedSegmentSum BEV pool."""
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_backbone.with_cp = False
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(args.device)
    model.eval()

    wrapper = BEVDetSegSumWrapper(model)
    wrapper.to(args.device).eval()

    bev_z, bev_h, bev_w = wrapper.bev_z, wrapper.bev_h, wrapper.bev_w
    num_points = args.num_points
    num_segments = int(bev_z * bev_h * bev_w)

    # Shape pre-pass: discover H_feat, W_feat, D from the actual backbone+neck.
    # feat_flat length = B·N·H_feat·W_feat; depth_flat length = B·N·D·H_feat·W_feat.
    # ranks_* must be sampled within their respective index spaces or gather() trips.
    with torch.no_grad():
        dummy = torch.randn(1 * 6, 3, 256, 704, device=args.device)
        f = model.img_backbone(dummy)
        f = model.img_neck(f)
        _, _, H_feat, W_feat = f.shape
        D = int(model.img_view_transformer.D)
    depth_size = 1 * 6 * D * H_feat * W_feat
    feat_size = 1 * 6 * H_feat * W_feat

    img = torch.randn(1, 6, 3, 256, 704, dtype=torch.float32,
                      device=args.device)
    ranks_depth = torch.randint(0, depth_size, (num_points,),
                               dtype=torch.int32, device=args.device)
    ranks_feat = torch.randint(0, feat_size, (num_points,),
                               dtype=torch.int32, device=args.device)
    ranks_bev = torch.randint(0, num_segments, (num_points,),
                              dtype=torch.int32, device=args.device)

    grid = [int(bev_w), int(bev_h), int(bev_z)]
    print("Exporting BEVDet ONNX (UnsortedSegmentSum BEV pool)...")
    print(f"  img shape: {tuple(img.shape)}  BEV grid: {grid}  "
          f"ranks N_Points: {num_points}")
    print(f"  depth_size: {depth_size}  feat_size: {feat_size}  "
          f"num_segments: {num_segments}")

    output_path = os.path.join(args.output, args.prefix + '.onnx')
    torch.onnx.export(
        wrapper,
        (img.float().contiguous(),
         ranks_depth.int().contiguous(),
         ranks_feat.int().contiguous(),
         ranks_bev.int().contiguous()),
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['img', 'ranks_depth', 'ranks_feat', 'ranks_bev'],
        output_names=['reg', 'height', 'dim', 'rot', 'vel', 'heatmap'],
        dynamic_axes=None,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    print(f"Successfully exported to: {output_path}")


if __name__ == '__main__':
    main()
