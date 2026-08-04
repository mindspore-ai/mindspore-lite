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
"""Export FlashOCC ONNX (torch) with UnsortedSegmentSum BEV pool.

The BEV pool replaces TRTBEVPoolv2 (CUDA plugin) with Gather+Gather+Mul +
ONE Custom(UnsortedSegmentSum) node — a built-in CANN op computing
y[segment_ids[i]] += x[i], exactly the BEV pool accumulation.

Key differences from the BEVDet Ascend export:
  - Neck (CustomFPN) returns a list -> take x[0]
  - BEV encoder is split into img_bev_encoder_backbone + img_bev_encoder_neck
  - OCC head (BEVOCCHead2D) for occupancy prediction, output (B,Dx,Dy,Dz,n_cls)
  - wocc / wdet3d flags control the output structure
  - Optional upsample (F.interpolate x2) before occ_head
  - grid_size [200, 200, 1] (not [128, 128, 1])
  - Exports TWO onnx: forward_ori (occ logits) + forward_with_argmax (occ labels)
  - Plugin import logic for custom mmdet3d_plugin modules
  - img limited to 6 cameras (temporal-frame robustness)
Run from FlashOCC/:
    python scripts/export_flashocc_onnx.py \
        --config projects/configs/flashocc/flashocc-r50.py \
        --checkpoint flashocc-r50-256x704.pth \
        --work_dir output \
        --prefix flashocc_r50_torch
"""
import argparse
import importlib
import os
import sys

sys.path.insert(0, os.getcwd()) # pylint: disable=wrong-import-position

import torch
from torch import nn
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor # pylint: disable=ungrouped-imports,wrong-import-position

class BEVPoolSegmentSum(torch.autograd.Function):
    """y[segment_ids[i]] += x[i] via CANN UnsortedSegmentSum."""

    @staticmethod
    def forward(ctx, x, segment_ids, num_segments):
        """Accumulate x into output indexed by segment_ids via scatter_add_."""
        del ctx
        _, C = x.shape
        ns = int(num_segments.item())
        out = torch.zeros(ns, C, dtype=x.dtype, device=x.device)
        out.scatter_add_(0, segment_ids.long().unsqueeze(-1).expand(-1, C), x)
        return out

    @staticmethod
    def symbolic(g, x, segment_ids, num_segments):
        """Emit a Custom(UnsortedSegmentSum) ONNX node for CANN."""
        return g.op("Custom", x, segment_ids, num_segments,
                    type_s="UnsortedSegmentSum",
                    input_names_s=["x", "segment_ids", "num_segments"],
                    optional_input_names_s=[],
                    output_names_s=["y"],
                    output_num_i=1,
                    input_index_i=[0, 1, 2])

class FlashOCCSegSumWrapper(nn.Module):
    """Full FlashOCC: backbone/neck/depth_net + UnsortedSegmentSum BEV pool
    + img_bev_encoder_backbone + img_bev_encoder_neck + occ_head
    (+ pts_bbox_head when wdet3d).

    Mirrors BEVDetOCCTRT.forward_ori but replaces TRTBEVPoolv2 with the
    UnsortedSegmentSum custom op (same technique as the BEVDet Ascend export).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        ivt = model.img_view_transformer
        grid_size = getattr(ivt, "grid_size", None)
        if grid_size is None:
            # FlashOCC default grid (x=200, y=200, z=1)
            self.bev_h, self.bev_w, self.bev_z = 200, 200, 1
        else:
            self.bev_w = int(grid_size[0].item())  # x -> W
            self.bev_h = int(grid_size[1].item())  # y -> H
            self.bev_z = int(grid_size[2].item())  # z -> Z
        self.bev_c = int(ivt.out_channels)

        # FlashOCC-specific flags (set by *TRT model classes)
        self.wocc = getattr(model, 'wocc', True)
        self.wdet3d = getattr(model, 'wdet3d', False)
        self.upsample = getattr(model, 'upsample', False)
        self.uni_train = getattr(model, 'uni_train', True)

    def result_serialize(self, outs_det3d=None, outs_occ=None):
        """Serialize outputs to a flat list — det3d first (6 per task head),
        then occ last.  Matches BEVDetOCCTRT.result_serialize."""
        outs_ = []
        if outs_det3d is not None:
            for out in outs_det3d:
                for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
                    outs_.append(out[0][key])
        if outs_occ is not None:
            outs_.append(outs_occ)
        return outs_

    def _bev_pool(self, x_4d, B, N, H_feat, W_feat,
                  ranks_depth, ranks_feat, ranks_bev):
        """UnsortedSegmentSum BEV pool — same logic as the BEVDet Ascend
        export but factored out for clarity."""
        D = self.model.img_view_transformer.D
        C_out = self.model.img_view_transformer.out_channels

        depth_5d = x_4d[:, :D].softmax(dim=1).view(B, N, D, H_feat, W_feat)
        tran_feat_5d = x_4d[:, D:D + C_out].view(B, N, C_out, H_feat, W_feat)
        tran_feat_5d = tran_feat_5d.permute(0, 1, 3, 4, 2).contiguous()

        # gather+mul (standard) + Custom UnsortedSegmentSum
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
        return x

    def forward_ori(self, img, ranks_depth, ranks_feat, ranks_bev):
        """Full forward producing raw occ logits (and optional det3d outputs)."""
        B, N, C_in, H, W = img.shape
        x = img.view(B * N, C_in, H, W)
        x = self.model.img_backbone(x)
        x = self.model.img_neck(x)
        # CustomFPN returns a list — take the single output (num_outs=1)
        x = x[0]

        _, _, H_feat, W_feat = x.shape
        x_4d = self.model.img_view_transformer.depth_net(x)

        # BEV pool via UnsortedSegmentSum (replaces TRTBEVPoolv2)
        x = self._bev_pool(x_4d, B, N, H_feat, W_feat,
                           ranks_depth, ranks_feat, ranks_bev)

        # BEV encoder: backbone -> neck (FlashOCC uses two separate modules,
        # unlike BEVDet which has a single bev_encoder)
        bev_feature = self.model.img_bev_encoder_backbone(x)
        occ_bev_feature = self.model.img_bev_encoder_neck(bev_feature)

        outs_occ = None
        if self.wocc:
            # uni_train gates upsample (matches BEVDetOCCTRT.forward_ori)
            if self.uni_train and self.upsample:
                occ_bev_feature = F.interpolate(
                    occ_bev_feature, scale_factor=2,
                    mode='bilinear', align_corners=True)
            outs_occ = self.model.occ_head(occ_bev_feature)

        outs_det3d = None
        if self.wdet3d:
            outs_det3d = self.model.pts_bbox_head([occ_bev_feature])

        outs = self.result_serialize(outs_det3d, outs_occ)
        return outs

    def forward_with_argmax(self, img, ranks_depth, ranks_feat, ranks_bev):
        """forward_ori then argmax over occ logits -> per-voxel class label."""
        outs = self.forward_ori(img, ranks_depth, ranks_feat, ranks_bev)
        pred_occ_label = outs[-1].argmax(-1)
        return pred_occ_label

    def forward(self, img, ranks_depth, ranks_feat, ranks_bev):
        """Default forward delegates to forward_ori."""
        return self.forward_ori(img, ranks_depth, ranks_feat, ranks_bev)

def parse_args():
    """Parse CLI arguments for FlashOCC ONNX export."""
    p = argparse.ArgumentParser(
        description='Export FlashOCC ONNX (torch_v3 + UnsortedSegmentSum BEV pool)')
    p.add_argument('--config', help='model config (relative to FlashOCC/)')
    p.add_argument('--checkpoint', help='checkpoint file')
    p.add_argument('--work_dir', help='output directory')
    p.add_argument('--prefix', default='flashocc_r50_torch_segmentsum')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--device', default='cpu')
    return p.parse_args()

def import_plugin(cfg, config_path):
    """Import custom mmdet3d_plugin modules so the registry is populated."""
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
    else:
        _module_dir = os.path.dirname(config_path)
    parts = _module_dir.split('/')
    _module_path = parts[0]
    for m in parts[1:]:
        _module_path = _module_path + '.' + m
    print(f'Importing plugin: {_module_path}')
    importlib.import_module(_module_path)

def _build_config(args):
    """Build and adjust FlashOCC config from CLI args."""
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_backbone.with_cp = False
    if not hasattr(cfg.model, 'wdet3d'):
        cfg.model.wdet3d = False
    if not hasattr(cfg.model, 'wocc'):
        cfg.model.wocc = True
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]
    import_plugin(cfg, args.config)
    return cfg

def _setup_test_pipeline(cfg):
    """Configure test pipeline (test_mode, replace ImageToTensor if needed)."""
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

def _build_test_loader(cfg):
    _setup_test_pipeline(cfg)
    test_loader_cfg = {
        'samples_per_gpu': 1, 'workers_per_gpu': 2,
        'dist': False, 'shuffle': False,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    return dataset, data_loader

def _build_model(cfg, args):
    """Build model, load checkpoint, set to eval mode."""
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        print(args.checkpoint, " does not exist!")
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model.to(args.device)
    model.eval()
    return model

def _get_export_inputs(model, data_loader, args):
    """Returns (img, ranks_bev, ranks_depth, ranks_feat, num_points, grid,
    wocc, wdet3d) extracted from sample 0."""
    data = next(iter(data_loader))
    inputs = [t.to(args.device) for t in data['img_inputs'][0]]
    # BEVDetOCCTRT.get_bev_pool_input returns a 5-tuple:
    # (ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths)
    metas = model.get_bev_pool_input(inputs)
    img = inputs[0]
    # Limit to 6 cameras (temporal-frame robustness, matches CUDA export)
    if img.shape[1] > 6:
        img = img[:, :6]
    ranks_bev, ranks_depth, ranks_feat = metas[0], metas[1], metas[2]
    num_points = ranks_bev.shape[0]
    grid = model.img_view_transformer.grid_size.tolist()
    wocc = getattr(model, 'wocc', True)
    wdet3d = getattr(model, 'wdet3d', False)
    return img, ranks_bev, ranks_depth, ranks_feat, num_points, grid, wocc, wdet3d

def _resolve_output_names(model, wocc, wdet3d):
    if wdet3d and not wocc:
        n_det = 6 * len(model.pts_bbox_head.task_heads)
        return [f'output_{j}' for j in range(n_det)]
    if wdet3d and wocc:
        n_det = 6 * len(model.pts_bbox_head.task_heads)
        return [f'output_{j}' for j in range(1 + n_det)]
    if not wdet3d and wocc:
        return [f'output_{j}' for j in range(1)]
    raise ValueError("At least one of wdet3d and wocc must be True!")

def _verify_onnx(output_path):
    try:
        import onnx  # pylint: disable=import-outside-toplevel
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print('ONNX Model Correct')
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'ONNX Model check failed: {e}')

def _export_onnx(wrapper, input_tensors, input_names, output_names,
                 output_path, opset, forward_method):
    """Export wrapper to ONNX file with given inputs/outputs."""
    print(f"\nExporting ONNX -> {output_path}")
    print(f"  output_names: {output_names}")
    wrapper.forward = forward_method
    torch.onnx.export(
        wrapper,
        input_tensors,
        output_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    print(f"Successfully exported to: {output_path}")
    _verify_onnx(output_path)

def main():
    """Entry point: export FlashOCC ONNX models."""
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    cfg = _build_config(args)
    _, data_loader = _build_test_loader(cfg)
    model = _build_model(cfg, args)

    (img, ranks_bev, ranks_depth, ranks_feat,
     num_points, grid, wocc, wdet3d) = _get_export_inputs(
        model, data_loader, args)

    print("Exporting FlashOCC ONNX (torch_v3 + UnsortedSegmentSum BEV pool)...")
    print(f"  img shape: {tuple(img.shape)}  BEV grid: {grid}  "
          f"ranks N_Points: {num_points}  wocc: {wocc}  wdet3d: {wdet3d}")

    wrapper = FlashOCCSegSumWrapper(model)
    wrapper.to(args.device).eval()

    output_names_ori = _resolve_output_names(model, wocc, wdet3d)
    input_names = ['img', 'ranks_depth', 'ranks_feat', 'ranks_bev']
    input_tensors = (img.float().contiguous(),
                     ranks_depth.int().contiguous(),
                     ranks_feat.int().contiguous(),
                     ranks_bev.int().contiguous())

    # Export 1: forward_ori (raw occ logits)
    _export_onnx(wrapper, input_tensors, input_names, output_names_ori,
                 os.path.join(args.work_dir, args.prefix + '.onnx'),
                 args.opset, wrapper.forward_ori)

    # Export 2: forward_with_argmax (per-voxel occ class label)
    _export_onnx(wrapper, input_tensors, input_names, ['cls_occ_label'],
                 os.path.join(args.work_dir,
                              args.prefix + '_with_argmax.onnx'),
                 args.opset, wrapper.forward_with_argmax)

if __name__ == '__main__':
    main()
