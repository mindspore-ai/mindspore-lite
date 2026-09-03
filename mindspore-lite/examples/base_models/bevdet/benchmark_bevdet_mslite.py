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
"""BEVDet benchmark with MindSpore Lite.

Handles three shape modes (auto-detected or explicit):
  fixed   — model compiled with fixed ranks (e.g., 179832); reuses sample 0's ranks
  dynamic — model compiled with ranks:-1 (pure dynamic); uses each sample's real ranks
  gear    — model compiled with ge.dynamicDims; pads each sample's ranks to the
            nearest upper dim

Pipeline: Main inference (NPU) → Decode (CPU PyTorch) → NMS (CPU numba)

Run from examples/base_models/bevdet/:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export PYTHONPATH=/path/to/BEVDet:/usr/local/lib/python3.10/site-packages
    python benchmark_bevdet_mslite.py \
        --config BEVDet/configs/bevdet/bevdet-r50.py \
        --checkpoint bevdet-r50.pth \
        --model output/bevdet_r50.mindir \
        --data-root BEVDet/data/nuscenes \
        --shape-mode fixed \
        --postprocessing \
        --eval \
        --output results/benchmark_mslite_result.txt
"""
import argparse
import gc
import os
import time

import numpy as np
import torch
import numba
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import replace_ImageToTensor
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


# ═══════════════════════════════════════════════════════════════════════════
# Numba Rotate NMS
# ═══════════════════════════════════════════════════════════════════════════

@numba.jit(nopython=True, cache=True)
def _box_corners(cx, cy, w, h, ry):
    """Compute 4 corners of a rotated 2D box."""
    cos_r = np.cos(ry)
    sin_r = np.sin(ry)
    hw, hh = w / 2.0, h / 2.0
    c = np.empty((4, 2))
    lx = (hw, -hw, -hw, hw)
    ly = (hh, hh, -hh, -hh)
    for i in range(4):
        c[i, 0] = cx + lx[i] * cos_r - ly[i] * sin_r
        c[i, 1] = cy + lx[i] * sin_r + ly[i] * cos_r
    return c


@numba.jit(nopython=True, cache=True)
def _signed_area(poly, n):
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return a / 2.0


@numba.jit(nopython=True, cache=True)
def _clip_sh(subject, ns, clip, nc):
    """Sutherland-Hodgman polygon clipping."""
    out = subject.copy()
    n_out = ns
    clip_ccw = _signed_area(clip, nc) > 0
    for i in range(nc):
        if n_out == 0:
            break
        p1x, p1y = clip[i, 0], clip[i, 1]
        p2x, p2y = clip[(i + 1) % nc, 0], clip[(i + 1) % nc, 1]
        dx, dy = p2x - p1x, p2y - p1y
        if clip_ccw:
            nx, ny = -dy, dx
        else:
            nx, ny = dy, -dx
        new_out = np.empty((8, 2))
        n_new = 0
        for j in range(n_out):
            k = (j + 1) % n_out
            x1, y1 = out[j, 0], out[j, 1]
            x2, y2 = out[k, 0], out[k, 1]
            d1 = (x1 - p1x) * nx + (y1 - p1y) * ny
            d2 = (x2 - p1x) * nx + (y2 - p1y) * ny
            if d1 >= 0:
                new_out[n_new, 0] = x1
                new_out[n_new, 1] = y1
                n_new += 1
                if d2 < 0:
                    t = d1 / (d1 - d2)
                    new_out[n_new, 0] = x1 + t * (x2 - x1)
                    new_out[n_new, 1] = y1 + t * (y2 - y1)
                    n_new += 1
            elif d2 >= 0:
                t = d1 / (d1 - d2)
                new_out[n_new, 0] = x1 + t * (x2 - x1)
                new_out[n_new, 1] = y1 + t * (y2 - y1)
                n_new += 1
        out = new_out
        n_out = n_new
    return out, n_out


@numba.jit(nopython=True, cache=True)
def _rotated_iou(b1, b2):
    """Compute rotated IoU between two 2D boxes."""
    cx1, cy1, w1, h1, r1 = b1[0], b1[1], b1[2], b1[3], b1[4]
    cx2, cy2, w2, h2, r2 = b2[0], b2[1], b2[2], b2[3], b2[4]
    rad = np.sqrt(w1 * w1 + h1 * h1) / 2.0 + np.sqrt(w2 * w2 + h2 * h2) / 2.0
    if (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 > rad * rad:
        return 0.0
    poly1 = _box_corners(cx1, cy1, w1, h1, r1)
    poly2 = _box_corners(cx2, cy2, w2, h2, r2)
    inter, ni = _clip_sh(poly1, 4, poly2, 4)
    if ni == 0:
        return 0.0
    area_inter = abs(_signed_area(inter, ni))
    area1 = w1 * h1
    area2 = w2 * h2
    return area_inter / (area1 + area2 - area_inter)


@numba.jit(nopython=True, cache=True)
def _rotate_nms(boxes, scores, thresh, post_max):
    """Numba-accelerated rotated NMS."""
    order = np.argsort(-scores)
    M = len(order)
    keep = np.empty(M, dtype=np.int64)
    n_keep = 0
    suppressed = np.zeros(M, dtype=np.bool_)
    for i in range(M):
        if suppressed[i]:
            continue
        keep[n_keep] = order[i]
        n_keep += 1
        for j in range(i + 1, M):
            if suppressed[j]:
                continue
            iou = _rotated_iou(boxes[order[i]], boxes[order[j]])
            if iou > thresh:
                suppressed[j] = True
    if 0 < post_max < n_keep:
        n_keep = post_max
    return keep[:n_keep]


def _empty_bbox_result():
    return {
        'boxes_3d': LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9),
        'scores_3d': torch.zeros(0),
        'labels_3d': torch.zeros(0, dtype=torch.long),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CPU Decode (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

def _atan2(y, x):
    return 2.0 * torch.atan(y / (torch.sqrt(x * x + y * y) + x))


def _gather_feat(feats, inds):
    dim = feats.size(2)
    inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
    return feats.gather(1, inds)


def _transpose_and_gather(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    return _gather_feat(feat, ind)


def decode_cpu(reg, height, dim, rot, vel, heatmap, head):
    """CPU Decode - 与 PyTorch decode 输出完全一致"""
    B, C, H, W = heatmap.shape
    coder = head.bbox_coder
    K = coder.max_num
    out_size_factor = coder.out_size_factor
    voxel_size = coder.voxel_size
    pc_range = coder.pc_range
    norm_bbox = head.norm_bbox  # 使用 head 的 norm_bbox

    # sigmoid
    heat = heatmap.sigmoid()

    # dim 处理
    if norm_bbox:
        dim = torch.exp(dim)

    # first topk (per class)
    topk_scores, topk_inds = torch.topk(heat.view(B, C, -1), K)

    # indices
    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds.float() / float(W)).int().float()
    topk_xs = (topk_inds % W).int().float()

    # second topk (global)
    topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), K)
    topk_clses = (topk_ind.float() / float(K)).int().float()

    # gather indices for global topk
    topk_inds_g = _gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, K)
    topk_ys_g = _gather_feat(topk_ys.view(B, -1, 1), topk_ind).view(B, K)
    topk_xs_g = _gather_feat(topk_xs.view(B, -1, 1), topk_ind).view(B, K)

    # gather features - reg
    reg_g = _transpose_and_gather(reg, topk_inds_g).view(B, K, 2)
    xs = topk_xs_g.view(B, K, 1) + reg_g[:, :, 0:1]
    ys = topk_ys_g.view(B, K, 1) + reg_g[:, :, 1:2]

    # gather features - rot (分别 gather 2 个 channel)
    rot_sine_g = _transpose_and_gather(rot[:, 0:1], topk_inds_g).view(B, K, 1)
    rot_cos_g = _transpose_and_gather(rot[:, 1:2], topk_inds_g).view(B, K, 1)
    rot_out = _atan2(rot_sine_g, rot_cos_g)

    # gather features - height, dim, vel
    hei_g = _transpose_and_gather(height, topk_inds_g).view(B, K, 1)
    dim_g = _transpose_and_gather(dim, topk_inds_g).view(B, K, 3)
    vel_g = _transpose_and_gather(vel, topk_inds_g).view(B, K, 2)

    # coordinate transform
    xs = xs * (out_size_factor * voxel_size[0]) + pc_range[0]
    ys = ys * (out_size_factor * voxel_size[1]) + pc_range[1]

    final_boxes = torch.cat([xs, ys, hei_g, dim_g, rot_out, vel_g], dim=2)
    return final_boxes, topk_score, topk_clses


def run_nms(model, boxes_np, scores_np, labels_np):
    """Score threshold + range filter + numba rotate NMS"""
    head = model.pts_bbox_head
    coder = head.bbox_coder
    test_cfg = head.test_cfg

    boxes = torch.from_numpy(boxes_np).float()
    scores = torch.from_numpy(scores_np).float()
    labels = torch.from_numpy(labels_np).long()

    mask = scores > coder.score_threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    pcr = torch.tensor(coder.post_center_range, dtype=torch.float)
    if pcr.shape[0] == 6:
        rmask = (boxes[:, :3] >= pcr[:3]).all(1) & \
                (boxes[:, :3] <= pcr[3:]).all(1)
        boxes, scores, labels = boxes[rmask], scores[rmask], labels[rmask]

    if boxes.shape[0] == 0:
        return _empty_bbox_result()

    nms_thr = test_cfg['nms_thr']
    if isinstance(nms_thr, list):
        nms_thr = nms_thr[0]
    pre_max = test_cfg.get('pre_max_size', 1000)
    post_max = test_cfg.get('post_max_size', 500)
    rescale = test_cfg.get('nms_rescale_factor', [[1.0]])
    if isinstance(rescale, list) and len(rescale) > 0:
        rescale = rescale[0]

    order = scores.argsort(descending=True)
    if 0 < pre_max < len(order):
        order = order[:pre_max]
    boxes_s = boxes[order].clone()
    scores_s = scores[order]
    labels_s = labels[order]

    for cid, rescale_val in enumerate(rescale):
        cmask = labels_s == cid
        if cmask.any():
            boxes_s[cmask, 3:6] *= rescale_val

    bev = torch.stack([
        boxes_s[:, 0], boxes_s[:, 1], boxes_s[:, 3],
        boxes_s[:, 4], boxes_s[:, 6],
    ], dim=1).numpy().astype(np.float64)
    sc = scores_s.numpy().astype(np.float64)

    keep_local = _rotate_nms(bev, sc, float(nms_thr), post_max)
    keep = order[torch.from_numpy(keep_local).long()]

    sel_boxes = boxes[keep]
    sel_scores = scores[keep]
    sel_labels = labels[keep].int()

    if sel_boxes.shape[0] > 0:
        sel_boxes[:, 2] = sel_boxes[:, 2] - sel_boxes[:, 5] * 0.5
    bboxes = LiDARInstance3DBoxes(sel_boxes, coder.code_size)
    return bbox3d2result(bboxes, sel_scores, sel_labels)


# ═══════════════════════════════════════════════════════════════════════════
# Args & Utils
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description='BEVDet benchmark with CPU Decode + CPU NMS')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--data-root', default=None,
                   help='override cfg.data.test.data_root and ann_file '
                        'to run from this dir without copying scripts into '
                        'the BEVDet repo')
    p.add_argument('--shape-mode', choices=['auto', 'fixed', 'dynamic', 'gear'],
                   default='fixed', help='shape mode for ranks')
    p.add_argument('--fixed-ranks-len', type=int, default=179832)
    p.add_argument('--gear-dims', default='179354,179655,179955',
                   help='comma-separated dim for gear mode')
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--samples', type=int, default=0,
                   help='number of samples to benchmark (0 = all)')
    p.add_argument('--postprocessing', action='store_true',
                   help='run postprocessing (decode + NMS)')
    p.add_argument('--eval', action='store_true',
                   help='run mAP/NDS evaluation on NuScenes val set')
    p.add_argument('--output', default=None)
    p.add_argument('--cpu-affinity', default=None,
                   help='comma-separated CPU core IDs (e.g., "0,1,2,3"). '
                        'If not specified, auto-detect from NPU NUMA node.')
    return p.parse_args()


def _override_data_root(cfg, data_root):
    """Override data_root and ann_file paths so scripts can run from this dir.

    BEVDet configs concatenate data_root + filename at parse time, so
    ann_file='data/nuscenes/xxx.pkl' already embeds the full prefix. Rewrite
    both fields by replacing the config's original data_root prefix with
    --data-root (the actual nuscenes dir).
    """
    if not data_root:
        return cfg
    data_root = os.path.abspath(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'--data-root not a directory: {data_root}')

    def _repath(ann, old_root):
        if old_root and ann.startswith(old_root):
            return os.path.join(data_root, ann[len(old_root):])
        if ann.startswith('data/'):
            return os.path.join(data_root, ann[len('data/'):])
        return ann

    def _patch(item):
        if not isinstance(item, dict):
            return
        old_root = item.get('data_root', '')
        if old_root:
            item['data_root'] = data_root + '/'
        if 'ann_file' in item:
            item['ann_file'] = _repath(item['ann_file'], old_root)
        if 'ann_files' in item:
            item['ann_files'] = [_repath(a, old_root)
                                 for a in item['ann_files']]
        for v in item.values():
            if isinstance(v, dict):
                _patch(v)
            elif isinstance(v, list):
                for x in v:
                    if isinstance(x, dict):
                        _patch(x)

    for key in ('train', 'val', 'test'):
        ds_cfg = cfg.data.get(key, None)
        if isinstance(ds_cfg, dict):
            _patch(ds_cfg)
        elif isinstance(ds_cfg, list):
            for x in ds_cfg:
                _patch(x)
    return cfg


def _rewrite_dataset_paths(dataset, data_root):
    """Rewrite relative data paths baked into the info pkl.

    The pkl stores cam/lidar paths like './data/nuscenes/samples/...', which
    only resolve when cwd is the BEVDet repo root. Point them at data_root so
    the dataloader finds images from any cwd.
    """
    prefixes = ('./data/nuscenes/', 'data/nuscenes/')
    data_root = os.path.abspath(data_root)

    def _fix(value):
        if isinstance(value, str):
            for pre in prefixes:
                if value.startswith(pre):
                    return os.path.join(data_root, value[len(pre):])
            return value
        if isinstance(value, dict):
            return {k: _fix(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_fix(v) for v in value]
        return value

    if getattr(dataset, 'data_infos', None):
        dataset.data_infos = [_fix(info) for info in dataset.data_infos]


def build_model_and_loader(args):
    """Build mmdet3d model and test data loader from args."""
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_backbone.with_cp = False
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]
    cfg = _override_data_root(cfg, args.data_root)
    cfg.data.test_dataloader.workers_per_gpu = 2
    assert cfg.data.test.test_mode
    default_args = {'samples_per_gpu': 1, 'workers_per_gpu': 0, 'dist': False,
                    'shuffle': False}
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    test_loader_cfg = {**default_args, **cfg.data.get('test_dataloader', {})}
    dataset = build_dataset(cfg.data.test)
    if args.data_root:
        _rewrite_dataset_paths(dataset, args.data_root)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to('cpu').eval()
    return cfg, model, dataset, data_loader


def detect_shape_mode(model_path, ctx, n_actual, fixed_len, gear_dims):
    """Auto-detect shape mode by trying resize with FRESH model per attempt."""
    import mindspore_lite as mslite
    test_len = n_actual + 1 if n_actual < 200000 else n_actual - 1
    for mode, shapes in [
                            ('dynamic', [[1, 6, 3, 256, 704], [test_len], [test_len], [test_len]]),
                            ('fixed', [[1, 6, 3, 256, 704], [fixed_len], [fixed_len], [fixed_len]]),
                        ] + [('gear', [[1, 6, 3, 256, 704], [d], [d], [d]]) for d in gear_dims]:
        try:
            m = mslite.Model()
            m.build_from_file(model_path, mslite.ModelType.MINDIR, ctx)
            mi = m.get_inputs()
            m.resize(mi, shapes)
            if mode == 'gear':
                return 'gear', m
            return mode, m
        except Exception:
            pass
    raise RuntimeError('Cannot detect shape mode — use explicit --shape-mode')


def pad_ranks_to_gear(rd, rf, rb, gear_dims):
    """Pad ranks to nearest upper dim."""
    n = len(rb)
    upper = [d for d in gear_dims if d >= n]
    if upper:
        target = min(upper)
    else:
        target = max(gear_dims)
    if target > n:
        pad = target - n
        rd = np.pad(rd, (0, pad), constant_values=0)
        rf = np.pad(rf, (0, pad), constant_values=0)
        rb = np.pad(rb, (0, pad), constant_values=0)
    elif target < n:
        rd, rf, rb = rd[:target], rf[:target], rb[:target]
    return rd, rf, rb, target


# ═══════════════════════════════════════════════════════════════════════════
# Constants for evaluation report
# ═══════════════════════════════════════════════════════════════════════════

CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian',
           'traffic_cone']
ERR_NAMES = {'mATE': 'mATE', 'mASE': 'mASE', 'mAOE': 'mAOE',
             'mAVE': 'mAVE', 'mAAE': 'mAAE'}


# ═══════════════════════════════════════════════════════════════════════════
# Main helpers (extracted from main for CCN/NLOC compliance)
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_runtime():
    """Parse args, adjust eval flag, import mslite, build model/loader, create Context."""
    args = parse_args()
    if args.eval:
        args.postprocessing = True
        print('  Note: --eval requires postprocessing, enabled automatically')

    import mindspore_lite as mslite
    gear_dims = [int(x) for x in args.gear_dims.split(',')]
    cfg, model, dataset, data_loader = build_model_and_loader(args)

    ctx = mslite.Context()
    ctx.target = ['ascend']
    ctx.ascend.device_id = args.device_id
    return args, mslite, gear_dims, cfg, model, dataset, data_loader, ctx


def _detect_numa_cpus(device_id):
    """Auto-detect CPU cores on the same NUMA node as the NPU device."""
    import re
    import subprocess
    try:
        result = subprocess.run(['npu-smi', 'info'], capture_output=True,
                                text=True, timeout=5, check=False)
        bus_ids = re.findall(r'\d{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d',
                             result.stdout)
        if device_id >= len(bus_ids):
            return None
        bus_id = bus_ids[device_id].lower()
        with open(f'/sys/bus/pci/devices/{bus_id}/numa_node', encoding='utf-8') as f:
            numa_node = int(f.read().strip())
        if numa_node < 0:
            return None
        with open(f'/sys/devices/system/node/node{numa_node}/cpulist', encoding='utf-8') as f:
            cpulist = f.read().strip()
        cpus = set()
        for part in cpulist.split(','):
            if '-' in part:
                s, e = part.split('-')
                cpus.update(range(int(s), int(e) + 1))
            else:
                cpus.add(int(part))
        return cpus if cpus else None
    except Exception:
        return None


def _stabilize_cpu(args):
    """Pin CPU affinity, disable GC, set torch threads for stable timing."""
    if args.cpu_affinity:
        cpus = set(int(x) for x in args.cpu_affinity.split(','))
    else:
        cpus = _detect_numa_cpus(args.device_id)
        if cpus is None:
            cpus = {0, 1, 2, 3}
    try:
        os.sched_setaffinity(0, cpus)
        print(f'  CPU affinity: {sorted(os.sched_getaffinity(0))}')
    except Exception:
        print(f'  CPU affinity: set failed (requested {sorted(cpus)})')
    torch.set_num_threads(1)
    gc.disable()
    print('  GC disabled, torch threads=1')


def _get_sample0_ranks(model, data_loader):
    """Extract sample 0 ranks for baseline + shape detection."""
    # Get sample 0 for ranks baseline + shape detection
    for i, data in enumerate(data_loader):
        if i == 0:
            inputs = [t.to('cpu') for t in data['img_inputs'][0]]
            metas = model.get_bev_pool_input(inputs)
            s0_ranks = {
                'ranks_depth': metas[1].int().cpu().numpy().astype(np.int32),
                'ranks_feat': metas[2].int().cpu().numpy().astype(np.int32),
                'ranks_bev': metas[0].int().cpu().numpy().astype(np.int32),
            }
            break
    return s0_ranks


def _build_ms_model(args, mslite, ctx, s0_n, gear_dims):
    """Detect shape mode or build model from file."""
    mode = args.shape_mode
    if mode == 'auto':
        mode, ms_model = detect_shape_mode(args.model, ctx, s0_n,
                                           args.fixed_ranks_len, gear_dims)
    else:
        ms_model = mslite.Model()
        ms_model.build_from_file(args.model, mslite.ModelType.MINDIR, ctx)
    return mode, ms_model


def _print_header_and_verify(args, mode, ms_model, s0_ranks, s0_n, gear_dims):
    """Print header, determine target_len, resize model, verify."""
    print(f'\n=== BEVDet CPU Decode Benchmark (shape_mode={mode}) ===')
    print(f'  model: {args.model}')
    print('  postprocessing: Decode (CPU) + NMS (CPU)')
    print(f'  gear_dims: {gear_dims}' if mode == 'gear' else '')
    print(f'  sample 0 ranks: {s0_n}')

    # Verify mode works on sample 0
    if mode == 'fixed':
        target_len = args.fixed_ranks_len
    elif mode == 'dynamic':
        target_len = s0_n
    elif mode == 'gear':
        _, _, _, target_len = pad_ranks_to_gear(
            s0_ranks['ranks_depth'], s0_ranks['ranks_feat'],
            s0_ranks['ranks_bev'], gear_dims)
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    ms_inputs = ms_model.get_inputs()
    ms_model.resize(ms_inputs, [[1, 6, 3, 256, 704], [target_len], [target_len], [target_len]])
    ms_inputs = ms_model.get_inputs()
    print(f'  verified: resize to [{target_len}] OK')
    return target_len, ms_inputs


def _warmup(ms_model, ms_inputs, s0_ranks, model, postprocessing):
    """Warm up NPU graph + numba JIT + PyTorch decode/NMS path."""
    print('  warming up NPU...')
    dummy_img = np.zeros((1, 6, 3, 256, 704), dtype=np.float32)
    for inp in ms_inputs:
        if inp.name == 'img':
            inp.set_data_from_numpy(dummy_img)
        elif inp.name in s0_ranks:
            inp.set_data_from_numpy(s0_ranks[inp.name])
    for _ in range(3):
        ms_outputs = ms_model.predict(ms_inputs)

    if postprocessing:
        print('  warming up numba JIT + PyTorch decode/NMS...')
        _db = np.ones((10, 9), dtype=np.float64)
        _ds = np.linspace(0.9, 0.1, 10).astype(np.float64)
        _rotate_nms(_db[:, :5], _ds, 0.2, 500)
        head = model.pts_bbox_head
        raw = [out.get_data_to_numpy() for out in ms_outputs]
        for _ in range(2):
            decode_cpu(torch.from_numpy(raw[0]), torch.from_numpy(raw[1]),
                      torch.from_numpy(raw[2]), torch.from_numpy(raw[3]),
                      torch.from_numpy(raw[4]), torch.from_numpy(raw[5]), head)


def _preprocess_sample(data, mode, s0_ranks, model, ms_model, ms_inputs,
                       resize_done, gear_dims, n_min, n_max, pad_count):
    """Preprocess one sample: extract inputs, compute ranks, resize, set feeds."""
    inputs = [t.to('cpu') for t in data['img_inputs'][0]]
    img_tensor = inputs[0]
    if img_tensor.dtype != torch.float32:
        img_np = img_tensor.float().numpy()
    else:
        img_np = img_tensor.numpy()

    if mode == 'fixed':
        rd, rf, rb = (s0_ranks['ranks_depth'], s0_ranks['ranks_feat'],
                      s0_ranks['ranks_bev'])
    else:
        metas_ = model.get_bev_pool_input(inputs)
        rd = metas_[1].int().cpu().numpy().astype(np.int32)
        rf = metas_[2].int().cpu().numpy().astype(np.int32)
        rb = metas_[0].int().cpu().numpy().astype(np.int32)
        n_actual = len(rb)
        n_min, n_max = min(n_min, n_actual), max(n_max, n_actual)
        if mode == 'dynamic':
            target_len = n_actual
        elif mode == 'gear':
            rd, rf, rb, target_len = pad_ranks_to_gear(rd, rf, rb, gear_dims)
            if target_len > n_actual:
                pad_count += 1

    if not resize_done:
        ms_model.resize(ms_inputs,
                        [[1, 6, 3, 256, 704], [target_len], [target_len], [target_len]])
        ms_inputs = ms_model.get_inputs()

    feeds = {'img': img_np, 'ranks_depth': rd, 'ranks_feat': rf,
             'ranks_bev': rb}
    for inp in ms_inputs:
        if inp.name in feeds:
            inp.set_data_from_numpy(feeds[inp.name])

    return ms_inputs, n_min, n_max, pad_count

def _postprocess_sample(ms_outputs, ms_inputs, model, args, i, head,
                        results, nan_sample_count):
    """Postprocess one sample: decode, NMS, NaN check, append results.
    Returns (results, nan_sample_count, decode_time, nms_time)."""
    raw_outputs = [out.get_data_to_numpy() for out in ms_outputs]
    reg_t = torch.from_numpy(raw_outputs[0])
    height_t = torch.from_numpy(raw_outputs[1])
    dim_t = torch.from_numpy(raw_outputs[2])
    rot_t = torch.from_numpy(raw_outputs[3])
    vel_t = torch.from_numpy(raw_outputs[4])
    heatmap_t = torch.from_numpy(raw_outputs[5])

    _td0 = time.perf_counter()
    final_boxes, final_scores, final_labels = decode_cpu(
        reg_t, height_t, dim_t, rot_t, vel_t, heatmap_t, head)
    _td1 = time.perf_counter()

    bad_boxes = torch.isnan(final_boxes).any() or torch.isinf(final_boxes).any()
    bad_scores = torch.isnan(final_scores).any() or torch.isinf(final_scores).any()
    if bad_boxes or bad_scores:
        print(f'WARNING: NaN/Inf detected in decode output at sample {i}')
        nan_sample_count += 1
        if bad_boxes:
            final_boxes = torch.nan_to_num(final_boxes, nan=0.0, posinf=1e6,
                                           neginf=-1e6)
        if bad_scores:
            final_scores = torch.nan_to_num(final_scores, nan=0.0,
                                            posinf=1.0, neginf=0.0)

    _tn0 = time.perf_counter()
    try:
        res = run_nms(model, final_boxes[0].numpy(),
                      final_scores[0].numpy(), final_labels[0].numpy())
        if args.eval:
            results.append(res)
    except Exception as e:
        print(f'Run NMS failed for [{ms_inputs[-1].name}], {str(e)}')
        if args.eval:
            results.append(_empty_bbox_result())
    _tn1 = time.perf_counter()

    return results, nan_sample_count, _td1 - _td0, _tn1 - _tn0


def _print_timing_breakdown(timed, num_warmup, t_prep, t_inf, t_decode,
                            t_nms, t_total):
    """Print timing breakdown."""
    if not t_total:
        return
    total_mean = np.mean(t_total) * 1000
    print(f'\n  === Timing Breakdown ({timed} samples, warmup={num_warmup}) ===')
    print('  (FPS excludes data preparation)\n')
    phases = [('Data prep (not counted)', t_prep, True),
              ('Inference (NPU)', t_inf, False),
              ('Decode (CPU)', t_decode, False),
              ('NMS (CPU)', t_nms, False)]
    print(f'  {"Phase":<26} {"Mean(ms)":>10} {"%":>6}')
    print(f'  {"-" * 26} {"-" * 10} {"-" * 6}')
    for name, arr, skip_pct in phases:
        m = np.mean(arr) * 1000 if arr else 0.0
        pct = 0 if skip_pct else m / total_mean * 100
        print(f'  {name:<26} {m:>10.2f} {pct:>5.1f}%')
    print(f'  {"Total":<26} {total_mean:>10.2f} {"100.0":>5}%')


def _print_overall(args, mode, total, timed, num_warmup, fps,
                   n_min, n_max, pad_count, nan_sample_count):
    """Print overall summary."""
    print('\n  === Overall ===')
    print(f'    shape_mode: {mode}')
    print(f'    Total samples: {total}')
    print(f'    Timed samples: {timed} (warmup={num_warmup})')
    print(f'    FPS: {fps:.2f} img/s')
    if fps > 0:
        print(f'    Inference time: {1000 / fps:.2f} ms')
    if mode == 'dynamic':
        print(f'    ranks range: {n_min}~{n_max} (per-sample)')
    elif mode == 'gear':
        print(f'    ranks range: {n_min}~{n_max}, '
              f'padded samples: {pad_count}/{total}')
    elif mode == 'fixed':
        print(f'    ranks: fixed {args.fixed_ranks_len} (reused sample 0)')
    print(f'    postprocessing: {args.postprocessing}, eval: {args.eval}')
    if nan_sample_count > 0:
        print(f'    WARNING: {nan_sample_count}/{total} samples had NaN/Inf in '
              f'decode output (silently replaced); metrics may be unreliable')


def _write_no_eval_report(args, mode, total, fps, nan_sample_count):
    """Write no-eval report file and return."""
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write('=' * 70 + '\n')
            f.write(f'BEVDet CPU Decode benchmark (shape_mode={mode}, '
                    f'no eval)\n')
            f.write('=' * 70 + '\n\n')
            f.write(f'model:      {args.model}\n')
            f.write(f'shape_mode: {mode}\n')
            f.write('postprocess: CPU Decode + CPU NMS\n')
            f.write(f'Samples:    {total}\n')
            f.write(f'FPS:        {fps:.2f} img/s\n')
            if fps > 0:
                f.write(f'Latency:    {1000 / fps:.2f} ms\n')
            if nan_sample_count > 0:
                f.write(f'NaN/Inf samples: {nan_sample_count}/{total} '
                        f'(metrics may be unreliable)\n')
            f.write('\n' + '=' * 70 + '\n')
        print(f'\n  Results saved to {args.output}')


def _run_evaluation(cfg, dataset, results):
    """Run NuScenes evaluation and return eval_result."""
    print(f'\n  Collected {len(results)} results for evaluation')
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    eval_kwargs.update({'metric': 'mAP'})
    print('\n  === Evaluation (NuScenes mAP/NDS) ===')
    try:
        eval_result = dataset.evaluate(results, **eval_kwargs)
    except Exception as e:
        print(f'  WARNING: evaluation crashed ({e}); using zero metrics.')
        _p = 'pts_bbox_NuScenes/'
        eval_result = {_p + k: 0.0 for k in
                       ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE',
                        'mAAE']}
    print(eval_result)
    return eval_result


def _write_eval_report_meta(f, args, mode, gear_dims, total, fps,
                            n_min, n_max, pad_count, nan_sample_count):
    """Write eval report header and metadata."""
    f.write('=' * 70 + '\n')
    f.write(f'BEVDet CPU Decode benchmark (shape_mode={mode})\n')
    f.write('=' * 70 + '\n\n')
    f.write(f'model:       {args.model}\n')
    f.write(f'shape_mode:  {mode}\n')
    f.write('postprocess: CPU Decode + CPU NMS\n')
    if mode == 'gear':
        f.write(f'gear_dims:   {gear_dims}\n')
    f.write(f'Config:      {args.config}\n')
    f.write(f'Checkpoint:  {args.checkpoint}\n')
    f.write(f'Samples:     {total}\n')
    f.write(f'FPS:         {fps:.2f} img/s\n')
    if fps > 0:
        f.write(f'Latency:     {1000 / fps:.2f} ms\n')
    if mode == 'dynamic':
        f.write(f'ranks range: {n_min}~{n_max} (per-sample)\n')
    elif mode == 'gear':
        f.write(f'ranks range: {n_min}~{n_max}, '
                f'padded: {pad_count}/{total}\n')
    elif mode == 'fixed':
        f.write(f'ranks: fixed {args.fixed_ranks_len} '
                f'(reused sample 0)\n')
    f.write('\n')
    if nan_sample_count > 0:
        f.write(f'NaN/Inf samples: {nan_sample_count}/{total} '
                f'(metrics may be unreliable)\n\n')
    f.write('-' * 70 + '\n')


def _write_eval_metrics(f, eval_result, prefix):
    """Write overall metrics and error metrics."""
    f.write('Overall Metrics\n')
    f.write('-' * 70 + '\n')
    f.write(f'{"mAP":<12} {eval_result[prefix + "mAP"]:.4f}\n')
    f.write(f'{"NDS":<12} {eval_result[prefix + "NDS"]:.4f}\n')
    f.write('\n')
    f.write('-' * 70 + '\n')
    f.write('Error Metrics (lower is better)\n')
    f.write('-' * 70 + '\n')
    for short, key in ERR_NAMES.items():
        f.write(f'{short:<12} '
                f'{eval_result.get(prefix + key, float("nan")):.4f}\n')
    f.write('\n')


def _write_eval_per_class(f, eval_result, prefix):
    """Write per-class AP and errors table."""
    f.write('-' * 70 + '\n')
    f.write('Per-Class AP and Errors\n')
    f.write('-' * 70 + '\n')
    hdr = (f'{"Class":<22} {"AP":>8} {"ATE":>8} {"ASE":>8} '
           f'{"AOE":>8} {"AVE":>8} {"AAE":>8}')
    f.write(hdr + '\n')
    f.write('-' * len(hdr) + '\n')
    for cls in CLASSES:
        ap_vals = [eval_result.get(
            f'{prefix}{cls}_AP_dist_{d}', 0.0)
            for d in [0.5, 1.0, 2.0, 4.0]]
        ap = sum(ap_vals) / len(ap_vals) if ap_vals else 0.0
        ate = eval_result.get(f'{prefix}{cls}_trans_err',
                              float('nan'))
        ase = eval_result.get(f'{prefix}{cls}_scale_err',
                              float('nan'))
        aoe = eval_result.get(f'{prefix}{cls}_orient_err',
                              float('nan'))
        ave = eval_result.get(f'{prefix}{cls}_vel_err',
                              float('nan'))
        aae = eval_result.get(f'{prefix}{cls}_attr_err',
                              float('nan'))

        def _fmt(v):
            return f'{v:.4f}' if v == v else '  nan'  # pylint: disable=comparison-with-itself

        f.write(f'{cls:<22} {ap:>8.4f} {_fmt(ate):>8} '
                f'{_fmt(ase):>8} {_fmt(aoe):>8} '
                f'{_fmt(ave):>8} {_fmt(aae):>8}\n')
    f.write('\n')


def _write_eval_per_distance(f, eval_result, prefix):
    """Write per-distance AP table."""
    f.write('-' * 70 + '\n')
    f.write('Raw Per-Distance AP\n')
    f.write('-' * 70 + '\n')
    hdr2 = (f'{"Class":<22} {"AP@0.5m":>10} {"AP@1.0m":>10} '
            f'{"AP@2.0m":>10} {"AP@4.0m":>10}')
    f.write(hdr2 + '\n')
    f.write('-' * len(hdr2) + '\n')
    for cls in CLASSES:
        vals = [eval_result.get(
            f'{prefix}{cls}_AP_dist_{d}', 0.0)
            for d in [0.5, 1.0, 2.0, 4.0]]
        f.write(f'{cls:<22} {vals[0]:>10.4f} {vals[1]:>10.4f} '
                f'{vals[2]:>10.4f} {vals[3]:>10.4f}\n')


def _write_eval_report(args, mode, gear_dims, total, fps,
                       n_min, n_max, pad_count, nan_sample_count,
                       eval_result):
    """Write evaluation report file."""
    if not args.output:
        return
    prefix = 'pts_bbox_NuScenes/'
    with open(args.output, 'w', encoding='utf-8') as f:
        _write_eval_report_meta(f, args, mode, gear_dims, total, fps,
                                n_min, n_max, pad_count, nan_sample_count)
        _write_eval_metrics(f, eval_result, prefix)
        _write_eval_per_class(f, eval_result, prefix)
        _write_eval_per_distance(f, eval_result, prefix)
        f.write('\n' + '=' * 70 + '\n')
    print(f'\n  Results saved to {args.output}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run BEVDet benchmark: inference, optional decode/NMS, optional eval."""
    args, mslite, gear_dims, cfg, model, dataset, data_loader, ctx = _prepare_runtime()

    s0_ranks = _get_sample0_ranks(model, data_loader)
    s0_n = len(s0_ranks['ranks_bev'])

    mode, ms_model = _build_ms_model(args, mslite, ctx, s0_n, gear_dims)

    _, ms_inputs = _print_header_and_verify(args, mode, ms_model, s0_ranks, s0_n, gear_dims)

    _stabilize_cpu(args)
    _warmup(ms_model, ms_inputs, s0_ranks, model, args.postprocessing)

    num_warmup = args.warmup
    t_prep, t_inf, t_decode, t_nms, t_total = [], [], [], [], []
    results = []
    n_min, n_max = 10 ** 9, 0
    pad_count = 0
    nan_sample_count = 0
    head = model.pts_bbox_head
    resize_done = mode == 'fixed'

    i = -1
    for i, data in enumerate(data_loader):
        if args.samples > 0 and i >= args.samples:
            break

        t0 = time.perf_counter()

        ms_inputs, n_min, n_max, pad_count = _preprocess_sample(
            data, mode, s0_ranks, model, ms_model, ms_inputs,
            resize_done, gear_dims, n_min, n_max, pad_count)

        t1 = time.perf_counter()

        ms_outputs = ms_model.predict(ms_inputs)

        t2 = time.perf_counter()

        if args.postprocessing:
            results, nan_sample_count, dt_decode, dt_nms = \
                _postprocess_sample(ms_outputs, ms_inputs, model, args, i,
                                    head, results, nan_sample_count)

        t3 = time.perf_counter()

        if i >= num_warmup:
            t_prep.append(t1 - t0)
            t_inf.append(t2 - t1)
            if args.postprocessing:
                t_decode.append(dt_decode)
                t_nms.append(dt_nms)
            t_total.append(t3 - t1)
            if (i + 1 - num_warmup) % 50 == 0:
                fps = (i + 1 - num_warmup) / sum(t_total)
                print(f'  Done image [{i + 1:>4}], fps: {fps:.2f} img/s')

    total = i + 1
    timed = max(total - num_warmup, 1)
    pure_inf = sum(t_total)
    fps = timed / pure_inf if pure_inf > 0 else 0

    _print_timing_breakdown(timed, num_warmup, t_prep, t_inf, t_decode,
                            t_nms, t_total)
    _print_overall(args, mode, total, timed, num_warmup, fps,
                   n_min, n_max, pad_count, nan_sample_count)

    gc.enable()

    eval_result = None

    if not args.eval:
        _write_no_eval_report(args, mode, total, fps, nan_sample_count)
        return

    eval_result = _run_evaluation(cfg, dataset, results)
    _write_eval_report(args, mode, gear_dims, total, fps,
                       n_min, n_max, pad_count, nan_sample_count,
                       eval_result)

if __name__ == '__main__':
    main()
