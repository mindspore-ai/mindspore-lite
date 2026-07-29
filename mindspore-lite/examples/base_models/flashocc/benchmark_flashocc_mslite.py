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
"""Universal benchmark for FlashOCC MindIR on Ascend NPU.

Mirrors tools/analysis_tools/benchmark_trt.py but replaces TensorRT with
MindSpore Lite. Supports wdet3d (3D detection) and wocc (occupancy),
with --postprocessing and --eval flags matching the TRT benchmark.

Handles three shape modes (auto-detected or explicit):
  fixed   — model compiled with fixed ranks; reuses sample 0's ranks
  dynamic — model compiled with ranks:-1; uses each sample's real ranks
  gear    — model compiled with ge.dynamicDims; pads to nearest gear dim

Usage (run from FlashOCC/):
    cd FlashOCC

    # Occupancy only (wocc=True, wdet3d=False):
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/benchmark_flashocc_mslite.py \
        --config projects/configs/flashocc/flashocc-r50.py \
        --checkpoint flashocc-r50-256x704.pth \
        --model output/flashocc_r50_with_argmax.mindir \
        --eval --output results/benchmark_flashocc_result.txt

    # Detection + occupancy (wdet3d=True, wocc=True):
    ASCEND_RT_VISIBLE_DEVICES=0 python scripts/benchmark_flashocc_mslite.py \
        --config projects/configs/panoptic-flashocc/panoptic-flashocc-r50-depth-pano.py \
        --checkpoint panoptic-flashocc.pth \
        --model output/panoptic.mindir \
        --postprocessing --eval

    # Explicit shape mode:
    --shape-mode gear --gear-dims 300674,300974,301274

"""
import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.getcwd()) # pylint: disable=wrong-import-position

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet.datasets import replace_ImageToTensor # pylint: disable=ungrouped-imports

from mmdet3d.core import bbox3d2result # pylint: disable=ungrouped-imports
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

class Tee:
    """Duplicate stdout to both console and a log file."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w', encoding='utf-8')  # pylint: disable=consider-using-with
        self.stdout = sys.stdout

    def write(self, msg):
        """Write message to both stdout and the log file."""
        self.stdout.write(msg)
        self.file.write(msg)

    def flush(self):
        """Flush both stdout and the log file."""
        self.stdout.flush()
        self.file.flush()

    def close(self):
        """Close the log file."""
        self.file.close()

def _empty_bbox_result():
    return {
        'boxes_3d': LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9),
        'scores_3d': torch.zeros(0),
        'labels_3d': torch.zeros(0, dtype=torch.long),
    }

def parse_args():
    """Parse CLI arguments for FlashOCC benchmark."""
    p = argparse.ArgumentParser(
        description='Full FlashOCC benchmark (wdet3d + wocc, fixed/dynamic/gear)')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--shape-mode', choices=['auto', 'fixed', 'dynamic', 'gear'],
                   default='auto')
    p.add_argument('--fixed-ranks-len', type=int, default=300974)
    p.add_argument('--gear-dims', default='300674,300974,301274')
    p.add_argument('--device-id', type=int, default=0)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--samples', type=int, default=0,
                   help='number of samples to benchmark (0 = all)')
    p.add_argument('--postprocessing', action='store_true',
                   help='run postprocessing (det3d decode+NMS and/or occ get_occ)')
    p.add_argument('--eval', action='store_true',
                   help='run evaluation (mAP/NDS for det3d, mIoU for occ)')
    p.add_argument('--output', default=None,
                   help='save benchmark results to this file')
    return p.parse_args()

def import_plugin(cfg, config_path):
    """Import custom plugin modules declared in the config."""
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        _module_dir = os.path.dirname(cfg.plugin_dir)
    else:
        _module_dir = os.path.dirname(config_path)
    parts = _module_dir.split('/')
    _module_path = parts[0]
    for m in parts[1:]:
        _module_path = _module_path + '.' + m
    print(f'Importing plugin: {_module_path}')
    importlib.import_module(_module_path)

def build_model_and_loader(args):
    """Build the FlashOCC model and test data loader from CLI args."""
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
    cfg.data.test_dataloader.workers_per_gpu = 0
    assert cfg.data.test.test_mode
    default_args = {"samples_per_gpu": 1, "workers_per_gpu": 0,
                    "dist": False, "shuffle": False}
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    test_loader_cfg = {**default_args, **cfg.data.get('test_dataloader', {})}
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to('cpu').eval()
    return cfg, model, dataset, data_loader

def detect_shape_mode(model_path, ctx, n_actual, fixed_len, gear_dims):
    """Auto-detect shape mode (fixed/dynamic/gear) by trial-resizing the model."""
    import mindspore_lite as mslite  # pylint: disable=import-outside-toplevel
    test_len = n_actual + 1 if n_actual < 400000 else n_actual - 1
    for mode, shapes in [
        ('dynamic', [[1, 6, 3, 256, 704], [test_len], [test_len], [test_len]]),
        ('fixed', [[1, 6, 3, 256, 704], [fixed_len], [fixed_len], [fixed_len]]),
    ] + [('gear', [[1, 6, 3, 256, 704], [d], [d], [d]]) for d in gear_dims]:
        try:
            m = mslite.Model()
            m.build_from_file(model_path, mslite.ModelType.MINDIR, ctx)
            mi = m.get_inputs()
            m.resize(mi, shapes)
            return mode, m
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    raise RuntimeError('Cannot detect shape mode — use explicit --shape-mode')

def pad_ranks_to_gear(rd, rf, rb, gear_dims):
    """Pad or truncate ranks arrays to the nearest gear dimension."""
    n = len(rb)
    upper = [d for d in gear_dims if d >= n]
    target = min(upper) if upper else max(gear_dims)
    if target > n:
        pad = target - n
        rd = np.pad(rd, (0, pad), constant_values=0)
        rf = np.pad(rf, (0, pad), constant_values=0)
        rb = np.pad(rb, (0, pad), constant_values=0)
    elif target < n:
        rd, rf, rb = rd[:target], rf[:target], rb[:target]
    return rd, rf, rb, target

def process_det3d(ms_outputs, model, num_det_outputs, det_offset=0):
    """Decode 3D detection outputs: deserialize -> get_bboxes -> bbox3d2result."""
    output_tensors = [
        torch.from_numpy(ms_outputs[det_offset + j].get_data_to_numpy()).float()
        for j in range(num_det_outputs)
    ]
    preds_dicts = model.result_deserialize(output_tensors)
    img_metas = [{"box_type_3d": LiDARInstance3DBoxes}]
    try:
        bbox_list = model.pts_bbox_head.get_bboxes(
            preds_dicts, img_metas, rescale=True)
        bbox_results = [bbox3d2result(b, s, l) for b, s, l in bbox_list]
        res = bbox_results[0]
        boxes = res.get('boxes_3d')
        if boxes is not None and len(boxes) > 0:
            bt = getattr(boxes, 'tensor', None)
            if bt is not None and (torch.isnan(bt).any() or torch.isinf(bt).any()):
                res = _empty_bbox_result()
        return res
    except Exception:  # pylint: disable=broad-exception-caught
        return _empty_bbox_result()

def process_occ(ms_outputs, model, is_argmax_model, occ_index=0):
    """Get occupancy prediction from model output."""
    if is_argmax_model:
        occ_pred = ms_outputs[occ_index].get_data_to_numpy()
        if occ_pred.ndim == 4:
            occ_pred = occ_pred[0]
        return occ_pred.astype(np.uint8)
    occ_logits = torch.from_numpy(
        ms_outputs[occ_index].get_data_to_numpy()).float()
    occ_preds = model.occ_head.get_occ(occ_logits)
    return occ_preds[0]


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def _setup_tee(args):
    """Redirect stdout to both console and file when --output is given."""
    if not args.output:
        return None
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    tee = Tee(args.output)
    sys.stdout = tee
    return tee

def _restore_stdout(tee):
    """Restore original stdout and close the tee file."""
    if tee is None:
        return
    sys.stdout = tee.stdout
    tee.close()

def _print_header(args):
    """Print benchmark configuration header."""
    print('=' * 70)
    print('FlashOCC MindSpore Lite Full Benchmark')
    print('=' * 70)
    print(f'Date:          {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Model:         {args.model}')
    print(f'Config:        {args.config}')
    print(f'Checkpoint:    {args.checkpoint}')
    print(f'Shape mode:    {args.shape_mode}')
    print(f'Samples:       {args.samples if args.samples > 0 else "all"}')
    print(f'Postprocessing:{args.postprocessing}')
    print(f'Eval:          {args.eval}')
    print(f'Warmup:        {args.warmup}')
    print(f'Device ID:     {args.device_id}')
    print('=' * 70)

def _build_mslite_context(args):
    """Build MindSpore Lite Ascend context."""
    import mindspore_lite as mslite  # pylint: disable=import-outside-toplevel
    ctx = mslite.Context()
    ctx.target = ['ascend']
    ctx.ascend.device_id = args.device_id
    return ctx

def _build_mslite_model(args, ctx, s0_n, gear_dims):
    """Build MindSpore Lite model, auto-detecting or using explicit shape mode."""
    import mindspore_lite as mslite  # pylint: disable=import-outside-toplevel
    if args.shape_mode == 'auto':
        return detect_shape_mode(args.model, ctx, s0_n,
                                 args.fixed_ranks_len, gear_dims)
    ms_model = mslite.Model()
    ms_model.build_from_file(args.model, mslite.ModelType.MINDIR, ctx)
    return args.shape_mode, ms_model

# ---------------------------------------------------------------------------
# Model / sample helpers
# ---------------------------------------------------------------------------
def _count_det_outputs(wdet3d, has_pts_bbox_head, model):
    """Count 3D detection output tensors based on task heads."""
    if wdet3d and has_pts_bbox_head:
        return 6 * len(model.pts_bbox_head.task_heads)
    return 0

def _analyze_outputs(ms_model, wdet3d, num_det_outputs):
    """Determine output structure: model type, occ index, total outputs."""
    ms_outputs_info = ms_model.get_outputs()
    is_argmax_model = any('cls_occ_label' in o.name for o in ms_outputs_info)
    model_type = 'forward_with_argmax' if is_argmax_model else 'forward_ori'
    occ_index = num_det_outputs if wdet3d else 0
    return {
        'is_argmax_model': is_argmax_model,
        'model_type': model_type,
        'num_total_outputs': len(ms_outputs_info),
        'num_det_outputs': num_det_outputs,
        'occ_index': occ_index,
    }

def _get_sample0_ranks(model, data_loader):
    """Extract BEV pool ranks (depth/feat/bev) from the first sample."""
    data = next(iter(data_loader))
    inputs = [t.to('cpu') for t in data['img_inputs'][0]]
    metas = model.get_bev_pool_input(inputs)
    return {
        'ranks_depth': metas[1].int().cpu().numpy().astype(np.int32),
        'ranks_feat': metas[2].int().cpu().numpy().astype(np.int32),
        'ranks_bev': metas[0].int().cpu().numpy().astype(np.int32),
    }

def _resolve_target_len(mode, s0_n, args, gear_dims, s0_ranks):
    """Resolve target ranks length for sample 0 verification."""
    if mode == 'fixed':
        return args.fixed_ranks_len
    if mode == 'dynamic':
        return s0_n
    if mode == 'gear':
        _, _, _, target_len = pad_ranks_to_gear(
            s0_ranks['ranks_depth'], s0_ranks['ranks_feat'],
            s0_ranks['ranks_bev'], gear_dims)
        return target_len
    return s0_n

def _prepare_sample_ranks(mode, s0_ranks, args, model, inputs, gear_dims):
    """Prepare ranks for a single sample based on shape mode.

    Returns (rd, rf, rb, target_len, n_actual, padded).
    """
    if mode == 'fixed':
        return (s0_ranks['ranks_depth'], s0_ranks['ranks_feat'],
                s0_ranks['ranks_bev'], args.fixed_ranks_len, None, False)
    metas_ = model.get_bev_pool_input(inputs)
    rd = metas_[1].int().cpu().numpy().astype(np.int32)
    rf = metas_[2].int().cpu().numpy().astype(np.int32)
    rb = metas_[0].int().cpu().numpy().astype(np.int32)
    n_actual = len(rb)
    if mode == 'dynamic':
        return rd, rf, rb, n_actual, n_actual, False
    rd, rf, rb, target_len = pad_ranks_to_gear(rd, rf, rb, gear_dims)
    return rd, rf, rb, target_len, n_actual, target_len > n_actual

def _verify_sample0(mode, s0_n, args, gear_dims, s0_ranks, ms_model):
    """Verify that model resize works on sample 0."""
    target_len = _resolve_target_len(mode, s0_n, args, gear_dims, s0_ranks)
    ms_inputs = ms_model.get_inputs()
    ms_model.resize(ms_inputs,
                    [[1, 6, 3, 256, 704], [target_len], [target_len], [target_len]])
    print(f'Verified: resize to [{target_len}] OK')

def _print_model_info(mode, model_type, s0_n, ms_model, out_info, num_det):
    """Print model structure and output layout."""
    print(f'\nShape mode:          {mode}')
    print(f'Model type:            {model_type}')
    print(f'Sample 0 ranks:        {s0_n}')
    print(f'Total outputs:         {out_info["num_total_outputs"]}')
    print(f'Det outputs [0:{num_det}], Occ output [{out_info["occ_index"]}]')
    print('\nMindIR inputs:')
    for inp in ms_model.get_inputs():
        print(f'  {inp.name}:  shape={list(inp.shape)}, dtype={inp.dtype}')
    print('MindIR outputs:')
    for out in ms_model.get_outputs():
        print(f'  {out.name}:  shape={list(out.shape)}, dtype={out.dtype}')

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def _run_postprocessing(ms_outputs, model, wdet3d, wocc, has_pts_bbox_head,
                        num_det_outputs, is_argmax_model, occ_index):
    """Run postprocessing on model outputs, return result dict."""
    result = {}
    if wdet3d and has_pts_bbox_head:
        result['pts_bbox'] = process_det3d(
            ms_outputs, model, num_det_outputs, det_offset=0)
    if wocc:
        result['pred_occ'] = process_occ(
            ms_outputs, model, is_argmax_model, occ_index=occ_index)
    return result

def _to_eval_entry(result, wdet3d, wocc):
    """Convert result dict to the entry expected by dataset.evaluate."""
    if wdet3d and wocc:
        return result
    if wdet3d and not wocc:
        return result.get('pts_bbox', _empty_bbox_result())
    if wocc:
        return result.get('pred_occ')
    return None

def _run_inference_loop(args, data_loader, model, ms_model, s0_ranks, mode,
                        gear_dims, wdet3d, wocc, has_pts_bbox_head, out_info):
    """Run the main inference loop.

    Returns (per_sample_times, pure_inf_time, results,
             n_min, n_max, pad_count, total).
    """
    num_warmup = args.warmup
    pure_inf_time = 0.0
    results = []
    per_sample_times = []
    n_min, n_max = 10 ** 9, 0
    pad_count = 0
    total = 0

    for i, data in enumerate(data_loader):
        if args.samples > 0 and i >= args.samples:
            break
        total = i + 1

        inputs = [t.to('cpu') for t in data['img_inputs'][0]]
        img = inputs[0].numpy().astype(np.float32)
        if img.shape[1] > 6:
            img = img[:, :6]

        rd, rf, rb, target_len, n_actual, padded = _prepare_sample_ranks(
            mode, s0_ranks, args, model, inputs, gear_dims)
        if n_actual is not None:
            n_min, n_max = min(n_min, n_actual), max(n_max, n_actual)
        if padded:
            pad_count += 1

        ms_inputs = ms_model.get_inputs()
        ms_model.resize(ms_inputs,
                        [[1, 6, 3, 256, 704], [target_len], [target_len], [target_len]])
        feeds = {'img': img, 'ranks_depth': rd, 'ranks_feat': rf,
                 'ranks_bev': rb}
        for inp in ms_inputs:
            if inp.name in feeds:
                inp.set_data_from_numpy(feeds[inp.name])

        start_time = time.perf_counter()
        ms_outputs = ms_model.predict(ms_inputs)

        if args.postprocessing:
            result = _run_postprocessing(
                ms_outputs, model, wdet3d, wocc, has_pts_bbox_head,
                out_info['num_det_outputs'], out_info['is_argmax_model'],
                out_info['occ_index'])
            if args.eval:
                results.append(_to_eval_entry(result, wdet3d, wocc))

        elapsed = time.perf_counter() - start_time
        per_sample_times.append(elapsed)

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % 50 == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                _print_progress(i + 1, args.samples, fps)

        if args.samples > 0 and (i + 1) == args.samples:
            break

    return (per_sample_times, pure_inf_time, results,
            n_min, n_max, pad_count, total)

def _print_progress(done, samples, fps):
    """Print periodic progress during inference."""
    total_label = samples if samples > 0 else 'all'
    print(f'Done image [{done:>4} / {total_label}], '
          f'fps: {fps:.2f} img / s')

# ---------------------------------------------------------------------------
# Output / evaluation helpers
# ---------------------------------------------------------------------------
def _print_ranks_stats(mode, n_min, n_max, pad_count, total, args):
    """Print ranks statistics based on shape mode."""
    if mode == 'dynamic':
        print(f'Ranks range:        {n_min}~{n_max} (per-sample)')
    elif mode == 'gear':
        print(f'Ranks range:        {n_min}~{n_max}, '
              f'padded: {pad_count}/{total}')
    elif mode == 'fixed':
        print(f'Ranks:              fixed {args.fixed_ranks_len} '
              f'(reused sample 0)')

def _print_summary(mode, model_type, wdet3d, wocc, total, num_warmup,
                   pure_inf_time, per_sample_times, n_min, n_max, pad_count,
                   args):
    """Print benchmark summary statistics."""
    timed = max(total - num_warmup, 1)
    fps = timed / pure_inf_time if pure_inf_time > 0 else 0
    latency = 1000 / fps if fps > 0 else 0

    print(f'\n{"=" * 70}')
    print('Overall')
    print(f'{"=" * 70}')
    print(f'Shape mode:         {mode}')
    print(f'Model type:         {model_type}')
    print(f'wdet3d:             {wdet3d}')
    print(f'wocc:               {wocc}')
    print(f'Total samples:      {total}')
    print(f'Timed samples:      {timed} (warmup={num_warmup})')
    print(f'FPS:                {fps:.2f} img/s')
    print(f'Inference time:     {latency:.2f} ms')

    timed_latencies = (per_sample_times[num_warmup:]
                       if len(per_sample_times) > num_warmup
                       else per_sample_times)
    if timed_latencies:
        print(f'Latency (min):      {min(timed_latencies) * 1000:.2f} ms')
        print(f'Latency (max):      {max(timed_latencies) * 1000:.2f} ms')
        print(f'Latency (mean):     {np.mean(timed_latencies) * 1000:.2f} ms')
        print(f'Latency (std):      {np.std(timed_latencies) * 1000:.2f} ms')

    _print_ranks_stats(mode, n_min, n_max, pad_count, total, args)
    print(f'Postprocessing:     {args.postprocessing}')
    print(f'Eval:               {args.eval}')

def _build_eval_kwargs(cfg, wdet3d, wocc):
    """Build evaluation kwargs and label from task flags.

    Returns (eval_kwargs, eval_label) where eval_label is None if neither
    task is enabled.
    """
    eval_kwargs = cfg.get('evaluation', {}).copy()
    for key in ['interval', 'tmpdir', 'start', 'gpu_collect',
                'save_best', 'rule']:
        eval_kwargs.pop(key, None)
    if wdet3d and wocc:
        eval_kwargs.update({"metric": ['mAP', 'mIoU']})
        return eval_kwargs, 'mAP/NDS + mIoU'
    if wdet3d and not wocc:
        eval_kwargs.update({"metric": 'mAP'})
        return eval_kwargs, 'mAP/NDS'
    if wocc:
        eval_kwargs.update({"metric": ['mIoU']})
        return eval_kwargs, 'mIoU'
    return eval_kwargs, None

def _run_evaluation(dataset, results, cfg, wdet3d, wocc, tee, args):
    """Run dataset evaluation and print results."""
    print(f'\nCollected {len(results)} results for evaluation')
    eval_kwargs, eval_label = _build_eval_kwargs(cfg, wdet3d, wocc)
    if eval_label is None:
        print('WARNING: neither wdet3d nor wocc is enabled')
        _restore_stdout(tee)
        return
    print(f'\n{"=" * 70}')
    print(f'Evaluation ({eval_label})')
    print(f'{"=" * 70}')
    try:
        eval_result = dataset.evaluate(results, **eval_kwargs)
        print(eval_result)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f'WARNING: evaluation failed ({e})')

    if args.output:
        print(f'\n{"=" * 70}')
        print(f'Results saved to: {args.output}')
        print(f'{"=" * 70}')
        _restore_stdout(tee)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Entry point: run FlashOCC MindSpore Lite benchmark."""
    args = parse_args()
    if args.eval:
        args.postprocessing = True
        print('Note: --eval requires postprocessing, enabled automatically')

    tee = _setup_tee(args)
    gear_dims = [int(x) for x in args.gear_dims.split(',')]

    _print_header(args)

    cfg, model, dataset, data_loader = build_model_and_loader(args)
    wdet3d = cfg.model.get('wdet3d', False)
    wocc = cfg.model.get('wocc', True)
    has_pts_bbox_head = getattr(model, 'pts_bbox_head', None) is not None
    num_det = _count_det_outputs(wdet3d, has_pts_bbox_head, model)

    print(f'\nwdet3d:                {wdet3d}')
    print(f'wocc:                  {wocc}')
    print(f'has pts_bbox_head:     {has_pts_bbox_head}')
    print(f'det output count:      {num_det}')

    ctx = _build_mslite_context(args)
    s0_ranks = _get_sample0_ranks(model, data_loader)
    s0_n = len(s0_ranks['ranks_bev'])

    mode, ms_model = _build_mslite_model(args, ctx, s0_n, gear_dims)
    out_info = _analyze_outputs(ms_model, wdet3d, num_det)
    _print_model_info(mode, out_info['model_type'], s0_n, ms_model, out_info,
                      num_det)
    _verify_sample0(mode, s0_n, args, gear_dims, s0_ranks, ms_model)

    (per_sample_times, pure_inf_time, results,
     n_min, n_max, pad_count, total) = _run_inference_loop(
        args, data_loader, model, ms_model, s0_ranks, mode, gear_dims,
        wdet3d, wocc, has_pts_bbox_head, out_info)

    _print_summary(mode, out_info['model_type'], wdet3d, wocc, total,
                   args.warmup, pure_inf_time, per_sample_times,
                   n_min, n_max, pad_count, args)

    if not args.eval:
        if args.output:
            print(f'\nResults saved to: {args.output}')
            _restore_stdout(tee)
        return

    _run_evaluation(dataset, results, cfg, wdet3d, wocc, tee, args)

if __name__ == '__main__':
    main()
