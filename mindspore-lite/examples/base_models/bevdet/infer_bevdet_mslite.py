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
"""MindIR inference on Ascend with BEVDet mmcv dataloader (BGR, cv2 pipeline).

Run from BEVDet/ so config data_root ('data/nuscenes/') resolves:
    cd BEVDet
    python scripts/infer_bevdet_mslite.py \
        --model output/bevdet_r50_ascend.mindir \
        --config config/bevdet/bevdet-r50.py \
        --checkpoint bevdet-r50.pth \
        --output results/mslite_sample0_result.txt
"""
import argparse
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.datasets import replace_ImageToTensor
try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

OUTPUT_NAMES = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
DETECTION_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]


def parse_args():
    """Parse command-line arguments for MindIR inference."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", help='model file', default='output/bevdet_r50_ascend.mindir')
    p.add_argument("--config", help='config file', default='config/bevdet/bevdet-r50.py')
    p.add_argument("--checkpoint", help='checkpoint file', default='bevdet-r50.pth')
    p.add_argument("--output", help='output file', default='results/mslite_sample0_result.txt')
    p.add_argument("--device", default="ascend")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--sample-idx", type=int, default=0)
    return p.parse_args()


def build_model_and_loader(args):
    """Build mmdet3d model and test data loader from args."""
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.img_backbone.with_cp = False
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    test_dataloader_default_args = {
        'samples_per_gpu': 1, 'workers_per_gpu': 2, 'dist': False, 'shuffle': False}
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to("cpu").eval()
    return model, data_loader


def get_sample_inputs(model, data_loader, sample_idx):
    """Extract img and ranks tensors for a specific sample index."""
    for i, data in enumerate(data_loader):
        if i != sample_idx:
            continue
        inputs = [t.to("cpu") for t in data['img_inputs'][0]]
        metas = model.get_bev_pool_input(inputs)
        img = inputs[0]
        ranks_bev, ranks_depth, ranks_feat = metas[0], metas[1], metas[2]
        return img, ranks_depth, ranks_feat, ranks_bev
    raise ValueError(f"sample_idx {sample_idx} not found")


def decode_detections(model, results):
    """Decode raw model outputs into bounding boxes, scores, and labels."""
    from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
    head = model.pts_bbox_head
    preds_dicts = [[{
        k: torch.from_numpy(np.ascontiguousarray(results[k])).float()
        for k in OUTPUT_NAMES
    }]]
    img_metas = [{'box_type_3d': LiDARInstance3DBoxes}]
    bboxes, scores, labels = head.get_bboxes(preds_dicts, img_metas)[0]
    boxes = bboxes.tensor.detach().cpu().numpy()
    return boxes, scores.detach().cpu().numpy(), labels.detach().cpu().numpy()


def main():
    """Run single-sample MindIR inference on Ascend NPU."""
    args = parse_args()
    print("=== BEVDet MindIR Inference (mmcv dataloader, BGR, Ascend) ===")
    print(f"  Model:      {args.model}")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")

    model, data_loader = build_model_and_loader(args)
    img, rd, rf, rb = get_sample_inputs(model, data_loader, args.sample_idx)
    num_points = rb.shape[0]
    print(f"  img shape:  {tuple(img.shape)}")
    print(f"  ranks N_Points: {num_points}")

    import mindspore_lite as mslite
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = args.device_id

    ms_model = mslite.Model()
    ms_model.build_from_file(args.model, mslite.ModelType.MINDIR, context)

    print("\n  MindIR inputs:")
    for inp in ms_model.get_inputs():
        print(f"    {inp.name}: shape={list(inp.shape)}, dtype={inp.dtype}")
    print("  MindIR outputs:")
    for out in ms_model.get_outputs():
        print(f"    {out.name}: shape={list(out.shape)}, dtype={out.dtype}")

    feeds_np = {
        "img":         img.cpu().numpy().astype(np.float32),
        "ranks_depth": rd.cpu().numpy().astype(np.int32),
        "ranks_feat":  rf.cpu().numpy().astype(np.int32),
        "ranks_bev":   rb.cpu().numpy().astype(np.int32),
    }

    ms_inputs = ms_model.get_inputs()
    model_ranks_len = ms_inputs[1].shape[0] if len(ms_inputs[1].shape) > 0 else -1
    is_dynamic = model_ranks_len == -1

    if is_dynamic:
        ms_model.resize(ms_inputs, [
            [1, 6, 3, 256, 704], [num_points], [num_points], [num_points]])
        ms_inputs = ms_model.get_inputs()
    elif model_ranks_len != num_points:
        print(f"ERROR: Fixed model expects ranks_len={model_ranks_len}, "
              f"but sample {args.sample_idx} has {num_points}. "
              f"Use the dynamic MindIR (--model output/bevdet_r50_dynamic.mindir) "
              f"for samples with different ranks.")
        sys.exit(1)

    for inp in ms_inputs:
        if inp.name in feeds_np:
            inp.set_data_from_numpy(feeds_np[inp.name])
        else:
            print(f"ERROR: MindIR input '{inp.name}' not found in feeds")
            sys.exit(1)

    ms_outputs = ms_model.predict(ms_inputs)
    results = {}
    for i, out in enumerate(ms_outputs):
        name = OUTPUT_NAMES[i] if i < len(OUTPUT_NAMES) else f"out_{i}"
        results[name] = out.get_data_to_numpy()

    print("\n  Output shapes:")
    for k, v in results.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    boxes, scores, labels = decode_detections(model, results)
    n = len(scores)
    order = scores.argsort()[::-1]

    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"=== MindIR Ascend decoded detections "
                f"— NuScenes sample {args.sample_idx} ===\n")
        f.write(f"model: {args.model}\n")
        f.write(f"Total: {n} objects (score>=0.1 + rotate-NMS "
                f"iou=0.2, max_per_img=500)\n")
        if n > 0:
            f.write(f"score range: {scores[order[-1]]:.3f} ~ "
                    f"{scores[order[0]]:.3f}\n")
            cls_counts = {}
            for i in order:
                c = DETECTION_CLASSES[int(labels[i])]
                cls_counts[c] = cls_counts.get(c, 0) + 1
            dist = ", ".join(f"{k}={v}" for k, v in sorted(
                cls_counts.items(), key=lambda x: -x[1]))
            f.write(f"class distribution: {dist}\n\n")
        hdr = (f"{'#':>3} {'class':<20} {'score':>6}  "
               f"{'x':>8} {'y':>8} {'z':>8}  "
               f"{'l':>6} {'w':>6} {'h':>6}  "
               f"{'yaw':>7} {'vx':>6} {'vy':>6}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")
        for rank, i in enumerate(order):
            c = DETECTION_CLASSES[int(labels[i])]
            x, y, z, l, w, h, yaw, vx, vy = boxes[i]
            f.write(f"{rank + 1:>3} {c:<20} {scores[i]:>6.3f}  "
                    f"{x:>8.2f} {y:>8.2f} {z:>8.2f}  "
                    f"{l:>6.2f} {w:>6.2f} {h:>6.2f}  "
                    f"{yaw:>7.3f} {vx:>6.2f} {vy:>6.2f}\n")

    print(f"\nWrote {n} detections to {args.output}")
    if n > 0:
        print(f"  score range: {scores[order[-1]]:.3f} ~ {scores[order[0]]:.3f}")
        i0 = order[0]
        c0 = DETECTION_CLASSES[int(labels[i0])]
        x, y, z = boxes[i0][:3]
        print(f"  top-1: {c0} {scores[i0]:.3f} @ ({x:.2f}, {y:.2f}, {z:.2f})")


if __name__ == "__main__":
    main()
