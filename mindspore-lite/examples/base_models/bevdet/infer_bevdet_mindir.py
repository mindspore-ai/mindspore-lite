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
"""BEVDet All-in-One MindSpore Lite inference.

Pipeline (mirrors BEVDet/tools/analysis_tools/benchmark_trt.py):
  1. Build BEVDet TRT model (for ranks computation + postprocessing head)
  2. Load real NuScenes sample via pre-processed npz + image preprocessing, or random inputs
  3. Compute ranks via model.get_bev_pool_input
  4. MindSpore Lite inference with 4 inputs: img + 3 ranks
  5. (Optional) Postprocessing: result_deserialize -> get_bboxes -> bbox3d2result
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from mmcv import Config
from mmcv.runner import load_checkpoint

try:
    from mmdet.utils import compat_cfg
    from mmdet3d.models import build_model
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes

try:
    import mindspore_lite as mslite
except ImportError:
    print("Error: mindspore_lite package not found.")
    print("Please install: pip install mindspore-lite")
    sys.exit(1)


DETECTION_CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
]

SEED = 1024

# Number of BEV-pool points for random/smoke-test mode. Matches the N_Points
# seen on the real NuScenes sample 0.
RANDOM_RANKS_N = 179832

BEVDET_CAM_ORDER = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]

# Mirrors bevdet-r50.py data_config — used for test-mode resize/crop
DATA_CONFIG = {
    'cams': BEVDET_CAM_ORDER,
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# ImageNet normalization (matches BEVDet's normalize_img)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================
# BEVDet model build (for ranks computation)
# ============================================================================

def build_bevdet_model(config_path: str, checkpoint_path: str,
                       device: str = 'cpu') -> torch.nn.Module:
    """Build BEVDet TRT model. Self-contained — used to call
    get_bev_pool_input for ranks computation.
    """
    cfg = Config.fromfile(config_path)
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


# ============================================================================
# Camera params extraction (pkl format auto-detection)
# ============================================================================

def quat_to_rotation(quat) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3x3 rotation matrix. Numpy only."""
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]
    ], dtype=np.float32)


def assemble_transform_from_quat(quat, tran) -> np.ndarray:
    """Assemble 4x4 homogeneous transform from quaternion + 3-vec translation."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_to_rotation(quat)
    T[:3, 3] = np.asarray(tran, dtype=np.float32)
    return T


def get_sensor_transforms(cam_info: dict):
    """Extract (sensor2ego, ego2global) 4x4 matrices from cam_info.

    Auto-detects pkl format:
      - Quaternion: keys 'sensor2ego_rotation' + 'sensor2ego_translation'
                    (+ ego2global_* equivalents)
      - Matrix:     keys 'sensor2ego' + 'ego2global' (4x4 each)
    """
    if 'sensor2ego_rotation' in cam_info:
        sensor2ego = assemble_transform_from_quat(
            cam_info['sensor2ego_rotation'],
            cam_info['sensor2ego_translation'])
        ego2global = assemble_transform_from_quat(
            cam_info['ego2global_rotation'],
            cam_info['ego2global_translation'])
    elif 'sensor2ego' in cam_info:
        sensor2ego = np.asarray(cam_info['sensor2ego'], dtype=np.float32)
        ego2global = np.asarray(cam_info['ego2global'], dtype=np.float32)
        if sensor2ego.shape != (4, 4):
            raise ValueError(
                f"sensor2ego matrix shape {sensor2ego.shape}, expected (4,4)")
    else:
        raise KeyError(
            "Cannot find camera transforms. Available cam_info keys: "
            f"{list(cam_info.keys())}. Expected either 'sensor2ego_rotation' "
            "(quaternion format) or 'sensor2ego' (matrix format).")
    return sensor2ego, ego2global


# ============================================================================
# Image preprocessing (matches BEVDet PrepareImageInputs test mode)
# ============================================================================

def sample_augmentation_test(H: int, W: int, data_config: dict):
    """Test-mode resize/crop params (no random augmentation).

    Mirrors PrepareImageInputs.sample_augmentation(is_train=False).
    Returns (resize, resize_dims, crop) where crop = (x0, y0, x1, y1).
    """
    fH, fW = data_config['input_size']
    resize = float(fW) / float(W) + data_config.get('resize_test', 0.0)
    resize_dims = (int(W * resize), int(H * resize))
    newW, newH = resize_dims
    crop_h = int((1 - np.mean(data_config['crop_h'])) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize, resize_dims, crop


def preprocess_image_with_aug(img_pil: Image.Image, intrin: np.ndarray,
                              data_config: dict):
    """Apply BEVDet test-mode preprocessing to one image.

    Mirrors PrepareImageInputs.img_transform (test path, flip=False, rotate=0)
    + normalize_img.

    Returns:
        img_norm:  [3, fH, fW] float32 (ImageNet-normalized)
        post_rot:  [3, 3] float32 — encodes cropped→original coord mapping
        post_tran: [3]    float32
        intrin:    [3, 3] float32 — unchanged (original image coords)
    """
    W, H = img_pil.size  # PIL: (width, height)
    resize, resize_dims, crop = sample_augmentation_test(H, W, data_config)

    # Transform image: resize → crop → normalize
    img = img_pil.resize(resize_dims, Image.BILINEAR)
    img = img.crop(crop)
    img = np.array(img, dtype=np.float32) / 255.0
    img = (img - IMG_MEAN) / IMG_STD
    img_norm = np.ascontiguousarray(img.transpose(2, 0, 1))  # [3, H, W]

    # post_rot / post_tran encode cropped→original image-coord mapping.
    # For test mode (flip=False, rotate=0):
    #   post_rot  = diag(resize, resize, 1)
    #   post_tran = (-crop_w, -crop_h, 0)
    post_rot = np.eye(3, dtype=np.float32)
    post_tran = np.zeros(3, dtype=np.float32)
    post_rot[0, 0] *= resize
    post_rot[1, 1] *= resize
    post_tran[0] -= crop[0]
    post_tran[1] -= crop[1]

    return img_norm, post_rot, post_tran, intrin


def load_sample_images_and_params(info: dict, data_root: str) -> dict:
    """Load 6 camera images + camera params from a NuScenes sample.

    Applies full BEVDet test-mode preprocessing. Returns dict of stacked
    arrays with batch dim:
        imgs:        [1, N, 3, H, W]
        sensor2egos: [1, N, 4, 4]
        ego2globals: [1, N, 4, 4]
        intrins:     [1, N, 3, 3]
        post_rots:   [1, N, 3, 3]
        post_trans:  [1, N, 3]
        bda:         [1, 4, 4]  (identity — no BEV aug at inference)
    """
    imgs, sensor2egos, ego2globals = [], [], []
    intrins, post_rots, post_trans = [], [], []

    for cam_name in BEVDET_CAM_ORDER:
        cam_info = info['cams'][cam_name]

        # Resolve image path with three-way fallback:
        #   - absolute path: use as-is
        #   - relative path that exists from cwd: use as-is (some pkls store
        #     the path already prefixed with data_root)
        #   - otherwise: prepend data_root (pkl stores bare filename)
        cam_path = cam_info['data_path']
        if Path(cam_path).is_absolute() or Path(cam_path).exists():
            full_path = cam_path
        elif data_root:
            full_path = str(Path(data_root) / cam_path)
        else:
            full_path = cam_path
        img_pil = Image.open(full_path).convert('RGB')

        # Intrinsics (original image coords)
        intrin = np.asarray(cam_info['cam_intrinsic'], dtype=np.float32)

        # Preprocess image + compute post_rot/post_trans
        img_norm, post_rot, post_tran, intrin = preprocess_image_with_aug(
            img_pil, intrin, DATA_CONFIG)

        # Sensor transforms (auto-detect format)
        sensor2ego, ego2global = get_sensor_transforms(cam_info)

        imgs.append(img_norm)
        intrins.append(intrin)
        sensor2egos.append(sensor2ego)
        ego2globals.append(ego2global)
        post_rots.append(post_rot)
        post_trans.append(post_tran)

    def _stack(arrs):
        return np.ascontiguousarray(np.stack(arrs)[None])

    return {
        'imgs':        _stack(imgs),
        'sensor2egos': _stack(sensor2egos),
        'ego2globals': _stack(ego2globals),
        'intrins':     _stack(intrins),
        'post_rots':   _stack(post_rots),
        'post_trans':  _stack(post_trans),
        'bda':         np.ascontiguousarray(np.eye(4, dtype=np.float32)[None]),
    }


# ============================================================================
# Random inputs (smoke test, no real data)
# ============================================================================

def generate_random_inputs(batch: int, num_cams: int, channels: int,
                           height: int, width: int, seed: int = SEED) -> dict:
    """Random imgs + dummy identity camera params (for smoke testing).

    imgs go through ImageNet normalization to match the real-data path.
    Seed is fixed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    imgs = rng.rand(batch, num_cams, channels, height, width).astype(np.float32)
    # Apply ImageNet norm so distribution roughly matches real preprocessed images
    imgs = (imgs - IMG_MEAN.reshape((1, 1, -1, 1, 1))) / IMG_STD.reshape((1, 1, -1, 1, 1))

    return {
        'imgs':        np.ascontiguousarray(imgs),
        'sensor2egos': np.ascontiguousarray(
            np.tile(np.eye(4, dtype=np.float32), (batch, num_cams, 1, 1))),
        'ego2globals': np.ascontiguousarray(
            np.tile(np.eye(4, dtype=np.float32), (batch, num_cams, 1, 1))),
        'intrins':     np.ascontiguousarray(
            np.tile(np.eye(3, dtype=np.float32), (batch, num_cams, 1, 1))),
        'post_rots':   np.ascontiguousarray(
            np.tile(np.eye(3, dtype=np.float32), (batch, num_cams, 1, 1))),
        'post_trans':  np.zeros((batch, num_cams, 3), dtype=np.float32),
        'bda':         np.ascontiguousarray(
            np.tile(np.eye(4, dtype=np.float32), (batch, 1, 1))),
    }


def generate_random_ranks(n_points: int = RANDOM_RANKS_N,
                          seed: int = SEED):
    """Random ranks (int32) for smoke-test mode — bypasses compute_ranks
    which would return None under identity intrinsics.

    Returns (ranks_depth, ranks_feat, ranks_bev), each shape [n_points],
    dtype int32 (matches ONNX/MindIR input dtype).
    """
    rng = np.random.RandomState(seed)
    return (
        rng.randint(0, 2**31 - 1, size=n_points, dtype=np.int32),
        rng.randint(0, 2**31 - 1, size=n_points, dtype=np.int32),
        rng.randint(0, 2**31 - 1, size=n_points, dtype=np.int32),
    )


# ============================================================================
# Ranks computation
# ============================================================================

def compute_ranks(model: torch.nn.Module, inputs: dict):
    """Compute (ranks_depth, ranks_feat, ranks_bev) via model.get_bev_pool_input.

    Reuses BEVDet's prepare_inputs + get_lidar_coor + voxel_pooling_prepare_v2.
    Returns numpy int32 arrays of shape [N_Points].

    Note: int32 (not int64) to match the ONNX export's example dtype —
    `model.get_bev_pool_input` returns `.int()` tensors, which the ONNX graph
    records as int32 inputs. Feeding int64 at inference causes
    "Unexpected input data type" errors in ONNX Runtime / MindSpore Lite.
    """
    bev_pool_inputs = [
        torch.from_numpy(inputs['imgs']),
        torch.from_numpy(inputs['sensor2egos']),
        torch.from_numpy(inputs['ego2globals']),
        torch.from_numpy(inputs['intrins']),
        torch.from_numpy(inputs['post_rots']),
        torch.from_numpy(inputs['post_trans']),
        torch.from_numpy(inputs['bda']),
    ]
    with torch.no_grad():
        metas = model.get_bev_pool_input(bev_pool_inputs)
    if metas[0] is None:
        raise RuntimeError(
            "ranks computation returned None; camera params may be invalid "
            "(e.g., identity intrinsics produce no points inside BEV grid)")
    ranks_bev, ranks_depth, ranks_feat, _, _ = metas
    return (
        ranks_depth.cpu().numpy().astype(np.int32),
        ranks_feat.cpu().numpy().astype(np.int32),
        ranks_bev.cpu().numpy().astype(np.int32),
    )


# ============================================================================
# NuScenes npz loader
# ============================================================================

def load_nuscenes_sample_from_npz(npz_path: str) -> dict:
    """Load a pre-processed NuScenes sample from npz (safe, no pickle).

    The npz must have been created by the data pre-processing step in README.
    It contains pre-built 4x4 (sensor2ego, ego2global) matrices so that
    downstream get_sensor_transforms can use them directly (matrix path),
    avoiding redundant quaternion→matrix conversion and precision loss.
    """
    data = np.load(npz_path)
    cams = {}
    for i in range(len(data['cam_names'])):
        name = str(data['cam_names'][i])
        cams[name] = {
            'data_path': str(data['data_paths'][i]),
            'cam_intrinsic': data['cam_intrinsics'][i],
            'sensor2ego': data['sensor2egos'][i],
            'ego2global': data['ego2globals'][i],
        }
    return {'cams': cams}


# ============================================================================
# Convenience: prepare inputs based on CLI args
# ============================================================================

def prepare_inputs(args):
    """Load real NuScenes data or generate seeded random inputs.

    Returns (inputs_dict, mode_str) where mode_str is 'NuScenes' or 'random'.
    """
    if args.sample_npz and Path(args.sample_npz).exists():
        info = load_nuscenes_sample_from_npz(args.sample_npz)
        inputs = load_sample_images_and_params(info, args.data_root)
        return inputs, "NuScenes"
    if args.sample_npz:
        print(f"  Warning: sample npz not found ({args.sample_npz}), using random input")
    inputs = generate_random_inputs(args.batch, args.num_cams, 3,
                                    args.imH, args.imW, seed=args.seed)
    return inputs, "random"


def add_data_args(parser: argparse.ArgumentParser):
    """Add standard data-related CLI args to a parser."""
    parser.add_argument("--config", default="BEVDet/configs/bevdet/bevdet-r50.py",
                        help="BEVDet config (for ranks computation model build)")
    parser.add_argument("--checkpoint", default="bevdet-dev2.1/bevdet-r50.pth",
                        help="BEVDet checkpoint (for ranks computation model build)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-cams", type=int, default=6)
    parser.add_argument("--imH", type=int, default=256)
    parser.add_argument("--imW", type=int, default=704)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for smoke-test inputs")
    parser.add_argument("--sample-npz", type=str, default=None,
                        help="Pre-processed NuScenes sample npz "
                             "(convert pkl via the data pre-processing step in README)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="NuScenes data root (e.g., data/nuscenes/)")
    return parser


# ============================================================================
# MindSpore Lite wrapper
# ============================================================================

class BEVDetAllInOneMsLite:
    """MindSpore Lite inference wrapper (4 inputs: img + 3 ranks)."""

    def __init__(self, model_path: str, device: str = "ascend", device_id: int = 0):
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)
        self.device = device

    def infer(self, imgs: np.ndarray, ranks_depth: np.ndarray,
              ranks_feat: np.ndarray, ranks_bev: np.ndarray):
        """Run inference. Returns list of raw output numpy arrays in MindIR
        order: [reg_0, height_0, dim_0, rot_0, vel_0, heatmap_0, ...] for
        each task head."""
        if imgs.dtype != np.float32:
            imgs = imgs.astype(np.float32)
        inputs_mslite = [
            mslite.Tensor(np.ascontiguousarray(imgs)),
            mslite.Tensor(np.ascontiguousarray(ranks_depth)),
            mslite.Tensor(np.ascontiguousarray(ranks_feat)),
            mslite.Tensor(np.ascontiguousarray(ranks_bev)),
        ]
        outputs = self.model.predict(inputs_mslite)
        return [o.get_data_to_numpy() for o in outputs]


# ============================================================================
# Postprocessing (mirrors BEVDet/tools/analysis_tools/benchmark_trt.py)
# ============================================================================

def postprocess(model: torch.nn.Module, outputs):
    """Decode raw head outputs into bbox results, mirroring
    BEVDet/tools/analysis_tools/benchmark_trt.py.

    Args:
        model:    BEVDet TRT model (provides pts_bbox_head + result_deserialize)
        outputs:  list of numpy arrays in MindIR output order
                  [reg_0, height_0, dim_0, rot_0, vel_0, heatmap_0, ...]
    Returns:
        list of bbox3d2result dicts: [{boxes_3d, scores_3d, labels_3d}]
    """
    n_task = len(model.pts_bbox_head.task_heads)
    expected = 6 * n_task
    if len(outputs) != expected:
        raise ValueError(
            f"Expected {expected} outputs (6 x {n_task} task heads), "
            f"got {len(outputs)}")

    ordered = [torch.from_numpy(arr) for arr in outputs]
    pred = model.result_deserialize(ordered)
    img_metas = [{"box_type_3d": LiDARInstance3DBoxes}]
    bbox_list = model.pts_bbox_head.get_bboxes(pred, img_metas, rescale=True)
    return [bbox3d2result(b, s, l) for b, s, l in bbox_list]


# ============================================================================
# Args + main
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to MindIR model")
    parser.add_argument("--device", default="ascend", choices=["ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--postproc", action="store_true",
                        help="Run benchmark_trt.py-style postprocessing "
                             "(result_deserialize -> get_bboxes -> bbox3d2result)")
    add_data_args(parser)
    return parser.parse_args()


def print_outputs(outputs, n_task: int):
    """print outputs."""
    keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
    print("--- Raw Output Shapes ---")
    for t in range(n_task):
        for k in keys:
            arr = outputs[t * 6 + keys.index(k)]
            print(f"  task{t}_{k}: {tuple(arr.shape)}")


def print_bbox_results(bbox_results):
    """Print top-K detection results per task head."""
    print("--- Postprocessing Results ---")
    for i, r in enumerate(bbox_results):
        n = len(r['labels_3d'])
        print(f"  Task {i}: {n} boxes")
        if n > 0:
            top = min(5, n)
            scores = np.asarray(r['scores_3d'])
            top_idx = np.argsort(-scores)[:top]
            for j in top_idx:
                cls = DETECTION_CLASSES[int(r['labels_3d'][j])] \
                    if int(r['labels_3d'][j]) < len(DETECTION_CLASSES) \
                    else f"class_{int(r['labels_3d'][j])}"
                print(f"    [{cls}] score={scores[j]:.3f}")


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    print("=== BEVDet MindIR Inference ===")
    print(f"  Model:      {args.model}")
    print(f"  Config:     {args.config}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    if args.sample_npz:
        print(f"  Data mode:  NuScenes (sample_npz={args.sample_npz})")
    else:
        print(f"  Data mode:  random (seed={args.seed})")
    print()

    # ---- 1. Build BEVDet model ----
    print("[1/4] Building BEVDet model (for ranks + postproc) ...")
    bevdet_model = build_bevdet_model(args.config, args.checkpoint, 'cpu')
    n_task = len(bevdet_model.pts_bbox_head.task_heads)
    print(f"  task_heads: {n_task}")

    # ---- 2. Prepare inputs (real NuScenes or random) ----
    print("[2/4] Preparing inputs ...")
    inputs, mode = prepare_inputs(args)
    print(f"  Mode:       {mode}")
    print(f"  imgs shape: {tuple(inputs['imgs'].shape)}")
    if mode == "NuScenes":
        print(f"  Sample npz: {args.sample_npz}")

    # ---- 3. Compute ranks ----
    if mode == "NuScenes":
        print("[3/4] Computing ranks from camera params ...")
        ranks_depth, ranks_feat, ranks_bev = compute_ranks(bevdet_model, inputs)
    else:
        print(f"[3/4] Using random ranks (N_Points={RANDOM_RANKS_N}) ...")
        ranks_depth, ranks_feat, ranks_bev = generate_random_ranks()
    print(f"  ranks N_Points: {ranks_bev.shape[0]}")

    # ---- 4. MindSpore Lite inference + (optional) postproc ----
    print(f"[4/4] MindSpore Lite inference (warmup={args.warmup}, runs={args.runs}) ...")
    model = BEVDetAllInOneMsLite(args.model, args.device, args.device_id)
    imgs_np = inputs['imgs'].astype(np.float32)

    for _ in range(args.warmup):
        model.infer(imgs_np, ranks_depth, ranks_feat, ranks_bev)

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        model.infer(imgs_np, ranks_depth, ranks_feat, ranks_bev)
        times.append((time.perf_counter() - start) * 1000)

    print(f"  Mean latency: {np.mean(times):.2f} ms")
    print(f"  Min latency:  {np.min(times):.2f} ms")
    print(f"  Max latency:  {np.max(times):.2f} ms")
    print()

    outputs = model.infer(imgs_np, ranks_depth, ranks_feat, ranks_bev)
    print_outputs(outputs, n_task)

    if args.postproc:
        print()
        bbox_results = postprocess(bevdet_model, outputs)
        print_bbox_results(bbox_results)


if __name__ == "__main__":
    main()
