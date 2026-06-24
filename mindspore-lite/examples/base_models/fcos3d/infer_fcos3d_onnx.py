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
"""FCOS3D ONNX inference (random-input perf + output shape)."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


SEED = 1024


def _read_proc_status_kb(key: str) -> Optional[int]:
    """read a field (KB) from /proc/self/status."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _memory_snapshot():
    """snapshot process memory."""
    return {"vmrss_kb": _read_proc_status_kb("VmRSS"),
            "vmhwm_kb": _read_proc_status_kb("VmHWM")}


def _random_input(img_h, img_w):
    """fixed-seed random image input."""
    rng = np.random.RandomState(SEED)
    return np.ascontiguousarray(rng.rand(1, 3, img_h, img_w).astype(np.float32))


class FCOS3DOnnx:
    """FCOS3D ONNXRuntime inference."""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime not installed.")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def forward(self, images):
        """run forward."""
        return self.sess.run(self.output_names, {self.input_names[0]: images})


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="FCOS3D ONNX inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--img-h", type=int, default=320)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main():
    """main entry."""
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    inferencer = FCOS3DOnnx(args.model, args.device)
    images = _random_input(args.img_h, args.img_w)
    print(f"Using random input, shape={images.shape}, seed={SEED}")

    mem0 = _memory_snapshot()
    for _ in range(args.warmup):
        inferencer.forward(images)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        inferencer.forward(images)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem1 = _memory_snapshot()

    outs = inferencer.forward(images)
    print("Output shapes:")
    for name, out in zip(["cls_score", "bbox_pred", "centerness", "dir_cls"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print("\nPerformance:")
    print(f"  warmup: {args.warmup}, runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  latency_ms_p99:  {float(np.percentile(lat_np, 99)):.3f}")
    print("\nMemory:")
    print(f"  VmRSS: {mem1['vmrss_kb']} KB (before: {mem0['vmrss_kb']} KB)")
    print(f"  VmHWM: {mem1['vmhwm_kb']} KB (before: {mem0['vmhwm_kb']} KB)")


if __name__ == "__main__":
    main()
