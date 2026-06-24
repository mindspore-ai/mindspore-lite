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
"""BEVFormer ONNX inference (random multi-view perf + output shape)."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


SEED = 1024


def _mem_kb(key):
    """read VmRSS/VmHWM (KB) from /proc/self/status."""
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


def _random_input(ncams, h, w):
    """fixed-seed random multi-view images."""
    rng = np.random.RandomState(SEED)
    return np.ascontiguousarray(rng.rand(1, ncams, 3, h, w).astype(np.float32))


class BEVFormerOnnx:
    """BEVFormer ONNXRuntime inference."""

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
    parser = argparse.ArgumentParser(description="BEVFormer ONNX inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ncams", type=int, default=6)
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

    net = BEVFormerOnnx(args.model, args.device)
    images = _random_input(args.ncams, args.img_h, args.img_w)
    print(f"Using random input, shape={images.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    try:
        for _ in range(args.warmup):
            net.forward(images)
        lat = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            net.forward(images)
            lat.append((time.perf_counter() - t0) * 1000.0)
        outs = net.forward(images)
        for name, out in zip(["cls_scores", "bbox_preds"], outs):
            print(f"  {name}: {tuple(out.shape)}")
        lat_np = np.array(lat, dtype=np.float32)
        print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}")
    except Exception as e:  # noqa
        print(f"(Custom deformable op cannot run on ORT: {e})")
        print("Use the MindIR path via infer_bevformer_mslite.py instead.")
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
