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
"""VAD MindSpore Lite inference (numpy-only, no torch)."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mindspore_lite as mslite
except ImportError:
    mslite = None


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


class VADMsLite:
    """VAD MindSpore Lite inference."""

    def __init__(self, model_path, device="cpu", device_id=0):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def forward(self, images):
        """run forward."""
        inputs = [mslite.Tensor(np.ascontiguousarray(images))]
        outputs = self.model.predict(inputs)
        return [o.get_data_to_numpy() for o in outputs]


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="VAD MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--img-h", type=int, default=320)
    parser.add_argument("--img-w", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main():
    """main entry."""
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    if mslite is None:
        print("Error: mindspore_lite not installed.")
        sys.exit(1)

    net = VADMsLite(args.model, args.device, args.device_id)
    images = _random_input(args.ncams, args.img_h, args.img_w)
    print(f"Using random input, shape={images.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    for _ in range(args.warmup):
        net.forward(images)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        net.forward(images)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")

    outs = net.forward(images)
    for name, out in zip(["ego_traj", "scores"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}, "
          f"p99: {float(np.percentile(lat_np, 99)):.3f}")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
