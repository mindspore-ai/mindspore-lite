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
"""MTR MindSpore Lite inference (numpy-only, no torch)."""

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


def _random_inputs(num_objects, obs_len, num_polylines, poly_feat):
    """fixed-seed random (actor_history, map_polylines)."""
    rng = np.random.RandomState(SEED)
    actor = np.ascontiguousarray(rng.rand(1, num_objects, obs_len, 9).astype(np.float32))
    mp = np.ascontiguousarray(rng.rand(1, num_polylines, poly_feat).astype(np.float32))
    return actor, mp


class MTRMsLite:
    """MTR MindSpore Lite inference."""

    def __init__(self, model_path, device="cpu", device_id=0):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def forward(self, actor_history, map_polylines):
        """run forward."""
        inputs = [mslite.Tensor(np.ascontiguousarray(actor_history)),
                  mslite.Tensor(np.ascontiguousarray(map_polylines))]
        outputs = self.model.predict(inputs)
        return [o.get_data_to_numpy() for o in outputs]


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="MTR MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-objects", type=int, default=64)
    parser.add_argument("--obs-len", type=int, default=11)
    parser.add_argument("--num-polylines", type=int, default=768)
    parser.add_argument("--poly-feat", type=int, default=9)
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

    net = MTRMsLite(args.model, args.device, args.device_id)
    actor, mp = _random_inputs(args.num_objects, args.obs_len, args.num_polylines, args.poly_feat)
    print(f"Using random input: actor_history={actor.shape}, map_polylines={mp.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    for _ in range(args.warmup):
        net.forward(actor, mp)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        net.forward(actor, mp)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")

    outs = net.forward(actor, mp)
    for name, out in zip(["trajectory", "scores"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}, "
          f"p99: {float(np.percentile(lat_np, 99)):.3f}")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
