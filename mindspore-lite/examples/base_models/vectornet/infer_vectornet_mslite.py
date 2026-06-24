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
"""VectorNet MindSpore Lite inference (numpy-only, no torch)."""

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


def _random_inputs(obs_len, agent_feat, map_num, map_feat):
    """fixed-seed random (agent_hist, map_polyline)."""
    rng = np.random.RandomState(SEED)
    agent = np.ascontiguousarray(rng.rand(1, obs_len, agent_feat).astype(np.float32))
    mp = np.ascontiguousarray(rng.rand(1, map_num, map_feat).astype(np.float32))
    return agent, mp


class VectorNetMsLite:
    """VectorNet MindSpore Lite inference."""

    def __init__(self, model_path, device="cpu", device_id=0):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def forward(self, agent_hist, map_polyline):
        """run forward."""
        inputs = [mslite.Tensor(np.ascontiguousarray(agent_hist)),
                  mslite.Tensor(np.ascontiguousarray(map_polyline))]
        outputs = self.model.predict(inputs)
        return [o.get_data_to_numpy() for o in outputs]


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="VectorNet MindSpore Lite inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--agent-feat", type=int, default=4)
    parser.add_argument("--map-poly-num", type=int, default=100)
    parser.add_argument("--map-feat", type=int, default=9)
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

    net = VectorNetMsLite(args.model, args.device, args.device_id)
    agent, mp = _random_inputs(args.obs_len, args.agent_feat, args.map_poly_num, args.map_feat)
    print(f"Using random input: agent_hist={agent.shape}, map_polyline={mp.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    for _ in range(args.warmup):
        net.forward(agent, mp)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        net.forward(agent, mp)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")

    outs = net.forward(agent, mp)
    for name, out in zip(["trajectory"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}, "
          f"p99: {float(np.percentile(lat_np, 99)):.3f}")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
