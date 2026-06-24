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
"""LaneGCN ONNX inference (random-input perf + output shape)."""

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


def _random_inputs(obs_len, actor_feat, lane_num, lane_feat):
    """fixed-seed random (actor_hist, lane_nodes)."""
    rng = np.random.RandomState(SEED)
    actor = np.ascontiguousarray(rng.rand(1, obs_len, actor_feat).astype(np.float32))
    lanes = np.ascontiguousarray(rng.rand(1, lane_num, lane_feat).astype(np.float32))
    return actor, lanes


class LaneGCNOnnx:
    """LaneGCN ONNXRuntime inference."""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime not installed.")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def forward(self, actor_hist, lane_nodes):
        """run forward."""
        feed = {self.input_names[0]: actor_hist, self.input_names[1]: lane_nodes}
        return self.sess.run(self.output_names, feed)


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="LaneGCN ONNX inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--obs-len", type=int, default=20)
    parser.add_argument("--agent-feat", type=int, default=6)
    parser.add_argument("--lane-num", type=int, default=500)
    parser.add_argument("--lane-feat", type=int, default=4)
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

    net = LaneGCNOnnx(args.model, args.device)
    actor, lanes = _random_inputs(args.obs_len, args.agent_feat, args.lane_num, args.lane_feat)
    print(f"Using random input: actor_hist={actor.shape}, lane_nodes={lanes.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    for _ in range(args.warmup):
        net.forward(actor, lanes)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        net.forward(actor, lanes)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")

    outs = net.forward(actor, lanes)
    for name, out in zip(["trajectory"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}, "
          f"p99: {float(np.percentile(lat_np, 99)):.3f}")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
