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
"""TCP ONNX inference (random-input perf + output shape)."""

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


def _random_inputs(img_h, img_w):
    """fixed-seed random (front_img, speed, command)."""
    rng = np.random.RandomState(SEED)
    front = np.ascontiguousarray(rng.rand(1, 3, img_h, img_w).astype(np.float32))
    speed = np.ascontiguousarray(rng.rand(1, 1).astype(np.float32))
    command = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    return front, speed, command


class TCPOnnx:
    """TCP ONNXRuntime inference."""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime not installed.")
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def forward(self, front_img, speed, command):
        """run forward."""
        feed = {self.input_names[0]: front_img,
                self.input_names[1]: speed,
                self.input_names[2]: command}
        return self.sess.run(self.output_names, feed)


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description="TCP ONNX inference")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--img-h", type=int, default=256)
    parser.add_argument("--img-w", type=int, default=512)
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

    net = TCPOnnx(args.model, args.device)
    front, speed, command = _random_inputs(args.img_h, args.img_w)
    print(f"Using random input: front_img={front.shape}, speed={speed.shape}, "
          f"command={command.shape}, seed={SEED}")

    rss0, hwm0 = _mem_kb("VmRSS"), _mem_kb("VmHWM")
    for _ in range(args.warmup):
        net.forward(front, speed, command)
    lat = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        net.forward(front, speed, command)
        lat.append((time.perf_counter() - t0) * 1000.0)
    rss1, hwm1 = _mem_kb("VmRSS"), _mem_kb("VmHWM")

    outs = net.forward(front, speed, command)
    for name, out in zip(["control", "trajectory"], outs):
        print(f"  {name}: {tuple(out.shape)}")

    lat_np = np.array(lat, dtype=np.float32)
    print(f"\nlatency_ms_mean: {float(lat_np.mean()):.3f}, "
          f"p99: {float(np.percentile(lat_np, 99)):.3f}")
    print(f"VmRSS: {rss1} KB (before {rss0} KB), VmHWM: {hwm1} KB (before {hwm0} KB)")


if __name__ == "__main__":
    main()
