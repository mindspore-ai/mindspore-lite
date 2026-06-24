#!/usr/bin/env python3
"""BMBC ONNXRuntime 推理脚本(精度基准 ground truth): 两帧 -> 中点中间帧。"""

import argparse
import os
import time

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None


def _read_proc_status_mb(key):
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
    except Exception:
        return None
    return None


def _memory_snapshot():
    return {"rss_mb": _read_proc_status_mb("VmRSS"), "hwm_mb": _read_proc_status_mb("VmHWM")}


def _load_image(path, height, width):
    img = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def _pick_providers(device):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


class BmbcOnnxInferencer:
    """BMBC ONNXRuntime 推理封装。"""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装")
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=_pick_providers(device))

    def forward(self, img0, img1):
        return self.sess.run(None, {"img0": img0, "img1": img1})[0]


def _parse_args():
    p = argparse.ArgumentParser(description="BMBC ONNXRuntime 推理")
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--img0", type=str, required=True)
    p.add_argument("--img1", type=str, required=True)
    p.add_argument("--output", type=str, default="./bmbc_mid_onnx.png")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    return p.parse_args()


def main():
    args = _parse_args()
    for key in ("img0", "img1"):
        if not os.path.exists(getattr(args, key)):
            raise FileNotFoundError(getattr(args, key))

    infer = BmbcOnnxInferencer(args.onnx, device=args.device)
    x0 = _load_image(args.img0, args.height, args.width)
    x1 = _load_image(args.img1, args.height, args.width)

    for _ in range(int(args.warmup)):
        _ = infer.forward(x0, x1)
    lat, mem0 = [], _memory_snapshot()
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = infer.forward(x0, x1)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem1 = _memory_snapshot()

    out = np.clip(out[0], 0.0, 1.0)
    Image.fromarray((np.transpose(out, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)).save(args.output)
    lat_np = np.array(lat, dtype=np.float32)
    print(f"[onnx] saved -> {args.output}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f} p50: {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  proc_rss_mb: {mem1['rss_mb']} (hwm={mem1['hwm_mb']})")


if __name__ == "__main__":
    main()
