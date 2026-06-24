#!/usr/bin/env python3
"""CAIN ONNXRuntime 推理脚本: 输入两帧, 输出中间帧(精度基准 ground truth)。"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None


def _read_proc_status_kb(key):
    """Read VmRSS/VmHWM(kB) from /proc/self/status."""
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
    return {"vmrss_mb": (_read_proc_status_kb("VmRSS") or 0) // 1024,
            "vmhwm_mb": (_read_proc_status_kb("VmHWM") or 0) // 1024}


def _load_image(path, height, width):
    """Load image, resize to (width,height), return float32 [1,3,H,W] in [0,1]."""
    img = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # [1,3,H,W]
    return arr


def _pick_providers(device):
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class CainOnnxInferencer:
    """CAIN ONNXRuntime 推理封装。"""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装: pip install onnxruntime")
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=_pick_providers(device))

    def forward(self, img0, img1):
        out = self.sess.run(None, {"img0": img0, "img1": img1})
        return out[0]  # [1,3,H,W]


def _parse_args():
    p = argparse.ArgumentParser(description="CAIN ONNXRuntime 推理")
    p.add_argument("--onnx", type=str, required=True, help="cain.onnx 路径")
    p.add_argument("--img0", type=str, required=True, help="起始帧图像")
    p.add_argument("--img1", type=str, required=True, help="结束帧图像")
    p.add_argument("--output", type=str, default="./cain_mid_onnx.png")
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

    infer = CainOnnxInferencer(args.onnx, device=args.device)
    x0 = _load_image(args.img0, args.height, args.width)
    x1 = _load_image(args.img1, args.height, args.width)

    for _ in range(int(args.warmup)):
        _ = infer.forward(x0, x1)
    lat = []
    mem_before = _memory_snapshot()
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        mid = infer.forward(x0, x1)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    mid = np.clip(mid[0], 0.0, 1.0)  # [3,H,W]
    mid = np.transpose(mid, (1, 2, 0))  # [H,W,3]
    Image.fromarray((mid * 255.0 + 0.5).astype(np.uint8)).save(args.output)

    lat_np = np.array(lat, dtype=np.float32)
    print(f"[onnx] saved mid frame -> {args.output}")
    print("Perf:")
    print(f"  input: {args.height}x{args.width}, warmup={args.warmup} runs={args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  latency_ms_p90:  {float(np.percentile(lat_np, 90)):.3f}")
    print(f"  proc_rss_mb: {mem_after['vmrss_mb']} (hwm={mem_after['vmhwm_mb']})")


if __name__ == "__main__":
    main()
