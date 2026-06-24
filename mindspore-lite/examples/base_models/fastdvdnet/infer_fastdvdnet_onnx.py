#!/usr/bin/env python3
"""FastDVDnet ONNXRuntime 推理脚本(精度基准 ground truth)。

demo: 读单张含噪帧, 复制成 5 帧序列, 配 noise_sigma, 输出去噪帧。
"""

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


def _load_frame(path, height, width):
    img = Image.open(path).convert("RGB").resize((int(width), int(height)), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3]


def _build_seq(frame_hw3, num_frames):
    """Replicate a single frame into a [1,num_frames,3,H,W] sequence."""
    fr = np.transpose(frame_hw3, (2, 0, 1))[None, ...]  # [1,3,H,W]
    return np.repeat(fr, num_frames, axis=0)[None, ...]  # [1,N,3,H,W]


def _pick_providers(device):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


class FastDvdNetOnnxInferencer:
    """FastDVDnet ONNXRuntime 推理封装。"""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装")
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=_pick_providers(device))

    def forward(self, seq, noise_sigma):
        return self.sess.run(None, {"seq": seq, "noise_sigma": noise_sigma})[0]


def _parse_args():
    p = argparse.ArgumentParser(description="FastDVDnet ONNXRuntime 推理")
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="含噪帧图像")
    p.add_argument("--output", type=str, default="./fastdvdnet_denoised_onnx.png")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--noise-sigma", type=float, default=5.0)
    p.add_argument("--num-frames", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    return p.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    infer = FastDvdNetOnnxInferencer(args.onnx, device=args.device)
    seq = _build_seq(_load_frame(args.input, args.height, args.width), args.num_frames)
    sigma = np.array([[args.noise_sigma]], dtype=np.float32)

    for _ in range(int(args.warmup)):
        _ = infer.forward(seq, sigma)
    lat, mem0 = [], _memory_snapshot()
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = infer.forward(seq, sigma)
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
