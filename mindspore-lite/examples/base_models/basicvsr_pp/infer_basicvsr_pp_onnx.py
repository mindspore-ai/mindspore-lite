#!/usr/bin/env python3
"""BasicVSR ONNXRuntime 推理脚本(精度基准): N 帧低清 -> N 帧 4x 超分(取中心帧保存)。"""

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


def _load_lr(path, h, w):
    img = Image.open(path).convert("RGB").resize((int(w), int(h)), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _build_seq(frame_hw3, num_frames):
    fr = np.transpose(frame_hw3, (2, 0, 1))[None, ...]
    return np.repeat(fr, num_frames, axis=0)[None, ...]


def _pick_providers(device):
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]


class BasicVsrPPOnnxInferencer:
    """BasicVSR ONNXRuntime 推理封装。"""

    def __init__(self, model_path, device="cpu"):
        if ort is None:
            raise RuntimeError("onnxruntime 未安装")
        self.sess = ort.InferenceSession(model_path, sess_options=ort.SessionOptions(),
                                         providers=_pick_providers(device))

    def forward(self, lr_seq):
        return self.sess.run(None, {"lr_seq": lr_seq})[0]


def _parse_args():
    p = argparse.ArgumentParser(description="BasicVSR ONNXRuntime 推理")
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="低清帧图像")
    p.add_argument("--output", type=str, default="./basicvsr_pp_sr_onnx.png")
    p.add_argument("--lr-height", type=int, default=64)
    p.add_argument("--lr-width", type=int, default=64)
    p.add_argument("--num-frames", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    return p.parse_args()


def main():
    args = _parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    infer = BasicVsrPPOnnxInferencer(args.onnx, device=args.device)
    seq = _build_seq(_load_lr(args.input, args.lr_height, args.lr_width), args.num_frames)

    for _ in range(int(args.warmup)):
        _ = infer.forward(seq)
    lat, mem0 = [], _memory_snapshot()
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = infer.forward(seq)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem1 = _memory_snapshot()

    center = out[0, out.shape[1] // 2]  # [3,4H,4W]
    center = np.clip(center, 0.0, 1.0)
    Image.fromarray((np.transpose(center, (1, 2, 0)) * 255.0 + 0.5).astype(np.uint8)).save(args.output)
    lat_np = np.array(lat, dtype=np.float32)
    print(f"[onnx] saved center SR -> {args.output}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f} p50: {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  proc_rss_mb: {mem1['rss_mb']} (hwm={mem1['hwm_mb']})")


if __name__ == "__main__":
    main()
