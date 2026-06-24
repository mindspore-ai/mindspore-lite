#!/usr/bin/env python3
"""LLaVA-1.5 skeleton ONNXRuntime inference: image + input_ids -> logits (single forward)."""

import argparse
import os
import time

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


def _read_proc_status_kb(key):
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
    rss = _read_proc_status_kb("VmRSS")
    hwm = _read_proc_status_kb("VmHWM")
    return {"vmrss_gb": round(rss / 1048576, 3) if rss else None,
            "vmhwm_gb": round(hwm / 1048576, 3) if hwm else None}


def _load_image(path, img_size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))[None, :].astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="LLaVA-1.5 skeleton ONNXRuntime inference")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=10)
    return p.parse_args()


def main():
    args = _parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime not installed: pip install onnxruntime")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(args.model, providers=providers)

    rng = np.random.default_rng(args.seed)
    image = (_load_image(args.image, args.img_size) if args.image and os.path.exists(args.image)
             else rng.standard_normal((1, 3, args.img_size, args.img_size)).astype(np.float32))
    input_ids = rng.integers(0, args.vocab_size, (1, args.seq_len)).astype(np.int64)

    feed = {"image": image, "input_ids": input_ids}
    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        sess.run(None, feed)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        outs = sess.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    logits = outs[0]
    next_tok = int(np.argmax(logits[0, -1]))
    print("Output:")
    print(f"  logits shape={logits.shape} dtype={logits.dtype}")
    print(f"  next_token(argmax of last pos)={next_tok}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  img: {image.shape}  seq: {args.seq_len}  device: {args.device}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.4f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.4f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
