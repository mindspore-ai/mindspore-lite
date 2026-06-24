#!/usr/bin/env python3
"""pi0 ONNXRuntime inference: image -> action chunk via Flow Matching (Euler ODE)."""

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


def euler_sample(step_fn, image, horizon, action_dim, num_steps, rng):
    x = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    dt = 1.0 / float(num_steps)
    for i in range(num_steps):
        t = np.array([i * dt], dtype=np.float32)
        v = step_fn(image, x, t)
        x = x + v * dt
    return x.astype(np.float32)


def _load_image(path, img_size):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))[None, :].astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="pi0 ONNXRuntime inference (Flow Matching)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--image", type=str, default="")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
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

    def step(img, x, t):
        return sess.run(None, {"image": img, "x_t": x, "t": t})[0].astype(np.float32)

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        euler_sample(step, image, args.horizon, args.action_dim, args.num_steps, rng)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        action = euler_sample(step, image, args.horizon, args.action_dim, args.num_steps, rng)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    print("Output:")
    print(f"  action shape={action.shape} dtype={action.dtype}")
    print(f"  action[0,0]={np.array2string(action[0, 0], precision=4)}")
    print(f"  action_abs_max={float(np.abs(action).max()):.6f}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  flow_steps: {args.num_steps}  horizon: {args.horizon}  action_dim: {args.action_dim}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  e2e_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  per_step_ms_mean: {float(lat_np.mean())/max(args.num_steps,1):.3f}")
    print(f"  e2e_ms_p50: {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
