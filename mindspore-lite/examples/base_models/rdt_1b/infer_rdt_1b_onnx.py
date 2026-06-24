#!/usr/bin/env python3
"""RDT-1B ONNXRuntime inference: cond -> bimanual action chunk via DDPM."""

import argparse
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


def ddpm_sample(step_fn, cond, horizon, action_dim, num_steps, rng):
    betas = np.linspace(1e-4, 2e-2, num_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas).astype(np.float32)
    x = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    for i in reversed(range(num_steps)):
        t = np.array([i], dtype=np.int64)
        eps = step_fn(x, t, cond)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        mean = (1.0 / np.sqrt(alpha)) * (x - (betas[i] / np.sqrt(1.0 - alpha_bar)) * eps)
        if i > 0:
            x = mean + np.sqrt(betas[i]) * rng.standard_normal(x.shape).astype(np.float32)
        else:
            x = mean
    return x.astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="RDT-1B ONNXRuntime inference (DDPM bimanual action chunk)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--action-dim", type=int, default=14)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--cond-dim", type=int, default=256)
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
    cond = rng.standard_normal((1, args.cond_dim)).astype(np.float32)

    def step(x, t, c):
        return sess.run(None, {"noisy_action": x, "timestep": t, "cond": c})[0].astype(np.float32)

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        ddpm_sample(step, cond, args.horizon, args.action_dim, args.num_steps, rng)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        action = ddpm_sample(step, cond, args.horizon, args.action_dim, args.num_steps, rng)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    print("Output:")
    print(f"  action shape={action.shape} dtype={action.dtype} (bimanual chunk)")
    print(f"  action[0,0]={np.array2string(action[0, 0], precision=4)}")
    print(f"  action_abs_max={float(np.abs(action).max()):.6f}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  ddpm_steps: {args.num_steps}  horizon: {args.horizon}  action_dim: {args.action_dim}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  e2e_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  per_step_ms_mean: {float(lat_np.mean())/max(args.num_steps,1):.3f}")
    print(f"  e2e_ms_p50: {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
