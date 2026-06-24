#!/usr/bin/env python3
"""Cosmos-Predict1 ONNXRuntime inference: cond -> video latent via DDPM."""

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


def ddpm_sample(step_fn, cond, num_tokens, latent_dim, num_steps, rng):
    betas = np.linspace(1e-4, 2e-2, num_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas).astype(np.float32)
    x = rng.standard_normal((1, num_tokens, latent_dim)).astype(np.float32)
    for i in reversed(range(num_steps)):
        t = np.array([i], dtype=np.int64)
        eps = step_fn(x, t, cond)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        mean = (1.0 / np.sqrt(alpha)) * (x - (betas[i] / np.sqrt(1.0 - alpha_bar)) * eps)
        x = mean + (np.sqrt(betas[i]) * rng.standard_normal(x.shape).astype(np.float32) if i > 0 else 0.0)
    return x.astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="Cosmos-Predict1 ONNXRuntime inference (DDPM latent)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
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
        return sess.run(None, {"noisy_latent": x, "timestep": t, "cond": c})[0].astype(np.float32)

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        ddpm_sample(step, cond, args.num_tokens, args.latent_dim, args.num_steps, rng)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        latent = ddpm_sample(step, cond, args.num_tokens, args.latent_dim, args.num_steps, rng)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    print("Output:")
    print(f"  latent shape={latent.shape} dtype={latent.dtype} (video latent; VAE decode omitted)")
    print(f"  latent_abs_max={float(np.abs(latent).max()):.6f}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  ddpm_steps: {args.num_steps}  num_tokens: {args.num_tokens}  latent_dim: {args.latent_dim}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  e2e_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  per_step_ms_mean: {float(lat_np.mean())/max(args.num_steps,1):.3f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
