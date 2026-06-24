#!/usr/bin/env python3
"""Octo policy ONNXRuntime inference: image + proprio -> action chunk via DDPM.

The exported ONNX is a single denoising step. This script runs the full DDPM
sampling loop in numpy and calls the ONNX once per step to predict the noise.
All inputs are numpy; image defaults to a seeded random tensor (pipeline
smoke-test), or a real image preprocessed with simple resize+normalize.
"""

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


def _ddpm_betas(num_steps, beta_start=1e-4, beta_end=2e-2):
    """Linear beta schedule for DDPM."""
    return np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)


def ddpm_sample(step_fn, image, proprio, horizon, action_dim, num_steps, rng):
    """Run DDPM sampling by repeatedly calling step_fn(image, proprio, t, x) -> eps."""
    betas = _ddpm_betas(num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas).astype(np.float32)
    x = rng.standard_normal((1, horizon, action_dim)).astype(np.float32)
    for i in reversed(range(num_steps)):
        t = np.array([i], dtype=np.int64)
        eps = step_fn(image, proprio, t, x)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i]
        mean = (1.0 / np.sqrt(alpha)) * (x - (betas[i] / np.sqrt(1.0 - alpha_bar)) * eps)
        if i > 0:
            z = rng.standard_normal(x.shape).astype(np.float32)
            x = mean + np.sqrt(betas[i]) * z
        else:
            x = mean
    return x.astype(np.float32)


def _load_image(path, img_size):
    """Load + resize + normalize an image to [1,3,H,W] float32; None -> seeded random handled by caller."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.transpose(arr, (2, 0, 1))[None, :].astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="Octo policy ONNXRuntime inference (DDPM action chunk)")
    p.add_argument("--model", type=str, required=True, help="ONNX path (octo_denoise.onnx)")
    p.add_argument("--image", type=str, default="", help="Optional image path; default seeded random.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--proprio-dim", type=int, default=7)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--num-steps", type=int, default=10, help="DDPM denoising steps.")
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
    if args.image and os.path.exists(args.image):
        image = _load_image(args.image, args.img_size)
    else:
        image = rng.standard_normal((1, 3, args.img_size, args.img_size)).astype(np.float32)
    proprio = rng.standard_normal((1, args.proprio_dim)).astype(np.float32)

    def step(img, prop, t, x):
        outs = sess.run(None, {"image": img, "proprio": prop, "timestep": t, "noisy_action": x})
        return outs[0].astype(np.float32)

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        ddpm_sample(step, image, proprio, args.horizon, args.action_dim, args.num_steps, rng)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        action = ddpm_sample(step, image, proprio, args.horizon, args.action_dim, args.num_steps, rng)
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    print("Output:")
    print(f"  action shape={action.shape} dtype={action.dtype}")
    print(f"  action[0,0,:6]={np.array2string(action[0, 0, :6], precision=4)}")
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
