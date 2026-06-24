#!/usr/bin/env python3
"""Legged Gym locomotion policy ONNXRuntime inference.

Loads the exported actor ONNX (``observation -> action``) and runs a forward
pass. The observation may come from:

  * a ``.npy`` file (``--obs-npy``),       shape ``[obs_dim]`` or ``[B, obs_dim]``
  * a comma-separated vector (``--obs``),  e.g. ``0.1,0.2,...``
  * a seeded random vector (default),       used for pipeline smoke-test / timing

Reports the action vector and end-to-end latency (warmup + N runs) plus the
process RSS memory snapshot (VmRSS / VmHWM) for host-memory accounting.
"""

import argparse
import os
import time
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None


def _read_proc_status_kb(key):
    """Read a memory field from /proc/self/status (KB)."""
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
    """Return VmRSS / VmHWM in GB for host-memory accounting."""
    rss = _read_proc_status_kb("VmRSS")
    hwm = _read_proc_status_kb("VmHWM")
    return {"vmrss_gb": round(rss / 1048576, 3) if rss else None,
            "vmhwm_gb": round(hwm / 1048576, 3) if hwm else None}


def _load_observation(args) -> np.ndarray:
    """Return a float32 observation array of shape [1, obs_dim]."""
    if args.obs_npy and os.path.exists(args.obs_npy):
        arr = np.load(args.obs_npy).astype(np.float32).reshape(-1)
        return arr[None, :]
    if args.obs:
        arr = np.asarray([float(x) for x in args.obs.split(",")], dtype=np.float32)
        return arr[None, :]
    rng = np.random.default_rng(args.seed)
    return rng.standard_normal((1, args.obs_dim)).astype(np.float32)


def _parse_args():
    p = argparse.ArgumentParser(description="Legged Gym policy ONNXRuntime inference")
    p.add_argument("--model", type=str, required=True, help="ONNX path (legged_gym_policy.onnx)")
    p.add_argument("--obs-npy", type=str, default="", help="Observation .npy file ([obs_dim] or [B,obs_dim]).")
    p.add_argument("--obs", type=str, default="", help="Comma-separated observation vector.")
    p.add_argument("--obs-dim", type=int, default=235)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=50)
    return p.parse_args()


def main():
    args = _parse_args()
    if ort is None:
        raise RuntimeError("onnxruntime not installed: pip install onnxruntime")

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.device == "cuda" else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(args.model, providers=providers)

    obs = _load_observation(args)
    if obs.shape[1] != args.obs_dim:
        raise RuntimeError(f"observation dim {obs.shape[1]} != --obs-dim {args.obs_dim}")

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        sess.run(None, {"observation": obs})
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        outs = sess.run(None, {"observation": obs})
        lat.append((time.perf_counter() - t0) * 1000.0)
    mem_after = _memory_snapshot()

    action = outs[0]
    print("Output:")
    print(f"  action shape={action.shape} dtype={action.dtype}")
    print(f"  action[0][:6]={np.array2string(action[0][:6], precision=4, max_line_width=120)}")
    print(f"  action_abs_max={float(np.abs(action).max()):.6f}")

    lat_np = np.asarray(lat, dtype=np.float32)
    print("Perf:")
    print(f"  batch_size: {obs.shape[0]}  obs_dim: {obs.shape[1]}")
    print(f"  warmup: {args.warmup}  runs: {args.runs}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.4f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.4f}")
    print(f"  latency_ms_p90:  {float(np.percentile(lat_np, 90)):.4f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
