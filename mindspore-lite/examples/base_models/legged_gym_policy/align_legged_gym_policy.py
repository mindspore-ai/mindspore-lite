#!/usr/bin/env python3
"""Accuracy alignment for the Legged Gym locomotion policy.

Compares the action produced by three paths on identical observation input:

  1. PyTorch ``PolicyMLP`` (fp32)              -- baseline (random-init demo)
  2. ONNX Runtime on the exported actor ONNX   -- export check
  3. MindSpore Lite on the converted MindIR    -- deploy check

Reports cosine similarity, max abs diff and mean abs diff of the action vector.
Input defaults to a seeded random observation; pass ``--obs`` / ``--obs-npy``
for a real observation. For a trained policy, re-export with ``--checkpoint``
first so all three paths share the same weights.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _cosine(a, b):
    """Cosine similarity between two flattened arrays."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _stats(name, a, b):
    """Print cosine / max_abs / mean_abs between a and b, return cosine."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    cos = _cosine(a, b)
    print(f"  {name:18s} shape={str(a.shape):18s} cos={cos:.6f} "
          f"max_abs={float(diff.max()):.6e} mean_abs={float(diff.mean()):.6e}")
    return cos


def _torch_baseline(args, obs):
    """Run the PyTorch PolicyMLP with the same (seeded) weights as the export."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import torch  # local import: baseline only
    from export_legged_gym_policy_onnx import PolicyMLP  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hidden = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    model = PolicyMLP(args.obs_dim, args.action_dim, hidden, args.activation,
                      output_tanh=not args.no_output_tanh).eval()
    with torch.no_grad():
        action = model(torch.from_numpy(obs)).cpu().numpy().astype(np.float32)
    return action


def _onnx_infer(onnx_path, obs):
    """Run ONNX Runtime inference."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"observation": obs.astype(np.float32)})[0].astype(np.float32)


def _mslite_infer(mindir_path, obs, device, device_id):
    """Run MindSpore Lite inference using repo helpers from the sibling infer script."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from infer_legged_gym_policy_mslite import _build_model, _run_model  # noqa: E402

    model, inputs = _build_model(mindir_path, device=device, device_id=device_id)
    outs = _run_model(model, inputs, {"observation": obs.astype(np.float32)})
    return outs[0].astype(np.float32)


def _load_obs(args) -> np.ndarray:
    """Return a float32 observation of shape [1, obs_dim]."""
    if args.obs_npy and os.path.exists(args.obs_npy):
        return np.load(args.obs_npy).astype(np.float32).reshape(1, -1)
    if args.obs:
        return np.asarray([float(x) for x in args.obs.split(",")], dtype=np.float32)[None, :]
    rng = np.random.default_rng(args.seed)
    return rng.standard_normal((1, args.obs_dim)).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Legged Gym policy alignment (PyTorch vs ONNX vs MSLite)")
    p.add_argument("--onnx", type=str, default="./legged_gym_policy_onnx/legged_gym_policy.onnx")
    p.add_argument("--mindir", type=str, default="./legged_gym_policy_onnx/legged_gym_policy_graph.mindir")
    p.add_argument("--obs", type=str, default="")
    p.add_argument("--obs-npy", type=str, default="")
    p.add_argument("--obs-dim", type=int, default=235)
    p.add_argument("--action-dim", type=int, default=18)
    p.add_argument("--hidden-dims", type=str, default="512,256,128")
    p.add_argument("--activation", type=str, default="elu")
    p.add_argument("--no-output-tanh", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-mslite", action="store_true")
    args = p.parse_args()

    obs = _load_obs(args)
    print("=" * 78)
    print("Legged Gym policy alignment: PyTorch vs ONNX vs MindSpore Lite")
    print("=" * 78)
    print(f"input: observation {obs.shape} {obs.dtype}"
          + (f"  (obs={args.obs})" if args.obs
             else (f"  (obs_npy={args.obs_npy})" if args.obs_npy else "  (seeded random)")))

    print("\n[PyTorch] PolicyMLP on CPU (fp32) ...")
    pt_action = _torch_baseline(args, obs)

    onx_action = None
    if not args.skip_onnx and os.path.exists(args.onnx):
        print(f"\n[ONNX] onnxruntime on {args.onnx} ...")
        onx_action = _onnx_infer(args.onnx, obs)
        print("  PyTorch vs ONNX:")
        _stats("action", pt_action, onx_action)
    elif not args.skip_onnx:
        print(f"\n[ONNX] skipped: {args.onnx} not found.")

    if not args.skip_mslite and os.path.exists(args.mindir):
        print(f"\n[MSLite] mindspore_lite on {args.mindir} (device={args.device}) ...")
        msl_action = _mslite_infer(args.mindir, obs, args.device, args.device_id)
        print("  PyTorch vs MSLite:")
        _stats("action", pt_action, msl_action)
        if onx_action is not None:
            print("  ONNX vs MSLite:")
            _stats("action", onx_action, msl_action)
    elif not args.skip_mslite:
        print(f"\n[MSLite] skipped: {args.mindir} not found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
