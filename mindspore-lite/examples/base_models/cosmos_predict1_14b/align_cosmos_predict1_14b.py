#!/usr/bin/env python3
"""Accuracy alignment for the Cosmos-Predict1 latent DiT denoiser (single step)."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _stats(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = np.abs(a - b)
    cos = _cosine(a, b)
    print(f"  {name:14s} shape={str(a.shape):24s} cos={cos:.6f} "
          f"max_abs={float(diff.max()):.6e} mean_abs={float(diff.mean()):.6e}")
    return cos


def _torch_baseline(args, noisy_latent, timestep, cond):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import torch  # local import: baseline only
    from export_cosmos_predict1_14b_onnx import CosmosDenoise  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = CosmosDenoise(args.num_tokens, args.latent_dim, args.cond_dim,
                          args.dim, args.depth, args.heads).eval()
    with torch.no_grad():
        noise = model(torch.from_numpy(noisy_latent), torch.from_numpy(timestep), torch.from_numpy(cond))
    return noise.cpu().numpy().astype(np.float32)


def _onnx_infer(onnx_path, noisy_latent, timestep, cond):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"noisy_latent": noisy_latent, "timestep": timestep, "cond": cond})[0].astype(np.float32)


def _mslite_infer(mindir_path, noisy_latent, timestep, cond, device, device_id):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from infer_cosmos_predict1_14b_mslite import _build_model, _run_model  # noqa: E402
    model, inputs = _build_model(mindir_path, device=device, device_id=device_id)
    return _run_model(model, inputs, {"noisy_latent": noisy_latent, "timestep": timestep, "cond": cond})[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Cosmos-Predict1 alignment (PyTorch vs ONNX vs MSLite)")
    p.add_argument("--onnx", type=str, default="./cosmos_predict1_14b_onnx/cosmos_denoise.onnx")
    p.add_argument("--mindir", type=str, default="./cosmos_predict1_14b_onnx/cosmos_denoise_graph.mindir")
    p.add_argument("--num-tokens", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--cond-dim", type=int, default=256)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-mslite", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    noisy_latent = rng.standard_normal((1, args.num_tokens, args.latent_dim)).astype(np.float32)
    timestep = np.array([args.timestep], dtype=np.int64)
    cond = rng.standard_normal((1, args.cond_dim)).astype(np.float32)

    print("=" * 78)
    print("Cosmos-Predict1 alignment: PyTorch vs ONNX vs MindSpore Lite (latent DiT)")
    print("=" * 78)
    print(f"input: noisy_latent {noisy_latent.shape}, timestep {timestep}, cond {cond.shape}")

    print("\n[PyTorch] CosmosDenoise on CPU (fp32) ...")
    pt_noise = _torch_baseline(args, noisy_latent, timestep, cond)

    onx_noise = None
    if not args.skip_onnx and os.path.exists(args.onnx):
        print(f"\n[ONNX] onnxruntime on {args.onnx} ...")
        onx_noise = _onnx_infer(args.onnx, noisy_latent, timestep, cond)
        print("  PyTorch vs ONNX:")
        _stats("noise", pt_noise, onx_noise)
    elif not args.skip_onnx:
        print(f"\n[ONNX] skipped: {args.onnx} not found.")

    if not args.skip_mslite and os.path.exists(args.mindir):
        print(f"\n[MSLite] mindspore_lite on {args.mindir} (device={args.device}) ...")
        msl_noise = _mslite_infer(args.mindir, noisy_latent, timestep, cond, args.device, args.device_id)
        print("  PyTorch vs MSLite:")
        _stats("noise", pt_noise, msl_noise)
        if onx_noise is not None:
            print("  ONNX vs MSLite:")
            _stats("noise", onx_noise, msl_noise)
    elif not args.skip_mslite:
        print(f"\n[MSLite] skipped: {args.mindir} not found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
