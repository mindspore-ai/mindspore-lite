#!/usr/bin/env python3
"""Accuracy alignment for the Octo single-step denoiser.

Compares the predicted noise of one denoising step across three paths on
identical (seeded) input:

  1. PyTorch ``OctoDenoiseNet`` (fp32)           -- baseline
  2. ONNX Runtime on the exported ``octo_denoise.onnx``
  3. MindSpore Lite on the converted MindIR

Reports cosine / max_abs / mean_abs of the predicted noise tensor. End-to-end
DDPM action-chunk alignment can be added once real weights are loaded (task-2).
"""

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
    print(f"  {name:14s} shape={str(a.shape):22s} cos={cos:.6f} "
          f"max_abs={float(diff.max()):.6e} mean_abs={float(diff.mean()):.6e}")
    return cos


def _torch_baseline(args, image, proprio, timestep, noisy_action):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import torch  # local import: baseline only
    from export_octo_onnx import OctoDenoiseNet  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = OctoDenoiseNet(
        img_size=args.img_size, patch=args.patch, dim=args.dim, trunk_depth=args.trunk_depth,
        heads=args.heads, num_readout=args.num_readout, proprio_dim=args.proprio_dim,
        action_dim=args.action_dim, horizon=args.horizon).eval()
    with torch.no_grad():
        noise = model(torch.from_numpy(image), torch.from_numpy(proprio),
                      torch.from_numpy(timestep), torch.from_numpy(noisy_action))
    return noise.cpu().numpy().astype(np.float32)


def _onnx_infer(onnx_path, image, proprio, timestep, noisy_action):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"image": image, "proprio": proprio, "timestep": timestep,
                           "noisy_action": noisy_action})
    return outs[0].astype(np.float32)


def _mslite_infer(mindir_path, image, proprio, timestep, noisy_action, device, device_id):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from infer_octo_mslite import _build_model, _run_model  # noqa: E402
    model, inputs = _build_model(mindir_path, device=device, device_id=device_id)
    outs = _run_model(model, inputs, {"image": image, "proprio": proprio, "timestep": timestep,
                                      "noisy_action": noisy_action})
    return outs[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Octo denoiser alignment (PyTorch vs ONNX vs MSLite)")
    p.add_argument("--onnx", type=str, default="./octo_onnx/octo_denoise.onnx")
    p.add_argument("--mindir", type=str, default="./octo_onnx/octo_denoise_graph.mindir")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--trunk-depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--num-readout", type=int, default=16)
    p.add_argument("--proprio-dim", type=int, default=7)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timestep", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-mslite", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((1, 3, args.img_size, args.img_size)).astype(np.float32)
    proprio = rng.standard_normal((1, args.proprio_dim)).astype(np.float32)
    timestep = np.array([args.timestep], dtype=np.int64)
    noisy_action = rng.standard_normal((1, args.horizon, args.action_dim)).astype(np.float32)

    print("=" * 78)
    print("Octo denoiser alignment: PyTorch vs ONNX vs MindSpore Lite")
    print("=" * 78)
    print(f"input: image {image.shape}, proprio {proprio.shape}, timestep {timestep}, "
          f"noisy_action {noisy_action.shape}")

    print("\n[PyTorch] OctoDenoiseNet on CPU (fp32) ...")
    pt_noise = _torch_baseline(args, image, proprio, timestep, noisy_action)

    onx_noise = None
    if not args.skip_onnx and os.path.exists(args.onnx):
        print(f"\n[ONNX] onnxruntime on {args.onnx} ...")
        onx_noise = _onnx_infer(args.onnx, image, proprio, timestep, noisy_action)
        print("  PyTorch vs ONNX:")
        _stats("noise", pt_noise, onx_noise)
    elif not args.skip_onnx:
        print(f"\n[ONNX] skipped: {args.onnx} not found.")

    if not args.skip_mslite and os.path.exists(args.mindir):
        print(f"\n[MSLite] mindspore_lite on {args.mindir} (device={args.device}) ...")
        msl_noise = _mslite_infer(args.mindir, image, proprio, timestep, noisy_action,
                                  args.device, args.device_id)
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
