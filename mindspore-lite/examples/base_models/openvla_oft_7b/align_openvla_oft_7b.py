#!/usr/bin/env python3
"""Accuracy alignment for the OpenVLA-OFT regression skeleton (single forward)."""

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


def _torch_baseline(args, image, task_tokens):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import torch  # local import: baseline only
    from export_openvla_oft_7b_onnx import OpenVLAOFTPolicy  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = OpenVLAOFTPolicy(args.img_size, args.patch, args.dim, args.depth, args.heads,
                             args.horizon, args.action_dim, args.vocab_size, args.task_len).eval()
    with torch.no_grad():
        action = model(torch.from_numpy(image), torch.from_numpy(task_tokens))
    return action.cpu().numpy().astype(np.float32)


def _onnx_infer(onnx_path, image, task_tokens):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"image": image, "task_tokens": task_tokens})[0].astype(np.float32)


def _mslite_infer(mindir_path, image, task_tokens, device, device_id):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from infer_openvla_oft_7b_mslite import _build_model, _run_model  # noqa: E402
    model, inputs = _build_model(mindir_path, device=device, device_id=device_id)
    return _run_model(model, inputs, {"image": image, "task_tokens": task_tokens})[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="OpenVLA-OFT alignment (PyTorch vs ONNX vs MSLite)")
    p.add_argument("--onnx", type=str, default="./openvla_oft_7b_onnx/openvla_oft_7b_policy.onnx")
    p.add_argument("--mindir", type=str, default="./openvla_oft_7b_onnx/openvla_oft_7b_policy_graph.mindir")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--task-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-mslite", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((1, 3, args.img_size, args.img_size)).astype(np.float32)
    task_tokens = rng.integers(0, args.vocab_size, (1, args.task_len)).astype(np.int64)

    print("=" * 78)
    print("OpenVLA-OFT alignment: PyTorch vs ONNX vs MindSpore Lite")
    print("=" * 78)
    print(f"input: image {image.shape}, task_tokens {task_tokens.shape}")

    print("\n[PyTorch] OpenVLAOFTPolicy on CPU (fp32) ...")
    pt_action = _torch_baseline(args, image, task_tokens)

    onx_action = None
    if not args.skip_onnx and os.path.exists(args.onnx):
        print(f"\n[ONNX] onnxruntime on {args.onnx} ...")
        onx_action = _onnx_infer(args.onnx, image, task_tokens)
        print("  PyTorch vs ONNX:")
        _stats("action", pt_action, onx_action)
    elif not args.skip_onnx:
        print(f"\n[ONNX] skipped: {args.onnx} not found.")

    if not args.skip_mslite and os.path.exists(args.mindir):
        print(f"\n[MSLite] mindspore_lite on {args.mindir} (device={args.device}) ...")
        msl_action = _mslite_infer(args.mindir, image, task_tokens, args.device, args.device_id)
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
