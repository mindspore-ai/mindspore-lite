#!/usr/bin/env python3
"""Accuracy alignment for the LLaVA-1.5 skeleton (single forward logits)."""

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


def _torch_baseline(args, image, input_ids):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import torch  # local import: baseline only
    from export_llava_1_5_7b_onnx import LlavaSkel  # noqa: E402

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = LlavaSkel(args.img_size, args.patch, args.dim, args.depth, args.heads,
                      args.vocab_size, args.seq_len).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(image), torch.from_numpy(input_ids))
    return logits.cpu().numpy().astype(np.float32)


def _onnx_infer(onnx_path, image, input_ids):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess.run(None, {"image": image, "input_ids": input_ids})[0].astype(np.float32)


def _mslite_infer(mindir_path, image, input_ids, device, device_id):
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from infer_llava_1_5_7b_mslite import _build_model, _run_model  # noqa: E402
    model, inputs = _build_model(mindir_path, device=device, device_id=device_id)
    return _run_model(model, inputs, {"image": image, "input_ids": input_ids})[0].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="LLaVA-1.5 alignment (PyTorch vs ONNX vs MSLite)")
    p.add_argument("--onnx", type=str, default="./llava_1_5_7b_onnx/llava_1_5_7b.onnx")
    p.add_argument("--mindir", type=str, default="./llava_1_5_7b_onnx/llava_1_5_7b_graph.mindir")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-onnx", action="store_true")
    p.add_argument("--skip-mslite", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    image = rng.standard_normal((1, 3, args.img_size, args.img_size)).astype(np.float32)
    input_ids = rng.integers(0, args.vocab_size, (1, args.seq_len)).astype(np.int64)

    print("=" * 78)
    print("LLaVA-1.5 alignment: PyTorch vs ONNX vs MindSpore Lite")
    print("=" * 78)
    print(f"input: image {image.shape}, input_ids {input_ids.shape}")

    print("\n[PyTorch] LlavaSkel on CPU (fp32) ...")
    pt_logits = _torch_baseline(args, image, input_ids)

    onx_logits = None
    if not args.skip_onnx and os.path.exists(args.onnx):
        print(f"\n[ONNX] onnxruntime on {args.onnx} ...")
        onx_logits = _onnx_infer(args.onnx, image, input_ids)
        print("  PyTorch vs ONNX:")
        _stats("logits", pt_logits, onx_logits)
    elif not args.skip_onnx:
        print(f"\n[ONNX] skipped: {args.onnx} not found.")

    if not args.skip_mslite and os.path.exists(args.mindir):
        print(f"\n[MSLite] mindspore_lite on {args.mindir} (device={args.device}) ...")
        msl_logits = _mslite_infer(args.mindir, image, input_ids, args.device, args.device_id)
        print("  PyTorch vs MSLite:")
        _stats("logits", pt_logits, msl_logits)
        if onx_logits is not None:
            print("  ONNX vs MSLite:")
            _stats("logits", onx_logits, msl_logits)
    elif not args.skip_mslite:
        print(f"\n[MSLite] skipped: {args.mindir} not found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
