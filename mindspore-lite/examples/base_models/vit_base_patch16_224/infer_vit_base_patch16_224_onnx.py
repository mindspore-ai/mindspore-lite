#!/usr/bin/env python3
"""ViT-B/16-224 unified ONNXRuntime inference script."""
import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from transformers import AutoConfig, AutoImageProcessor
except Exception:
    AutoConfig = None
    AutoImageProcessor = None


def _read_proc_status_kb(key: str) -> Optional[int]:
    """Read process memory status from /proc/self/status."""
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


def _memory_snapshot() -> dict:
    return {
        "vmrss_kb": _read_proc_status_kb("VmRSS"),
        "vmhwm_kb": _read_proc_status_kb("VmHWM"),
    }


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _load_images(paths: List[str]) -> List[Image.Image]:
    out = []
    for p in paths:
        out.append(Image.open(p).convert("RGB"))
    return out


def _pick_providers(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@dataclass
class VitOutputs:
    logits: np.ndarray
    probs: np.ndarray


class VitOnnxInferencer:
    """ViT unified-model inference class using ONNXRuntime."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str = "cpu",
    ):
        if ort is None:
            raise RuntimeError("onnxruntime not installed. Please install: pip install onnxruntime")
        if AutoConfig is None or AutoImageProcessor is None:
            raise RuntimeError("transformers not installed or incompatible.")
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be cpu or cuda")

        self.cfg = AutoConfig.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        providers = _pick_providers(device)
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)

    def preprocess(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="np")
        pixel_values = inputs["pixel_values"].astype(np.float32)
        return pixel_values

    def forward(self, pixel_values: np.ndarray) -> np.ndarray:
        outputs = self.sess.run(None, {"pixel_values": pixel_values})
        return outputs[0]

    def infer(self, image_paths: List[str]) -> VitOutputs:
        images = _load_images(image_paths)
        pixel_values = self.preprocess(images)
        logits = self.forward(pixel_values)
        probs = _softmax(logits, axis=-1)
        return VitOutputs(logits=logits, probs=probs)

    def topk(self, probs: np.ndarray, k: int = 5) -> List[List[Tuple[int, float, str]]]:
        """Return top-k predictions with labels."""
        k = int(k)
        results = []
        for b in range(int(probs.shape[0])):
            row = probs[b]
            idx = np.argsort(row)[::-1][:k].tolist()
            one = []
            for i in idx:
                label = getattr(self.cfg, "id2label", {}).get(int(i), str(i))
                one.append((int(i), float(row[i]), str(label)))
            results.append(one)
        return results


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="ViT-B/16-224 ONNXRuntime inference (CPU/GPU)")
    p.add_argument("--model-id", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--image", type=str, required=True, help="Image path, or comma-separated paths for batch")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--model", type=str, required=True, help="Unified ONNX path (vit_unified.onnx)")
    return p.parse_args()


def main():
    args = _parse_args()

    image_paths = [s for s in args.image.split(",") if s]
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    inferencer = VitOnnxInferencer(
        model_id=args.model_id,
        model_path=args.model,
        device=args.device,
    )

    pixel_values = inferencer.preprocess(_load_images(image_paths))

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        _ = inferencer.forward(pixel_values)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        logits = inferencer.forward(pixel_values)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)
    mem_after = _memory_snapshot()

    probs = _softmax(logits, axis=-1)
    topk = inferencer.topk(probs, k=args.topk)

    print("TopK:")
    for b, one in enumerate(topk):
        print(f"[{b}]")
        for idx, score, label in one:
            print(f"  {idx}\t{score:.6f}\t{label}")

    lat_np = np.array(lat, dtype=np.float32)
    print("Perf:")
    print(f"  batch_size: {pixel_values.shape[0]}")
    print(f"  warmup: {int(args.warmup)} runs: {int(args.runs)}")
    print(f"  latency_ms_mean: {float(lat_np.mean()):.3f}")
    print(f"  latency_ms_p50:  {float(np.percentile(lat_np, 50)):.3f}")
    print(f"  latency_ms_p90:  {float(np.percentile(lat_np, 90)):.3f}")
    print(f"  latency_ms_p99:  {float(np.percentile(lat_np, 99)):.3f}")
    print(f"  mem_before: {mem_before}")
    print(f"  mem_after:  {mem_after}")


if __name__ == "__main__":
    main()
