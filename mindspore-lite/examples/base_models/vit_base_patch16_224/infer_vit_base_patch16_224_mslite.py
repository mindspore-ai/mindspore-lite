#!/usr/bin/env python3
"""ViT-B/16-224 unified MindSpore Lite inference script."""
import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite  # type: ignore
except Exception:
    mslite = None

try:
    from transformers import AutoConfig, AutoImageProcessor
except Exception:
    AutoConfig = None
    AutoImageProcessor = None


def _read_proc_status_kb(key: str):
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
    return {"vmrss_kb": _read_proc_status_kb("VmRSS"), "vmhwm_kb": _read_proc_status_kb("VmHWM")}


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


@dataclass
class VitOutputs:
    logits: np.ndarray
    probs: np.ndarray


class VitMsLiteInferencer:
    """ViT unified-model inference class using MindSpore Lite runtime."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str = "cpu",
        device_id: int = 0,
    ):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        if AutoConfig is None or AutoImageProcessor is None:
            raise RuntimeError("transformers not installed or incompatible.")
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        self.cfg = AutoConfig.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = int(device_id)

        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)

    def preprocess(self, images: List[Image.Image]) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="np")
        return inputs["pixel_values"].astype(np.float32)

    def forward(self, pixel_values: np.ndarray) -> np.ndarray:
        outputs = self.model.predict([mslite.Tensor(pixel_values)])
        return outputs[0].get_data_to_numpy()

    def infer(self, image_paths: List[str]) -> VitOutputs:
        pixel_values = self.preprocess(_load_images(image_paths))
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
    p = argparse.ArgumentParser(description="ViT-B/16-224 MindSpore Lite inference (MindIR)")
    p.add_argument("--model-id", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--image", type=str, required=True, help="Image path, or comma-separated paths for batch")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--model", type=str, required=True, help="Unified MindIR path (vit_unified.mindir)")
    return p.parse_args()


def main():
    args = _parse_args()

    image_paths = [s for s in args.image.split(",") if s]
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    inferencer = VitMsLiteInferencer(
        model_id=args.model_id,
        model_path=args.model,
        device=args.device,
        device_id=args.device_id,
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
