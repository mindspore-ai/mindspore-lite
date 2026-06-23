#!/usr/bin/env python3
"""CLIP-ViT-Base-Patch32 vision tower MindSpore Lite inference script (MindIR).

Loads the converted MindIR (`pixel_values -> image_embeds, last_hidden_state`)
and runs image-embedding inference. No torch dependency: preprocessing is numpy
+ PIL, and the model is driven by `mindspore_lite`.

Two helpers follow the repo convention:
  - `_build_model`: build a `mindspore_lite.Model` from a MindIR file on the
    requested device (cpu / ascend), returning (model, inputs).
  - `_run_model`: name-match the model inputs against a kwargs dict and cast
    every value to the input tensor's declared dtype before predict().

Zero-shot classification (optional, `--zero-shot`) needs text embeddings; those
are computed on CPU via HuggingFace `transformers` when requested. The deployed
MindIR itself is vision-only.
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite  # type: ignore
except Exception:
    mslite = None

try:
    from transformers import AutoImageProcessor
except Exception:
    AutoImageProcessor = None


# CLIP OpenAI normalization constants (matches CLIPImageProcessor defaults).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_CROP = 224
_CLIP_RESIZE = 224


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


def _preprocess_numpy(images: List[Image.Image]) -> np.ndarray:
    """CLIP preprocessing in pure numpy: resize(224,bicubic) -> center crop -> normalize."""
    out = []
    for img in images:
        w, h = img.size
        short = min(w, h)
        if short != _CLIP_RESIZE:
            new_w = max(1, round(w * _CLIP_RESIZE / short))
            new_h = max(1, round(h * _CLIP_RESIZE / short))
            img = img.resize((new_w, new_h), Image.BICUBIC)
        w, h = img.size
        left = (w - _CLIP_CROP) // 2
        top = (h - _CLIP_CROP) // 2
        img = img.crop((left, top, left + _CLIP_CROP, top + _CLIP_CROP))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.asarray(_CLIP_MEAN, dtype=np.float32)) / np.asarray(_CLIP_STD, dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1))
        out.append(arr)
    return np.stack(out, axis=0).astype(np.float32)


def _build_model(
    model_path: str,
    device: str = "cpu",
    device_id: int = 0,
) -> Tuple["mslite.Model", List["mslite.Tensor"]]:
    """Build a MindSpore Lite Model from a MindIR file and return (model, inputs)."""
    if mslite is None:
        raise RuntimeError("mindspore_lite not installed.")
    if device not in ["cpu", "ascend"]:
        raise ValueError("device must be cpu or ascend")

    context = mslite.Context()
    context.target = [device]
    if device == "ascend":
        context.ascend.device_id = int(device_id)

    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model, model.get_inputs()


_MS_DTYPE = {
    "FLOAT32": np.float32, "FLOAT16": np.float16, "FLOAT64": np.float64,
    "INT32": np.int32, "INT64": np.int64, "INT16": np.int16, "INT8": np.int8,
    "UINT8": np.uint8, "UINT32": np.uint32, "UINT64": np.uint64, "BOOL": np.bool_,
}


def _run_model(
    model: "mslite.Model",
    inputs: List["mslite.Tensor"],
    feed: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """Name-match inputs and cast each feed value to the input tensor dtype, then run."""
    name_to_tensor = {t.name: t for t in inputs}
    feed_keys = set(feed.keys())
    if feed_keys != set(name_to_tensor.keys()):
        raise RuntimeError(
            f"Input name mismatch. Model expects {sorted(name_to_tensor.keys())}, "
            f"got {sorted(feed_keys)}."
        )

    ordered = []
    for t in inputs:
        arr = np.ascontiguousarray(feed[t.name])
        target = _MS_DTYPE.get(getattr(t.dtype, "name", str(t.dtype)), np.float32)
        if arr.dtype != target:
            arr = arr.astype(target)
        # Build a fresh Tensor FROM the array (constructor infers shape/dtype),
        # matching the model input by position.
        ordered.append(mslite.Tensor(arr))
    outputs = model.predict(ordered)
    return [o.get_data_to_numpy() for o in outputs]


def _compute_text_embeds(model_id: str, prompts: List[str]) -> np.ndarray:
    """Compute CLIP text embeddings on CPU via transformers (for zero-shot)."""
    import torch  # local import: text-encoder path only
    from transformers import AutoProcessor, CLIPModel

    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    inputs = processor(text=prompts, padding="max_length", max_length=77, truncation=True,
                       return_tensors="pt")
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
    return text_embeds.cpu().numpy().astype(np.float32)


def _zero_shot(image_embeds: np.ndarray, prompts: List[str], model_id: str) -> List[List[Tuple[str, float]]]:
    """Compute cosine image*text similarities and softmax with CLIP logit_scale ~ 100."""
    text_embeds = _compute_text_embeds(model_id, prompts)
    img = image_embeds / np.maximum(np.linalg.norm(image_embeds, axis=1, keepdims=True), 1e-8)
    txt = text_embeds / np.maximum(np.linalg.norm(text_embeds, axis=1, keepdims=True), 1e-8)
    sims = img @ txt.T
    probs = _softmax(sims * 100.0, axis=-1)
    results = []
    for b in range(int(probs.shape[0])):
        order = np.argsort(probs[b])[::-1]
        results.append([(prompts[i], float(probs[b, i])) for i in order])
    return results


@dataclass
class ClipVisionOutputs:
    image_embeds: np.ndarray
    last_hidden_state: np.ndarray


class ClipVisionMsLiteInferencer:
    """CLIP vision unified-model inference class using MindSpore Lite runtime."""

    def __init__(
        self,
        model_id: str,
        model_path: str,
        device: str = "cpu",
        device_id: int = 0,
        use_processor: bool = True,
    ):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        self.model_id = model_id
        self.processor = None
        if use_processor:
            if AutoImageProcessor is None:
                raise RuntimeError("transformers not installed or incompatible.")
            self.processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)

        self.model, self.inputs = _build_model(model_path, device=device, device_id=device_id)

    def preprocess(self, images: List[Image.Image]) -> np.ndarray:
        if self.processor is not None:
            inputs = self.processor(images=images, return_tensors="np")
            return inputs["pixel_values"].astype(np.float32)
        return _preprocess_numpy(images)

    def forward(self, pixel_values: np.ndarray) -> ClipVisionOutputs:
        outs = _run_model(self.model, self.inputs, {"pixel_values": pixel_values})
        # Order matches export: image_embeds, last_hidden_state.
        return ClipVisionOutputs(image_embeds=outs[0], last_hidden_state=outs[1])

    def infer(self, image_paths: List[str]) -> ClipVisionOutputs:
        pixel_values = self.preprocess(_load_images(image_paths))
        return self.forward(pixel_values)


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="CLIP-ViT-Base-Patch32 MindSpore Lite inference (MindIR)")
    p.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--image", type=str, required=True,
                   help="Image path, or comma-separated paths for batch")
    p.add_argument("--model", type=str, required=True, help="Unified MindIR path (clip_vision.mindir)")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "ascend"])
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--no-processor", action="store_true",
                   help="Use pure-numpy preprocessing instead of AutoImageProcessor.")
    p.add_argument("--zero-shot", type=str, default="",
                   help="Comma-separated candidate labels for zero-shot classification. "
                   "Requires transformers (text encoder runs on CPU).")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=20)
    return p.parse_args()


def main():
    args = _parse_args()

    image_paths = [s for s in args.image.split(",") if s]
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    inferencer = ClipVisionMsLiteInferencer(
        model_id=args.model_id,
        model_path=args.model,
        device=args.device,
        device_id=args.device_id,
        use_processor=not args.no_processor,
    )

    pixel_values = inferencer.preprocess(_load_images(image_paths))

    mem_before = _memory_snapshot()
    for _ in range(int(args.warmup)):
        _ = inferencer.forward(pixel_values)
    lat = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = inferencer.forward(pixel_values)
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000.0)
    mem_after = _memory_snapshot()

    print("Output:")
    print(f"  image_embeds      shape={out.image_embeds.shape} dtype={out.image_embeds.dtype}")
    print(f"  last_hidden_state shape={out.last_hidden_state.shape} dtype={out.last_hidden_state.dtype}")
    print(f"  embed_norm[0]     ={float(np.linalg.norm(out.image_embeds[0])):.6f}")

    if args.zero_shot:
        prompts = [s.strip() for s in args.zero_shot.split(",") if s.strip()]
        if not prompts:
            raise ValueError("--zero-shot requires at least one label")
        results = _zero_shot(out.image_embeds, prompts, args.model_id)
        print("ZeroShot:")
        for b, one in enumerate(results):
            print(f"[{b}]")
            for label, score in one:
                print(f"  {score:.6f}\t{label}")

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
