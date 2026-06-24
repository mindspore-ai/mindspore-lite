#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""MindSpore Lite inference script for IDEA-Research/grounding-dino-base.

Loads the converted MindIR model on Ascend and produces grounded detection
results (boxes + scores + text labels) from one image + a list of class
labels. The pipeline mirrors :mod:`infer_grounding_dino_base_onnx.py` so the
two outputs can be compared directly.

The MindSpore Lite runtime is restricted to numpy / PIL — there is no
``import torch`` in this file. All torch ops used during export have numpy
equivalents implemented here.
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except Exception:
    mslite = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

PIXEL_H = 800
PIXEL_W = 1333
MAX_TEXT_LEN = 256
SPECIAL_TOKENS = (101, 102, 1012, 1029)


def _resize_to_grounding_dino_size(img: Image.Image) -> Image.Image:
    """Resize so the shorter edge >= 800 and the longer edge <= 1333."""
    w, h = img.size
    short = min(h, w)
    long = max(h, w)
    scale = 1.0
    if short < 800:
        scale = 800 / short
    if long * scale > 1333:
        scale = 1333 / long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _preprocess_image(img: Image.Image, mean: List[float],
                      std: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Resize, normalize, pad to (PIXEL_H, PIXEL_W); return pixel_values, pixel_mask."""
    img = _resize_to_grounding_dino_size(img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std,
                                                                  dtype=np.float32)
    arr = arr.transpose(2, 0, 1)

    _, h, w = arr.shape
    pad_h = PIXEL_H - h
    pad_w = PIXEL_W - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Resized image ({h}, {w}) exceeds canvas "
                         f"({PIXEL_H}, {PIXEL_W}).")

    pixel_values = np.zeros((3, PIXEL_H, PIXEL_W), dtype=np.float32)
    pixel_values[:, :h, :w] = arr
    pixel_mask = np.zeros((PIXEL_H, PIXEL_W), dtype=np.int32)
    pixel_mask[:h, :w] = 1
    return pixel_values, pixel_mask


def _build_text_inputs(tokenizer, text_labels: List[str]) -> Tuple[np.ndarray, ...]:
    """Tokenize, pad to MAX_TEXT_LEN, and build text_self_attention_masks/position_ids."""
    text = ". ".join(text_labels)
    enc = tokenizer(text, padding="max_length", max_length=MAX_TEXT_LEN,
                    truncation=True, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int32)
    token_type_ids = enc["token_type_ids"].astype(np.int32)
    attention_mask = enc["attention_mask"].astype(np.int32)
    text_self_attention_masks = _build_text_self_attention_masks(input_ids)
    text_position_ids = _build_text_position_ids(input_ids)
    return (input_ids, token_type_ids, attention_mask, text_self_attention_masks,
            text_position_ids)


def _build_text_self_attention_masks(input_ids: np.ndarray) -> np.ndarray:
    """Reproduce ``generate_masks_with_special_tokens_and_transfer_map`` in numpy."""
    batch, num = input_ids.shape
    eye = np.eye(num, dtype=bool)
    masks = np.broadcast_to(eye, (batch, num, num)).copy()
    for b in range(batch):
        specials = np.isin(input_ids[b], np.asarray(SPECIAL_TOKENS))
        idxs = np.nonzero(specials)[0]
        prev = -1
        for col in idxs:
            if col in (0, num - 1):
                masks[b, col, col] = True
            else:
                lo = prev + 1
                masks[b, lo:col + 1, lo:col + 1] = True
            prev = col
    return masks.astype(bool)


def _build_text_position_ids(input_ids: np.ndarray) -> np.ndarray:
    """Block-local position ids (0 at every special token, 1..k inside)."""
    batch, num = input_ids.shape
    pids = np.zeros((batch, num), dtype=np.int32)
    for b in range(batch):
        specials = np.isin(input_ids[b], np.asarray(SPECIAL_TOKENS))
        idxs = np.nonzero(specials)[0]
        prev = -1
        for col in idxs:
            if col not in (0, num - 1):
                lo = prev + 1
                length = col - lo
                pids[b, lo:col + 1] = np.arange(length + 1, dtype=np.int32)
            prev = col
    return pids


def _center_to_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) -> (x0, y0, x1, y1)."""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h],
                    axis=-1)


def _get_phrase_labels(probs: np.ndarray, input_ids: np.ndarray,
                       tokenizer, text_threshold: float) -> str:
    """Reproduce ``get_phrases_from_posmap`` for a single detection."""
    num = probs.shape[0]
    posmap = probs > text_threshold
    posmap = posmap.copy()
    posmap[0] = False
    posmap[num - 1:] = False
    nonzero = np.nonzero(posmap)[0]
    if len(nonzero) == 0:
        return ""
    token_ids = [int(input_ids[i]) for i in nonzero]
    phrase = tokenizer.decode(token_ids, skip_special_tokens=True)
    return phrase.strip()


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid; clamps large-magnitude logits to avoid overflow."""
    x = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _post_process(logits: np.ndarray, pred_boxes: np.ndarray,
                  input_ids: np.ndarray, tokenizer,
                  threshold: float, text_threshold: float,
                  target_size: Tuple[int, int]) -> List[dict]:
    """Return one dict per batch item with scores/boxes/text_labels."""
    probs = _sigmoid(logits)
    scores = probs.max(axis=-1)
    boxes_corners = _center_to_corners(pred_boxes)

    img_h, img_w = target_size
    scale = np.asarray([img_w, img_h, img_w, img_h], dtype=np.float32)
    boxes_corners = boxes_corners * scale

    results = []
    for b in range(logits.shape[0]):
        keep = scores[b] > threshold
        scores_b = scores[b][keep]
        boxes_b = boxes_corners[b][keep]
        probs_b = probs[b][keep]
        labels = []
        for prob_row in probs_b:
            labels.append(_get_phrase_labels(prob_row, input_ids[b], tokenizer,
                                             text_threshold))
        results.append({"scores": scores_b, "boxes": boxes_b,
                        "text_labels": labels})
    return results


def _build_mslite_inputs(model, feed_dict: dict) -> List:
    """Match the model inputs by name; fall back to a known preferred order."""
    inputs_info = model.get_inputs()
    name_to_input = {inp.name: inp for inp in inputs_info}
    matched = []
    for inp in inputs_info:
        arr = feed_dict.get(inp.name)
        if arr is None:
            raise ValueError(f"Missing input for '{inp.name}'")
        arr = np.ascontiguousarray(arr)
        inp.shape = list(arr.shape)
        inp.dtype = _np_to_mslite_dtype(arr.dtype, inp.dtype)
        inp.set_data_from_numpy(arr)
        matched.append(inp)
    del name_to_input
    return matched


def _np_to_mslite_dtype(np_dtype, mslite_dtype_hint):
    """Map numpy dtype to mslite.DataType, falling back to ``mslite_dtype_hint``."""
    del mslite_dtype_hint  # the inferred dtype from the MindIR is already correct
    if np.issubdtype(np_dtype, np.floating):
        return mslite.DataType.FLOAT32
    if np.issubdtype(np_dtype, np.bool_):
        return mslite.DataType.BOOL
    if np.issubdtype(np_dtype, np.integer):
        return mslite.DataType.INT32
    raise TypeError(f"Unsupported numpy dtype {np_dtype}")


class GroundingDinoMsLiteInferencer:
    """Wraps the MSLite model + tokenizer for grounded detection."""

    def __init__(self, mindir_path: str, model_dir: str, device_id: int = 0):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed.")
        if not os.path.exists(mindir_path):
            raise FileNotFoundError(mindir_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.context = mslite.Context()
        self.context.target = ["ascend"]
        self.context.ascend.device_id = int(device_id)
        self.model = mslite.Model()
        self.model.build_from_file(mindir_path, mslite.ModelType.MINDIR,
                                   self.context)

    def _build_feed(self, img: Image.Image, text_labels: List[str]) -> dict:
        pixel_values, pixel_mask = _preprocess_image(img, list(self.mean),
                                                     list(self.std))
        (input_ids, token_type_ids, attention_mask,
         text_self_attention_masks, text_position_ids) = _build_text_inputs(
            self.tokenizer, text_labels)
        return {
            "pixel_values": pixel_values[None].astype(np.float32),
            "pixel_mask": pixel_mask[None].astype(np.int32),
            "input_ids": input_ids.astype(np.int32),
            "token_type_ids": token_type_ids.astype(np.int32),
            "attention_mask": attention_mask.astype(np.int32),
            "text_self_attention_masks": text_self_attention_masks.astype(bool),
            "text_position_ids": text_position_ids.astype(np.int32),
        }

    def run(self, img: Image.Image, text_labels: List[str],
            threshold: float = 0.25,
            text_threshold: float = 0.25) -> Tuple[List[dict], float, float, float]:
        """Run inference and return (results, input_build_ms, infer_ms, post_ms)."""
        t0 = time.perf_counter()
        feed = self._build_feed(img, text_labels)
        t1 = time.perf_counter()
        inputs = _build_mslite_inputs(self.model, feed)
        outputs = self.model.predict(inputs)
        t2 = time.perf_counter()

        logits = outputs[0].get_data_to_numpy()
        pred_boxes = outputs[1].get_data_to_numpy()
        results = _post_process(logits, pred_boxes, feed["input_ids"],
                                self.tokenizer, threshold, text_threshold,
                                target_size=(img.size[1], img.size[0]))
        t3 = time.perf_counter()
        return results, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t3 - t2) * 1000.0


def _parse_args():
    """Parse CLI flags."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",
                   default="./outputs/grounding_dino_base.mindir",
                   help="Path to the converted MindIR model.")
    p.add_argument("--model-dir",
                   default="/data/llj/models/model_weight/grounding-dino-base",
                   help="Path to the HuggingFace model directory.")
    p.add_argument("--image", required=True, help="Path to the input image.")
    p.add_argument("--text", default="a cat", help="Comma-separated labels.")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Confidence score threshold.")
    p.add_argument("--text-threshold", type=float, default=0.25,
                   help="Per-token text score threshold.")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    return p.parse_args()


def _run_once(inferencer: GroundingDinoMsLiteInferencer, img: Image.Image,
              text_labels: List[str], args) -> Tuple[List[dict], float, float, float, float]:
    """Run one inference; return (results, input_build_ms, infer_ms, post_ms, e2e_ms)."""
    results, input_build_ms, infer_ms, post_ms = inferencer.run(
        img, text_labels, args.threshold, args.text_threshold)
    e2e_ms = input_build_ms + infer_ms + post_ms
    return results, input_build_ms, infer_ms, post_ms, e2e_ms


def main():
    """CLI entry point for MSLite inference."""
    args = _parse_args()
    img = Image.open(args.image).convert("RGB")
    text_labels = [s.strip() for s in args.text.split(",") if s.strip()]
    if not text_labels:
        text_labels = [args.text]

    inferencer = GroundingDinoMsLiteInferencer(args.model, args.model_dir,
                                               device_id=args.device_id)

    for _ in range(int(args.warmup)):
        _run_once(inferencer, img, text_labels, args)

    results_list = []
    input_build_times = []
    infer_times = []
    post_times = []
    e2e_times = []
    for _ in range(int(args.runs)):
        res, input_build_ms, infer_ms, post_ms, e2e_ms = _run_once(
            inferencer, img, text_labels, args)
        results_list.append(res)
        input_build_times.append(input_build_ms)
        infer_times.append(infer_ms)
        post_times.append(post_ms)
        e2e_times.append(e2e_ms)

    results = results_list[-1]
    for b, res in enumerate(results):
        print(f"[batch {b}] detected {len(res['scores'])} objects")
        for box, score, label in zip(res["boxes"], res["scores"],
                                     res["text_labels"]):
            box_list = [round(float(x), 2) for x in box.tolist()]
            print(f"  label='{label}' score={float(score):.4f} box={box_list}")

    print("Perf:")
    print(f"  warmup: {args.warmup} runs: {args.runs}")
    print(f"  input_build_ms_mean:  {float(np.mean(input_build_times)):.3f}  "
          f"(image resize/normalize/pad + text tokenization)")
    print(f"  inference_ms_mean:    {float(np.mean(infer_times)):.3f}  "
          f"(model forward on Ascend)")
    print(f"  postprocess_ms_mean:  {float(np.mean(post_times)):.3f}  "
          f"(sigmoid + thresholding + phrase extraction)")
    print(f"  e2e_ms_mean:          {float(np.mean(e2e_times)):.3f}")

if __name__ == "__main__":
    main()
