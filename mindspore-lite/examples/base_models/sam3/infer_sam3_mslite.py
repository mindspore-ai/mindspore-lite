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
"""MindSpore Lite inference script for SAM3 image model.

Loads the three converted MindIR models (image_encoder, language_encoder,
decoder) on Ascend and produces text-prompted instance segmentation results
(masks, boxes, scores) from one image + a text prompt.

This script does NOT import torch. All computation uses numpy / PIL /
mindspore_lite only.

Usage:
    python infer_sam3_mslite.py --mindir-dir ./mindir \
        --image path/to/image.jpg --prompt "a dog"
    python infer_sam3_mslite.py --mindir-dir ./mindir --align-check \
        --onnx-dir ./onnx
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import mindspore_lite as mslite


IMAGE_SIZE = 1008
CONTEXT_LENGTH = 32
SOT_TOKEN = 49406
EOT_TOKEN = 49407
CONF_THRESHOLD = 0.5

_DTYPE_MAP = {
    "FLOAT32": np.float32,
    "FLOAT16": np.float16,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool_,
}


def _mslite_dtype_to_np(dtype) -> np.dtype:
    """Convert mslite DataType to numpy dtype."""
    key = str(dtype).rsplit('.', maxsplit=1)[-1]
    return np.dtype(_DTYPE_MAP.get(key, np.float32))


def preprocess_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """Load and preprocess an image for SAM3 inference.

    Args:
        image_path: Path to the input image.

    Returns:
        Tuple of (preprocessed image [1,3,1008,1008] float32, orig_h, orig_w).
    """
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis, ...], orig_h, orig_w


def tokenize_text(text: str) -> np.ndarray:
    """Tokenize a text prompt using the SAM3 CLIP BPE tokenizer.

    Args:
        text: Text prompt string.

    Returns:
        Token array [1, 32] int32.
    """
    import pkg_resources
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    tokens = tokenizer(text, context_length=CONTEXT_LENGTH)
    return tokens.numpy().astype(np.int64)


def _load_model(mindir_path: str) -> mslite.Model:
    """Load a MindIR model and build it for Ascend inference.

    Args:
        mindir_path: Path to the MindIR file.

    Returns:
        Built mslite.Model ready for inference.
    """
    context = mslite.Context()
    context.target = ["ascend"]
    model = mslite.Model()
    model.build_from_file(mindir_path, mslite.ModelType.MINDIR, context)
    return model


def _set_input(tensor: mslite.Tensor, data: np.ndarray) -> None:
    """Set numpy data into an mslite input tensor."""
    tensor.set_data_from_numpy(data)


def run_full_pipeline(
    img_model: mslite.Model,
    lang_model: mslite.Model,
    dec_model: mslite.Model,
    image: np.ndarray,
    text_tokens: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full SAM3 inference pipeline across three MindIR models.

    Args:
        img_model: Image encoder MindIR model.
        lang_model: Language encoder MindIR model.
        dec_model: Decoder MindIR model.
        image: Preprocessed image [1, 3, 1008, 1008] float32.
        text_tokens: Tokenized text [1, 32] int32.

    Returns:
        Tuple of (pred_logits, pred_boxes, pred_masks, presence_logit).
    """
    t0 = time.time()

    img_inputs = img_model.get_inputs()
    _set_input(img_inputs[0], image.astype(np.float32))
    img_outputs = img_model.predict(img_inputs)
    img_out_map = {out.name: out.get_data_to_numpy() for out in img_outputs}
    t1 = time.time()

    lang_inputs = lang_model.get_inputs()
    _set_input(lang_inputs[0], text_tokens.astype(np.int32))
    lang_outputs = lang_model.predict(lang_inputs)
    lang_out_map = {out.name: out.get_data_to_numpy() for out in lang_outputs}
    t2 = time.time()

    dec_inputs = dec_model.get_inputs()
    for inp in dec_inputs:
        if inp.name in img_out_map:
            data = img_out_map[inp.name]
        elif inp.name in lang_out_map:
            data = lang_out_map[inp.name]
        else:
            raise KeyError(f"Cannot find input '{inp.name}' in encoder outputs")
        if data.dtype != _mslite_dtype_to_np(inp.dtype):
            data = data.astype(_mslite_dtype_to_np(inp.dtype))
        _set_input(inp, data)
    dec_outputs = dec_model.predict(dec_inputs)
    dec_results = [out.get_data_to_numpy() for out in dec_outputs]
    t3 = time.time()

    print(f"  Image encoder:    {t1 - t0:.3f}s")
    print(f"  Language encoder: {t2 - t1:.3f}s")
    print(f"  Decoder:          {t3 - t2:.3f}s")
    print(f"  Total:            {t3 - t0:.3f}s")

    return dec_results[0], dec_results[1], dec_results[2], dec_results[3]


def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    pred_masks: np.ndarray,
    presence_logit: np.ndarray,
    orig_h: int,
    orig_w: int,
    conf_threshold: float = CONF_THRESHOLD,
) -> List[dict]:
    """Post-process SAM3 outputs to produce final detection results.

    Args:
        pred_logits: [1, 200, 1] detection scores (pre-sigmoid).
        pred_boxes: [1, 200, 4] bounding boxes (cxcywh, normalized).
        pred_masks: [1, 200, H, W] mask logits.
        presence_logit: [1, 1] presence score (pre-sigmoid).
        orig_h: Original image height.
        orig_w: Original image width.
        conf_threshold: Confidence threshold.

    Returns:
        List of detection dicts with keys: score, box, mask.
    """
    scores = 1.0 / (1.0 + np.exp(-pred_logits[0]))
    presence = 1.0 / (1.0 + np.exp(-presence_logit[0]))
    scores = (scores * presence).squeeze(-1)

    keep = scores > conf_threshold
    results = []
    for i, _ in enumerate(keep):
        if not keep[i]:
            continue
        cx, cy, w, h = pred_boxes[0, i]
        x0 = (cx - w / 2) * orig_w
        y0 = (cy - h / 2) * orig_h
        x1 = (cx + w / 2) * orig_w
        y1 = (cy + h / 2) * orig_h

        mask_logit = pred_masks[0, i]
        mask_img = Image.fromarray(mask_logit.astype(np.float32), mode="F")
        mask_img = mask_img.resize((orig_w, orig_h), Image.BILINEAR)
        mask = (1.0 / (1.0 + np.exp(-np.asarray(mask_img)))) > 0.5

        results.append({
            "score": float(scores[i]),
            "box": [float(x0), float(y0), float(x1), float(y1)],
            "mask": mask,
        })
    return results


def _find_mindir(mindir_dir: str, name: str) -> str:
    """Find the MindIR file for a model.

    Args:
        mindir_dir: Directory containing MindIR files.
        name: Model name prefix (e.g. 'sam3_image_encoder').

    Returns:
        Path to the MindIR file.
    """
    plain_path = os.path.join(mindir_dir, f"{name}.mindir")
    if os.path.exists(plain_path):
        return plain_path
    graph_path = os.path.join(mindir_dir, f"{name}_graph.mindir")
    if os.path.exists(graph_path):
        return graph_path
    raise FileNotFoundError(f"Cannot find {name} MindIR in {mindir_dir}")


def align_check(mindir_dir: str, onnx_dir: str) -> None:
    """Verify MindIR outputs match ONNX outputs with cosine similarity.

    Args:
        mindir_dir: Directory containing MindIR models.
        onnx_dir: Directory containing ONNX models.
    """
    import onnxruntime as ort

    np.random.seed(42)
    dummy_image = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    dummy_text = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
    dummy_text[0, 0] = SOT_TOKEN
    dummy_text[0, 1] = 320
    dummy_text[0, 2] = EOT_TOKEN

    print("Loading MindIR models...")
    img_model = _load_model(_find_mindir(mindir_dir, "sam3_image_encoder"))
    lang_model = _load_model(_find_mindir(mindir_dir, "sam3_language_encoder"))
    dec_model = _load_model(_find_mindir(mindir_dir, "sam3_decoder"))

    print("Loading ONNX models...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    img_sess = ort.InferenceSession(
        os.path.join(onnx_dir, "sam3_image_encoder.onnx"),
        sess_options=opts, providers=["CPUExecutionProvider"])
    lang_sess = ort.InferenceSession(
        os.path.join(onnx_dir, "sam3_language_encoder.onnx"),
        sess_options=opts, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(
        os.path.join(onnx_dir, "sam3_decoder.onnx"),
        sess_options=opts, providers=["CPUExecutionProvider"])

    print("Running ONNX inference...")
    img_onnx = img_sess.run(None, {img_sess.get_inputs()[0].name: dummy_image})
    img_onnx_map = {out.name: arr for out, arr in zip(img_sess.get_outputs(), img_onnx)}
    lang_onnx = lang_sess.run(None, {lang_sess.get_inputs()[0].name: dummy_text})
    lang_onnx_map = {out.name: arr for out, arr in zip(lang_sess.get_outputs(), lang_onnx)}

    dec_feed = {}
    for inp in dec_sess.get_inputs():
        if inp.name in img_onnx_map:
            dec_feed[inp.name] = img_onnx_map[inp.name]
        elif inp.name in lang_onnx_map:
            dec_feed[inp.name] = lang_onnx_map[inp.name]
        else:
            raise KeyError(f"Cannot find ONNX input '{inp.name}' in encoder outputs")
    dec_onnx = dec_sess.run(None, dec_feed)

    print("Running MindIR inference...")
    img_inputs = img_model.get_inputs()
    _set_input(img_inputs[0], dummy_image.astype(np.float32))
    img_ms = img_model.predict(img_inputs)
    img_ms_arr = [out.get_data_to_numpy() for out in img_ms]
    img_ms_map = {out.name: out.get_data_to_numpy() for out in img_ms}

    lang_inputs = lang_model.get_inputs()
    _set_input(lang_inputs[0], dummy_text.astype(np.int32))
    lang_ms = lang_model.predict(lang_inputs)
    lang_ms_arr = [out.get_data_to_numpy() for out in lang_ms]
    lang_ms_map = {out.name: out.get_data_to_numpy() for out in lang_ms}

    dec_inputs = dec_model.get_inputs()
    for inp in dec_inputs:
        if inp.name in img_ms_map:
            data = img_ms_map[inp.name]
        elif inp.name in lang_ms_map:
            data = lang_ms_map[inp.name]
        else:
            raise KeyError(f"Cannot find input '{inp.name}'")
        np_dtype = _mslite_dtype_to_np(inp.dtype)
        if data.dtype != np_dtype:
            data = data.astype(np_dtype)
        _set_input(inp, data)
    dec_ms = dec_model.predict(dec_inputs)
    dec_ms_arr = [out.get_data_to_numpy() for out in dec_ms]

    print("\n=== Image Encoder Alignment (MindIR vs ONNX) ===")
    _print_alignment(
        ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
         "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2"],
        img_onnx, img_ms_arr
    )

    print("\n=== Language Encoder Alignment (MindIR vs ONNX) ===")
    _print_alignment(["language_features", "language_mask"], lang_onnx, lang_ms_arr)

    print("\n=== Decoder Alignment (MindIR vs ONNX) ===")
    _print_alignment(
        ["pred_logits", "pred_boxes", "pred_masks", "presence_logit"],
        dec_onnx, dec_ms_arr
    )


def _print_alignment(names, ref_outputs, ms_outputs):
    """Print alignment comparison table."""
    print(f"{'Output':<25} {'Cosine Sim':>12} {'Max Abs Err':>12} {'Status':>6}")
    print("-" * 57)
    for name, r_out, m_out in zip(names, ref_outputs, ms_outputs):
        r = r_out.flatten()
        m = m_out.flatten()
        if r.dtype in (np.bool_, bool) or m.dtype in (np.bool_, bool):
            match = np.all(r.astype(bool) == m.astype(bool))
            cos_sim = 1.0 if match else 0.0
            max_err = 0.0 if match else 1.0
        else:
            cos_sim = float(np.dot(r, m) / (np.linalg.norm(r) * np.linalg.norm(m) + 1e-8))
            max_err = float(np.max(np.abs(r - m)))
        status = "PASS" if cos_sim > 0.99 else "FAIL"
        print(f"{name:<25} {cos_sim:>12.6f} {max_err:>12.6f} {status:>6}")


def main():
    """Parse arguments and run MSLite inference or alignment check."""
    parser = argparse.ArgumentParser(description="SAM3 MindSpore Lite inference")
    parser.add_argument("--mindir-dir", type=str, default="./mindir")
    parser.add_argument("--onnx-dir", type=str, default="./onnx")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--align-check", action="store_true")
    args = parser.parse_args()

    if args.align_check:
        align_check(args.mindir_dir, args.onnx_dir)
        return

    if args.image is None:
        print("Error: --image is required (or use --align-check)")
        return

    print("Loading MindIR models...")
    img_model = _load_model(_find_mindir(args.mindir_dir, "sam3_image_encoder"))
    lang_model = _load_model(_find_mindir(args.mindir_dir, "sam3_language_encoder"))
    dec_model = _load_model(_find_mindir(args.mindir_dir, "sam3_decoder"))

    print(f"Preprocessing image: {args.image}")
    image, orig_h, orig_w = preprocess_image(args.image)
    text_tokens = tokenize_text(args.prompt)
    print(f"  Image: {orig_w}x{orig_h}, prompt: '{args.prompt}'")

    print("Running inference...")
    pred_logits, pred_boxes, pred_masks, presence_logit = run_full_pipeline(
        img_model, lang_model, dec_model, image, text_tokens
    )
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  pred_boxes:  {pred_boxes.shape}")
    print(f"  pred_masks:  {pred_masks.shape}")
    print(f"  presence:    {presence_logit.shape}")

    results = postprocess(
        pred_logits, pred_boxes, pred_masks, presence_logit,
        orig_h, orig_w, args.conf_threshold
    )
    print(f"\nDetected {len(results)} objects (threshold={args.conf_threshold}):")
    for i, r in enumerate(results):
        print(f"  [{i}] score={r['score']:.4f} box={r['box']}")


if __name__ == "__main__":
    main()
