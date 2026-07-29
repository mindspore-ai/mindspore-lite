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
"""ONNX Runtime inference script for SAM3 image model.

Loads the three exported ONNX models (image_encoder, language_encoder,
decoder) and runs text-prompted instance segmentation on an input image.
Also supports torch-vs-ONNX alignment verification.

Usage:
    python infer_sam3_onnx.py --onnx-dir ./onnx --image path/to/image.jpg \
        --prompt "a dog"
    python infer_sam3_onnx.py --onnx-dir ./onnx --align-check \
        --checkpoint /path/to/sam3.1_multiplex.pt
"""

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = 1008
CONTEXT_LENGTH = 32
SOT_TOKEN = 49406
EOT_TOKEN = 49407
CONF_THRESHOLD = 0.5


def preprocess_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """Load and preprocess an image for SAM3 inference.

    Args:
        image_path: Path to the input image.

    Returns:
        Tuple of (preprocessed image [1,3,1008,1008], orig_h, orig_w).
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
        Token array [1, 32] int64.
    """
    import pkg_resources
    from sam3.model.tokenizer_ve import SimpleTokenizer

    bpe_path = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    tokens = tokenizer(text, context_length=CONTEXT_LENGTH)
    return tokens.numpy().astype(np.int64)


def create_session(onnx_path: str) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Uses ORT_DISABLE_ALL to avoid graph optimization that can introduce
    numerical differences with large transformer models.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        ONNX Runtime InferenceSession.
    """
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
    )
    return session


def run_full_pipeline(
    img_session: ort.InferenceSession,
    lang_session: ort.InferenceSession,
    dec_session: ort.InferenceSession,
    image: np.ndarray,
    text_tokens: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full SAM3 inference pipeline across three ONNX models.

    Args:
        img_session: Image encoder ONNX session.
        lang_session: Language encoder ONNX session.
        dec_session: Decoder ONNX session.
        image: Preprocessed image [1, 3, 1008, 1008] float32.
        text_tokens: Tokenized text [1, 32] int64.

    Returns:
        Tuple of (pred_logits, pred_boxes, pred_masks, presence_logit).
    """
    t0 = time.time()
    img_inputs = {img_session.get_inputs()[0].name: image}
    img_outputs = img_session.run(None, img_inputs)
    img_out_names = [out.name for out in img_session.get_outputs()]
    img_out_map = dict(zip(img_out_names, img_outputs))
    t1 = time.time()

    lang_inputs = {lang_session.get_inputs()[0].name: text_tokens}
    lang_outputs = lang_session.run(None, lang_inputs)
    lang_out_names = [out.name for out in lang_session.get_outputs()]
    lang_out_map = dict(zip(lang_out_names, lang_outputs))
    t2 = time.time()

    dec_feed = {}
    for inp in dec_session.get_inputs():
        if inp.name in img_out_map:
            dec_feed[inp.name] = img_out_map[inp.name]
        elif inp.name in lang_out_map:
            dec_feed[inp.name] = lang_out_map[inp.name]
        else:
            raise KeyError(f"Cannot find ONNX input '{inp.name}' in encoder outputs")
    dec_outputs = dec_session.run(None, dec_feed)
    t3 = time.time()

    print(f"  Image encoder:  {t1 - t0:.3f}s")
    print(f"  Language encoder: {t2 - t1:.3f}s")
    print(f"  Decoder:         {t3 - t2:.3f}s")
    print(f"  Total:           {t3 - t0:.3f}s")

    return dec_outputs[0], dec_outputs[1], dec_outputs[2], dec_outputs[3]


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
        pred_masks: [1, 200, 288, 288] mask logits.
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
    for i in range(len(keep)):
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


def align_check(onnx_dir: str, checkpoint_path: str) -> None:
    """Verify ONNX outputs match PyTorch outputs with cosine similarity.

    Args:
        onnx_dir: Directory containing the three ONNX models.
        checkpoint_path: Path to the SAM3 checkpoint.
    """
    import torch
    from export_sam3_onnx import build_model, ImageEncoderWrapper, LanguageEncoderWrapper, DecoderWrapper

    print("Building PyTorch model...")
    model = build_model(checkpoint_path)

    np.random.seed(42)
    dummy_image = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    dummy_text = np.zeros((1, CONTEXT_LENGTH), dtype=np.int64)
    dummy_text[0, 0] = SOT_TOKEN
    dummy_text[0, 1] = 320
    dummy_text[0, 2] = EOT_TOKEN

    print("Running PyTorch reference...")
    with torch.no_grad():
        img_wrapper = ImageEncoderWrapper(model)
        img_wrapper.eval()
        img_torch = img_wrapper(torch.from_numpy(dummy_image))

        lang_wrapper = LanguageEncoderWrapper(model)
        lang_wrapper.eval()
        lang_torch = lang_wrapper(torch.from_numpy(dummy_text))

        dec_wrapper = DecoderWrapper(model)
        dec_wrapper.eval()
        dec_torch = dec_wrapper(*img_torch, *lang_torch)

    print("Loading ONNX sessions...")
    img_sess = create_session(os.path.join(onnx_dir, "sam3_image_encoder.onnx"))
    lang_sess = create_session(os.path.join(onnx_dir, "sam3_language_encoder.onnx"))
    dec_sess = create_session(os.path.join(onnx_dir, "sam3_decoder.onnx"))

    print("Running ONNX inference...")
    img_onnx = img_sess.run(None, {img_sess.get_inputs()[0].name: dummy_image})
    img_out_names = [out.name for out in img_sess.get_outputs()]
    img_out_map = dict(zip(img_out_names, img_onnx))

    lang_onnx = lang_sess.run(None, {lang_sess.get_inputs()[0].name: dummy_text})
    lang_out_names = [out.name for out in lang_sess.get_outputs()]
    lang_out_map = dict(zip(lang_out_names, lang_onnx))

    dec_feed = {}
    for inp in dec_sess.get_inputs():
        if inp.name in img_out_map:
            dec_feed[inp.name] = img_out_map[inp.name]
        elif inp.name in lang_out_map:
            dec_feed[inp.name] = lang_out_map[inp.name]
    dec_onnx = dec_sess.run(None, dec_feed)

    print("\n=== Image Encoder Alignment ===")
    _print_alignment(
        ["backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2",
         "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2"],
        img_torch, img_onnx
    )

    print("\n=== Language Encoder Alignment ===")
    _print_alignment(["language_features", "language_mask"], lang_torch, lang_onnx)

    print("\n=== Decoder Alignment ===")
    _print_alignment(
        ["pred_logits", "pred_boxes", "pred_masks", "presence_logit"],
        dec_torch, dec_onnx
    )


def _print_alignment(names, torch_outputs, onnx_outputs):
    """Print alignment comparison table."""
    print(f"{'Output':<25} {'Cosine Sim':>12} {'Max Abs Err':>12} {'Status':>6}")
    print("-" * 57)
    all_pass = True
    for name, t_out, o_out in zip(names, torch_outputs, onnx_outputs):
        t = t_out.numpy().flatten() if hasattr(t_out, 'numpy') else t_out.flatten()
        o = o_out.flatten()
        if t.dtype in (np.bool_, bool):
            match = np.all(t == o)
            cos_sim = 1.0 if match else 0.0
            max_err = 0.0 if match else 1.0
        else:
            cos_sim = float(np.dot(t, o) / (np.linalg.norm(t) * np.linalg.norm(o) + 1e-8))
            max_err = float(np.max(np.abs(t - o)))
        status = "PASS" if cos_sim > 0.99 else "FAIL"
        if cos_sim <= 0.99:
            all_pass = False
        print(f"{name:<25} {cos_sim:>12.6f} {max_err:>12.6f} {status:>6}")
    return all_pass


def main():
    """Parse arguments and run ONNX inference or alignment check."""
    parser = argparse.ArgumentParser(description="SAM3 ONNX inference")
    parser.add_argument("--onnx-dir", type=str, default="./onnx")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a dog")
    parser.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--align-check", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default="/home/BYD/SAM3/weight/sam3.1_multiplex.pt")
    args = parser.parse_args()

    if args.align_check:
        align_check(args.onnx_dir, args.checkpoint)
        return

    if args.image is None:
        print("Error: --image is required (or use --align-check)")
        return

    print("Loading ONNX models...")
    img_sess = create_session(os.path.join(args.onnx_dir, "sam3_image_encoder.onnx"))
    lang_sess = create_session(os.path.join(args.onnx_dir, "sam3_language_encoder.onnx"))
    dec_sess = create_session(os.path.join(args.onnx_dir, "sam3_decoder.onnx"))

    print(f"Preprocessing image: {args.image}")
    image, orig_h, orig_w = preprocess_image(args.image)
    text_tokens = tokenize_text(args.prompt)
    print(f"  Image: {orig_w}x{orig_h}, prompt: '{args.prompt}'")

    print("Running inference...")
    pred_logits, pred_boxes, pred_masks, presence_logit = run_full_pipeline(
        img_sess, lang_sess, dec_sess, image, text_tokens
    )

    results = postprocess(
        pred_logits, pred_boxes, pred_masks, presence_logit,
        orig_h, orig_w, args.conf_threshold
    )
    print(f"\nDetected {len(results)} objects (threshold={args.conf_threshold}):")
    for i, r in enumerate(results):
        print(f"  [{i}] score={r['score']:.4f} box={r['box']}")


if __name__ == "__main__":
    main()
