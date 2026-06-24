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

"""ONNXRuntime end-to-end inference for Salesforce/blip-vqa-base.

Runs the 3 ONNX sub-models (vision encoder, text encoder, text decoder) with
onnxruntime. Image preprocessing matches BlipImageProcessor (resize / center
crop / rescale / normalize with CLIP mean & std) using only PIL + numpy. The
question is tokenized with AutoTokenizer and padded to a fixed length. The
answer is greedy-decoded from the text decoder.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _load_image(path_or_url: str) -> Image.Image:
    """Load an image from a local path or http(s) URL as RGB."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request
        from io import BytesIO

        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _resize_short(image: Image.Image, size: int) -> Image.Image:
    """Resize so the shorter edge equals ``size`` (bilinear, BLIP default)."""
    w, h = image.size
    if h <= w:
        new_h, new_w = size, int(round(size * w / h))
    else:
        new_h, new_w = int(round(size * h / w)), size
    return image.resize((new_w, new_h), Image.BILINEAR)


def _center_crop(image: Image.Image, crop: int) -> Image.Image:
    """Center-crop to ``crop`` x ``crop``."""
    w, h = image.size
    top = (h - crop) // 2
    left = (w - crop) // 2
    return image.crop((left, top, left + crop, top + crop))


def preprocess_image(image: Image.Image, image_size: int = 384) -> np.ndarray:
    """Reproduce BlipImageProcessor for a single RGB image.

    Steps: resize shorter edge to image_size -> center crop image_size x
    image_size -> rescale 1/255 -> normalize with CLIP mean/std. Output shape
    is (1, 3, H, W) float32 (NCHW).
    """
    img = _resize_short(image, image_size)
    img = _center_crop(img, image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # -> 1,C,H,W
    return np.ascontiguousarray(arr, dtype=np.float32)


def _tokenize_question(
    tokenizer, question: str, question_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Tokenize the question with special tokens and pad to question_len."""
    enc = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=question_len,
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    return input_ids, attention_mask


def _build_providers(device: str) -> List[str]:
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class BlipVqaOnnxInferencer:
    """End-to-end BLIP VQA inference over 3 ONNX sub-models."""

    def __init__(
        self,
        vision_path: str,
        text_encoder_path: str,
        text_decoder_path: str,
        tokenizer_id: str,
        device: str = "cpu",
        image_size: int = 384,
        question_len: int = 20,
    ):
        if ort is None:
            raise RuntimeError(
                "onnxruntime not installed. pip install onnxruntime"
            )
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed or incompatible.")
        if device not in ["cpu", "cuda"]:
            raise ValueError("device must be cpu or cuda")

        self.image_size = int(image_size)
        self.question_len = int(question_len)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

        providers = _build_providers(device)
        so = ort.SessionOptions()
        self.vision_sess = ort.InferenceSession(vision_path, sess_options=so, providers=providers)
        self.text_enc_sess = ort.InferenceSession(
            text_encoder_path, sess_options=so, providers=providers
        )
        self.text_dec_sess = ort.InferenceSession(
            text_decoder_path, sess_options=so, providers=providers
        )

        self.bos_id = int(getattr(self.tokenizer, "bos_token_id", 30522) or 30522)
        self.sep_id = int(getattr(self.tokenizer, "sep_token_id", 102) or 102)
        self.pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

    def _run_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        out = self.vision_sess.run(None, {"pixel_values": pixel_values})
        return out[0]

    def _run_text_encoder(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        image_embeds: np.ndarray,
    ) -> np.ndarray:
        image_attn = np.ones((1, image_embeds.shape[1]), dtype=np.int64)
        out = self.text_enc_sess.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds,
                "image_attention_mask": image_attn,
            },
        )
        return out[0]

    def _run_decoder_step(
        self,
        decoder_input_ids: np.ndarray,
        question_embeds: np.ndarray,
        question_len: int,
    ) -> np.ndarray:
        q_attn = np.ones((1, question_len), dtype=np.int64)
        out = self.text_dec_sess.run(
            None,
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": question_embeds,
                "encoder_attention_mask": q_attn,
            },
        )
        return out[0]

    def _greedy_decode(
        self, question_embeds: np.ndarray, max_answer_len: int
    ) -> List[int]:
        """Greedy-decode the answer from the text decoder.

        Each step re-feeds the full answer prefix (no KV-cache export).
        """
        generated = [self.bos_id]
        q_len = int(question_embeds.shape[1])
        for _ in range(max_answer_len):
            prefix = np.array([generated], dtype=np.int64)
            logits = self._run_decoder_step(prefix, question_embeds, q_len)
            next_id = int(np.argmax(logits[0, -1, :]))
            generated.append(next_id)
            if next_id == self.sep_id:
                break
        # Drop the leading BOS and any trailing SEP.
        answer_ids = [t for t in generated[1:] if t != self.sep_id and t != self.pad_id]
        return answer_ids

    def infer(
        self,
        image_path_or_url: str,
        question: str,
        max_answer_len: int = 10,
    ) -> Tuple[str, dict]:
        """Run full pipeline: preprocess -> vision -> text-encoder -> decode."""
        t_e2e = time.perf_counter()

        t0 = time.perf_counter()
        image = _load_image(image_path_or_url)
        pixel_values = preprocess_image(image, self.image_size)
        input_ids, attention_mask = _tokenize_question(
            self.tokenizer, question, self.question_len
        )
        t_pre = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        image_embeds = self._run_vision(pixel_values)
        t_vis = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        question_embeds = self._run_text_encoder(
            input_ids, attention_mask, image_embeds
        )
        t_enc = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        answer_ids = self._greedy_decode(question_embeds, max_answer_len)
        t_dec = (time.perf_counter() - t0) * 1000.0

        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        timing = {
            "preprocess_ms": t_pre,
            "vision_ms": t_vis,
            "text_encoder_ms": t_enc,
            "decode_ms": t_dec,
            "e2e_ms": (time.perf_counter() - t_e2e) * 1000.0,
        }
        return answer, timing


def _parse_args():
    p = argparse.ArgumentParser(
        description="BLIP VQA end-to-end ONNXRuntime inference."
    )
    p.add_argument("--vision-model", type=str, required=True)
    p.add_argument("--text-encoder-model", type=str, required=True)
    p.add_argument("--text-decoder-model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="Salesforce/blip-vqa-base")
    p.add_argument("--image", type=str, required=True, help="Image path or URL")
    p.add_argument("--question", type=str, required=True, help="Question text")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--question-len", type=int, default=20)
    p.add_argument("--max-answer-len", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()


def main():
    """Run BLIP VQA ONNX inference and print answer + timing."""
    args = _parse_args()
    for f in (args.vision_model, args.text_encoder_model, args.text_decoder_model):
        if not Path(f).exists():
            raise FileNotFoundError(f)

    inferencer = BlipVqaOnnxInferencer(
        vision_path=args.vision_model,
        text_encoder_path=args.text_encoder_model,
        text_decoder_path=args.text_decoder_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        image_size=args.image_size,
        question_len=args.question_len,
    )
    answer, timing = inferencer.infer(
        args.image, args.question, max_answer_len=args.max_answer_len
    )

    print("\n" + "=" * 50)
    print(f"Image:    {args.image}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print("-" * 50)
    print(
        "Timing(ms): "
        f"preprocess={timing['preprocess_ms']:.3f} "
        f"vision={timing['vision_ms']:.3f} "
        f"text_encoder={timing['text_encoder_ms']:.3f} "
        f"decode={timing['decode_ms']:.3f} "
        f"e2e={timing['e2e_ms']:.3f}"
    )
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
