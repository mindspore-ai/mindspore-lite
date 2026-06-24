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

"""MindSpore Lite end-to-end inference for Salesforce/blip-vqa-base.

Loads the 3 converted MindIR sub-models (vision encoder, text encoder, text
decoder) on Ascend (or CPU) and runs the BLIP VQA pipeline. The core inference
path uses only numpy + mindspore_lite + PIL (no torch). Image preprocessing
matches BlipImageProcessor (resize / center crop / rescale / normalize with
CLIP mean & std). The question is tokenized with AutoTokenizer and padded to a
fixed length. The answer is greedy-decoded from the text decoder.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import mindspore_lite as mslite
except ImportError:
    mslite = None

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


def _np_dtype_to_mslite(dtype: np.dtype):
    """Map a numpy dtype to the corresponding mslite DataType."""
    dt = np.dtype(dtype)
    if dt == np.dtype(np.float16):
        return mslite.DataType.FLOAT16
    if dt == np.dtype(np.float32):
        return mslite.DataType.FLOAT32
    if dt == np.dtype(np.int32):
        return mslite.DataType.INT32
    if dt == np.dtype(np.int64):
        return mslite.DataType.INT64
    raise TypeError(f"unsupported numpy dtype for mslite.Tensor: {dt}")


_MS_TO_NP = {
    "FLOAT32": np.float32, "FLOAT16": np.float16, "FLOAT64": np.float64,
    "INT32": np.int32, "INT64": np.int64, "INT16": np.int16, "INT8": np.int8,
    "UINT8": np.uint8, "UINT32": np.uint32, "UINT64": np.uint64, "BOOL": np.bool_,
}


def _to_mslite_tensor(np_array: np.ndarray) -> mslite.Tensor:
    """Wrap a numpy array as an mslite.Tensor (constructor infers shape/dtype)."""
    return mslite.Tensor(np.ascontiguousarray(np_array))


def _build_model(model_path: str, context: mslite.Context) -> mslite.Model:
    """Build a MindIR model from file."""
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model


def _run_model(
    model: mslite.Model, feed_dict: dict, preferred_order: List[str]
) -> list:
    """Run a model, binding inputs by name or preferred order.

    Inputs are dtype-cast to whatever the model expects (queried from the
    model's input tensors). Returns the list of output numpy arrays.
    """
    inputs = model.get_inputs()
    tensors = []
    if inputs:
        # If all input names are known, bind by name with dtype cast.
        name_to_input = {
            getattr(t, "name", None): t for t in inputs if getattr(t, "name", "")
        }
        if name_to_input and all(k in feed_dict for k in name_to_input):
            for t in inputs:
                arr = feed_dict[t.name]
                target = _MS_TO_NP.get(getattr(t.dtype, "name", str(t.dtype)), np.float32)
                if arr.dtype != target:
                    arr = arr.astype(target)
                ts = _to_mslite_tensor(arr)
                tensors.append(ts)
        else:
            for key in preferred_order:
                tensors.append(_to_mslite_tensor(feed_dict[key]))
    else:
        for key in preferred_order:
            tensors.append(_to_mslite_tensor(feed_dict[key]))
    outputs = model.predict(tensors)
    return [o.get_data_to_numpy() for o in outputs]


class BlipVqaMsLiteInferencer:
    """End-to-end BLIP VQA inference over 3 MindIR sub-models."""

    def __init__(
        self,
        vision_path: str,
        text_encoder_path: str,
        text_decoder_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
        image_size: int = 384,
        question_len: int = 20,
    ):
        if mslite is None:
            raise RuntimeError("mindspore_lite not installed.")
        if AutoTokenizer is None:
            raise RuntimeError("transformers not installed or incompatible.")
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        self.device = str(device)
        self.image_size = int(image_size)
        self.question_len = int(question_len)

        self.context = mslite.Context()
        self.context.target = [self.device]
        if device == "ascend":
            self.context.ascend.device_id = int(device_id)

        print(f"Loading vision model: {vision_path}")
        self.vision_model = _build_model(vision_path, self.context)
        print(f"Loading text encoder model: {text_encoder_path}")
        self.text_encoder_model = _build_model(text_encoder_path, self.context)
        print(f"Loading text decoder model: {text_decoder_path}")
        self.text_decoder_model = _build_model(text_decoder_path, self.context)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.bos_id = int(getattr(self.tokenizer, "bos_token_id", 30522) or 30522)
        self.sep_id = int(getattr(self.tokenizer, "sep_token_id", 102) or 102)
        self.pad_id = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

    def _run_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        out = _run_model(
            self.vision_model,
            {"pixel_values": pixel_values.astype(np.float32)},
            preferred_order=["pixel_values"],
        )
        return out[0]

    def _run_text_encoder(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        image_embeds: np.ndarray,
    ) -> np.ndarray:
        image_attn = np.ones((1, image_embeds.shape[1]), dtype=np.int64)
        out = _run_model(
            self.text_encoder_model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_embeds": image_embeds.astype(np.float32),
                "image_attention_mask": image_attn,
            },
            preferred_order=[
                "input_ids",
                "attention_mask",
                "image_embeds",
                "image_attention_mask",
            ],
        )
        return out[0]

    def _run_decoder_step(
        self,
        decoder_input_ids: np.ndarray,
        question_embeds: np.ndarray,
    ) -> np.ndarray:
        q_attn = np.ones((1, self.question_len), dtype=np.int64)
        out = _run_model(
            self.text_decoder_model,
            {
                "decoder_input_ids": decoder_input_ids,
                "encoder_hidden_states": question_embeds.astype(np.float32),
                "encoder_attention_mask": q_attn,
            },
            preferred_order=[
                "decoder_input_ids",
                "encoder_hidden_states",
                "encoder_attention_mask",
            ],
        )
        return out[0]

    def _greedy_decode(
        self, question_embeds: np.ndarray, max_answer_len: int
    ) -> List[int]:
        """Greedy-decode the answer from the text decoder (no KV-cache)."""
        generated = [self.bos_id]
        for _ in range(max_answer_len):
            prefix = np.array([generated], dtype=np.int64)
            logits = self._run_decoder_step(prefix, question_embeds)
            next_id = int(np.argmax(logits[0, -1, :]))
            generated.append(next_id)
            if next_id == self.sep_id:
                break
        answer_ids = [
            t for t in generated[1:] if t != self.sep_id and t != self.pad_id
        ]
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
        description="BLIP VQA MindSpore Lite inference (vision + text-encoder + decoder)"
    )
    p.add_argument("--vision-model", type=str, required=True, help="vision MindIR")
    p.add_argument("--text-encoder-model", type=str, required=True)
    p.add_argument("--text-decoder-model", type=str, required=True)
    p.add_argument("--tokenizer", type=str, default="Salesforce/blip-vqa-base")
    p.add_argument("--image", type=str, required=True, help="Image path or URL")
    p.add_argument("--question", type=str, required=True, help="Question text")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--question-len", type=int, default=20)
    p.add_argument("--max-answer-len", type=int, default=10)
    p.add_argument(
        "--device", type=str, default="ascend", choices=["ascend", "cpu"]
    )
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args()


def main():
    """Run BLIP VQA MindSpore Lite inference and print answer + timing."""
    args = _parse_args()
    for f in (args.vision_model, args.text_encoder_model, args.text_decoder_model):
        if not Path(f).exists():
            raise FileNotFoundError(f)

    inferencer = BlipVqaMsLiteInferencer(
        vision_path=args.vision_model,
        text_encoder_path=args.text_encoder_model,
        text_decoder_path=args.text_decoder_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
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
