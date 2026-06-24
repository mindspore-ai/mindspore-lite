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

"""Alignment check for Salesforce/blip-vqa-base (MSLite vs HuggingFace).

Runs the HuggingFace reference model ``BlipForQuestionAnswering`` on CPU
(autoregressive ``generate``) and the exported ONNX / MindIR pipeline on the
same image + question, then compares:

  * Answer string exact match.
  * Cosine similarity of the first-step decoder logits (BOS -> first token).

Both paths feed identical preprocessed ``pixel_values`` / ``input_ids`` /
``attention_mask`` so any divergence comes from the model export, not from
preprocessing.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is required for the HF reference path.") from exc

try:
    from transformers import (
        AutoTokenizer,
        BlipForQuestionAnswering,
        BlipImageProcessor,
    )
except ImportError as exc:
    raise SystemExit(
        "transformers is required for the HF reference path."
    ) from exc

from infer_blip_vqa_base_onnx import (
    BlipVqaOnnxInferencer,
    preprocess_image,
    _tokenize_question,
)


def _load_image(path_or_url: str) -> Image.Image:
    """Load an image from a local path or http(s) URL as RGB."""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request
        from io import BytesIO

        with urllib.request.urlopen(path_or_url) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened vectors."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HfReference:
    """HuggingFace BlipForQuestionAnswering reference on CPU."""

    def __init__(self, model_id: str, image_size: int, question_len: int):
        self.model_id = model_id
        self.image_size = int(image_size)
        self.question_len = int(question_len)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = BlipImageProcessor.from_pretrained(model_id)
        print(f"Loading HF reference model {model_id} (float32, CPU) ...")
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self.model.eval().to("cpu")

    def preprocess(self, image: Image.Image, question: str):
        """Use the official HF BlipImageProcessor for the reference path."""
        pv = self.processor(images=[image], return_tensors="pt")["pixel_values"]
        pv = pv.to(torch.float32)
        input_ids, attention_mask = _tokenize_question(
            self.tokenizer, question, self.question_len
        )
        input_ids_t = torch.from_numpy(input_ids).to(torch.long)
        attention_mask_t = torch.from_numpy(attention_mask).to(torch.long)
        return pv, input_ids_t, attention_mask_t

    def first_step_logits(
        self, pixel_values, input_ids, attention_mask
    ) -> np.ndarray:
        """Return the decoder logits at the first step (decoder_input_ids=BOS).

        This mirrors the export's first decode step so logits can be compared
        token-for-token against the ONNX / MindIR pipeline.
        """
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values, return_dict=True
            )
            image_embeds = vision_outputs.last_hidden_state
            image_attn = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
            )
            question_outputs = self.model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn,
                return_dict=True,
            )
            question_embeds = question_outputs.last_hidden_state
            bos_ids = torch.full(
                (question_embeds.size(0), 1),
                fill_value=self.model.decoder_start_token_id,
                dtype=torch.long,
                device=question_embeds.device,
            )
            q_attn = torch.ones(
                question_embeds.size()[:-1],
                dtype=torch.long,
                device=question_embeds.device,
            )
            answer_output = self.model.text_decoder(
                input_ids=bos_ids,
                encoder_hidden_states=question_embeds,
                encoder_attention_mask=q_attn,
                return_dict=True,
            )
        return answer_output.logits[0, -1, :].cpu().numpy().astype(np.float32)

    def generate_answer(
        self, pixel_values, input_ids, attention_mask, max_answer_len: int
    ) -> str:
        """Run the autoregressive HF generate to obtain the reference answer."""
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=int(max_answer_len),
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def _gather_onnx_first_step_logits(
    inferencer: BlipVqaOnnxInferencer,
    pixel_values: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-run the ONNX pipeline up to the first decode logits.

    Returns (first_step_logits, question_embeds).
    """
    image_embeds = inferencer._run_vision(pixel_values)
    question_embeds = inferencer._run_text_encoder(
        input_ids, attention_mask, image_embeds
    )
    prefix = np.array([[inferencer.bos_id]], dtype=np.int64)
    logits = inferencer._run_decoder_step(prefix, question_embeds)
    return logits[0, -1, :], question_embeds


def run_alignment(args) -> int:
    """Run alignment between HF reference and ONNX pipeline."""
    image = _load_image(args.image)

    # Shared numpy preprocessing for the ONNX path (matches the export shapes).
    pv_np = preprocess_image(image, args.image_size)
    input_ids_np, attn_np = _tokenize_question(
        AutoTokenizer.from_pretrained(args.tokenizer), args.question, args.question_len
    )

    ref = HfReference(args.model_id, args.image_size, args.question_len)
    pv_ref, input_ids_ref, attn_ref = ref.preprocess(image, args.question)

    # Sanity: HF image processor output should match our numpy preprocessing.
    pv_diff = float(np.max(np.abs(pv_ref.cpu().numpy() - pv_np)))
    if pv_diff > 1e-3:
        print(
            f"Warning: pixel_values max abs diff between HF processor and numpy "
            f"preprocess = {pv_diff:.6f} (expected ~0). Check preprocessing."
        )

    print("Computing HF reference first-step logits ...")
    ref_logits = ref.first_step_logits(pv_ref, input_ids_ref, attn_ref)
    ref_answer = ref.generate_answer(
        pv_ref, input_ids_ref, attn_ref, args.max_answer_len
    )

    print("Running ONNX pipeline ...")
    onnx = BlipVqaOnnxInferencer(
        vision_path=args.vision_model,
        text_encoder_path=args.text_encoder_model,
        text_decoder_path=args.text_decoder_model,
        tokenizer_id=args.tokenizer,
        device="cpu",
        image_size=args.image_size,
        question_len=args.question_len,
    )
    onnx_logits, _ = _gather_onnx_first_step_logits(
        onnx, pv_np, input_ids_np, attn_np
    )
    onnx_answer, _ = onnx.infer(
        args.image, args.question, max_answer_len=args.max_answer_len
    )

    cos = _cosine(onnx_logits, ref_logits)
    max_abs = float(np.max(np.abs(onnx_logits - ref_logits)))

    print("\n" + "=" * 60)
    print("Alignment Report: BLIP VQA (ONNX vs HF reference, CPU)")
    print("=" * 60)
    print(f"Image:     {args.image}")
    print(f"Question:  {args.question}")
    print("-" * 60)
    print(f"HF answer:   '{ref_answer}'")
    print(f"ONNX answer: '{onnx_answer}'")
    answer_match = ref_answer.strip().lower() == onnx_answer.strip().lower()
    print(f"Answer exact-match (case-insensitive): {answer_match}")
    print("-" * 60)
    print(f"First-step logits cosine similarity: {cos:.6f}")
    print(f"First-step logits max abs diff:      {max_abs:.6f}")
    print("=" * 60)

    cos_ok = cos >= float(args.cosine_threshold)
    print(f"Cosine >= {args.cosine_threshold}: {'PASS' if cos_ok else 'FAIL'}")
    print(f"Answer match: {'PASS' if answer_match else 'FAIL'}")
    return 0 if (cos_ok and answer_match) else 1


def _parse_args():
    p = argparse.ArgumentParser(
        description="Align BLIP VQA ONNX pipeline against HF reference (CPU)."
    )
    p.add_argument("--vision-model", type=str, required=True)
    p.add_argument("--text-encoder-model", type=str, required=True)
    p.add_argument("--text-decoder-model", type=str, required=True)
    p.add_argument("--model-id", type=str, default="Salesforce/blip-vqa-base")
    p.add_argument("--tokenizer", type=str, default="Salesforce/blip-vqa-base")
    p.add_argument("--image", type=str, required=True, help="Image path or URL")
    p.add_argument("--question", type=str, required=True, help="Question text")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--question-len", type=int, default=20)
    p.add_argument("--max-answer-len", type=int, default=10)
    p.add_argument("--cosine-threshold", type=float, default=0.999)
    return p.parse_args()


def main():
    """Entry point for the BLIP VQA alignment check."""
    args = _parse_args()
    for f in (args.vision_model, args.text_encoder_model, args.text_decoder_model):
        if not Path(f).exists():
            raise FileNotFoundError(f)
    return run_alignment(args)


if __name__ == "__main__":
    sys.exit(main())
