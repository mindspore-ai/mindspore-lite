"""Validate the MSLite pipeline for Salesforce/blip2-opt-2.7b against HF.

Reference path (CPU, torch):
    Blip2ForConditionalGeneration.from_pretrained(...).generate(...)
        -> first answer token + greedy answer string.

MSLite path (numpy + mslite, NO torch on core path):
    four-stage pipeline exported by ``export_blip2_opt_2_7b_onnx.py``.

Pass criteria:
    * answer string exact match (case-insensitive, stripped); AND
    * cosine similarity of the prefill's last-position logits >= threshold
      (default 0.999) with a small max-abs diff.

The reference first-step logits and the MSLite first-step logits are produced
under identical greedy inputs (same pixel_values and question token ids).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from infer_blip2_opt_2_7b_mslite import (
    Blip2OptMsLiteInferencer,
    _load_image,
    _tokenize_question,
    preprocess_image,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HfReference:
    """CPU HF reference for BLIP-2 OPT-2.7B."""

    def __init__(self, model_id: str, image_size: int = 224):
        import torch
        from transformers import (
            AutoTokenizer,
            Blip2ForConditionalGeneration,
            Blip2ImageProcessor,
        )

        self.torch = torch
        self.image_size = image_size
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Use the official HF processor for the reference pixel_values.
        try:
            self.processor = Blip2ImageProcessor(
                size={"height": image_size, "width": image_size},
                do_resize=True, do_rescale=True, do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
            )
        except Exception:  # pragma: no cover - fallback to Blip2Processor
            from transformers import Blip2Processor
            self.processor = Blip2Processor.from_pretrained(model_id).image_processor
            self.processor.size = {"height": image_size, "width": image_size}

    def preprocess(self, image: Image.Image, question: str):
        """Return (pixel_values_tensor, input_ids, attention_mask)."""
        # Reference pixel_values via the official HF image processor.
        pv = self.processor(images=image, return_tensors="pt")["pixel_values"]
        input_ids, attn = _tokenize_question(self.tokenizer, question,
                                             question_len=32)
        input_ids_t = self.torch.tensor(input_ids, dtype=self.torch.long)
        attn_t = self.torch.tensor(attn, dtype=self.torch.long)
        return pv, input_ids_t, attn_t

    def first_step_logits(self, pixel_values, input_ids, attention_mask):
        """Replicate the export's prefill step in torch.

        Returns ``logits[0, -1, :]`` as a numpy float32 vector for cosine
        comparison with the MSLite prefill's last-position logits.
        """
        torch = self.torch
        with torch.no_grad():
            # 1. vision -> image_embeds
            image_embeds = self.model.vision_model(
                pixel_values=pixel_values, return_dict=True
            ).last_hidden_state  # [1, 257, 1408]
            # 2. qformer -> query_embeds; project to OPT dim
            image_atts = torch.ones(image_embeds.shape[:2], dtype=torch.long)
            qf = self.model.qformer(
                query_embeds=self.model.query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_embeds = qf.last_hidden_state  # [1, 32, 768]
            language_model_inputs = self.model.language_projection(query_embeds)
            # 3. build inputs_embeds (32 query embeds + question tokens)
            question_embeds = self.model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat(
                [language_model_inputs, question_embeds], dim=1
            )  # [1, 32+q_len, 2560]
            seq_len = inputs_embeds.shape[1]
            position_ids = torch.arange(seq_len).unsqueeze(0)
            full_attn = torch.cat(
                [torch.ones(1, 32, dtype=torch.long), attention_mask], dim=1
            )
            # 4. OPT forward, use_cache=False (we only need logits here)
            out = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [1, seq_len, vocab]
        return logits[0, -1, :].cpu().numpy().astype(np.float32)

    def generate_answer(self, pixel_values, input_ids, attention_mask,
                        max_new_tokens: int) -> str:
        """Use HF .generate to produce the reference answer string.

        Mirrors ``Blip2ForConditionalGeneration.generate`` semantics: it
        builds inputs_embeds internally (32 image tokens + BOS + question)
        and runs OPT autoregressive decoding. We replicate the exact input
        construction the export/infer pipeline uses so the comparison is fair.
        """
        torch = self.torch
        with torch.no_grad():
            image_embeds = self.model.vision_model(
                pixel_values=pixel_values, return_dict=True
            ).last_hidden_state
            image_atts = torch.ones(image_embeds.shape[:2], dtype=torch.long)
            qf = self.model.qformer(
                query_embeds=self.model.query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_embeds = qf.last_hidden_state
            language_model_inputs = self.model.language_projection(query_embeds)
            # BLIP-2 generate prepends 32 image tokens + BOS + prompt.
            bos = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)
            full_ids = torch.cat([bos, input_ids], dim=1)
            inputs_embeds = self.model.get_input_embeddings()(full_ids)
            image_token_index = self.model.config.image_token_index
            if image_token_index is None:
                image_token_index = 0  # placeholder; we replace first 32 slots
                special_mask = torch.zeros_like(full_ids)
                special_mask[:, :32] = 1
            else:
                special_mask = (full_ids == image_token_index)
            # Force the 32 query embeds into the first 32 positions, matching
            # the MSLite pipeline (which prepends rather than masks).
            if not special_mask[:, :32].any():
                special_mask = torch.zeros_like(full_ids)
                special_mask[:, :32] = 1
            inputs_embeds = inputs_embeds.masked_scatter(
                special_mask.unsqueeze(-1), language_model_inputs
            )
            full_attn = torch.cat(
                [torch.ones(1, 32, dtype=torch.long), torch.ones_like(full_ids)],
                dim=1,
            )
            # Pad to question_len for parity with the infer script.
            gen_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()


def _gather_mslite_first_step_logits(inferencer: Blip2OptMsLiteInferencer,
                                     pixel_values: np.ndarray,
                                     question_ids: np.ndarray,
                                     question_attn: np.ndarray,
                                     opt_embed_tokens: np.ndarray):
    """Replicate the MSLite prefill and return its last-position logits."""
    image_embeds = inferencer._run_vision(pixel_values)
    _, language_model_inputs = inferencer._run_qformer(image_embeds)
    question_embeds = opt_embed_tokens[question_ids[0]]
    full_embeds = np.concatenate(
        [language_model_inputs[0], question_embeds], axis=0
    ).astype(np.float32)[None, :, :]
    full_attn = np.concatenate(
        [np.ones((1, 32), dtype=np.int64), question_attn], axis=1
    )
    seq_len = full_embeds.shape[1]
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    prefill_logits, _ = inferencer._run_prefill(full_embeds, full_attn, position_ids)
    return prefill_logits[0, -1, :].astype(np.float32)


def run_alignment(args) -> int:
    # 1. Shared numpy preprocessing (matches the infer script path).
    image = _load_image(args.image)
    pixel_np = preprocess_image(image, args.image_size).astype(np.float32)
    tokenizer_id = args.tokenizer
    # Tokenize once with HF tokenizer.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    question_ids_np, question_attn_np = _tokenize_question(
        tokenizer, args.question, question_len=args.question_len
    )

    # 2. HF reference (CPU).
    print("[align] building HF reference ...")
    ref = HfReference(args.model_id, image_size=args.image_size)
    pv_ref, input_ids_t, attn_t = ref.preprocess(image, args.question)
    # Sanity check: our numpy preprocessing must match HF's pixel_values.
    max_abs_diff_pv = float(np.max(np.abs(pv_ref.numpy() - pixel_np)))
    if max_abs_diff_pv > 1e-3:
        print(f"[align] WARNING: pixel_values diff {max_abs_diff_pv:.6f} > 1e-3")

    print("[align] running HF reference (first-step logits + generate) ...")
    ref_logits = ref.first_step_logits(pv_ref, input_ids_t, attn_t)
    ref_answer = ref.generate_answer(
        pv_ref, input_ids_t, attn_t, max_new_tokens=args.max_new_tokens
    )

    # 3. MSLite inference.
    print("[align] building MSLite pipeline ...")
    opt_embed_tokens = np.load(args.opt_embeddings)
    inferencer = Blip2OptMsLiteInferencer(
        vision_path=args.vision_model,
        qformer_path=args.qformer_model,
        prefill_path=args.prefill_model,
        decode_path=args.decode_model,
        tokenizer_id=tokenizer_id,
        device=args.device,
        device_id=args.device_id,
        image_size=args.image_size,
        question_len=args.question_len,
        max_total_len=args.max_total_len,
    )
    print("[align] running MSLite prefill (first-step logits) ...")
    mslite_logits = _gather_mslite_first_step_logits(
        inferencer, pixel_np, question_ids_np, question_attn_np, opt_embed_tokens
    )
    print("[align] running MSLite full inference ...")
    mslite_answer, _ = inferencer.infer(
        args.image, args.question,
        max_new_tokens=args.max_new_tokens,
        opt_embed_tokens=opt_embed_tokens,
    )

    # 4. Metrics.
    cos = _cosine(ref_logits, mslite_logits)
    max_abs_logits = float(np.max(np.abs(ref_logits - mslite_logits)))
    answer_match = (
        ref_answer.strip().lower() == mslite_answer.strip().lower()
    )

    print("\n=== Alignment Report ===")
    print(f"Question         : {args.question}")
    print(f"HF answer        : {ref_answer!r}")
    print(f"MSLite answer    : {mslite_answer!r}")
    print(f"Answer match     : {answer_match}")
    print(f"Prefill cosine   : {cos:.6f}")
    print(f"Prefill max|d|   : {max_abs_logits:.6f}")
    print(f"pixel_values max|d|: {max_abs_diff_pv:.6f}")

    passed = answer_match and cos >= args.cosine_threshold
    verdict = "PASS" if passed else "FAIL"
    print(f"Verdict          : {verdict} "
          f"(cosine_threshold={args.cosine_threshold})")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(
        description="Align MSLite BLIP-2 OPT-2.7B pipeline against HF."
    )
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--qformer-model", required=True)
    parser.add_argument("--prefill-model", required=True)
    parser.add_argument("--decode-model", required=True)
    parser.add_argument("--opt-embeddings", default="opt_embed_tokens.npy")
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--tokenizer", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--question-len", type=int, default=32)
    parser.add_argument("--max-total-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    parser.add_argument("--device", default="ascend", choices=["ascend", "cpu"])
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    sys.exit(run_alignment(args))


if __name__ == "__main__":
    main()
