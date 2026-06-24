"""MindSpore Lite inference for Salesforce/blip2-opt-2.7b.

Pure numpy + mslite + PIL implementation (NO torch on the core inference path).
Image preprocessing mirrors the HF ``BlipImageProcessor``: resize to 224x224,
rescale by 1/255, normalize with the OpenAI CLIP mean/std.

Greedy decoding runs against the two ONNX-exported OPT modules (prefill +
decode). The KV cache is carried as a single fp16 tensor of shape
``[64, 1, 32, max_total_len, 80]`` between decode steps; only the current
column is rewritten per step.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import mindspore_lite as mslite

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    print(f"transformers is required for tokenization: {exc}", file=sys.stderr)
    sys.exit(1)


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# OPT / GPT2 tokenizer special tokens (set by Salesforce/blip2-opt-2.7b).
DEFAULT_BOS_ID = 2      # OPT </s> reused as BOS by BLIP-2
DEFAULT_EOS_ID = 2      # OPT </s>
DEFAULT_PAD_ID = 1      # OPT <pad>
IMAGE_TOKEN_INDEX = -1  # BLIP-2 placeholder; we never feed it to OPT directly
NUM_QUERY_TOKENS = 32


# ---------------------------------------------------------------------------
# Preprocessing helpers (numpy + PIL)
# ---------------------------------------------------------------------------
def _load_image(path_or_url: str) -> Image.Image:
    """Load an image from a local path or http(s) URL; return RGB PIL."""
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url) as resp:
            img = Image.open(resp)
    else:
        img = Image.open(path_or_url)
    return img.convert("RGB")


def preprocess_image(image: Image.Image, image_size: int = 224) -> np.ndarray:
    """Resize, rescale and normalize a PIL image to a CHW float32 array."""
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC
    arr = (arr - CLIP_MEAN) / CLIP_STD  # broadcast over H, W
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return np.expand_dims(arr, axis=0)  # NCHW [1,3,224,224]


def _tokenize_question(tokenizer, question: str, question_len: int):
    """Tokenize + pad the question to ``question_len`` (right padding)."""
    enc = tokenizer(
        question,
        padding="max_length",
        truncation=True,
        max_length=question_len - 1,  # reserve 1 slot for BOS prefix
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype(np.int64)  # [1, q-1]
    bos = np.array([[tokenizer.bos_token_id]], dtype=np.int64)
    input_ids = np.concatenate([bos, input_ids], axis=1)  # [1, question_len]
    attn = np.ones_like(input_ids, dtype=np.int64)
    # Tokenizer may have already padded; mask any remaining pad positions.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    attn[input_ids == pad_id] = 0
    return input_ids, attn


# ---------------------------------------------------------------------------
# MSLite helpers
# ---------------------------------------------------------------------------
def _np_dtype_to_mslite(dtype):
    mapping = {
        np.dtype(np.float16): mslite.DataType.FLOAT16,
        np.dtype(np.float32): mslite.DataType.FLOAT32,
        np.dtype(np.int32): mslite.DataType.INT32,
        np.dtype(np.int64): mslite.DataType.INT64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _to_mslite_tensor(np_array: np.ndarray) -> mslite.Tensor:
    """Wrap a numpy array as an mslite.Tensor (constructor infers shape/dtype)."""
    return mslite.Tensor(np.ascontiguousarray(np_array))


def _build_model(model_path: str, context: mslite.Context) -> mslite.Model:
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)
    return model


def _run_model(model: mslite.Model, feed_dict: dict, preferred_order: list):
    """Run ``model`` and return a list of output numpy arrays.

    Inputs are matched by name when every declared input name is present in
    ``feed_dict``; otherwise they are bound in ``preferred_order``.
    """
    inputs = model.get_inputs()
    named = all(inp.name in feed_dict for inp in inputs)
    bound = []
    if named:
        for inp in inputs:
            arr = feed_dict[inp.name]
            if arr.dtype != _mslite_dtype_to_np(inp.dtype):
                arr = arr.astype(_mslite_dtype_to_np(inp.dtype))
            inp_t = _to_mslite_tensor(arr)
            inp_t.name = inp.name
            bound.append(inp_t)
    else:
        if len(preferred_order) != len(inputs):
            raise RuntimeError(
                f"Need {len(inputs)} inputs in preferred_order, got "
                f"{len(preferred_order)}"
            )
        for inp, arr in zip(inputs, preferred_order):
            if arr.dtype != _mslite_dtype_to_np(inp.dtype):
                arr = arr.astype(_mslite_dtype_to_np(inp.dtype))
            inp_t = _to_mslite_tensor(arr)
            inp_t.name = inp.name
            bound.append(inp_t)
    outputs = model.predict(bound)
    return [np.array(out.get_data_to_numpy(), copy=True) for out in outputs]


def _mslite_dtype_to_np(dtype) -> np.dtype:
    mapping = {
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.INT64: np.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported mslite dtype: {dtype}")
    return np.dtype(mapping[dtype])


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------
class Blip2OptMsLiteInferencer:
    """Four-stage MSLite inferencer for BLIP-2 OPT-2.7B.

    Args:
        vision_path, qformer_path, prefill_path, decode_path: MINDIR file paths.
        tokenizer_id: HF tokenizer id (defaults to the BLIP-2 checkpoint).
        device: "ascend" or "cpu".
        device_id: Ascend device id.
        image_size: square vision input size (default 224).
        question_len: fixed padded question length (default 32).
        max_total_len: fixed max total sequence length (default 256).
    """

    def __init__(
        self,
        vision_path: str,
        qformer_path: str,
        prefill_path: str,
        decode_path: str,
        tokenizer_id: str = "Salesforce/blip2-opt-2.7b",
        device: str = "ascend",
        device_id: int = 0,
        image_size: int = 224,
        question_len: int = 32,
        max_total_len: int = 256,
    ):
        if device not in ("ascend", "cpu"):
            raise ValueError(f"device must be ascend/cpu, got {device}")
        self.device = device
        self.image_size = image_size
        self.question_len = question_len
        self.max_total_len = max_total_len
        self.num_query_tokens = NUM_QUERY_TOKENS

        context = mslite.Context()
        context.target = [device]
        if device == "ascend":
            context.ascend.device_id = device_id

        self.vision_model = _build_model(vision_path, context)
        self.qformer_model = _build_model(qformer_path, context)
        self.prefill_model = _build_model(prefill_path, context)
        self.decode_model = _build_model(decode_path, context)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.bos_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else DEFAULT_BOS_ID
        )
        self.eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else DEFAULT_EOS_ID
        )

    # -- stage 1: vision encoder -------------------------------------------
    def _run_vision(self, pixel_values: np.ndarray) -> np.ndarray:
        out = _run_model(
            self.vision_model,
            {"pixel_values": pixel_values},
            preferred_order=[pixel_values],
        )
        return out[0]  # image_embeds [1, 257, 1408]

    # -- stage 2: qformer + language_projection ----------------------------
    def _run_qformer(self, image_embeds: np.ndarray):
        out = _run_model(
            self.qformer_model,
            {"image_embeds": image_embeds},
            preferred_order=[image_embeds],
        )
        query_embeds = out[0]              # [1, 32, 768]
        language_model_inputs = out[1]     # [1, 32, 2560]
        return query_embeds, language_model_inputs

    # -- stage 3: prefill --------------------------------------------------
    def _run_prefill(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
    ):
        out = _run_model(
            self.prefill_model,
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            preferred_order=[inputs_embeds, attention_mask, position_ids],
        )
        logits = out[0]
        past_kv = out[1]
        return logits, past_kv

    # -- stage 4: decode step ---------------------------------------------
    def _run_decode(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_kv: np.ndarray,
        cache_pos: int,
    ):
        cache_pos_arr = np.array([cache_pos], dtype=np.int64)
        out = _run_model(
            self.decode_model,
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_kv,
                "cache_pos": cache_pos_arr,
            },
            preferred_order=[
                inputs_embeds, attention_mask, position_ids, past_kv, cache_pos_arr
            ],
        )
        logits = out[0]
        new_past = out[1]
        return logits, new_past

    # -- top-level inference ----------------------------------------------
    def infer(
        self,
        image_path_or_url: str,
        question: str,
        max_new_tokens: int = 32,
        opt_embed_tokens: np.ndarray = None,
    ):
        """Run end-to-end inference.

        Args:
            image_path_or_url: local path or URL to the input image.
            question: question / prompt string.
            max_new_tokens: maximum number of answer tokens to generate.
            opt_embed_tokens: OPT ``embed_tokens.weight`` numpy matrix of shape
                ``[vocab, 2560]``. Required because the OPT embedding lookup is
                not part of the exported prefill/decode modules (they take
                ``inputs_embeds`` directly). The align script and the README
                show how to load this from the checkpoint without keeping torch
                in the hot path.

        Returns:
            (answer_str, timing_dict).
        """
        if opt_embed_tokens is None:
            raise ValueError(
                "opt_embed_tokens is required: pass the OPT embed_tokens.weight "
                "numpy array ([vocab, 2560]). See README for how to dump it once."
            )

        t_e2e = time.perf_counter()

        # 1. preprocess
        t0 = time.perf_counter()
        image = _load_image(image_path_or_url)
        pixel_values = preprocess_image(image, self.image_size).astype(np.float32)
        question_ids, question_attn = _tokenize_question(
            self.tokenizer, question, self.question_len
        )
        t_pre = (time.perf_counter() - t0) * 1000

        # 2. vision encoder
        t0 = time.perf_counter()
        image_embeds = self._run_vision(pixel_values)
        t_vis = (time.perf_counter() - t0) * 1000

        # 3. qformer + projection
        t0 = time.perf_counter()
        _, language_model_inputs = self._run_qformer(image_embeds)
        t_qf = (time.perf_counter() - t0) * 1000

        # 4. build inputs_embeds for prefill
        q_len = question_ids.shape[1]
        seq_len = self.num_query_tokens + q_len  # 32 + question_len
        question_embeds = opt_embed_tokens[question_ids[0]]  # [q_len, 2560]
        full_embeds = np.concatenate(
            [language_model_inputs[0], question_embeds], axis=0
        ).astype(np.float32)[None, :, :]  # [1, 32+q_len, 2560]
        full_attn = np.concatenate(
            [np.ones((1, self.num_query_tokens), dtype=np.int64), question_attn],
            axis=1,
        )  # [1, 32+q_len]

        # 5. prefill
        t0 = time.perf_counter()
        position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        prefill_logits, past_kv = self._run_prefill(full_embeds, full_attn, position_ids)
        t_prefill = (time.perf_counter() - t0) * 1000

        # 6. greedy decode loop
        t0 = time.perf_counter()
        generated = [int(np.argmax(prefill_logits[0, -1, :]))]
        decode_steps = 0
        if generated[-1] != self.eos_id:
            attn_full = np.zeros((1, self.max_total_len), dtype=np.int64)
            attn_full[0, :seq_len] = full_attn[0, :seq_len]
            cache_pos = seq_len
            for _ in range(max_new_tokens - 1):
                if generated[-1] == self.eos_id:
                    break
                if cache_pos >= self.max_total_len:
                    break
                step_embed = opt_embed_tokens[[generated[-1]]].astype(np.float32)
                step_embed = step_embed[None, :, :]  # [1, 1, 2560]
                attn_full[0, cache_pos] = 1
                step_pos = np.array([[cache_pos]], dtype=np.int64)
                step_logits, past_kv = self._run_decode(
                    step_embed, attn_full, step_pos, past_kv, cache_pos
                )
                next_id = int(np.argmax(step_logits[0, -1, :]))
                generated.append(next_id)
                cache_pos += 1
                decode_steps += 1
                if next_id == self.eos_id:
                    break
        t_decode = (time.perf_counter() - t0) * 1000

        e2e_ms = (time.perf_counter() - t_e2e) * 1000

        # Strip EOS (and pad) before decoding to text.
        text_ids = [t for t in generated if t != self.eos_id]
        answer = self.tokenizer.decode(text_ids, skip_special_tokens=True).strip()

        decode_avg = t_decode / max(decode_steps, 1)
        timing = {
            "preprocess_ms": t_pre,
            "vision_ms": t_vis,
            "qformer_ms": t_qf,
            "prefill_ms": t_prefill,
            "decode_total_ms": t_decode,
            "decode_steps": decode_steps,
            "decode_avg_ms": decode_avg,
            "e2e_ms": e2e_ms,
        }
        return answer, timing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _dump_opt_embed_tokens(model_id: str, out_path: str):
    """One-time helper to dump OPT embed_tokens to a .npy (uses torch)."""
    import torch  # noqa: late import; only for the offline dump helper
    from transformers import Blip2ForConditionalGeneration

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    weight = model.language_model.get_input_embeddings().weight.detach().cpu().numpy()
    np.save(out_path, weight)
    print(f"[dump] wrote OPT embed_tokens -> {out_path}  shape={weight.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="MSLite inference for Salesforce/blip2-opt-2.7b."
    )
    parser.add_argument("--vision-model", required=True)
    parser.add_argument("--qformer-model", required=True)
    parser.add_argument("--prefill-model", required=True)
    parser.add_argument("--decode-model", required=True)
    parser.add_argument("--tokenizer", default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--opt-embeddings",
                        default="opt_embed_tokens.npy",
                        help="Path to OPT embed_tokens .npy (see --dump-embeddings).")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--question-len", type=int, default=32)
    parser.add_argument("--max-total-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="ascend", choices=["ascend", "cpu"])
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dump-embeddings", action="store_true",
                        help="Dump OPT embed_tokens to --opt-embeddings and exit.")
    parser.add_argument("--model-id", default="Salesforce/blip2-opt-2.7b",
                        help="Used only with --dump-embeddings.")
    args = parser.parse_args()

    if args.dump_embeddings:
        _dump_opt_embed_tokens(args.model_id, args.opt_embeddings)
        return

    opt_embed_tokens = np.load(args.opt_embeddings)

    inferencer = Blip2OptMsLiteInferencer(
        vision_path=args.vision_model,
        qformer_path=args.qformer_model,
        prefill_path=args.prefill_model,
        decode_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
        image_size=args.image_size,
        question_len=args.question_len,
        max_total_len=args.max_total_len,
    )
    answer, timing = inferencer.infer(
        args.image, args.question,
        max_new_tokens=args.max_new_tokens,
        opt_embed_tokens=opt_embed_tokens,
    )
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print(
        "Timing(ms): "
        f"preprocess={timing['preprocess_ms']:.2f} "
        f"vision={timing['vision_ms']:.2f} "
        f"qformer={timing['qformer_ms']:.2f} "
        f"prefill={timing['prefill_ms']:.2f} "
        f"decode_total={timing['decode_total_ms']:.2f} "
        f"decode_steps={timing['decode_steps']} "
        f"decode_avg={timing['decode_avg_ms']:.2f} "
        f"e2e={timing['e2e_ms']:.2f}"
    )
    if timing["decode_steps"] > 0:
        tok_s = timing["decode_steps"] / (timing["decode_total_ms"] / 1000.0)
        print(f"Throughput: {tok_s:.2f} tok/s (decode)")


if __name__ == "__main__":
    main()
