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
"""
Infer Qwen2.5-0.5B model with ONNX Runtime.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("Error: onnxruntime package not found.")
    print("Please install: pip install onnxruntime")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: transformers package not found.")
    print("Please install: pip install transformers")
    sys.exit(1)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    position_ids = np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int64)


def _tokenize_prompt(tokenizer, prompt: str, use_chat_template: bool):
    """Tokenize the prompt, optionally applying chat template."""
    if (
        use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
        return input_ids.astype(np.int64), attention_mask.astype(np.int64)

    enc = tokenizer(prompt, return_tensors="np")
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
    return input_ids.astype(np.int64), attention_mask.astype(np.int64)


def _build_ort_session(model_path: Path, providers, low_mem: bool):
    sess_options = ort.SessionOptions()
    if low_mem:
        sess_options.enable_mem_pattern = False
        sess_options.enable_cpu_mem_arena = False
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
    return ort.InferenceSession(
        str(model_path), sess_options=sess_options, providers=providers
    )


def _get_ort_input_numpy_dtype(
    session: ort.InferenceSession, input_name: str, fallback
):
    """Get the numpy dtype for an ONNX input by inspecting the session metadata."""
    for inp in session.get_inputs():
        if inp.name != input_name:
            continue
        if inp.type == "tensor(float16)":
            return np.float16
        if inp.type == "tensor(float)":
            return np.float32
        return fallback
    return fallback


def _run_onnx(
    prefill_path,
    decode_path,
    tokenizer_path,
    prompt,
    max_new_tokens,
    device,
    use_chat_template,
    low_mem,
):
    """Run end-to-end ONNX inference with prefill-decode pipeline."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    print(f"Loading prefill model from {prefill_path}...")
    prefill_session = _build_ort_session(Path(prefill_path), providers, low_mem)
    print(f"Loading decode model from {decode_path}...")
    decode_session = _build_ort_session(Path(decode_path), providers, low_mem)

    print(f"Tokenizing prompt: '{prompt}'")
    input_ids, attention_mask = _tokenize_prompt(tokenizer, prompt, use_chat_template)
    position_ids = _compute_position_ids(attention_mask)
    print(f"Input shape: {input_ids.shape}")

    prefill_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    logits, past_kv = prefill_session.run(None, prefill_inputs)

    generated_ids = []
    next_token = int(np.argmax(logits[0, -1]))
    generated_ids.append(next_token)

    past_kv_dtype = _get_ort_input_numpy_dtype(
        decode_session, "past_key_values", np.float16
    )
    past_kv = past_kv.astype(past_kv_dtype, copy=False)

    cur_attention_mask = attention_mask
    cur_pos = int(position_ids[0, -1])

    for step in range(max_new_tokens - 1):
        print(f"Step {step + 1}/{max_new_tokens - 1}", end="\r")
        next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
        cur_attention_mask = np.concatenate(
            [cur_attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
        )
        next_position_ids = np.array([[cur_pos + 1]], dtype=np.int64)

        decode_inputs = {
            "input_ids": next_input_ids,
            "attention_mask": cur_attention_mask,
            "position_ids": next_position_ids,
            "past_key_values": past_kv,
        }
        logits, past_kv = decode_session.run(None, decode_inputs)
        past_kv = past_kv.astype(past_kv_dtype, copy=False)
        cur_pos += 1

        next_token = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token)
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            print(f"\nEOS token reached at step {step + 1}")
            break

    print("\n" + "=" * 50)
    print("Generated text:")
    print("=" * 50)
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print("=" * 50)
    return generated_ids


def main():
    """Main entry point for ONNX inference."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B ONNX inference (prefill + decode)"
    )
    parser.add_argument(
        "--prefill", type=str, required=True, help="Path to prefill model (.onnx)"
    )
    parser.add_argument(
        "--decode", type=str, required=True, help="Path to decode model (.onnx)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace tokenizer path",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, how are you?", help="Input prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="ONNXRuntime device",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template and run with raw prompt tokenization",
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="ONNX Runtime low-memory session options (may reduce peak memory)",
    )

    args = parser.parse_args()

    _run_onnx(
        prefill_path=args.prefill,
        decode_path=args.decode,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        use_chat_template=not args.no_chat_template,
        low_mem=args.low_mem,
    )


if __name__ == "__main__":
    main()
