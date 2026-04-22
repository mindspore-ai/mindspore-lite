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
ONNX Runtime inference for Qwen2-0.5B (prefill + decode split).
CLI: python infer_qwen2_onnx.py --prefill <path> --decode <path> --tokenizer <path>
API: from infer_qwen2_onnx import Qwen2OnnxInferencer; inferencer = Qwen2OnnxInferencer(...);
result = inferencer.generate(...)
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
    """position_ids = cumsum(mask) - 1, zeros where mask == 0."""
    position_ids = np.cumsum(attention_mask.astype(np.int64), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int64)


def _tokenize_prompt(tokenizer, prompt: str, use_chat_template: bool):
    """Tokenize with optional chat template."""
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
    return enc["input_ids"].astype(np.int64), enc.get(
        "attention_mask", np.ones_like(enc["input_ids"])
    ).astype(np.int64)


def _build_ort_session(model_path: Path, providers, low_mem: bool):
    """Build ONNX Runtime session with optional low-memory config."""
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


def _get_ort_input_dtype(session: ort.InferenceSession, input_name: str, fallback):
    """Map ONNX dtype string to numpy dtype."""
    for inp in session.get_inputs():
        if inp.name != input_name:
            continue
        if inp.type == "tensor(float16)":
            return np.float16
        if inp.type == "tensor(float)":
            return np.float32
        return fallback
    return fallback


class Qwen2OnnxInferencer:
    """
    Qwen2-0.5B inference via ONNX Runtime (prefill + decode split).

    Args:
        prefill_path:  Path to qwen2_llm_prefill.onnx
        decode_path:   Path to qwen2_llm_decode.onnx
        tokenizer_path: HuggingFace tokenizer path or local directory
        device:        "cpu" or "cuda"
        low_mem:       Enable low-memory session options
    """

    def __init__(
            self,
            prefill_path: str,
            decode_path: str,
            tokenizer_path: str,
            device: str = "cpu",
            low_mem: bool = False,
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.prefill_session = _build_ort_session(
            Path(prefill_path), providers, low_mem
        )
        self.decode_session = _build_ort_session(Path(decode_path), providers, low_mem)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.past_kv_dtype = _get_ort_input_dtype(
            self.decode_session, "past_key_values", np.float16
        )
        self._stop_tokens = set()
        if self.tokenizer.eos_token_id is not None:
            self._stop_tokens.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, "im_end_token_id") and self.tokenizer.im_end_token_id is not None:
            self._stop_tokens.add(self.tokenizer.im_end_token_id)
        self._stop_tokens.add(151645)

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 128,
            use_chat_template: bool = True,
    ):
        """
        End-to-end text generation (prefill → decode loop).

        Returns:
            generated_ids (list[int]): token IDs generated
        """
        input_ids, attention_mask = _tokenize_prompt(
            self.tokenizer, prompt, use_chat_template
        )
        position_ids = _compute_position_ids(attention_mask)

        prefill_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        logits, past_kv = self.prefill_session.run(None, prefill_inputs)

        generated_ids = []
        next_token = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token)

        past_kv = past_kv.astype(self.past_kv_dtype, copy=False)
        cur_attention_mask = attention_mask
        cur_pos = int(position_ids[0, -1])

        for _ in range(max_new_tokens - 1):
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
            logits, past_kv = self.decode_session.run(None, decode_inputs)
            past_kv = past_kv.astype(self.past_kv_dtype, copy=False)
            cur_pos += 1

            next_token = int(np.argmax(logits[0, -1]))
            generated_ids.append(next_token)
            if (
                    next_token in self._stop_tokens
            ):
                break

        return generated_ids

    def infer(self, prompt: str, max_new_tokens: int = 128, use_chat_template: bool = True):
        """Alias for generate that also returns decoded string."""
        ids = self.generate(prompt, max_new_tokens, use_chat_template)
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return text, ids


def _run_onnx(prefill, decode, tokenizer, prompt, max_new_tokens, device, use_chat_template, low_mem):
    """CLI runner."""
    inferencer = Qwen2OnnxInferencer(
        prefill_path=prefill,
        decode_path=decode,
        tokenizer_path=tokenizer,
        device=device,
        low_mem=low_mem,
    )
    print(f"Prompt: {prompt}")
    print(f"Max new tokens: {max_new_tokens}")
    print("=" * 50)
    text, _ = inferencer.infer(prompt, max_new_tokens, use_chat_template)
    print("Generated text:")
    print(text)
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-0.5B ONNX inference (prefill + decode split)"
    )
    parser.add_argument(
        "--prefill", type=str, required=True, help="Path to prefill .onnx"
    )
    parser.add_argument(
        "--decode", type=str, required=True, help="Path to decode .onnx"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models",
        help="HuggingFace tokenizer path (default: models)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello, how are you?", help="Input prompt"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128, help="Max new tokens to generate"
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
        help="Disable chat template",
    )
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="Enable low-memory ONNX session options",
    )

    args = parser.parse_args()

    _run_onnx(
        prefill=args.prefill,
        decode=args.decode,
        tokenizer=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        use_chat_template=not args.no_chat_template,
        low_mem=args.low_mem,
    )


if __name__ == "__main__":
    main()
