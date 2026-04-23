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
MindSpore Lite inference for Qwen2-0.5B (prefill + decode split MindIR).
CLI: python infer_qwen2_mindir.py --prefill-model <path> --decode-model <path> --tokenizer <path>
API: from infer_qwen2_mindir import Qwen2MindIrInferencer; inferencer.generate(...)
"""

import argparse
import sys
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install: pip install mindspore-lite transformers")
    sys.exit(1)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    """position_ids = cumsum(mask) - 1, zeros where mask == 0. dtype int32."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


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
        return input_ids.astype(np.int32), attention_mask.astype(np.int32)

    enc = tokenizer(prompt, return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc.get(
        "attention_mask", np.ones_like(enc["input_ids"])
    ).astype(np.int32)


class Qwen2MindIrInferencer:
    """
    Qwen2-0.5B inference via MindSpore Lite (prefill + decode split MindIR).

    Args:
        prefill_model_path:  Path to qwen2_llm_prefill.mindir
        decode_model_path:   Path to qwen2_llm_decode.mindir
        tokenizer_path:      HuggingFace tokenizer path or local directory
        device:              "cpu" or "ascend"
        device_id:           Device ID for ascend (default 0)
    """

    def __init__(
            self,
            prefill_model_path: str,
            decode_model_path: str,
            tokenizer_path: str,
            device: str = "ascend",
            device_id: int = 0,
    ):
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        self._stop_tokens = set()
        if self.eos_token_id is not None:
            self._stop_tokens.add(self.eos_token_id)
        if hasattr(self.tokenizer, "im_end_token_id") and self.tokenizer.im_end_token_id is not None:
            self._stop_tokens.add(self.tokenizer.im_end_token_id)
        self._stop_tokens.add(151645)

    def _prepare_inputs(self, text: str, max_length: int, use_chat_template: bool = True):
        """Tokenize text and compute position_ids."""
        input_ids, attention_mask = _tokenize_prompt(
            self.tokenizer, text, use_chat_template
        )
        if input_ids.shape[-1] > max_length:
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        position_ids = _compute_position_ids(attention_mask)
        return input_ids.astype(np.int32), attention_mask.astype(np.int32), position_ids

    def generate(self, text: str, max_new_tokens: int = 128, max_length: int = 2048, use_chat_template: bool = True):
        """
        Text generation via prefill + autoregressive decode loop.

        Returns:
            str: decoded generated text
        """
        input_ids, attention_mask, position_ids = self._prepare_inputs(
            text, max_length, use_chat_template
        )

        prefill_inputs = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ]
        prefill_outputs = self.prefill_model.predict(prefill_inputs)
        logits = prefill_outputs[0].get_data_to_numpy()
        past_kv = prefill_outputs[1].get_data_to_numpy()

        generated_ids = []
        next_token = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token)

        cur_attention_mask = attention_mask
        cur_pos = int(position_ids[0, -1])

        for _ in range(max_new_tokens - 1):
            if generated_ids[-1] in self._stop_tokens:
                break

            next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int32)
            cur_attention_mask = np.concatenate(
                [cur_attention_mask, np.ones((1, 1), dtype=np.int32)], axis=1
            )
            next_position_ids = np.array([[cur_pos + 1]], dtype=np.int32)

            decode_inputs = [
                mslite.Tensor(next_input_ids),
                mslite.Tensor(cur_attention_mask),
                mslite.Tensor(next_position_ids),
                mslite.Tensor(past_kv),
            ]
            decode_outputs = self.decode_model.predict(decode_inputs)
            logits = decode_outputs[0].get_data_to_numpy()
            past_kv = decode_outputs[1].get_data_to_numpy()
            cur_pos += 1

            generated_ids.append(int(np.argmax(logits[0, -1])))

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-0.5B MindSpore Lite inference (prefill + decode split MindIR)"
    )
    parser.add_argument(
        "--prefill-model",
        type=str,
        required=True,
        help="Path to qwen2_llm_prefill.mindir",
    )
    parser.add_argument(
        "--decode-model",
        type=str,
        required=True,
        help="Path to qwen2_llm_decode.mindir",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models",
        help="Tokenizer path (default: models)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="ascend",
        choices=["cpu", "ascend"],
        help="Device for inference (cpu/ascend)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID for ascend",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template",
    )

    args = parser.parse_args()

    inferencer = Qwen2MindIrInferencer(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)

    result = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        use_chat_template=not args.no_chat_template,
    )

    print("\n" + "=" * 60)
    print(f"Generated Response: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
