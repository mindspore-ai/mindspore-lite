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
Infer Qwen3-0.6B with MindSpore Lite using MindIR model.
"""

import sys
import argparse
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    """
    Compute position ids for MindIR inference.
    """
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


class Qwen3Inferencer:
    """
    Qwen3-0.6B inferencer using split MindIR (prefill + decode).
    """

    def __init__(
        self,
        prefill_model_path: str,
        decode_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
    ):
        """
        Initialize Qwen3Inferencer.
        """
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def _prepare_inputs(self, text: str, max_length: int):
        """
        Prepare input tensors for MindIR inference.
        """
        enc = self.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc.get("attention_mask", np.ones_like(input_ids)).astype(
            np.int32
        )
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 2048,
    ):
        """
        Generate text using MindIR inference.
        """
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)

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
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
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
    """
    Main function for Qwen3-0.6B inference with MindSpore Lite.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Inference with MindSpore Lite using split MindIR (prefill + decode)"
    )
    parser.add_argument(
        "--prefill-model",
        type=str,
        required=True,
        help="Path to qwen3_llm_prefill.mindir",
    )
    parser.add_argument(
        "--decode-model",
        type=str,
        required=True,
        help="Path to qwen3_llm_decode.mindir",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B-Instruct",
        help="Tokenizer model ID or path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="你好，请介绍一下你自己。",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--max-length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--device", type=str, default="ascend", help="Device for inference (cpu/ascend)"
    )
    parser.add_argument("--device-id", type=int, default=0, help="Device ID for ascend")

    args = parser.parse_args()

    inferencer = Qwen3Inferencer(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)

    print("\nGenerating response...")
    result = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )

    print("\n" + "=" * 60)
    print(f"Generated Response: {result}")
    print("=" * 60)


if __name__ == "__main__":
    main()
