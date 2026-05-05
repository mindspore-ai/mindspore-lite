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
Infer Qwen3-1.7B with MindSpore Lite using split MindIR (prefill + decode).
"""

import argparse
import sys

import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)


def _compute_position_ids(attention_mask):
    """
    Compute position ids from attention mask.
    """
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """
    Convert numpy array to MindSpore Lite tensor.
    """
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """
    Build MindSpore Lite model inputs.
    """
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]

    ok_by_name = True
    for t in inputs:
        name = getattr(t, "name", None)
        if name is None or name not in feed_dict:
            ok_by_name = False
            break
    if ok_by_name:
        return [_mslite_tensor(feed_dict[t.name]) for t in inputs]

    if preferred_order:
        return [_mslite_tensor(feed_dict[k]) for k in preferred_order]

    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


class Qwen317BInferencer:
    """
    Qwen3-1.7B inferencer with MindSpore Lite.
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
        Initialize Qwen3-1.7B inferencer.
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
        Prepare inputs for Qwen3-1.7B inference.
        """
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
        )
        if isinstance(enc, dict):
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
        else:
            input_ids = enc
            attention_mask = np.ones_like(input_ids)

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        if max_length is not None and int(max_length) > 0:
            max_length = int(max_length)
            if input_ids.shape[1] > max_length:
                input_ids = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _stream_print_token(self, token_id: int):
        """
        Stream print token text.
        """
        token_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if token_text:
            print(token_text, end="", flush=True)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        stream: bool = True,
    ):
        """
        Generate text using Qwen3-1.7B.
        """
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)

        prefill_feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        inputs = _build_mslite_inputs(
            self.prefill_model,
            prefill_feed,
            preferred_order=["input_ids", "attention_mask", "position_ids"],
        )
        prefill_outputs = self.prefill_model.predict(inputs)
        logits = prefill_outputs[0].get_data_to_numpy()
        past_kv = prefill_outputs[1].get_data_to_numpy()

        generated_ids = [int(np.argmax(logits[0, -1]))]
        if stream:
            self._stream_print_token(generated_ids[-1])
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

            decode_feed = {
                "input_ids": next_input_ids,
                "attention_mask": cur_attention_mask,
                "position_ids": next_position_ids,
                "past_key_values": past_kv,
            }
            inputs = _build_mslite_inputs(
                self.decode_model,
                decode_feed,
                preferred_order=[
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "past_key_values",
                ],
            )
            decode_outputs = self.decode_model.predict(inputs)
            logits = decode_outputs[0].get_data_to_numpy()
            past_kv = decode_outputs[1].get_data_to_numpy()
            cur_pos += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                self._stream_print_token(generated_ids[-1])

        if stream:
            print()

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B Inference with MindSpore Lite (prefill + decode)"
    )
    parser.add_argument(
        "--prefill-model", type=str, required=True, help="Path to prefill .mindir"
    )
    parser.add_argument(
        "--decode-model", type=str, required=True, help="Path to decode .mindir"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="./Qwen3-1.7B", help="Tokenizer path"
    )
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="ascend", choices=["cpu", "ascend"]
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inferencer = Qwen317BInferencer(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id,
    )

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
