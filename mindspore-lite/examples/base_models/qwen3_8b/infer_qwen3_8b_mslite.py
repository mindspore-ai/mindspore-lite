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
Infer Qwen3-8B with MindSpore Lite using split MindIR (prefill + decode).
"""

import argparse
import sys
import time

import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    print("Please install them first.")
    sys.exit(1)

KV_CACHE_LEN = 512
PREFILL_GEARS = [64, 128, 256]

DECODE_INPUT_ORDER = [
    "input_ids", "attention_mask", "position_ids",
    "past_key_cache", "past_value_cache",
]
PREFILL_INPUT_ORDER = ["input_ids", "attention_mask", "position_ids"]


def _compute_position_ids(attention_mask):
    """Compute cumulative position ids from attention mask, zeroing padded positions."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """Wrap a numpy array as a MindSpore Lite Tensor."""
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build model input tensor list, matching by name or falling back to preferred_order."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]

    ok_by_name = all(
        getattr(t, "name", None) is not None and getattr(t, "name", None) in feed_dict
        for t in inputs
    )
    if ok_by_name:
        return [_mslite_tensor(feed_dict[t.name]) for t in inputs]

    if preferred_order:
        return [_mslite_tensor(feed_dict[k]) for k in preferred_order]

    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}"
    )


def _run_prefill(inferencer, input_ids, attention_mask, position_ids):
    """Run prefill model and return (logits, past_k, past_v, actual_len, prefill_ms)."""
    prefill_feed = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    inputs = _build_mslite_inputs(inferencer.prefill_model, prefill_feed,
                                   preferred_order=PREFILL_INPUT_ORDER)
    t0 = time.time()
    outputs = inferencer.prefill_model.predict(inputs)
    prefill_ms = (time.time() - t0) * 1000.0
    logits = outputs[0].get_data_to_numpy()
    past_k = outputs[1].get_data_to_numpy()
    past_v = outputs[2].get_data_to_numpy()

    actual_len = int(attention_mask[0].sum())
    if past_k.shape[3] != KV_CACHE_LEN or past_v.shape[3] != KV_CACHE_LEN:
        raise RuntimeError(
            f"prefill cache len mismatch, expected {KV_CACHE_LEN}, "
            f"got k={past_k.shape}, v={past_v.shape}"
        )
    return logits, past_k, past_v, actual_len, prefill_ms


def _init_decode_state(actual_len, logits):
    """Initialize decode loop state: attention mask, first token id, valid length."""
    cur_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
    if actual_len > 0:
        cur_mask[0, :actual_len] = 1
    last_idx = max(actual_len - 1, 0)
    first_token = int(np.argmax(logits[0, last_idx]))
    return cur_mask, first_token, int(actual_len)


def _run_single_decode(inferencer, token_id, cur_mask, position, past_k, past_v):
    """Run one decode step, return (logits, past_k, past_v, decode_ms)."""
    next_ids = np.array([[token_id]], dtype=np.int32)
    next_pos = np.array([[position]], dtype=np.int32)
    decode_feed = {
        "input_ids": next_ids,
        "attention_mask": cur_mask,
        "position_ids": next_pos,
        "past_key_cache": past_k,
        "past_value_cache": past_v,
    }
    inputs = _build_mslite_inputs(inferencer.decode_model, decode_feed,
                                   preferred_order=DECODE_INPUT_ORDER)
    t1 = time.time()
    outputs = inferencer.decode_model.predict(inputs)
    decode_ms = (time.time() - t1) * 1000.0
    return (outputs[0].get_data_to_numpy(),
            outputs[1].get_data_to_numpy(),
            outputs[2].get_data_to_numpy(),
            decode_ms)


def _tokenize_prompt(tokenizer, text):
    """Tokenize prompt using chat template, return (input_ids, attention_mask)."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, return_tensors="np",
    )
    # apply_chat_template returns different types across transformers versions:
    #   4.x -> numpy.ndarray (the input_ids directly)
    #   5.x -> BatchEncoding (has input_ids/attention_mask keys, but is NOT a
    #          dict subclass, so isinstance(enc, dict) is False). Detect by attr.
    if hasattr(enc, "input_ids"):
        input_ids = np.asarray(enc["input_ids"])
        attention_mask = np.asarray(enc["attention_mask"])
    elif isinstance(enc, dict):
        input_ids = np.asarray(enc["input_ids"])
        attention_mask = np.asarray(enc.get("attention_mask", np.ones_like(input_ids)))
    else:
        input_ids = np.asarray(enc)
        attention_mask = np.ones_like(input_ids)

    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    if attention_mask.ndim == 1:
        attention_mask = attention_mask[None, :]
    return input_ids, attention_mask


def _select_prefill_gear(seq_len):
    """Select the nearest prefill gear size >= seq_len."""
    seq_len = int(seq_len)
    for gear in PREFILL_GEARS:
        if seq_len <= gear:
            return gear
    return PREFILL_GEARS[-1]


def _truncate_to_max_length(input_ids, attention_mask, max_length):
    """Truncate input sequences to max_length from the right side."""
    max_length = int(max_length) if max_length and int(max_length) > 0 else KV_CACHE_LEN
    max_length = min(max_length, PREFILL_GEARS[-1])
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
    return input_ids, attention_mask


def _pad_to_gear(tokenizer, input_ids, attention_mask, gear_len):
    """Pad input_ids and attention_mask so seq_len matches the prefill gear."""
    seq_len = int(input_ids.shape[1])
    if gear_len <= seq_len:
        return input_ids, attention_mask
    pad_len = gear_len - seq_len
    pad_id = int(tokenizer.pad_token_id)
    pad_ids = np.full((input_ids.shape[0], pad_len), pad_id, dtype=input_ids.dtype)
    pad_mask = np.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
    input_ids = np.concatenate([input_ids, pad_ids], axis=1)
    attention_mask = np.concatenate([attention_mask, pad_mask], axis=1)
    return input_ids, attention_mask


class Qwen38BInferencer:
    """Qwen3-8B inferencer with MindSpore Lite (prefill + decode split)."""

    def __init__(self, prefill_model_path, decode_model_path, tokenizer_id,
                 device="ascend", device_id=0):
        """Load prefill/decode MindIR models, tokenizer, and set up device context."""
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
            prefill_model_path, mslite.ModelType.MINDIR, self.context)

        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(
            decode_model_path, mslite.ModelType.MINDIR, self.context)

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def _select_prefill_gear(self, seq_len):
        """Select the nearest prefill gear size >= seq_len."""
        return _select_prefill_gear(seq_len)

    def _prepare_inputs(self, text, max_length):
        """Tokenize prompt, truncate, pad to gear, and return (ids, mask, position_ids)."""
        input_ids, attention_mask = _tokenize_prompt(self.tokenizer, text)
        input_ids, attention_mask = _truncate_to_max_length(input_ids, attention_mask, max_length)

        gear_len = self._select_prefill_gear(int(input_ids.shape[1]))
        input_ids, attention_mask = _pad_to_gear(self.tokenizer, input_ids, attention_mask, gear_len)

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _stream_print_delta(self, generated_ids, prev_text):
        """Decode generated ids, print only the incremental delta since prev_text."""
        new_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if prev_text and new_text.startswith(prev_text):
            delta = new_text[len(prev_text):]
        else:
            n = min(len(prev_text), len(new_text))
            i = 0
            while i < n and prev_text[i] == new_text[i]:
                i += 1
            delta = new_text[i:]
        if delta:
            delta = delta.replace("�", "")
        if delta:
            print(delta, end="", flush=True)
        return new_text

    def generate(self, text, max_new_tokens=128, max_length=4096, stream=True):
        """Generate text: prefill prompt, then auto-regressive decode loop."""
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)
        logits, past_k, past_v, actual_len, prefill_ms = _run_prefill(
            self, input_ids, attention_mask, position_ids)

        cur_mask, first_token, valid_len = _init_decode_state(actual_len, logits)
        generated_ids = [first_token]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        decode_times = []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break

            cur_mask[0, valid_len] = 1
            logits, past_k, past_v, decode_ms = _run_single_decode(
                self, generated_ids[-1], cur_mask, valid_len, past_k, past_v)
            decode_times.append(decode_ms)
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        if stream:
            print()

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0.0
        total_ms = prefill_ms + total_decode_ms
        perf = {
            "prefill_ms": prefill_ms, "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms, "total_ms": total_ms,
            "num_generated": len(generated_ids),
            "throughput_tok_s": len(generated_ids) / (total_ms / 1000.0) if total_ms > 0 else 0.0,
        }
        return result, perf


def _parse_args():
    """Parse command-line arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B Inference with MindSpore Lite (prefill + decode)")
    parser.add_argument("--prefill-model", type=str, required=True,
                        help="Path to prefill .mindir")
    parser.add_argument("--decode-model", type=str, required=True,
                        help="Path to decode .mindir")
    parser.add_argument("--tokenizer", type=str, default="./Qwen3-8B",
                        help="Tokenizer path")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def main():
    """Parse args, build inferencer, run generation, and print results."""
    args = _parse_args()
    inferencer = Qwen38BInferencer(
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
    _, perf = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    print("\n--- Performance ---")
    print(f"  Prefill:           {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:      {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode Step:   {perf['avg_decode_ms']:.2f} ms")
    print(f"  Total:             {perf['total_ms']:.2f} ms")
    print(f"  Tokens Generated:  {perf['num_generated']}")
    print(f"  Throughput:        {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
