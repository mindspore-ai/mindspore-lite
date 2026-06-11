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
Infer Qwen2-7B-Instruct with MindSpore Lite using split MindIR (prefill + decode).
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


def _compute_position_ids(attention_mask):
    """Compute position ids from attention mask by cumulative sum."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """Convert a numpy array to a MindSpore Lite tensor."""
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build MindSpore Lite model inputs by matching tensor names or preferred order."""
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


def _extract_token_ids(enc):
    """Extract input_ids and attention_mask from tokenizer output in various formats."""
    if hasattr(enc, "__getitem__") and "input_ids" in enc:
        input_ids = np.array(enc["input_ids"])
        attention_mask = np.array(enc.get("attention_mask", np.ones_like(input_ids)))
    elif isinstance(enc, np.ndarray):
        input_ids = enc
        attention_mask = np.ones_like(input_ids)
    else:
        input_ids = np.array(enc)
        attention_mask = np.ones_like(input_ids)
    return input_ids, attention_mask


def _ensure_2d(input_ids, attention_mask):
    """Ensure input_ids and attention_mask have 2 dimensions (batch, seq)."""
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    if attention_mask.ndim == 1:
        attention_mask = attention_mask[None, :]
    return input_ids, attention_mask


def _truncate_to_max_length(input_ids, attention_mask, max_length):
    """Truncate input sequences to max_length from the right."""
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
    return input_ids, attention_mask


def _pad_to_gear(input_ids, attention_mask, gear_len, pad_id):
    """Pad input_ids and attention_mask to the specified gear length."""
    seq_len = int(input_ids.shape[1])
    if gear_len <= seq_len:
        return input_ids, attention_mask
    pad_len = gear_len - seq_len
    pad_ids = np.full((input_ids.shape[0], pad_len), pad_id, dtype=input_ids.dtype)
    pad_mask = np.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
    input_ids = np.concatenate([input_ids, pad_ids], axis=1)
    attention_mask = np.concatenate([attention_mask, pad_mask], axis=1)
    return input_ids, attention_mask


def _select_prefill_gear(seq_len):
    """Select the nearest prefill gear size >= seq_len within configured gears."""
    seq_len = int(seq_len)
    for gear in PREFILL_GEARS:
        if seq_len <= gear:
            return gear
    return PREFILL_GEARS[-1]


class Qwen27BInferencer:
    """Qwen2-7B-Instruct inferencer with MindSpore Lite."""

    def __init__(self, prefill_model_path, decode_model_path, tokenizer_id,
                 device="ascend", device_id=0):
        """Initialize Qwen2-7B inferencer with models, tokenizer, and device context."""
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        self.context = self._create_context(device, device_id)
        self.prefill_model = self._load_model(prefill_model_path, "prefill")
        self.decode_model = self._load_model(decode_model_path, "decode")
        self.tokenizer = self._load_tokenizer(tokenizer_id)
        self.eos_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _create_context(device, device_id):
        """Create and configure MindSpore Lite context for the target device."""
        print(f"Initializing MindSpore Lite context for {device}...")
        context = mslite.Context()
        context.target = [device]
        if device == "ascend":
            context.ascend.device_id = device_id
        return context

    def _load_model(self, model_path, label):
        """Load a MindIR model file and build it for inference."""
        print(f"Loading {label} model from {model_path}...")
        model = mslite.Model()
        model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)
        return model

    @staticmethod
    def _load_tokenizer(tokenizer_id):
        """Load tokenizer from the specified path and ensure pad_token is set."""
        print(f"Loading tokenizer from {tokenizer_id}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _prepare_inputs(self, text, max_length):
        """Tokenize input text and prepare padded inputs matching prefill gear dimensions."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="np",
        )
        input_ids, attention_mask = _extract_token_ids(enc)
        input_ids, attention_mask = _ensure_2d(input_ids, attention_mask)

        max_length = min(int(max_length) or KV_CACHE_LEN, PREFILL_GEARS[-1])
        input_ids, attention_mask = _truncate_to_max_length(input_ids, attention_mask, max_length)

        gear_len = _select_prefill_gear(int(input_ids.shape[1]))
        input_ids, attention_mask = _pad_to_gear(
            input_ids, attention_mask, gear_len, int(self.tokenizer.pad_token_id))

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _stream_print_delta(self, generated_ids, prev_text):
        """Print incremental decoded text delta and return the full decoded text."""
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

    def _run_prefill(self, input_ids, attention_mask, position_ids):
        """Run prefill inference and return logits, KV cache, and timing."""
        feed = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        inputs = _build_mslite_inputs(
            self.prefill_model, feed, preferred_order=["input_ids", "attention_mask", "position_ids"])
        t0 = time.time()
        outputs = self.prefill_model.predict(inputs)
        prefill_ms = (time.time() - t0) * 1000.0

        logits = outputs[0].get_data_to_numpy()
        past_k = outputs[1].get_data_to_numpy()
        past_v = outputs[2].get_data_to_numpy()
        return logits, past_k, past_v, prefill_ms

    def _validate_cache_shape(self, past_k, past_v):
        """Validate that KV cache has the expected shape."""
        if int(past_k.shape[3]) != KV_CACHE_LEN or int(past_v.shape[3]) != KV_CACHE_LEN:
            raise RuntimeError(
                f"prefill cache len mismatch, expected {KV_CACHE_LEN}, "
                f"got k={past_k.shape}, v={past_v.shape}")

    def _run_decode_step(self, token_id, cur_attention_mask, valid_len, past_k, past_v):
        """Run a single decode step and return updated logits, cache, and timing."""
        next_input_ids = np.array([[token_id]], dtype=np.int32)
        cur_attention_mask[0, valid_len] = 1
        next_position_ids = np.array([[valid_len]], dtype=np.int32)

        decode_feed = {
            "input_ids": next_input_ids, "attention_mask": cur_attention_mask,
            "position_ids": next_position_ids,
            "past_key_cache": past_k, "past_value_cache": past_v,
        }
        inputs = _build_mslite_inputs(
            self.decode_model, decode_feed,
            preferred_order=["input_ids", "attention_mask", "position_ids",
                              "past_key_cache", "past_value_cache"])

        t1 = time.time()
        outputs = self.decode_model.predict(inputs)
        decode_ms = (time.time() - t1) * 1000.0

        logits = outputs[0].get_data_to_numpy()
        past_k = outputs[1].get_data_to_numpy()
        past_v = outputs[2].get_data_to_numpy()
        return logits, past_k, past_v, decode_ms

    def generate(self, text, max_new_tokens=128, max_length=4096, stream=True):
        """Generate text using Qwen2-7B with prefill-decode split and return result + performance."""
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)

        logits, past_k, past_v, prefill_ms = self._run_prefill(input_ids, attention_mask, position_ids)
        self._validate_cache_shape(past_k, past_v)

        actual_len = int(attention_mask[0].sum())
        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        cur_attention_mask[0, :actual_len] = 1

        last_idx = max(actual_len - 1, 0)
        generated_ids = [int(np.argmax(logits[0, last_idx]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        valid_len, decode_times = int(actual_len), []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break

            logits, past_k, past_v, decode_ms = self._run_decode_step(
                generated_ids[-1], cur_attention_mask, valid_len, past_k, past_v)
            decode_times.append(decode_ms)
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        if stream:
            print()

        perf = self._compute_performance(prefill_ms, decode_times, generated_ids)
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result, perf

    @staticmethod
    def _compute_performance(prefill_ms, decode_times, generated_ids):
        """Compute performance metrics from timing data."""
        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0.0
        total_ms = prefill_ms + total_decode_ms
        num_generated = len(generated_ids)
        throughput = num_generated / (total_ms / 1000.0) if total_ms > 0 else 0.0
        return {
            "prefill_ms": prefill_ms, "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms, "total_ms": total_ms,
            "num_generated": num_generated, "throughput_tok_s": throughput,
        }


def _parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Qwen2-7B-Instruct Inference with MindSpore Lite (prefill + decode)")
    parser.add_argument("--prefill-model", type=str, required=True, help="Path to prefill .mindir")
    parser.add_argument("--decode-model", type=str, required=True, help="Path to decode .mindir")
    parser.add_argument("--tokenizer", type=str, default="./Qwen2-7B-Instruct/Qwen2-7B-Instruct",
                        help="Tokenizer path")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def _print_performance(perf):
    """Print performance metrics in a formatted layout."""
    print("\n--- Performance ---")
    print(f"  Prefill:           {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:      {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode Step:   {perf['avg_decode_ms']:.2f} ms")
    print(f"  Total:             {perf['total_ms']:.2f} ms")
    print(f"  Tokens Generated:  {perf['num_generated']}")
    print(f"  Throughput:        {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)


def main():
    """Main entry point: parse args, run inference, and print results with performance."""
    args = _parse_args()

    inferencer = Qwen27BInferencer(
        prefill_model_path=args.prefill_model, decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer, device=args.device, device_id=args.device_id)

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(
        args.prompt, max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    _print_performance(perf)


if __name__ == "__main__":
    main()
