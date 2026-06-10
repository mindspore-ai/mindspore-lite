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
Infer Qwen2.5-3B with MindSpore Lite using split MindIR (prefill + decode).
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
PREFILL_GEAR_MIN = 128
PREFILL_GEAR_MAX = 256
PREFILL_GEAR_STEP = 128


def _compute_position_ids(attention_mask):
    """Compute position ids from attention mask via cumulative sum."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """Convert a numpy array to a MindSpore Lite tensor."""
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """Build MindSpore Lite model inputs from a name-to-array dict.

    Tries to match by input name first; falls back to preferred_order.
    """
    inputs = model.get_inputs()
    if not inputs:
        order = preferred_order or list(feed_dict.keys())
        return [_mslite_tensor(feed_dict[k]) for k in order]

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
    """Extract input_ids and attention_mask from tokenizer output.

    Handles dict, BatchEncoding, numpy array, and tensor-like outputs.
    """
    if isinstance(enc, dict) or (hasattr(enc, "__getitem__") and hasattr(enc, "keys")):
        input_ids = np.array(enc["input_ids"])
        am = enc.get("attention_mask")
        attention_mask = np.array(am) if am is not None else np.ones_like(input_ids)
    elif hasattr(enc, "numpy"):
        input_ids = enc.numpy()
        attention_mask = np.ones_like(input_ids)
    else:
        input_ids = np.array(enc)
        attention_mask = np.ones_like(input_ids)
    return input_ids, attention_mask


def _ensure_2d(*arrays):
    """Ensure all arrays have 2 dimensions by adding batch dim if needed."""
    result = []
    for a in arrays:
        if a.ndim == 1:
            a = a[None, :]
        result.append(a)
    return result


def _pad_to_gear(input_ids, attention_mask, gear_len, pad_token_id):
    """Pad input_ids and attention_mask to the specified gear length."""
    pad_len = gear_len - input_ids.shape[1]
    if pad_len <= 0:
        return input_ids, attention_mask
    pad_ids = np.full((input_ids.shape[0], pad_len), pad_token_id, dtype=input_ids.dtype)
    pad_mask = np.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
    return (np.concatenate([input_ids, pad_ids], axis=1),
            np.concatenate([attention_mask, pad_mask], axis=1))


class Qwen253BInferencer:
    """Qwen2.5-3B inferencer with MindSpore Lite (prefill + decode)."""

    def __init__(self, prefill_model_path: str, decode_model_path: str,
                 tokenizer_id: str, device: str = "ascend", device_id: int = 0):
        """Load prefill/decode MindIR models and tokenizer."""
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        if device == "ascend":
            self.context.ascend.device_id = device_id

        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(prefill_model_path, mslite.ModelType.MINDIR, self.context)

        print(f"Loading decode model from {decode_model_path}...")
        self.decode_model = mslite.Model()
        self.decode_model.build_from_file(decode_model_path, mslite.ModelType.MINDIR, self.context)

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

    def _select_prefill_gear(self, seq_len: int) -> int:
        """Round up seq_len to the nearest prefill gear (10~200, step 10)."""
        seq_len = int(seq_len)
        if seq_len <= PREFILL_GEAR_MIN:
            return PREFILL_GEAR_MIN
        if seq_len >= PREFILL_GEAR_MAX:
            return PREFILL_GEAR_MAX
        return int(((seq_len + PREFILL_GEAR_STEP - 1) // PREFILL_GEAR_STEP) * PREFILL_GEAR_STEP)

    def _prepare_inputs(self, text: str, max_length: int):
        """Tokenize chat input and pad to the nearest prefill gear."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="np")
        input_ids, attention_mask = _extract_token_ids(enc)
        input_ids, attention_mask = _ensure_2d(input_ids, attention_mask)

        max_length = min(int(max_length) or KV_CACHE_LEN, PREFILL_GEAR_MAX)
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        gear_len = self._select_prefill_gear(input_ids.shape[1])
        input_ids, attention_mask = _pad_to_gear(
            input_ids, attention_mask, gear_len, int(self.tokenizer.pad_token_id))

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _stream_print_delta(self, generated_ids, prev_text: str):
        """Print incremental decoded text delta; return full decoded text so far."""
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
        """Run prefill model and return (logits, past_k, past_v, prefill_ms)."""
        inputs = _build_mslite_inputs(
            self.prefill_model,
            {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids},
            preferred_order=["input_ids", "attention_mask", "position_ids"])
        t0 = time.time()
        outputs = self.prefill_model.predict(inputs)
        prefill_ms = (time.time() - t0) * 1000
        logits = outputs[0].get_data_to_numpy()
        past_k = outputs[1].get_data_to_numpy()
        past_v = outputs[2].get_data_to_numpy()
        return logits, past_k, past_v, prefill_ms

    def _run_decode_step(self, next_token, cur_attention_mask, next_pos, past_k, past_v):
        """Run single decode step; return (logits, past_k, past_v, decode_ms)."""
        decode_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int32),
            "attention_mask": cur_attention_mask,
            "position_ids": np.array([[next_pos]], dtype=np.int32),
            "past_key_cache": past_k,
            "past_value_cache": past_v,
        }
        inputs = _build_mslite_inputs(
            self.decode_model, decode_feed,
            preferred_order=["input_ids", "attention_mask", "position_ids",
                             "past_key_cache", "past_value_cache"])
        t1 = time.time()
        outputs = self.decode_model.predict(inputs)
        decode_ms = (time.time() - t1) * 1000
        return (outputs[0].get_data_to_numpy(),
                outputs[1].get_data_to_numpy(),
                outputs[2].get_data_to_numpy(),
                decode_ms)

    def _decode_loop(self, first_token, actual_len, max_new_tokens,
                     cur_attention_mask, past_k, past_v, stream, streamed_text):
        """Run auto-regressive decode loop until EOS or max_new_tokens."""
        generated_ids = [first_token]
        valid_len = int(actual_len)
        decode_times = []

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break

            cur_attention_mask[0, valid_len] = 1
            logits, past_k, past_v, decode_ms = self._run_decode_step(
                generated_ids[-1], cur_attention_mask, valid_len, past_k, past_v)
            decode_times.append(decode_ms)
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        return generated_ids, decode_times, past_k, past_v, streamed_text

    def generate(self, text: str, max_new_tokens: int = 128,
                 max_length: int = 4096, stream: bool = True):
        """Generate text using Qwen2.5-3B with prefill + decode pipeline."""
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)
        logits, past_k, past_v, prefill_ms = self._run_prefill(input_ids, attention_mask, position_ids)

        actual_len = int(attention_mask[0].sum())
        if past_k.shape[3] != KV_CACHE_LEN or past_v.shape[3] != KV_CACHE_LEN:
            raise RuntimeError(
                f"prefill cache len mismatch, expected {KV_CACHE_LEN}, "
                f"got k={past_k.shape}, v={past_v.shape}")

        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        cur_attention_mask[0, :actual_len] = 1
        first_token = int(np.argmax(logits[0, max(actual_len - 1, 0)]))
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta([first_token], streamed_text)

        generated_ids, decode_times, _, _, streamed_text = self._decode_loop(
            first_token, actual_len, max_new_tokens,
            cur_attention_mask, past_k, past_v, stream, streamed_text)
        if stream:
            print()

        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
        total_ms = prefill_ms + total_decode_ms
        throughput = len(generated_ids) / (total_ms / 1000) if total_ms > 0 else 0
        perf = {
            "prefill_ms": prefill_ms,
            "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms,
            "total_ms": total_ms,
            "generated_tokens": len(generated_ids),
            "throughput_tok_s": throughput,
        }
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True), perf


def _print_perf(perf):
    """Print performance metrics in a formatted table."""
    print("=" * 60)
    print("\n[Performance]")
    print(f"  Prefill:        {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:   {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode:     {perf['avg_decode_ms']:.2f} ms/step")
    print(f"  Total Time:     {perf['total_ms']:.2f} ms")
    print(f"  Tokens:         {perf['generated_tokens']}")
    print(f"  Throughput:     {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)


def _parse_infer_args():
    """Parse command-line arguments for MindSpore Lite inference."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B Inference with MindSpore Lite (prefill + decode)")
    parser.add_argument("--prefill-model", type=str, required=True,
                        help="Path to prefill .mindir")
    parser.add_argument("--decode-model", type=str, required=True,
                        help="Path to decode .mindir")
    parser.add_argument("--tokenizer", type=str, default="./Qwen2.5-3B-Instruct",
                        help="Tokenizer path")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="ascend", choices=["cpu", "ascend"])
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def main():
    """Main entry: load models, run inference, print result and performance."""
    args = _parse_infer_args()

    inferencer = Qwen253BInferencer(
        prefill_model_path=args.prefill_model,
        decode_model_path=args.decode_model,
        tokenizer_id=args.tokenizer,
        device=args.device,
        device_id=args.device_id)

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(
        args.prompt, max_new_tokens=args.max_new_tokens, max_length=args.max_length)

    _print_perf(perf)


if __name__ == "__main__":
    main()
