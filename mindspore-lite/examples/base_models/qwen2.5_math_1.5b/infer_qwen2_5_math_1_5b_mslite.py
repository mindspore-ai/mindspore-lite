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
Infer Qwen2.5-Math-1.5B with MindSpore Lite using split MindIR (prefill + decode).
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
PREFILL_GEAR_MIN = 10
PREFILL_GEAR_MAX = 200
PREFILL_GEAR_STEP = 10


def _compute_position_ids(attention_mask):
    """Compute position ids from attention mask."""
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


def _mslite_tensor(np_array):
    """Convert numpy array to MindSpore Lite tensor."""
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model: mslite.Model, feed_dict, preferred_order=None):
    """Build MindSpore Lite model inputs."""
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


class Qwen215BInferencer:
    """Qwen2.5-Math-1.5B inferencer with MindSpore Lite."""

    def __init__(
        self,
        prefill_model_path: str,
        decode_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
    ):
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

    def _select_prefill_gear(self, seq_len: int) -> int:
        seq_len = int(seq_len)
        if seq_len <= PREFILL_GEAR_MIN:
            return PREFILL_GEAR_MIN
        if seq_len >= PREFILL_GEAR_MAX:
            return PREFILL_GEAR_MAX
        return int(((seq_len + PREFILL_GEAR_STEP - 1) // PREFILL_GEAR_STEP) * PREFILL_GEAR_STEP)

    def _prepare_inputs(self, text: str, max_length: int):
        """Prepare inputs for Qwen2.5-Math-1.5B inference."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
        )
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

        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if attention_mask.ndim == 1:
            attention_mask = attention_mask[None, :]

        max_length = int(max_length) if max_length is not None and int(max_length) > 0 else KV_CACHE_LEN
        max_length = min(max_length, PREFILL_GEAR_MAX)
        if input_ids.shape[1] > max_length:
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]

        seq_len = int(input_ids.shape[1])
        gear_len = self._select_prefill_gear(seq_len)
        if gear_len > seq_len:
            pad_len = gear_len - seq_len
            pad_id = int(self.tokenizer.pad_token_id)
            pad_ids = np.full((input_ids.shape[0], pad_len), pad_id, dtype=input_ids.dtype)
            pad_mask = np.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
            input_ids = np.concatenate([input_ids, pad_ids], axis=1)
            attention_mask = np.concatenate([attention_mask, pad_mask], axis=1)

        input_ids = input_ids.astype(np.int32, copy=False)
        attention_mask = attention_mask.astype(np.int32, copy=False)
        position_ids = _compute_position_ids(attention_mask)
        return input_ids, attention_mask, position_ids

    def _stream_print_delta(self, generated_ids, prev_text: str):
        """Print incremental decoded text delta in stream mode."""
        new_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
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
        print("\n[Prefill] Running prefill ...", end="", flush=True)
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict(inputs)
        prefill_ms = (time.time() - t0) * 1000
        print(f" done in {prefill_ms:.2f} ms")

        logits = prefill_outputs[0].get_data_to_numpy()
        past_k = prefill_outputs[1].get_data_to_numpy()
        past_v = prefill_outputs[2].get_data_to_numpy()

        print(f"  logits shape:      {logits.shape}, dtype={logits.dtype}")
        print(f"  past_k shape:      {past_k.shape}, dtype={past_k.dtype}")
        print(f"  past_v shape:      {past_v.shape}, dtype={past_v.dtype}")

        if int(past_k.shape[3]) != KV_CACHE_LEN or int(past_v.shape[3]) != KV_CACHE_LEN:
            raise RuntimeError(
                f"prefill cache len mismatch, expected {KV_CACHE_LEN}, "
                f"got k={past_k.shape}, v={past_v.shape}"
            )
        return logits, past_k, past_v, prefill_ms

    def _run_decode_loop(self, past_k, past_v, actual_len, generated_ids,
                         max_new_tokens, stream, streamed_text):
        """Run autoregressive decode loop. Returns (generated_ids, decode_times, streamed_text)."""
        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        if actual_len > 0:
            cur_attention_mask[0, :actual_len] = 1
        valid_len = int(actual_len)

        print(f"\n[Decode] Running {max_new_tokens - 1} decode steps "
              f"(KV_CACHE_LEN={KV_CACHE_LEN}) ...")
        decode_times = []
        for step_i in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                print(f"  [step {step_i}] EOS token reached, stopping early.")
                break
            if valid_len >= KV_CACHE_LEN:
                print(f"  [step {step_i}] KV cache full (valid_len={valid_len}), stopping.")
                break

            next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int32)
            cur_attention_mask[0, valid_len] = 1
            next_position_ids = np.array([[valid_len]], dtype=np.int32)

            decode_feed = {
                "input_ids": next_input_ids,
                "attention_mask": cur_attention_mask,
                "position_ids": next_position_ids,
                "past_key_cache": past_k,
                "past_value_cache": past_v,
            }
            inputs = _build_mslite_inputs(
                self.decode_model,
                decode_feed,
                preferred_order=[
                    "input_ids", "attention_mask", "position_ids",
                    "past_key_cache", "past_value_cache",
                ],
            )

            t1 = time.time()
            decode_outputs = self.decode_model.predict(inputs)
            decode_ms = (time.time() - t1) * 1000
            decode_times.append(decode_ms)

            logits = decode_outputs[0].get_data_to_numpy()
            past_k = decode_outputs[1].get_data_to_numpy()
            past_v = decode_outputs[2].get_data_to_numpy()
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

            if step_i == 0 or (step_i + 1) % 32 == 0:
                print(f"  [step {step_i:>3d}] decode={decode_ms:.2f}ms, "
                      f"logits={logits.shape}, valid_len={valid_len}")

        if stream:
            print()

        return generated_ids, decode_times, streamed_text

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        stream: bool = True,
    ):
        """Generate text using Qwen2.5-Math-1.5B."""
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)

        actual_input_len = int(attention_mask[0].sum())
        gear_len = int(input_ids.shape[1])
        print("\n[Input Info]")
        print(f"  actual_input_len:  {actual_input_len}")
        print(f"  padded_input_len:  {gear_len} (gear)")
        print(f"  input_ids shape:   {input_ids.shape}, dtype={input_ids.dtype}")
        print(f"  attn_mask shape:   {attention_mask.shape}, dtype={attention_mask.dtype}")
        print(f"  position_ids shape:{position_ids.shape}, dtype={position_ids.dtype}")

        logits, past_k, past_v, prefill_ms = self._run_prefill(
            input_ids, attention_mask, position_ids)

        actual_len = int(attention_mask[0].sum())
        last_idx = max(actual_len - 1, 0)
        generated_ids = [int(np.argmax(logits[0, last_idx]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        generated_ids, decode_times, streamed_text = self._run_decode_loop(
            past_k, past_v, actual_len, generated_ids,
            max_new_tokens, stream, streamed_text)

        total_decode_ms = sum(decode_times)
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0
        min_decode_ms = min(decode_times) if decode_times else 0
        max_decode_ms = max(decode_times) if decode_times else 0
        total_ms = prefill_ms + total_decode_ms
        throughput = len(generated_ids) / (total_ms / 1000) if total_ms > 0 else 0

        perf = {
            "prefill_ms": prefill_ms,
            "total_decode_ms": total_decode_ms,
            "avg_decode_ms": avg_decode_ms,
            "min_decode_ms": min_decode_ms,
            "max_decode_ms": max_decode_ms,
            "total_ms": total_ms,
            "generated_tokens": len(generated_ids),
            "throughput_tok_s": throughput,
            "actual_input_len": actual_input_len,
            "gear_len": gear_len,
        }

        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result, perf


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Math-1.5B Inference with MindSpore Lite (prefill + decode)"
    )
    parser.add_argument(
        "--prefill-model", type=str, required=True, help="Path to prefill .mindir"
    )
    parser.add_argument(
        "--decode-model", type=str, required=True, help="Path to decode .mindir"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="./Qwen2.5-Math-1.5B", help="Tokenizer path"
    )
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--device", type=str, default="ascend", choices=["cpu", "ascend"]
    )
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    inferencer = Qwen215BInferencer(
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
    result, perf = inferencer.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )

    print("=" * 60)
    print("\n[Performance]")
    print(f"  Input Tokens:     {perf['actual_input_len']} (padded to {perf['gear_len']})")
    print(f"  Output Tokens:    {perf['generated_tokens']}")
    print(f"  Prefill:          {perf['prefill_ms']:.2f} ms")
    print(f"  Total Decode:     {perf['total_decode_ms']:.2f} ms")
    print(f"  Avg Decode:       {perf['avg_decode_ms']:.2f} ms/step")
    print(f"  Min Decode:       {perf['min_decode_ms']:.2f} ms/step")
    print(f"  Max Decode:       {perf['max_decode_ms']:.2f} ms/step")
    print(f"  Total Time:       {perf['total_ms']:.2f} ms")
    print(f"  Throughput:       {perf['throughput_tok_s']:.2f} tok/s")
    print("=" * 60)
    print("\n[Full Output Text]")
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
