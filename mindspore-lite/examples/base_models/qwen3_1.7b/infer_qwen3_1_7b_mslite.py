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
import json
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

# Map numpy dtype -> mslite.DataType
_NP_TO_MSLITE_DTYPE = {
    np.dtype(np.float32): mslite.DataType.FLOAT32,
    np.dtype(np.float16): mslite.DataType.FLOAT16,
    np.dtype(np.int32): mslite.DataType.INT32,
    np.dtype(np.int64): mslite.DataType.INT64,
}


def _np_dtype_to_mslite(dt):
    return _NP_TO_MSLITE_DTYPE.get(np.dtype(dt), mslite.DataType.FLOAT32)


def _compute_position_ids(attention_mask):
    """
    Compute position ids from attention mask.
    """
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


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
        self._dev = f"{device}:{device_id}" if device == "ascend" else "cpu"
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

        # Zero-copy buffers (lazily initialized after first decode predict)
        self._zc_inputs = None   # [input_ids, attn_mask, pos_ids, past_k, past_v] (device)
        self._zc_outputs = None  # [logits, present_k, present_v] (device)
        # KV dtype (detected from model, can be float16 or float32)
        self._kv_np_dtype = np.float32

    def _zc_setup(self, kv_shape, logits_shape, kv_np_dtype):
        """Create all device tensors for zero-copy decode. Call after 1st predict."""
        dev = self._dev
        self._kv_np_dtype = np.dtype(kv_np_dtype)
        # Small inputs — updated via set_data_from_numpy each step
        t_input_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=dev)
        t_attention_mask = mslite.Tensor(
            shape=[1, KV_CACHE_LEN], dtype=mslite.DataType.INT32, device=dev)
        t_position_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=dev)

        # KV cache ping-pong buffers (two pairs: in_K/V, out_K/V)
        kv_mslite_dtype = _np_dtype_to_mslite(self._kv_np_dtype)
        t_in_k = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_in_v = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_out_k = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_out_v = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)

        # Output buffer: logits on device (only D2H copy for argmax)
        dtype_logits = kv_mslite_dtype
        t_logits = mslite.Tensor(
            shape=list(logits_shape), dtype=dtype_logits, device=dev)

        # Input order: input_ids, attention_mask, position_ids, past_key_cache, past_value_cache
        self._zc_inputs = [t_input_ids, t_attention_mask, t_position_ids,
                           t_in_k, t_in_v]
        # Output order: logits, present_key_cache, present_value_cache
        self._zc_outputs = [t_logits, t_out_k, t_out_v]

    def _select_prefill_gear(self, seq_len: int) -> int:
        seq_len = int(seq_len)
        if seq_len <= PREFILL_GEAR_MIN:
            return PREFILL_GEAR_MIN
        if seq_len >= PREFILL_GEAR_MAX:
            return PREFILL_GEAR_MAX
        return int(((seq_len + PREFILL_GEAR_STEP - 1) // PREFILL_GEAR_STEP) * PREFILL_GEAR_STEP)

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
        """Print streaming delta between previous text and current decode."""
        new_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if prev_text and new_text.startswith(prev_text):
            delta = new_text[len(prev_text) :]
        else:
            n = min(len(prev_text), len(new_text))
            i = 0
            while i < n and prev_text[i] == new_text[i]:
                i += 1
            delta = new_text[i:]
        if delta:
            delta = delta.replace("\uFFFD", "")
        if delta:
            print(delta, end="", flush=True)
        return new_text

    def _run_prefill(self, input_ids, attention_mask, position_ids):
        """Run prefill inference with warmup. Returns (logits_np, kv_k_dev, kv_v_dev, elapsed_ms)."""
        dev = self._dev

        def _make_inputs():
            return [
                mslite.Tensor(input_ids),
                mslite.Tensor(attention_mask),
                mslite.Tensor(position_ids),
            ]

        # Warmup: trigger Ascend graph compilation
        warmup_out = self.prefill_model.predict(_make_inputs())

        # Pre-allocate output device tensors for KV cache (from warmup output shapes)
        kv_k_dev_out = mslite.Tensor(
            shape=list(warmup_out[1].shape),
            dtype=warmup_out[1].dtype,
            device=dev,
        )
        kv_v_dev_out = mslite.Tensor(
            shape=list(warmup_out[2].shape),
            dtype=warmup_out[2].dtype,
            device=dev,
        )
        # Logits output buffer (on device, but will be read to CPU for argmax)
        logits_dev_out = mslite.Tensor(
            shape=list(warmup_out[0].shape),
            dtype=warmup_out[0].dtype,
            device=dev,
        )

        # Timed inference with pre-allocated output buffers
        start = time.perf_counter()
        self.prefill_model.predict(_make_inputs(), outputs=[logits_dev_out, kv_k_dev_out, kv_v_dev_out])
        elapsed = (time.perf_counter() - start) * 1000.0

        logits_np = logits_dev_out.get_data_to_numpy()
        return logits_np, kv_k_dev_out, kv_v_dev_out, elapsed

    def _prime_decode(self, token_id, cur_attention_mask, valid_len, kv_k_dev, kv_v_dev):
        """Run one decode step to determine output shapes and dtype. KV inputs are device Tensors."""
        # Determine KV dtype from decode model
        decode_model_inputs = self.decode_model.get_inputs()
        kv_np_dtype = np.float32
        for t in decode_model_inputs:
            if getattr(t, "name", "") == "past_key_cache":
                kv_np_dtype = np.float16 if t.dtype == mslite.DataType.FLOAT16 else np.float32
                break

        input_ids_np = np.array([[token_id]], dtype=np.int32)
        position_ids_np = np.array([[valid_len]], dtype=np.int32)

        # Build input list directly: use device tensors for KV cache
        prime_inputs = []
        for t in decode_model_inputs:
            name = getattr(t, "name", "")
            if name == "input_ids":
                prime_inputs.append(mslite.Tensor(input_ids_np))
            elif name == "attention_mask":
                prime_inputs.append(mslite.Tensor(cur_attention_mask))
            elif name == "position_ids":
                prime_inputs.append(mslite.Tensor(position_ids_np))
            elif name == "past_key_cache":
                prime_inputs.append(kv_k_dev)
            elif name == "past_value_cache":
                prime_inputs.append(kv_v_dev)

        prime_out = self.decode_model.predict(prime_inputs)
        logits_shape = prime_out[0].shape
        kv_shape = list(prime_out[1].shape)
        return kv_shape, logits_shape, kv_np_dtype

    def _print_perf_summary(self, prefill_ms, decode_times):
        """Print performance summary."""
        total_decode_ms = sum(decode_times) if decode_times else 0.0
        avg_decode_ms = total_decode_ms / len(decode_times) if decode_times else 0.0
        total_ms = prefill_ms + total_decode_ms
        throughput = len(decode_times) / (total_decode_ms / 1000.0) if total_decode_ms > 0 else 0.0
        print(f"\n{'='*60}")
        print("Performance Summary (zero-copy + ping-pong):")
        print(f"  Device:                    {self._dev}")
        print(f"  Prefill (ms):              {prefill_ms:<12.2f}")
        print(f"  Total Decode (ms):         {total_decode_ms:<12.2f}")
        print(f"  Num decode steps:          {len(decode_times)}")
        print(f"  Avg decode step (ms):      {avg_decode_ms:<12.2f}")
        print(f"  Total (ms):                {total_ms:<12.2f}")
        print(f"  Throughput (tok/s):        {throughput:<12.2f}")
        print(f"{'='*60}")

    def _dump_calib(self, path, text, input_ids, attention_mask, position_ids, generated_ids):
        """Dump one calibration record to JSONL file."""
        record = {
            "ts": int(time.time()),
            "prompt": text,
            "prefill_input_ids": input_ids.astype(np.int64).tolist(),
            "prefill_attention_mask": attention_mask.astype(np.int64).tolist(),
            "prefill_position_ids": position_ids.astype(np.int64).tolist(),
            "generated_ids": [int(x) for x in generated_ids],
            "kv_cache_len": int(KV_CACHE_LEN),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 4096,
        stream: bool = True,
        dump_calib_path=None,
    ):
        """
        Generate text using Qwen3-1.7B.
        """
        input_ids, attention_mask, position_ids = self._prepare_inputs(text, max_length)

        # ── Prefill (zero-copy: KV cache stays on device) ──
        logits, kv_k_dev, kv_v_dev, prefill_ms = self._run_prefill(
            input_ids, attention_mask, position_ids
        )

        actual_len = int(attention_mask[0].sum())
        if int(list(kv_k_dev.shape)[3]) != KV_CACHE_LEN or int(list(kv_v_dev.shape)[3]) != KV_CACHE_LEN:
            raise RuntimeError(
                f"prefill cache len mismatch, expected {KV_CACHE_LEN}, got k={kv_k_dev.shape}, v={kv_v_dev.shape}"
            )

        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=np.int32)
        if actual_len > 0:
            cur_attention_mask[0, :actual_len] = 1
        last_idx = max(actual_len - 1, 0)
        generated_ids = [int(np.argmax(logits[0, last_idx]))]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)
        valid_len = int(actual_len)

        # ── Prime: one decode step to determine shapes and dtype ──
        kv_shape, logits_shape, kv_np_dtype = self._prime_decode(
            generated_ids[-1], cur_attention_mask, valid_len, kv_k_dev, kv_v_dev
        )

        # ── Create device tensors for zero-copy decode ──
        self._zc_setup(kv_shape, logits_shape, kv_np_dtype)
        # Swap prefill KV device tensors directly into decode inputs (no numpy copy)
        self._zc_inputs[3] = kv_k_dev
        self._zc_inputs[4] = kv_v_dev

        # ── Zero-copy decode loop with ping-pong KV cache ──
        decode_inputs = self._zc_inputs
        decode_outputs = self._zc_outputs
        decode_times = []

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                break
            if valid_len >= KV_CACHE_LEN:
                break

            cur_attention_mask[0, valid_len] = 1
            decode_inputs[0].set_data_from_numpy(
                np.array([[generated_ids[-1]]], dtype=np.int32))
            decode_inputs[1].set_data_from_numpy(cur_attention_mask)
            decode_inputs[2].set_data_from_numpy(
                np.array([[valid_len]], dtype=np.int32))

            decode_start = time.perf_counter()
            outputs = self.decode_model.predict(decode_inputs, outputs=decode_outputs)
            decode_step_ms = (time.perf_counter() - decode_start) * 1000.0
            decode_times.append(decode_step_ms)

            logits = outputs[0].get_data_to_numpy()
            decode_inputs[3], decode_outputs[1] = decode_outputs[1], decode_inputs[3]
            decode_inputs[4], decode_outputs[2] = decode_outputs[2], decode_inputs[4]

            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        if stream:
            print()

        self._print_perf_summary(prefill_ms, decode_times)

        if dump_calib_path:
            self._dump_calib(dump_calib_path, text, input_ids, attention_mask, position_ids, generated_ids)

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
        "--dump-calib",
        type=str,
        default="",
        help="Append one JSONL record for PTQ calibration (input_ids/mask/pos + generated_ids).",
    )
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
        dump_calib_path=(args.dump_calib or None),
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
