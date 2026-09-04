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
"""Qwen3-0.6B MindSpore Lite unified inference script.

Supports three modes:
  1. prefill_decode: Full conversation generation with KV cache (Scene A)
  2. prefill_only:   Prefill outputs [batch, 1, vocab] only (Scene B)
  3. common_prefix:  Prefix + suffix models for common-prefix caching (Scene C)
"""

import sys
import argparse
import time
import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError:
    print("Error: mindspore_lite or transformers package not found.")
    sys.exit(1)


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


_DTYPE_MAP = {
    np.dtype("int32"): mslite.DataType.INT32,
    np.dtype("float16"): mslite.DataType.FLOAT16,
    np.dtype("float32"): mslite.DataType.FLOAT32,
}


def _np_to_mslite_dtype(np_dtype):
    return _DTYPE_MAP[np.dtype(np_dtype)]


def _tokenize(text, tokenizer, max_length=2048, use_chat_template=True,
              enable_thinking=False, system_prompt=None):
    """Tokenize text into input_ids/attention_mask/position_ids.

    enable_thinking:
      - True  → 场景A（prefill+decode），保留 Qwen3 thinking 模式（输出含思考过程）
      - False → 场景B/C，关闭 thinking，直接输出选项/答案

    system_prompt:
      - 若提供，则在 chat template 中添加 system 消息，引导模型直接输出答案
      - 用于场景B/C，使模型输出选项字母（如 "A"）
    """
    if (
        use_chat_template
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        enc = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="np",
            enable_thinking=enable_thinking,
        )
        if hasattr(enc, "keys") and "input_ids" in enc:
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
        else:
            input_ids = enc
            attention_mask = np.ones_like(input_ids)
    else:
        enc = tokenizer(
            text, return_tensors="np", padding=False, truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
    input_ids = input_ids.astype(np.int32)
    attention_mask = attention_mask.astype(np.int32)
    position_ids = _compute_position_ids(attention_mask)
    return input_ids, attention_mask, position_ids


def _pad_to_bucket(input_ids, attention_mask, position_ids, bucket, pad_token_id):
    """Right-pad to target bucket length."""
    actual_len = int(input_ids.shape[1])
    pad_len = bucket - actual_len
    if pad_len > 0:
        input_ids = np.concatenate(
            [input_ids, np.full((1, pad_len), pad_token_id, dtype=np.int32)], axis=1
        )
        attention_mask = np.concatenate(
            [attention_mask, np.zeros((1, pad_len), dtype=np.int32)], axis=1
        )
        position_ids = np.concatenate(
            [position_ids, np.zeros((1, pad_len), dtype=np.int32)], axis=1
        )
    return input_ids, attention_mask, position_ids, actual_len, pad_len


def _next_bucket(seq_len, buckets):
    for b in buckets:
        if b >= seq_len:
            return b
    return buckets[-1]


# ---------------------------------------------------------------------------
# Prefill + Decode inferencer (Scene A)
# ---------------------------------------------------------------------------

class Qwen3PrefillDecodeInferencer:
    """Zero-copy prefill + autoregressive decode inferencer."""

    def __init__(
        self,
        prefill_model_path: str,
        decode_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
        decode_buckets=None,
        prefill_buckets=None,
    ):
        if device not in ["cpu", "ascend"]:
            raise ValueError("device must be cpu or ascend")
        if device != "ascend":
            raise ValueError("Zero-copy path requires device='ascend'")

        self._use_pre_alloc = None

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        self.context.ascend.device_id = device_id
        self.device_id = device_id
        self.device_str = f"ascend:{int(device_id)}"

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
        self.decode_buckets = (
            sorted(int(b) for b in decode_buckets) if decode_buckets else []
        )
        self.prefill_buckets = (
            sorted(int(b) for b in prefill_buckets) if prefill_buckets else []
        )

        self._bucket_io_cache = {}
        self._probe_pre_alloc_support()

    def _probe_pre_alloc_support(self):
        """Probe whether the model supports zero-copy pre-allocated output buffers."""
        if self._use_pre_alloc is not None:
            return
        if not self.decode_buckets:
            self._use_pre_alloc = False
            return

        smallest_bucket = self.decode_buckets[0]
        io = self._get_bucket_io(smallest_bucket)

        syn_input_ids = np.zeros((1, 1), dtype=np.int32)
        syn_attn = np.zeros((1, smallest_bucket + 1), dtype=np.int32)
        syn_attn[0, smallest_bucket] = 1
        syn_pos = np.array([[0]], dtype=np.int32)
        syn_past = np.zeros(
            (56, 1, 8, smallest_bucket, 128), dtype=np.float16
        )

        io["t_input_ids"].set_data_from_numpy(syn_input_ids)
        io["t_attention_mask"].set_data_from_numpy(syn_attn)
        io["t_position_ids"].set_data_from_numpy(syn_pos)
        io["t_past_in"].set_data_from_numpy(syn_past)

        inputs = [
            io["t_input_ids"],
            io["t_attention_mask"],
            io["t_position_ids"],
            io["t_past_in"],
        ]
        try:
            self.decode_model.predict(inputs, outputs=io["out_bufs"])
            self._use_pre_alloc = True
            print("[zero-copy] pre-allocated outputs: enabled (probe OK)")
        except (RuntimeError, ValueError) as e:
            self._use_pre_alloc = False
            print(
                f"[zero-copy] pre-allocated outputs disabled (probe failed: "
                f"{type(e).__name__}: {e!s:.120s}); using plain predict"
            )

    def _get_bucket_io(self, bucket_kv_len: int):
        """Get or create cached input/output tensors for a given decode bucket."""
        if bucket_kv_len in self._bucket_io_cache:
            return self._bucket_io_cache[bucket_kv_len]

        amask_len = bucket_kv_len + 1
        device_str = self.device_str

        t_input_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
        )
        t_attention_mask = mslite.Tensor(
            shape=[1, amask_len], dtype=mslite.DataType.INT32, device=device_str
        )
        t_position_ids = mslite.Tensor(
            shape=[1, 1], dtype=mslite.DataType.INT32, device=device_str
        )
        t_past_in = mslite.Tensor(
            shape=[56, 1, 8, bucket_kv_len, 128],
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )
        t_past_out = mslite.Tensor(
            shape=[56, 1, 8, bucket_kv_len + 1, 128],
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )

        outs_info = self.decode_model.get_outputs()
        logits_shape = [int(x) if int(x) > 0 else 1 for x in outs_info[0].shape]
        if len(logits_shape) == 3:
            logits_shape = [1, 1, logits_shape[2]]
        elif len(logits_shape) == 2:
            logits_shape = [1, logits_shape[1]]
        else:
            logits_shape = [1, 1, logits_shape[-1]]
        t_logits = mslite.Tensor(
            shape=logits_shape,
            dtype=mslite.DataType.FLOAT16,
            device=device_str,
        )

        io = {
            "t_input_ids": t_input_ids,
            "t_attention_mask": t_attention_mask,
            "t_position_ids": t_position_ids,
            "t_past_in": t_past_in,
            "t_past_out": t_past_out,
            "t_logits": t_logits,
            "out_bufs": [t_logits, t_past_out],
        }
        self._bucket_io_cache[bucket_kv_len] = io
        return io

    def _predict(self, inputs, out_bufs):
        if self._use_pre_alloc:
            return self.decode_model.predict(inputs, outputs=out_bufs)
        return self.decode_model.predict(inputs)

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 2048,
        use_chat_template: bool = True,
        enable_thinking: bool = True,
    ):
        """Run prefill + autoregressive decode and return the generated text.

        enable_thinking=True (default): 保留 Qwen3 thinking 模式，输出含思考过程，
            与原始 README 输出一致（如 "好的，用户问我的介绍..."）。
        """
        input_ids, attention_mask, position_ids = _tokenize(
            text, self.tokenizer, max_length, use_chat_template,
            enable_thinking=enable_thinking,
        )

        actual_seq_len = int(input_ids.shape[1])
        if self.prefill_buckets:
            target_seq_len = _next_bucket(actual_seq_len, self.prefill_buckets)
            if target_seq_len < actual_seq_len:
                raise ValueError(
                    f"prompt seq_len={actual_seq_len} exceeds max prefill "
                    f"bucket {self.prefill_buckets[-1]}"
                )
            input_ids, attention_mask, position_ids, _, pad_len = _pad_to_bucket(
                input_ids, attention_mask, position_ids,
                target_seq_len, int(self.tokenizer.pad_token_id),
            )
            print(f"[prefill] seq_len={actual_seq_len} -> bucket={target_seq_len} (pad {pad_len})")

        prefill_inputs = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ]
        print("Running LLM prefill...")
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict(prefill_inputs)
        logits = prefill_outputs[0].get_data_to_numpy()
        past_kv = prefill_outputs[1].get_data_to_numpy()
        prefill_ms = (time.time() - t0) * 1000
        print(f"Prefill time: {prefill_ms:.2f} ms")

        if actual_seq_len != int(input_ids.shape[1]):
            past_kv = past_kv[:, :, :, :actual_seq_len, :]

        generated_ids = []
        # logits is [batch, 1, vocab] (slice_last) or [batch, seq, vocab]
        next_token = int(np.argmax(logits[0, -1]))
        generated_ids.append(next_token)

        cur_pos = int(position_ids[0, actual_seq_len - 1])

        print("Running LLM decode...")
        decode_times = []
        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                break

            next_input_ids = np.array([[generated_ids[-1]]], dtype=np.int32)
            next_position_ids = np.array([[cur_pos + 1]], dtype=np.int32)

            if self.decode_buckets:
                cur_kv_len = past_kv.shape[3]
                target_len = _next_bucket(cur_kv_len, self.decode_buckets)
                if target_len > cur_kv_len:
                    pad = np.zeros(
                        (
                            past_kv.shape[0], past_kv.shape[1], past_kv.shape[2],
                            target_len - cur_kv_len, past_kv.shape[4],
                        ),
                        dtype=past_kv.dtype,
                    )
                    past_kv_in = np.concatenate([past_kv, pad], axis=3)
                else:
                    past_kv_in = past_kv
                attn_in = np.zeros((1, target_len + 1), dtype=np.int32)
                attn_in[0, :cur_kv_len] = 1
                attn_in[0, target_len] = 1
            else:
                target_len = past_kv.shape[3]
                past_kv_in = past_kv
                attn_in = np.ones((1, target_len + 1), dtype=np.int32)

            io = self._get_bucket_io(target_len)

            io["t_input_ids"].set_data_from_numpy(next_input_ids)
            io["t_attention_mask"].set_data_from_numpy(attn_in)
            io["t_position_ids"].set_data_from_numpy(next_position_ids)
            io["t_past_in"].set_data_from_numpy(past_kv_in)

            inputs = [
                io["t_input_ids"],
                io["t_attention_mask"],
                io["t_position_ids"],
                io["t_past_in"],
            ]
            t_step = time.time()
            decode_outputs = self._predict(inputs, io["out_bufs"])
            decode_times.append((time.time() - t_step) * 1000)

            logits = decode_outputs[0].get_data_to_numpy()
            new_past_kv = decode_outputs[1].get_data_to_numpy()

            if self.decode_buckets and new_past_kv.shape[3] != past_kv.shape[3] + 1:
                tail_idx = past_kv_in.shape[3]
                new_kv = new_past_kv[:, :, :, tail_idx : tail_idx + 1, :]
                past_kv = np.concatenate([past_kv, new_kv], axis=3)
            else:
                past_kv = new_past_kv
            cur_pos += 1

            generated_ids.append(int(np.argmax(logits[0, -1])))

        total_decode_ms = sum(decode_times)
        avg_decode_ms = (
            total_decode_ms / len(decode_times) if decode_times else 0.0
        )
        print(
            f"Total decode time: {total_decode_ms:.2f} ms, "
            f"avg decode step: {avg_decode_ms:.2f} ms, steps: {len(decode_times)}"
        )

        total_ms = prefill_ms + total_decode_ms
        throughput = (
            len(generated_ids) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        print(
            f"Total time: {total_ms:.2f} ms, throughput: {throughput:.2f} tok/s"
        )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prefill-only inferencer (slice_last & no_cache modes)
# ---------------------------------------------------------------------------

class Qwen3PrefillInferencer:
    """Prefill-only inferencer (Scene B: single token classification)."""

    def __init__(
        self,
        prefill_model_path: str,
        tokenizer_id: str,
        device: str = "ascend",
        device_id: int = 0,
        prefill_buckets=None,
    ):
        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        self.context.ascend.device_id = device_id

        print(f"Loading prefill model from {prefill_model_path}...")
        self.prefill_model = mslite.Model()
        self.prefill_model.build_from_file(
            prefill_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.prefill_buckets = (
            sorted(int(b) for b in prefill_buckets) if prefill_buckets else []
        )

    def generate_first_token(
        self, text: str, max_length: int = 2048, use_chat_template: bool = True,
        system_prompt: str = None,
    ):
        """Run prefill to generate the first token logits (Scene B)."""
        input_ids, attention_mask, position_ids = _tokenize(
            text, self.tokenizer, max_length, use_chat_template,
            enable_thinking=False,  # 场景B：关闭 thinking，直接输出选项
            system_prompt=system_prompt,
        )
        actual_seq_len = int(input_ids.shape[1])

        if self.prefill_buckets:
            target = _next_bucket(actual_seq_len, self.prefill_buckets)
            if target < actual_seq_len:
                raise ValueError(
                    f"prompt seq_len={actual_seq_len} exceeds max bucket "
                    f"{self.prefill_buckets[-1]}"
                )
            input_ids, attention_mask, position_ids, _, pad_len = (
                _pad_to_bucket(
                    input_ids, attention_mask, position_ids,
                    target, int(self.tokenizer.pad_token_id),
                )
            )
            print(f"[prefill] seq_len={actual_seq_len} -> bucket={target} (pad {pad_len})")

        prefill_inputs = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ]
        print("Running LLM prefill...")
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict(prefill_inputs)
        prefill_ms = (time.time() - t0) * 1000

        logits = prefill_outputs[0].get_data_to_numpy()
        next_token = int(np.argmax(logits[0, -1]))

        print(f"Prefill time: {prefill_ms:.2f} ms")
        print(f"Output logits shape: {logits.shape}")
        print(f"Predicted token id: {next_token}")
        decoded = self.tokenizer.decode([next_token], skip_special_tokens=False)
        print(f"Decoded token: {decoded!r}")

        return next_token, decoded, prefill_ms


# ---------------------------------------------------------------------------
# Common-prefix inferencer (prefix + suffix modes)
# ---------------------------------------------------------------------------

class Qwen3CommonPrefixInferencer:
    """Inferencer using prefix KV cache + suffix model."""

    def __init__(
        self,
        prefix_model_path: str,
        suffix_model_path: str,
        tokenizer_id: str,
        prefix_seq_len: int = 768,
        suffix_buckets: list = None,
        device: str = "ascend",
        device_id: int = 0,
    ):
        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        self.context.ascend.device_id = device_id
        self.device_str = f"ascend:{int(device_id)}"

        print(f"Loading prefix model from {prefix_model_path}...")
        self.prefix_model = mslite.Model()
        self.prefix_model.build_from_file(
            prefix_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading suffix model from {suffix_model_path}...")
        self.suffix_model = mslite.Model()
        self.suffix_model.build_from_file(
            suffix_model_path, mslite.ModelType.MINDIR, self.context
        )

        print(f"Loading tokenizer from {tokenizer_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prefix_seq_len = prefix_seq_len
        self.suffix_buckets = suffix_buckets or [128, 256, 384, 512, 640]
        self._prefix_kv = None
        self._prefix_kv_tensor = mslite.Tensor(
            shape=[56, 1, 8, int(prefix_seq_len), 128],
            dtype=mslite.DataType.FLOAT16,
            device=self.device_str,
        )

    def compute_prefix_cache(self, prefix_text: str):
        """Run prefix model once to compute KV cache."""
        enc = self.tokenizer(
            prefix_text, return_tensors="np", padding=False, truncation=True,
            max_length=self.prefix_seq_len,
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc.get("attention_mask", np.ones_like(input_ids)).astype(np.int32)
        position_ids = _compute_position_ids(attention_mask)

        actual_len = int(input_ids.shape[1])
        if actual_len > self.prefix_seq_len:
            raise ValueError(
                f"Prefix token length {actual_len} exceeds {self.prefix_seq_len}"
            )
        self._prefix_actual_len = actual_len
        pad_len = self.prefix_seq_len - actual_len
        if pad_len > 0:
            pad_token = int(self.tokenizer.pad_token_id)
            input_ids = np.concatenate(
                [input_ids, np.full((1, pad_len), pad_token, dtype=np.int32)], axis=1
            )
            attention_mask = np.concatenate(
                [attention_mask, np.zeros((1, pad_len), dtype=np.int32)], axis=1
            )
            position_ids = np.concatenate(
                [position_ids, np.zeros((1, pad_len), dtype=np.int32)], axis=1
            )

        print(f"[prefix] tokens={actual_len}, padded to {self.prefix_seq_len}")
        inputs = [mslite.Tensor(input_ids), mslite.Tensor(attention_mask),
                  mslite.Tensor(position_ids)]
        prefix_out_buf = self._prefix_kv_tensor
        print("Running prefix model...")
        t0 = time.time()
        try:
            outputs = self.prefix_model.predict(inputs, outputs=[prefix_out_buf])
        except (RuntimeError, ValueError):
            outputs = self.prefix_model.predict(inputs)
        prefix_ms = (time.time() - t0) * 1000

        prefix_kv = outputs[0]
        self._prefix_kv = prefix_kv
        if prefix_kv is not self._prefix_kv_tensor:
            self._prefix_kv_tensor = prefix_kv
        print(f"Prefix KV cache shape: {tuple(int(x) for x in prefix_kv.shape)}")
        print(f"Prefix model time: {prefix_ms:.2f} ms")
        return prefix_kv, prefix_ms

    def infer_suffix(self, suffix_text: str, use_chat_template: bool = True):
        """Run suffix model with prefix KV cache + user suffix tokens."""
        if self._prefix_kv is None:
            raise RuntimeError("Must call compute_prefix_cache() first")

        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            enc = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": suffix_text}],
                tokenize=True, add_generation_prompt=True, return_tensors="np",
                enable_thinking=False,
            )
            suffix_input_ids = enc["input_ids"] if hasattr(enc, "keys") and "input_ids" in enc else enc
        else:
            enc = self.tokenizer(suffix_text, return_tensors="np", padding=False)
            suffix_input_ids = enc["input_ids"]

        suffix_input_ids = suffix_input_ids.astype(np.int32)
        suffix_len = int(suffix_input_ids.shape[1])

        target_suffix_len = _next_bucket(suffix_len, self.suffix_buckets)
        pad_len = target_suffix_len - suffix_len
        if pad_len > 0:
            pad_token = int(self.tokenizer.pad_token_id)
            suffix_input_ids = np.concatenate(
                [np.full((1, pad_len), pad_token, dtype=np.int32), suffix_input_ids],
                axis=1,
            )

        print(f"[suffix] tokens={suffix_len}, padded to {target_suffix_len}")

        prefix_len = self.prefix_seq_len
        prefix_actual = getattr(self, "_prefix_actual_len", prefix_len)
        prefix_mask = np.concatenate([
            np.ones((1, prefix_actual), dtype=np.int32),
            np.zeros((1, prefix_len - prefix_actual), dtype=np.int32),
        ], axis=1)
        suffix_mask = np.ones((1, target_suffix_len), dtype=np.int32)
        if pad_len > 0:
            suffix_mask[:, :pad_len] = 0
        full_attention_mask = np.concatenate([prefix_mask, suffix_mask], axis=1)

        suffix_positions = np.arange(
            prefix_len, prefix_len + target_suffix_len, dtype=np.int32
        ).reshape(1, -1)
        if pad_len > 0:
            suffix_positions[:, :pad_len] = 0

        inputs = [
            mslite.Tensor(suffix_input_ids),
            mslite.Tensor(full_attention_mask),
            mslite.Tensor(suffix_positions),
            self._prefix_kv_tensor,
        ]

        print("Running suffix model...")
        t0 = time.time()
        outputs = self.suffix_model.predict(inputs)
        suffix_ms = (time.time() - t0) * 1000

        logits = outputs[0].get_data_to_numpy()
        token_id = int(np.argmax(logits[0, -1]))
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)

        print(f"Suffix model time: {suffix_ms:.2f} ms")
        print(f"Output logits shape: {logits.shape}")
        print(f"Predicted token id: {token_id}")
        print(f"Decoded token: {decoded!r}")

        return token_id, decoded, suffix_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Inference (supports prefill_decode, slice_last, no_cache, common_prefix)"
    )
    parser.add_argument(
        "--mode", type=str, default="prefill_only",
        choices=["prefill_decode", "prefill_only", "common_prefix"],
        help="Inference mode: prefill_decode (Scene A), prefill_only (Scene B, default), "
             "or common_prefix (Scene C)",
    )
    parser.add_argument("--prefill-model", type=str, default=None,
                        help="Prefill MindIR model path (for prefill_decode/slice_last/no_cache modes)")
    parser.add_argument("--decode-model", type=str, default=None,
                        help="Decode MindIR model path (for prefill_decode mode)")
    parser.add_argument("--prefix-model", type=str, default=None,
                        help="Prefix MindIR model path (for common_prefix mode)")
    parser.add_argument("--suffix-model", type=str, default=None,
                        help="Suffix MindIR model path (for common_prefix mode)")
    parser.add_argument("--tokenizer", type=str,
                        default="./Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下你自己。")
    parser.add_argument("--system-prompt", type=str,
                        default="You are a helpful assistant. Answer questions concisely.",
                        help="System prompt for slice_last/common_prefix modes (Scene B/C)")
    parser.add_argument("--prefix-text", type=str,
                        default="You are a helpful assistant. Answer questions concisely.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max new tokens to generate (prefill_decode mode)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--device", type=str, default="ascend")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefill-buckets", type=str, default="128,256,512,1024,480,640",
                        help="Prefill seq_len buckets (prefill_decode/prefill_only modes)")
    parser.add_argument("--decode-buckets", type=str,
                        default="16,32,64,96,128,192,256,384,512,768,1024,1536,2048",
                        help="Decode seq_len buckets (prefill_decode mode)")
    parser.add_argument("--prefix-seq-len", type=int, default=768,
                        help="Prefix model bucket size (common_prefix mode)")
    parser.add_argument("--suffix-buckets", type=str, default="128,256,384,512,640",
                        help="Suffix seq_len buckets (common_prefix mode)")
    parser.add_argument("--force-no-pre-alloc", action="store_true",
                        help="Skip pre-alloc probe; use plain predict (prefill_decode mode)")
    args = parser.parse_args()

    if args.mode == "prefill_decode":
        if not args.prefill_model or not args.decode_model:
            print("Error: --prefill-model and --decode-model are required for prefill_decode mode")
            sys.exit(1)
        prefill_buckets = [int(b) for b in args.prefill_buckets.split(",") if b]
        decode_buckets = [int(b) for b in args.decode_buckets.split(",") if b]
        inferencer = Qwen3PrefillDecodeInferencer(
            prefill_model_path=args.prefill_model,
            decode_model_path=args.decode_model,
            tokenizer_id=args.tokenizer,
            device=args.device,
            device_id=args.device_id,
            decode_buckets=decode_buckets,
            prefill_buckets=prefill_buckets,
        )
        if args.force_no_pre_alloc:
            inferencer._use_pre_alloc = False
            print("[zero-copy] pre-alloc disabled by --force-no-pre-alloc flag")

        print(f"\n{'=' * 60}")
        print("Mode: prefill_decode")
        print(f"Input Prompt: {args.prompt}")
        print(f"{'=' * 60}")
        result = inferencer.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            use_chat_template=not args.no_chat_template,
        )
        print(f"\n{'=' * 60}")
        print(f"Generated Response: {result}")
        print(f"{'=' * 60}")

    elif args.mode == "prefill_only":
        if not args.prefill_model:
            print(f"Error: --prefill-model is required for {args.mode} mode")
            sys.exit(1)
        prefill_buckets = [int(b) for b in args.prefill_buckets.split(",") if b]
        inferencer = Qwen3PrefillInferencer(
            prefill_model_path=args.prefill_model,
            tokenizer_id=args.tokenizer,
            device=args.device,
            device_id=args.device_id,
            prefill_buckets=prefill_buckets,
        )
        print(f"\n{'=' * 60}")
        print(f"Mode: {args.mode}")
        print(f"Input Prompt: {args.prompt}")
        print(f"{'=' * 60}")
        token_id, decoded, prefill_ms = inferencer.generate_first_token(
            args.prompt, max_length=args.max_length,
            use_chat_template=not args.no_chat_template,
            system_prompt=args.system_prompt,
        )
        print(f"\n{'=' * 60}")
        print(f"First token id:    {token_id}")
        print(f"Decoded token:     {decoded!r}")
        print(f"Prefill latency:   {prefill_ms:.2f} ms")
        print(f"{'=' * 60}")

    elif args.mode == "common_prefix":
        if not args.prefix_model or not args.suffix_model:
            print("Error: --prefix-model and --suffix-model are required for common_prefix mode")
            sys.exit(1)
        suffix_buckets = [int(b) for b in args.suffix_buckets.split(",") if b]
        inferencer = Qwen3CommonPrefixInferencer(
            prefix_model_path=args.prefix_model,
            suffix_model_path=args.suffix_model,
            tokenizer_id=args.tokenizer,
            prefix_seq_len=args.prefix_seq_len,
            suffix_buckets=suffix_buckets,
            device=args.device,
            device_id=args.device_id,
        )
        print(f"\n{'=' * 60}")
        print("Mode: common_prefix")
        print(f"Prefix text: {args.prefix_text}")
        print(f"{'=' * 60}")
        _, prefix_ms = inferencer.compute_prefix_cache(args.prefix_text)

        print(f"\n{'=' * 60}")
        print(f"User prompt: {args.prompt}")
        print(f"{'=' * 60}")
        token_id, decoded, suffix_ms = inferencer.infer_suffix(
            args.prompt, use_chat_template=not args.no_chat_template
        )

        total_ms = prefix_ms + suffix_ms
        print(f"\n{'=' * 60}")
        print(f"Prefix model time:  {prefix_ms:.2f} ms")
        print(f"Suffix model time:  {suffix_ms:.2f} ms")
        print(f"Total time:         {total_ms:.2f} ms")
        print(f"Predicted token id: {token_id}")
        print(f"Decoded token:      {decoded!r}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
