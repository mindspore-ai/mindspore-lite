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
  2. prefill_only:   Prefill outputs a single token_id (Scene B)
  3. common_prefix:  Prefix + suffix models for common-prefix caching (Scene C)

All models fuse "Slice hoisting + ArgMax" into the graph (always enabled at
export time): the lm_head MatMul only runs on the last real token's hidden
state (seq_len× less FLOPs), and the graph outputs token_id [batch, 1, 1] INT
(8 bytes D2H) instead of full logits [batch, 1, vocab] FP32 (~600 KB D2H).
This supports greedy decoding only.

Scene A decode uses zero-copy (Scatter in-place KV update): past_key_values is
a gear-sized padded buffer; the new K/V is Scatter-written in-place at
position_ids, so present has the SAME shape as past. The inference loop uses
ping-pong device buffers (two per gear), so KV never round-trips through the
host. Per-step host traffic is just input_ids(4B) + attention_mask(≤4KB) +
position_ids(4B) H2D + token_id(8B) D2H.
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


_MSLITE_TO_NP = {
    mslite.DataType.INT32: np.int32,
    mslite.DataType.INT64: np.int64,
    mslite.DataType.FLOAT16: np.float16,
    mslite.DataType.FLOAT32: np.float32,
}


def _np_dtype(dt):
    return _MSLITE_TO_NP.get(dt, np.float32)


def _shape_with_gear(meta_shape, gear):
    """Replace the dynamic dim (-1) of a [56,1,8,-1,128]-style shape with gear."""
    return [int(gear) if int(d) == -1 else int(d) for d in meta_shape]


def _compute_position_ids(attention_mask: np.ndarray) -> np.ndarray:
    position_ids = np.cumsum(attention_mask.astype(np.int32), axis=-1) - 1
    position_ids = np.where(attention_mask > 0, position_ids, 0)
    return position_ids.astype(np.int32)


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
# Prefill + Decode inferencer (Scene A) — zero-copy ping-pong
# ---------------------------------------------------------------------------

class Qwen3PrefillDecodeInferencer:
    """Prefill + zero-copy decode inferencer (Scene A).

    Prefill model outputs (token_id, present_key_values): ArgMax is fused
    in-graph, so the first token comes directly from the graph output.

    Decode model uses Scatter in-place KV update: past_key_values is a
    gear-sized padded buffer; the new K/V is written at position_ids inside
    the graph, so present_key_values has the SAME shape as past_key_values.
    The loop uses ping-pong device buffers (two per gear): each step's output
    buffer becomes the next step's input, so KV never round-trips through
    the host. ArgMax is fused in the decode graph too (token_id output).

    Per-step host traffic: input_ids(4B) + attention_mask(≤4KB) +
    position_ids(4B) H2D + token_id(8B) D2H.
    """

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
        if device != "ascend":
            raise ValueError("prefill_decode mode requires device='ascend'")
        if not decode_buckets:
            raise ValueError("decode_buckets is required (must match decode ini ge.dynamicDims)")

        print(f"Initializing MindSpore Lite context for {device}...")
        self.context = mslite.Context()
        self.context.target = [device]
        self.context.ascend.device_id = device_id
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
        self.decode_buckets = sorted(int(b) for b in decode_buckets)
        self.prefill_buckets = (
            sorted(int(b) for b in prefill_buckets) if prefill_buckets else []
        )
        self.max_total_len = self.decode_buckets[-1]

        # Static dtype/shape metadata from the built models.
        pf_out = self.prefill_model.get_outputs()
        self._pf_token_dtype = pf_out[0].dtype
        self._pf_kv_meta_shape = [int(d) for d in pf_out[1].shape]
        self._pf_kv_dtype = pf_out[1].dtype

        dc_in = self.decode_model.get_inputs()
        dc_out = self.decode_model.get_outputs()
        self._dc_ids_dtype = dc_in[0].dtype
        self._dc_mask_dtype = dc_in[1].dtype
        self._dc_pos_dtype = dc_in[2].dtype
        self._dc_kv_meta_shape = [int(d) for d in dc_in[3].shape]
        self._dc_kv_dtype = dc_in[3].dtype
        self._dc_token_dtype = dc_out[0].dtype

        # Device buffer caches: decode gear I/O sets (ping-pong KV pair).
        self._gear_io = {}

    def _gear_io_for(self, gear):
        """Device buffers for decode at this gear (ping-pong KV pair)."""
        if gear not in self._gear_io:
            kv_shape = _shape_with_gear(self._dc_kv_meta_shape, gear)
            self._gear_io[gear] = {
                "t_ids": mslite.Tensor(
                    shape=[1, 1], dtype=self._dc_ids_dtype, device=self.device_str
                ),
                "t_mask": mslite.Tensor(
                    shape=[1, gear], dtype=self._dc_mask_dtype, device=self.device_str
                ),
                "t_pos": mslite.Tensor(
                    shape=[1, 1], dtype=self._dc_pos_dtype, device=self.device_str
                ),
                "t_token": mslite.Tensor(
                    shape=[1, 1, 1], dtype=self._dc_token_dtype, device=self.device_str
                ),
                "t_kv_a": mslite.Tensor(
                    shape=kv_shape, dtype=self._dc_kv_dtype, device=self.device_str
                ),
                "t_kv_b": mslite.Tensor(
                    shape=kv_shape, dtype=self._dc_kv_dtype, device=self.device_str
                ),
            }
        return self._gear_io[gear]

    def _seed_kv_buffer(self, src_np, valid_len, gear):
        """One-time KV copy into a gear's device buffer: H2D of valid prefix."""
        io = self._gear_io_for(gear)
        if hasattr(src_np, "get_data_to_numpy"):
            src_np = src_np.get_data_to_numpy()
        src = src_np[:, :, :, :valid_len, :]
        kv_np = np.zeros(
            _shape_with_gear(self._dc_kv_meta_shape, gear),
            dtype=_np_dtype(self._dc_kv_dtype),
        )
        kv_np[:, :, :, :valid_len] = src
        io["t_kv_a"].set_data_from_numpy(kv_np)
        return io["t_kv_a"]

    def _prime_gear(self, gear, t_in_kv, ids_np, mask_np, pos_np):
        """One untimed plain decode predict at this gear (triggers model resize)."""
        prime_out = self.decode_model.predict(
            [
                mslite.Tensor(ids_np),
                mslite.Tensor(mask_np),
                mslite.Tensor(pos_np),
                t_in_kv,
            ]
        )
        got = [int(d) for d in prime_out[1].shape]
        if got[3] != gear:
            raise RuntimeError(
                f"decode prime at gear {gear}: present seq dim {got[3]} != {gear}"
            )
        return prime_out

    def generate(
        self,
        text: str,
        max_new_tokens: int = 128,
        max_length: int = 2048,
        use_chat_template: bool = True,
        enable_thinking: bool = True,
    ):
        """Run prefill + zero-copy decode and return the generated text.

        enable_thinking=True (default): 保留 Qwen3 thinking 模式，输出含思考过程，
            与原始 README 输出一致（如 "好的，用户问我的介绍..."）。
        """
        input_ids, attention_mask, position_ids = _tokenize(
            text, self.tokenizer, max_length, use_chat_template,
            enable_thinking=enable_thinking,
        )

        actual_seq_len = int(input_ids.shape[1])

        # Total sequence (prompt + generation) is capped by the max decode gear.
        budget = self.max_total_len - int(max_new_tokens)
        if actual_seq_len > budget:
            print(
                f"[truncate] prompt seq_len={actual_seq_len} > budget={budget} "
                f"(max gear {self.max_total_len} - max_new {max_new_tokens}); keeping tail"
            )
            input_ids = input_ids[:, -budget:]
            attention_mask = attention_mask[:, -budget:]
            position_ids = _compute_position_ids(attention_mask)
            actual_seq_len = budget

        # ---- Prefill ----
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

        print("Running LLM prefill...")
        pf_inputs = [
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ]
        # Warmup (untimed): trigger the graph resize/compile at this bucket.
        self.prefill_model.predict(pf_inputs)
        t0 = time.perf_counter()
        pf_out = self.prefill_model.predict(pf_inputs)
        prefill_ms = (time.perf_counter() - t0) * 1000.0
        # ArgMax fused in the prefill graph: output 0 is already token_id.
        generated_ids = [int(pf_out[0].get_data_to_numpy().flatten()[0])]
        valid_len = actual_seq_len
        pf_kv_np = pf_out[1].get_data_to_numpy()
        print(f"Prefill time: {prefill_ms:.2f} ms")

        # ---- Zero-copy decode loop ----
        print("Running LLM decode (zero-copy ping-pong KV on device)...")
        decode_times = []
        gear_step_ms = {}
        switch_ms = []
        gear = _next_bucket(valid_len + 1, self.decode_buckets)
        if gear < valid_len + 1:
            raise ValueError(
                f"sequence length {valid_len + 1} exceeds max decode "
                f"bucket {self.decode_buckets[-1]}"
            )
        print(f"[decode] start gear={gear}")

        # Prefill -> decode handoff: one H2D copy of the valid KV prefix.
        t0 = time.perf_counter()
        t_in_kv = self._seed_kv_buffer(pf_kv_np, valid_len, gear)
        switch_ms.append((time.perf_counter() - t0) * 1000.0)
        print(f"[decode] prefill->decode KV handoff: H2D {switch_ms[-1]:.2f} ms")

        io = self._gear_io_for(gear)
        t_out_kv = io["t_kv_b"]
        ids_np = _np_dtype(self._dc_ids_dtype)
        mask_np_dtype = _np_dtype(self._dc_mask_dtype)
        pos_np = _np_dtype(self._dc_pos_dtype)

        # Prime the decode model at the starting gear (untimed).
        prime_mask = np.zeros((1, gear), dtype=mask_np_dtype)
        prime_mask[0, : valid_len + 1] = 1
        self._prime_gear(
            gear, t_in_kv,
            np.array([[generated_ids[-1]]], dtype=ids_np),
            prime_mask, np.array([[valid_len]], dtype=pos_np),
        )

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(
                self.eos_token_id
            ):
                break
            if valid_len >= self.max_total_len:
                print(f"[decode] KV buffer full at max gear {self.max_total_len}; stop")
                break

            # Gear switch when the current buffer is full.
            if valid_len >= gear:
                next_gear = _next_bucket(valid_len + 1, self.decode_buckets)
                t0 = time.perf_counter()
                t_in_kv = self._seed_kv_buffer(t_in_kv, valid_len, next_gear)
                gear = next_gear
                io = self._gear_io_for(gear)
                t_out_kv = io["t_kv_b"]
                switch_mask = np.zeros((1, gear), dtype=mask_np_dtype)
                switch_mask[0, : valid_len + 1] = 1
                self._prime_gear(
                    gear, t_in_kv,
                    np.array([[generated_ids[-1]]], dtype=ids_np),
                    switch_mask, np.array([[valid_len]], dtype=pos_np),
                )
                switch_ms.append((time.perf_counter() - t0) * 1000.0)
                print(
                    f"[decode] gear switch -> {gear} "
                    f"(copy+prime {switch_ms[-1]:.2f} ms, valid_len={valid_len})"
                )

            # attention_mask: ones on [0, valid_len+1).
            attn_mask = np.zeros((1, gear), dtype=mask_np_dtype)
            attn_mask[0, : valid_len + 1] = 1

            t_step = time.perf_counter()
            io["t_ids"].set_data_from_numpy(
                np.array([[generated_ids[-1]]], dtype=ids_np)
            )
            io["t_mask"].set_data_from_numpy(attn_mask)
            io["t_pos"].set_data_from_numpy(np.array([[valid_len]], dtype=pos_np))
            decode_outputs = self.decode_model.predict(
                [io["t_ids"], io["t_mask"], io["t_pos"], t_in_kv],
                outputs=[io["t_token"], t_out_kv],
            )
            # ArgMax fused in the decode graph: 8-byte D2H per step.
            token = int(decode_outputs[0].get_data_to_numpy().flatten()[0])
            step_ms = (time.perf_counter() - t_step) * 1000.0
            decode_times.append(step_ms)
            gear_step_ms.setdefault(gear, []).append(step_ms)

            # Ping-pong swap: next step's input KV is this step's output buffer.
            t_in_kv, t_out_kv = t_out_kv, t_in_kv
            valid_len += 1
            generated_ids.append(token)

        # ---- Summary ----
        total_decode_ms = sum(decode_times)
        avg_decode_ms = (
            total_decode_ms / len(decode_times) if decode_times else 0.0
        )
        total_ms = prefill_ms + total_decode_ms
        throughput = (
            len(generated_ids) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        print(
            f"Total decode time: {total_decode_ms:.2f} ms, "
            f"avg decode step: {avg_decode_ms:.2f} ms, steps: {len(decode_times)}"
        )
        for g in sorted(gear_step_ms):
            times = gear_step_ms[g]
            print(
                f"  gear {g}: {len(times)} steps, "
                f"avg {sum(times) / len(times):.2f} ms"
            )
        if switch_ms:
            print(
                f"Gear-switch/handoff one-time costs (copy+prime): {len(switch_ms)} "
                f"(total {sum(switch_ms):.2f} ms, excluded from decode avg)"
            )
        print(
            f"Total time: {total_ms:.2f} ms, throughput: {throughput:.2f} tok/s"
        )

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prefill-only inferencer (Scene B)
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
        """Run prefill to generate the first token (Scene B)."""
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

        print("Running LLM prefill...")
        t0 = time.time()
        prefill_outputs = self.prefill_model.predict([
            mslite.Tensor(input_ids),
            mslite.Tensor(attention_mask),
            mslite.Tensor(position_ids),
        ])
        prefill_ms = (time.time() - t0) * 1000

        # ArgMax is fused into the graph: output 0 is already token_id [batch, 1, 1].
        token_out = prefill_outputs[0].get_data_to_numpy()
        next_token = int(token_out.flatten()[0])
        print(f"Prefill time: {prefill_ms:.2f} ms")
        print(f"Output token_id shape: {token_out.shape}")
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
        """Run prefix model once to compute KV cache.

        prefix_text is wrapped as a system message via apply_chat_template so
        that the prefix tokens follow the chat format (<|im_start|>system\\n...
        <|im_end|>\\n). The suffix model then appends <|im_start|>user\\n...
        <|im_end|>\\n<|im_start|>assistant\\n via apply_chat_template, producing
        a coherent chat sequence. Without this wrapping, the model sees raw
        text for prefix and chat-templated suffix, breaking the context and
        causing degenerate outputs (e.g., <|im_end|>).
        """
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None)
        ):
            messages = [{"role": "system", "content": prefix_text}]
            enc = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,  # no assistant header; suffix adds it
                return_tensors="np",
                enable_thinking=False,
            )
            if hasattr(enc, "keys") and "input_ids" in enc:
                input_ids = enc["input_ids"]
                attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
            else:
                input_ids = enc
                attention_mask = np.ones_like(input_ids)
        else:
            enc = self.tokenizer(
                prefix_text, return_tensors="np", padding=False, truncation=True,
                max_length=self.prefix_seq_len,
            )
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", np.ones_like(input_ids))
        input_ids = input_ids.astype(np.int32)
        attention_mask = attention_mask.astype(np.int32)
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

        # Natural single-sequence RoPE: real suffix tokens continue right after the
        # actual prefix length, not after the right-padded prefix_len (base-768).
        # The base-768 positions shift the rotary phase and, with the added-token
        # remap above, are what flips the reference Z into the deployment C.
        suffix_positions = np.zeros((1, target_suffix_len), dtype=np.int32)
        suffix_positions[:, pad_len:] = np.arange(
            prefix_actual, prefix_actual + suffix_len, dtype=np.int32
        ).reshape(1, -1)

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

        # ArgMax is fused into the graph: output 0 is already token_id [batch, 1, 1].
        out0 = outputs[0].get_data_to_numpy()
        token_id = int(out0.flatten()[0])
        print(f"Suffix model time: {suffix_ms:.2f} ms")
        print(f"Output token_id shape: {out0.shape}")
        print(f"Predicted token id: {token_id}")
        decoded = self.tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"Decoded token: {decoded!r}")

        return token_id, decoded, suffix_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B Inference (supports prefill_decode, prefill_only, common_prefix)"
    )
    parser.add_argument(
        "--mode", type=str, default="prefill_only",
        choices=["prefill_decode", "prefill_only", "common_prefix"],
        help="Inference mode: prefill_decode (Scene A), prefill_only (Scene B, default), "
             "or common_prefix (Scene C)",
    )
    parser.add_argument("--prefill-model", type=str, default=None,
                        help="Prefill MindIR model path (for prefill_decode/prefill_only modes)")
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
                        help="System prompt for prefill_only/common_prefix modes (Scene B/C)")
    parser.add_argument("--prefix-text", type=str,
                        default="You are a helpful assistant. Answer questions concisely.")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max new tokens to generate (prefill_decode mode)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--device", type=str, default="ascend")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--prefill-buckets", type=str, default="128,512,1024,2048",
                        help="Prefill seq_len buckets, must match prefill ini ge.dynamicDims")
    parser.add_argument("--decode-buckets", type=str, default="256,640,1152,2176",
                        help="Decode gear sizes (zero-copy KV cache buckets), must match "
                             "decode ini ge.dynamicDims")
    parser.add_argument("--prefix-seq-len", type=int, default=768,
                        help="Prefix model bucket size (common_prefix mode)")
    parser.add_argument("--suffix-buckets", type=str, default="128,256,384,512,640",
                        help="Suffix seq_len buckets (common_prefix mode)")
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
