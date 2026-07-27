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
"""Qwen2.5-7B inference via MindSpore Lite MindIR on Ascend — unified 1p / 2p / 4p.

One script supports three deployment modes, selected by the number of device IDs:

* ``--device-ids 0``           -> **1p** single-chip (zero-copy decode)
* ``--device-ids 0,1``         -> **2p** tensor-parallel on 2 chips (one card, HCCS)
* ``--device-ids 2,3,4,5``     -> **4p** tensor-parallel on 4 chips (two cards)

1p path
-------
Single process. Prefill + decode both run on one chip. The decode loop is
**zero-copy**: KV cache stays resident on the device via ping-pong buffers and
never round-trips through the host (only the small logits tensor is D2H'd per
step for argmax). The prefill graph emits decode-compatible KV directly, so the
prefill->decode handoff is a pure device-tensor swap on the same chip.

2p/4p path
----------
Megatron-style tensor parallelism with in-graph HCCL AllReduce (provider=ge).
One worker process per rank (+ a driver). Each rank builds its prefill-rank and
decode-rank MindIR, runs prefill, hands its per-rank KV slice to its decode, and
runs the decode loop; the driver orchestrates and takes logits from rank 0.
For 4p the decode is exported with extra layer-0 "tap" outputs to work around a
GE graph-optimization miscompile, so 7 output buffers are allocated (the extra 4
are ignored).

Launch with infer.sh (<device-ids>) which generates the rank_table/config for
the TP path and picks the right model directory.
"""

import argparse
import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError as exc:
    print(f"Error: missing dependency ({exc}). pip install mindspore-lite transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared dtype maps
# ---------------------------------------------------------------------------
_NP_TO_MSLITE_DTYPE = {
    np.dtype(np.float32): mslite.DataType.FLOAT32,
    np.dtype(np.float16): mslite.DataType.FLOAT16,
    np.dtype(np.int32): mslite.DataType.INT32,
    np.dtype(np.int64): mslite.DataType.INT64,
}
_MS_DTYPE_TO_NP = {
    mslite.DataType.FLOAT16: np.float16, mslite.DataType.FLOAT32: np.float32,
    mslite.DataType.FLOAT64: np.float64, mslite.DataType.INT32: np.int32,
    mslite.DataType.INT64: np.int64, mslite.DataType.BOOL: np.bool_,
}


# ===========================================================================
# 1P PATH — single-chip, zero-copy decode
# ===========================================================================

KV_CACHE_LEN = 256
# Dynamic seq dims the 1p prefill graph was compiled with (ge.dynamicDims).
PREFILL_SEQ_DIMS = (32, 64, 128)


def _np_dtype_to_mslite(dt):
    return _NP_TO_MSLITE_DTYPE.get(np.dtype(dt), mslite.DataType.FLOAT32)


def _mslite_tensor(np_array, target_dtype=None):
    """Convert numpy array to MindSpore Lite tensor, casting dtype if needed."""
    if target_dtype is not None:
        np_dtype = _MS_DTYPE_TO_NP.get(target_dtype)
        if np_dtype is not None and np_array.dtype != np_dtype:
            np_array = np_array.astype(np_dtype)
    return mslite.Tensor(np_array)


def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    """Build MindSpore Lite model inputs, auto-casting dtypes to match."""
    inputs = model.get_inputs()
    if not inputs:
        if preferred_order:
            return [_mslite_tensor(feed_dict[k]) for k in preferred_order]
        return [_mslite_tensor(v) for v in feed_dict.values()]
    ok_by_name = all(getattr(t, "name", None) in feed_dict for t in inputs)
    if ok_by_name:
        return [_mslite_tensor(feed_dict[t.name], t.dtype) for t in inputs]
    if preferred_order:
        dtype_map = {}
        for idx, key in enumerate(preferred_order):
            if idx < len(inputs):
                dtype_map[key] = inputs[idx].dtype
        return [_mslite_tensor(feed_dict[k], dtype_map.get(k)) for k in preferred_order]
    raise RuntimeError(
        f"input mismatch. model inputs={[getattr(x, 'name', '') for x in inputs]} "
        f"feed keys={list(feed_dict.keys())}")


def _pick_prefill_seq(real_len):
    """Smallest compiled prefill dim >= real_len."""
    for dim in PREFILL_SEQ_DIMS:
        if dim >= real_len:
            return dim
    raise ValueError(
        f"prompt length {real_len} exceeds max prefill dim "
        f"{PREFILL_SEQ_DIMS[-1]}; shorten the prompt or extend ge.dynamicDims.")


class Qwen257BInferencer:
    """Qwen2.5-7B single-chip inferencer: prefill + decode both on Ascend MindIR (zero-copy decode)."""

    DECODE_INPUT_ORDER = ["input_ids", "attention_mask", "position_ids",
                          "past_key_cache", "past_value_cache"]

    def __init__(self, prefill_model_path, decode_model_path, tokenizer_path,
                 prefill_device_id=0, decode_device_id=0):
        if prefill_device_id != decode_device_id:
            print(f"NOTE: prefill (dev {prefill_device_id}) and decode (dev {decode_device_id}) "
                  "on different chips; KV handoff crosses chips via one D2H+H2D round-trip.")
        self.prefill_device_id = prefill_device_id
        self.decode_device_id = decode_device_id
        self._dev = f"ascend:{decode_device_id}"

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.prefill = self._load_model(prefill_model_path, prefill_device_id, "prefill")
        self.decode = self._load_model(decode_model_path, decode_device_id, "decode")

        self._zc_inputs = None
        self._zc_outputs = None
        self._zc_kv_np_dtype = np.float32
        self._zc_ready = False
        self._prefill_compiled = False
        self._pf_out_dev = None

    @staticmethod
    def _load_model(path, device_id, tag):
        print(f"Loading {tag} MindIR from {path} (ascend device {device_id})...")
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = device_id
        model = mslite.Model()
        model.build_from_file(path, mslite.ModelType.MINDIR, ctx)
        print(f"{tag} model loaded.")
        return model

    def _zc_setup(self):
        """Create all device tensors for the zero-copy decode loop (decode device)."""
        dev = self._dev
        ins = self.decode.get_inputs()
        outs = self.decode.get_outputs()
        by_name = {getattr(t, "name", ""): t for t in ins}
        t_ids = by_name.get("input_ids") or ins[0]
        t_attn = by_name.get("attention_mask") or ins[1]
        t_pos = by_name.get("position_ids") or ins[2]
        t_pk = by_name.get("past_key_cache") or ins[3]
        t_pv = by_name.get("past_value_cache") or ins[4]
        assert t_pk.shape == t_pv.shape, f"KV input shape mismatch: {t_pk.shape} vs {t_pv.shape}"
        assert t_pk.dtype == t_pv.dtype, f"KV input dtype mismatch: {t_pk.dtype} vs {t_pv.dtype}"

        kv_shape = [int(d) for d in t_pk.shape]
        kv_mslite_dtype = t_pk.dtype
        self._zc_kv_np_dtype = _MS_DTYPE_TO_NP[kv_mslite_dtype]
        logits_shape = [int(d) for d in outs[0].shape]
        logits_dtype = outs[0].dtype

        t_input_ids = mslite.Tensor(shape=[int(d) for d in t_ids.shape], dtype=t_ids.dtype, device=dev)
        t_attention_mask = mslite.Tensor(shape=[int(d) for d in t_attn.shape], dtype=t_attn.dtype, device=dev)
        t_position_ids = mslite.Tensor(shape=[int(d) for d in t_pos.shape], dtype=t_pos.dtype, device=dev)
        # KV ping-pong: two distinct buffer pairs (MUST NOT alias in/out -- Scatter hazard)
        t_in_k = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_in_v = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_out_k = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_out_v = mslite.Tensor(shape=kv_shape, dtype=kv_mslite_dtype, device=dev)
        t_logits = mslite.Tensor(shape=logits_shape, dtype=logits_dtype, device=dev)

        self._zc_inputs = [t_input_ids, t_attention_mask, t_position_ids, t_in_k, t_in_v]
        self._zc_outputs = [t_logits, t_out_k, t_out_v]
        self._zc_ready = True
        print(f"[zero-copy] decode device buffers on {dev}: KV shape={kv_shape} "
              f"dtype={kv_mslite_dtype}, logits shape={logits_shape} dtype={logits_dtype}")

    def _stream_print_delta(self, generated_ids, prev_text):
        """Print only the newly-decoded delta (token-wise streaming output)."""
        new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
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

    def _prefill(self, input_ids, attention_mask, position_ids):
        """Run prefill; return (first_token, k_dev, v_dev, prefill_ms, warmup_ms)."""
        feed = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}
        inputs = _build_mslite_inputs(self.prefill, feed, preferred_order=list(feed.keys()))
        pf_dev = f"ascend:{self.prefill_device_id}"
        warmup_ms = 0.0
        if not self._prefill_compiled:
            t0 = time.perf_counter()
            warmup_out = self.prefill.predict(inputs)
            warmup_ms = (time.perf_counter() - t0) * 1000
            self._prefill_compiled = True
            self._pf_out_dev = [
                mslite.Tensor(shape=[int(d) for d in warmup_out[0].shape], dtype=warmup_out[0].dtype, device=pf_dev),
                mslite.Tensor(shape=[int(d) for d in warmup_out[1].shape], dtype=warmup_out[1].dtype, device=pf_dev),
                mslite.Tensor(shape=[int(d) for d in warmup_out[2].shape], dtype=warmup_out[2].dtype, device=pf_dev),
            ]
            print(f"[prefill] static output buffers on {pf_dev}: logits={list(warmup_out[0].shape)}, "
                  f"KV={list(warmup_out[1].shape)} dtype={warmup_out[1].dtype}")
        logits_dev, k_dev, v_dev = self._pf_out_dev
        t0 = time.perf_counter()
        self.prefill.predict(inputs, outputs=[logits_dev, k_dev, v_dev])
        prefill_ms = (time.perf_counter() - t0) * 1000
        logits_np = logits_dev.get_data_to_numpy()
        first_token = int(np.argmax(logits_np[0, -1, :]))
        return first_token, k_dev, v_dev, prefill_ms, warmup_ms

    def _decode_zerocopy(self, first_token, pf_k_dev, pf_v_dev, real_len, max_new_tokens, stream):
        """Zero-copy decode loop: device-resident KV ping-pong, only logits D2H."""
        if not self._zc_ready:
            self._zc_setup()
        kv_np = self._zc_kv_np_dtype
        t_hand = time.perf_counter()
        if self.prefill_device_id == self.decode_device_id:
            assert pf_k_dev.dtype == self._zc_inputs[3].dtype
            assert pf_v_dev.dtype == self._zc_inputs[4].dtype
            self._zc_inputs[3] = pf_k_dev
            self._zc_inputs[4] = pf_v_dev
        else:
            k_np = pf_k_dev.get_data_to_numpy().astype(kv_np, copy=False)
            v_np = pf_v_dev.get_data_to_numpy().astype(kv_np, copy=False)
            self._zc_inputs[3].set_data_from_numpy(k_np)
            self._zc_inputs[4].set_data_from_numpy(v_np)
        handoff_ms = (time.perf_counter() - t_hand) * 1000

        ids_np = _MS_DTYPE_TO_NP[self._zc_inputs[0].dtype]
        pos_np = _MS_DTYPE_TO_NP[self._zc_inputs[2].dtype]
        attn_np = _MS_DTYPE_TO_NP[self._zc_inputs[1].dtype]

        generated_ids = [first_token]
        streamed_text = ""
        if stream:
            streamed_text = self._stream_print_delta(generated_ids, streamed_text)

        valid_len = real_len
        cur_attention_mask = np.zeros((1, KV_CACHE_LEN), dtype=attn_np)
        cur_attention_mask[0, :valid_len] = 1

        decode_inputs = self._zc_inputs
        decode_outputs = self._zc_outputs
        decode_times = []

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and generated_ids[-1] == int(self.eos_token_id):
                break
            if valid_len >= KV_CACHE_LEN:
                break
            cur_attention_mask[0, valid_len] = 1
            decode_inputs[0].set_data_from_numpy(np.array([[generated_ids[-1]]], dtype=ids_np))
            decode_inputs[1].set_data_from_numpy(cur_attention_mask)
            decode_inputs[2].set_data_from_numpy(np.array([[valid_len]], dtype=pos_np))
            td0 = time.perf_counter()
            outputs = self.decode.predict(decode_inputs, outputs=decode_outputs)
            decode_times.append(time.perf_counter() - td0)
            logits = outputs[0].get_data_to_numpy()
            # ping-pong swap: this step's KV output becomes next step's KV input
            decode_inputs[3], decode_outputs[1] = decode_outputs[1], decode_inputs[3]
            decode_inputs[4], decode_outputs[2] = decode_outputs[2], decode_inputs[4]
            valid_len += 1
            generated_ids.append(int(np.argmax(logits[0, -1, :])))
            if stream:
                streamed_text = self._stream_print_delta(generated_ids, streamed_text)
        if stream:
            print()
        return generated_ids, decode_times, handoff_ms

    def generate(self, text, max_new_tokens=128, stream=True):
        """Generate text from a prompt string. Returns (decoded_text, perf_dict)."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_tensors="np")
        if hasattr(enc, "keys"):
            input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
        elif isinstance(enc, np.ndarray):
            input_ids = enc.astype(np.int32)
        else:
            input_ids = np.asarray(enc, dtype=np.int32)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        return self.generate_from_ids(input_ids, max_new_tokens, stream)

    def generate_from_ids(self, input_ids, max_new_tokens=128, stream=True, decode_only=False):
        """Run prefill + zero-copy decode from a (1, N) int32 token-id array.

        Returns (decoded_text, perf_dict). ``decode_only`` skips decoding (prefill bench).
        ``perf_dict`` carries ``decode_step_ms`` (per-step curve).
        """
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        real_len = int(input_ids.shape[1])
        seq = _pick_prefill_seq(real_len)
        if real_len > seq:
            input_ids = input_ids[:, -seq:]
            real_len = seq
        padded_ids = np.zeros((1, seq), dtype=np.int32)
        padded_ids[0, :real_len] = input_ids[0]
        attn_mask = np.zeros((1, seq), dtype=np.int32)
        attn_mask[0, :real_len] = 1
        cum = np.cumsum(attn_mask[0], dtype=np.int32) - 1
        position_ids = np.where(attn_mask[0] > 0, cum, 0).astype(np.int32)[None, :]

        print(f"[Prefill: {real_len} tokens -> seq dim {seq}]")
        t0 = time.perf_counter()
        first_token, pf_k_dev, pf_v_dev, prefill_ms, warmup_ms = self._prefill(
            padded_ids, attn_mask, position_ids)
        t_prefill = time.perf_counter() - t0

        if decode_only:
            perf = {
                "prefill_ms": t_prefill * 1000, "prefill_steady_ms": prefill_ms, "warmup_ms": warmup_ms,
                "handoff_ms": 0.0, "total_decode_ms": 0.0, "avg_decode_ms": 0.0,
                "total_ms": t_prefill * 1000, "num_decode_steps": 0,
                "decode_step_ms": [], "input_len": real_len, "output_len": 1,
                "first_token": int(first_token), "generated_ids": [int(first_token)],
            }
            return "", perf

        t0 = time.perf_counter()
        generated_ids, decode_times, handoff_ms = self._decode_zerocopy(
            first_token, pf_k_dev, pf_v_dev, real_len, max_new_tokens, stream)
        t_decode = time.perf_counter() - t0

        total_decode = sum(decode_times)
        avg_decode = total_decode / len(decode_times) if decode_times else 0.0
        perf = {
            "prefill_ms": t_prefill * 1000, "prefill_steady_ms": prefill_ms, "warmup_ms": warmup_ms,
            "handoff_ms": handoff_ms, "total_decode_ms": total_decode * 1000,
            "avg_decode_ms": avg_decode * 1000, "total_ms": (t_prefill + t_decode) * 1000,
            "num_decode_steps": len(decode_times), "input_len": real_len, "output_len": len(generated_ids),
            "decode_step_ms": [t * 1000 for t in decode_times],
            "generated_ids": [int(x) for x in generated_ids],
        }
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True), perf

    @staticmethod
    def print_perf(perf):
        """Print prefill/decode perf dict to stdout."""
        print("=" * 60)
        print(f"  Input tokens:        {perf['input_len']}")
        print(f"  Output tokens:       {perf['output_len']}")
        print(f"  Prefill warmup:      {perf['warmup_ms']:.2f} ms (one-time, untimed)")
        print(f"  Prefill (steady):    {perf['prefill_steady_ms']:.2f} ms")
        print(f"  Prefill (incl handoff):{perf['prefill_ms']:.2f} ms")
        print(f"  KV handoff:          {perf['handoff_ms']:.2f} ms")
        print(f"  Total Decode:        {perf['total_decode_ms']:.2f} ms")
        print(f"  Avg decode step:     {perf['avg_decode_ms']:.2f} ms")
        print(f"  Total time:          {perf['total_ms']:.2f} ms")
        if perf["output_len"] > 1 and perf["total_decode_ms"] > 0:
            throughput = (perf["output_len"] - 1) / (perf["total_decode_ms"] / 1000)
            print(f"  Decode throughput:   {throughput:.1f} tok/s")
        print("=" * 60)


def run_1p(prefill_model, decode_model, model_id, prompt, max_new_tokens, device_id,
           prefill_device_id=None, decode_device_id=None):
    """1p single-chip inference (zero-copy decode)."""
    pf_dev = prefill_device_id if prefill_device_id is not None else device_id
    dc_dev = decode_device_id if decode_device_id is not None else device_id
    inferencer = Qwen257BInferencer(prefill_model, decode_model, model_id, pf_dev, dc_dev)
    print("\n" + "=" * 60)
    print(f"Input Prompt: {prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(prompt, max_new_tokens=max_new_tokens)
    print("\n--- Performance (1p single-chip) ---")
    Qwen257BInferencer.print_perf(perf)


# ===========================================================================
# TP PATH (2p/4p) — tensor parallel, in-graph HCCL AllReduce
# ===========================================================================

NUM_LAYERS = 28
NUM_ATTN_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
VOCAB = 152064
PREFILL_SEQ = 64


def _tensor(np_array, dtype=None):
    if dtype is not None:
        npd = _MS_DTYPE_TO_NP.get(dtype)
        if npd is not None and np_array.dtype != npd:
            np_array = np_array.astype(npd)
    return mslite.Tensor(np.ascontiguousarray(np_array))


def _tokenize(tokenizer, text):
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, return_tensors="np")
    ids = np.asarray(enc["input_ids"] if hasattr(enc, "keys") else enc, dtype=np.int64)
    return ids.reshape(1, -1) if ids.ndim == 1 else ids


def _reconstruct_kv_tp(prefill_kv):
    """Prefill emits decode-compatible KV directly; no host-side reshape needed."""
    return prefill_kv


def _tp_worker(prefill_path, decode_path, device_id, rank_id, config_file,
               prompt_q, prefill_logits_q, step_q, decode_out_q, ready_q, warmup=2, tp_size=2):
    """One TP rank: build prefill+decode, warmup, prefill, decode loop. os._exit at end
    (HCCL communicator destroy is collective and would deadlock on normal return)."""
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = device_id
    ctx.ascend.rank_id = rank_id
    ctx.ascend.provider = "ge"
    # Match qwen3_8b: online-GE TP uses enforce_fp32 (== GE force_fp32) so 2p/4p
    # logits stay numerically close to the 1p offline path. Set via Context, NOT
    # the HCCL config_file (writing force_fp32 there triggers port-16666 clashes).
    ctx.ascend.precision_mode = "enforce_fp32"

    kv_per = NUM_KV_HEADS // tp_size

    print(f"[rank{rank_id}] building prefill...", flush=True)
    pf = mslite.Model()
    pf.build_from_file(prefill_path, mslite.ModelType.MINDIR, ctx, config_file)
    pf_in_dtypes = [t.dtype for t in pf.get_inputs()]

    print(f"[rank{rank_id}] building decode...", flush=True)
    dc = mslite.Model()
    dc.build_from_file(decode_path, mslite.ModelType.MINDIR, ctx, config_file)
    dc_in_dtypes = [t.dtype for t in dc.get_inputs()]

    # Host-side decode warmup buffers (KV cache shape is static, seq-independent).
    kv_shape_host = [NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM]
    dc_warmup_out = [
        mslite.Tensor(np.zeros((1, 1, VOCAB), np.float32)),
        mslite.Tensor(np.zeros(kv_shape_host, np.float16)),
        mslite.Tensor(np.zeros(kv_shape_host, np.float16)),
    ]
    # TP=4 decode is exported with barrier outputs (every 4 layers, stacked into one tensor)
    if tp_size >= 4:
        n_barriers = NUM_LAYERS // 4
        dc_warmup_out += [
            mslite.Tensor(np.zeros((n_barriers, 1, 1, 3584), np.float16)),
        ]
    dc_dummy_kv = np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)
    dc_dummy = [np.array([[1]], np.int64), np.ones((1, KV_CACHE_LEN), np.int64),
                np.array([[5]], np.int64), dc_dummy_kv, dc_dummy_kv]

    # ready BEFORE warmup: 2p/4p prefill is exported static-per-seq, so the warmup
    # dummy must use the actual prompt seq (received from driver). Decode warmup is
    # seq-independent (KV cache shape fixed).
    ready_q.put(1)

    pids, pam, ppos, _, seq = prompt_q.get()
    # Prefill logits shape: 2p slices to last token [1,1,VOCAB]; 4p emits full-seq.
    _pf_seq = 1 if tp_size < 4 else seq
    pf_warmup_out = [
        mslite.Tensor(np.zeros((1, _pf_seq, VOCAB), np.float32)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
    ]
    pf_dummy = [np.zeros((1, seq), np.int64), np.ones((1, seq), np.int64),
                np.arange(seq, dtype=np.int64).reshape(1, -1)]
    for _ in range(warmup):
        pf.predict([_tensor(a, pf_in_dtypes[i]) for i, a in enumerate(pf_dummy)], pf_warmup_out)
        dc.predict([_tensor(a, dc_in_dtypes[i]) for i, a in enumerate(dc_dummy)], dc_warmup_out)
    print(f"[rank{rank_id}] warmup done (seq={seq}, {warmup} rounds)", flush=True)

    pf_inputs = [_tensor(a, pf_in_dtypes[i]) for i, a in enumerate([pids, pam, ppos])]
    pf_outputs = [
        mslite.Tensor(np.zeros((1, _pf_seq, VOCAB), np.float32)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
    ]
    _pf0 = time.perf_counter()
    pf_outs = pf.predict(pf_inputs, pf_outputs)
    _pf1 = time.perf_counter()
    pf_logits = pf_outs[0].get_data_to_numpy()
    _pf2 = time.perf_counter()
    first_token = int(np.argmax(pf_logits[0, -1, :]))
    prefill_logits_q.put((first_token, (_pf1 - _pf0) * 1000))  # (token, steady predict ms)
    print(f"[rank{rank_id}] prefill-profile predict={(_pf1-_pf0)*1000:.2f}ms "
          f"D2H(logits+KV)={(_pf2-_pf1)*1000:.2f}ms", flush=True)
    # Decode loop (host-side KV round-trip; GE online path doesn't support device output buffers)
    past_k = _reconstruct_kv_tp(pf_outs[1].get_data_to_numpy())
    past_v = _reconstruct_kv_tp(pf_outs[2].get_data_to_numpy())

    dc_outputs = dc_warmup_out
    past_k_np, past_v_np = past_k, past_v
    # ---- profile: per-step decode timing breakdown (worker side) ----
    _t = {"h2d": 0.0, "predict": 0.0, "d2h_logits": 0.0, "d2h_kv": 0.0, "ipc_out": 0.0}
    _n = 0
    while True:
        step = step_q.get()
        if step is None:
            break
        dids, dam, dpos = step
        feed = [dids, dam, dpos, past_k_np, past_v_np]
        _t0 = time.perf_counter()
        inputs = [_tensor(a, dc_in_dtypes[i]) for i, a in enumerate(feed)]
        _t1 = time.perf_counter()
        outs = dc.predict(inputs, dc_outputs)
        _t2 = time.perf_counter()
        logits = outs[0].get_data_to_numpy()
        _t3 = time.perf_counter()
        past_k_np = outs[1].get_data_to_numpy()
        past_v_np = outs[2].get_data_to_numpy()
        _t4 = time.perf_counter()
        # IPC: only send the sampled token id (8B), not the full logits (0.6MB).
        decode_out_q.put(int(np.argmax(logits[0, -1, :])))
        _t5 = time.perf_counter()
        _t["h2d"] += _t1 - _t0
        _t["predict"] += _t2 - _t1
        _t["d2h_logits"] += _t3 - _t2
        _t["d2h_kv"] += _t4 - _t3
        _t["ipc_out"] += _t5 - _t4
        _n += 1
    if _n > 0:
        print(f"[rank{rank_id}] decode-profile n={_n} "
              f"predict={_t['predict']*1000/_n:.2f}ms "
              f"H2D(KV+tok)={_t['h2d']*1000/_n:.2f}ms "
              f"D2H(logits)={_t['d2h_logits']*1000/_n:.2f}ms "
              f"D2H(KV)={_t['d2h_kv']*1000/_n:.2f}ms "
              f"IPC_out={_t['ipc_out']*1000/_n:.2f}ms", flush=True)
    os._exit(0)


def _tp_prepare_input(tok, prompt, seq_len, seed):
    """Build (pids, pam, ppos, real_len, seq). seq_len set → random bench input."""
    if seq_len is not None:
        real_len = int(seq_len)
        seq = real_len
        rng = np.random.default_rng(seed)
        pids = rng.integers(100, 5000, size=(1, seq)).astype(np.int64)
        pam = np.ones((1, seq), dtype=np.int64)
        ppos = np.arange(seq, dtype=np.int64).reshape(1, -1)
    else:
        input_ids = _tokenize(tok, prompt)
        real_len = int(input_ids.shape[1])
        seq = PREFILL_SEQ
        if real_len > seq:
            input_ids = input_ids[:, -seq:]
            real_len = seq
        pids = np.zeros((1, seq), np.int64)
        pids[0, :real_len] = input_ids[0]
        pam = np.zeros((1, seq), np.int64)
        pam[0, :real_len] = 1
        cum = np.cumsum(pam[0], dtype=np.int64) - 1
        ppos = np.where(pam[0] > 0, cum, 0).astype(np.int64)[None, :]
    return pids, pam, ppos, real_len, seq


def _tp_spawn_workers(prefill_ranks, decode_ranks, device_ids, config_file,
                      warmup, tp_size):
    """Spawn one worker process per rank; return (procs, 5 queue lists)."""
    prompt_qs = [Queue() for _ in range(tp_size)]
    pf_logits_qs = [Queue() for _ in range(tp_size)]
    step_qs = [Queue() for _ in range(tp_size)]
    out_qs = [Queue() for _ in range(tp_size)]
    ready_qs = [Queue() for _ in range(tp_size)]
    procs = []
    for r in range(tp_size):
        p = Process(target=_tp_worker, args=(
            prefill_ranks[r], decode_ranks[r], device_ids[r], r, config_file,
            prompt_qs[r], pf_logits_qs[r], step_qs[r], out_qs[r], ready_qs[r],
            warmup, tp_size))
        p.start()
        procs.append(p)
    return procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs


def _tp_decode_loop(tok, step_qs, out_qs, generated, eos, max_new_tokens,
                    valid_len, stream, streamed, tp_size):
    """Run decode loop; return (generated, decode_times, streamed)."""
    decode_times = []
    cur_am = np.zeros((1, KV_CACHE_LEN), np.int64)
    cur_am[0, :valid_len] = 1
    for _ in range(max_new_tokens - 1):
        if eos is not None and generated[-1] == int(eos):
            break
        if valid_len >= KV_CACHE_LEN:
            break
        cur_am[0, valid_len] = 1
        step = (np.array([[generated[-1]]], np.int64), cur_am,
                np.array([[valid_len]], np.int64))
        for r in range(tp_size):
            step_qs[r].put(step)
        td0 = time.perf_counter()
        token_r0 = out_qs[0].get()  # worker sends only the argmax token id
        decode_times.append(time.perf_counter() - td0)
        valid_len += 1
        generated.append(token_r0)
        if stream:
            txt = tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            delta = txt[len(streamed):] if txt.startswith(streamed) else txt
            if delta:
                print(delta.replace("�", ""), end="", flush=True)
            streamed = txt
    return generated, decode_times, streamed


def _tp_print_perf(perf):
    print("=" * 60)
    print(f"  Input tokens:     {perf['input_len']}")
    print(f"  Output tokens:    {perf['output_len']}")
    print(f"  Prefill (ms):     {perf['prefill_ms']:.2f}")
    print(f"  Total Decode (ms): {perf['total_decode_ms']:.2f}")
    print(f"  Avg decode step:  {perf['avg_decode_ms']:.2f}")
    if perf["output_len"] > 1 and perf["total_decode_ms"] > 0:
        print(f"  Decode throughput: {(perf['output_len']-1)/(perf['total_decode_ms']/1000):.1f} tok/s")
    print("=" * 60)


def run_tp(prefill_ranks, decode_ranks, tokenizer_path, config_file, prompt,
           max_new_tokens, device_ids, warmup=2, stream=True, tp_size=2,
           seq_len=None, prefill_only=False, seed=1234):
    """2p/4p tensor-parallel inference (multi-process + HCCL).

    bench mode (seq_len is not None): random input of length seq_len, perf only.
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if prefill_only:
        max_new_tokens = 1  # prefill-only bench: decode loop range(0) skips

    pids, pam, ppos, real_len, seq = _tp_prepare_input(tok, prompt, seq_len, seed)
    procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs = _tp_spawn_workers(
        prefill_ranks, decode_ranks, device_ids, config_file, warmup, tp_size)

    print("Waiting for workers to build + warmup...", flush=True)
    for r in range(tp_size):
        ready_qs[r].get()
    print("All workers ready. Starting timed inference.", flush=True)

    print(f"[Prefill: {real_len} tokens -> seq {seq}]")
    # prefill timed by rank0's steady predict (excludes GE first-compile warmup).
    for r in range(tp_size):
        prompt_qs[r].put((pids, pam, ppos, real_len, seq))
    first_token, prefill_ms = pf_logits_qs[0].get()

    generated = [first_token]
    streamed = ""
    if stream:
        streamed = tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(streamed, end="", flush=True)

    generated, decode_times, streamed = _tp_decode_loop(
        tok, step_qs, out_qs, generated, tok.eos_token_id, max_new_tokens,
        real_len, stream, streamed, tp_size)
    if stream:
        print()

    total_decode = sum(decode_times)
    avg_decode = total_decode / len(decode_times) if decode_times else 0.0
    perf = {"prefill_ms": prefill_ms, "total_decode_ms": total_decode * 1000,
            "avg_decode_ms": avg_decode * 1000, "input_len": real_len, "output_len": len(generated),
            "decode_step_ms": [t * 1000 for t in decode_times],
            "generated_ids": [int(x) for x in generated]}

    for r in range(tp_size):
        step_qs[r].put(None)
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    _tp_print_perf(perf)
    return tok.decode(generated, skip_special_tokens=True), perf


# ===========================================================================
# main: dispatch on the number of device IDs
# ===========================================================================

def _build_arg_parser():
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Qwen2.5-7B inference (1p/2p/4p) via MindSpore Lite MindIR on Ascend")
    parser.add_argument("--device-ids", type=str, required=True,
                        help="Comma-separated Ascend device IDs. Count selects mode: "
                             "1 -> 1p single-chip; 2 -> TP=2; 4 -> TP=4")
    parser.add_argument("--prefill-model", type=str, default=None,
                        help="1p: path to single-shard prefill _graph.mindir")
    parser.add_argument("--decode-model", type=str, default=None,
                        help="1p: path to single-shard decode _graph.mindir")
    parser.add_argument("--prefill-device-id", type=int, default=None,
                        help="1p: override prefill device (split across two chips)")
    parser.add_argument("--decode-device-id", type=int, default=None,
                        help="1p: override decode device (split across two chips)")
    parser.add_argument("--prefill-ranks", type=str, default=None,
                        help="TP: comma-separated prefill-rank MindIR paths")
    parser.add_argument("--decode-ranks", type=str, default=None,
                        help="TP: comma-separated decode-rank MindIR paths")
    parser.add_argument("--config-file", type=str, default=None,
                        help="TP: ascend_context config (rank_table_file + plugin_custom_ops)")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="TP: tensor-parallel size (inferred from device count if omitted)")
    parser.add_argument("--model-id", type=str, default="./Qwen2.5-7B-Instruct",
                        help="Tokenizer / model path")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍一下你自己")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2,
                        help="TP warm-up rounds (each = 1 prefill + 1 decode)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="bench: random-input length (prefill seq); skips tokenizer")
    parser.add_argument("--decode-steps", type=int, default=None,
                        help="bench: number of decode steps (defaults to --max-new-tokens)")
    parser.add_argument("--prefill-only", action="store_true",
                        help="bench: time prefill only, skip decode")
    parser.add_argument("--json-out", type=str, default=None,
                        help="bench: write perf dict as JSON to this path")
    parser.add_argument("--seed", type=int, default=1234, help="bench: rng seed for random input")
    parser.add_argument("--prefill-dims", type=str, default=None,
                        help="bench: override compiled prefill seq buckets (comma-sep), e.g. '512,1024'")
    return parser


def _dump_perf_json(path, payload):
    """Write perf dict as JSON (utf-8) to path."""
    import json as _json
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=2)


def _run_1p_mode(args, device_ids, parser):
    """1p single-chip dispatch (bench or accuracy mode)."""
    if not args.prefill_model or not args.decode_model:
        parser.error("1p mode requires --prefill-model and --decode-model")
    pf_dev = args.prefill_device_id if args.prefill_device_id is not None else device_ids[0]
    dc_dev = args.decode_device_id if args.decode_device_id is not None else device_ids[0]
    inferencer = Qwen257BInferencer(args.prefill_model, args.decode_model, args.model_id, pf_dev, dc_dev)

    if args.seq_len is not None:
        if args.prefill_dims:
            global PREFILL_SEQ_DIMS  # noqa: PLW0603 - override compiled seq buckets at runtime
            PREFILL_SEQ_DIMS = tuple(int(x) for x in args.prefill_dims.split(","))
        rng = np.random.default_rng(args.seed)
        ids = rng.integers(100, 5000, size=(1, args.seq_len)).astype(np.int32)
        decode_steps = args.decode_steps if args.decode_steps is not None else args.max_new_tokens
        print(f"\n[BENCH 1p] seq_len={args.seq_len} decode_steps={decode_steps} "
              f"prefill_only={args.prefill_only}")
        _, perf = inferencer.generate_from_ids(
            ids, max_new_tokens=decode_steps, stream=False, decode_only=args.prefill_only)
        Qwen257BInferencer.print_perf(perf)
        if args.json_out:
            _dump_perf_json(args.json_out, {"tp": 1, "seq_len": args.seq_len, **perf})
        return

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n--- Performance (1p single-chip) ---")
    Qwen257BInferencer.print_perf(perf)
    if args.json_out:
        _dump_perf_json(args.json_out, {"tp": 1, "prompt": args.prompt, **perf})


def _run_tp_mode(args, device_ids, tp_size, parser):
    """2p/4p tensor-parallel dispatch."""
    if not args.prefill_ranks or not args.decode_ranks or not args.config_file:
        parser.error(f"TP={tp_size} mode requires --prefill-ranks, --decode-ranks and --config-file")
    prefill_ranks = args.prefill_ranks.split(",")
    decode_ranks = args.decode_ranks.split(",")
    if len(prefill_ranks) != tp_size or len(decode_ranks) != tp_size:
        parser.error(f"--prefill-ranks/--decode-ranks must have {tp_size} entries")
    decode_steps = args.decode_steps if args.decode_steps is not None else args.max_new_tokens
    if args.seq_len is not None:
        print(f"\n[BENCH TP={tp_size}] seq_len={args.seq_len} decode_steps={decode_steps} "
              f"prefill_only={args.prefill_only}")
    else:
        print("\n" + "=" * 60)
        print(f"Input Prompt: {args.prompt}")
        print("=" * 60)
        print("Generated Response: ", end="", flush=True)
    _, perf = run_tp(prefill_ranks, decode_ranks, args.model_id, args.config_file, args.prompt,
                     decode_steps, device_ids, args.warmup,
                     stream=(args.json_out is None and args.seq_len is None), tp_size=tp_size,
                     seq_len=args.seq_len, prefill_only=args.prefill_only, seed=args.seed)
    if args.json_out:
        _dump_perf_json(args.json_out, {"tp": tp_size, "seq_len": args.seq_len, **perf})


def main():
    """Entry point: parse args and dispatch on the number of device IDs."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    device_ids = [int(x) for x in args.device_ids.split(",")]
    tp_size = args.tp_size if args.tp_size is not None else len(device_ids)
    if tp_size == 1:
        _run_1p_mode(args, device_ids, parser)
    else:
        _run_tp_mode(args, device_ids, tp_size, parser)


if __name__ == "__main__":
    main()
