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
Infer Qwen3-8B on Ascend via MindSpore Lite MindIR (prefill + decode).
No PyTorch is used.

Auto-dispatches by the number of device IDs:
  * 1 device  -> single-chip zero-copy decode (KV ping-pong resident on device)
  * 2/4 devices -> tensor-parallel multi-process (HCCL, one worker per rank)

Usage:
  python infer_qwen3_8b_mslite.py --device-ids 2            # single-chip (1p)
  python infer_qwen3_8b_mslite.py --device-ids 2,3          # TP=2
  python infer_qwen3_8b_mslite.py --device-ids 2,3,4,5      # TP=4

Model paths auto-resolve from the device count (1p/2p/4p output dirs produced
by export_and_convert.sh); override with --prefill-ranks / --decode-ranks (TP)
or --prefill-model / --decode-model (1p).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np

# Set the HCCL NPU socket port range BEFORE importing mindspore_lite so GE/HCCL
# read it at the C-library init (the default port 16666 otherwise collides
# across same-card ranks in TP). Placed at module top so both the driver and
# spawn-forked workers inherit it before any GE init.
os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "21500-21600")

try:
    import mindspore_lite as mslite
    from transformers import AutoTokenizer
except ImportError as exc:
    print(f"Error: missing dependency ({exc}). pip install mindspore-lite transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model constants (Qwen3-8B)
# ---------------------------------------------------------------------------
KV_CACHE_LEN = 256
# Dynamic seq dims the prefill graph was compiled with (ge.dynamicDims).
PREFILL_SEQ_DIMS = (32, 64, 128)

# TP multi-process buffer constants (Qwen3-8B architecture).
NUM_LAYERS = 36
NUM_ATTN_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
VOCAB = 151936
PREFILL_SEQ = 64

# numpy dtype <-> mslite.DataType
_NP_TO_MSLITE_DTYPE = {
    np.dtype(np.float32): mslite.DataType.FLOAT32,
    np.dtype(np.float16): mslite.DataType.FLOAT16,
    np.dtype(np.int32): mslite.DataType.INT32,
    np.dtype(np.int64): mslite.DataType.INT64,
}
_MS_DTYPE_TO_NP = {
    mslite.DataType.FLOAT16: np.float16,
    mslite.DataType.FLOAT32: np.float32,
    mslite.DataType.FLOAT64: np.float64,
    mslite.DataType.INT32: np.int32,
    mslite.DataType.INT64: np.int64,
    mslite.DataType.BOOL: np.bool_,
}


def _mslite_tensor(np_array, target_dtype=None):
    """Convert numpy array to MindSpore Lite tensor, casting dtype if needed."""
    if target_dtype is not None:
        np_dtype = _MS_DTYPE_TO_NP.get(target_dtype)
        if np_dtype is not None and np_array.dtype != np_dtype:
            np_array = np_array.astype(np_dtype)
    return mslite.Tensor(np_array)


def _tp_tensor(np_array, dtype=None):
    """Wrap a contiguous numpy array as an mslite Tensor for TP workers."""
    if dtype is not None:
        npd = _MS_DTYPE_TO_NP.get(dtype)
        if npd is not None and np_array.dtype != npd:
            np_array = np_array.astype(npd)
    return mslite.Tensor(np.ascontiguousarray(np_array))


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
        f"feed keys={list(feed_dict.keys())}"
    )


def _pick_prefill_seq(real_len):
    """Smallest compiled prefill dim >= real_len."""
    for dim in PREFILL_SEQ_DIMS:
        if dim >= real_len:
            return dim
    raise ValueError(
        f"prompt length {real_len} exceeds max prefill dim "
        f"{PREFILL_SEQ_DIMS[-1]}; shorten the prompt or extend ge.dynamicDims."
    )


# ===========================================================================
# Path: single-chip (1p) — zero-copy decode
# ===========================================================================
class Qwen38BInferencer:
    """Qwen3-8B single-chip inferencer: prefill + decode both on Ascend MindIR (zero-copy decode)."""

    DECODE_INPUT_ORDER = [
        "input_ids", "attention_mask", "position_ids",
        "past_key_cache", "past_value_cache",
    ]

    def __init__(
        self,
        prefill_model_path,
        decode_model_path,
        tokenizer_path,
        prefill_device_id=0,
        decode_device_id=0,
    ):
        """Load tokenizer and the prefill/decode MindIR models onto Ascend."""
        if prefill_device_id != decode_device_id:
            print(
                "NOTE: prefill and decode are on different chips "
                f"(prefill dev {prefill_device_id}, decode dev {decode_device_id}). "
                "The zero-copy decode loop runs on the decode chip; the prefill->decode "
                "KV handoff crosses chips via a single D2H+H2D round-trip "
                "(same-chip handoff is a pure device-tensor swap)."
            )

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
        """Build a MindSpore Lite model from a MindIR file on a given Ascend device."""
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
        print(f"[zero-copy] decode device buffers on {dev}: "
              f"KV shape={kv_shape} dtype={kv_mslite_dtype}, "
              f"logits shape={logits_shape} dtype={logits_dtype}")

    def _stream_print_delta(self, generated_ids, prev_text):
        """Print incremental decoded text delta in stream mode."""
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

    def _prefill(self, input_ids, attention_mask, position_ids, real_len):
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
            print(f"[prefill] static output buffers on {pf_dev}: "
                  f"logits={list(warmup_out[0].shape)}, "
                  f"KV={list(warmup_out[1].shape)} dtype={warmup_out[1].dtype}")

        logits_dev, k_dev, v_dev = self._pf_out_dev
        t0 = time.perf_counter()
        self.prefill.predict(inputs, outputs=[logits_dev, k_dev, v_dev])
        prefill_ms = (time.perf_counter() - t0) * 1000

        logits_np = logits_dev.get_data_to_numpy()
        first_token = int(np.argmax(logits_np[0, real_len - 1, :]))
        _dbg = os.environ.get("QWEN_DUMP_PREFILL")
        if _dbg:
            np.save(f"{_dbg}_1p.npy", logits_np[0, real_len - 1, :].astype(np.float32))
            print(f"[1p] dumped prefill logits (argmax={first_token}) to {_dbg}_1p.npy", flush=True)
        return first_token, k_dev, v_dev, prefill_ms, warmup_ms

    def _decode_zerocopy(self, first_token, pf_k_dev, pf_v_dev, real_len, max_new_tokens, stream):
        """Zero-copy decode loop: device-resident KV ping-pong, only logits D2H."""
        if not self._zc_ready:
            self._zc_setup()

        kv_np = self._zc_kv_np_dtype
        t_hand = time.perf_counter()
        if self.prefill_device_id == self.decode_device_id:
            assert pf_k_dev.dtype == self._zc_inputs[3].dtype, (
                f"KV dtype mismatch on same-chip handoff: "
                f"prefill {pf_k_dev.dtype} vs decode {self._zc_inputs[3].dtype}")
            assert pf_v_dev.dtype == self._zc_inputs[4].dtype, (
                f"KV dtype mismatch on same-chip handoff: "
                f"prefill {pf_v_dev.dtype} vs decode {self._zc_inputs[4].dtype}")
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

        Returns (decoded_text, perf_dict). ``decode_only`` skips decoding and
        returns only the prefill perf (used by the prefill-only bench points).
        ``perf_dict`` carries ``decode_step_ms`` (per-step curve) for plotting
        decode latency vs KV-cache occupancy.
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
            padded_ids, attn_mask, position_ids, real_len)
        t_prefill = time.perf_counter() - t0

        if decode_only:
            perf = {
                "prefill_ms": t_prefill * 1000, "prefill_steady_ms": prefill_ms, "warmup_ms": warmup_ms,
                "handoff_ms": 0.0, "total_decode_ms": 0.0, "avg_decode_ms": 0.0,
                "total_ms": t_prefill * 1000, "num_decode_steps": 0,
                "decode_step_ms": [], "input_len": real_len, "output_len": 1, "first_token": int(first_token),
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
        """Print a single-chip performance summary."""
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


# ===========================================================================
# Path: tensor-parallel (2p/4p) — multi-process HCCL
# ===========================================================================
def _tokenize(tokenizer, text):
    """Apply chat template and return (1, seq) int64 input_ids."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True, add_generation_prompt=True, return_tensors="np")
    ids = np.asarray(enc["input_ids"] if hasattr(enc, "keys") else enc, dtype=np.int64)
    return ids.reshape(1, -1) if ids.ndim == 1 else ids


def _reconstruct_kv_tp(prefill_kv, valid_len, seq, tp_size):
    """Prefill emits decode-compatible (num_layers,1,kv_per,256,128) directly; padding
    garbage at positions [valid_len..seq] is harmless — decode attention_mask masks it.
    Passthrough."""
    del valid_len, seq, tp_size
    return prefill_kv


def _tp_unified_worker(prefill_path, decode_path, device_id, rank_id, config_file,
                       prompt_q, prefill_logits_q, step_q, decode_out_q, ready_q,
                       built_q, start_warmup_q, warmup=2, tp_size=2, decode_only=False):
    """Run one TP rank: build prefill+decode, warmup, then serve prefill+decode calls."""
    # Force-set the HCCL port range INSIDE the worker so GE/HCCL never falls back
    # to the default NPU adapter port (16666) — forked workers occasionally miss
    # the shell env, and 16666 collides with other TP sessions on the shared NPU.
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "21500-21600")
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = device_id
    ctx.ascend.rank_id = rank_id
    ctx.ascend.provider = "ge"
    # Qwen3-8B requires fp32 compute on the TP online-GE path too (same fp16
    # attention/KV overflow as 1p — without it, decode emits "Neo Neo Neo..."
    # garbage). Set via the Context precision_mode (mslite uses its own enum
    # names: enforce_fp32 == GE's force_fp32). NOT in the HCCL config_file:
    # putting [acl_init_options] there makes GE bind port 16666 eagerly at
    # Context creation, colliding with the other rank and aborting GE init.
    ctx.ascend.precision_mode = "enforce_fp32"

    kv_per = NUM_KV_HEADS // tp_size

    pf = None
    pf_in_dtypes = None
    if not decode_only:
        print(f"[rank{rank_id}] building prefill...", flush=True)
        pf = mslite.Model()
        pf.build_from_file(prefill_path, mslite.ModelType.MINDIR, ctx, config_file)
        pf_in_dtypes = [t.dtype for t in pf.get_inputs()]

    print(f"[rank{rank_id}] building decode...", flush=True)
    dc = mslite.Model()
    dc.build_from_file(decode_path, mslite.ModelType.MINDIR, ctx, config_file)
    dc_in_dtypes = [t.dtype for t in dc.get_inputs()]

    # Signal build (GE online compile + TBE kernel compile) done. The driver
    # staggers worker starts on this signal so no two ranks cold-compile the
    # same TBE custom-op kernels concurrently (which corrupts the TBE compile
    # cache and kills workers with "main process disappeared").
    built_q.put(1)

    # Barrier: wait until ALL ranks have finished building before starting
    # warmup. Warmup calls predict (HCCL AllReduce), which requires every rank
    # to participate simultaneously — with only the build stagger, early ranks
    # warm up while later ranks are still building → HCCL timeout
    # (Communication_Error_Get_Socket: "collective communication operator is
    # started too late or is not started by some NPUs"). The driver releases
    # this barrier once every rank's built_q has been collected.
    start_warmup_q.get()

    pf_warmup_out = [
        mslite.Tensor(np.zeros((1, PREFILL_SEQ, VOCAB), np.float32)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
    ]
    dc_warmup_out = [
        mslite.Tensor(np.zeros((1, 1, VOCAB), np.float32)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
    ]
    # TP=4 decode carries DEBUG_TAP (4 extra layer-0 intermediate outputs); allocate
    # buffers and ignore them (only outs[0..2] are used).
    if tp_size >= 4:
        num_heads_local = NUM_ATTN_HEADS // tp_size
        dc_warmup_out += [
            mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),  # tap_attn_out
            mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),  # tap_post_attn
            mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),  # tap_post_mlp
            mslite.Tensor(np.zeros((1, 1, num_heads_local * HEAD_DIM), np.float16)),  # tap_raw_out
        ]
    pf_dummy = [np.zeros((1, PREFILL_SEQ), np.int64),
                np.ones((1, PREFILL_SEQ), np.int64),
                np.arange(PREFILL_SEQ, dtype=np.int64).reshape(1, -1)]
    dc_dummy_kv = np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)
    dc_dummy = [np.array([[1]], np.int64), np.ones((1, KV_CACHE_LEN), np.int64),
                np.array([[5]], np.int64), dc_dummy_kv, dc_dummy_kv]

    for _ in range(warmup):
        if not decode_only:
            pf_feed = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate(pf_dummy)]
            pf.predict(pf_feed, pf_warmup_out)
        dc_feed = [_tp_tensor(a, dc_in_dtypes[i]) for i, a in enumerate(dc_dummy)]
        dc.predict(dc_feed, dc_warmup_out)
    print(f"[rank{rank_id}] warmup done", flush=True)
    ready_q.put(1)

    if decode_only:
        # Hybrid mode: driver sends (first_token, past_k, past_v) from a 1p prefill
        first_token, past_k_np, past_v_np = prompt_q.get()
        print(f"[rank{rank_id}] received 1p-prefill KV (first_token={first_token})", flush=True)
    else:
        prompt_data = prompt_q.get()
        pids, pam, ppos, real_len, seq = prompt_data
        pf_inputs = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate([pids, pam, ppos])]
        pf_outputs = [
            mslite.Tensor(np.zeros((1, seq, VOCAB), np.float32)),
            mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
            mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, KV_CACHE_LEN, HEAD_DIM), np.float16)),
        ]
        pf_outs = pf.predict(pf_inputs, pf_outputs)
        pf_logits = pf_outs[0].get_data_to_numpy()
        first_token = int(np.argmax(pf_logits[0, real_len - 1, :]))
        prefill_logits_q.put(first_token)
        past_k_np = _reconstruct_kv_tp(pf_outs[1].get_data_to_numpy(), real_len, seq, tp_size)
        past_v_np = _reconstruct_kv_tp(pf_outs[2].get_data_to_numpy(), real_len, seq, tp_size)

    dc_outputs = dc_warmup_out
    while True:
        step = step_q.get()
        if step is None:
            break
        dids, dam, dpos = step
        feed = [dids, dam, dpos, past_k_np, past_v_np]
        inputs = [_tp_tensor(a, dc_in_dtypes[i]) for i, a in enumerate(feed)]
        outs = dc.predict(inputs, dc_outputs)
        logits = outs[0].get_data_to_numpy()
        past_k_np = outs[1].get_data_to_numpy()
        past_v_np = outs[2].get_data_to_numpy()
        decode_out_q.put(logits)
    os._exit(0)


def _tp_prepare_prompt(tok, prompt):
    """Tokenize prompt → (pids, pam, ppos, real_len, seq)."""
    input_ids = _tokenize(tok, prompt)
    real_len = int(input_ids.shape[1])
    seq = PREFILL_SEQ
    if real_len > seq:
        input_ids = input_ids[:, -seq:]
        real_len = seq
    pids = np.zeros((1, seq), dtype=np.int64)
    pids[0, :real_len] = input_ids[0]
    pam = np.zeros((1, seq), np.int64)
    pam[0, :real_len] = 1
    cum = np.cumsum(pam[0], dtype=np.int64) - 1
    ppos = np.where(pam[0] > 0, cum, 0).astype(np.int64)[None, :]
    return pids, pam, ppos, real_len, seq


def _tp_spawn_unified_workers(prefill_ranks, decode_ranks, device_ids, config_file,
                              warmup, tp_size, use_hybrid):
    """Spawn workers with staggered builds (avoid TBE cache race + GE port 16666 race)."""
    prompt_qs = [Queue() for _ in range(tp_size)]
    pf_logits_qs = [Queue() for _ in range(tp_size)]
    step_qs = [Queue() for _ in range(tp_size)]
    out_qs = [Queue() for _ in range(tp_size)]
    ready_qs = [Queue() for _ in range(tp_size)]
    built_qs = [Queue() for _ in range(tp_size)]
    start_warmup_qs = [Queue() for _ in range(tp_size)]
    procs = []
    for r in range(tp_size):
        p = Process(target=_tp_unified_worker, args=(
            prefill_ranks[r], decode_ranks[r], device_ids[r], r, config_file,
            prompt_qs[r], pf_logits_qs[r], step_qs[r], out_qs[r], ready_qs[r],
            built_qs[r], start_warmup_qs[r], warmup, tp_size, use_hybrid))
        p.start()
        procs.append(p)
        # Stagger builds so ranks don't cold-compile the same TBE kernels
        # concurrently (race kills workers); the sleep lets rank r's GE port
        # binding settle before rank r+1 inits (default port 16666 races).
        if r + 1 < tp_size:
            print(f"Waiting for rank{r} build before starting rank{r + 1}...", flush=True)
            built_qs[r].get()
            time.sleep(8)
    built_qs[tp_size - 1].get()
    print(f"All {tp_size} ranks built. Releasing warmup barrier "
          "(synchronizes HCCL AllReduce start)...", flush=True)
    for r in range(tp_size):
        start_warmup_qs[r].put(1)
    return procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs


def _tp_hybrid_prefill(device_ids, pids, pam, ppos, real_len, kv_per, prompt_qs, tp_size):
    """1p prefill on device 0 for correct KV, then shard KV to all ranks."""
    print(f"[Hybrid: running 1p prefill on device {device_ids[0]} for correct KV...]")
    pf_1p_path = "./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0_graph.mindir"
    ctx1p = mslite.Context()
    ctx1p.target = ["ascend"]
    ctx1p.ascend.device_id = device_ids[0]
    pf1p = mslite.Model()
    pf1p.build_from_file(pf_1p_path, mslite.ModelType.MINDIR, ctx1p)
    pf1p_inputs = [_tp_tensor(a.astype(np.int32)) for a in [pids, pam, ppos]]
    pf1p_outs = pf1p.predict(pf1p_inputs)
    pf1p_logits = pf1p_outs[0].get_data_to_numpy()
    first_token = int(np.argmax(pf1p_logits[0, real_len - 1, :]))
    full_k = pf1p_outs[1].get_data_to_numpy()
    full_v = pf1p_outs[2].get_data_to_numpy()
    print(f"[Hybrid: 1p prefill done, first_token={first_token}, sharding KV for {tp_size} ranks...]")
    for r in range(tp_size):
        sk = full_k[:, :, r * kv_per:(r + 1) * kv_per, :, :].copy()
        sv = full_v[:, :, r * kv_per:(r + 1) * kv_per, :, :].copy()
        prompt_qs[r].put((first_token, sk, sv))
    return first_token


def _tp_decode_step(step_qs, out_qs, generated, valid_len, cur_am, tp_size):
    """Dispatch one decode step to all ranks; return (token, step_seconds)."""
    cur_am[0, valid_len] = 1
    step = (np.array([[generated[-1]]], np.int64), cur_am, np.array([[valid_len]], np.int64))
    for r in range(tp_size):
        step_qs[r].put(step)
    td0 = time.perf_counter()
    logits_r0 = out_qs[0].get()
    return int(np.argmax(logits_r0[0, -1, :])), time.perf_counter() - td0


def _tp_run_prefill(use_hybrid, device_ids, pids, pam, ppos, real_len, seq,
                    kv_per, prompt_qs, pf_logits_qs, tp_size):
    """Run prefill (hybrid 1p or normal TP); return (first_token, prefill_ms)."""
    if use_hybrid:
        first_token = _tp_hybrid_prefill(device_ids, pids, pam, ppos, real_len, kv_per,
                                         prompt_qs, tp_size)
        print(f"[Prefill (hybrid 1p): {real_len} tokens -> seq {seq}]")
        return first_token, 0.0
    print(f"[Prefill: {real_len} tokens -> seq {seq}]")
    t0 = time.perf_counter()  # precede put: worker may finish during the gap
    for r in range(tp_size):
        prompt_qs[r].put((pids, pam, ppos, real_len, seq))
    first_token = pf_logits_qs[0].get()
    return first_token, (time.perf_counter() - t0) * 1000


def _tp_stream_decode(tok, step_qs, out_qs, first_token, eos, max_new_tokens,
                      valid_len, stream, tp_size):
    """Run decode loop with optional streaming; return (generated, decode_times)."""
    generated = [first_token]
    streamed = tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False) if stream else ""
    if stream:
        print(streamed, end="", flush=True)
    cur_am = np.zeros((1, KV_CACHE_LEN), np.int64)
    cur_am[0, :valid_len] = 1
    decode_times = []
    for _ in range(max_new_tokens - 1):
        if eos is not None and generated[-1] == int(eos):
            break
        if valid_len >= KV_CACHE_LEN:
            break
        token, step_s = _tp_decode_step(step_qs, out_qs, generated, valid_len, cur_am, tp_size)
        decode_times.append(step_s)
        valid_len += 1
        generated.append(token)
        if stream:
            txt = tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            delta = txt[len(streamed):] if txt.startswith(streamed) else txt
            if delta:
                print(delta.replace("�", ""), end="", flush=True)
            streamed = txt
    if stream:
        print()
    return generated, decode_times


def _tp_finalize(procs, step_qs, tp_size):
    """Signal workers to exit and join."""
    for r in range(tp_size):
        step_qs[r].put(None)
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()


def run_tp_infer(prefill_ranks, decode_ranks, tokenizer_path, config_file,
                 prompt, max_new_tokens, device_ids, warmup=2, stream=True, tp_size=2,
                 use_hybrid=None):
    """Drive TP inference: spawn one worker per rank, feed prompt, stream decode output."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pids, pam, ppos, real_len, seq = _tp_prepare_prompt(tok, prompt)
    use_hybrid = (tp_size >= 4) if use_hybrid is None else use_hybrid
    kv_per = NUM_KV_HEADS // tp_size
    procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs = _tp_spawn_unified_workers(
        prefill_ranks, decode_ranks, device_ids, config_file, warmup, tp_size, use_hybrid)

    print("Waiting for warmup...", flush=True)
    for r in range(tp_size):
        ready_qs[r].get()
    print("All workers ready. Starting timed inference.", flush=True)

    first_token, prefill_ms = _tp_run_prefill(
        use_hybrid, device_ids, pids, pam, ppos, real_len, seq, kv_per,
        prompt_qs, pf_logits_qs, tp_size)
    generated, decode_times = _tp_stream_decode(
        tok, step_qs, out_qs, first_token, tok.eos_token_id, max_new_tokens,
        real_len, stream, tp_size)

    total_decode = sum(decode_times)
    avg_decode = total_decode / len(decode_times) if decode_times else 0.0
    perf = {"prefill_ms": prefill_ms, "total_decode_ms": total_decode * 1000,
            "avg_decode_ms": avg_decode * 1000, "input_len": real_len,
            "output_len": len(generated),
            "decode_step_ms": [t * 1000 for t in decode_times],
            "generated_ids": [int(x) for x in generated]}

    _tp_finalize(procs, step_qs, tp_size)
    return tok.decode(generated, skip_special_tokens=True), perf


def _print_tp_perf(perf, tp_size):
    """Print a TP performance summary."""
    print("=" * 60)
    print(f"--- Performance (TP={tp_size} prefill+decode) ---")
    print(f"  Input tokens:     {perf['input_len']}")
    print(f"  Output tokens:    {perf['output_len']}")
    print(f"  Prefill (ms):     {perf['prefill_ms']:.2f}")
    print(f"  Total Decode (ms): {perf['total_decode_ms']:.2f}")
    print(f"  Avg decode step:  {perf['avg_decode_ms']:.2f}")
    if perf["output_len"] > 1 and perf["total_decode_ms"] > 0:
        print(f"  Decode throughput: {(perf['output_len']-1)/(perf['total_decode_ms']/1000):.1f} tok/s")
    print("=" * 60)


# ===========================================================================
# Auto path resolution + HCCL config (so the script is callable directly)
# ===========================================================================
def _auto_model_dir(tp_size):
    """Pick the ONNX/MindIR output dir that export_and_convert.sh produced."""
    if tp_size == 1:
        return "./qwen3_8b_onnx"
    if tp_size == 4:
        return "./qwen3_8b_tp4_onnx"
    return "./qwen3_8b_tp_onnx"


def _rank_paths(model_dir, tp_size, sub):
    """Build the list of rank MindIR paths for a sub-graph (prefill/decode)."""
    return [f"{model_dir}/{sub}/qwen3_8b_llm_{sub}_rank{r}_graph.mindir" for r in range(tp_size)]


def _write_hccl_config(device_ids, run_dir):
    """Write rank_table.json + config_file.ini for the given device ids; return config path."""
    os.makedirs(run_dir, exist_ok=True)
    devs_json = ",".join(
        f'{{"device_id":"{d}","rank_id":"{i}"}}' for i, d in enumerate(device_ids))
    rank_table = (
        '{"version":"1.0","server_count":"1","server_list":['
        '{"server_id":"127.0.0.1","device":[' + devs_json + '],'
        '"host_nic_ip":"reserve"}],"status":"completed"}'
    )
    with open(os.path.join(run_dir, "rank_table.json"), "w", encoding="utf-8") as f:
        f.write(rank_table)
    config_path = os.path.join(run_dir, "config_file.ini")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[ascend_context]\n")
        f.write(f"rank_table_file={run_dir}/rank_table.json\n")
        f.write("plugin_custom_ops=All\n")
    return config_path


# ===========================================================================
# CLI + dispatch
# ===========================================================================
def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B inference (MindSpore Lite MindIR on Ascend) — 1p / 2p / 4p auto-dispatch")
    parser.add_argument("--device-ids", type=str, required=True,
                        help="comma-separated Ascend device ids (count decides parallelism: 1/2/4)")
    parser.add_argument("--model-id", type=str, default="./Qwen3-8B",
                        help="tokenizer / weights path")
    parser.add_argument("--prompt", type=str, default="你好，请用一句话介绍一下你自己")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3,
                        help="TP warm-up rounds (each = 1 prefill + 1 decode); 1p ignores this")
    # optional explicit model path overrides (otherwise auto-resolved from device count)
    parser.add_argument("--prefill-model", type=str, default=None,
                        help="1p override: path to prefill _graph.mindir")
    parser.add_argument("--decode-model", type=str, default=None,
                        help="1p override: path to decode _graph.mindir")
    parser.add_argument("--prefill-ranks", type=str, default=None,
                        help="TP override: comma-separated prefill rank MindIR paths")
    parser.add_argument("--decode-ranks", type=str, default=None,
                        help="TP override: comma-separated decode rank MindIR paths")
    parser.add_argument("--config-file", type=str, default=None,
                        help="TP override: HCCL config_file.ini (auto-generated if omitted)")
    # ---- bench mode (perf sweep with random inputs; bypasses tokenizer) ----
    parser.add_argument("--seq-len", type=int, default=None,
                        help="bench: random-input length (prefill seq); skips tokenizer")
    parser.add_argument("--decode-steps", type=int, default=None,
                        help="bench: number of decode steps (defaults to --max-new-tokens)")
    parser.add_argument("--prefill-only", action="store_true",
                        help="bench: time prefill only, skip decode")
    parser.add_argument("--json-out", type=str, default=None,
                        help="bench: write perf dict as JSON to this path")
    parser.add_argument("--seed", type=int, default=1234, help="bench: rng seed for random input")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="TP=4 bench: disable 1p-prefill hybrid so native 4p prefill is timed "
                             "(prefill result will be corrupt — perf use only)")
    parser.add_argument("--prefill-dims", type=str, default=None,
                        help="bench: override compiled prefill seq buckets (comma-sep), e.g. "
                             "'512,1024' for the kv1024 long-seq variant long-seq variant")
    return parser.parse_args()


def _run_single_chip(args, device_ids):
    """Run the single-chip (1p) zero-copy path."""
    model_dir = _auto_model_dir(1)
    prefill_model = args.prefill_model or _rank_paths(model_dir, 1, "prefill")[0]
    decode_model = args.decode_model or _rank_paths(model_dir, 1, "decode")[0]
    inferencer = Qwen38BInferencer(
        prefill_model_path=prefill_model, decode_model_path=decode_model,
        tokenizer_path=args.model_id, prefill_device_id=device_ids[0], decode_device_id=device_ids[0])

    if args.seq_len is not None:
        # ---- bench mode: random input of chosen length, perf only ----
        if args.prefill_dims:
            global PREFILL_SEQ_DIMS
            PREFILL_SEQ_DIMS = tuple(int(x) for x in args.prefill_dims.split(","))
        rng = np.random.default_rng(args.seed)
        # common token range (avoid special/eos tokens at the extremes)
        ids = rng.integers(100, 5000, size=(1, args.seq_len)).astype(np.int32)
        decode_steps = args.decode_steps if args.decode_steps is not None else args.max_new_tokens
        print(f"\n[BENCH 1p] seq_len={args.seq_len} decode_steps={decode_steps} "
              f"prefill_only={args.prefill_only}")
        _, perf = inferencer.generate_from_ids(
            ids, max_new_tokens=decode_steps, stream=False, decode_only=args.prefill_only)
        print("--- Performance ---")
        Qwen38BInferencer.print_perf(perf)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({"tp": 1, "seq_len": args.seq_len, **perf}, f, indent=2)
        return

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = inferencer.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    print("\n--- Performance ---")
    Qwen38BInferencer.print_perf(perf)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tp": 1, "prompt": args.prompt, **perf}, f, indent=2)


def _run_tensor_parallel(args, device_ids, tp_size):
    """Run the tensor-parallel (2p/4p) multi-process path."""
    model_dir = _auto_model_dir(tp_size)
    prefill_ranks = (args.prefill_ranks.split(",") if args.prefill_ranks
                     else _rank_paths(model_dir, tp_size, "prefill"))
    decode_ranks = (args.decode_ranks.split(",") if args.decode_ranks
                    else _rank_paths(model_dir, tp_size, "decode"))
    config_file = args.config_file or _write_hccl_config(device_ids, os.path.join(os.getcwd(), "tp_run"))

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = run_tp_infer(
        prefill_ranks, decode_ranks, args.model_id, config_file, args.prompt,
        args.max_new_tokens, device_ids, warmup=args.warmup, stream=not args.json_out, tp_size=tp_size,
        use_hybrid=False if args.no_hybrid else None)
    _print_tp_perf(perf, tp_size)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tp": tp_size, **perf}, f, indent=2)


def main():
    """Parse args and dispatch to single-chip or tensor-parallel path by device count."""
    # Use 'spawn' (not the default 'fork') so each TP worker is a fresh process
    # that reads HCCL_NPU_SOCKET_PORT_RANGE cleanly at GE init. With 'fork',
    # workers inherit the driver's imported-GE state and the port-range env var
    # is ignored → every rank tries the default NPU adapter port 16666 →
    # "Initialize GE failed ... port 16666 already bound" on the 2nd+ rank.
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    args = _parse_args()
    device_ids = [int(x) for x in args.device_ids.split(",")]
    tp_size = len(device_ids)

    print(f"=== TP_SIZE={tp_size}  devices={args.device_ids} ===")
    if tp_size == 1:
        _run_single_chip(args, device_ids)
    elif tp_size in (2, 4):
        _run_tensor_parallel(args, device_ids, tp_size)
    else:
        raise ValueError(f"unsupported device count {tp_size}: use 1, 2, or 4 device ids")


if __name__ == "__main__":
    main()
