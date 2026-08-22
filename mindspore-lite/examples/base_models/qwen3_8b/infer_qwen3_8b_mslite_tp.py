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

All paths use the GE provider, ModelGroup(SHARE_WEIGHT) for weight sharing
between prefill and decode, and zero-copy decode (KV ping-pong resident on
device) for minimum latency.

1p (single-chip) uses a multi-static-bucket GE flow instead of ge.dynamicDims:
  * 6 prefill seq buckets: {512, 1024, 1664, 2048, 2816, 3072}
  * 6 decode KV buckets:  {seq + 512} for every prefill bucket
Each bucket has its own GE compiled graph and its own ModelGroup(SHARE_WEIGHT)
so that switching buckets releases the previous compiled graph, avoiding TBE
subprocess OOM when all 12 graphs are built concurrently. Bucket configs
live under configs/ (1p), configs/tp2/ (TP=2), configs/tp4/ (TP=4).

Auto-dispatches by the number of device IDs:
  * 1 device  -> single-chip GE + per-bucket weight sharing + zero-copy decode
  * 2/4 devices -> tensor-parallel multi-process (HCCL, one worker per rank)
                 with GE provider and zero-copy KV handoff

Usage:
  python infer_qwen3_8b_mslite_tp.py --device-ids 0,1       # TP=2
  python infer_qwen3_8b_mslite_tp.py --device-ids 0,1,2,3   # TP=4
  # 1p: use infer_qwen3_8b_mslite_1p.py instead

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
KV_CACHE_LEN = 256  # TP default; 1p uses dynamic KV via per-bucket cfgs
# Prefill seq buckets (multi-static-bucket GE flow via 6 separate cfgs).
PREFILL_SEQ_DIMS = (512, 1024, 1664, 2048, 2816, 3072)
# Max output tokens per prefill bucket; decode KV = prefill_seq + MAX_OUTPUT_TOKENS.
MAX_OUTPUT_TOKENS = 512
# TP path uses a fixed prefill seq for warmup dummy inputs.
TP_PREFILL_SEQ = 64

# TP multi-process buffer constants (Qwen3-8B architecture).
NUM_LAYERS = 36
NUM_ATTN_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 4096
VOCAB = 151936

# numpy dtype <-> mslite.DataType
_NP_TO_MSLITE_DTYPE = {
    np.dtype(np.float32): mslite.DataType.FLOAT32,
    np.dtype(np.float16): mslite.DataType.FLOAT16,
    np.dtype(np.int32): mslite.DataType.INT32,
    np.dtype(np.int64): mslite.DataType.INT64,
}
MS_DTYPE_TO_NP = {
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
        np_dtype = MS_DTYPE_TO_NP.get(target_dtype)
        if np_dtype is not None and np_array.dtype != np_dtype:
            np_array = np_array.astype(np_dtype)
    return mslite.Tensor(np_array)


def _tp_tensor(np_array, dtype=None):
    """Wrap a contiguous numpy array as an mslite Tensor for TP workers."""
    if dtype is not None:
        npd = MS_DTYPE_TO_NP.get(dtype)
        if npd is not None and np_array.dtype != npd:
            np_array = np_array.astype(npd)
    return mslite.Tensor(np.ascontiguousarray(np_array))


def build_mslite_inputs(model, feed_dict, preferred_order=None):
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
        f"{PREFILL_SEQ_DIMS[-1]}; shorten the prompt or extend PREFILL_SEQ_DIMS.")


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
    """Pad prefill KV to kv_len = seq + 512 (or slice already-padded graph output).

    Prefill now pads KV inside the graph to kv_len = seq + MAX_OUTPUT_TOKENS;
    this helper is kept for TP decode paths where a shard-by-KV-head copy is
    needed. If the physical KV length already >= kv_len, it just slices.
    """
    del valid_len, tp_size
    kv_len = seq + MAX_OUTPUT_TOKENS
    shape = list(prefill_kv.shape)
    phys_len = shape[3]
    if phys_len >= kv_len:
        return prefill_kv[:, :, :, :kv_len, :].copy()
    padded = np.zeros(shape[:3] + [kv_len] + [shape[4]], dtype=prefill_kv.dtype)
    padded[:, :, :, :seq, :] = prefill_kv[:, :, :, :seq, :]
    return padded


class _DecodeZeroCopyIO:
    """Zero-copy I/O for the decode loop: KV cache stays on device with ping-pong reuse.

    Corresponds to references/zero_copy_inference.md:
      * All inputs/outputs are pre-allocated as device tensors (device=\"ascend:<id>\");
      * predict(inputs, outputs=...) uses pre-allocated output buffers to avoid internal reallocation;
      * Large KV tensors (past_k/past_v) alternate between two buffers: this step's output
        is directly the next step's input, with no Host<->Device copy;
      * Only the small logits tensor is copied back for argmax (on the driver side).

    Usage:
      io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size)
      io.load_kv(past_k_np, past_v_np)
      for step in ...: io.predict_step(dc, dids, dam, dpos)
    """

    def __init__(self, device_id, kv_per, kv_len, dc_in_dtypes, tp_size):
        device_str = f"ascend:{int(device_id)}"
        kv_dtype = dc_in_dtypes[3]
        kv_shape = [NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM]
        # numpy dtype mapping (set_data_from_numpy requires an exact dtype match)
        self._np_ids = MS_DTYPE_TO_NP[dc_in_dtypes[0]]
        self._np_attn = MS_DTYPE_TO_NP[dc_in_dtypes[1]]
        self._np_pos = MS_DTYPE_TO_NP[dc_in_dtypes[2]]
        self._np_kv = MS_DTYPE_TO_NP[kv_dtype]
        # Small-input device tensors (updated via set_data_from_numpy each step)
        self.t_ids = mslite.Tensor(shape=[1, 1], dtype=dc_in_dtypes[0], device=device_str)
        self.t_attn = mslite.Tensor(shape=[1, kv_len], dtype=dc_in_dtypes[1], device=device_str)
        self.t_pos = mslite.Tensor(shape=[1, 1], dtype=dc_in_dtypes[2], device=device_str)
        # KV cache dual buffers (ping-pong): A/B alternate as input/output
        self.t_k_a = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
        self.t_v_a = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
        self.t_k_b = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
        self.t_v_b = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
        # logits output buffer (small tensor, copied back for argmax)
        self.t_logits = mslite.Tensor(
            shape=[1, 1, VOCAB], dtype=mslite.DataType.FLOAT32, device=device_str)
        # DEBUG_TAP outputs for TP>=4 (host buffers, not part of ping-pong)
        self.extra_out = []
        if tp_size >= 4:
            num_heads_local = NUM_ATTN_HEADS // tp_size
            self.extra_out = [
                mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),
                mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),
                mslite.Tensor(np.zeros((1, 1, HIDDEN_SIZE), np.float16)),
                mslite.Tensor(np.zeros((1, 1, num_heads_local * HEAD_DIM), np.float16)),
            ]
        # Current input/output buffer pointers (first round: in=A, out=B)
        self._k_in, self._k_out = self.t_k_a, self.t_k_b
        self._v_in, self._v_out = self.t_v_a, self.t_v_b

    def load_kv(self, past_k_np, past_v_np):
        """Write the prefill KV outputs into device buffers for the first time (one-time H2D)."""
        self._k_in.set_data_from_numpy(
            np.ascontiguousarray(past_k_np.astype(self._np_kv)))
        self._v_in.set_data_from_numpy(
            np.ascontiguousarray(past_v_np.astype(self._np_kv)))

    def predict_step(self, dc, dids, dam, dpos):
        """Run one decode step: only small inputs are updated; large KV tensors ping-pong on device."""
        self.t_ids.set_data_from_numpy(
            np.ascontiguousarray(np.asarray(dids).astype(self._np_ids)))
        self.t_attn.set_data_from_numpy(
            np.ascontiguousarray(np.asarray(dam).astype(self._np_attn)))
        self.t_pos.set_data_from_numpy(
            np.ascontiguousarray(np.asarray(dpos).astype(self._np_pos)))
        inputs = [self.t_ids, self.t_attn, self.t_pos, self._k_in, self._v_in]
        outputs = [self.t_logits, self._k_out, self._v_out] + self.extra_out
        outs = dc.predict(inputs, outputs=outputs)
        # ping-pong: this step's output buffer becomes the next step's input
        self._k_in, self._k_out = self._k_out, self._k_in
        self._v_in, self._v_out = self._v_out, self._v_in
        return outs


def _tp_unified_worker(prefill_path, decode_path, device_id, rank_id,
                       pf_config_file, dc_config_file, bucket_seq, kv_len,
                       prompt_q, prefill_logits_q, step_q, decode_out_q, ready_q,
                       built_q, start_warmup_q, warmup=2, tp_size=2, decode_only=False,
                       perf_sweep=False, bucket_q=None):
    """Run one TP rank: build prefill+decode, warmup, then serve prefill+decode calls.

    2p bucketed dynamicDims flow:
      * pf_config_file / dc_config_file each carry their sub-graph's own
        [ge_graph_options] (ge.inputShape + ge.dynamicDims + ge.dynamicNodeType=1),
        so a single online-GE build serves all 6 seq/KV buckets. One bucket is
        exercised per process lifetime (chosen by the driver from prompt length),
        so warmup pins that bucket's compiled shape.
      * bucket_seq = padded prefill seq for this run; kv_len = bucket_seq + 512
        = decode KV length (= prefill KV output length; NOT the max 3584).
    4p/hybrid keeps the fixed-shape path: bucket_seq=TP_PREFILL_SEQ, kv_len=KV_CACHE_LEN,
    both config files = the plain HCCL config (no ge_graph_options).

    perf_sweep mode: workers stay alive across all 6 buckets. bucket_q receives
    (bucket_seq, kv_len) per bucket or None to exit. After each bucket's decode,
    step_q receives None to end the decode loop, then the worker loops back to
    bucket_q for the next bucket. Output tensors are re-created per bucket since
    kv_len differs across buckets.
    """
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
    # ctx.ascend.precision_mode = "enforce_fp32"

    kv_per = NUM_KV_HEADS // tp_size

    pf = None
    pf_in_dtypes = None
    if not decode_only:
        print(f"[rank{rank_id}] building prefill (seq={bucket_seq}, kv_len={kv_len})...", flush=True)
        pf = mslite.Model()
        pf.build_from_file(prefill_path, mslite.ModelType.MINDIR, ctx, pf_config_file)
        pf_in_dtypes = [t.dtype for t in pf.get_inputs()]

    print(f"[rank{rank_id}] building decode (kv_len={kv_len})...", flush=True)
    dc = mslite.Model()
    dc.build_from_file(decode_path, mslite.ModelType.MINDIR, ctx, dc_config_file)
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

    # Prefill KV output length == seq (no graph-side padding); host-side
    # padding to kv_len = seq + 512 is done by _reconstruct_kv_tp.
    # Decode KV in/out length == kv_len.
    pf_warmup_out = [
        mslite.Tensor(np.zeros((1, 1, VOCAB), np.float32)),  # prefill logits now (1,1,vocab) = real-last
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, bucket_seq, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, bucket_seq, HEAD_DIM), np.float16)),
    ]
    dc_warmup_out = [
        mslite.Tensor(np.zeros((1, 1, VOCAB), np.float32)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM), np.float16)),
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
    # Warmup MUST hit the selected bucket's compiled shape (seq=bucket_seq, KV=kv_len);
    # otherwise the online-GE dynamicDims graph specializes a different profile than the
    # one the timed prefill/decode calls use (memory: TP warmup dummy must use the bucket
    # seq, not a hard-coded 64).
    pf_dummy = [np.zeros((1, bucket_seq), np.int64),
                np.ones((1, bucket_seq), np.int64),
                np.arange(bucket_seq, dtype=np.int64).reshape(1, -1)]
    dc_dummy_kv = np.zeros((NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM), np.float16)
    dc_dummy = [np.array([[1]], np.int64), np.ones((1, kv_len), np.int64),
                np.array([[5]], np.int64), dc_dummy_kv, dc_dummy_kv]

    for _ in range(warmup):
        if not decode_only:
            pf_feed = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate(pf_dummy)]
            pf.predict(pf_feed, pf_warmup_out)
        dc_feed = [_tp_tensor(a, dc_in_dtypes[i]) for i, a in enumerate(dc_dummy)]
        dc.predict(dc_feed, dc_warmup_out)
    print(f"[rank{rank_id}] warmup done", flush=True)
    ready_q.put(1)

    # --- perf_sweep: multi-bucket loop (workers stay alive across all buckets) ---
    if perf_sweep and not decode_only and pf is not None:
        while True:
            bucket_info = bucket_q.get()
            if bucket_info is None:
                break
            cur_seq, cur_kv_len = bucket_info
            prompt_data = prompt_q.get()
            pids, pam, ppos, real_len, seq = prompt_data
            pf_inputs = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate([pids, pam, ppos])]
            pf_outputs = [
                mslite.Tensor(np.zeros((VOCAB,), np.float32)),  # logits now 1D [vocab] = real-last token
                mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, cur_seq + MAX_OUTPUT_TOKENS, HEAD_DIM), np.float16)),
                mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, cur_seq + MAX_OUTPUT_TOKENS, HEAD_DIM), np.float16)),
            ]
            pf_outs = pf.predict(pf_inputs, pf_outputs)
            pf_logits = pf_outs[0].get_data_to_numpy()
            first_token = int(np.argmax(pf_logits.reshape(-1)))
            prefill_logits_q.put(first_token)
            past_k_np = _reconstruct_kv_tp(pf_outs[1].get_data_to_numpy(), real_len, seq, tp_size)
            past_v_np = _reconstruct_kv_tp(pf_outs[2].get_data_to_numpy(), real_len, seq, tp_size)
            kv_out_len = int(past_k_np.shape[3])
            assert kv_out_len == cur_kv_len, (
                f"[rank{rank_id}] prefill KV out len {kv_out_len} != bucket kv_len {cur_kv_len} "
                f"(seq={seq}); host-side padding failed")
            print(f"[rank{rank_id}] 核心点 OK: prefill KV padded to kv_len == {kv_out_len} "
                  f"(seq bucket {seq}, NOT max 3584)", flush=True)
            # Zero-copy: pre-allocated device tensors + KV ping-pong (large tensors stay on device)
            zc_io = _DecodeZeroCopyIO(device_id, kv_per, cur_kv_len, dc_in_dtypes, tp_size)
            zc_io.load_kv(past_k_np, past_v_np)
            while True:
                step = step_q.get()
                if step is None:
                    break
                dids, dam, dpos = step
                outs = zc_io.predict_step(dc, dids, dam, dpos)
                logits = outs[0].get_data_to_numpy()
                decode_out_q.put(logits)
        os._exit(0)

    # --- Single-bucket mode (original path) ---
    if decode_only:
        # Hybrid mode: driver sends (first_token, past_k, past_v) from a 1p prefill
        first_token, past_k_np, past_v_np = prompt_q.get()
        print(f"[rank{rank_id}] received 1p-prefill KV (first_token={first_token})", flush=True)
    else:
        prompt_data = prompt_q.get()
        pids, pam, ppos, real_len, seq = prompt_data
        pf_inputs = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate([pids, pam, ppos])]
        pf_outputs = [
            mslite.Tensor(np.zeros((VOCAB,), np.float32)),  # logits now 1D [vocab] = real-last token
            mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, seq + MAX_OUTPUT_TOKENS, HEAD_DIM), np.float16)),
            mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, seq + MAX_OUTPUT_TOKENS, HEAD_DIM), np.float16)),
        ]
        pf_outs = pf.predict(pf_inputs, pf_outputs)
        pf_logits = pf_outs[0].get_data_to_numpy()
        first_token = int(np.argmax(pf_logits.reshape(-1)))
        prefill_logits_q.put(first_token)
        past_k_np = _reconstruct_kv_tp(pf_outs[1].get_data_to_numpy(), real_len, seq, tp_size)
        past_v_np = _reconstruct_kv_tp(pf_outs[2].get_data_to_numpy(), real_len, seq, tp_size)
        # Key point: prefill outputs seq-length KV; _reconstruct_kv_tp pads to
        # kv_len = seq + 512 on host side. Assert the padded length matches.
        kv_out_len = int(past_k_np.shape[3])
        assert kv_out_len == kv_len, (
            f"[rank{rank_id}] prefill KV out len {kv_out_len} != bucket kv_len {kv_len} "
            f"(seq={seq}); host-side padding failed")
        print(f"[rank{rank_id}] 核心点 OK: prefill KV padded to kv_len == {kv_out_len} "
              f"(seq bucket {seq}, NOT max 3584)", flush=True)

    # Zero-copy: pre-allocated device tensors + KV ping-pong (large tensors stay on device)
    zc_io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size)
    zc_io.load_kv(past_k_np, past_v_np)
    while True:
        step = step_q.get()
        if step is None:
            break
        dids, dam, dpos = step
        outs = zc_io.predict_step(dc, dids, dam, dpos)
        logits = outs[0].get_data_to_numpy()
        decode_out_q.put(logits)
    os._exit(0)


def _tp_prepare_prompt(tok, prompt, seq=TP_PREFILL_SEQ, force_len=None):
    """Tokenize prompt → (pids, pam, ppos, real_len, seq), padding to bucket `seq`.

    force_len (perf sweep only): synthesize exactly that many valid tokens instead
    of tokenizing `prompt`, so a caller can deterministically hit a target bucket.
    Output text is meaningless under force_len — timings + KV shapes are the point.
    """
    if force_len is not None:
        filler = int(tok.encode("A", add_special_tokens=False)[0])
        input_ids = np.full((1, int(force_len)), filler, dtype=np.int64)
    else:
        input_ids = _tokenize(tok, prompt)
    real_len = int(input_ids.shape[1])
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


def _tp_spawn_unified_workers(prefill_ranks, decode_ranks, device_ids,
                              pf_config_file, dc_config_file, bucket_seq, kv_len,
                              warmup, tp_size, use_hybrid, perf_sweep=False):
    """Spawn workers with staggered builds (avoid TBE cache race + GE port 16666 race).

    Returns (procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs, bucket_qs).
    bucket_qs is None when perf_sweep=False (single-bucket backward-compatible path).
    """
    prompt_qs = [Queue() for _ in range(tp_size)]
    pf_logits_qs = [Queue() for _ in range(tp_size)]
    step_qs = [Queue() for _ in range(tp_size)]
    out_qs = [Queue() for _ in range(tp_size)]
    ready_qs = [Queue() for _ in range(tp_size)]
    built_qs = [Queue() for _ in range(tp_size)]
    start_warmup_qs = [Queue() for _ in range(tp_size)]
    bucket_qs = [Queue() for _ in range(tp_size)] if perf_sweep else None
    procs = []
    for r in range(tp_size):
        p = Process(target=_tp_unified_worker, args=(
            prefill_ranks[r], decode_ranks[r], device_ids[r], r,
            pf_config_file, dc_config_file, bucket_seq, kv_len,
            prompt_qs[r], pf_logits_qs[r], step_qs[r], out_qs[r], ready_qs[r],
            built_qs[r], start_warmup_qs[r], warmup, tp_size, use_hybrid,
            perf_sweep, bucket_qs[r] if bucket_qs else None))
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
    return procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs, bucket_qs


def _tp_hybrid_prefill(device_ids, pids, pam, ppos, kv_per, prompt_qs, tp_size):
    """1p prefill on device 0 for correct KV, then shard KV to all ranks.

    1p prefill outputs seq-length KV (no graph-side padding); pad to kv_len
    = seq + 512 on host before sharding to decode workers.
    """
    print(f"[Hybrid: running 1p prefill on device {device_ids[0]} for correct KV...]")
    pf_1p_path = "./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0_graph.mindir"
    ctx1p = mslite.Context()
    ctx1p.target = ["ascend"]
    ctx1p.ascend.device_id = device_ids[0]
    ctx1p.ascend.provider = "ge"
    pf1p = mslite.Model()
    pf1p.build_from_file(pf_1p_path, mslite.ModelType.MINDIR, ctx1p)
    pf1p_inputs = [_tp_tensor(a.astype(np.int32)) for a in [pids, pam, ppos]]
    pf1p_outs = pf1p.predict(pf1p_inputs)
    pf1p_logits = pf1p_outs[0].get_data_to_numpy()
    first_token = int(np.argmax(pf1p_logits.reshape(-1)))  # logits now 1D [vocab] = real-last token
    full_k = pf1p_outs[1].get_data_to_numpy()
    full_v = pf1p_outs[2].get_data_to_numpy()
    seq = int(pids.shape[1])
    kv_len = seq + MAX_OUTPUT_TOKENS
    print(f"[Hybrid: 1p prefill done, first_token={first_token}, sharding KV for {tp_size} ranks...]")
    # Prefill already pads KV to kv_len inside the graph; shard by KV head to each rank
    for r in range(tp_size):
        sk = full_k[:, :, r * kv_per:(r + 1) * kv_per, :kv_len, :]
        sv = full_v[:, :, r * kv_per:(r + 1) * kv_per, :kv_len, :]
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
        first_token = _tp_hybrid_prefill(device_ids, pids, pam, ppos, kv_per,
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
                      valid_len, stream, tp_size, kv_len=KV_CACHE_LEN):
    """Run decode loop with optional streaming; return (generated, decode_times, truncated)."""
    generated = [first_token]
    streamed = tok.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False) if stream else ""
    if stream:
        print(streamed, end="", flush=True)
    cur_am = np.zeros((1, kv_len), np.int64)
    cur_am[0, :valid_len] = 1
    decode_times = []
    truncated = False
    for _ in range(max_new_tokens - 1):
        if eos is not None and generated[-1] == int(eos):
            break
        if valid_len >= kv_len:
            # KV cache for the selected bucket is full — stop and flag truncation.
            truncated = True
            print(f"\n[warning] output length hit KV cache size ({kv_len}); truncating.", flush=True)
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
    return generated, decode_times, truncated


def _tp_finalize(procs, step_qs, tp_size):
    """Signal workers to exit and join."""
    for r in range(tp_size):
        step_qs[r].put(None)
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()


def run_tp_infer(prefill_ranks, decode_ranks, tokenizer_path,
                 pf_config_file, dc_config_file, prompt, max_new_tokens, device_ids,
                 warmup=2, stream=True, tp_size=2, use_hybrid=None, seq=None, kv_len=None,
                 prompt_tokens=None):
    """Drive TP inference: spawn one worker per rank, feed prompt, stream decode output.

    `seq`/`kv_len` are the selected bucket dims (from _pick_prefill_seq on real_len);
    when None they are derived from the prompt length so a bare call still buckets.
    `prompt_tokens` (perf sweep) forces a synthetic prompt of exactly that many tokens
    so the caller can deterministically drive each bucket; overrides `prompt`/`seq`."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if prompt_tokens is not None:
        seq = _pick_prefill_seq(int(prompt_tokens))
    elif seq is None:
        probe = _tokenize(tok, prompt)
        seq = _pick_prefill_seq(int(probe.shape[1]))
    if kv_len is None:
        kv_len = seq + MAX_OUTPUT_TOKENS
    pids, pam, ppos, real_len, seq = _tp_prepare_prompt(tok, prompt, seq, force_len=prompt_tokens)
    print(f"[Bucket] real_len={real_len} -> prefill_seq={seq}, kv_len={kv_len}", flush=True)
    use_hybrid = tp_size >= 4 if use_hybrid is None else use_hybrid
    kv_per = NUM_KV_HEADS // tp_size
    procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs, _ = _tp_spawn_unified_workers(
        prefill_ranks, decode_ranks, device_ids, pf_config_file, dc_config_file,
        seq, kv_len, warmup, tp_size, use_hybrid)

    print("Waiting for warmup...", flush=True)
    for r in range(tp_size):
        ready_qs[r].get()
    print("All workers ready. Starting timed inference.", flush=True)

    first_token, prefill_ms = _tp_run_prefill(
        use_hybrid, device_ids, pids, pam, ppos, real_len, seq, kv_per,
        prompt_qs, pf_logits_qs, tp_size)
    generated, decode_times, truncated = _tp_stream_decode(
        tok, step_qs, out_qs, first_token, tok.eos_token_id, max_new_tokens,
        real_len, stream, tp_size, kv_len)

    total_decode = sum(decode_times)
    avg_decode = total_decode / len(decode_times) if decode_times else 0.0
    perf = {"prefill_ms": prefill_ms, "total_decode_ms": total_decode * 1000,
            "avg_decode_ms": avg_decode * 1000, "input_len": real_len,
            "output_len": len(generated), "prefill_seq": seq, "kv_len": kv_len,
            "truncated": truncated,
            "decode_step_ms": [t * 1000 for t in decode_times],
            "generated_ids": [int(x) for x in generated]}

    _tp_finalize(procs, step_qs, tp_size)
    return tok.decode(generated, skip_special_tokens=True), perf


def run_tp_perf_sweep(prefill_ranks, decode_ranks, tokenizer_path,
                      pf_config_file, dc_config_file, device_ids,
                      bucket_tokens, repeats, max_new_tokens, tp_size=2):
    """Drive TP perf sweep: spawn workers ONCE, loop over all 6 buckets with repeats.

    Unlike run_tp_infer (which spawns/finalizes workers per call), this function
    builds prefill+decode once, warms up with the first bucket, then loops over
    all buckets sending (bucket_seq, kv_len) + prompt data via queues. Workers
    stay alive until all buckets are done.

    Protocol:
      1. Driver sends (seq, kv_len) to bucket_q for each repeat of each bucket.
      2. Driver sends prompt_data to prompt_q; worker runs prefill, returns first_token.
      3. Driver runs decode loop via step_q/out_q, then sends None to step_q to end decode.
      4. After all buckets: driver sends None to bucket_q to exit workers.

    Returns a list of per-bucket steady-state perf dicts (repeats[1:] averaged).
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_hybrid = tp_size >= 4
    kv_per = NUM_KV_HEADS // tp_size

    # Use first bucket for initial warmup (HCCL sync + first-shape lazy compile)
    first_seq = _pick_prefill_seq(bucket_tokens[0])
    first_kv_len = first_seq + MAX_OUTPUT_TOKENS

    procs, prompt_qs, pf_logits_qs, step_qs, out_qs, ready_qs, bucket_qs = \
        _tp_spawn_unified_workers(
            prefill_ranks, decode_ranks, device_ids, pf_config_file, dc_config_file,
            first_seq, first_kv_len, 2, tp_size, use_hybrid, perf_sweep=True)

    print("Waiting for warmup...", flush=True)
    for r in range(tp_size):
        ready_qs[r].get()
    print("All workers ready. Starting perf sweep (single build, multi-bucket).",
          flush=True)

    results = []
    for ntok in bucket_tokens:
        seq = _pick_prefill_seq(ntok)
        kv_len = seq + MAX_OUTPUT_TOKENS
        print(f"\n===== bucket: prompt-tokens={ntok}  seq={seq}  kv_len={kv_len}  "
              f"repeats={repeats} =====", flush=True)

        bucket_perfs = []
        for rep in range(repeats):
            # Send bucket params to all workers
            for r in range(tp_size):
                bucket_qs[r].put((seq, kv_len))

            # Prepare synthetic prompt for this bucket
            pids, pam, ppos, real_len, _ = _tp_prepare_prompt(
                tok, "", seq, force_len=ntok)

            # Run prefill
            first_token, prefill_ms = _tp_run_prefill(
                use_hybrid, device_ids, pids, pam, ppos, real_len, seq, kv_per,
                prompt_qs, pf_logits_qs, tp_size)

            # Run decode (no streaming for perf measurement)
            generated, decode_times, truncated = _tp_stream_decode(
                tok, step_qs, out_qs, first_token, tok.eos_token_id,
                max_new_tokens, real_len, False, tp_size, kv_len)

            # End this repeat's decode loop (worker loops back to bucket_q)
            for r in range(tp_size):
                step_qs[r].put(None)

            total_decode = sum(decode_times)
            avg_decode = total_decode / len(decode_times) if decode_times else 0.0
            perf = {
                "prompt_tokens": ntok,
                "prefill_seq": seq,
                "kv_len": kv_len,
                "prefill_ms": round(prefill_ms, 2),
                "avg_decode_ms": round(avg_decode * 1000, 2),
                "output_len": len(generated),
                "truncated": truncated,
            }
            tag = "warmup(含懒编译)" if rep == 0 else f"steady #{rep}"
            print(f"  [{tag}] prefill={perf['prefill_ms']}ms  "
                  f"decode_avg={perf['avg_decode_ms']}ms  "
                  f"output_len={perf['output_len']}", flush=True)
            bucket_perfs.append(perf)

        # Steady = repeats 2+ (or all if only 1 repeat)
        steady = bucket_perfs[1:] if len(bucket_perfs) > 1 else bucket_perfs
        steady_perf = {
            "prompt_tokens": ntok,
            "prefill_seq": seq,
            "kv_len": kv_len,
            "prefill_ms": round(sum(p["prefill_ms"] for p in steady) / len(steady), 2),
            "avg_decode_ms": round(sum(p["avg_decode_ms"] for p in steady) / len(steady), 2),
            "output_len": steady[-1]["output_len"],
            "truncated": steady[-1]["truncated"],
        }
        results.append(steady_perf)
        print(f"  [bucket {ntok}] steady: prefill_avg={steady_perf['prefill_ms']}ms  "
              f"decode_avg={steady_perf['avg_decode_ms']}ms", flush=True)

    # Signal workers to exit
    for r in range(tp_size):
        bucket_qs[r].put(None)
    for p in procs:
        p.join(timeout=15)
        if p.is_alive():
            p.terminate()

    return results


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


def _write_rank_table(device_ids, run_dir):
    """Write rank_table.json for the given device ids; return its path."""
    os.makedirs(run_dir, exist_ok=True)
    devs_json = ",".join(
        f'{{"device_id":"{d}","rank_id":"{i}"}}' for i, d in enumerate(device_ids))
    rank_table = (
        '{"version":"1.0","server_count":"1","server_list":['
        '{"server_id":"127.0.0.1","device":[' + devs_json + '],'
        '"host_nic_ip":"reserve"}],"status":"completed"}'
    )
    rank_table_path = os.path.join(run_dir, "rank_table.json")
    with open(rank_table_path, "w", encoding="utf-8") as f:
        f.write(rank_table)
    return rank_table_path


# precision_mode is set on the Context (enforce_fp32), NOT in the config_file:
# putting an [acl_init_options]-style precision key there makes GE bind port
# 16666 eagerly at Context creation, colliding with the other rank. So we drop
# any precision_mode line when grafting [ge_session_options].
_GE_SESSION_DROP_KEYS = ("precision_mode",)


def _extract_ge_section(cfg_path, section):
    """Return the raw non-comment lines under [<section>] of a cfg (excluding header)."""
    lines = []
    in_section = False
    with open(cfg_path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_section = stripped == f"[{section}]"
                continue
            if in_section and stripped and not stripped.startswith("#"):
                lines.append(stripped)
    return lines


def _extract_ge_graph_options(cfg_path):
    """Grafts a tp2 dynamicDims cfg's ge.inputShape / ge.dynamicDims / ge.dynamicNodeType
    into the HCCL config_file so a single online-GE build serves all 6 seq/KV buckets."""
    return _extract_ge_section(cfg_path, "ge_graph_options")


def _extract_ge_session_options(cfg_path):
    """Graft the cfg's [ge_session_options] GE tuning knobs (constLifecycle, formatMode,
    atomicCleanPolicy, event, staticMemoryPolicy) into the HCCL config_file so the TP
    online-GE build matches the 1p session tuning. precision_mode is dropped here and
    kept on the Context (enforce_fp32) to avoid the eager port-16666 bind on the other rank."""
    return [ln for ln in _extract_ge_section(cfg_path, "ge_session_options")
            if not any(ln.split("=", 1)[0].strip().endswith(k) for k in _GE_SESSION_DROP_KEYS)]


def _write_hccl_config_with_ge(device_ids, run_dir, ge_cfg_path, tag):
    """Write config_file_<tag>.ini merging the HCCL rank-table with one cfg's
    [ge_session_options] (tuning knobs, minus precision_mode) and [ge_graph_options]
    (dynamicDims). One file per sub-graph (prefill/decode) since each carries its own
    ge.inputShape / ge.dynamicDims. Returns the path.

    Also propagates model_cache_mode / model_cache_dir from [ascend_context] and the
    [lite_inner_group] section so GE compilation cache and Session sharing take effect
    on the TP path."""
    rank_table_path = _write_rank_table(device_ids, run_dir)
    session_lines = _extract_ge_session_options(ge_cfg_path)
    ge_lines = _extract_ge_graph_options(ge_cfg_path)
    cache_lines = [ln for ln in _extract_ge_section(ge_cfg_path, "ascend_context")
                   if ln.startswith("model_cache_mode=") or ln.startswith("model_cache_dir=")]
    group_lines = _extract_ge_section(ge_cfg_path, "lite_inner_group")
    config_path = os.path.join(run_dir, f"config_file_{tag}.ini")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[ascend_context]\n")
        f.write(f"rank_table_file={rank_table_path}\n")
        f.write("plugin_custom_ops=All\n")
        for line in cache_lines:
            f.write(line + "\n")
        if session_lines:
            f.write("\n[ge_session_options]\n")
            for line in session_lines:
                f.write(line + "\n")
        if ge_lines:
            f.write("\n[ge_graph_options]\n")
            for line in ge_lines:
                f.write(line + "\n")
        if group_lines:
            f.write("\n[lite_inner_group]\n")
            for line in group_lines:
                f.write(line + "\n")
    return config_path


def _write_hccl_config(device_ids, run_dir):
    """Write rank_table.json + config_file.ini for the given device ids; return config path."""
    rank_table_path = _write_rank_table(device_ids, run_dir)
    config_path = os.path.join(run_dir, "config_file.ini")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("[ascend_context]\n")
        f.write(f"rank_table_file={rank_table_path}\n")
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
    # ---- TP GE config dir (dynamicDims ge_prefill.cfg / ge_decode.cfg; auto-resolved if omitted) ----
    parser.add_argument("--bucket-cfg-dir", type=str, default=None,
                        help="TP prefill/decode: directory containing ge_prefill.cfg and "
                             "ge_decode.cfg (default: configs/tp2 for TP=2, configs/tp4 for TP=4)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="write perf dict as JSON to this path")
    parser.add_argument("--prompt-tokens", type=int, default=None,
                        help="perf sweep (TP=2): synthesize a prompt of exactly N tokens "
                             "to deterministically drive one prefill bucket (overrides --prompt)")
    parser.add_argument("--perf-sweep", action="store_true", default=False,
                        help="TP=2 perf mode: single process, build once, loop over all 6 "
                             "buckets with repeats. Replaces the infer.sh 6x process loop.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="perf sweep: repeats per bucket (first = warmup/lazy-compile, "
                             "steady = repeats 2+). Only used with --perf-sweep.")
    return parser.parse_args()


def _run_single_chip(args, device_ids):
    """Run the single-chip (1p) dynamic single-graph path.

    1p uniformly uses infer_qwen3_8b_mslite_1p.DynamicBucketRunner (single prefill mindir +
    single decode mindir, configs/ge_prefill.cfg / ge_decode.cfg, ge.dynamicDims 8 buckets,
    compile once and infer across buckets, KV slicing).
    The old multi-static bucket cfgs (ge_prefill_bucket_*.cfg) are no longer used (removed).
    """
    # Deferred import to avoid a circular dependency (infer_qwen3_8b_mslite_1p imports this module's helpers at the top).
    from infer_qwen3_8b_mslite_1p import DynamicBucketRunner

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    runner = DynamicBucketRunner(device_ids[0], tok)

    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    txt, perf = runner.run(args.prompt, max_new_tokens=args.max_new_tokens)
    print(f"\n{txt[:300]}")
    print("\n--- Performance (1p 动态单图) ---")
    print(f"  real_len={perf['real_len']}  prefill_seq={perf['prefill_seq']}  "
          f"kv_len={perf['kv_len']}")
    print(f"  KV: phys={perf['kv_out_phys']} -> pad到 {perf['kv_padded_to']}  "
          f"核心点OK={perf['kv_len_ok']}")
    print(f"  prefill_steady={perf['prefill_ms']}ms  "
          f"decode_first={perf['decode_first_ms']}ms  decode_min={perf['decode_min_ms']}ms  "
          f"truncated={perf['truncated']}")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tp": 1, "prompt": args.prompt, **perf}, f, indent=2)


def _run_tensor_parallel(args, device_ids, tp_size):
    """Run the tensor-parallel (2p/4p) multi-process path."""
    model_dir = _auto_model_dir(tp_size)
    if args.prefill_ranks:
        prefill_ranks = args.prefill_ranks.split(",")
    else:
        prefill_ranks = _rank_paths(model_dir, tp_size, "prefill")
    if args.decode_ranks:
        decode_ranks = args.decode_ranks.split(",")
    else:
        decode_ranks = _rank_paths(model_dir, tp_size, "decode")
    run_dir = os.path.join(os.getcwd(), "tp_run")

    if tp_size == 2:
        # 2p bucketed dynamicDims: graft each sub-graph's [ge_graph_options]
        # (ge.inputShape + ge.dynamicDims + ge.dynamicNodeType=1) into the HCCL
        # config so a single online-GE build serves all 6 seq/KV buckets. seq/
        # kv_len are chosen from the prompt length inside run_tp_infer (seq=None).
        cfg_dir = args.bucket_cfg_dir or "configs/tp2"
        pf_config_file = _write_hccl_config_with_ge(
            device_ids, run_dir, os.path.join(cfg_dir, "ge_prefill.cfg"), "prefill")
        dc_config_file = _write_hccl_config_with_ge(
            device_ids, run_dir, os.path.join(cfg_dir, "ge_decode.cfg"), "decode")
        seq = kv_len = None
        print(f"[TP2] bucketed dynamicDims cfgs: {pf_config_file} / {dc_config_file}")
    else:
        # 4p keeps the fixed-shape path (plain HCCL config, no ge_graph_options).
        config_file = args.config_file or _write_hccl_config(device_ids, run_dir)
        pf_config_file = dc_config_file = config_file
        seq, kv_len = TP_PREFILL_SEQ, KV_CACHE_LEN

    # ---- perf sweep: single build, multi-bucket loop (replaces 6x process) ----
    if args.perf_sweep:
        bucket_tokens = list(PREFILL_SEQ_DIMS)
        print("\n" + "=" * 60)
        print(f"[perf-sweep] TP={tp_size}  buckets={bucket_tokens}  "
              f"repeats={args.repeats}  max_new_tokens={args.max_new_tokens}")
        print("=" * 60)
        results = run_tp_perf_sweep(
            prefill_ranks, decode_ranks, args.model_id,
            pf_config_file, dc_config_file, device_ids,
            bucket_tokens, args.repeats, args.max_new_tokens, tp_size=tp_size)
        # Print summary table
        print("\n" + "=" * 60)
        print(f"=== TP={tp_size} perf-sweep summary (steady-state, repeats 2+) ===")
        print("=" * 60)
        hdr = f"{'ntok':>6} {'seq':>6} {'kv_len':>7} {'prefill_ms':>11} {'decode_ms':>10} {'trunc':>6}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            print(f"{r['prompt_tokens']:>6} {r['prefill_seq']:>6} {r['kv_len']:>7} "
                  f"{r['prefill_ms']:>11} {r['avg_decode_ms']:>10} "
                  f"{str(r['truncated']):>6}")
        ok = all(r.get("kv_len") == (r.get("prefill_seq") or 0) + MAX_OUTPUT_TOKENS
                 for r in results)
        print(f"\n核心点校验 (kv_len == prefill_seq + {MAX_OUTPUT_TOKENS}): "
              f"{'PASS' if ok else 'FAIL'} ({len(results)}/6 档)")
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({"tp": tp_size, "buckets": results}, f, indent=2)
        return

    # ---- single-bucket mode (original path) ----
    print("\n" + "=" * 60)
    print(f"Input Prompt: {args.prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    _, perf = run_tp_infer(
        prefill_ranks, decode_ranks, args.model_id, pf_config_file, dc_config_file,
        args.prompt, args.max_new_tokens, device_ids, warmup=args.warmup,
        stream=not args.json_out, tp_size=tp_size, use_hybrid=None, seq=seq, kv_len=kv_len,
        prompt_tokens=args.prompt_tokens)
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
