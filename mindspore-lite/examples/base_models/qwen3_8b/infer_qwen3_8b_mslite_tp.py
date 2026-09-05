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
live under configs/.

Auto-dispatches by the number of device IDs:
  * 1 device  -> single-chip GE + per-bucket weight sharing + zero-copy decode
  * 2/4 devices -> tensor-parallel multi-process (HCCL, one worker per rank)
                 with GE provider and zero-copy KV handoff

Usage:
  python infer_qwen3_8b_mslite_tp.py --device-ids 0,1       # TP=2
  python infer_qwen3_8b_mslite_tp.py --device-ids 0,1,2,3   # TP=4
  # The same entry point handles one, two, or four devices.

Model paths auto-resolve from the device count (1p/2p/4p output dirs produced
by export_and_convert.sh); override with --prefill-ranks / --decode-ranks (TP)
or --prefill-model / --decode-model (1p).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing import Process, Queue

import numpy as np

# Set the HCCL NPU socket port range BEFORE importing mindspore_lite so GE/HCCL
# read it at the C-library init (the default port 16666 otherwise collides
# across same-card ranks in TP). Placed at module top so both the driver and
# spawn-forked workers inherit it before any GE init.
os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "21500-21600"

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

    NOTE: no longer used on the main TP path — the prefill graph pads KV to
    kv_len = seq + MAX_OUTPUT_TOKENS inside the graph, and the decode now takes
    the prefill DEVICE output tensors directly (zero-copy attach_kv). Kept for
    reference / legacy paths where a shard-by-KV-head host copy is needed.
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

    KV handoff from prefill is zero-copy as well:
      * attach_kv(k_dev, v_dev): the prefill graph's DEVICE output tensors (written by
        prefill predict(outputs=...) into pre-allocated device tensors of shape kv_len)
        are used directly as the first decode step's k_in/v_in — the KV never crosses
        the host. Only the small inputs (ids/attn/pos) are updated per step via
        set_data_from_numpy; the large KV tensors ping-pong on device.
      * load_kv(past_k_np, past_v_np): fallback for cross-process KV handoff
        (hybrid/decode-only path), one-time H2D from host numpy.

    Usage:
      io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size)
      io.attach_kv(pf_k_dev, pf_v_dev)    # zero-copy handoff, or
      io.load_kv(past_k_np, past_v_np)    # one-time H2D
      for step in ...: io.predict_step(dc, dids, dam, dpos)
    """

    def __init__(self, device_id, kv_per, kv_len, dc_in_dtypes, tp_size,
                 dual_buffers=True):
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
        # KV cache ping-pong buffers. dual_buffers=True (host-KV handoff): A is the
        # first input, B the first output. dual_buffers=False (zero-copy attach_kv):
        # the prefill device output tensors are the first input, B is the first output,
        # so A is not allocated (saves one KV pair of device memory).
        if dual_buffers:
            self.t_k_a = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
            self.t_v_a = mslite.Tensor(shape=kv_shape, dtype=kv_dtype, device=device_str)
        else:
            self.t_k_a = self.t_v_a = None
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
        # Current input/output buffer pointers (set by load_kv / attach_kv)
        self._k_in = self._v_in = None
        self._k_out = self._v_out = None

    def load_kv(self, past_k_np, past_v_np):
        """Write the prefill KV outputs into device buffers for the first time (one-time H2D).

        Host-KV handoff (hybrid/decode-only path): KV arrives from the driver process as
        host numpy; write it into the pre-allocated device buffers once, then ping-pong
        on device.
        """
        assert self.t_k_a is not None, "load_kv requires dual_buffers=True"
        self._k_in, self._v_in = self.t_k_a, self.t_v_a
        self._k_out, self._v_out = self.t_k_b, self.t_v_b
        self._k_in.set_data_from_numpy(
            np.ascontiguousarray(past_k_np.astype(self._np_kv)))
        self._v_in.set_data_from_numpy(
            np.ascontiguousarray(past_v_np.astype(self._np_kv)))

    def attach_kv(self, k_dev, v_dev):
        """Zero-copy KV handoff: use the prefill's device output tensors directly
        as the first decode step's k_in/v_in.

        Prefill writes KV into pre-allocated DEVICE tensors (shape == decode kv_len, same
        device); those tensors become the first decode inputs with no Host<->Device copy.
        Only the small ids/attn/pos inputs are updated per step via set_data_from_numpy;
        the large KV tensors ping-pong between the attached tensors and t_k_b/t_v_b.
        """
        self._k_in, self._v_in = k_dev, v_dev
        self._k_out, self._v_out = self.t_k_b, self.t_v_b

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


def _worker_build(prefill_path, decode_path, device_id, rank_id, pf_config_file,
                  dc_config_file, bucket_seq, kv_len, decode_only):
    """Build prefill + decode GE models; return their dtypes for tensor allocation."""
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

    # Prefill output specs (dtypes) — used to pre-allocate DEVICE output tensors so the
    # prefill->decode KV handoff is zero-copy: prefill writes KV straight into
    # device memory and decode's first k_in/v_in reference those tensors directly.
    pf_out_dtypes = [t.dtype for t in pf.get_outputs()] if pf is not None else []
    pf_kv_dtype = pf_out_dtypes[1] if len(pf_out_dtypes) > 1 else dc_in_dtypes[3]
    return pf, pf_in_dtypes, dc, dc_in_dtypes, pf_out_dtypes, pf_kv_dtype


def _worker_warmup(pf, pf_in_dtypes, dc, dc_in_dtypes, kv_per, bucket_seq, kv_len,
                   tp_size, warmup, decode_only, rank_id, built_q, start_warmup_q,
                   ready_q):
    """Signal build done, wait for the warmup barrier, then warm up compiled shapes."""
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

    # Warmup only: prefill/decode outputs are host numpy buffers here (no timing,
    # no handoff). Shapes MUST match the real graph outputs:
    #   * prefill logits is 1D [vocab] (real-last token, see Qwen3LlmPrefill);
    #   * prefill K/V output length == kv_len = bucket_seq + MAX_OUTPUT_TOKENS
    #     (the graph pads 512 empty KV slots inside the graph);
    #   * decode K/V in/out length == kv_len, logits [1, 1, vocab].
    pf_warmup_out = [
        mslite.Tensor(np.zeros((VOCAB,), np.float32)),  # prefill logits 1D [vocab] = real-last token
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM), np.float16)),
        mslite.Tensor(np.zeros((NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM), np.float16)),
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


def _worker_prefill_bucket(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype, dc_in_dtypes,
                           device_id, rank_id, kv_per, kv_len, prompt_q,
                           prefill_logits_q, tp_size):
    """Prefill one bucket into DEVICE output tensors; return the zero-copy decode IO."""
    dev = f"ascend:{device_id}"
    prompt_data = prompt_q.get()
    pids, pam, ppos, _, seq = prompt_data
    pf_inputs = [_tp_tensor(a, pf_in_dtypes[i]) for i, a in enumerate([pids, pam, ppos])]
    # Zero-copy prefill->decode KV handoff: pre-allocated DEVICE output
    # tensors, reused directly as the decode's first k_in/v_in (no D2H/H2D).
    # Tensors are re-created per bucket since kv_len differs across buckets.
    pf_logits_dtype = pf_out_dtypes[0] if pf_out_dtypes else mslite.DataType.FLOAT32
    pf_outputs = [
        mslite.Tensor(shape=[VOCAB], dtype=pf_logits_dtype, device=dev),
        mslite.Tensor(shape=[NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM],
                      dtype=pf_kv_dtype, device=dev),
        mslite.Tensor(shape=[NUM_LAYERS, 1, kv_per, kv_len, HEAD_DIM],
                      dtype=pf_kv_dtype, device=dev),
    ]
    pf_outs = pf.predict(pf_inputs, pf_outputs)
    pf_logits = pf_outs[0].get_data_to_numpy()
    first_token = int(np.argmax(pf_logits.reshape(-1)))
    prefill_logits_q.put(first_token)
    # Key point: prefill graph pads KV to kv_len = seq + 512 inside the graph
    # (NOT max 3584); the device output buffer shape must match.
    kv_out_len = int(pf_outs[1].shape[3])
    assert kv_out_len == kv_len, (
        f"[rank{rank_id}] prefill KV out len {kv_out_len} != bucket kv_len "
        f"{kv_len} (seq={seq}); graph-side padding failed")
    print(f"[rank{rank_id}] key-point OK: prefill KV device output == kv_len {kv_out_len} "
          f"(seq bucket {seq}, NOT max 3584)", flush=True)
    # Zero-copy: pre-allocated device tensors + KV ping-pong (large tensors stay on device)
    zc_io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size,
                              dual_buffers=False)
    if pf_kv_dtype == dc_in_dtypes[3]:
        # Zero-copy: prefill device KV feeds decode's k_in/v_in directly (no D2H/H2D)
        zc_io.attach_kv(pf_outs[1], pf_outs[2])
    else:
        print(f"[rank{rank_id}] KV dtype mismatch (pf={pf_kv_dtype} "
              f"dc={dc_in_dtypes[3]}); falling back to one-time H2D", flush=True)
        zc_io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size)
        zc_io.load_kv(pf_outs[1].get_data_to_numpy(), pf_outs[2].get_data_to_numpy())
    return zc_io


def _worker_decode_loop(dc, step_q, decode_out_q, zc_io):
    """Consume decode steps until None; send logits to the driver."""
    while True:
        step = step_q.get()
        if step is None:
            break
        dids, dam, dpos = step
        outs = zc_io.predict_step(dc, dids, dam, dpos)
        logits = outs[0].get_data_to_numpy()
        decode_out_q.put(logits)


def _worker_perf_sweep(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype, dc, dc_in_dtypes,
                       device_id, rank_id, kv_per, prompt_q, prefill_logits_q,
                       step_q, decode_out_q, bucket_q, tp_size):
    """Multi-bucket loop: one prefill + decode per bucket until bucket_q is None."""
    while True:
        bucket_info = bucket_q.get()
        if bucket_info is None:
            break
        _, cur_kv_len = bucket_info
        zc_io = _worker_prefill_bucket(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype,
                                       dc_in_dtypes, device_id, rank_id, kv_per,
                                       cur_kv_len, prompt_q, prefill_logits_q, tp_size)
        _worker_decode_loop(dc, step_q, decode_out_q, zc_io)


def _worker_single_bucket(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype, dc,
                          dc_in_dtypes, device_id, rank_id, kv_per, kv_len,
                          prompt_q, prefill_logits_q, step_q, decode_out_q,
                          tp_size, decode_only):
    """Single-bucket mode: hybrid 1p-prefill KV or TP prefill, then the decode loop."""
    if decode_only:
        # Hybrid mode: driver sends (first_token, past_k, past_v) from a 1p prefill
        first_token, past_k_np, past_v_np = prompt_q.get()
        print(f"[rank{rank_id}] received 1p-prefill KV (first_token={first_token})", flush=True)
        # KV comes from the driver process (host numpy): one-time H2D into the
        # pre-allocated device buffers, then the decode loop stays on device.
        zc_io = _DecodeZeroCopyIO(device_id, kv_per, kv_len, dc_in_dtypes, tp_size)
        zc_io.load_kv(past_k_np, past_v_np)
    else:
        zc_io = _worker_prefill_bucket(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype,
                                       dc_in_dtypes, device_id, rank_id, kv_per,
                                       kv_len, prompt_q, prefill_logits_q, tp_size)
    _worker_decode_loop(dc, step_q, decode_out_q, zc_io)


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
    os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "21500-21600"
    pf, pf_in_dtypes, dc, dc_in_dtypes, pf_out_dtypes, pf_kv_dtype = _worker_build(
        prefill_path, decode_path, device_id, rank_id, pf_config_file, dc_config_file,
        bucket_seq, kv_len, decode_only)
    kv_per = NUM_KV_HEADS // tp_size
    _worker_warmup(pf, pf_in_dtypes, dc, dc_in_dtypes, kv_per, bucket_seq, kv_len,
                   tp_size, warmup, decode_only, rank_id, built_q, start_warmup_q,
                   ready_q)
    # --- perf_sweep: multi-bucket loop (workers stay alive across all buckets) ---
    if perf_sweep and not decode_only and pf is not None:
        _worker_perf_sweep(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype, dc,
                           dc_in_dtypes, device_id, rank_id, kv_per, prompt_q,
                           prefill_logits_q, step_q, decode_out_q, bucket_q, tp_size)
        os._exit(0)
    # --- Single-bucket mode (original path) ---
    _worker_single_bucket(pf, pf_in_dtypes, pf_out_dtypes, pf_kv_dtype, dc,
                          dc_in_dtypes, device_id, rank_id, kv_per, kv_len, prompt_q,
                          prefill_logits_q, step_q, decode_out_q, tp_size, decode_only)
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
                              pf_config_files, dc_config_files, bucket_seq, kv_len,
                              warmup, tp_size, use_hybrid, perf_sweep=False):
    """Spawn workers with staggered builds (avoid TBE cache race + GE port 16666 race).

    pf_config_files/dc_config_files are per-rank config paths (each carries its own
    ge.graph_compiler_cache_dir=<base>/rank{r}); worker r builds with files[r].
    TP=2 starts rank1 before rank0 (the last cold-compiler historically crashed
    with TBE kernel compile-race "main process disappeared"); TP=4 keeps 0..3.

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
    # Stagger order: TP=2 starts rank1 BEFORE rank0. The rank that cold-compiles
    # the shared TBE custom-op kernels LAST historically crashes with "main process
    # disappeared" / task_distribute errors (TBE compile-cache race); starting
    # rank1 first lets it own the cache entries, then rank0 builds behind it
    # (built_qs[r].get() + sleep(8) between starts). TP=4 keeps 0..3 order.
    start_order = [1, 0] if tp_size == 2 else list(range(tp_size))
    for i, r in enumerate(start_order):
        p = Process(target=_tp_unified_worker, args=(
            prefill_ranks[r], decode_ranks[r], device_ids[r], r,
            pf_config_files[r], dc_config_files[r], bucket_seq, kv_len,
            prompt_qs[r], pf_logits_qs[r], step_qs[r], out_qs[r], ready_qs[r],
            built_qs[r], start_warmup_qs[r], warmup, tp_size, use_hybrid,
            perf_sweep, bucket_qs[r] if bucket_qs else None))
        p.start()
        procs.append(p)
        # Stagger builds so ranks don't cold-compile the same TBE kernels
        # concurrently (race kills workers); the sleep lets rank r's GE port
        # binding settle before the next rank inits (default port 16666 races).
        if i + 1 < len(start_order):
            nxt = start_order[i + 1]
            print(f"Waiting for rank{r} build before starting rank{nxt}...", flush=True)
            built_qs[r].get()
            time.sleep(8)
    built_qs[start_order[-1]].get()
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
                 pf_config_files, dc_config_files, prompt, max_new_tokens, device_ids,
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
        prefill_ranks, decode_ranks, device_ids, pf_config_files, dc_config_files,
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


def _perf_sweep_bucket(tok, ntok, repeats, tp_size, kv_per, seq, kv_len, bucket_qs,
                      prompt_qs, pf_logits_qs, step_qs, out_qs, use_hybrid,
                      device_ids, max_new_tokens):
    """Run one perf-sweep bucket: repeats of prefill+decode; return the steady perf."""
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
    print(f"  [bucket {ntok}] steady: prefill_avg={steady_perf['prefill_ms']}ms  "
          f"decode_avg={steady_perf['avg_decode_ms']}ms", flush=True)
    return steady_perf


def run_tp_perf_sweep(prefill_ranks, decode_ranks, tokenizer_path,
                      pf_config_files, dc_config_files, device_ids,
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
            prefill_ranks, decode_ranks, device_ids, pf_config_files, dc_config_files,
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
        results.append(_perf_sweep_bucket(
            tok, ntok, repeats, tp_size, kv_per, seq, kv_len, bucket_qs,
            prompt_qs, pf_logits_qs, step_qs, out_qs, use_hybrid, device_ids,
            max_new_tokens))

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
    return "./qwen3_8b_tp_quant_onnx"


def _rank_paths(model_dir, tp_size, sub):
    """Build the list of rank MindIR paths for a sub-graph (prefill/decode).

    TP>=2 exports are per-rank dirs (rank{r}/{sub}/...) so each rank's ONNX
    external-data files never collide; 1p keeps the flat layout.
    """
    if tp_size > 1:
        return [f"{model_dir}/rank{r}/{sub}/qwen3_8b_llm_{sub}_rank{r}_graph.mindir"
                for r in range(tp_size)]
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
# precision_mode must NOT be dropped: configs carry
# `ge.exec.precision_mode=must_keep_origin_dtype` under [ge_session_options];
# dropping it makes GE re-specialize the prefill fp16 KV outputs to fp32
# ("KV dtype mismatch (pf=FLOAT32 dc=FLOAT16)"), disabling zero-copy attach_kv.
_GE_SESSION_DROP_KEYS = ()


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
    atomicCleanPolicy, event, staticMemoryPolicy, precision_mode=must_keep_origin_dtype)
    into the HCCL config_file so the TP online-GE build matches the 1p session tuning
    (in particular keeping the fp16 KV output dtype)."""
    return [ln for ln in _extract_ge_section(cfg_path, "ge_session_options")
            if not any(ln.split("=", 1)[0].strip().endswith(k) for k in _GE_SESSION_DROP_KEYS)]


def _write_hccl_config_with_ge(device_ids, run_dir, ge_cfg_path, tag, cache_suffix=None):
    """Write config_file_<tag>.ini merging the HCCL rank-table with one cfg's
    [ge_session_options] (tuning knobs) and [ge_graph_options] (dynamicDims).
    One file per sub-graph (prefill/decode) since each carries its own
    ge.inputShape / ge.dynamicDims. Returns the path.

    cache_suffix: per-rank GE compile-cache sub-dir. The cfg carries
    `ge.graph_compiler_cache_dir=<base>`; each rank gets <base>/rank{suffix} so
    the two ranks never cold-compile the same TBE kernels into one directory
    (that race kills workers with "main process disappeared" / task_distribute
    errors and forces a full recompile on every run). A separate config file is
    generated per (sub-graph, rank).

    Also propagates model_cache_mode / model_cache_dir from [ascend_context] and the
    [lite_inner_group] section so GE compilation cache and Session sharing take effect
    on the TP path."""
    rank_table_path = _write_rank_table(device_ids, run_dir)
    session_lines = _extract_ge_session_options(ge_cfg_path)
    if cache_suffix is not None:
        # Per-rank cache sub-dir so the two ranks never compile into the same
        # directory concurrently. GE requires the cache dir to pre-exist
        # (E13026 "The cache directory does not exist"), so mkdir -p the full
        # path and write it into the config.
        new_session = []
        for ln in session_lines:
            if ln.startswith("ge.graph_compiler_cache_dir="):
                base = ln.split("=", 1)[1].strip().rstrip("/")
                cache_dir = f"{base}/rank{cache_suffix}"
                os.makedirs(cache_dir, exist_ok=True)
                new_session.append(f"ge.graph_compiler_cache_dir={cache_dir}")
            else:
                new_session.append(ln)
        session_lines = new_session
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
# Unified common-prefix runners (1P / TP)
# ===========================================================================
COMMON_PREFIX_BUCKET = 768
COMMON_SUFFIX_BUCKET = 896
BUCKET_SEQ_DIMS = (512, 896, 1024, 1664, 2048, 2560, 2816, 3072)

def _pick_seq(real_len):
    for dim in BUCKET_SEQ_DIMS:
        if dim >= real_len:
            return dim
    raise ValueError(f"prompt length {real_len} exceeds max bucket {BUCKET_SEQ_DIMS[-1]}")

class _DecodeLoopMixin:
    """Reusable zero-copy decode loop shared by all runners.

    Hosts must provide: self.decode, self.eos_token_id, self.t_dc_* tensors
    and self._dc_*_np dtypes (allocated by the runner's own __init__).
    """

    def _decode(self, first, pf_k, pf_v, real_len, kv_len, max_new_tokens):
        """Decode loop. pf_k/pf_v are prefill outputs (device tensors or numpy arrays)."""
        kv_heads = getattr(self, "num_kv_heads", NUM_KV_HEADS)
        din = self.decode.get_inputs()
        self.decode.resize(din, [[1, 1], [1, kv_len], [1, 1],
                                 [NUM_LAYERS, 1, kv_heads, kv_len, HEAD_DIM],
                                 [NUM_LAYERS, 1, kv_heads, kv_len, HEAD_DIM]])
        gen = [first]
        valid = real_len
        cur_attn = np.zeros((1, kv_len), dtype=self._dc_attn_np)
        cur_attn[0, :valid] = 1
        step_ms = []
        truncated = False

        kv_shape = [NUM_LAYERS, 1, kv_heads, kv_len, HEAD_DIM]
        self.t_dc_attn.shape = [1, kv_len]
        self.t_dc_k_a.shape = kv_shape
        self.t_dc_v_a.shape = kv_shape
        self.t_dc_k_b.shape = kv_shape
        self.t_dc_v_b.shape = kv_shape

        k_buf = np.zeros(kv_shape, dtype=self._dc_kv_np)
        v_buf = np.zeros(kv_shape, dtype=self._dc_kv_np)
        for src, dst in ((pf_k, k_buf), (pf_v, v_buf)):
            try:
                arr = src.get_data_to_numpy() if hasattr(src, 'get_data_to_numpy') else np.asarray(src)
            except (RuntimeError, ValueError, TypeError):
                arr = None
            if arr is None:
                continue
            arr = np.asarray(arr)
            while arr.ndim < 5:
                arr = arr[np.newaxis, ...]
            phys = min(arr.shape[3], kv_len)
            dst[:, :, :, :phys, :] = arr[:, :, :, :phys, :]
        self.t_dc_k_a.set_data_from_numpy(np.ascontiguousarray(k_buf))
        self.t_dc_v_a.set_data_from_numpy(np.ascontiguousarray(v_buf))
        k_in, k_out = self.t_dc_k_a, self.t_dc_k_b
        v_in, v_out = self.t_dc_v_a, self.t_dc_v_b

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and gen[-1] == int(self.eos_token_id):
                break
            if valid >= kv_len:
                truncated = True
                break
            cur_attn[0, valid] = 1
            self.t_dc_ids.set_data_from_numpy(np.array([[gen[-1]]], dtype=self._dc_ids_np))
            self.t_dc_attn.set_data_from_numpy(cur_attn.copy())
            self.t_dc_pos.set_data_from_numpy(np.array([[valid]], dtype=self._dc_pos_np))
            inputs = [self.t_dc_ids, self.t_dc_attn, self.t_dc_pos, k_in, v_in]
            outputs = [self.t_dc_logits, k_out, v_out]
            t0 = time.perf_counter()
            self.decode.predict(inputs, outputs=outputs)
            step_ms.append((time.perf_counter() - t0) * 1000)
            lg = self.t_dc_logits.get_data_to_numpy()
            k_in, k_out = k_out, k_in
            v_in, v_out = v_out, v_in
            valid += 1
            nid = int(np.argmax(lg[0, -1, :])) if lg.ndim == 3 else int(np.argmax(lg.reshape(-1)))
            gen.append(nid)
        return gen, step_ms, truncated

class DynamicBucketRunner(_DecodeLoopMixin):
    """Single dynamic prefill graph + single dynamic decode graph, SHARE_WEIGHT, resize bucket selection, per-bucket KV slicing."""

    def __init__(self, device_id, tokenizer, phase="both"):
        """phase: 'both' normal inference; 'prefill'/'decode' builds only the corresponding model (for per-process prof collection)."""
        self.device_id = int(device_id)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.phase = phase
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = self.device_id
        ctx.ascend.provider = "ge"
        t0 = time.perf_counter()
        if phase in ("both", "prefill"):
            self.prefill = mslite.Model()
            self.prefill.build_from_file(PREFILL_PATH, mslite.ModelType.MINDIR, ctx, PREFILL_CFG)
            self._pf_build_s = time.perf_counter() - t0
            print(f"[build] prefill={self._pf_build_s:.1f}s", flush=True)
            print(f"[build] prefill inputs={[list(t.shape) for t in self.prefill.get_inputs()]}",
                  flush=True)
        if phase in ("both", "decode"):
            self.decode = mslite.Model()
            self.decode.build_from_file(DECODE_PATH, mslite.ModelType.MINDIR, ctx, DECODE_CFG)
            self._build_s = time.perf_counter() - t0
            print(f"[build] decode total={self._build_s:.1f}s", flush=True)
            print(f"[build] decode inputs={[list(t.shape) for t in self.decode.get_inputs()]}",
                  flush=True)
        if phase == "both":
            self._mg = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
            self._mg.add_model([self.prefill, self.decode])
            print(f"[build] both graphs total={self._build_s:.1f}s (SHARE_WEIGHT)", flush=True)

        self.max_seq = BUCKET_SEQ_DIMS[-1]
        self.max_kv_len = self.max_seq + MAX_OUTPUT_TOKENS
        dev = f"ascend:{self.device_id}"

        self._pf_kv_dtype = mslite.DataType.FLOAT16
        self.t_pf_logits = mslite.Tensor(
            shape=[VOCAB], dtype=mslite.DataType.FLOAT32, device=dev)
        self.t_pf_k = mslite.Tensor(
            shape=[NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM],
            dtype=self._pf_kv_dtype, device=dev)
        self.t_pf_v = mslite.Tensor(
            shape=[NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM],
            dtype=self._pf_kv_dtype, device=dev)
        if phase in ("both", "prefill"):
            _ = self.t_pf_k.get_data_to_numpy()
            _ = self.t_pf_v.get_data_to_numpy()

        if phase in ("both", "decode"):
            din = self.decode.get_inputs()
            self._dc_ids_np = MS_DTYPE_TO_NP[din[0].dtype]
            self._dc_attn_np = MS_DTYPE_TO_NP[din[1].dtype]
            self._dc_pos_np = MS_DTYPE_TO_NP[din[2].dtype]
            self._dc_kv_np = MS_DTYPE_TO_NP[din[3].dtype]
            kv_shape = [NUM_LAYERS, 1, NUM_KV_HEADS, self.max_kv_len, HEAD_DIM]
            self.t_dc_ids = mslite.Tensor(shape=[1, 1], dtype=din[0].dtype, device=dev)
            self.t_dc_attn = mslite.Tensor(shape=[1, self.max_kv_len], dtype=din[1].dtype, device=dev)
            self.t_dc_pos = mslite.Tensor(shape=[1, 1], dtype=din[2].dtype, device=dev)
            self.t_dc_k_a = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
            self.t_dc_v_a = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
            self.t_dc_k_b = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
            self.t_dc_v_b = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
            self.t_dc_logits = mslite.Tensor(
                shape=[VOCAB], dtype=mslite.DataType.FLOAT32, device=dev)

    def _prep_inputs(self, text):
        """Tokenize and pad prompt → (padded, attn, pos, real, seq, kv_len)."""
        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, return_tensors="np")
        ids = np.asarray(enc["input_ids"] if hasattr(enc, "keys") else enc, dtype=np.int32).reshape(1, -1)
        real = int(ids.shape[1])
        seq = _pick_seq(real)
        kv_len = seq + MAX_OUTPUT_TOKENS
        padded = np.zeros((1, seq), np.int32)
        padded[0, :real] = ids[0, :real]
        attn = np.zeros((1, seq), np.int32)
        attn[0, :real] = 1
        cum = np.cumsum(attn[0], dtype=np.int32) - 1
        pos = np.where(attn[0] > 0, cum, 0).astype(np.int32)[None, :]
        return padded, attn, pos, real, seq, kv_len

    def _prefill(self, padded, attn, pos, seq, kv_len):
        """Run prefill with preallocated device output buffers and return first token + KV tensors."""
        self.prefill.resize(self.prefill.get_inputs(), [[1, seq], [1, seq], [1, seq]])
        feed = build_mslite_inputs(
            self.prefill, {"input_ids": padded, "attention_mask": attn, "position_ids": pos},
            preferred_order=["input_ids", "attention_mask", "position_ids"])

        kv_shape = [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]
        self.t_pf_k.shape = kv_shape
        self.t_pf_v.shape = kv_shape

        t0 = time.perf_counter()
        self.prefill.predict(feed, outputs=[self.t_pf_logits, self.t_pf_k, self.t_pf_v])
        pf_ms = (time.perf_counter() - t0) * 1000

        # logits 小张量 D2H 做 argmax
        logits = self.t_pf_logits.get_data_to_numpy().astype(np.float32)
        flat = logits.reshape((-1, VOCAB))
        if flat.shape[0] > 1:
            real = int(attn.sum())
            first = int(np.argmax(flat[real - 1, :]))
        else:
            first = int(np.argmax(flat[0, :]))

        kv_out_phys = kv_len
        return first, self.t_pf_k, self.t_pf_v, pf_ms, kv_out_phys

    def run_prefill(self, text):
        """Run prefill only (for prof mode). Returns perf dict."""
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        _, _, _, pf_ms, kv_phys = self._prefill(padded, attn, pos, seq, kv_len)
        return {"real_len": real, "prefill_seq": seq, "kv_len": kv_len,
                "prefill_ms": round(pf_ms, 1), "kv_out_phys": kv_phys}

    def run_decode_only(self, seq, kv_len, max_new_tokens=32):
        """Run decode only with dummy (zero) KV — for per-process prof decode collection.

        Does not build/run prefill: KV is zero-filled and the decode process contains only decode ops,
        avoiding mixing prefill and decode data (profiling is per-process).
        Returns perf dict.
        """
        din = self.decode.get_inputs()
        self.decode.resize(din, [[1, 1], [1, kv_len], [1, 1],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM],
                                 [NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM]])
        ids_np = MS_DTYPE_TO_NP[din[0].dtype]
        attn_np = MS_DTYPE_TO_NP[din[1].dtype]
        pos_np = MS_DTYPE_TO_NP[din[2].dtype]
        kv_np = MS_DTYPE_TO_NP[din[3].dtype]
        first = 1
        valid = seq
        cur_attn = np.zeros((1, kv_len), dtype=attn_np)
        cur_attn[0, :valid] = 1
        gen = [first]
        step_ms = []
        truncated = False

        dc_buf = self._dc_buffers[kv_len]
        t_ids, t_attn, t_pos = dc_buf["ids"], dc_buf["attn"], dc_buf["pos"]
        t_k_a, t_v_a = dc_buf["k_a"], dc_buf["v_a"]
        t_k_b, t_v_b = dc_buf["k_b"], dc_buf["v_b"]
        t_logits = dc_buf["logits"]
        dummy_k = np.zeros([NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM], dtype=kv_np)
        dummy_v = np.zeros([NUM_LAYERS, 1, NUM_KV_HEADS, kv_len, HEAD_DIM], dtype=kv_np)
        t_k_a.set_data_from_numpy(np.ascontiguousarray(dummy_k))
        t_v_a.set_data_from_numpy(np.ascontiguousarray(dummy_v))
        k_in, k_out = t_k_a, t_k_b
        v_in, v_out = t_v_a, t_v_b

        for _ in range(max_new_tokens - 1):
            if self.eos_token_id is not None and gen[-1] == int(self.eos_token_id):
                break
            if valid >= kv_len:
                truncated = True
                break
            cur_attn[0, valid] = 1
            t_ids.set_data_from_numpy(np.array([[gen[-1]]], dtype=ids_np))
            t_attn.set_data_from_numpy(cur_attn.copy())
            t_pos.set_data_from_numpy(np.array([[valid]], dtype=pos_np))
            inputs = [t_ids, t_attn, t_pos, k_in, v_in]
            outputs = [t_logits, k_out, v_out]
            t0 = time.perf_counter()
            o = self.decode.predict(inputs, outputs=outputs)
            step_ms.append((time.perf_counter() - t0) * 1000)
            lg = o[0].get_data_to_numpy()
            k_in, k_out = k_out, k_in
            v_in, v_out = v_out, v_in
            valid += 1
            nid = int(np.argmax(lg[0, -1, :])) if lg.ndim == 3 else int(np.argmax(lg.reshape(-1)))
            gen.append(nid)
        return {"prefill_seq": seq, "kv_len": kv_len,
                "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
                "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
                "decode_steps": len(step_ms), "output_len": len(gen), "truncated": truncated}

    def run_decode(self, text, max_new_tokens=32):
        """Run prefill (for KV) + decode only (for prof mode). Returns perf dict."""
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        first, pf_k, pf_v, _, _ = self._prefill(padded, attn, pos, seq, kv_len)
        gen, step_ms, trunc = self._decode(first, pf_k, pf_v, real, kv_len, max_new_tokens)
        return {"real_len": real, "prefill_seq": seq, "kv_len": kv_len,
                "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
                "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
                "decode_steps": len(step_ms), "output_len": len(gen), "truncated": trunc}

    def run(self, text, max_new_tokens=32):
        padded, attn, pos, real, seq, kv_len = self._prep_inputs(text)
        first, pf_k, pf_v, pf_ms, kv_phys = self._prefill(padded, attn, pos, seq, kv_len)
        gen, step_ms, trunc = self._decode(first, pf_k, pf_v, real, kv_len, max_new_tokens)
        txt = self.tokenizer.decode(gen, skip_special_tokens=True)
        return txt, {
            "real_len": real, "prefill_seq": seq, "kv_len": kv_len,
            "kv_out_phys": kv_phys, "kv_padded_to": kv_len,
            "kv_len_ok": (pf_k.shape[3] >= kv_len),
            "prefill_ms": round(pf_ms, 1),
            "decode_steps": len(step_ms),
            "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
            "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
            "decode_tok_s": round(1000.0 / min(step_ms), 2) if step_ms else 0.0,
            "output_len": len(gen), "truncated": trunc,
            "output_preview": txt[:80].replace("\n", " "),
            "generated_ids": [int(x) for x in gen],
        }


class CommonPrefixRunner(_DecodeLoopMixin):
    """1P prefix-cache runner; the prefix graph executes once per system prompt."""

    def __init__(self, device_id, tokenizer, common_prefix, prefix_role="system",
                 rank_id=0, prefix_model=None, suffix_model=None,
                 decode_model=None, prefix_config=None, suffix_config=None,
                 decode_config=None, num_kv_heads=NUM_KV_HEADS):
        self.device_id = int(device_id)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.phase = "both"
        self.num_kv_heads = int(num_kv_heads)
        ctx = mslite.Context()
        ctx.target = ["ascend"]
        ctx.ascend.device_id = self.device_id
        ctx.ascend.rank_id = int(rank_id)
        ctx.ascend.provider = "ge"
        self.prefix = mslite.Model()
        self.suffix = mslite.Model()
        self.decode = mslite.Model()
        self._mg = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT)
        self._mg.add_model([self.prefix, self.suffix, self.decode])
        self.prefix.build_from_file(
            prefix_model, mslite.ModelType.MINDIR, ctx, prefix_config)
        self.suffix.build_from_file(
            suffix_model, mslite.ModelType.MINDIR, ctx, suffix_config)
        self.decode.build_from_file(
            decode_model, mslite.ModelType.MINDIR, ctx, decode_config)

        self.max_seq = COMMON_PREFIX_BUCKET + COMMON_SUFFIX_BUCKET
        self.max_kv_len = self.max_seq + MAX_OUTPUT_TOKENS
        dev = f"ascend:{self.device_id}"
        suffix_out = self.suffix.get_outputs()
        prefix_kv_shape = [
            NUM_LAYERS, 1, self.num_kv_heads, COMMON_PREFIX_BUCKET, HEAD_DIM]
        suffix_kv_shape = [
            NUM_LAYERS, 1, self.num_kv_heads, self.max_kv_len, HEAD_DIM]
        # GE dynamic graphs expose the active logical output shape, while the
        # zero-copy buffer must cover the maximum configured bucket.  The
        # converted graph descriptor may report FP32 for KV even though GE's
        # force_fp16 graph physically produces/consumes FP16 KV, so do not use
        # get_outputs()[].dtype for these buffers.
        self.t_prefix_k = mslite.Tensor(
            shape=prefix_kv_shape, dtype=mslite.DataType.FLOAT16, device=dev)
        self.t_prefix_v = mslite.Tensor(
            shape=prefix_kv_shape, dtype=mslite.DataType.FLOAT16, device=dev)
        self.t_suffix_logits = mslite.Tensor(
            shape=[1, 1, VOCAB], dtype=suffix_out[0].dtype, device=dev)
        self.t_suffix_k = mslite.Tensor(
            shape=suffix_kv_shape, dtype=mslite.DataType.FLOAT16, device=dev)
        self.t_suffix_v = mslite.Tensor(
            shape=suffix_kv_shape, dtype=mslite.DataType.FLOAT16, device=dev)
        din = self.decode.get_inputs()
        self._dc_ids_np = MS_DTYPE_TO_NP[din[0].dtype]
        self._dc_attn_np = MS_DTYPE_TO_NP[din[1].dtype]
        self._dc_pos_np = MS_DTYPE_TO_NP[din[2].dtype]
        self._dc_kv_np = MS_DTYPE_TO_NP[din[3].dtype]
        kv_shape = [NUM_LAYERS, 1, self.num_kv_heads, self.max_kv_len, HEAD_DIM]
        self.t_dc_ids = mslite.Tensor(shape=[1, 1], dtype=din[0].dtype, device=dev)
        self.t_dc_attn = mslite.Tensor(
            shape=[1, self.max_kv_len], dtype=din[1].dtype, device=dev)
        self.t_dc_pos = mslite.Tensor(shape=[1, 1], dtype=din[2].dtype, device=dev)
        self.t_dc_k_a = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
        self.t_dc_v_a = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
        self.t_dc_k_b = mslite.Tensor(shape=kv_shape, dtype=din[3].dtype, device=dev)
        self.t_dc_v_b = mslite.Tensor(shape=kv_shape, dtype=din[4].dtype, device=dev)
        self.t_dc_logits = mslite.Tensor(
            shape=[VOCAB], dtype=mslite.DataType.FLOAT32, device=dev)
        self.prefix_role = prefix_role
        self._prepare_prefix(common_prefix)

    @staticmethod
    def _ids(encoded):
        value = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return np.asarray(value, dtype=np.int32).reshape(1, -1)

    def _prepare_prefix(self, text):
        """Cache the common prefix text and optionally build the prefix KV now."""
        self.common_prefix_text = text
        if self.prefix_role == "user":
            # Token boundaries can merge across "prefix + suffix". Defer the
            # one-time prefix build until the first complete user request lets
            # us cut at an exact offset-mapped token boundary.
            self._prefix_ids_raw = None
            self.prefix_k = self.prefix_v = None
            self.prefix_actual_len = 0
            self.prefix_ms = 0.0
            return
        messages = [{"role": "system", "content": text}]
        ids = self._ids(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_tensors="np"))
        self._cache_prefix_ids(ids)

    def _cache_prefix_ids(self, ids):
        """Run the prefix graph once and keep its KV on device for reuse."""
        self._prefix_ids_raw = ids.copy()
        self.prefix_actual_len = int(ids.shape[1])
        if self.prefix_actual_len > COMMON_PREFIX_BUCKET:
            raise ValueError(
                f"common prefix has {self.prefix_actual_len} tokens; maximum is "
                f"{COMMON_PREFIX_BUCKET}")
        pad_id = int(self.tokenizer.pad_token_id)
        pids = np.full((1, COMMON_PREFIX_BUCKET), pad_id, np.int32)
        pids[:, :self.prefix_actual_len] = ids
        pmask = np.zeros((1, COMMON_PREFIX_BUCKET), np.int32)
        pmask[:, :self.prefix_actual_len] = 1
        ppos = np.zeros((1, COMMON_PREFIX_BUCKET), np.int32)
        ppos[:, :self.prefix_actual_len] = np.arange(
            self.prefix_actual_len, dtype=np.int32)
        self.prefix.resize(self.prefix.get_inputs(), [
            [1, COMMON_PREFIX_BUCKET], [1, COMMON_PREFIX_BUCKET],
            [1, COMMON_PREFIX_BUCKET]])
        feed = build_mslite_inputs(
            self.prefix, {"input_ids": pids, "attention_mask": pmask,
                          "position_ids": ppos},
            preferred_order=["input_ids", "attention_mask", "position_ids"])
        outputs = None
        t0 = 0.0
        for repeat in range(4):
            if repeat == 3:
                t0 = time.perf_counter()
            outputs = self.prefix.predict(
                feed, outputs=[self.t_prefix_k, self.t_prefix_v])
        self.prefix_ms = (time.perf_counter() - t0) * 1000
        self.prefix_k, self.prefix_v = outputs[0], outputs[1]
        print(f"[prefix cache] actual={self.prefix_actual_len}, "
              f"bucket={COMMON_PREFIX_BUCKET}, repeats=4, "
              f"last_latency={self.prefix_ms:.1f}ms",
              flush=True)

    def _prepare_suffix(self, user_text):
        """Tokenize the user suffix, split it from the cached prefix, and pad to the suffix bucket."""
        if self.prefix_role == "user":
            messages = [{"role": "user",
                         "content": self.common_prefix_text + user_text}]
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer(
                rendered, add_special_tokens=False,
                return_offsets_mapping=True)
            full = np.asarray(
                encoded["input_ids"], dtype=np.int32).reshape(1, -1)
            content_start = rendered.find(self.common_prefix_text + user_text)
            if content_start < 0:
                raise ValueError("cannot locate user content in rendered chat template")
            boundary = content_start + len(self.common_prefix_text)
            split = 0
            for i, offset in enumerate(encoded["offset_mapping"]):
                if offset[1] <= boundary:
                    split = i + 1
                else:
                    break
            if split <= 0:
                raise ValueError("common prefix does not end on a usable token boundary")
            if self._prefix_ids_raw is None:
                self._cache_prefix_ids(full[:, :split])
        else:
            messages = [{"role": "system", "content": self.common_prefix_text},
                        {"role": "user", "content": user_text}]
            full = self._ids(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="np"))
        # Prefer an exact token-prefix split. Some tokenizers do not round-trip
        # decoded system text identically; fail loudly instead of duplicating it.
        plen = self.prefix_actual_len
        if full.shape[1] < plen or not np.array_equal(full[:, :plen], self._prefix_ids_raw):
            raise ValueError("chat template does not preserve the cached system prefix")
        ids = full[:, plen:]
        real = int(ids.shape[1])
        if real <= 0 or real > COMMON_SUFFIX_BUCKET:
            raise ValueError(
                f"suffix has {real} tokens; supported range is 1..{COMMON_SUFFIX_BUCKET}")
        pad = COMMON_SUFFIX_BUCKET - real
        pad_id = int(self.tokenizer.pad_token_id)
        sids = np.full((1, COMMON_SUFFIX_BUCKET), pad_id, np.int32)
        sids[:, pad:] = ids
        full_mask = np.zeros(
            (1, COMMON_PREFIX_BUCKET + COMMON_SUFFIX_BUCKET), np.int32)
        full_mask[:, :plen] = 1
        full_mask[:, COMMON_PREFIX_BUCKET + pad:] = 1
        spos = np.zeros((1, COMMON_SUFFIX_BUCKET), np.int32)
        spos[:, pad:] = np.arange(plen, plen + real, dtype=np.int32)
        return sids, full_mask, spos, real, pad

    def run(self, text, max_new_tokens=32):
        """Run suffix prefill (reusing cached prefix KV) and the decode loop."""
        sids, mask, spos, suffix_real, suffix_pad = self._prepare_suffix(text)
        self.suffix.resize(self.suffix.get_inputs(), [
            [1, COMMON_SUFFIX_BUCKET],
            [1, COMMON_PREFIX_BUCKET + COMMON_SUFFIX_BUCKET],
            [1, COMMON_SUFFIX_BUCKET],
            [NUM_LAYERS, 1, self.num_kv_heads, COMMON_PREFIX_BUCKET, HEAD_DIM],
            [NUM_LAYERS, 1, self.num_kv_heads, COMMON_PREFIX_BUCKET, HEAD_DIM]])
        small = build_mslite_inputs(
            self.suffix, {"input_ids": sids, "attention_mask": mask,
                          "position_ids": spos},
            preferred_order=["input_ids", "attention_mask", "position_ids"])
        out = None
        t0 = 0.0
        for repeat in range(4):
            if repeat == 3:
                t0 = time.perf_counter()
            out = self.suffix.predict(
                small + [self.prefix_k, self.prefix_v],
                outputs=[self.t_suffix_logits, self.t_suffix_k,
                         self.t_suffix_v])
        suffix_ms = (time.perf_counter() - t0) * 1000
        logits = out[0].get_data_to_numpy()
        first = int(np.argmax(logits.reshape(-1, VOCAB)[-1]))

        # Compact right-padded prefix + left-padded suffix into the contiguous
        # cache layout expected by decode, then reserve MAX_OUTPUT_TOKENS slots.
        valid = self.prefix_actual_len + suffix_real
        kv_len = self.max_kv_len
        compact = []
        suffix_start = COMMON_PREFIX_BUCKET + suffix_pad
        for src in out[1:3]:
            arr = src.get_data_to_numpy()
            buf = np.zeros(
                [NUM_LAYERS, 1, self.num_kv_heads, kv_len, HEAD_DIM],
                dtype=self._dc_kv_np)
            buf[:, :, :, :self.prefix_actual_len, :] = \
                arr[:, :, :, :self.prefix_actual_len, :]
            buf[:, :, :, self.prefix_actual_len:valid, :] = \
                arr[:, :, :, suffix_start:suffix_start + suffix_real, :]
            compact.append(buf)
        gen, step_ms, trunc = self._decode(
            first, compact[0], compact[1], valid, kv_len, max_new_tokens)
        decoded = self.tokenizer.decode(gen, skip_special_tokens=True)
        prefix_ms = round(self.prefix_ms, 1)
        suffix_ms_rounded = round(suffix_ms, 1)
        return decoded, {
            "prefix_actual_len": self.prefix_actual_len,
            "prefix_bucket": COMMON_PREFIX_BUCKET,
            "suffix_actual_len": suffix_real,
            "suffix_bucket": COMMON_SUFFIX_BUCKET,
            "prefix_ms_once": prefix_ms,
            "suffix_ms": suffix_ms_rounded,
            "prefill_total_ms": round(prefix_ms + suffix_ms_rounded, 1),
            "decode_first_ms": round(step_ms[0], 1) if step_ms else 0.0,
            "decode_min_ms": round(min(step_ms), 1) if step_ms else 0.0,
            "decode_avg_ms": round(sum(step_ms) / len(step_ms), 1) if step_ms else 0.0,
            "decode_total_ms": round(sum(step_ms), 1),
            "decode_steps": len(step_ms),
            "output_len": len(gen), "truncated": trunc,
            "generated_ids": [int(x) for x in gen],
        }



def _worker(rank, device_id, args, barrier, result_q):
    """TP worker: build one CommonPrefixRunner and stream (rank, text, perf, error) back."""
    try:
        os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] = "21500-21600"
        impl = sys.modules[__name__]

        # A TP=2 rank owns half of the eight KV heads.  Reuse the thoroughly
        # tested common-prefix runner while replacing its per-rank graph data.
        base = args.model_dir
        prefix_model = f"{base}/rank{rank}/prefix/qwen3_8b_prefix_rank{rank}_graph.mindir"
        suffix_model = f"{base}/rank{rank}/suffix/qwen3_8b_suffix_rank{rank}_graph.mindir"
        decode_model = f"{base}/rank{rank}/decode/qwen3_8b_llm_decode_rank{rank}_graph.mindir"

        tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        runner = impl.CommonPrefixRunner(
            device_id, tok, args.prefix, prefix_role="user", rank_id=rank,
            prefix_model=prefix_model, suffix_model=suffix_model,
            decode_model=decode_model, prefix_config=args.prefix_cfgs[rank],
            suffix_config=args.suffix_cfgs[rank],
            decode_config=args.decode_cfgs[rank],
            num_kv_heads=NUM_KV_HEADS // args.tp_size)
        barrier.wait()
        text, perf = runner.run(args.suffix, args.max_new_tokens)
        result_q.put((rank, text, perf, None))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError,
            IndexError, AttributeError):
        result_q.put((rank, None, None, traceback.format_exc()))


def run_common_prefix_tp(args):
    """Run multi-rank common-prefix inference for a compatible namespace."""
    devices = [int(x) for x in args.device_ids.split(",")]
    if len(devices) not in (2, 4):
        raise ValueError("common-prefix TP requires two or four device IDs")
    args.tp_size = len(devices)

    # Keep stable graph configuration under configs/tp2.  HCCL topology is
    # runtime state, so merge the current rank table into per-rank temporary
    # configs instead of hard-coding rank_table_file in those base files.
    run_dir = os.path.join(os.getcwd(), "tp_run")
    cfg_dir = args.config_dir
    def runtime_cfgs(base_name, tag):
        return [
            _write_hccl_config_with_ge(
                devices, run_dir, os.path.join(cfg_dir, base_name),
                f"common_{tag}_r{rank}", cache_suffix=str(rank))
            for rank in range(args.tp_size)
        ]
    args.prefix_cfgs = runtime_cfgs(
        "qwen3_8b_llm_prefill_prefix.config", "prefix")
    args.suffix_cfgs = runtime_cfgs(
        "qwen3_8b_llm_prefill_suffix.config", "suffix")
    args.decode_cfgs = runtime_cfgs(
        "qwen3_8b_llm_decode.config", "decode")

    mp.set_start_method("spawn", force=True)
    barrier = mp.Barrier(args.tp_size)
    result_q = mp.Queue()
    workers = [mp.Process(target=_worker,
                          args=(rank, dev, args, barrier, result_q))
               for rank, dev in enumerate(devices)]
    for proc in workers:
        proc.start()
    results = [result_q.get() for _ in workers]
    for proc in workers:
        proc.join()
    errors = [f"rank{r}:\n{err}" for r, _, _, err in results if err]
    if errors:
        raise RuntimeError("\n".join(errors))
    results.sort()
    if results[0][1] != results[1][1]:
        raise RuntimeError("TP ranks produced different token sequences")
    print(results[0][1])
    perf = results[0][2]
    print(f"[TP{args.tp_size} common-prefix]", perf)
    print(f"Prefill: total={perf['prefill_total_ms']} ms "
          f"(prefix={perf['prefix_ms_once']} ms, suffix={perf['suffix_ms']} ms)")
    print(f"Decoder: total={perf['decode_total_ms']} ms, "
          f"steps={perf['decode_steps']}, first={perf['decode_first_ms']} ms, "
          f"avg={perf['decode_avg_ms']} ms, min={perf['decode_min_ms']} ms")



# ===========================================================================
# CLI + dispatch
# ===========================================================================
def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Qwen3-8B inference (MindSpore Lite MindIR on Ascend) — 1p / 2p / 4p auto-dispatch")
    parser.add_argument("--device-ids", "--device-id", dest="device_ids",
                        type=str, required=True,
                        help="comma-separated Ascend device ids (count decides parallelism: 1/2/4)")
    parser.add_argument("--model-id", type=str, default="./Qwen3-8B",
                        help="tokenizer / weights path")
    parser.add_argument("--common-prefix-text", default="你好，")
    parser.add_argument("--suffix-prompt", default="请用一句话介绍一下你自己")
    parser.add_argument("--common-model-dir", required=True,
                        help="common-prefix model root supplied by the caller")
    parser.add_argument("--common-config-dir", default=None,
                        help="config directory; defaults to configs for 1P and configs/tpN for TP")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--json-out", type=str, default=None,
                        help="write perf dict as JSON to this path")
    return parser.parse_args()


def _run_single_chip(args, device_ids):
    """Run the single-chip (1p) dynamic single-graph path.

    The one-device path uses the common-prefix runner defined in this module.
    """
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_dir = args.common_model_dir
    config_dir = args.common_config_dir
    runner = CommonPrefixRunner(
        device_ids[0], tok, args.common_prefix_text, prefix_role="user",
        prefix_model=f"{model_dir}/prefix/qwen3_8b_prefix_rank0_graph.mindir",
        suffix_model=f"{model_dir}/suffix/qwen3_8b_suffix_rank0_graph.mindir",
        decode_model=f"{model_dir}/decode/qwen3_8b_llm_decode_rank0_graph.mindir",
        prefix_config=f"{config_dir}/qwen3_8b_llm_prefill_prefix.config",
        suffix_config=f"{config_dir}/qwen3_8b_llm_prefill_suffix.config",
        decode_config=f"{config_dir}/qwen3_8b_llm_decode.config")
    run_prompt = args.suffix_prompt

    print("\n" + "=" * 60)
    full_prompt = args.common_prefix_text + args.suffix_prompt
    print(f"Input Prompt: {full_prompt}")
    print("=" * 60)
    print("Generated Response: ", end="", flush=True)
    txt, perf = runner.run(run_prompt, max_new_tokens=args.max_new_tokens)
    print(f"\n{txt[:300]}")
    print("\n--- Performance (1p common prefix) ---")
    print(perf)
    print(f"Prefill: total={perf['prefill_total_ms']} ms "
          f"(prefix={perf['prefix_ms_once']} ms, suffix={perf['suffix_ms']} ms)")
    print(f"Decoder: total={perf['decode_total_ms']} ms, "
          f"steps={perf['decode_steps']}, first={perf['decode_first_ms']} ms, "
          f"avg={perf['decode_avg_ms']} ms, min={perf['decode_min_ms']} ms")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tp": 1, "prompt": full_prompt, **perf}, f, indent=2)


def _run_tensor_parallel(args, device_ids, tp_size):
    """Run the tensor-parallel (2p/4p) multi-process path."""
    if tp_size in (2, 4):
        from types import SimpleNamespace
        run_common_prefix_tp(SimpleNamespace(
            device_ids=args.device_ids,
            model_id=args.model_id,
            model_dir=args.common_model_dir,
            prefix=args.common_prefix_text,
            suffix=args.suffix_prompt,
            config_dir=args.common_config_dir,
            max_new_tokens=args.max_new_tokens))
        return
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
        # Per-rank config files carry per-rank GE compile caches
        # (ge.graph_compiler_cache_dir=<base>/rank{r}) so the two ranks never
        # cold-compile the same TBE kernels into one directory.
        cfg_dir = args.bucket_cfg_dir
        if not cfg_dir:
            raise ValueError("--bucket-cfg-dir is required")
        pf_config_files = [
            _write_hccl_config_with_ge(
                device_ids, run_dir, os.path.join(cfg_dir, "tp2/qwen3_8b_llm_prefill.config"),
                f"prefill_r{r}", cache_suffix=str(r)) for r in range(tp_size)]
        dc_config_files = [
            _write_hccl_config_with_ge(
                device_ids, run_dir, os.path.join(cfg_dir, "tp2/qwen3_8b_llm_decode.config"),
                f"decode_r{r}", cache_suffix=str(r)) for r in range(tp_size)]
        seq = kv_len = None
        print(f"[TP2] bucketed dynamicDims cfgs: {pf_config_files[0]} / {dc_config_files[0]} "
              f"(per-rank GE cache under ge_cache/rank{{r}})")
    else:
        # 4p keeps the fixed-shape path (plain HCCL config, no ge_graph_options).
        config_file = args.config_file
        if not config_file:
            raise ValueError("--config-file is required")
        pf_config_files = [config_file] * tp_size
        dc_config_files = [config_file] * tp_size
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
            pf_config_files, dc_config_files, device_ids,
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
    txt, perf = run_tp_infer(
        prefill_ranks, decode_ranks, args.model_id, pf_config_files, dc_config_files,
        args.prompt, args.max_new_tokens, device_ids, warmup=args.warmup,
        stream=not args.json_out, tp_size=tp_size, use_hybrid=None, seq=seq, kv_len=kv_len,
        prompt_tokens=args.prompt_tokens)
    print("result:",txt)
    _print_tp_perf(perf, tp_size)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"tp": tp_size, "output": txt, **perf}, f, indent=2)


def main():
    """Parse args and dispatch to single-chip or tensor-parallel path by device count."""
    # Use 'spawn' (not the default 'fork') so each TP worker is a fresh process
    # that reads HCCL_NPU_SOCKET_PORT_RANGE cleanly at GE init. With 'fork',
    # workers inherit the driver's imported-GE state and the port-range env var
    # is ignored → every rank tries the default NPU adapter port 16666 →
    # "Initialize GE failed ... port 16666 already bound" on the 2nd+ rank.
    mp.set_start_method("spawn", force=True)

    args = _parse_args()
    device_ids = [int(x) for x in args.device_ids.split(",")]
    tp_size = len(device_ids)
    if args.common_config_dir is None:
        args.common_config_dir = (
            "configs" if tp_size == 1 else f"configs/tp{tp_size}")

    print(f"=== TP_SIZE={tp_size}  devices={args.device_ids} ===")
    if tp_size == 1:
        _run_single_chip(args, device_ids)
    elif tp_size in (2, 4):
        _run_tensor_parallel(args, device_ids, tp_size)
    else:
        raise ValueError(f"unsupported device count {tp_size}: use 1, 2, or 4 device ids")


if __name__ == "__main__":
    main()
