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
Comm-compute (CC) overlap exchangers for the Boogu-Image SP attention/FFN.

The SP attention pipelines in ``attention_common.py`` take an *exchanger* that
owns the collective exchanges (Q, K/V, out); this module provides the
overlapping flavors: the same collectives are posted on a dedicated comm
stream as soon as their inputs are ready, and the waits are deferred to
where the pipeline needs the data — so compute lanes (K/V projections +
RoPE, FFN halves, out projection) fill the communication window.

- ``CommComputeAsyncExchanger`` (``sp_attention_overlap``, installed as
  ``usp_single_stream._sp_attention``):
  a2a(Q) overlaps the K/V projections + RoPE and gather-K is posted right
  after K's norm/RoPE so the V projection fills its window; the out exchange
  (reverse a2a) is synchronous — no matmul is independent of it, so
  deferring it would only pipeline the launch.

- ``CommComputeLaunchExchanger`` (``sp_joint_attention_ffn_interleaved`` /
  ``sp_img_self_attention_ffn_interleaved``): the FFN is the overlap lane —
  linear_1/linear_3 run while a2a(Q) and the KV gather are in flight,
  swiglu + linear_2 run while the reverse a2a is in flight. The FFN always
  goes through ``feed_forward.linear_1/linear_3/swiglu/linear_2`` module
  attributes (never re-implemented math) so a later Quant pass can swap
  those linears without touching this file.

All collectives go through ``async_collectives.py``; stream sync uses npu
Events. Callers must run inside an initialized dist group with
world_size > 1 (guarded in ``boost.py``).
"""

import torch

from lite_boost.model.booguimage.attention_common import (
    attention_bnsd,
    double_stream_img_self_sp_attention,
    double_stream_joint_sp_attention,
    project_out,
    resolve_apply_rotary_emb,
    single_stream_sp_attention,
)
from lite_boost.model.booguimage.async_collectives import (
    all_gather_kv_finalize,
    all_gather_kv_launch,
    all_gather_seq_4d_async,
    all_to_all_4d_async,
    all_to_all_4d_finalize,
    all_to_all_4d_launch,
)

_comm_stream = None


def _get_comm_stream():
    global _comm_stream
    if _comm_stream is None:
        _comm_stream = torch.npu.Stream()
    return _comm_stream


class CommComputeAsyncExchanger:
    """Comm-compute overlap for the single-stream pipeline.

    Q and K exchanges are posted on the comm stream as soon as their inputs
    are ready: a2a(Q) hides under the K/V projections + RoPE, and gather-K
    is posted right after K's norm/RoPE so the V projection fills its
    window. V's gather cannot be hidden (it posts after V exists and no
    compute is left before the wait). The out exchange (reverse a2a) is
    synchronous — to_out consumes the a2a's output, so deferring it only
    pipelines the launch.
    """

    natural_head_order = False

    def __init__(self, comm=None):
        self._comm = comm if comm is not None else _get_comm_stream()

    def exchange_q(self, query):
        """Post a2a(Q) on the comm stream; wait in ``wait_qkv``."""
        q_ready = torch.npu.Event()
        q_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(q_ready)
            query_full, q_work = all_to_all_4d_async(
                query, scatter_idx=2, gather_idx=1
            )
            q_post = torch.npu.Event()
            q_post.record()
        return query_full, q_work, q_post

    def exchange_k(self, key):
        """Post gather-K on the comm stream; wait in ``wait_qkv``."""
        k_ready = torch.npu.Event()
        k_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(k_ready)
            key_full, k_work = all_gather_seq_4d_async(key)
            k_post = torch.npu.Event()
            k_post.record()
        return key_full, k_work, k_post

    def exchange_v(self, value):
        """Post gather-V on the comm stream; wait in ``wait_qkv``."""
        v_ready = torch.npu.Event()
        v_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(v_ready)
            value_full, v_work = all_gather_seq_4d_async(value)
            v_post = torch.npu.Event()
            v_post.record()
        return value_full, v_work, v_post

    def wait_qkv(self, q_handle, k_handle, v_handle):
        """Sync the compute stream to the three exchanges.

        The collectives are already waited and post-processed on the comm
        stream (inside the ``*_async`` primitives); the post events recorded
        after them order everything the compute stream consumes.
        """
        query_full, _, q_post = q_handle
        key_full, _, k_post = k_handle
        value_full, _, v_post = v_handle
        torch.npu.current_stream().wait_event(q_post)
        torch.npu.current_stream().wait_event(k_post)
        torch.npu.current_stream().wait_event(v_post)
        return query_full, key_full, value_full

    def post_out(self, out):
        """Reverse out exchange, inline on the current stream."""
        return all_to_all_4d_async(out, scatter_idx=1, gather_idx=2)

    def wait_out(self, out_handle):
        """Wait for the out exchange on the compute stream."""
        out_local, out_work = out_handle
        if out_work is not None:
            out_work.wait()
        return out_local

    def finish_out(self, attn, out, head_dim):
        """Projection chain after ``_attention_core`` already waited the out
        exchange."""
        return project_out(attn, out, head_dim)


class CommComputeLaunchExchanger:
    """Comm-compute overlap for the double-stream pipelines.

    Uses the ``*_launch``/``*_finalize`` split style: the collective is
    posted on the comm stream inside ``exchange_*``/``post_out`` and only
    waited + post-processed in ``wait_*``, letting the FFN lane run in
    between (the pipeline bodies place the FFN halves exactly there).
    """

    natural_head_order = False

    def __init__(self, comm=None):
        self._comm = comm if comm is not None else _get_comm_stream()
        self._key = None

    def exchange_q(self, query):
        q_ready = torch.npu.Event()
        q_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(q_ready)
            q_handle = all_to_all_4d_launch(query, scatter_idx=2, gather_idx=1)
        return q_handle

    def exchange_k(self, key):
        """Stash K; the fused KV gather posts in ``exchange_v`` (one HCCL
        call for both streams — the double-stream choreography keeps the fused
        collective, the FFN lane fills its whole window)."""
        self._key = key

    def exchange_v(self, value):
        kv_ready = torch.npu.Event()
        kv_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(kv_ready)
            kv_handle = all_gather_kv_launch(self._key, value)
        return kv_handle

    def wait_qkv(self, q_handle, k_handle, v_handle):
        del k_handle
        query_full = all_to_all_4d_finalize(*q_handle)
        key_full, value_full = all_gather_kv_finalize(*v_handle)
        return query_full, key_full, value_full

    def post_out(self, out):
        out_ready = torch.npu.Event()
        out_ready.record()
        with torch.npu.stream(self._comm):
            self._comm.wait_event(out_ready)
            out_handle = all_to_all_4d_launch(out, scatter_idx=1, gather_idx=2)
        return out_handle

    def wait_out(self, out_handle):
        return all_to_all_4d_finalize(*out_handle)


def sp_attention_overlap(attn, hidden_local, mask_sdpa, rope_local,
                    base_sequence_length=None, backend=None):
    """Single-stream SP attention with a2a hidden under QKV / out matmuls.

    Drop-in replacement for ``usp_single_stream._sp_attention`` (installed via
    ``usp.set_sp_attention``); same contract: [B, S/P, D] in, [B, S/P, D] out.
    ``base_sequence_length`` is unused here but kept for signature parity.
    """
    return single_stream_sp_attention(
        attn, hidden_local, mask_sdpa, rope_local,
        CommComputeAsyncExchanger(), resolve_apply_rotary_emb(),
        backend=attention_bnsd if backend is None else backend,
    )


def sp_joint_attention_ffn_interleaved(
    attn,
    img_norm1_out,
    instruct_norm1_out,
    joint_mask_full,
    rope_local,
    encoder_seq_lengths_local,
    seq_lengths_local,
    # FFN lane:
    ffn_norm2_out,
    ffn_scale_mlp,
    ffn_shift_mlp,
    feed_forward,
    ffn_norm1_mod,
    backend=None,
):
    """Double-stream joint attention overlapped with the img FFN matmuls.

    Returns ``(hidden_attn, mlp_out)``; ``hidden_attn`` is the merged
    [instruct | img] attention output after the per-stream output
    projections and ``to_out``, ``mlp_out`` the img FFN result. The block
    forward scatters ``hidden_attn`` back into the two streams.
    """
    return double_stream_joint_sp_attention(
        attn, img_norm1_out, instruct_norm1_out, joint_mask_full,
        rope_local, encoder_seq_lengths_local, seq_lengths_local,
        CommComputeLaunchExchanger(), resolve_apply_rotary_emb(),
        ffn_norm2_out=ffn_norm2_out, ffn_scale_mlp=ffn_scale_mlp,
        ffn_shift_mlp=ffn_shift_mlp, feed_forward=feed_forward,
        ffn_norm1_mod=ffn_norm1_mod,
        backend=attention_bnsd if backend is None else backend,
    )


def sp_img_self_attention_ffn_interleaved(
    attn,
    img_norm3_out,
    img_mask_full,
    rope_local,
    # FFN lane:
    ffn_norm2_out,
    ffn_scale_mlp,
    ffn_shift_mlp,
    feed_forward,
    ffn_norm1_mod,
    backend=None,
):
    """Double-stream image self-attention overlapped with the instruct FFN matmuls.

    Mirrors ``sp_joint_attention_ffn_interleaved``; returns ``(out_local,
    mlp_out)`` for the instruct stream FFN.
    """
    return double_stream_img_self_sp_attention(
        attn, img_norm3_out, img_mask_full, rope_local,
        CommComputeLaunchExchanger(), resolve_apply_rotary_emb(),
        ffn_norm2_out=ffn_norm2_out, ffn_scale_mlp=ffn_scale_mlp,
        ffn_shift_mlp=ffn_shift_mlp, feed_forward=feed_forward,
        ffn_norm1_mod=ffn_norm1_mod,
        backend=attention_bnsd if backend is None else backend,
    )
