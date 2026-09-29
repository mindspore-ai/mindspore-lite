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
Shared attention bodies for the BooguImage kernels, built from two orthogonal
axes (plus one seam for future overlap patterns):

- *exchanger*: owns the collective exchanges (Q, K, V, out).
  ``IdentityExchanger`` passes tensors through (single-card), ``SyncExchanger``
  runs the Ulysses collectives synchronously (plain SP), and
  ``comm_compute_overlap.py``
  supplies the comm-compute overlapping exchangers that post the same
  collectives on a dedicated comm stream and defer the waits. K and V are
  exchanged separately: the single-stream comm-compute flavor posts gather-K
  right after K is final and lets the V projection fill that window, while
  the double-stream comm-compute flavor stashes K and posts one fused K/V
  gather (its window is filled by the FFN lane anyway).

- *backend*: the attention evaluation itself, taking the post-exchange BSND
  ``[B, S, N, D]`` tensors plus a precomputed ``kv_idx`` so GQA is one
  explicit head-indexing step everywhere (required under Ulysses, where the
  head a2a breaks native broadcast; value-identical to ``enable_gqa``
  broadcast on a single card). ``attention_bnsd`` transposes to [B, N, S, D] and
  calls ``F.scaled_dot_product_attention``; ``attention_bsnd`` feeds
  [B, S, N, D] straight into the fused ``npu_fusion_attention`` with no
  transposes.

- *lane*: the compute that fills a comm-compute exchange window.
  ``FeedForwardLane`` runs the FFN halves in it; a future fused-FFN op is
  another lane with the same two steps, no body changes. Exchangers,
  backends and lanes are model-free, so new overlap combinations and new
  models (the planned shared attention base class) reuse them as-is.

The bodies touch only ``attn`` / ``attn.processor`` attributes (projections,
stream concat/split, per-stream out projections) and the FFN through
``feed_forward.linear_1/3/2`` and ``feed_forward.swiglu`` — never
re-implemented math — so a later Quant pass can swap those linears without
touching the kernels.
"""

import math

import torch.nn.functional as F
from lite_boost.model.booguimage.npu_kernels import npu_fusion_attention_bsnd
from lite_boost.parallel.context_parallel import (
    all_gather_seq,
    all_to_all_4d,
    gqa_kv_head_index,
)

_apply_rotary_emb = None


def resolve_apply_rotary_emb():
    """Lazily resolve boogu's ``apply_rotary_emb`` (NPU-patched by
    ``boost.py``); fall back to the lite_boost NPU kernel."""
    global _apply_rotary_emb
    if _apply_rotary_emb is None:
        try:
            import boogu.models.attention_processor as ap
            _apply_rotary_emb = ap.apply_rotary_emb
        except ImportError:
            from lite_boost.model.booguimage.npu_kernels import npu_apply_rotary_emb
            _apply_rotary_emb = npu_apply_rotary_emb
    return _apply_rotary_emb


def split_heads(x, num_heads):
    """Split the last dim of a projection output: [B, S, H*D] -> [B, S, H, D]."""
    return x.view(x.shape[0], -1, num_heads, x.shape[-1] // num_heads)


def norm_rope_cast(x, norm, rope, rope_fn, dtype):
    """QK-norm (optional) then rotary (optional), cast back to ``dtype``."""
    if norm is not None:
        x = norm(x)
    if rope is not None:
        x = rope_fn(x, rope, use_real=False)
    return x.to(dtype)


def softmax_scale(attn, sequence_length, base_sequence_length=None):
    """NTK-style scaling when ``base_sequence_length`` is set, else ``attn.scale``."""
    if base_sequence_length is not None:
        return math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
    return attn.scale


def fit_mask(mask, target_len):
    """Pad (False) or truncate a bool padding-mask to ``target_len`` columns."""
    cur = mask.shape[-1]
    if cur < target_len:
        return F.pad(mask, (0, target_len - cur), value=False)
    if cur > target_len:
        return mask[..., :target_len]
    return mask


def prepare_mask_4d(mask, batch_size, query_len, crop=False):
    """Bool mask -> 4-D SDPA form (True = keep). ``None`` passes through.

    2-D padding masks broadcast over the query rows; 3-D (e.g. causal)
    masks are unsqueezed; 4-D passes through. With ``crop=True`` the
    result is truncated to ``[query_len, query_len]`` — the SP kernels'
    safety net for self-attention over the padded joint sequence; never
    enable it for cross-attention where K length may differ from Q length.
    """
    if mask is None:
        return None
    mask = mask.bool()
    if mask.dim() == 2:
        mask = mask.view(batch_size, 1, 1, -1).expand(batch_size, 1, query_len, -1)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError(
            f"Attention mask can only be 2D, 3D or 4D, got {mask.dim()}D mask"
        )
    if crop and (mask.shape[-1] != query_len or mask.shape[-2] != query_len):
        mask = mask[..., :query_len, :query_len]
    return mask


def gqa_kv_idx(query, num_heads, kv_heads):
    """KV-head index for each local query head (explicit GQA mapping).

    After the (Ulysses or identity) exchange the local query heads are the
    contiguous block ``[rank * N_loc, (rank+1) * N_loc)`` of the full head
    order (identity = one block covering every head, world size 1); each maps
    to its KV group ``global_q_head // ratio`` — value-identical to the
    ``enable_gqa`` broadcast on a single card. The local head count comes
    from the post-exchange query, so it holds for both exchanger flavors.
    """
    heads_local = query.shape[2]
    ratio = num_heads // kv_heads
    return gqa_kv_head_index(heads_local, ratio, query.device)


def attention_bnsd(query, key, value, kv_idx, mask, scale):
    """Backend: BNSD round-trip — transpose to [B, N, S, D], SDPA, back.

    GQA is the explicit ``kv_idx`` index_select, except ``kv_idx=None``
    (Identity exchanger: query heads in natural order) where SDPA's native
    ``enable_gqa`` broadcast is used instead; returns [B, N, S, D]
    contiguous, ready for the reverse out exchange.
    """
    query = query.transpose(1, 2)
    if kv_idx is not None:
        key = key.index_select(2, kv_idx).transpose(1, 2)
        value = value.index_select(2, kv_idx).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scale,
        )
    else:
        out = F.scaled_dot_product_attention(
            query, key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=mask, scale=scale, enable_gqa=True,
        )
    return out.transpose(1, 2).contiguous()


def attention_bsnd(query, key, value, kv_idx, mask, scale):
    """Backend: fused ``npu_fusion_attention`` straight in BSND (no transposes).

    ``head_num`` is the LOCAL query-head count in the tensors (H/P under SP).
    ``kv_idx=None`` (Identity exchanger: query heads in natural order) skips
    the GQA head expansion and lets the fused kernel broadcast natively -
    the K/V tensors stay at their projected head count instead of being
    materialized up to the query-head count (4x for this model's 28:7 GQA).
    """
    if kv_idx is not None:
        key = key.index_select(2, kv_idx)
        value = value.index_select(2, kv_idx)
    return npu_fusion_attention_bsnd(
        query, key, value, attn_mask=mask, scale=scale, num_heads=query.shape[2],
    )


def project_out(attn, out, head_dim):
    """Flatten heads and run the output projection chain (to_out[0] -> to_out[1])."""
    out = out.reshape(out.shape[0], -1, attn.heads * head_dim)
    out = attn.to_out[0](out)
    if len(attn.to_out) > 1 and attn.to_out[1] is not None:
        out = attn.to_out[1](out)
    return out


def feed_forward_gate_up(feed_forward, ffn_norm1_mod, ffn_norm2_out, ffn_scale_mlp,
                ffn_shift_mlp):
    """Modulated FFN input plus the gate/up projections (first half)."""
    ffn_input = ffn_norm1_mod(
        (1 + ffn_scale_mlp.unsqueeze(1)) * ffn_norm2_out + ffn_shift_mlp.unsqueeze(1)
    )
    return feed_forward.linear_1(ffn_input), feed_forward.linear_3(ffn_input)


def feed_forward_down(feed_forward, ffn_h1, ffn_h2):
    """swiglu + down projection (second FFN half)."""
    return feed_forward.linear_2(feed_forward.swiglu(ffn_h1, ffn_h2))


class FeedForwardLane:
    """The FFN halves as the comm-compute overlap lane: gate/up projections fill the
    window between the collective posts and their waits, swiglu + down
    projection fill the window after the out exchange is posted.

    New overlap patterns (e.g. a fused FFN op replacing linear_1/linear_3)
    provide their own lane with the same two steps; the pipeline bodies and
    exchangers never change.
    """

    def __init__(self, feed_forward, ffn_norm1_mod, ffn_norm2_out,
                 ffn_scale_mlp, ffn_shift_mlp):
        self._args = (feed_forward, ffn_norm1_mod, ffn_norm2_out,
                      ffn_scale_mlp, ffn_shift_mlp)

    def before_wait(self):
        """Runs between the V exchange post and ``wait_qkv``."""
        return feed_forward_gate_up(*self._args)

    def after_post(self, ffn_h1, ffn_h2):
        """Runs between the out exchange post and ``wait_out``."""
        return feed_forward_down(self._args[0], ffn_h1, ffn_h2)


def _feed_forward_lane(feed_forward, ffn_norm1_mod, ffn_norm2_out, ffn_scale_mlp,
              ffn_shift_mlp):
    """``FeedForwardLane`` when an FFN is supplied, else ``None`` (single-card
    processor contract: the block owns the FFN, the body only attends)."""
    if feed_forward is None:
        return None
    return FeedForwardLane(feed_forward, ffn_norm1_mod, ffn_norm2_out, ffn_scale_mlp,
                   ffn_shift_mlp)


class IdentityExchanger:
    """Single-card exchange strategy: every step passes tensors through.

    Handles are already-final tensors; the ``wait_*`` steps pass through.
    Query heads keep their natural order (``natural_head_order``), so the
    backends can take their native GQA broadcast instead of the explicit
    K/V head map.
    """

    natural_head_order = True

    def exchange_q(self, query):
        """Pass the query through."""
        return query

    def exchange_k(self, key):
        """Pass the key through."""
        return key

    def exchange_v(self, value):
        """Pass the value through."""
        return value

    def wait_qkv(self, q_handle, k_handle, v_handle):
        return q_handle, k_handle, v_handle

    def post_out(self, out):
        """Pass the attention output through."""
        return out

    def wait_out(self, out_handle):
        return out_handle

    def finish_out(self, attn, out, head_dim):
        return project_out(attn, out, head_dim)


class SyncExchanger:
    """SP exchange strategy running every collective synchronously on the
    current stream — the plain Ulysses flavor.

    Handles are already-final tensors; the ``wait_*`` steps pass through.
    The head a2a breaks the native GQA broadcast (``natural_head_order``):
    the explicit K/V head map is required.
    """

    natural_head_order = False

    def exchange_q(self, query):
        """[B, S_loc, H, D] -> [B, S_pad, H/P, D] (scatter heads, gather seq)."""
        return all_to_all_4d(query, scatter_idx=2, gather_idx=1)

    def exchange_k(self, key):
        """All-gather K along the sequence dim (weights unsharded under SP)."""
        return all_gather_seq(key, dim=1)

    def exchange_v(self, value):
        """All-gather V along the sequence dim."""
        return all_gather_seq(value, dim=1)

    def wait_qkv(self, q_handle, k_handle, v_handle):
        return q_handle, k_handle, v_handle

    def post_out(self, out):
        """Reverse Ulysses exchange: [B, S_pad, H/P, D] -> [B, S_loc, H, D]."""
        return all_to_all_4d(out, scatter_idx=1, gather_idx=2)

    def wait_out(self, out_handle):
        return out_handle

    def finish_out(self, attn, out, head_dim):
        return project_out(attn, out, head_dim)


def double_stream_joint_out(processor, attn, out, encoder_seq_lengths, seq_lengths):
    """Double-stream joint attention tail: split the joint output per stream
    (processor-canonical), apply the per-stream output projections, merge and
    run the shared to_out chain."""
    instruct_attn_out, img_attn_out = processor._split_instruction_image_features(
        [out], encoder_seq_lengths, seq_lengths,
    )[0]
    hidden_attn = processor._concat_instruction_image_features(
        [processor.img_out(img_attn_out)],
        [processor.instruct_out(instruct_attn_out)],
        encoder_seq_lengths, seq_lengths,
    )[0]
    # hidden_attn spans the full inner dim (H * head_dim); recover head_dim.
    return project_out(attn, hidden_attn, hidden_attn.shape[-1] // attn.heads)


def _attention_core(attn, q_handle, k_handle, value, mask, exchanger, backend,
             kv_heads, scale, ffn_lane=None, crop_mask=False):
    """Shared attention middle, from the K/V exchange posts onward.

    The caller posts the Q exchange (``q_handle``) right after the Q
    projection so the comm-compute flavor can hide its a2a under the K/V
    projections, and posts the K exchange (``k_handle``) right after K is
    final so V's projection fills that window; ``value`` is the local V
    tensor, exchanged inside. This runs the overlap lane's first step in
    that window, waits, runs the backend (explicit GQA head selection +
    attention evaluation), posts the reverse out exchange and runs the
    lane's second step in that window, then waits.
    ``ffn_lane`` fills the two windows (``None`` = attention only);
    ``crop_mask`` truncates the prepared mask to the query width (SP
    self-attention safety net over the padded joint sequence).
    Returns ``(out, lane_out)`` — ``lane_out`` is the FFN result when a lane
    is present, else ``None``.
    """
    v_handle = exchanger.exchange_v(value)
    ffn_h1 = ffn_h2 = None
    if ffn_lane is not None:
        ffn_h1, ffn_h2 = ffn_lane.before_wait()

    query, key, value = exchanger.wait_qkv(q_handle, k_handle, v_handle)
    # Head-order-preserving exchanges can use the backends' native GQA
    # broadcast (identity: skips the K/V head materialization, 4x on this
    # model); the Ulysses head a2a scrambles the head order and needs the
    # explicit map.
    if exchanger.natural_head_order:
        kv_idx = None
    else:
        kv_idx = gqa_kv_idx(query, attn.heads, kv_heads)
    # post-exchange tensors are BSND: seq is dim 1
    mask = prepare_mask_4d(mask, query.shape[0], query.shape[1], crop=crop_mask)
    out = backend(query, key, value, kv_idx, mask, scale).to(query.dtype)

    out_handle = exchanger.post_out(out)
    lane_out = ffn_lane.after_post(ffn_h1, ffn_h2) if ffn_lane is not None else None
    return exchanger.wait_out(out_handle), lane_out


def single_stream_sp_attention(attn, hidden_local, mask_sdpa, rope_local, exchanger,
                          rope_fn, backend=attention_bnsd, base_sequence_length=None,
                          encoder_hidden_states=None):
    """Attention for one single-stream block on a local chunk.

    hidden_local: [B, S, D]; mask_sdpa: [B, 1, 1, S_pad], 2-D, or None;
    rope_local:   [B, S/P, D/2] complex or None. Returns [B, S, D].
    K/V are projected from ``encoder_hidden_states`` (defaults to
    ``hidden_local`` — self-attention, the SP case; boogu's processor passes
    the encoder tensor explicitly).
    """
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_local
    scale = softmax_scale(attn, hidden_local.shape[1], base_sequence_length)

    q_raw = attn.to_q(hidden_local)
    head_dim = q_raw.shape[-1] // attn.heads
    kv_heads = attn.to_k.out_features // head_dim

    query = norm_rope_cast(
        split_heads(q_raw, attn.heads), attn.norm_q, rope_local, rope_fn,
        q_raw.dtype,
    )
    q_handle = exchanger.exchange_q(query)  # overlaps the K/V projections

    k_raw = attn.to_k(encoder_hidden_states)
    key = norm_rope_cast(
        split_heads(k_raw, kv_heads), attn.norm_k, rope_local, rope_fn,
        k_raw.dtype,
    )
    k_handle = exchanger.exchange_k(key)    # overlaps the V projection
    v_raw = attn.to_v(encoder_hidden_states)
    value = split_heads(v_raw, kv_heads)

    out, _ = _attention_core(attn, q_handle, k_handle, value, mask_sdpa, exchanger,
                      backend, kv_heads, scale)
    return exchanger.finish_out(attn, out, head_dim)


def double_stream_joint_sp_attention(attn, img_norm1_out, instruct_norm1_out, joint_mask_full,
                     rope_local, encoder_seq_lengths_local, seq_lengths_local,
                     exchanger, rope_fn, ffn_norm2_out=None, ffn_scale_mlp=None,
                     ffn_shift_mlp=None, feed_forward=None, ffn_norm1_mod=None,
                     backend=attention_bnsd, base_sequence_length=None):
    """Joint attention for one double-stream block (+ img FFN lane when an
    FFN is supplied).

    Concat/split of the two streams goes through the processor's
    ``_concat_instruction_image_features`` /
    ``_split_instruction_image_features`` with the given lengths
    (boogu-canonical, B>1-safe; for B=1 — the SP contract — value-identical
    to a plain cat).
    Returns ``(hidden_attn, mlp_out)``; ``hidden_attn`` is the merged
    [instruct | img] attention output after the per-stream output
    projections and ``to_out``, ``mlp_out`` the img FFN result (``None``
    without a lane).
    """
    processor = attn.processor
    dtype = img_norm1_out.dtype

    img_query = processor.img_to_q(img_norm1_out)
    instruct_query = processor.instruct_to_q(instruct_norm1_out)
    query = processor._concat_instruction_image_features(
        [img_query], [instruct_query],
        encoder_seq_lengths_local, seq_lengths_local,
    )[0]

    head_dim = query.shape[-1] // attn.heads
    kv_heads = processor.img_to_k.out_features // head_dim
    query = norm_rope_cast(
        split_heads(query, attn.heads), attn.norm_q, rope_local, rope_fn, dtype,
    )
    q_handle = exchanger.exchange_q(query)  # overlaps the K/V projections

    key, value = processor._concat_instruction_image_features(
        [processor.img_to_k(img_norm1_out), processor.img_to_v(img_norm1_out)],
        [processor.instruct_to_k(instruct_norm1_out),
         processor.instruct_to_v(instruct_norm1_out)],
        encoder_seq_lengths_local, seq_lengths_local,
    )
    key = norm_rope_cast(
        split_heads(key, kv_heads), attn.norm_k, rope_local, rope_fn, dtype,
    )
    k_handle = exchanger.exchange_k(key)
    value = split_heads(value, kv_heads)

    out, mlp_out = _attention_core(
        attn, q_handle, k_handle, value, joint_mask_full, exchanger, backend,
        kv_heads, softmax_scale(attn, max(seq_lengths_local), base_sequence_length),
        ffn_lane=_feed_forward_lane(feed_forward, ffn_norm1_mod, ffn_norm2_out,
                           ffn_scale_mlp, ffn_shift_mlp),
        crop_mask=True,
    )
    out = out.reshape(img_norm1_out.shape[0], -1,
                      attn.heads * head_dim).type_as(query)
    hidden_attn = double_stream_joint_out(processor, attn, out,
                               encoder_seq_lengths_local, seq_lengths_local)
    return hidden_attn, mlp_out


def double_stream_img_self_sp_attention(attn, img_norm3_out, img_mask_full, rope_local,
                        exchanger, rope_fn, ffn_norm2_out=None,
                        ffn_scale_mlp=None, ffn_shift_mlp=None,
                        feed_forward=None, ffn_norm1_mod=None,
                        backend=attention_bnsd, base_sequence_length=None):
    """Image self-attention for one double-stream block (+ instruct FFN lane
    when an FFN is supplied). Mirrors ``double_stream_joint_sp_attention``; returns
    ``(out_local, mlp_out)``.
    """
    dtype = img_norm3_out.dtype

    q_raw = attn.to_q(img_norm3_out)
    head_dim = q_raw.shape[-1] // attn.heads
    kv_heads = attn.to_k.out_features // head_dim
    query = norm_rope_cast(
        split_heads(q_raw, attn.heads), attn.norm_q, rope_local, rope_fn, dtype,
    )
    q_handle = exchanger.exchange_q(query)

    k_raw = attn.to_k(img_norm3_out)
    key = norm_rope_cast(
        split_heads(k_raw, kv_heads), attn.norm_k, rope_local, rope_fn, dtype,
    )
    k_handle = exchanger.exchange_k(key)
    v_raw = attn.to_v(img_norm3_out)
    value = split_heads(v_raw, kv_heads)

    out, mlp_out = _attention_core(
        attn, q_handle, k_handle, value, img_mask_full, exchanger, backend,
        kv_heads, softmax_scale(attn, img_norm3_out.shape[1], base_sequence_length),
        ffn_lane=_feed_forward_lane(feed_forward, ffn_norm1_mod, ffn_norm2_out,
                           ffn_scale_mlp, ffn_shift_mlp),
        crop_mask=True,
    )
    return project_out(attn, out, head_dim), mlp_out
