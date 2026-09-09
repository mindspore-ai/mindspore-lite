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
Ulysses Sequence Parallel (USP) for Boogu-Image single-stream blocks.

``boost_sp_single_stream`` installs ``sp_single_stream_block_forward`` as the
block forward; the TP counterpart lives in ``tp.py``.

Each rank holds the FULL weights but only a slice of the (padded) joint
sequence. Inside attention, Q/K/V are exchanged with ``all_to_all`` so every
rank computes attention for all sequence positions of its local head subset
(Ulysses), then the results are exchanged back. Only the first block of the
SP region splits the incoming full sequence and only the last block gathers
the full sequence back; the transformer forward between those boundaries
runs on local chunks with no further communication outside attention.

RoPE freqs are sliced to the local chunk before entering the block, so the
shared NPU-safe rotary kernel (float32 real math, cached cos/sin) applies
unchanged.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from lite_boost.model.booguimage.npu_kernels import npu_apply_rotary_emb
from lite_boost.parallel.context_parallel import (
    all_gather_seq,
    all_to_all_4d,
    gqa_kv_head_index,
    pad_split_seq,
)


def _sp_attn(attn, hidden_local, mask_sdpa, rope_local):
    """Ulysses attention for one single-stream block on a local sequence chunk.

    hidden_local: [B, S/P, D]; mask_sdpa: [B, 1, 1, S_pad] or None;
    rope_local:   [B, S/P, D/2] complex or None.
    Returns [B, S/P, D].
    """
    world_size = dist.get_world_size()
    b, sloc, _ = hidden_local.shape

    query = attn.to_q(hidden_local)
    key = attn.to_k(hidden_local)
    value = attn.to_v(hidden_local)

    head_dim = query.shape[-1] // attn.heads
    kv_heads = key.shape[-1] // head_dim

    query = query.view(b, -1, attn.heads, head_dim)
    key = key.view(b, -1, kv_heads, head_dim)
    value = value.view(b, -1, kv_heads, head_dim)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    if rope_local is not None:
        query = npu_apply_rotary_emb(query, rope_local, use_real=False)
        key = npu_apply_rotary_emb(key, rope_local, use_real=False)

    dtype = query.dtype
    query, key = query.to(dtype), key.to(dtype)

    # Ulysses forward: [B, S/P, H, D] -> [B, S_pad, H/P, D]
    query = all_to_all_4d(query, scatter_idx=2, gather_idx=1)
    # K/V hold all heads locally already (weights unsharded under SP); gather
    # the full sequence so every rank can attend over all positions.
    key = all_gather_seq(key, dim=1)    # [B, S_pad, H_kv, D]
    value = all_gather_seq(value, dim=1)

    heads_local = attn.heads // world_size
    qk_ratio = attn.heads // kv_heads

    # After all_to_all the query heads on this rank are the contiguous block
    # [rank*heads_local, (rank+1)*heads_local) of the full head order; index
    # the gathered K/V heads each local query group maps to (GQA).
    kv_idx = gqa_kv_head_index(heads_local, qk_ratio, query.device)
    key = key.index_select(2, kv_idx).transpose(1, 2)    # [B, H/P, S_pad, D]
    value = value.index_select(2, kv_idx).transpose(1, 2)
    query = query.transpose(1, 2)                        # [B, H/P, S_pad, D]

    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask_sdpa, scale=attn.scale,
    )
    out = out.transpose(1, 2).contiguous()  # [B, S_pad, H/P, D]

    # Ulysses reverse: [B, S_pad, H/P, D] -> [B, S/P, H, D]
    out = all_to_all_4d(out, scatter_idx=1, gather_idx=2)

    out = out.reshape(b, sloc, attn.heads * head_dim)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out


def sp_single_stream_block_forward(
    self, hidden_states, attention_mask, image_rotary_emb, temb=None,
):
    """USP replacement for ``BooguImageSingleStreamTransformerBlock.forward``.

    ``hidden_states`` is the FULL joint sequence for the first block of the
    SP region (``_lb_sp_split`` set) and a local chunk afterwards. The output
    is a local chunk, except for the last block (``_lb_sp_gather`` set) which
    returns the gathered full sequence (padding stripped).
    """
    if getattr(self, "enable_taylorseer", False):
        # TaylorSeer skips/replaces block forwards with cached full-sequence
        # results, which breaks the split/gather boundary contract of SP.
        raise NotImplementedError(
            "TaylorSeer is incompatible with sequence parallelism"
        )

    world_size = dist.get_world_size()
    split_input = getattr(self, "_lb_sp_split", False)
    gather_output = getattr(self, "_lb_sp_gather", False)

    if attention_mask is not None and attention_mask.dim() != 2:
        raise ValueError(
            f"SP single-stream forward expects a 2-D [B, S] mask, "
            f"got shape {tuple(attention_mask.shape)}"
        )
    seq_full = attention_mask.shape[1] if attention_mask is not None else hidden_states.shape[1]
    seq_pad = (world_size - seq_full % world_size) % world_size

    if split_input:
        hs, _ = pad_split_seq(hidden_states, seq_pad, dim=1)
    else:
        hs = hidden_states

    if image_rotary_emb is None:
        rope_local = None
    else:
        rope_full = image_rotary_emb
        if seq_pad:
            pad_shape = list(rope_full.shape)
            pad_shape[1] = seq_pad
            rope_full = torch.cat(
                [rope_full, rope_full.new_zeros(pad_shape)], dim=1,
            )
        rope_local, _ = pad_split_seq(rope_full, 0, dim=1)

    if attention_mask is not None:
        mask_bool = attention_mask.bool()
        if seq_pad:
            mask_bool = F.pad(mask_bool, (0, seq_pad), value=False)
        # [B, S_pad] -> [B, 1, 1, S_pad]; the shared NPU SDPA wrapper expands
        # the broadcast mask on NPU before calling the fused kernel.
        mask_sdpa = mask_bool.view(mask_bool.shape[0], 1, 1, -1)
    else:
        mask_sdpa = None

    if self.modulation:
        norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hs, temb)
        attn_output = _sp_attn(self.attn, norm_hidden_states, mask_sdpa, rope_local)
        hs = hs + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
        mlp_output = self.feed_forward(self.ffn_norm1(hs) * (1 + scale_mlp.unsqueeze(1)))
        hs = hs + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
    else:
        norm_hidden_states = self.norm1(hs)
        attn_output = _sp_attn(self.attn, norm_hidden_states, mask_sdpa, rope_local)
        hs = hs + self.norm2(attn_output)
        mlp_output = self.feed_forward(self.ffn_norm1(hs))
        hs = hs + self.ffn_norm2(mlp_output)

    if gather_output:
        hs = all_gather_seq(hs, dim=1)
        if seq_pad:
            hs = hs[:, :seq_full]
    return hs


def boost_sp_single_stream(transformer, world_size=None):
    """Patch the single-stream blocks for Ulysses sequence parallelism.

    The first block of the region splits the incoming full sequence; the
    last block gathers the full sequence back, so the surrounding transformer
    forward and the double-stream stage need no changes. Refiner blocks run
    before the SP region on the full sequence and are left unpatched.
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if world_size <= 1:
        return transformer

    num_heads = transformer.config.num_attention_heads
    if num_heads % world_size != 0:
        raise ValueError(
            f"num_attention_heads ({num_heads}) must be divisible by "
            f"world_size ({world_size})"
        )

    from boogu.models.transformers.transformer_boogu import (
        BooguImageSingleStreamTransformerBlock,
    )
    cls = BooguImageSingleStreamTransformerBlock
    if not getattr(cls, "_lb_sp_patched", False):
        cls._lb_original_forward = cls.forward
        cls.forward = sp_single_stream_block_forward
        cls._lb_sp_patched = True

    layers = list(transformer.single_stream_layers)
    for i, blk in enumerate(layers):
        blk._lb_sp_split = i == 0
        blk._lb_sp_gather = i == len(layers) - 1
    return transformer
