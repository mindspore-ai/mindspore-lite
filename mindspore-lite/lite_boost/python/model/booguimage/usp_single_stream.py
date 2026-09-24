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

from lite_boost.model.booguimage.attention_common import (
    SyncExchanger,
    attention_bnsd,
    attention_bsnd,
    resolve_apply_rotary_emb,
    single_stream_sp_attention,
)
from lite_boost.parallel.context_parallel import (
    all_gather_seq,
    pad_split_seq,
)


def set_sp_attention(attention_fn):
    """Override the attention kernel used by ``sp_single_stream_block_forward``.

    Used by ``comm_compute_overlap.py`` to swap in the comm-compute overlapping variant;
    pass ``None`` to restore the default Ulysses kernel.
    """
    global _sp_attention
    _sp_attention = attention_fn if attention_fn is not None else _sp_attention_default


def _sp_attention_backend(attn, hidden_local, mask_sdpa, rope_local, backend):
    """Ulysses attention for one single-stream block on a local chunk.

    hidden_local: [B, S/P, D]; mask_sdpa: [B, 1, 1, S_pad] or None;
    rope_local:   [B, S/P, D/2] complex or None. Returns [B, S/P, D].
    """
    return single_stream_sp_attention(
        attn, hidden_local, mask_sdpa, rope_local, SyncExchanger(),
        resolve_apply_rotary_emb(), backend=backend,
    )


def _sp_attention_default(attn, hidden_local, mask_sdpa, rope_local):
    """Ulysses attention for one single-stream block (SDPA backend)."""
    return _sp_attention_backend(attn, hidden_local, mask_sdpa, rope_local, attention_bnsd)


def _sp_attention_bsnd(attn, hidden_local, mask_sdpa, rope_local):
    """Ulysses attention for one single-stream block (fused BSND backend)."""
    return _sp_attention_backend(attn, hidden_local, mask_sdpa, rope_local, attention_bsnd)


# Current single-stream attention kernel; swapped by ``set_sp_attention``.
_sp_attention = _sp_attention_default


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
    # Pad to a multiple of 2*world so S_local is even - HCCL all_to_all hits
    # a ~2x transit-time cliff on odd per-rank sequence counts.
    align = 2 * world_size
    seq_pad = (align - seq_full % align) % align

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
        attn_output = _sp_attention(self.attn, norm_hidden_states, mask_sdpa, rope_local)
        hs = hs + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
        mlp_output = self.feed_forward(self.ffn_norm1(hs) * (1 + scale_mlp.unsqueeze(1)))
        hs = hs + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
    else:
        norm_hidden_states = self.norm1(hs)
        attn_output = _sp_attention(self.attn, norm_hidden_states, mask_sdpa, rope_local)
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
