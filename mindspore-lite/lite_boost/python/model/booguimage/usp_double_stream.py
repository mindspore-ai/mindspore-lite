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
Ulysses Sequence Parallel (USP) for Boogu-Image double-stream blocks.

Counterpart of ``usp_single_stream.py`` for
``BooguImageDoubleStreamTransformerBlock``.
Each rank holds the FULL weights but only a slice of the padded
instruction / image sequences; the joint attention exchanges heads/sequence
via all_to_all exactly like the single-stream case, and the image
self-attention runs on the local image chunk.

Every block re-derives the per-sample local lengths, refits the masks and
rebuilds the rope slices from the FULL arguments the transformer passes to
each block call — only the first block of the stage splits the incoming
sequences (``_lb_sp_split``) and only the last block gathers them back
(``_lb_sp_gather``), mirroring the single-stream contract.

Assumptions (relaxing them requires reworking the split/gather boundary):

- ``batch_size == 1``. The per-sample layout of the joint sequence
  (instruct first, then img, padded per sample) cannot be sequence-sharded
  sample-by-sample; with B=1 the padded joint sequence is exactly
  ``[instruct | img]`` and the flat per-rank split is unambiguous.

- Instruct / img sequence lengths are fixed within one transformer forward
  (true for Boogu-Image: masks/rope are built once per forward). Each
  stream is padded to a multiple of ``2 * world`` so the per-rank chunk
  lengths match across ranks and the local ``[instruct | img]`` split stays
  on rank boundaries.

- The attention masks are padding masks (validity independent of the query
  row): they are refit to the padded width and broadcast over the local
  query rows after the head exchange; they are never sequence-sliced.

TaylorSeer is not supported (same restriction as the single-stream SP).
Enabling ``_lb_sp_cc_overlap`` on a block (set by ``boost.py``) routes the
attention/FFN work through the comm-compute overlapping kernels in
``comm_compute_overlap.py``; ``_lb_sp_bsnd`` picks the fused-attention backend
(``attention_bsnd``) over the SDPA one — the two flags are orthogonal, giving a
2x2 kernel dispatch per block. All flavors share one signature: they
receive the FFN lane inputs and return ``(attn_out, mlp_out)`` so the block
forward is identical in every mode, and the FFN always goes through
``feed_forward.linear_1/linear_3/swiglu/linear_2`` module attributes (never
re-implemented math) so a later Quant pass can swap those linears without
touching this file.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F

from lite_boost.model.booguimage.attention_common import (
    SyncExchanger,
    attention_bnsd,
    attention_bsnd,
    double_stream_img_self_sp_attention,
    double_stream_joint_sp_attention,
    fit_mask,
    resolve_apply_rotary_emb,
)
from lite_boost.parallel.context_parallel import (
    all_gather_seq,
    get_sp_rank,
    get_sp_size,
)


def _make_double_stream_kernels(backend):
    """Build the synchronous Ulysses double-stream kernel pair for one backend."""
    def _sp_joint_attention(attn, img_norm1_out, instruct_norm1_out, joint_mask_full,
                       rope_local, encoder_seq_lengths_local, seq_lengths_local,
                       ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
                       feed_forward, ffn_norm1_mod):
        """Synchronous Ulysses joint attention + img FFN lane.

        Returns ``(hidden_attn, mlp_out)`` where ``hidden_attn`` is the merged
        [instruct | img] attention output after the per-stream output
        projections and ``to_out``, and ``mlp_out`` is the img FFN result.
        """
        return double_stream_joint_sp_attention(
            attn, img_norm1_out, instruct_norm1_out, joint_mask_full, rope_local,
            encoder_seq_lengths_local, seq_lengths_local,
            SyncExchanger(), resolve_apply_rotary_emb(),
            ffn_norm2_out=ffn_norm2_out, ffn_scale_mlp=ffn_scale_mlp,
            ffn_shift_mlp=ffn_shift_mlp, feed_forward=feed_forward,
            ffn_norm1_mod=ffn_norm1_mod, backend=backend,
        )

    def _sp_img_self_attention(attn, img_norm3_out, img_mask_full, rope_local,
                          ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
                          feed_forward, ffn_norm1_mod):
        """Synchronous Ulysses image self-attention + instruct FFN lane.

        Mirrors ``_sp_joint_attention``; returns ``(out_local, mlp_out)`` where
        ``mlp_out`` is the instruct FFN result.
        """
        return double_stream_img_self_sp_attention(
            attn, img_norm3_out, img_mask_full, rope_local,
            SyncExchanger(), resolve_apply_rotary_emb(),
            ffn_norm2_out=ffn_norm2_out, ffn_scale_mlp=ffn_scale_mlp,
            ffn_shift_mlp=ffn_shift_mlp, feed_forward=feed_forward,
            ffn_norm1_mod=ffn_norm1_mod, backend=backend,
        )

    return _sp_joint_attention, _sp_img_self_attention


_double_stream_kernels_bnsd = _make_double_stream_kernels(attention_bnsd)
_double_stream_kernels_bsnd = _make_double_stream_kernels(attention_bsnd)


def _make_cc_double_stream_kernels(backend):
    """Build the comm-compute overlapping double-stream kernel pair for one backend."""
    from lite_boost.model.booguimage.comm_compute_overlap import (
        sp_img_self_attention_ffn_interleaved as img_self_attention_fn,
        sp_joint_attention_ffn_interleaved as joint_attention_fn,
    )

    def _cc_joint_attn(attn, img_norm1_out, instruct_norm1_out, joint_mask_full,
                       rope_local, encoder_seq_lengths_local, seq_lengths_local,
                       ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
                       feed_forward, ffn_norm1_mod):
        """CC joint attention kernel bound to one backend."""
        return joint_attention_fn(
            attn, img_norm1_out, instruct_norm1_out, joint_mask_full, rope_local,
            encoder_seq_lengths_local, seq_lengths_local,
            ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
            feed_forward, ffn_norm1_mod, backend=backend,
        )

    def _cc_img_self_attn(attn, img_norm3_out, img_mask_full, rope_local,
                          ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
                          feed_forward, ffn_norm1_mod):
        """CC image self-attention kernel bound to one backend."""
        return img_self_attention_fn(
            attn, img_norm3_out, img_mask_full, rope_local,
            ffn_norm2_out, ffn_scale_mlp, ffn_shift_mlp,
            feed_forward, ffn_norm1_mod, backend=backend,
        )

    return _cc_joint_attn, _cc_img_self_attn


def _local_seq_lengths(encoder_seq_lengths, seq_lengths, align, world):
    """Per-sample LOCAL chunk lengths for the two streams.

    Each stream is padded independently to ``align``; the local instruct
    chunk is ``instruct_local_len`` long and the local img chunk fills up to
    ``instruct_local_len + img_local_len``. Without explicit per-sample lengths
    (``encoder_seq_lengths is None``) returns ``(None, None)``.
    """
    if encoder_seq_lengths is None:
        return None, None
    encoder_seq_lengths_local = []
    seq_lengths_local = []
    for enc_s, full_s in zip(encoder_seq_lengths, seq_lengths):
        s_instruct_pad = (enc_s + align - 1) // align * align
        s_img = full_s - enc_s
        s_img_pad = (s_img + align - 1) // align * align
        instruct_local_len = s_instruct_pad // world
        img_local_len = s_img_pad // world
        encoder_seq_lengths_local.append(instruct_local_len)
        seq_lengths_local.append(instruct_local_len + img_local_len)
    return encoder_seq_lengths_local, seq_lengths_local


def _fit_local_masks(joint_attention_mask, img_attention_mask, world,
                     encoder_seq_lengths_local, seq_lengths_local):
    """Refit the padding masks to the padded per-rank widths.

    The masks are padding masks covering [instruct | img] (joint) and
    [img] (img-only); they are refit to the padded widths — never
    sequence-sliced. Without per-sample lengths the masks pass through.
    """
    if seq_lengths_local is None:
        return joint_attention_mask, img_attention_mask
    joint_seq_target = seq_lengths_local[0] * world
    if joint_attention_mask is not None:
        joint_attention_mask = fit_mask(joint_attention_mask, joint_seq_target)
    if img_attention_mask is not None:
        img_mask_seq_target = (
            seq_lengths_local[0] - encoder_seq_lengths_local[0]
        ) * world
        img_attention_mask = fit_mask(img_attention_mask, img_mask_seq_target)
    return joint_attention_mask, img_attention_mask


def _split_joint_rope(rotary_emb, encoder_seq_lengths, align, world, rank):
    """Split the joint [instruct | img] rope per stream, pad each to
    ``align`` and keep this rank's chunk.

    The two streams are padded INDEPENDENTLY, so the split must happen per
    stream first — padding the concatenated vector as one would keep the
    img tokens' positions attached to the wrong slots.
    """
    if rotary_emb is None or encoder_seq_lengths is None:
        return rotary_emb
    instruct_rope_full_len = encoder_seq_lengths[0]
    img_rope_full_len = rotary_emb.shape[1] - instruct_rope_full_len
    pad_instruct_rope = (align - instruct_rope_full_len % align) % align
    pad_img_rope = (align - img_rope_full_len % align) % align
    instruct_rope_local_len = (instruct_rope_full_len + pad_instruct_rope) // world
    img_rope_local_len = (img_rope_full_len + pad_img_rope) // world

    instruct_rope = rotary_emb[:, :instruct_rope_full_len]
    img_rope = rotary_emb[:, instruct_rope_full_len:]
    if pad_instruct_rope:
        instruct_rope = F.pad(instruct_rope, (0, 0, 0, pad_instruct_rope), value=0)
    if pad_img_rope:
        img_rope = F.pad(img_rope, (0, 0, 0, pad_img_rope), value=0)
    instruct_rope_local = instruct_rope[
        :, rank * instruct_rope_local_len: (rank + 1) * instruct_rope_local_len
    ]
    img_rope_local = img_rope[
        :, rank * img_rope_local_len: (rank + 1) * img_rope_local_len
    ]
    return torch.cat([instruct_rope_local, img_rope_local], dim=1).contiguous()


def _slice_img_rope(image_rotary_emb, align, world, rank):
    """Pad the img rope to ``align`` and keep this rank's chunk."""
    if image_rotary_emb is None:
        return None
    img_rope_full_len = image_rotary_emb.shape[1]
    pad_img_rope = (align - img_rope_full_len % align) % align
    img_rope_local_len = (img_rope_full_len + pad_img_rope) // world
    if pad_img_rope:
        img_rope_pad = image_rotary_emb.new_zeros(
            image_rotary_emb.shape[0], pad_img_rope, image_rotary_emb.shape[2]
        )
        image_rotary_emb = torch.cat([image_rotary_emb, img_rope_pad], dim=1)
    return image_rotary_emb[
        :, rank * img_rope_local_len: (rank + 1) * img_rope_local_len
    ].contiguous()


def _split_streams(img_hidden_states, instruct_hidden_states, align, world,
                   rank):
    """Pad both streams to ``align`` and keep this rank's chunk."""
    img_full_len = img_hidden_states.shape[1]
    instruct_full_len = instruct_hidden_states.shape[1]

    pad_img = (align - img_full_len % align) % align
    pad_instruct = (align - instruct_full_len % align) % align
    img_local_len = (img_full_len + pad_img) // world
    instruct_local_len = (instruct_full_len + pad_instruct) // world

    if pad_img:
        img_hidden_states = F.pad(
            img_hidden_states, (0, 0, 0, pad_img), value=0
        )
    if pad_instruct:
        instruct_hidden_states = F.pad(
            instruct_hidden_states, (0, 0, 0, pad_instruct), value=0
        )
    img_hidden_states = img_hidden_states[
        :, rank * img_local_len: (rank + 1) * img_local_len
    ].contiguous()
    instruct_hidden_states = instruct_hidden_states[
        :, rank * instruct_local_len: (rank + 1) * instruct_local_len
    ].contiguous()
    return img_hidden_states, instruct_hidden_states


def _gated_add(hidden, gate, norm, out):
    """Gated residual add: ``hidden + tanh(gate) * norm(out)``."""
    return hidden + gate.unsqueeze(1).tanh() * norm(out)


def _scatter_joint_attn_out(block, joint_attn_out, img_hidden_states,
                            instruct_hidden_states, encoder_seq_lengths_local,
                            seq_lengths_local):
    """Scatter the merged [instruct | img] attention output back into the
    two per-stream tensors.

    With B=1 the local chunk is [instruct | img] exactly, so plain slicing
    keeps the streams aligned; the zeros keep the same shape contract as
    the original processor for B>1 layouts.
    """
    batch_local = img_hidden_states.shape[0]
    instruct_local_len = instruct_hidden_states.shape[1]
    img_local_len = img_hidden_states.shape[1]

    instruct_attn_out = instruct_hidden_states.new_zeros(
        batch_local, instruct_local_len, block.hidden_size
    )
    img_attn_out = img_hidden_states.new_zeros(
        batch_local, img_local_len, block.hidden_size
    )
    for i, (enc_len, seq_len) in enumerate(
        zip(encoder_seq_lengths_local, seq_lengths_local)
    ):
        instruct_attn_out[i, :enc_len] = joint_attn_out[i, :enc_len]
        img_attn_out[i, :seq_len - enc_len] = joint_attn_out[i, enc_len:seq_len]
    return instruct_attn_out, img_attn_out


def _modulated_double_stream_forward(
    block, img_hidden_states, instruct_hidden_states, temb,
    joint_attention_mask, img_attention_mask, rotary_emb, image_rotary_emb,
    encoder_seq_lengths_local, seq_lengths_local,
    joint_attention_fn, img_self_attention_fn,
):
    """Attention + FFN tail of one modulated double-stream block.

    The img stream runs the joint attention (with the instruct stream) and
    its own FFN lane; the instruct stream contributes only its normed
    queries/keys to the joint attention and runs its FFN under the img
    self-attention call. Returns the updated ``(img, instruct)`` streams.
    """
    img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = block.img_norm1(
        img_hidden_states, temb
    )
    img_norm2_out, img_shift_mlp, _, _ = block.img_norm2(img_hidden_states, temb)
    img_norm3_out, img_gate_self, _, _ = block.img_norm3(img_hidden_states, temb)

    (
        instruct_norm1_out, instruct_gate_msa, instruct_scale_mlp,
        instruct_gate_mlp,
    ) = block.instruct_norm1(instruct_hidden_states, temb)
    instruct_norm2_out, instruct_shift_mlp, _, _ = block.instruct_norm2(
        instruct_hidden_states, temb
    )

    joint_attn_out, img_mlp_out = joint_attention_fn(
        block.img_instruct_attn,
        img_norm1_out,
        instruct_norm1_out,
        joint_attention_mask,
        rotary_emb,
        encoder_seq_lengths_local,
        seq_lengths_local,
        # img FFN lane:
        img_norm2_out,
        img_scale_mlp,
        img_shift_mlp,
        block.img_feed_forward,
        block.img_ffn_norm1,
    )

    instruct_attn_out, img_attn_out = _scatter_joint_attn_out(
        block, joint_attn_out, img_hidden_states, instruct_hidden_states,
        encoder_seq_lengths_local, seq_lengths_local,
    )

    img_self_attn_out, instruct_mlp_out = img_self_attention_fn(
        block.img_self_attn,
        img_norm3_out,
        img_attention_mask,
        image_rotary_emb,
        # instruct FFN lane:
        instruct_norm2_out,
        instruct_scale_mlp,
        instruct_shift_mlp,
        block.instruct_feed_forward,
        block.instruct_ffn_norm1,
    )

    img_hidden_states = _gated_add(
        img_hidden_states, img_gate_msa, block.img_attn_norm, img_attn_out
    )
    img_hidden_states = _gated_add(
        img_hidden_states, img_gate_self, block.img_self_attn_norm,
        img_self_attn_out
    )
    img_hidden_states = _gated_add(
        img_hidden_states, img_gate_mlp, block.img_ffn_norm2, img_mlp_out
    )

    instruct_hidden_states = _gated_add(
        instruct_hidden_states, instruct_gate_msa, block.instruct_attn_norm,
        instruct_attn_out
    )
    instruct_hidden_states = _gated_add(
        instruct_hidden_states, instruct_gate_mlp, block.instruct_ffn_norm2,
        instruct_mlp_out
    )
    return img_hidden_states, instruct_hidden_states


def _gather_and_strip_outputs(img_hidden_states, instruct_hidden_states,
                              encoder_seq_lengths, seq_lengths,
                              img_full_len_orig, instruct_full_len_orig):
    """All-gather the per-rank chunks and strip the padding.

    Strip lengths come from the transformer-passed per-sample lengths
    (available on every block); the split-block capture is the fallback
    (single-block stages without explicit lengths).
    """
    img_hidden_states = all_gather_seq(img_hidden_states, dim=1)
    instruct_hidden_states = all_gather_seq(instruct_hidden_states, dim=1)
    if encoder_seq_lengths is not None:
        instruct_strip_len = encoder_seq_lengths[0]
        img_strip_len = seq_lengths[0] - encoder_seq_lengths[0]
    else:
        instruct_strip_len = instruct_full_len_orig
        img_strip_len = img_full_len_orig
    if img_strip_len is not None and img_strip_len < img_hidden_states.shape[1]:
        img_hidden_states = img_hidden_states[:, :img_strip_len]
    if (
        instruct_strip_len is not None
        and instruct_strip_len < instruct_hidden_states.shape[1]
    ):
        instruct_hidden_states = instruct_hidden_states[:, :instruct_strip_len]
    return img_hidden_states, instruct_hidden_states


def sp_double_stream_block_forward(
    self, img_hidden_states, instruct_hidden_states, img_attention_mask,
    joint_attention_mask, image_rotary_emb, rotary_emb, temb=None,
    encoder_seq_lengths=None, seq_lengths=None,
):
    """USP replacement for ``BooguImageDoubleStreamTransformerBlock.forward``.

    See the module docstring for the batch_size=1 / fixed-layout assumptions.
    ``_lb_sp_cc_overlap`` (set per block by ``boost.py``) routes attention
    and FFN through the comm-compute overlapping kernels in
    ``comm_compute_overlap.py`` instead of the synchronous kernels in this module.
    """
    if getattr(self, "enable_taylorseer", False):
        raise NotImplementedError(
            "TaylorSeer is incompatible with sequence parallelism"
        )
    if img_hidden_states.shape[0] != 1:
        raise ValueError(
            "Double-stream SP assumes batch_size == 1, got "
            f"{img_hidden_states.shape[0]}; see usp_double_stream.py module docstring"
        )

    world = get_sp_size()
    rank = get_sp_rank()
    split_input = getattr(self, "_lb_sp_split", False)
    gather_output = getattr(self, "_lb_sp_gather", False)
    bsnd = getattr(self, "_lb_sp_bsnd", False)
    if getattr(self, "_lb_sp_cc_overlap", False):
        joint_attention_fn, img_self_attention_fn = _make_cc_double_stream_kernels(
            attention_bsnd if bsnd else attention_bnsd)
    else:
        joint_attention_fn, img_self_attention_fn = (
            _double_stream_kernels_bsnd if bsnd else _double_stream_kernels_bnsd)

    img_full_len_orig = img_hidden_states.shape[1] if split_input else None
    instruct_full_len_orig = instruct_hidden_states.shape[1] if split_input else None

    align = 2 * world
    encoder_seq_lengths_local, seq_lengths_local = _local_seq_lengths(
        encoder_seq_lengths, seq_lengths, align, world,
    )
    joint_attention_mask, img_attention_mask = _fit_local_masks(
        joint_attention_mask, img_attention_mask, world,
        encoder_seq_lengths_local, seq_lengths_local,
    )
    rotary_emb = _split_joint_rope(
        rotary_emb, encoder_seq_lengths, align, world, rank
    )
    image_rotary_emb = _slice_img_rope(image_rotary_emb, align, world, rank)

    if split_input:
        img_hidden_states, instruct_hidden_states = _split_streams(
            img_hidden_states, instruct_hidden_states, align, world, rank
        )

    if self.modulation:
        img_hidden_states, instruct_hidden_states = (
            _modulated_double_stream_forward(
                self, img_hidden_states, instruct_hidden_states, temb,
                joint_attention_mask, img_attention_mask, rotary_emb,
                image_rotary_emb, encoder_seq_lengths_local,
                seq_lengths_local, joint_attention_fn, img_self_attention_fn,
            )
        )
    else:
        raise NotImplementedError(
            "Double-stream SP only supports modulated blocks "
            "(self.modulation == True)"
        )

    if gather_output:
        img_hidden_states, instruct_hidden_states = _gather_and_strip_outputs(
            img_hidden_states, instruct_hidden_states, encoder_seq_lengths,
            seq_lengths, img_full_len_orig, instruct_full_len_orig,
        )

    return img_hidden_states, instruct_hidden_states


def boost_sp_double_stream(transformer, world_size=None, cc_overlap=False,
                           bsnd=False):
    """Patch the double-stream blocks for Ulysses sequence parallelism.

    Must be called AFTER ``boost_sp_single_stream`` (which owns the split /
    gather boundaries of the transformer forward): the double-stream stage
    consumes the FULL sequences produced by the context refiners and hands
    FULL sequences to the single-stream stage. ``cc_overlap`` routes every
    block through the comm-compute overlapping kernels in ``comm_compute_overlap.py``;
    ``bsnd`` routes the synchronous kernels through the fused-attention
    backend (no effect on the comm-compute kernels' backend choice).
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if world_size <= 1:
        return transformer

    from boogu.models.transformers.transformer_boogu import (
        BooguImageDoubleStreamTransformerBlock,
    )
    cls = BooguImageDoubleStreamTransformerBlock
    if not getattr(cls, "_lb_sp_patched", False):
        cls._lb_original_forward = cls.forward
        cls.forward = sp_double_stream_block_forward
        cls._lb_sp_patched = True

    layers = list(transformer.double_stream_layers)
    for i, blk in enumerate(layers):
        blk._lb_sp_split = i == 0
        blk._lb_sp_gather = i == len(layers) - 1
        blk._lb_sp_cc_overlap = cc_overlap
        blk._lb_sp_bsnd = bsnd
    return transformer
