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
BSND fused attention for BooguImage processors.

The single-card flavor of the shared attention bodies in ``attention_common.py``:
``IdentityExchanger`` (no collectives) + ``attention_bsnd`` backend (fused
``npu_fusion_attention`` consuming the [B, S, N, D] layout the QKV
projections already produce, so no BNSD transpose or materialization happens
between projection and attention — see ``npu_kernels.py``).

Known divergence from boogu's original processors (shipped behavior of
838f8a05, kept): 3-D masks take the ``unsqueeze(1)`` path of
``prepare_mask_4d`` instead of boogu's causal-mask rebuild — boogu's model
only sends 2-D padding masks in practice. 2-D masks go through
``prepare_mask_4d``'s broadcast path, value-identical to boogu's
``_prepare_sdpa_padding_mask``.

``install_bsnd_processors`` swaps the ``__call__`` of
``BooguImageAttnProcessor`` (single-stream) and
``BooguImageDoubleStreamSelfAttnProcessor`` (double-stream joint attention);
the Flash2Varlen processor variants are left untouched. When SP or the
comm-compute overlap patch is active it owns the double-stream joint attention, so
``install_bsnd_processors`` skips it and only patches the single-stream
processor (the SP/CC kernels pick the backend via the ``bsnd`` option).
"""

import logging

from lite_boost.model.booguimage.attention_common import (
    IdentityExchanger,
    attention_bsnd,
    double_stream_joint_sp_attention,
    resolve_apply_rotary_emb,
    single_stream_sp_attention,
)

logger = logging.getLogger(__name__)

_ORIG_SINGLE_STREAM_CALL = None
_ORIG_DOUBLE_STREAM_CALL = None


def _bsnd_single_stream_call(
    self, attn, hidden_states, encoder_hidden_states,
    attention_mask=None, image_rotary_emb=None, base_sequence_length=None,
):
    """Single-stream processor with BSND fused attention (no BNSD transposes)."""
    return single_stream_sp_attention(
        attn, hidden_states, attention_mask, image_rotary_emb,
        IdentityExchanger(), resolve_apply_rotary_emb(), backend=attention_bsnd,
        base_sequence_length=base_sequence_length,
        encoder_hidden_states=encoder_hidden_states,
    )


def _bsnd_double_stream_call(
    self, attn, img_hidden_states, instruct_hidden_states,
    joint_attention_mask=None, rotary_emb=None, encoder_seq_lengths=None,
    seq_lengths=None, base_sequence_length=None,
):
    """Double-stream joint processor with BSND fused attention."""
    hidden_attn, _ = double_stream_joint_sp_attention(
        attn, img_hidden_states, instruct_hidden_states, joint_attention_mask,
        rotary_emb, encoder_seq_lengths, seq_lengths,
        IdentityExchanger(), resolve_apply_rotary_emb(), backend=attention_bsnd,
        base_sequence_length=base_sequence_length,
    )
    return hidden_attn


def _get_processor_classes():
    from boogu.models.attention_processor import (
        BooguImageAttnProcessor,
        BooguImageDoubleStreamSelfAttnProcessor,
    )
    return BooguImageAttnProcessor, BooguImageDoubleStreamSelfAttnProcessor


def install_bsnd_processors(patch_double_stream=True):
    """Swap the processor ``__call__`` for the BSND fused-kernel variants.

    Idempotent. ``patch_double_stream=False`` leaves the double-stream
    processor alone (SP / comm-compute overlap owns it there).
    """
    global _ORIG_SINGLE_STREAM_CALL, _ORIG_DOUBLE_STREAM_CALL

    single_cls, double_cls = _get_processor_classes()

    if _ORIG_SINGLE_STREAM_CALL is None:
        _ORIG_SINGLE_STREAM_CALL = single_cls.__call__
        single_cls.__call__ = _bsnd_single_stream_call
    if patch_double_stream and _ORIG_DOUBLE_STREAM_CALL is None:
        _ORIG_DOUBLE_STREAM_CALL = double_cls.__call__
        double_cls.__call__ = _bsnd_double_stream_call
    logger.info(
        "bsnd: processors patched (single_stream=%s, double_stream=%s)",
        single_cls.__name__, double_cls.__name__ if patch_double_stream else "skipped",
    )


def revert_bsnd_processors():
    """Restore the original processor ``__call__`` methods."""
    global _ORIG_SINGLE_STREAM_CALL, _ORIG_DOUBLE_STREAM_CALL
    single_cls, double_cls = _get_processor_classes()
    if _ORIG_SINGLE_STREAM_CALL is not None:
        single_cls.__call__ = _ORIG_SINGLE_STREAM_CALL
        _ORIG_SINGLE_STREAM_CALL = None
    if _ORIG_DOUBLE_STREAM_CALL is not None:
        double_cls.__call__ = _ORIG_DOUBLE_STREAM_CALL
        _ORIG_DOUBLE_STREAM_CALL = None
    logger.info("bsnd: processors reverted")
