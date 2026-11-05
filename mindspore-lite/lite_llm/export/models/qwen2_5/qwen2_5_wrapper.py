# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen2.5 NNRT ONNX wrapper (bias projections, no per-head Q/K norm).

Qwen2.5 attention uses the default base behaviour: projections carry q/k/v
biases (handled by ``nn.Linear`` automatically — no special wrapper code),
and there is no per-head Q/K RMSNorm.  The class exists so the exporter names
a concrete subclass and remains a seam for future Qwen2 variants.
"""

from models._base.nnrt_decoder_wrapper import NnrtDecoderWrapper, NnrtOpSet
from mslite_llm_ops import MsAttention, MsTransRopeScatterNDUpdate


def resolve_head_dim(config):
    """Single source of truth for the Qwen2.5 head dimension.

    The exporter (rope/cache shapes, size constraints) and the op set
    (operator attribute) must agree: prefer an explicit ``config.head_dim``
    and fall back to ``hidden_size // num_attention_heads``.
    """
    num_heads = int(config.num_attention_heads)
    return int(getattr(config, "head_dim", None) or config.hidden_size // num_heads)


class Qwen2OpSet(NnrtOpSet):
    """Fuse layout conversion, full-head RoPE, cache update and attention.

    ``MsTransRopeScatterNDUpdate`` writes the packed-NZ caches in place and
    returns only ``rope_q``, and ``MsAttention`` consumes those caches
    directly instead of ``MsGroupMatmul`` + ``MsAddSoftmax``.
    """

    def __init__(self, config):
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = resolve_head_dim(config)

    def prepare_qkv(self, states, position_embeddings, past_key_value, valid_seq_len):
        """Update packed NZ caches in place; the fused op returns only ``rope_q``."""
        query = MsTransRopeScatterNDUpdate.apply(
            *states, *position_embeddings, *past_key_value, valid_seq_len,
            self.num_heads, self.head_dim, "NZ",
        )
        return query, past_key_value[0], past_key_value[1]

    def attention(self, states, mask, valid_seq_len, dropout_p=0.0):
        """Fuse inference attention over the in-place-updated packed caches."""
        if dropout_p:
            return super().attention(states, mask, valid_seq_len, dropout_p)
        query, key, value = states
        scale = query.new_full((1,), self.head_dim ** -0.5)
        # The eager reference computes in fp32; the device kernel outputs fp16.
        return MsAttention.apply(
            query, key, value, mask.squeeze(0).squeeze(0), scale, valid_seq_len,
        ).to(query.dtype)


class Qwen2NnrtWrapper(NnrtDecoderWrapper):
    """NNRT wrapper for Qwen2.5 — fused cache/attention op set."""

    def __init__(self, hf_model, config, op_set=None):
        super().__init__(hf_model, config, Qwen2OpSet(config) if op_set is None else op_set)
