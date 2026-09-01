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
"""Qwen3 NNRT ONNX wrapper (per-head Q/K RMSNorm, bias-free projections).

Qwen3-specific differences vs the base Qwen2 path:

* per-head Q/K RMSNorm (``q_norm`` / ``k_norm``, shared ``[head_dim]``
  weight) applied on the ``(bsz, q_len, n_heads, head_dim)`` view before the
  head-dim transpose — exported as ``MsRmsNorm`` nodes at
  ``/model/layers.{i}/self_attn/{q,k}_norm/MsRmsNorm`` (the GGUF loader maps
  these names);
* bias-free projections (``attention_bias=False``) — ``nn.Linear(bias=False)``
  emits no bias Add, so no wrapper code is needed;
* ``head_dim`` may differ from ``hidden // heads`` (MiniMind-3: 96), read
  from config by the base class.
"""

from models._base.nnrt_decoder_wrapper import (
    NnrtAttention,
    NnrtDecoderWrapper,
    NnrtRmsNorm,
)


class Qwen3Attention(NnrtAttention):
    """Qwen3 attention: adds per-head Q/K RMSNorm adapters (``apply_qk_norm`` hook)."""

    def __init__(self, attn, config, ops):
        super().__init__(attn, config, ops)
        self.q_norm = NnrtRmsNorm(attn.q_norm, ops)
        self.k_norm = NnrtRmsNorm(attn.k_norm, ops)

    def apply_qk_norm(self, query_states, key_states, value_states):
        """Apply per-head Q/K RMSNorm before the head-dim transpose.

        The adapters read only the HF module's ``weight`` /
        ``variance_epsilon`` and call the op set's ``rmsnorm`` primitive, so
        the wrapper never depends on the transformers ``Qwen3RMSNorm``
        ``forward`` (a version coupling point) and the per-head norms follow
        the injected kernel policy.
        """
        return self.q_norm(query_states), self.k_norm(key_states), value_states


class Qwen3NnrtWrapper(NnrtDecoderWrapper):
    """NNRT wrapper for Qwen3 — selects the per-head-QK-norm attention adapter."""

    attn_module = Qwen3Attention
