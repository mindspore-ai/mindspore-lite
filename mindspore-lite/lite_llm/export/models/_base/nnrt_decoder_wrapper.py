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
"""Base ``torch.nn.Module`` hierarchy for NNRT (Kirin NPU) ONNX export.

Two orthogonal variation axes, two independent seams:

* **Architecture axis** (Qwen2 / Qwen3 / MiniCPM / ...): the wrapper selects
  an attention adapter via the ``attn_module`` class attribute and adapters
  override hooks (``NnrtAttention.apply_qk_norm`` for per-head Q/K norm).
  This axis decides *what* the graph computes.
* **Operator-set axis** (which ``torch_custom`` fused kernels realize each
  primitive): inject an ``NnrtOpSet`` subclass via ``op_set=``.  Different
  model specs / targets may fuse differently (e.g. unfused Add+Softmax
  instead of ``MsAddSoftmax``) while sharing the same architecture wrapper —
  no per-combination subclass explosion.

The hierarchy mirrors the HF ``*ForCausalLM`` module layout so traced node
and initializer names match the pre-refactor (monkey-patch) exports exactly.
Downstream tooling depends on those names — the GGUF loaders map GGUF
tensors by ONNX node name (``/model/layers.{i}/self_attn/q_proj/MatMul_quant``,
``/model/layers.{i}/input_layernorm/MsRmsNorm``, ``/model/norm/MsRmsNorm``)
and ``apply_shared_weight`` locates the lm_head MatMul by name::

    NnrtDecoderWrapper                     (root; lm_head at ``lm_head.*``)
    ├── model: NnrtDecoderCore             → ``model.*`` scope
    │   ├── layers.{i}: NnrtDecoderLayer
    │   │   ├── self_attn: NnrtAttention   → q/k/v/o_proj (HF Linears, shared)
    │   │   │   └── (Qwen3: q_norm / k_norm: NnrtRmsNorm)
    │   │   ├── input_layernorm: NnrtRmsNorm       → ops.rmsnorm
    │   │   ├── post_attention_layernorm: NnrtRmsNorm
    │   │   └── mlp (HF MLP — vanilla SwiGLU, called as-is, never patched)
    │   └── norm: NnrtRmsNorm              → ``/model/norm/MsRmsNorm``
    └── lm_head (HF Linear, shared)

transformers is only touched at load time (``AutoModelForCausalLM.from_pretrained``
etc., stable public API); the adapters hold plain references to the loaded
submodules and never call a transformers ``forward`` for norm/attention.

NNRT graph I/O contract (matches ``lite_llm/src/executor/nnrt/nnrt_executor.cc``):

    inputs:  [valid_seq_len, lmhead_idx, rope_cos, rope_sin, inputs_embeds,
              attention_mask, embedding_weight] + past_key_i/past_val_i (per layer)
    outputs: [logits, out_key_i/out_val_i]   (KV updated in place on device)

The I/O contract is op-set independent (fixed by the NPU runtime), so
swapping an op set never breaks the runtime contract — only the internal
node types change.
"""

import logging
import math
import os
import sys

import torch
from torch import nn

# custom_ops torch_custom layer (vendored under custom_ops/) is the single
# source of truth for the Ms* operator ONNX contracts (eager reference +
# ``custom::`` symbolic).  Source-checkout bootstrap — no-op once the
# mslite-llm-export wheel (which packages torch_custom) is installed.
_CUSTOM_OPS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "custom_ops"
)
if _CUSTOM_OPS_DIR not in sys.path:
    sys.path.insert(0, _CUSTOM_OPS_DIR)

# pylint: disable=wrong-import-position  # adapters resolve via _CUSTOM_OPS_DIR
from torch_custom.ms_add_softmax import MsAddSoftmax  # noqa: E402
from torch_custom.ms_group_matmul import MsGroupMatmul  # noqa: E402
from torch_custom.ms_rotary_pos_emb import MsRotaryPosEmb  # noqa: E402
from torch_custom.ms_rms_norm import MsRmsNorm  # noqa: E402
from torch_custom.ms_scatter_nd import MsScatterND  # noqa: E402

logger = logging.getLogger(__name__)


class NnrtOpSet:
    """Operator-set policy: which ``torch_custom`` kernels realize each primitive.

    The decoder forward loop is architecture math (reshape / scale / residual);
    every fused-kernel call is delegated to this policy so a model spec or
    target with a different fusion strategy can swap kernels without touching
    the wrapper.  Override only the primitives that differ, e.g.::

        class UnfusedSpecOpSet(NnrtOpSet):
            \"\"\"Spec that lets the NPU compiler fuse Add+Softmax itself.\"\"\"
            def mask_softmax(self, weights, mask):
                return (weights + mask).softmax(dim=-1)

    Coupling note: the ONNX postprocess pass ``fuse_add_rmsnorm``
    (``utils/onnx_postprocess.py``) assumes the *default* op set (plain
    ``Add`` + ``MsRmsNorm`` patterns get fused into ``MsAddRmsNorm``).  An op
    set that changes ``rmsnorm`` / matmul primitives must keep the exporter's
    fusion passes consistent (skip or extend them accordingly).
    """

    def rope(self, query, key, cos, sin):
        """Rotary position embedding on BNSD q/k with [B, S, D] cos/sin tables."""
        return MsRotaryPosEmb.apply(query, key, cos, sin)

    def kv_scatter(self, past, current_pos, current):
        """Scatter-update one KV cache tensor (BNSD): ``past[:, :, pos:pos+seq, :] = cur``.

        NNRT contract: pure ``MsScatterND`` — the device kernel writes in
        place.  No mask input.  Current state is cast to fp16 (cache dtype).
        """
        return MsScatterND.apply(past, current_pos, current.to(torch.float16), "BNSD")

    def qk_matmul(self, query, key):
        """Q @ K^T (BNSD, GQA-aware)."""
        return MsGroupMatmul.apply(query, key, True)

    def pv_matmul(self, weights, value):
        """P @ V (BNSD)."""
        return MsGroupMatmul.apply(weights, value, False)

    def mask_softmax(self, weights, mask):
        """Fused (weights + mask).softmax(-1)."""
        return MsAddSoftmax.apply(weights, mask)

    def rmsnorm(self, hidden, weight, eps):
        """RMSNorm over the last dim."""
        return MsRmsNorm.apply(hidden, weight, eps)


class NnrtRmsNorm(nn.Module):
    """RMSNorm adapter: keeps the traced module scope, delegates to the op set.

    Sharing the HF module's ``weight`` / ``variance_epsilon`` but never
    calling its ``forward`` is the decoupling point — the transformers
    RMSNorm implementation may change across versions; only the two plain
    attributes are read.  Being a real ``nn.Module`` gives the exported node
    its scope name (``…/input_layernorm/MsRmsNorm``), which the GGUF loaders
    rely on.
    """

    def __init__(self, src_norm, ops):
        super().__init__()
        self.weight = src_norm.weight
        self.variance_epsilon = src_norm.variance_epsilon
        self.ops = ops

    def forward(self, hidden):
        return self.ops.rmsnorm(hidden, self.weight, self.variance_epsilon)


class NnrtAttention(nn.Module):
    """Attention adapter holding the HF projections (shared ``nn.Linear``).

    Implements the standard RoPE + GQA attention math with every fused kernel
    delegated to the op set.  ``apply_qk_norm`` is the architecture hook
    (default no-op); Qwen3 subclasses add per-head Q/K RMSNorm adapters.
    """

    def __init__(self, attn, config, ops):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.o_proj
        self.attention_dropout = getattr(attn, "attention_dropout", 0.0)
        self.ops = ops
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(
            getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        )

    def apply_qk_norm(self, query_states, key_states, value_states):
        """Architecture hook: per-head Q/K norm before the head-dim transpose."""
        return query_states, key_states, value_states

    def forward(self, hidden_states, attention_mask, rope_cos, rope_sin,
                past_key_value, valid_seq_len):
        """Project → heads → qk-norm → RoPE → KV scatter → GQA attention."""
        bsz = hidden_states.size(0)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_kv_heads, self.head_dim)

        query_states, key_states, value_states = self.apply_qk_norm(
            query_states, key_states, value_states
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # RoPE consumes BNSD q/k and [B, S, D] cos/sin.
        query_states, key_states = self.ops.rope(query_states, key_states, rope_cos, rope_sin)

        key_states = self.ops.kv_scatter(past_key_value[0], valid_seq_len, key_states)
        value_states = self.ops.kv_scatter(past_key_value[1], valid_seq_len, value_states)

        attn_weights = self.ops.qk_matmul(query_states, key_states)
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        attn_weights = self.ops.mask_softmax(attn_weights, attention_mask)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = self.ops.pv_matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, -1, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, [key_states, value_states]


class NnrtDecoderLayer(nn.Module):
    """Decoder-layer adapter: pre-norm residual wiring around attention + MLP."""

    def __init__(self, layer, config, ops, attn_module):
        super().__init__()
        self.self_attn = attn_module(layer.self_attn, config, ops)
        self.input_layernorm = NnrtRmsNorm(layer.input_layernorm, ops)
        self.post_attention_layernorm = NnrtRmsNorm(layer.post_attention_layernorm, ops)
        # HF MLP is vanilla SwiGLU (down_proj(silu(gate) * up)) built from
        # plain nn.Linear — historically never monkey-patched; called as-is.
        self.mlp = layer.mlp

    def forward(self, hidden_states, attention_mask, rope_cos, rope_sin,
                past_key_value, valid_seq_len):
        """Pre-norm residual wiring: x + attn(norm(x)), then x + mlp(norm(x))."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states, attention_mask, rope_cos, rope_sin, past_key_value, valid_seq_len
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, present_key_value


class NnrtDecoderCore(nn.Module):
    """Inner decoder (held as ``wrapper.model``) → traced ``model.*`` scope.

    ``embed_tokens`` is deliberately not held: the embedding lookup runs
    CPU-side and enters the graph as ``inputs_embeds``; the tied lm_head
    weight becomes the ``embedding_weight`` graph input in postprocess.
    """

    def __init__(self, inner, config, ops, attn_module):
        super().__init__()
        self.layers = nn.ModuleList(
            [NnrtDecoderLayer(layer, config, ops, attn_module) for layer in inner.layers]
        )
        self.norm = NnrtRmsNorm(inner.norm, ops)

    def forward(self, inputs_embeds, attention_mask, rope_cos, rope_sin,
                past_key_values, valid_seq_len):
        """Run all decoder layers; returns (final hidden, interleaved KV list)."""
        hidden_states = inputs_embeds
        presents = []
        for i, layer in enumerate(self.layers):
            hidden_states, present = layer(
                hidden_states, attention_mask, rope_cos, rope_sin,
                past_key_values[i], valid_seq_len,
            )
            presents.extend(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class NnrtDecoderWrapper(nn.Module):
    """Base NNRT ONNX wrapper for a Qwen-style HF causal-LM.

    Selects the attention adapter via the ``attn_module`` class attribute
    (architecture axis) and the fused kernels via ``op_set`` (operator-set
    axis).  The forward loop, attention math and KV cache handling are shared
    here so adding a new model, a new transformers version, or a new fusion
    strategy does not require monkey-patching transformers or forking the
    loop.
    """

    attn_module = NnrtAttention

    def __init__(self, hf_model, config, op_set=None):
        """Build the adapter hierarchy over the loaded HF model's submodules.

        ``op_set`` defaults to ``NnrtOpSet`` (the standard ``Ms*`` kernels);
        pass a subclass to realize the same architecture with a different
        fused-operator set.
        """
        super().__init__()
        self.config = config
        # Plain-object policy (not nn.Module): no params/buffers leak into tracing.
        self.ops = op_set if op_set is not None else NnrtOpSet()
        self.model = NnrtDecoderCore(hf_model.model, config, self.ops, self.attn_module)
        self.lm_head = hf_model.lm_head
        self.num_layers = int(config.num_hidden_layers)
        self.hidden_size = int(config.hidden_size)

    def forward(self, input_ids=None, valid_seq_len=None, lmhead_idx=None,
                rope_cos=None, rope_sin=None, inputs_embeds=None,
                attention_mask=None, past_key_values=None):
        """NNRT-contract forward: embedding lookup stays CPU-side (graph input).

        Returns ``(logits, out_key_0, out_val_0, …)`` matching the exporter's
        ``output_names``.
        """
        del input_ids  # NNRT: embedding lookup is CPU-side; inputs_embeds is required.
        hidden_states, presents = self.model(
            inputs_embeds, attention_mask, rope_cos, rope_sin,
            past_key_values, valid_seq_len,
        )
        hidden_states = hidden_states[:, lmhead_idx]
        logits = self.lm_head(hidden_states).float()
        return (logits, *presents)
