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
Export Kev-4B (Qwen3.5-4B-Base with the Kev LoRA merged in) to two ONNX graphs.

Kev is not a text generator. It is a causal backbone whose hidden states feed a pointer head
that scores the options of a question, so serving needs one graph per *row kind*, not
prefill/decode:

  kev_state   state row ``[<|fim_prefix|>] + state``  ->  conv / recurrent / KV caches
  kev_branch  one question row + those caches         ->  hidden states for the pointer head

The pointer head itself stays in numpy on the host (``head.npz``); both graphs end at the
last hidden state the head reads, so the exported graphs contain no head weights.

Both graphs are batch-1 and fully dynamic in their sequence dimensions, which holds because
every expensive subgraph is exported as the CANN operator that does the same arithmetic:

* linear attention -> ``ChunkGatedDeltaRule``. The reference chunked delta rule unrolls
  ``seq_len // 64`` chunks at trace time and would freeze the dummy length into the graph, so it
  cannot be used for the deployed graph. This operator additionally needs the mslite vendor
  package; the ones below are CANN built-ins.
* norms -> ``GemmaRmsNorm`` for the zero-centred form, ``RmsNorm`` plus the silu(gate) multiply
  for the gated one, instead of the eight elementwise kernels of the eager chain.
* the MLP's silu -> ``Swish``. The gate and up projections stay two separate MatMuls on purpose:
  merging them into one 18432-wide matmul makes GE fall back from MatMulV3 to MatMulV2, which
  costs more than the elementwise kernels it saves.
* full attention -> ``PromptFlashAttention`` on both graphs, replacing the eager
  mask-add / softmax / matmul chain. The operator has to be given ``pre_tokens`` /
  ``next_tokens`` explicitly: without them a cross-attention call (branch: query length ``Lb``
  against key length ``Ls + Lb``) only matches the reference while the key length is at most 64,
  and computes something else above that. With both windows open the bool mask alone decides
  what is visible, which is what the branch needs (see the class docstring).
* the delta rule's q/k normalization -> ``L2Normalize``, one kernel instead of ``Square`` plus a
  fused reduce/rsqrt/multiply.

Every fusion is unconditional: each one is checked against Kev's own torch implementation along
the whole state -> branch path, so a fusion that changed the result is a bug rather than a
configuration choice.
"""

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    # dynamo would try to trace the Custom op wrappers; the legacy exporter does not need it.
    import torch._dynamo

    torch._dynamo.disable()
except ImportError:
    pass

from transformers import AutoModelForCausalLM

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5MLP, Qwen3_5RMSNorm, Qwen3_5RMSNormGated, apply_rotary_pos_emb,
    )
except ImportError:
    print("Error: transformers with the qwen3_5 architecture is required (>= 5.19).")
    sys.exit(1)


LINEAR_ATTENTION = "linear_attention"
# torch.nn.Softplus() is F.softplus at the default beta=1, threshold=20; the module form keeps
# the call resolvable by static analysis, which cannot see into torch.nn.functional.
_SOFTPLUS = torch.nn.Softplus()
# Checksum of the masking value: additive masks are built on the host, and -10000 stays finite in
# fp16 (finfo(fp16).min is -65504, and two of those add up to -inf after a matmul).
MASK_FILL = -10000.0
# Custom-operator integration follows one shape for every operator:
#   <OP>_OP                 the operator name the Custom node carries
#   <Op>Function            a torch.autograd.Function whose symbolic() emits exactly that node
#   _<op>_custom(...)       adapter used by the traced model when the layouts differ
#   _install_<group>_ops()  class-level monkey patch, paired with _patched_<module>_forward()
CHUNK_GATED_DELTA_RULE_OP = "ChunkGatedDeltaRule"
GEMMA_RMS_NORM_OP = "GemmaRmsNorm"
RMS_NORM_OP = "RmsNorm"
# CANN Swish is y = x * sigmoid(scale * x); silu is the scale = 1 case.
SWISH_OP = "Swish"
PROMPT_FLASH_ATTENTION_OP = "PromptFlashAttention"
# The delta rule's q/k normalization. The operator clamps the squared sum at its own default
# (1e-4, a norm floor of 1e-2) and ignores the eps attribute, which is far below the norm of any
# real activation row.
L2_NORMALIZE_OP = "L2Normalize"
# "No window limit" for PromptFlashAttention's pre_tokens / next_tokens. The operator's defaults
# truncate cross-attention past 64 key columns; at or below 64 both settings agree, which is why
# only the branch graph can tell them apart.
INT_MAX = 2147483647
# Linear-attention key head dim, set from the loaded config; the Custom op scales q by 1/sqrt(d).
KEY_HEAD_DIM = 128


def _l2norm(x, dim=-1, eps=1e-6):
    """L2-normalize the input tensor along ``dim``."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _chunk_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state):
    """
    Reference chunked Gated DeltaNet rule (fp32 math).

    Trace-time ``range()`` loops make its graph shape-specific: it only runs inside
    ``ChunkGatedDeltaRuleFunction.forward`` during tracing, where the graph length is the
    exported dummy length.
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, _ = key.shape
    value_head_dim = value.shape[-1]
    chunk_size = 64
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    query = query * (1 / (query.shape[-1] ** 0.5))

    value_beta = value * beta.unsqueeze(-1)
    key_beta = key * beta.unsqueeze(-1)
    query, key, value, key_beta, value_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, key_beta, value_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size).cumsum(-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp()).tril()
    strict_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    attn = -((key_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(strict_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ value_beta
    key_cumdecay = attn @ (key_beta * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(batch_size, num_heads, key.shape[-1], value_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    out = torch.zeros_like(value)
    for i in range(query.shape[2]):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        new_value = v_i - key_cumdecay[:, :, i] @ state
        out[:, :, i] = ((q_i * g[:, :, i, :, None].exp()) @ state
                        + (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]) @ new_value)
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2)
                 @ new_value)
    out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])[:, :, :sequence_length]
    return out.transpose(1, 2).contiguous().to(initial_dtype), state if output_final_state else None


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Export the Ascend ChunkGatedDeltaRule Custom operator (7 inputs, 2 outputs)."""

    @staticmethod
    def forward(ctx, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Run the PyTorch reference implementation used while tracing."""
        del ctx, actual_seq_lengths
        batch_size = state.shape[0]
        sequence_length = query.shape[0] // batch_size
        num_key_heads, key_dim = query.shape[1], query.shape[2]
        num_value_heads, value_dim = value.shape[1], value.shape[2]
        query_4d = query.reshape(batch_size, sequence_length, num_key_heads, key_dim)
        key_4d = key.reshape(batch_size, sequence_length, num_key_heads, key_dim)
        value_4d = value.reshape(batch_size, sequence_length, num_value_heads, value_dim)
        beta_3d = beta.reshape(batch_size, sequence_length, num_value_heads)
        g_3d = g_decay.reshape(batch_size, sequence_length, num_value_heads)
        if num_value_heads > num_key_heads:
            repeat = num_value_heads // num_key_heads
            query_4d = query_4d.repeat_interleave(repeat, dim=2)
            key_4d = key_4d.repeat_interleave(repeat, dim=2)
        out, final_state = _chunk_gated_delta_rule(
            query_4d, key_4d, value_4d, g_3d, beta_3d,
            initial_state=state.transpose(-2, -1).contiguous().float(),
            output_final_state=True,
        )
        out = out.reshape(-1, num_value_heads, value_dim).half()
        final_state = final_state.transpose(-2, -1).contiguous().half()
        return out, final_state

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Emit the Custom node with the attributes registered by the Ascend OpDef."""
        out, final_state = graph.op(
            "Custom",
            query, key, value, beta, state, actual_seq_lengths, g_decay,
            type_s=CHUNK_GATED_DELTA_RULE_OP,
            input_names_s=["query", "key", "value", "beta", "initial_state",
                           "actual_seq_lengths", "g"],
            optional_input_names_s=["g"],
            output_names_s=["out", "final_state"],
            output_num_i=2,
            input_index_i=list(range(7)),
            scale_value_f=1.0 / (KEY_HEAD_DIM ** 0.5),
            dtype_i=10,
            outputs=2,
        )
        out.setType(value.type().with_dtype(torch.float16))
        final_state.setType(state.type().with_dtype(torch.float16))
        return out, final_state


def _norm_op_node(graph, op_type, x, gamma, eps):
    """Emit a Custom node carrying a CANN built-in norm op, returning its y output.

    The Custom primitive is only the transport: GE resolves ``op_type`` in the CANN operator
    library, so both ops keep their own shape inference and no vendor kernel is involved. Each
    declares a second ``rstd`` output that nothing consumes; leaving it untyped lets the built-in
    infer-shape function fill it in, which is why this beats a hand-written op here.
    """
    y, _ = graph.op(  # rstd is left untyped; GE's infer-shape supplies it
        "Custom", x, gamma, outputs=2,
        type_s=op_type,
        input_names_s=["x", "gamma"],
        output_names_s=["y", "rstd"],
        output_num_i=2,
        epsilon_f=float(eps),
    )
    y.setType(x.type())
    return y


class GemmaRmsNormFunction(torch.autograd.Function):
    """Export Qwen3.5's zero-centred RMSNorm as the CANN built-in ``GemmaRmsNorm``.

    GemmaRmsNorm is y = rmsnorm(x, eps) * (1 + gamma) computed in fp32 and rounded once -- the
    same expression as ``Qwen3_5RMSNorm.forward``, which stores its weight as a deviation from 1.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        del ctx
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        return (norm * (1.0 + weight.float())).type_as(x)

    @staticmethod
    def symbolic(graph, x, weight, eps):
        return _norm_op_node(graph, GEMMA_RMS_NORM_OP, x, weight, eps)


class RmsNormFunction(torch.autograd.Function):
    """Export the gated RMSNorm as CANN ``RmsNorm`` plus the silu(gate) multiply.

    RmsNorm covers norm(x) * weight in one kernel; this CANN version has no gated-norm variant,
    so the gate factor stays three elementwise nodes on a [rows, head_dim] tensor.
    """

    @staticmethod
    def forward(ctx, x, weight, gate, eps):
        del ctx
        input_dtype = x.dtype
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        # Replicates the module: the normalized value is rounded to the input dtype before the
        # weight, and the silu factor stays fp32 until the closing cast.
        norm = weight * norm.to(input_dtype)
        return (norm * F.silu(gate.float())).to(input_dtype)

    @staticmethod
    def symbolic(graph, x, weight, gate, eps):
        y = _norm_op_node(graph, RMS_NORM_OP, x, weight, eps)
        sigmoid = graph.op("Sigmoid", gate)
        sigmoid.setType(gate.type())
        silu_gate = graph.op("Mul", gate, sigmoid)
        silu_gate.setType(gate.type())
        out = graph.op("Mul", y, silu_gate)
        out.setType(x.type())
        return out


def _patched_rms_norm_forward(self, x):
    """Drop-in replacement for ``Qwen3_5RMSNorm.forward``."""
    return GemmaRmsNormFunction.apply(x, self.weight, self.eps)


def _patched_rms_norm_gated_forward(self, x, gate):
    """Drop-in replacement for ``Qwen3_5RMSNormGated.forward``."""
    return RmsNormFunction.apply(x, self.weight, gate, self.variance_epsilon)


def _install_norm_ops():
    """Patch the two norm modules at class level; every instance is ours in this exporter."""
    Qwen3_5RMSNorm.forward = _patched_rms_norm_forward
    Qwen3_5RMSNormGated.forward = _patched_rms_norm_gated_forward


class SwishFunction(torch.autograd.Function):
    """Export the MLP's silu as the CANN built-in ``Swish``.

    ``Qwen3_5MLP`` writes silu as Sigmoid followed by Mul on a [tokens, intermediate] tensor.
    GE leaves that pair as two full-size elementwise kernels here, so the fused form saves one
    round trip of the whole activation through HBM.
    """

    @staticmethod
    def forward(ctx, x):
        del ctx
        return F.silu(x)

    @staticmethod
    def symbolic(graph, x):
        y = graph.op("Custom", x, outputs=1, type_s=SWISH_OP,
                     input_names_s=["x"], output_names_s=["y"], output_num_i=1, scale_f=1.0)
        y.setType(x.type())
        return y


def _patched_mlp_forward(self, x):
    """Drop-in replacement for ``Qwen3_5MLP.forward`` with the activation routed to Swish."""
    return self.down_proj(SwishFunction.apply(self.gate_proj(x)) * self.up_proj(x))


def _install_mlp_ops(text_model):
    """Patch the MLP class after checking every layer shares the activation this assumes."""
    for layer in text_model.layers:
        if type(layer.mlp.act_fn).__name__ != "SiLUActivation":
            raise SystemExit(
                f"MLP activation is {type(layer.mlp.act_fn).__name__}, not SiLU; "
                "exporting it as Swish would change the model")
    Qwen3_5MLP.forward = _patched_mlp_forward


class PromptFlashAttentionFunction(torch.autograd.Function):
    """Export full attention as the CANN built-in ``PromptFlashAttention``.

    Both graphs use it: the state graph's causal self-attention (query and key length ``S``) and
    the branch graph's cross-attention (query ``Lb`` against key ``Ls + Lb``). ``pre_tokens`` and
    ``next_tokens`` must be exported explicitly; with the operator's defaults a branch-shaped call
    only agrees with the reference up to 64 key columns and computes something else above that
    (measured: key 128 with an all-visible mask still misses a numpy reference by 0.18). With both
    windows open the bool mask alone decides visibility, so the state graph's causal mask and the
    branch's "prefix visible, row causal" mask are both honoured at every length probed (key up to
    2304).
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_kv_heads, scale):
        del ctx
        repeat = num_heads // num_kv_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        weights = torch.matmul(query, key.transpose(2, 3)) * scale
        weights = weights.masked_fill(atten_mask, MASK_FILL)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(weights, value)

    @staticmethod
    def symbolic(graph, query, key, value, atten_mask, num_heads, num_kv_heads, scale):
        """Export one Custom node; ``sparse_mode=0`` takes the mask as passed, True = masked."""
        out = graph.op(
            "Custom", query, key, value, atten_mask, outputs=1,
            type_s=PROMPT_FLASH_ATTENTION_OP,
            input_names_s=["query", "key", "value", "atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_kv_heads),
            scale_value_f=float(scale),
            input_layout_s="BNSD",
            sparse_mode_i=0,
            # Both windows open, otherwise the operator's own sparse window (not the exported
            # mask) decides what a cross-attention call may see once the key length exceeds 64.
            pre_tokens_i=INT_MAX,
            next_tokens_i=INT_MAX,
        )
        out.setType(query.type())
        return out


class L2NormalizeFunction(torch.autograd.Function):
    """Export the delta rule's q/k normalization as the CANN built-in ``L2Normalize``.

    The eager form costs two kernels (a ``Square`` and one fused reduce/rsqrt/multiply) over the
    whole q and k tensors; the built-in does it in one, 0.026 ms against 0.066 ms per call at
    S=2048.
    """

    @staticmethod
    def forward(ctx, x, eps):
        del ctx
        return _l2norm(x, dim=-1, eps=eps)

    @staticmethod
    def symbolic(graph, x, eps):
        """Export one Custom node; the axis is the last dimension, as in ``_l2norm``."""
        y = graph.op("Custom", x, outputs=1, type_s=L2_NORMALIZE_OP,
                     input_names_s=["x"], output_names_s=["y"], output_num_i=1,
                     axis_i=[-1], eps_f=float(eps))
        y.setType(x.type())
        return y


def _l2_normalize_custom(x):
    """q/k normalization of the delta rule on the CANN ``L2Normalize`` operator."""
    return L2NormalizeFunction.apply(x, 1e-6)


def _chunk_gated_delta_rule_custom(query, key, value, g, beta, initial_state):
    """Adapt the model's [B, L, H, D] layout to the flat [tokens, H, D] Custom-op layout."""
    batch_size, sequence_length = query.shape[0], query.shape[1]
    token_count = batch_size * sequence_length
    num_value_heads, value_dim = value.shape[2], value.shape[3]
    query_flat = _l2_normalize_custom(query).reshape(
        token_count, query.shape[2], query.shape[3]).half()
    key_flat = _l2_normalize_custom(key).reshape(
        token_count, key.shape[2], key.shape[3]).half()
    value_flat = value.reshape(token_count, num_value_heads, value_dim).half()
    beta_flat = beta.reshape(token_count, num_value_heads).half()
    g_flat = g.reshape(token_count, num_value_heads).float()
    if initial_state is None:
        state_custom = torch.zeros(batch_size, num_value_heads, value_dim, query.shape[3],
                                   dtype=torch.float16, device=query.device)
    else:
        state_custom = initial_state.transpose(-2, -1).contiguous().half()
    seq_lengths = torch.onnx.operators.shape_as_tensor(query)[1:2].to(torch.int32).expand(batch_size)
    out, state_out = ChunkGatedDeltaRuleFunction.apply(
        query_flat.contiguous(), key_flat.contiguous(), value_flat.contiguous(),
        beta_flat.contiguous(), state_custom, seq_lengths, g_flat.contiguous(),
    )
    out = out.reshape(batch_size, sequence_length, num_value_heads, value_dim).to(query.dtype)
    return out, state_out.transpose(-2, -1).contiguous().float()


def _delta_rule(layer, mixed_qkv, z, gates, initial_state=None):
    """Gated DeltaNet tail shared by both graphs: split q/k/v, run the rule, gate, project.

    ``gates`` is ``(beta, g)``. ``initial_state`` is None on the state graph (the rule starts
    from zeros) and the cached recurrent state on the branch graph.
    """
    batch_size, seq_len = mixed_qkv.shape[0], mixed_qkv.shape[-1]
    query, key, value = torch.split(
        mixed_qkv.transpose(1, 2), [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, layer.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, layer.head_v_dim)
    beta, g = gates
    core_attn_out, state = _chunk_gated_delta_rule_custom(query, key, value, g, beta, initial_state)
    state = state.float() if state is not None else None
    core_attn_out = layer.norm(core_attn_out.reshape(-1, layer.head_v_dim),
                               z.reshape(-1, layer.head_v_dim))
    return layer.out_proj(core_attn_out.reshape(batch_size, seq_len, -1)), state


def _linear_attn_gates(layer, hidden_states):
    """beta = sigmoid(in_proj_b) and the fp32 decay g of the delta rule."""
    beta = layer.in_proj_b(hidden_states).sigmoid()
    g = -layer.A_log.float().exp() * _SOFTPLUS(layer.in_proj_a(hidden_states).float()
                                               + layer.dt_bias)
    return beta, g


def _causal_conv(layer, full_qkv, left_context):
    """Depthwise causal conv over ``left_context`` cached columns followed by the row tokens.

    ``conv1d`` pads both sides by ``k-1``, so its output is ``left_context + length + k - 1``
    wide: the first ``left_context`` entries belong to the cached columns and the last ``k-1``
    look into the right padding, so dropping both ends leaves one output per row token. Both
    slice offsets are constant, which keeps the graph dynamic-shape safe.
    """
    return layer.conv1d(full_qkv)[:, :, left_context:-(layer.conv_kernel_size - 1)]


def _linear_attn_state(layer, hidden_states):
    """State graph: one Gated DeltaNet layer from scratch, returning its output and both caches."""
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    projected = layer.in_proj_qkv(hidden_states)
    # The conv state is the pre-convolution tail: the k columns the next row needs as left context.
    # Taking them before the transpose keeps the slice on a contiguous [batch, k, channels] block;
    # after it, the same columns are a k-element gather per channel over the whole [C, seq] tensor.
    conv_state = projected[:, -layer.conv_kernel_size:, :].transpose(1, 2)
    mixed_qkv = projected.transpose(1, 2)
    z = layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, layer.head_v_dim)
    gates = _linear_attn_gates(layer, hidden_states)
    mixed_qkv = F.silu(_causal_conv(layer, mixed_qkv, 0))
    out, recurrent_state = _delta_rule(layer, mixed_qkv, z, gates)
    return out, conv_state, recurrent_state


def _linear_attn_branch(layer, hidden_states, conv_state, recurrent_state):
    """Branch graph: one Gated DeltaNet layer continuing from the cached conv/recurrent state."""
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    mixed_qkv = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    z = layer.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, layer.head_v_dim)
    gates = _linear_attn_gates(layer, hidden_states)
    # conv_state holds the k columns the conv needs as left context, so prepending it and
    # dropping both padded ends yields exactly one output per new token.
    full_qkv = torch.cat([conv_state, mixed_qkv], dim=-1).to(layer.conv1d.weight.dtype)
    conv_out = _causal_conv(layer, full_qkv, conv_state.shape[-1])
    out, _ = _delta_rule(layer, F.silu(conv_out).to(hidden_states.dtype), z, gates,
                         recurrent_state)
    return out


def _full_attn_forward(layer, hidden_states, position_embeddings, attention_mask, past=None):
    """Full-attention layer; ``past`` is the cached (key, value) on the branch graph."""
    batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    head_dim = layer.head_dim
    num_heads = layer.config.num_attention_heads
    num_kv_heads = layer.config.num_key_value_heads
    qkv = layer.q_proj(hidden_states).reshape(batch_size, seq_len, -1, head_dim * 2)
    query_states, gate = torch.chunk(qkv, 2, dim=-1)
    gate = gate.reshape(batch_size, seq_len, -1)
    query_states = layer.q_norm(query_states).transpose(1, 2)
    key_states = layer.k_norm(
        layer.k_proj(hidden_states).reshape(batch_size, seq_len, -1, head_dim)
    ).transpose(1, 2)
    value_states = layer.v_proj(hidden_states).reshape(
        batch_size, seq_len, -1, head_dim
    ).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    if past is not None:
        key_states = torch.cat([past[0], key_states], dim=2)
        value_states = torch.cat([past[1], value_states], dim=2)
    scaling = getattr(layer, "scaling", 1.0 / (head_dim ** 0.5))
    # The additive mask turns into the operator's bool form; -10000 is the only value the mask
    # builder writes, so any negative entry means masked. ``past`` only decides how long the key
    # axis is: the operator takes the concatenated caches like the eager path.
    attn_output = PromptFlashAttentionFunction.apply(
        query_states, key_states, value_states, attention_mask < -1,
        num_heads, num_kv_heads, scaling)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    return layer.o_proj(attn_output), key_states, value_states


def _layer_map(layers):
    """Per decoder layer: its kind and the index of its cache slot."""
    mapping, linear_idx, full_idx = [], 0, 0
    for layer in layers:
        if layer.block_type == LINEAR_ATTENTION:
            mapping.append((LINEAR_ATTENTION, linear_idx))
            linear_idx += 1
        else:
            mapping.append(("full_attention", full_idx))
            full_idx += 1
    return mapping


class KevStateModel(torch.nn.Module):
    """State graph: one row in, the three caches the branch graph continues from out."""

    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, input_ids, position_ids, attention_mask):
        """Run the state row and stack the caches."""
        hidden_states = self.text_model.embed_tokens(input_ids)
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        caches = {"conv": [], "recurrent": [], "kv": []}
        for layer in self.text_model.layers:
            residual = hidden_states
            attn_out = self._attention(layer, layer.input_layernorm(hidden_states),
                                       position_embeddings, attention_mask, caches)
            hidden_states = residual + attn_out
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states))
        return (torch.stack(caches["conv"]), torch.stack(caches["recurrent"]),
                torch.stack(caches["kv"]))

    def _attention(self, layer, hidden_states, position_embeddings, attention_mask, caches):
        """Dispatch one layer and record its cache."""
        if layer.block_type == LINEAR_ATTENTION:
            out, conv_state, recurrent_state = _linear_attn_state(layer.linear_attn, hidden_states)
            caches["conv"].append(conv_state)
            caches["recurrent"].append(recurrent_state)
            return out
        out, key_states, value_states = _full_attn_forward(
            layer.self_attn, hidden_states, position_embeddings, attention_mask)
        caches["kv"].extend([key_states, value_states])
        return out


class KevBranchModel(torch.nn.Module):
    """Branch graph: one question row continuing the state caches, out come its hidden states."""

    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model
        self.layer_map = _layer_map(text_model.layers)

    def forward(self, input_ids, position_ids, attention_mask,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """Run the branch row over the cached state, returning the final hidden states."""
        hidden_states = self.text_model.embed_tokens(input_ids)
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        for index, layer in enumerate(self.text_model.layers):
            kind, slot = self.layer_map[index]
            residual = hidden_states
            attn_in = layer.input_layernorm(hidden_states)
            if kind == LINEAR_ATTENTION:
                attn_out = _linear_attn_branch(layer.linear_attn, attn_in,
                                               past_conv_states[slot], past_recurrent_states[slot])
            else:
                past = (past_kv_cache[2 * slot], past_kv_cache[2 * slot + 1])
                attn_out, _, _ = _full_attn_forward(layer.self_attn, attn_in,
                                                    position_embeddings, attention_mask, past)
            hidden_states = residual + attn_out
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states))
        return self.text_model.norm(hidden_states)


def _causal_mask(length, past_len, dtype=torch.float16, device=None):
    """Additive causal mask for one row: [1, 1, length, past_len + length]."""
    total = past_len + length
    row = torch.arange(length, device=device).unsqueeze(1)
    col = torch.arange(total, device=device).unsqueeze(0)
    allowed = col <= (past_len + row)
    mask = torch.zeros(1, 1, length, total, dtype=dtype, device=device)
    return mask.masked_fill(~allowed[None, None], MASK_FILL)


def _dummies(args):
    """Example inputs: only the shapes matter, but the masks must be the real causal ones.

    The branch row continues a state of ``past_len`` tokens that starts at position 0, so its
    positions start at ``past_len`` and its mask has ``past_len`` prefix columns -- which is why
    ``--dummy-past-len`` defaults to ``--dummy-state-len`` rather than to an unrelated length.
    """
    device = args.device
    state_len, branch_len = args.dummy_state_len, args.dummy_branch_len
    past_len = args.dummy_past_len if args.dummy_past_len is not None else state_len
    return {
        "state_ids": torch.randint(0, args.vocab_size, (1, state_len),
                                   dtype=torch.int64, device=device),
        "state_pos": torch.arange(state_len, device=device).unsqueeze(0),
        "state_mask": _causal_mask(state_len, 0, device=device),
        "branch_ids": torch.randint(0, args.vocab_size, (1, branch_len),
                                    dtype=torch.int64, device=device),
        "branch_pos": torch.arange(past_len, past_len + branch_len,
                                   device=device).unsqueeze(0),
        "branch_mask": _causal_mask(branch_len, past_len, device=device),
        "past_conv": torch.randn(args.num_linear, 1, args.conv_dim, args.conv_kernel_size,
                                 dtype=torch.float16, device=device),
        "past_recurrent": torch.randn(args.num_linear, 1, args.num_v_heads,
                                      args.key_head_dim, args.value_head_dim, device=device),
        "past_kv": torch.randn(2 * args.num_full, 1, args.num_kv_heads, past_len,
                               args.head_dim, dtype=torch.float16, device=device),
    }


def _export(graph, path, spec, dynamic_axes):
    """Run the legacy exporter (the only one that understands old-style ``symbolic``).

    ``spec`` holds the example inputs plus their input/output names.
    """
    from torch.onnx import utils as onnx_utils

    print(f"--- exporting {path.name} (dummy tokens: {spec['inputs'][0].shape[-1]})")
    with torch.no_grad():
        onnx_utils.export(
            graph, spec["inputs"], str(path), input_names=spec["input_names"],
            output_names=spec["output_names"], opset_version=14,
            do_constant_folding=True, dynamic_axes=dynamic_axes,
        )


def _check_custom_nodes(path, expected):
    """Fail the export when a Custom operator is missing, duplicated, or left half-fused.

    ``expected`` is exact for the operators whose count is structural: CGDR is one per
    linear-attention layer, ``L2Normalize`` two per linear-attention layer, ``Swish`` one per
    live MLP, and the gated ``RmsNorm`` one per live linear-attention layer (the state graph
    drops the norm of a last layer that is a linear one: only its recurrent state is used, and
    the norm plus ``out_proj`` after the rule feed nothing). ``GemmaRmsNorm``
    is not -- the exporter drops nodes that never reach an output, so the state graph keeps 102
    of the model's 105 norms (the closing norm and the last layer's dead q_norm and
    post-attention norm disappear). What must hold instead is that every norm was fused: no
    eager chain is left, and ``Pow`` / ``ReduceMean`` do not appear at all.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts, leftover = {}, {}
    for node in model.graph.node:
        if node.op_type == "Custom":
            for attr in node.attribute:
                if attr.name == "type":
                    name = attr.s.decode()
                    counts[name] = counts.get(name, 0) + 1
        elif node.op_type in ("Pow", "ReduceMean"):
            leftover[node.op_type] = leftover.get(node.op_type, 0) + 1
    print(f"    {path.name}: {counts}, eager norm ops left: {leftover or '{}'}")
    wanted = {name: count for name, count in expected.items() if count}
    found = {name: count for name, count in counts.items() if name != GEMMA_RMS_NORM_OP}
    if found != wanted or leftover or counts.get(GEMMA_RMS_NORM_OP, 0) == 0:
        raise SystemExit(
            f"{path.name}: expected every RMSNorm fused with Custom nodes {wanted},"
            f" found {counts}, leftover norm ops {leftover}")


def _dynamic_axes():
    """Symbolic sequence axes of both graphs."""
    state = {"input_ids": {1: "seq_len"}, "position_ids": {1: "seq_len"},
             "attention_mask": {2: "seq_len", 3: "seq_len"},
             "present_kv_cache": {3: "seq_len"}}
    branch = {"input_ids": {1: "branch_len"}, "position_ids": {1: "branch_len"},
              "attention_mask": {2: "branch_len", 3: "total_len"},
              "past_kv_cache": {3: "past_len"}, "last_hidden_state": {1: "branch_len"}}
    return state, branch


def export_graphs(models, dummies, args):
    """Export both graphs, one subdirectory each, and check their Custom node count.

    The graphs must not share a directory: both write their large weights as external data
    files named after the tensor, and the two graphs reuse the same traced MatMul numbers for
    different shapes, so one export would silently overwrite the other's weights.
    """
    state_path = args.output_dir / "state" / "kev_state.onnx"
    branch_path = args.output_dir / "branch" / "kev_branch.onnx"
    for path in (state_path, branch_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    state_dynamic, branch_dynamic = _dynamic_axes()
    state_spec = {
        "inputs": (dummies["state_ids"], dummies["state_pos"], dummies["state_mask"]),
        "input_names": ["input_ids", "position_ids", "attention_mask"],
        "output_names": ["present_conv_states", "present_recurrent_states", "present_kv_cache"],
    }
    branch_spec = {
        "inputs": (dummies["branch_ids"], dummies["branch_pos"], dummies["branch_mask"],
                   dummies["past_conv"], dummies["past_recurrent"], dummies["past_kv"]),
        "input_names": ["input_ids", "position_ids", "attention_mask",
                        "past_conv_states", "past_recurrent_states", "past_kv_cache"],
        "output_names": ["last_hidden_state"],
    }
    _export(models["state"], state_path, state_spec, state_dynamic)
    _export(models["branch"], branch_path, branch_spec, branch_dynamic)
    # The state graph outputs caches, not hidden states, so its last layer's MLP is dead code --
    # and so is that layer's attention when it is a full-attention one, whose KV cache comes from
    # the projections before the attention itself. The branch graph keeps every layer.
    mlp_counts = {"state": args.num_layers - 1, "branch": args.num_layers}
    pfa_counts = {"state": args.num_full - args.last_full, "branch": args.num_full}
    # When that dead last layer is a linear one, the gated norm after the delta rule goes with
    # it: the rule still runs for the recurrent cache, but nothing reads its output.
    gated_counts = {"state": args.num_linear - args.last_linear, "branch": args.num_linear}
    for kind, path in (("state", state_path), ("branch", branch_path)):
        expected = {CHUNK_GATED_DELTA_RULE_OP: args.num_linear,
                    RMS_NORM_OP: gated_counts[kind],
                    SWISH_OP: mlp_counts[kind],
                    PROMPT_FLASH_ATTENTION_OP: pfa_counts[kind],
                    L2_NORMALIZE_OP: 2 * args.num_linear}
        _check_custom_nodes(path, expected)


def _model_meta(text_model, args):
    """Fill the shape constants the dummy caches need from the loaded config."""
    config = text_model.config
    args.num_layers = len(config.layer_types)
    args.num_linear = sum(1 for kind in config.layer_types if kind == LINEAR_ATTENTION)
    args.num_full = args.num_layers - args.num_linear
    args.last_full = int(config.layer_types[-1] != LINEAR_ATTENTION)
    args.last_linear = int(config.layer_types[-1] == LINEAR_ATTENTION)
    linear = next(layer.linear_attn for layer in text_model.layers
                  if layer.block_type == LINEAR_ATTENTION)
    args.conv_dim = linear.conv_dim
    args.conv_kernel_size = linear.conv_kernel_size
    args.num_v_heads = linear.num_v_heads
    args.key_head_dim = linear.head_k_dim
    global KEY_HEAD_DIM
    KEY_HEAD_DIM = linear.head_k_dim
    args.value_head_dim = linear.head_v_dim
    args.num_kv_heads = config.num_key_value_heads
    args.head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads)
    args.vocab_size = config.vocab_size


def parse_args(argv=None):
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", type=Path, required=True,
                        help="merged fp32 checkpoint directory (scripts/merge_lora.py output)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/onnx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dummy-state-len", type=int, default=64)
    parser.add_argument("--dummy-branch-len", type=int, default=16)
    parser.add_argument("--dummy-past-len", type=int, default=None,
                        help="cache length for the branch trace (default: --dummy-state-len)")
    return parser.parse_args(argv)


def main(argv=None):
    """Load the merged checkpoint in fp16 and export both graphs."""
    args = parse_args(argv)
    _install_norm_ops()
    merged = args.merged.expanduser().resolve()
    if not (merged / "config.json").is_file():
        raise SystemExit(
            f"--merged is not a local checkpoint directory: {args.merged}\n"
            f"  resolved against the current directory ({Path.cwd()}) to {merged}\n"
            "  pass the directory written by scripts/merge_lora.py; a path that does not\n"
            "  exist is treated by transformers as a Hub repo id, and loading then fails\n"
            "  with a network error against huggingface.co")
    print(f"--- loading merged checkpoint: {merged}")
    text_model = AutoModelForCausalLM.from_pretrained(
        str(merged), dtype=torch.float16, local_files_only=True).model
    text_model.eval().to(args.device)
    _model_meta(text_model, args)
    _install_mlp_ops(text_model)
    models = {"state": KevStateModel(text_model).to(args.device).eval(),
              "branch": KevBranchModel(text_model).to(args.device).eval()}
    export_graphs(models, _dummies(args), args)
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
