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

Both graphs are batch-1 and fully dynamic in their sequence dimensions. That is only possible
with the ``ChunkGatedDeltaRule`` Ascend Custom operator: the reference chunked delta rule
unrolls ``seq_len // 64`` chunks at trace time and would freeze the dummy length into the
graph. ``--no-enable-cgdr-custom`` keeps the reference implementation for single-shape
debugging and then exports fixed shapes on purpose.
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
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
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
USE_CUSTOM_CGDR = True
# Linear-attention key head dim, set from the loaded config; the Custom op scales q by 1/sqrt(d).
KEY_HEAD_DIM = 128


def _l2norm(x, dim=-1, eps=1e-6):
    """L2-normalize the input tensor along ``dim``."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _chunk_gated_delta_rule(query, key, value, g, beta, initial_state, output_final_state):
    """
    Reference chunked Gated DeltaNet rule (fp32 math).

    Trace-time ``range()`` loops make its graph shape-specific: it is only used with
    ``--no-enable-cgdr-custom``, where the fixed-shape dummy length is the graph length.
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
        out = out.reshape(-1, num_value_heads, value_dim).to(torch.float16)
        final_state = final_state.transpose(-2, -1).contiguous().to(torch.float16)
        return out, final_state

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Emit the Custom node with the attributes registered by the Ascend OpDef."""
        out, final_state = graph.op(
            "Custom",
            query, key, value, beta, state, actual_seq_lengths, g_decay,
            type_s="ChunkGatedDeltaRule",
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


def _chunk_gated_delta_rule_custom(query, key, value, g, beta, initial_state):
    """Adapt the model's [B, L, H, D] layout to the flat [tokens, H, D] Custom-op layout."""
    batch_size, sequence_length = query.shape[0], query.shape[1]
    token_count = batch_size * sequence_length
    num_value_heads, value_dim = value.shape[2], value.shape[3]
    query_flat = _l2norm(query, dim=-1).reshape(token_count, query.shape[2], query.shape[3]).half()
    key_flat = _l2norm(key, dim=-1).reshape(token_count, key.shape[2], key.shape[3]).half()
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
    if USE_CUSTOM_CGDR:
        core_attn_out, state = _chunk_gated_delta_rule_custom(
            query, key, value, g, beta, initial_state
        )
    else:
        repeat = layer.num_v_heads // layer.num_k_heads
        query, key = _l2norm(query, dim=-1), _l2norm(key, dim=-1)
        core_attn_out, state = _chunk_gated_delta_rule(
            query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2),
            value, g, beta, initial_state, True,
        )
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
    mixed_qkv = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    # The conv state is the pre-convolution tail: the k columns the next row needs as left context.
    conv_state = mixed_qkv[:, :, -layer.conv_kernel_size:]
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
    repeat = num_heads // num_kv_heads
    key_full = key_states.repeat_interleave(repeat, dim=1)
    value_full = value_states.repeat_interleave(repeat, dim=1)
    scaling = getattr(layer, "scaling", 1.0 / (head_dim ** 0.5))
    attn_weights = torch.matmul(query_states, key_full.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_full)
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


def _custom_node_count(path):
    """Count the nodes that actually call ChunkGatedDeltaRule.

    Every Ascend Custom op is exported with ``op_type == "Custom"``; the operator identity is the
    node's ``type`` string attribute. Counting bare ``Custom`` nodes would also accept 24
    unrelated custom ops.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return sum(1 for node in model.graph.node
               if node.op_type == "Custom"
               and any(attr.name == "type" and attr.s == b"ChunkGatedDeltaRule"
                       for attr in node.attribute))


def _dynamic_axes():
    """Dynamic axes per graph; empty for the fixed-shape reference export."""
    if not USE_CUSTOM_CGDR:
        print("WARNING: the reference delta rule freezes the dummy lengths into the graph")
        return {}, {}
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
    expected = args.num_linear if USE_CUSTOM_CGDR else 0
    for path in (state_path, branch_path):
        found = _custom_node_count(path)
        print(f"    {path.name}: {found} ChunkGatedDeltaRule nodes")
        if found != expected:
            raise SystemExit(f"{path.name}: expected {expected} Custom nodes, found {found}")


def _model_meta(text_model, args):
    """Fill the shape constants the dummy caches need from the loaded config."""
    config = text_model.config
    args.num_linear = sum(1 for kind in config.layer_types if kind == LINEAR_ATTENTION)
    args.num_full = len(config.layer_types) - args.num_linear
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
    parser.add_argument("--enable-cgdr-custom", action=argparse.BooleanOptionalAction,
                        default=True, help="use the Ascend ChunkGatedDeltaRule Custom operator")
    return parser.parse_args(argv)


def main(argv=None):
    """Load the merged checkpoint in fp16 and export both graphs."""
    global USE_CUSTOM_CGDR
    args = parse_args(argv)
    USE_CUSTOM_CGDR = args.enable_cgdr_custom
    merged = args.merged.expanduser().resolve()
    if not (merged / "config.json").is_file():
        raise SystemExit(
            f"--merged is not a local checkpoint directory: {args.merged}\n"
            f"  resolved against the current directory ({Path.cwd()}) to {merged}\n"
            "  pass the directory written by scripts/merge_lora.py; a path that does not\n"
            "  exist is treated by transformers as a Hub repo id, and loading then fails\n"
            "  with a network error against huggingface.co")
    print(f"--- loading merged checkpoint in fp16: {merged}")
    text_model = AutoModelForCausalLM.from_pretrained(
        str(merged), dtype=torch.float16, local_files_only=True).model
    text_model.eval().to(args.device)
    _model_meta(text_model, args)
    models = {"state": KevStateModel(text_model).to(args.device).eval(),
              "branch": KevBranchModel(text_model).to(args.device).eval()}
    export_graphs(models, _dummies(args), args)
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
