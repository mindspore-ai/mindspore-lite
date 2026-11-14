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
# torch.nn.functional members are callable; pylint mis-reports them.
# pylint: disable=not-callable
"""Qwen3.5-4B TP=2 prefill ONNX export (new, self-contained).

Shards the decoder with tensor parallelism:
  * full attention: q/k/v column-parallel (8 q heads / 2 kv heads per rank),
    o_proj row-parallel + AllReduce;
  * linear attention (GatedDeltaNet): qkv/z/b/a column-parallel (8 k heads /
    16 v heads per rank), conv1d channels sharded, CGDR per rank,
    out_proj row-parallel + AllReduce;
  * MLP: gate/up column-parallel, down row-parallel + AllReduce;
  * embedding / norms / lm_head replicated (identical logits per rank).

This script only exports the *prefill* graph at a fixed sequence length so it can
be converted with `--optimize=none` and run with provider=ge + HCCL (one process
per rank). It does not modify the 1p example.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from transformers import Qwen3_5ForConditionalGeneration
except ImportError:
    print("Error: transformers package not found or version too low.")
    sys.exit(1)

TP_SIZE = 2
TP_RANK = 0

# Qwen3.5-4B text config
HIDDEN = 2560
NUM_LAYERS = 32
NUM_Q_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
LIN_NK = 16          # linear key heads
LIN_NV = 32          # linear value heads
LIN_DK = 128
LIN_DV = 128
MLP_INTER = 9216
CONV_DIM = 2 * LIN_NK * LIN_DK + LIN_NV * LIN_DV
CONV_K = 4


def _l2norm(x, dim=-1, eps=1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


class AllReduceSum(torch.autograd.Function):
    """AllReduce(sum) across the TP group; GE lowers it to HcomAllReduce."""

    @staticmethod
    def forward(ctx, x):
        del ctx
        return x

    @staticmethod
    def symbolic(graph, x):
        """Emit the AllReduce Custom node consumed by the Ascend backend."""
        y = graph.op(
            "Custom", x,
            type_s="AllReduce",
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
            op_s="sum",
            group_s="hccl_world_group",
            rank_size_i=int(TP_SIZE),
            fusion_i=0,
        )
        y.setType(x.type())
        return y


def allreduce_sum(x):
    if TP_SIZE <= 1:
        return x
    return AllReduceSum.apply(x)


class Conv1dExport(torch.autograd.Function):
    """Depthwise causal conv1d that stays a real ONNX Conv without tracing it.

    fp16 F.conv1d on CPU is emulated (~12 s for the 4096-channel GDN conv), so
    the tracing forward only allocates the output and the symbolic reuses
    torch's own conv1d exporter.
    """

    @staticmethod
    def forward(ctx, x, w, stride, padding, dilation, groups, kernel_size):
        del ctx
        if torch.onnx.is_in_onnx_export():
            k = int(kernel_size)
            out_len = (x.shape[-1] + 2 * padding - dilation * (k - 1) - 1) // stride + 1
            return x.new_empty((x.shape[0], w.shape[0], out_len))
        return F.conv1d(x, w, None, stride=stride, padding=padding,
                        dilation=dilation, groups=groups)

    @staticmethod
    def symbolic(g, x, w, stride, padding, dilation, groups, kernel_size):
        # Emit explicit INTS attributes: torch's conv1d symbolic writes scalar
        # ints for 1-D convs, which mslite's ONNX Conv parser rejects.
        return g.op(
            "Conv", x, w,
            strides_i=[int(stride)],
            pads_i=[int(padding), int(padding)],
            dilations_i=[int(dilation)],
            group_i=int(groups),
            kernel_shape_i=[int(kernel_size)],
        )


class TrimDynamic(torch.autograd.Function):
    """Slice x's last axis to ref's last axis with a dynamic ONNX Slice.

    Keeps the causal-conv output length aligned with the (dynamic) token axis
    without baking a Python int into the graph.
    """

    @staticmethod
    def forward(ctx, x, ref):
        del ctx
        if torch.onnx.is_in_onnx_export():
            return x.new_empty((x.shape[0], x.shape[1], ref.shape[2]))
        return x[:, :, :ref.shape[2]]

    @staticmethod
    def symbolic(g, x, ref):
        """Emit a dynamic Slice that trims x to the reference length."""
        shape = g.op("Shape", ref)
        idx = g.op("Constant", value_t=torch.tensor([2], dtype=torch.long))
        end = g.op("Gather", shape, idx, axis_i=0)
        starts = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        axes = g.op("Constant", value_t=torch.tensor([2], dtype=torch.long))
        steps = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        y = g.op("Slice", x, starts, end, axes, steps)
        y.setType(x.type())
        return y


class ChunkGatedDeltaRuleTP(torch.autograd.Function):
    """TP-local ChunkGatedDeltaRule Custom op (q/k 8 heads, v 16 heads)."""

    @staticmethod
    def forward(ctx, query, key, value, beta, state, actual_seq_lengths, g_decay):
        del ctx, key, beta, actual_seq_lengths, g_decay
        if not torch.onnx.is_in_onnx_export():
            raise RuntimeError("ChunkGatedDeltaRuleTP is export-only")
        token_count, num_v_heads, value_dim = (int(v) for v in value.shape)
        return (
            query.new_empty((token_count, num_v_heads, value_dim)),
            state.new_empty(tuple(state.shape)),
        )

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths, g_decay):
        """Emit the TP-local ChunkGatedDeltaRule Custom node."""
        # JIT Value types can lack concrete sizes during tracing; the TP shapes
        # are fixed by this script.
        num_v_heads = LIN_NV // TP_SIZE
        value_dim = LIN_DV
        key_dim = LIN_DK
        batch = 1
        scale_value = 1.0 / (float(key_dim) ** 0.5)
        out, final_state = graph.op(
            "Custom",
            query, key, value, beta, state, actual_seq_lengths, g_decay,
            type_s="ChunkGatedDeltaRule",
            input_names_s=[
                "query", "key", "value", "beta", "initial_state",
                "actual_seq_lengths", "g",
            ],
            optional_input_names_s=["g"],
            output_names_s=["out", "final_state"],
            output_num_i=2,
            input_index_i=list(range(7)),
            scale_value_f=scale_value,
            dtype_i=10,
            outputs=2,
            outputs_shape_s=(
                f"3,-1,{num_v_heads},{value_dim},"
                f"4,{batch},{num_v_heads},{value_dim},{key_dim},"
            ),
        )
        out.setType(value.type().with_dtype(torch.float16))
        final_state.setType(state.type().with_dtype(torch.float16))
        return out, final_state


def _chunk_gated_delta_rule_tp(query, key, value, g, beta):
    """Adapter: local-head CGDR with host-side Q/K L2 normalization."""
    batch_size, _, num_key_heads, key_dim = query.shape
    num_value_heads = value.shape[2]
    value_dim = value.shape[3]
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)
    # Keep the token axis dynamic: -1 instead of a traced Python int.
    query_flat = query.reshape(-1, num_key_heads, key_dim).contiguous().half()
    key_flat = key.reshape(-1, num_key_heads, key_dim).contiguous().half()
    value_flat = value.reshape(-1, num_value_heads, value_dim).contiguous().half()
    beta_flat = beta.reshape(-1, num_value_heads).contiguous().half()
    g_flat = g.reshape(-1, num_value_heads).contiguous().float()
    state_custom = torch.zeros(
        batch_size, num_value_heads, value_dim, key_dim,
        dtype=torch.float16, device=query.device,
    )
    query_shape = torch.onnx.operators.shape_as_tensor(query)
    actual_seq_lengths = query_shape[1:2].to(torch.int32).expand(batch_size)
    out, state_out = ChunkGatedDeltaRuleTP.apply(
        query_flat, key_flat, value_flat, beta_flat,
        state_custom, actual_seq_lengths, g_flat,
    )
    out = out.reshape(batch_size, -1, num_value_heads, value_dim)
    state_out = state_out.transpose(-2, -1).contiguous().float()
    return out, state_out


# ---------------------------------------------------------------------------
# Weight sharding
# ---------------------------------------------------------------------------
def _rows(w, start, n):
    return w[start:start + n].detach().contiguous().half()


def _cols(w, start, n):
    return w[:, start:start + n].detach().contiguous().half()


def _local_conv_weight(conv_w, rank):
    """q/k/v channel slices of the depthwise conv weight [8192, 1, 4]."""
    k_per = LIN_NK // TP_SIZE * LIN_DK      # 8 * 128 = 1024
    v_per = LIN_NV // TP_SIZE * LIN_DV      # 16 * 128 = 2048
    q = conv_w[rank * k_per:(rank + 1) * k_per].detach()
    k = conv_w[LIN_NK * LIN_DK + rank * k_per:
               LIN_NK * LIN_DK + (rank + 1) * k_per].detach()
    v = conv_w[2 * LIN_NK * LIN_DK + rank * v_per:
               2 * LIN_NK * LIN_DK + (rank + 1) * v_per].detach()
    return torch.cat([q, k, v], dim=0).contiguous().half()


def shard_weights(text_model, rank):
    """Return {(layer_idx, tag): tensor} for this rank (weights stored [in, out])."""
    shards = {}
    q_per = NUM_Q_HEADS // TP_SIZE * HEAD_DIM      # 2048
    kv_per = NUM_KV_HEADS // TP_SIZE * HEAD_DIM    # 512
    k_per = LIN_NK // TP_SIZE * LIN_DK             # 1024
    v_per = LIN_NV // TP_SIZE * LIN_DV             # 2048
    inter_per = MLP_INTER // TP_SIZE               # 4608
    for i, layer in enumerate(text_model.layers):
        layer_type = getattr(layer, "block_type", None) or getattr(layer, "layer_type", None)
        if layer_type == "linear_attention":
            la = layer.linear_attn
            qw = la.in_proj_qkv.weight
            shards[(i, "qkv")] = torch.cat([
                qw[rank * k_per:(rank + 1) * k_per],
                qw[LIN_NK * LIN_DK + rank * k_per:
                   LIN_NK * LIN_DK + (rank + 1) * k_per],
                qw[2 * LIN_NK * LIN_DK + rank * v_per:
                   2 * LIN_NK * LIN_DK + (rank + 1) * v_per],
            ], dim=0).detach().contiguous().half()
            shards[(i, "z")] = _rows(la.in_proj_z.weight, rank * v_per, v_per)
            shards[(i, "b")] = _rows(la.in_proj_b.weight, rank * LIN_NV // TP_SIZE,
                                     LIN_NV // TP_SIZE)
            shards[(i, "a")] = _rows(la.in_proj_a.weight, rank * LIN_NV // TP_SIZE,
                                     LIN_NV // TP_SIZE)
            shards[(i, "out")] = _cols(la.out_proj.weight, rank * v_per, v_per)
            shards[(i, "conv")] = _local_conv_weight(la.conv1d.weight, rank)
            shards[(i, "a_log")] = la.A_log.detach()[rank * LIN_NV // TP_SIZE:
                                                     (rank + 1) * LIN_NV // TP_SIZE].float().contiguous()
            shards[(i, "dt_bias")] = la.dt_bias.detach()[rank * LIN_NV // TP_SIZE:
                                                         (rank + 1) * LIN_NV // TP_SIZE].float().contiguous()
        else:
            sa = layer.self_attn
            # q_proj rows are laid out per head as [head][q(D), gate(D)], so a
            # contiguous 8-head slice keeps q/gate interleaving and GQA mapping
            # (local q head j -> local kv head j // 4) intact.
            q_pair = NUM_Q_HEADS // TP_SIZE * HEAD_DIM * 2
            shards[(i, "q")] = _rows(sa.q_proj.weight, rank * q_pair, q_pair)
            shards[(i, "k")] = _rows(sa.k_proj.weight, rank * kv_per, kv_per)
            shards[(i, "v")] = _rows(sa.v_proj.weight, rank * kv_per, kv_per)
            shards[(i, "o")] = _cols(sa.o_proj.weight, rank * q_per, q_per)
        mlp = layer.mlp
        shards[(i, "gate")] = _rows(mlp.gate_proj.weight, rank * inter_per, inter_per)
        shards[(i, "up")] = _rows(mlp.up_proj.weight, rank * inter_per, inter_per)
        shards[(i, "down")] = _cols(mlp.down_proj.weight, rank * inter_per, inter_per)
    return shards


# ---------------------------------------------------------------------------
# Sharded prefill model
# ---------------------------------------------------------------------------
class Qwen35PrefillTP(torch.nn.Module):
    """TP-sharded Qwen3.5 prefill graph for one rank."""
    def __init__(self, text_model, lm_head, image_token_id):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self.image_token_id = int(image_token_id)
        self.config = text_model.config
        shards = shard_weights(text_model, TP_RANK)
        self._names = {}
        for (i, tag), tensor in shards.items():
            name = f"w_{i}_{tag}"
            self.register_buffer(name, tensor, persistent=False)
            self._names[(i, tag)] = name

    def _w(self, i, tag):
        return getattr(self, self._names[(i, tag)])

    @staticmethod
    def _lin(x, w):
        # F.linear dispatches to an optimized CPU kernel while tracing; a plain
        # fp16 torch.matmul on CPU is emulated and thousands of times slower.
        return F.linear(x, w)

    def _linear_attn(self, layer, hidden):
        """Run one TP-sharded GatedDeltaNet layer for prefill."""
        la = layer.linear_attn
        i = self._layer_idx
        batch = hidden.shape[0]
        ksz = int(la.conv1d.kernel_size[0])
        pad = ksz - 1
        mixed = self._lin(hidden, self._w(i, "qkv"))
        mixed = mixed.transpose(1, 2)
        conv_state = mixed[:, :, -pad:]
        # Left-pad then pad=0 conv keeps the output length equal to S without a
        # Python-int slice, so the seq axis stays dynamic. (torch.cat instead of
        # F.pad: constant_pad_nd trips the legacy exporter peephole pass.)
        conv_ref = mixed
        padded = Conv1dExport.apply(
            mixed, self._w(i, "conv"), 1, pad, 1,
            int(mixed.shape[1]), ksz,
        )
        mixed = F.silu(TrimDynamic.apply(padded, conv_ref)).transpose(1, 2)
        q_size = LIN_NK // TP_SIZE * LIN_DK
        v_size = LIN_NV // TP_SIZE * LIN_DV
        query, key, value = torch.split(mixed, [q_size, q_size, v_size], dim=-1)
        nk_loc = LIN_NK // TP_SIZE
        nv_loc = LIN_NV // TP_SIZE
        query = query.reshape(batch, -1, nk_loc, LIN_DK)
        key = key.reshape(batch, -1, nk_loc, LIN_DK)
        value = value.reshape(batch, -1, nv_loc, LIN_DV)
        z = self._lin(hidden, self._w(i, "z")).reshape(batch, -1, nv_loc, LIN_DV)
        b = self._lin(hidden, self._w(i, "b")).sigmoid()
        a = self._lin(hidden, self._w(i, "a"))
        g = -self._w(i, "a_log").exp() * F.softplus(a.float() + self._w(i, "dt_bias"))
        core, rec_state = _chunk_gated_delta_rule_tp(query, key, value, g=g, beta=b)
        core = core.reshape(-1, LIN_DV)
        z = z.reshape(-1, LIN_DV)
        core = la.norm(core, z).reshape(batch, -1, v_size)
        partial = self._lin(core, self._w(i, "out"))
        return allreduce_sum(partial), conv_state, rec_state

    def _full_attn(self, layer, hidden, position_embeddings, attn_mask):
        """Run one TP-sharded full-attention layer for prefill."""
        sa = layer.self_attn
        i = self._layer_idx
        batch = hidden.shape[0]
        q_heads = NUM_Q_HEADS // TP_SIZE
        kv_heads = NUM_KV_HEADS // TP_SIZE
        qkv = self._lin(hidden, self._w(i, "q")).view(batch, -1, q_heads, HEAD_DIM * 2)
        query, gate = torch.chunk(qkv, 2, dim=-1)
        gate = gate.reshape(batch, -1, q_heads * HEAD_DIM)
        query = sa.q_norm(query).transpose(1, 2)
        key = sa.k_norm(
            self._lin(hidden, self._w(i, "k")).view(batch, -1, kv_heads, HEAD_DIM)
        ).transpose(1, 2)
        value = self._lin(hidden, self._w(i, "v")).view(
            batch, -1, kv_heads, HEAD_DIM).transpose(1, 2)
        cos, sin = position_embeddings
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_rep = key.repeat_interleave(q_heads // kv_heads, dim=1)
        value_rep = value.repeat_interleave(q_heads // kv_heads, dim=1)
        scaling = getattr(sa, "scaling", 1.0 / (HEAD_DIM ** 0.5))
        scores = torch.matmul(query, key_rep.transpose(2, 3)) * scaling
        scores = scores + attn_mask
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        out = torch.matmul(scores, value_rep)
        out = out.transpose(1, 2).reshape(batch, -1, q_heads * HEAD_DIM).contiguous()
        out = out * torch.sigmoid(gate)
        partial = self._lin(out, self._w(i, "o"))
        return allreduce_sum(partial), key, value

    def forward(self, input_ids, position_ids, image_embeds, attn_mask):
        """Build the per-rank prefill graph outputs from the dummy inputs."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        image_mask = input_ids == self.image_token_id
        image_values = image_embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        image_indices = image_mask.long().cumsum(dim=1).clamp(min=1) - 1
        image_values = image_values[image_indices]
        inputs_embeds = torch.where(
            image_mask.unsqueeze(-1), image_values, inputs_embeds)

        mm_position_ids = position_ids[1:] if position_ids.shape[0] == 4 else position_ids
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, mm_position_ids)

        hidden = inputs_embeds
        present_conv, present_rec, present_kv = [], [], []
        for i, layer in enumerate(self.text_model.layers):
            self._layer_idx = i
            layer_type = getattr(layer, "block_type", None) or getattr(layer, "layer_type", None)
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if layer_type == "linear_attention":
                out, conv_s, rec_s = self._linear_attn(layer, hidden)
                present_conv.append(conv_s)
                present_rec.append(rec_s)
            else:
                out, pk, pv = self._full_attn(
                    layer, hidden, position_embeddings, attn_mask)
                present_kv.append(pk)
                present_kv.append(pv)
            hidden = residual + out

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            gate = self._lin(hidden, self._w(i, "gate"))
            up = self._lin(hidden, self._w(i, "up"))
            down = self._lin(mlp.act_fn(gate) * up, self._w(i, "down"))
            hidden = residual + allreduce_sum(down)

        hidden = self.text_model.norm(hidden)
        logits = self.lm_head(hidden)
        return (
            logits,
            torch.stack(present_conv, dim=0),
            torch.stack(present_rec, dim=0),
            torch.stack(present_kv, dim=0),
        )


def export_rank(model, rank, out_dir, seq_len, dynamic_seq=False, const_fold=True,
                image_tokens=16):
    """Export one TP rank's prefill ONNX graph."""
    global TP_RANK
    TP_RANK = rank
    text_model = model.model.language_model
    lm_head = model.lm_head
    text_model.eval()
    lm_head.eval()
    wrapper = Qwen35PrefillTP(text_model, lm_head, model.config.image_token_id)
    wrapper.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.int64)
    n_img = min(int(image_tokens), max(seq_len - 4, 0))
    input_ids[0, 4:4 + n_img] = int(model.config.image_token_id)
    base = torch.arange(seq_len, dtype=torch.int64).view(1, -1)
    position_ids = base.unsqueeze(0).expand(4, 1, seq_len).contiguous()
    image_embeds = torch.randn(int(image_tokens), HIDDEN, dtype=torch.float16)
    # Additive causal mask is now a model input so the seq axis stays dynamic.
    neg = torch.finfo(torch.float16).min
    ar = torch.arange(seq_len)
    attn_mask = torch.where(
        ar[None, :] > ar[:, None],
        torch.full((seq_len, seq_len), neg, dtype=torch.float16),
        torch.zeros((seq_len, seq_len), dtype=torch.float16),
    )[None, None]

    rank_dir = Path(out_dir) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    path = rank_dir / f"qwen3_5_llm_prefill_rank{rank}.onnx"
    print(f"Exporting rank{rank} prefill seq={seq_len} to {path} ...", flush=True)
    with torch.no_grad():
        from torch.onnx import utils as onnx_utils
        dynamic_axes = None
        if dynamic_seq:
            dynamic_axes = {
                "input_ids": {1: "seq_len"},
                "position_ids": {2: "seq_len"},
                "attn_mask": {2: "q_seq", 3: "k_seq"},
                "logits": {1: "seq_len"},
                "present_kv_cache": {3: "kv_seq"},
            }
        onnx_utils.export(
            wrapper,
            (input_ids, position_ids, image_embeds, attn_mask),
            str(path),
            input_names=["input_ids", "position_ids", "image_embeds", "attn_mask"],
            output_names=["logits", "present_conv_states",
                          "present_recurrent_states", "present_kv_cache"],
            opset_version=14,
            do_constant_folding=const_fold,
            dynamic_axes=dynamic_axes,
        )
    print(f"rank{rank} exported.", flush=True)
    del wrapper
    import gc
    gc.collect()


# ===========================================================================
# Decode (TP=2): RGDR 16 v-head/rank, IFA 2 kv-head/rank, sharded states
# ===========================================================================
class RecurrentGatedDeltaRuleTP(torch.autograd.Function):
    """TP-local RecurrentGatedDeltaRule (NV=16, NK=8)."""

    @staticmethod
    def forward(ctx, query, key, value, beta, state, actual_seq_lengths,
                ssm_state_indices, g_decay, num_accepted_tokens):
        del ctx, query, key, beta, actual_seq_lengths, g_decay
        del ssm_state_indices, num_accepted_tokens
        if not torch.onnx.is_in_onnx_export():
            raise RuntimeError("RecurrentGatedDeltaRuleTP is export-only")
        token_count, num_v_heads, value_dim = (int(v) for v in value.shape)
        return (
            value.new_empty((token_count, num_v_heads, value_dim)),
            state.new_empty(tuple(state.shape)),
        )

    @staticmethod
    def symbolic(graph, query, key, value, beta, state, actual_seq_lengths,
                 ssm_state_indices, g_decay, num_accepted_tokens):
        """Emit the TP-local RecurrentGatedDeltaRule Custom node."""
        num_v = LIN_NV // TP_SIZE
        scale_value = 1.0 / (float(LIN_DK) ** 0.5)
        out, state_out = graph.op(
            "Custom",
            query, key, value, beta, state, actual_seq_lengths,
            ssm_state_indices, g_decay, num_accepted_tokens,
            type_s="RecurrentGatedDeltaRule",
            input_names_s=["query", "key", "value", "beta", "state",
                           "actual_seq_lengths", "ssm_state_indices", "g", "gk",
                           "num_accepted_tokens"],
            optional_input_names_s=["g", "gk", "num_accepted_tokens"],
            input_index_i=[0, 1, 2, 3, 4, 5, 6, 7, 9],
            output_names_s=["out", "state"],
            output_num_i=2,
            scale_value_f=scale_value,
            output_types_i=[42, 42],
            outputs_shape_s=f"3,-1,{num_v},{LIN_DV},4,-1,{num_v},{LIN_DV},{LIN_DK},",
            dtype_i=10,
            outputs=2,
        )
        out.setType(value.type().with_dtype(torch.float16))
        state_out.setType(state.type().with_dtype(torch.float16))
        return out, state_out


def _rgdr_tp(query, key, value, g, beta, initial_state):
    """Run one TP-sharded RGDR decode step and return the new state."""
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)
    batch_size, seq_len, num_k, dk = query.shape
    num_v, dv = value.shape[2], value.shape[3]
    qf = query.reshape(-1, num_k, dk).to(torch.float16).contiguous()
    kf = key.reshape(-1, num_k, dk).to(torch.float16).contiguous()
    vf = value.reshape(-1, num_v, dv).to(torch.float16).contiguous()
    bf = beta.reshape(-1, num_v).to(torch.float16).contiguous()
    gf = g.reshape(-1, num_v).float().contiguous()
    state_custom = initial_state.transpose(-2, -1).contiguous().to(torch.float16)
    actual = torch.full((batch_size,), seq_len, dtype=torch.int32, device=query.device)
    ssm = torch.zeros(batch_size * seq_len, dtype=torch.int32, device=query.device)
    nacc = torch.full((batch_size,), seq_len, dtype=torch.int32, device=query.device)
    out, state_out = RecurrentGatedDeltaRuleTP.apply(
        qf, kf, vf, bf, state_custom, actual, ssm, gf, nacc)
    out = out.reshape(batch_size, seq_len, num_v, dv)
    state_out = state_out.transpose(-2, -1).contiguous().float()
    return out, state_out


class IncreFlashAttentionTP(torch.autograd.Function):
    """TP-sharded Incremental FlashAttention Custom operator (BSND)."""
    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                num_kv_heads):
        """Reference forward used while tracing the IFA Custom node."""
        del ctx
        key_ref, value_ref = key, value
        if 0 < num_kv_heads < num_heads:
            rep = num_heads // num_kv_heads
            key_ref = key_ref.repeat_interleave(rep, dim=1)
            value_ref = value_ref.repeat_interleave(rep, dim=1)
        attn = torch.matmul(query, key_ref.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(attn, value_ref)

    @staticmethod
    def symbolic(graph, query, key, value, atten_mask, num_heads, scale_value,
                 num_kv_heads):
        """Emit the IncreFlashAttention Custom node."""
        inputs = [query, key, value]
        idx = [0, 1, 2]
        if atten_mask is not None:
            inputs.append(atten_mask)
            idx.append(3)
        out = graph.op(
            "Custom", *inputs,
            type_s="IncreFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=idx,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BNSD",
            num_key_value_heads_i=int(num_kv_heads),
            block_size_i=0,
            inner_precise_i=1,
        )
        out.setType(query.type())
        return out


class ScatterUpdateTP(torch.autograd.Function):
    """TP-sharded Scatter update used to write the KV cache."""
    @staticmethod
    def forward(ctx, var, indices, updates, axis):
        del ctx
        axis = int(axis)
        result = var.clone()
        positions = indices.to(torch.int64).reshape(-1)
        index = positions.reshape(-1, 1, 1, 1).expand(
            updates.size(0), updates.size(1), 1, updates.size(3))
        result.scatter_(axis, index, updates.to(var.dtype))
        return result

    @staticmethod
    def symbolic(graph, var, indices, updates, axis):
        """Emit the Scatter Custom node used for KV-cache updates."""
        out = graph.op(
            "Custom", var, indices, updates,
            input_index_i=[0, 1, 2],
            input_names_s=["var", "indices", "updates"],
            optional_input_names_s=[],
            output_names_s=["var"],
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis),
        )
        out.setType(var.type())
        return out


def _scatter_kv_tp(past, update, cache_pos):
    indices = cache_pos.reshape(-1).to(torch.int64)
    return ScatterUpdateTP.apply(past, indices, update.to(past.dtype), 2)


def _expand_kv_tp(cache, num_heads, num_kv_heads):
    rep = num_heads // num_kv_heads
    b, _, cap, d = cache.shape
    return cache.unsqueeze(2).expand(b, num_kv_heads, rep, cap, d).reshape(
        b, num_heads, cap, d)


class Qwen35DecodeTP(torch.nn.Module):
    """TP-sharded Qwen3.5 decode graph for one rank."""
    def __init__(self, text_model, lm_head):
        super().__init__()
        self.text_model = text_model
        self.lm_head = lm_head
        self._names = {}
        shards = shard_weights(text_model, TP_RANK)
        for (i, tag), tensor in shards.items():
            name = f"d_{i}_{tag}"
            self.register_buffer(name, tensor, persistent=False)
            self._names[(i, tag)] = name
        self._idx = 0

    def _w(self, i, tag):
        return getattr(self, self._names[(i, tag)])

    @staticmethod
    def _lin(x, w):
        return F.linear(x, w)

    def _linear_attn(self, layer, hidden, conv_in, rec_in):
        """Run one TP-sharded GatedDeltaNet layer for decode."""
        la = layer.linear_attn
        i = self._idx
        mixed = self._lin(hidden, self._w(i, "qkv"))
        mixed_t = mixed.transpose(1, 2)
        new = torch.cat([conv_in, mixed_t], dim=-1)
        state_len = conv_in.shape[-1]
        conv_out = new[:, :, -state_len:]
        w = self._w(i, "conv")
        y = new[:, :, 0] * w[:, 0, 0].view(1, -1)
        for k in range(1, int(w.shape[-1])):
            y = y + new[:, :, k] * w[:, 0, k].view(1, -1)
        mixed = F.silu(y).unsqueeze(-1).transpose(1, 2)
        q_size = LIN_NK // TP_SIZE * LIN_DK
        v_size = LIN_NV // TP_SIZE * LIN_DV
        q, k, v = torch.split(mixed, [q_size, q_size, v_size], dim=-1)
        q = q.reshape(1, -1, LIN_NK // TP_SIZE, LIN_DK)
        k = k.reshape(1, -1, LIN_NK // TP_SIZE, LIN_DK)
        v = v.reshape(1, -1, LIN_NV // TP_SIZE, LIN_DV)
        z = self._lin(hidden, self._w(i, "z")).reshape(
            1, -1, LIN_NV // TP_SIZE, LIN_DV)
        b = self._lin(hidden, self._w(i, "b")).sigmoid()
        a = self._lin(hidden, self._w(i, "a"))
        g = -self._w(i, "a_log").exp() * F.softplus(a.float() + self._w(i, "dt_bias"))
        core, rec_out = _rgdr_tp(q, k, v, g, b, rec_in)
        core = la.norm(core.reshape(-1, LIN_DV), z.reshape(-1, LIN_DV))
        core = core.reshape(1, -1, v_size)
        partial = self._lin(core, self._w(i, "out"))
        return allreduce_sum(partial), conv_out, rec_out

    def _full_attn(self, layer, hidden, pos_emb, attn_mask, past_k, past_v, cache_pos):
        """Run one TP-sharded full-attention layer for decode."""
        sa = layer.self_attn
        i = self._idx
        q_heads = NUM_Q_HEADS // TP_SIZE
        kv_heads = NUM_KV_HEADS // TP_SIZE
        qkv = self._lin(hidden, self._w(i, "q")).view(1, -1, q_heads, HEAD_DIM * 2)
        query, gate = torch.chunk(qkv, 2, dim=-1)
        gate = gate.reshape(1, -1, q_heads * HEAD_DIM)
        query = sa.q_norm(query).transpose(1, 2)
        key = sa.k_norm(self._lin(hidden, self._w(i, "k")).view(
            1, -1, kv_heads, HEAD_DIM)).transpose(1, 2)
        value = self._lin(hidden, self._w(i, "v")).view(
            1, -1, kv_heads, HEAD_DIM).transpose(1, 2)
        cos, sin = pos_emb
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key_cache = _scatter_kv_tp(past_k, key, cache_pos)
        value_cache = _scatter_kv_tp(past_v, value, cache_pos)
        key_attn = _expand_kv_tp(key_cache, q_heads, kv_heads)
        value_attn = _expand_kv_tp(value_cache, q_heads, kv_heads)
        scale = getattr(sa, "scaling", 1.0 / (HEAD_DIM ** 0.5))
        attn = IncreFlashAttentionTP.apply(
            query.to(key_cache.dtype).contiguous(),
            key_attn.contiguous(), value_attn.contiguous(),
            attn_mask, q_heads, scale, q_heads)
        attn = attn.to(hidden.dtype).transpose(1, 2).reshape(1, -1, q_heads * HEAD_DIM)
        attn = attn * torch.sigmoid(gate)
        partial = self._lin(attn, self._w(i, "o"))
        return allreduce_sum(partial), key_cache, value_cache

    def forward(self, input_ids, attention_mask, position_ids,
                past_conv_states, past_recurrent_states, past_kv_cache):
        """Build the per-rank decode graph outputs from the dummy inputs."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        q_len = input_ids.shape[1]
        max_seq = past_kv_cache.shape[3]
        total_valid = attention_mask.sum(dim=1, keepdim=True)
        cache_pos = (total_valid - q_len).reshape(())
        kv_range = torch.arange(max_seq, device=inputs_embeds.device,
                                dtype=torch.int64)
        allowed = kv_range.view(1, -1) <= cache_pos.view(-1, 1)
        allowed = allowed & attention_mask.to(torch.bool)
        attn_mask = (~allowed).view(-1, 1, 1, max_seq)

        mm_pos = position_ids[1:] if position_ids.shape[0] == 4 else position_ids
        pos_emb = self.text_model.rotary_emb(inputs_embeds, mm_pos)

        hidden = inputs_embeds
        present_conv, present_rec, present_kv = [], [], []
        linear_idx, kv_idx = 0, 0
        for i, layer in enumerate(self.text_model.layers):
            self._idx = i
            lt = getattr(layer, "block_type", None) or getattr(layer, "layer_type", None)
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if lt == "linear_attention":
                out, conv_s, rec_s = self._linear_attn(
                    layer, hidden,
                    past_conv_states[linear_idx], past_recurrent_states[linear_idx])
                present_conv.append(conv_s)
                present_rec.append(rec_s)
                linear_idx += 1
            else:
                out, pk, pv = self._full_attn(
                    layer, hidden, pos_emb, attn_mask,
                    past_kv_cache[kv_idx], past_kv_cache[kv_idx + 1], cache_pos)
                present_kv.append(pk)
                present_kv.append(pv)
                kv_idx += 2
            hidden = residual + out

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            mlp = layer.mlp
            gate = self._lin(hidden, self._w(i, "gate"))
            up = self._lin(hidden, self._w(i, "up"))
            down = self._lin(mlp.act_fn(gate) * up, self._w(i, "down"))
            hidden = residual + allreduce_sum(down)

        hidden = self.text_model.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits,
                torch.stack(present_conv, dim=0),
                torch.stack(present_rec, dim=0),
                torch.stack(present_kv, dim=0))


def export_decode_rank(model, rank, out_dir, cap=4608, const_fold=True):
    """Export one TP rank's decode ONNX graph."""
    global TP_RANK
    TP_RANK = rank
    text_model = model.model.language_model
    lm_head = model.lm_head
    text_model.eval()
    lm_head.eval()
    wrapper = Qwen35DecodeTP(text_model, lm_head)
    wrapper.eval()

    ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64)
    mask = torch.zeros(1, cap, dtype=torch.int64)
    mask[0, :512] = 1
    pos = torch.arange(512, 513, dtype=torch.int64).view(1, 1)
    position_ids = pos.unsqueeze(0).expand(4, 1, 1).contiguous()
    past_conv = torch.zeros(24, 1, CONV_DIM // TP_SIZE, CONV_K - 1, dtype=torch.float16)
    past_rec = torch.zeros(24, 1, LIN_NV // TP_SIZE, LIN_DK, LIN_DV,
                           dtype=torch.float32)
    past_kv = torch.zeros(16, 1, NUM_KV_HEADS // TP_SIZE, cap, HEAD_DIM,
                          dtype=torch.float16)
    rank_dir = Path(out_dir) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    path = rank_dir / f"qwen3_5_llm_decode_rank{rank}.onnx"
    print(f"Exporting decode rank{rank} cap={cap} -> {path}", flush=True)
    with torch.no_grad():
        from torch.onnx import utils as onnx_utils
        onnx_utils.export(
            wrapper,
            (ids, mask, position_ids, past_conv, past_rec, past_kv),
            str(path),
            input_names=["input_ids", "attention_mask", "position_ids",
                         "past_conv_states", "past_recurrent_states", "past_kv_cache"],
            output_names=["logits", "present_conv_states",
                          "present_recurrent_states", "present_kv_cache"],
            opset_version=14,
            do_constant_folding=const_fold,
        )
    print(f"decode rank{rank} exported.", flush=True)
    del wrapper
    import gc
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3.5-4B TP2 models")
    parser.add_argument("--model-id", type=str, default="./Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="./qwen3_5_4b_tp2_onnx")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=4608)
    parser.add_argument("--vision-image-size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--component", type=str, default="all",
                        choices=["all", "vision", "prefill", "decode"])
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--dynamic-seq", action="store_true", default=True)
    parser.add_argument("--static-seq", dest="dynamic_seq", action="store_false")
    parser.add_argument("--no-const-fold", action="store_true")
    parser.add_argument("--image-tokens", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading model {args.model_id} (fp16, cpu) ...", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map="cpu",
        low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval()

    patch = int(model.config.vision_config.patch_size)
    merge = int(model.config.vision_config.spatial_merge_size)
    grid = max(int(args.vision_image_size) // patch, 1)
    image_tokens = (grid // merge) ** 2
    if args.no_vision:
        image_tokens = 16
    if args.image_tokens > 0:
        image_tokens = args.image_tokens
    print(f"image_tokens={image_tokens} no_vision={args.no_vision}")

    if args.component in ("all", "vision") and not args.no_vision:
        from export_qwen3_5_4b_onnx import export_vision_tower
        export_vision_tower(model, Path(args.output_dir) / "vision",
                            "cpu", args.vision_image_size)

    ranks = [0, 1] if args.rank < 0 else [args.rank]
    const_fold = not args.no_const_fold
    if args.component in ("all", "prefill"):
        for rank in ranks:
            export_rank(model, rank, args.output_dir, args.seq_len,
                        args.dynamic_seq, const_fold, image_tokens)
    if args.component in ("all", "decode"):
        for rank in ranks:
            export_decode_rank(model, rank, args.output_dir,
                              cap=args.max_seq_len, const_fold=const_fold)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
