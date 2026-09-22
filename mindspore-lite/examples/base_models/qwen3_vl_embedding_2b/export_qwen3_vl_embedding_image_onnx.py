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
Export Qwen3-VL-Embedding-2B (image path) to ONNX as two chunks.

- `qwen3_vl_embedding_2b_vision.onnx`: vision tower exported for a fixed patch grid.
  Input `pixel_values [num_patches, 1536]`, outputs `image_embeds [num_merged, 2048]`
  and the three deepstack features `ds0/ds1/ds2 [num_merged, 2048]`. The learned
  position embedding and the vision rotary tables only depend on the image grid, so
  they are pre-computed for `--vision-image-size` (1024 by default, which gives a
  64x64 patch grid, 4096 patches and 1024 merged tokens) and baked into the graph. The
  inference script must use the same image size (`--image-size`), and a different size
  needs a new export of this chunk.
- `qwen3_vl_embedding_2b_text_image.onnx`: text backbone taking `input_ids`,
  `attention_mask`, 3-row mrope `position_ids` plus the vision outputs above. The image
  features are scattered into the `<|image_pad|>` positions inside the graph, so the
  host never has to read the token embedding table; their row count is fixed by
  `--vision-image-size`, so this chunk is re-exported whenever the image size changes.
  For text-only inputs pass dummy features of the same row count: MindSpore Lite on
  Ascend does not support size-0 tensors.

The two chunks are exported in float32 on purpose. The MindSpore Lite converter
rejects FLOAT16 type declarations for several parsers (for example `Clip` fails with
`do not support data_type: 10`), so the ONNX models must stay float32.

Every ONNX file is written together with a `<name>.onnx.data` file (the weights are
too large for a single protobuf message). Keep the two files in the same directory
when converting them with `converter_lite`.

The CANN fused operators are optional and disabled by default, so the exported graphs
stay as close to the reference implementation as possible. Pass
`--use-fused-gelu-tanh-nz`, `--use-fused-rms-norm-nz` and/or
`--use-fused-qk-norm-rope-bsh` to emit `Custom(FusedGeluTanhNZ)`,
`Custom(FusedRmsNormNZ)` and/or `Custom(FusedQKNormRopeBSH)` instead of the plain
operator graphs; the fused variants need a CANN version providing those operators.

Example:
    python export_qwen3_vl_embedding_image_onnx.py --model-id ./Qwen3-VL-Embedding-2B
    python export_qwen3_vl_embedding_image_onnx.py --module vision --vision-image-size 256

Then convert both graphs for Ascend, for example:

    converter_lite --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \\
        --modelFile=qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision.onnx \\
        --outputFile=qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b_vision
"""

import argparse
import gc
import os
import sys
import time
from collections import Counter

import onnx
import torch

try:
    from transformers import AutoModel
except ImportError:
    print("Error: the transformers package was not found or is too old.")
    print("Please install or upgrade it, for example: pip install --upgrade transformers")
    sys.exit(1)

# `torch.autograd.Function` subclasses below intentionally override `forward` with a
# different signature (the symbolic one), and `torch.nn.functional` is not understood
# by pylint's inference, which makes these checks noisy false positives.
# pylint: disable=abstract-method,arguments-differ,not-callable

V_HIDDEN = 1024
V_HEADS = 16
V_HEAD_DIM = 64
V_INTER = 4096
V_OUT = 2048
V_MERGE = 2
V_DEEPSTACK_LAYERS = (5, 11, 17)
V_IN_CHANNELS = 3
V_T_PATCH = 2
V_PATCH = 16
V_IN_DIM = V_IN_CHANNELS * V_T_PATCH * V_PATCH * V_PATCH

_GELU_TANH_BETA = 0.044715
_GELU_TANH_NEG_ALPHA = -1.5957691216057308

# The CANN fusion switches are exposed as `--use-fused-*` CLI flags (see
# `_parse_args`) and default to the plain (non-fused) graphs, which are easier to
# convert and debug.


def _remove_isnan_nodes(onnx_model):
    """Replace Where(IsNaN(x), default, x) with Identity(x); drop IsNaN nodes."""
    isnan_nodes = [n for n in onnx_model.graph.node if n.op_type == "IsNaN"]
    if not isnan_nodes:
        print("No IsNaN nodes found in the model")
        return onnx_model
    print(f"Found {len(isnan_nodes)} IsNaN nodes, replacing with Identity...")
    remove = set()
    add = []
    for isnan_node in isnan_nodes:
        isnan_output = isnan_node.output[0]
        for node in onnx_model.graph.node:
            if node.op_type == "Where" and isnan_output in node.input:
                if len(node.input) == 3 and node.input[0] == isnan_output:
                    identity = onnx.helper.make_node(
                        "Identity",
                        inputs=[node.input[2]],
                        outputs=[node.output[0]],
                        name=node.name + "_identity",
                    )
                    add.append(identity)
                    remove.add(node.name)
        remove.add(isnan_node.name)
    new_nodes = [n for n in onnx_model.graph.node if n.name not in remove]
    new_nodes.extend(add)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)
    print(f"Removed {len(isnan_nodes)} IsNaN nodes")
    return onnx_model


class _CannPromptFlashAttentionBSH(torch.autograd.Function):
    """QK^T + softmax + V -> Custom(PromptFlashAttention) in BSH layout."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """PyTorch reference impl returning the [B, S, N, D] CANN PFA layout."""
        del ctx
        b, s, _ = query.shape
        d = query.shape[-1] // num_heads
        q_ref = query.reshape(b, s, num_heads, d).transpose(1, 2)
        kd = key.shape[-1] // num_key_value_heads
        k_ref = key.reshape(b, s, num_key_value_heads, kd).transpose(1, 2)
        v_ref = value.reshape(b, s, num_key_value_heads, kd).transpose(1, 2)
        if num_key_value_heads < num_heads:
            rep = num_heads // num_key_value_heads
            k_ref = k_ref.repeat_interleave(rep, dim=1)
            v_ref = v_ref.repeat_interleave(rep, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(q_ref, k_ref.transpose(2, 3)) * scale
        if atten_mask is not None:
            attn = attn + atten_mask
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_out = torch.matmul(attn, v_ref)
        attn_out = attn_out.transpose(1, 2).reshape(b, s, num_heads * d)
        return attn_out

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, num_key_value_heads, scale_value):
        """Emit the ONNX Custom node mapping to the CANN PromptFlashAttention op."""
        y = g.op(
            "Custom",
            query,
            key,
            value,
            atten_mask,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value", "atten_mask"],
            optional_input_names_s=["atten_mask"],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2, 3],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BSH",
            inner_precise_i=4,
        )
        y.setType(query.type())
        return y


class _CannPromptFlashAttentionBSHNonCausal(torch.autograd.Function):
    """QK^T + softmax + V -> Custom(PromptFlashAttention), no mask, BSH layout."""

    @staticmethod
    def forward(ctx, query, key, value, num_heads, num_key_value_heads, scale_value):
        """PyTorch reference impl returning the [B, S, N, D] CANN PFA layout."""
        del ctx
        b, s, _ = query.shape
        d = query.shape[-1] // num_heads
        q_ref = query.reshape(b, s, num_heads, d).transpose(1, 2)
        kd = key.shape[-1] // num_key_value_heads
        k_ref = key.reshape(b, s, num_key_value_heads, kd).transpose(1, 2)
        v_ref = value.reshape(b, s, num_key_value_heads, kd).transpose(1, 2)
        if num_key_value_heads < num_heads:
            rep = num_heads // num_key_value_heads
            k_ref = k_ref.repeat_interleave(rep, dim=1)
            v_ref = v_ref.repeat_interleave(rep, dim=1)
        scale = float(scale_value)
        attn = torch.matmul(q_ref, k_ref.transpose(2, 3)) * scale
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_out = torch.matmul(attn, v_ref)
        attn_out = attn_out.transpose(1, 2).reshape(b, s, num_heads * d)
        return attn_out

    @staticmethod
    def symbolic(g, query, key, value, num_heads, num_key_value_heads, scale_value):
        """Emit the ONNX Custom node mapping to the CANN PromptFlashAttention op."""
        y = g.op(
            "Custom",
            query,
            key,
            value,
            type_s="PromptFlashAttention",
            input_names_s=["query", "key", "value"],
            optional_input_names_s=[],
            output_names_s=["attention_out"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            num_heads_i=int(num_heads),
            num_key_value_heads_i=int(num_key_value_heads),
            scale_value_f=float(scale_value),
            input_layout_s="BSH",
            inner_precise_i=4,
        )
        y.setType(query.type())
        return y


class _CannRotaryMul(torch.autograd.Function):
    """rotate_half(x) applied with cos/sin -> Custom(RotaryMul)."""

    @staticmethod
    def forward(ctx, x, r1, r2):
        """PyTorch reference impl of the CANN RotaryMul op."""
        del ctx
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        y = x * r1 + rotated * r2
        return y

    @staticmethod
    def symbolic(g, x, r1, r2):
        """Emit the ONNX Custom node mapping to the CANN RotaryMul op."""
        sizes = x.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 4:
                dims[0] = -1
                dims[2] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])
        y = g.op(
            "Custom",
            x,
            r1,
            r2,
            type_s="RotaryMul",
            input_names_s=["x", "r1", "r2"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1, 2],
            output_shapes_s=out_shapes,
        )
        y.setType(x.type())
        return y


class _CannSwiGlu(torch.autograd.Function):
    """silu(x[:d]) * x[d:] -> Custom(SwiGlu)."""

    @staticmethod
    def forward(ctx, x, dim):
        """PyTorch reference impl of the CANN SwiGlu op."""
        del ctx
        d = int(dim)
        split = x.shape[d] // 2
        a, b = torch.split(x, [split, split], dim=d)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim):
        """Emit the ONNX Custom node mapping to the CANN SwiGlu op."""
        sizes = x.type().sizes()
        if sizes is None:
            out_shapes = ""
        else:
            dims = [int(d) if d is not None else -1 for d in list(sizes)]
            if len(dims) == 3:
                dims[0] = -1
                dims[1] = -1
            out_shapes = ",".join([str(len(dims))] + [str(i) for i in dims])
        y = g.op(
            "Custom",
            x,
            type_s="SwiGlu",
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
            dim_i=int(dim),
            output_shapes_s=out_shapes,
        )
        y.setType(x.type())
        return y


class _CannMatMulV2(torch.autograd.Function):
    """x1 @ x2 -> Custom(MatMulV2)."""

    @staticmethod
    def forward(ctx, x1, x2):
        """PyTorch reference impl of the CANN MatMulV2 op."""
        del ctx
        return torch.matmul(x1, x2)

    @staticmethod
    def symbolic(g, x1, x2):
        """Emit the ONNX Custom node mapping to the CANN MatMulV2 op."""
        x1_sizes = x1.type().sizes()
        x2_sizes = x2.type().sizes()
        if x1_sizes is None or x2_sizes is None:
            out_shapes = ""
        else:
            m = x1_sizes[0]
            n = x2_sizes[1] if len(x2_sizes) >= 2 else None
            dims = [
                int(m) if m is not None else -1,
                int(n) if n is not None else -1,
            ]
            out_shapes = ",".join(["2"] + [str(i) for i in dims])
        y = g.op(
            "Custom",
            x1,
            x2,
            type_s="MatMulV2",
            input_names_s=["x1", "x2"],
            optional_input_names_s=[],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1],
            output_shapes_s=out_shapes,
        )
        y.setType(x1.type())
        return y


class _CannFusedGeluTanhNZ(torch.autograd.Function):
    """gelu(x, approximate="tanh") -> Custom(FusedGeluTanhNZ)."""

    @staticmethod
    def forward(ctx, x):
        """PyTorch reference impl of the tanh-approximated GELU."""
        del ctx
        x3 = x * x * x
        z = _GELU_TANH_NEG_ALPHA * (x + _GELU_TANH_BETA * x3)
        return x / (1.0 + torch.exp(z))

    @staticmethod
    def symbolic(g, x):
        """Emit the ONNX Custom node mapping to the CANN FusedGeluTanhNZ op."""
        y = g.op(
            "Custom",
            x,
            type_s="FusedGeluTanhNZ",
            input_names_s=["x"],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0],
        )
        y.setType(x.type())
        return y


class _CannFusedRmsNormNZ(torch.autograd.Function):
    """RmsNorm -> Custom(FusedRmsNormNZ)."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon):
        """PyTorch reference impl of the CANN FusedRmsNormNZ op."""
        del ctx
        var = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + float(epsilon)) * gamma

    @staticmethod
    def symbolic(g, x, gamma, epsilon):
        """Emit the ONNX Custom node mapping to the CANN FusedRmsNormNZ op."""
        y = g.op(
            "Custom",
            x,
            gamma,
            type_s="FusedRmsNormNZ",
            input_names_s=["x", "gamma"],
            output_names_s=["y"],
            output_num_i=1,
            input_index_i=[0, 1],
            epsilon_f=float(epsilon),
        )
        y.setType(x.type())
        return y


class _CannFusedQKNormRopeBSH(torch.autograd.Function):
    """QK-norm + RoPE on BSH q/k -> Custom(FusedQKNormRopeBSH)."""

    @staticmethod
    def forward(ctx, q, k, gamma_q, gamma_k, cos, sin, epsilon):
        """PyTorch reference impl of the fused q/k RMS-norm followed by RoPE."""
        del ctx
        eps = float(epsilon)
        d = gamma_q.shape[-1]
        half = d // 2
        c3 = cos.to(torch.float32)[..., None, :]
        s3 = sin.to(torch.float32)[..., None, :]

        def norm_rope(x, gamma):
            shape = x.shape
            h = x.shape[-1] // d
            x3 = x.to(torch.float32).view(*shape[:-1], h, d)
            ms = (x3 * x3).mean(dim=-1, keepdim=True) + eps
            xn = x3 * (1.0 / torch.sqrt(ms)) * gamma.to(torch.float32).view(1, 1, d)
            a, b = xn[..., :half], xn[..., half:]
            out = torch.cat([a * c3 - b * s3, b * c3 + a * s3], dim=-1)
            return out.to(x.dtype).view(*shape[:-1], h * d)

        return norm_rope(q, gamma_q), norm_rope(k, gamma_k)

    @staticmethod
    def symbolic(g, q, k, gamma_q, gamma_k, cos, sin, epsilon):
        """Emit the ONNX Custom node mapping to the CANN FusedQKNormRopeBSH op."""
        q_out, k_out = g.op(
            "Custom",
            q,
            k,
            gamma_q,
            gamma_k,
            cos,
            sin,
            type_s="FusedQKNormRopeBSH",
            input_names_s=["q", "k", "gamma_q", "gamma_k", "cos", "sin"],
            output_names_s=["q_out", "k_out"],
            output_num_i=2,
            input_index_i=[0, 1, 2, 3, 4, 5],
            epsilon_f=float(epsilon),
            output_shapes_s="",
            outputs=2,
        )
        q_out.setType(q.type())
        k_out.setType(k.type())
        return q_out, k_out


def _linear(linear_mod, x, enable_bmm2mm_fusion):
    """Linear layer as a plain MatMul (+ bias), or as Custom(MatMulV2) when fused."""
    if enable_bmm2mm_fusion:
        return _linear_2d(linear_mod, x)
    out = torch.matmul(x, linear_mod.weight.t())
    if linear_mod.bias is not None:
        out = out + linear_mod.bias
    return out


def _linear_2d(linear_mod, x):
    """Linear via Custom(MatMulV2), flattening any leading dims to 2D first."""
    if x.dim() == 2:
        out = _CannMatMulV2.apply(x, linear_mod.weight.t())
        if linear_mod.bias is not None:
            out = out + linear_mod.bias
        return out
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    out2d = _CannMatMulV2.apply(x2d, linear_mod.weight.t())
    if linear_mod.bias is not None:
        out2d = out2d + linear_mod.bias
    return out2d.reshape(*orig_shape[:-1], -1)


def _linear_slice(src_linear, r0, r1, x):
    """Slice of a fused linear weight (used to split qkv/fc1 without extra weights)."""
    out = torch.matmul(x, src_linear.weight[r0:r1].t())
    if src_linear.bias is not None:
        out = out + src_linear.bias[r0:r1]
    return out


def _get_rmsnorm_epsilon(norm_mod):
    """Read the epsilon of an RMSNorm module regardless of its attribute name."""
    for attr in ("variance_epsilon", "eps", "epsilon"):
        val = getattr(norm_mod, attr, None)
        if val is not None:
            return float(val)
    return 1e-6


def _cann_rotary_mul(x, cos, sin):
    """Apply Custom(RotaryMul)."""
    return _CannRotaryMul.apply(x, cos, sin)


def cann_apply_rotary_pos_emb(query, key, cos, sin):
    """Apply the CANN rotary embedding to query and key."""
    query_out = _cann_rotary_mul(query, cos, sin)
    key_out = _cann_rotary_mul(key, cos, sin)
    return query_out, key_out


def _make_bool_causal_mask(attention_mask, q_len, k_len, past_len):
    """Build the boolean (True == masked) causal + padding mask used by CANN PFA."""
    device = attention_mask.device
    ar_q = torch.arange(q_len, device=device)
    ar_k = torch.arange(k_len, device=device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    padding = 1.0 - attention_mask.to(torch.float)
    mask_val = causal.to(torch.float)[None, None, :, :] + padding[:, None, None, :]
    mask = mask_val > 0.5
    mask = mask.expand(attention_mask.shape[0], 1, q_len, k_len)
    return mask


def _cann_attn_forward_bsh(attn_mod, hidden_states, position_embeddings, bool_mask, rotarymul):
    """Attention forward emitting Custom(RotaryMul) and Custom(PromptFlashAttention)."""
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    b, s = hidden_states.shape[0], hidden_states.shape[1]
    hidden_shape = (b, s, -1, head_dim)
    query_states = _linear(attn_mod.q_proj, hidden_states, False).view(hidden_shape)
    key_states = _linear(attn_mod.k_proj, hidden_states, False).view(hidden_shape)
    value_states = _linear(attn_mod.v_proj, hidden_states, False).view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = attn_mod.q_norm(query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = attn_mod.k_norm(key_states)

    cos, sin = position_embeddings
    if rotarymul:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        query_states, key_states = cann_apply_rotary_pos_emb(query_states, key_states, cos, sin)
    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    query_bsh = query_states.reshape(b, s, num_heads * head_dim)
    key_bsh = key_states.reshape(b, s, num_kv_heads * head_dim)
    value_bsh = value_states.reshape(b, s, num_kv_heads * head_dim)
    attn_out = _CannPromptFlashAttentionBSH.apply(
        query_bsh, key_bsh, value_bsh, bool_mask, int(num_heads), int(num_kv_heads), float(scaling),
    )
    attn_out = _linear(attn_mod.o_proj, attn_out, False)
    return attn_out, key_states, value_states


def _cann_attn_forward_bsh_fused_qkrope(attn_mod, hidden_states, position_embeddings, bool_mask):
    """Attention forward fusing q/k norm+RoPE, then Custom(PromptFlashAttention)."""
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    b, s = hidden_states.shape[0], hidden_states.shape[1]

    query_states = _linear(attn_mod.q_proj, hidden_states, False)
    key_states = _linear(attn_mod.k_proj, hidden_states, False)
    value_states = _linear(attn_mod.v_proj, hidden_states, False).view(b, s, -1, head_dim)

    cos, sin = position_embeddings
    half = head_dim // 2
    eps = _get_rmsnorm_epsilon(attn_mod.q_norm)
    query_states, key_states = _CannFusedQKNormRopeBSH.apply(
        query_states, key_states,
        attn_mod.q_norm.weight, attn_mod.k_norm.weight,
        cos[..., :half], sin[..., :half],
        eps,
    )
    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))
    query_bsh = query_states.reshape(b, s, num_heads * head_dim)
    key_bsh = key_states.reshape(b, s, num_kv_heads * head_dim)
    value_bsh = value_states.reshape(b, s, num_kv_heads * head_dim)
    attn_out = _CannPromptFlashAttentionBSH.apply(
        query_bsh, key_bsh, value_bsh, bool_mask, int(num_heads), int(num_kv_heads), float(scaling),
    )
    attn_out = _linear(attn_mod.o_proj, attn_out, False)
    return attn_out, key_states, value_states


def _cann_mlp_forward(mlp_mod, hidden_states, enable_swiglu=True, enable_bmm2mm=False):
    """MLP forward with the SwiGlu fusion switch."""
    if enable_swiglu:
        gate_up = torch.cat(
            [
                _linear(mlp_mod.gate_proj, hidden_states, enable_bmm2mm),
                _linear(mlp_mod.up_proj, hidden_states, enable_bmm2mm),
            ],
            dim=-1,
        )
        gate_up = _CannSwiGlu.apply(gate_up, -1)
    else:
        gate = _linear(mlp_mod.gate_proj, hidden_states, enable_bmm2mm)
        up = _linear(mlp_mod.up_proj, hidden_states, enable_bmm2mm)
        gate_up = torch.nn.functional.silu(gate) * up
    return _linear(mlp_mod.down_proj, gate_up, enable_bmm2mm)


def _vision_mlp_forward(mlp_mod, hidden_n, use_fused_gelu_tanh_nz=False):
    """Vision MLP forward, slicing the fused fc1 weight into per-shard linears."""
    h_parts = [
        _linear_slice(mlp_mod.linear_fc1, i * V_HIDDEN, (i + 1) * V_HIDDEN, hidden_n)
        for i in range(V_INTER // V_HIDDEN)
    ]
    h = torch.cat(h_parts, dim=-1)
    if use_fused_gelu_tanh_nz:
        h = _CannFusedGeluTanhNZ.apply(h)
    else:
        h = torch.nn.functional.gelu(h, approximate="tanh")
    return _linear(mlp_mod.linear_fc2, h, False)


def _cann_vision_attn_forward(attn_mod, hidden_normed, cos, sin):
    """Vision attention forward: split qkv, Custom(RotaryMul), non-causal PFA."""
    b, n = hidden_normed.shape[0], hidden_normed.shape[1]
    q = _linear_slice(attn_mod.qkv, 0, V_HIDDEN, hidden_normed).view(b, n, V_HEADS, V_HEAD_DIM)
    k = _linear_slice(attn_mod.qkv, V_HIDDEN, 2 * V_HIDDEN, hidden_normed).view(b, n, V_HEADS, V_HEAD_DIM)
    v = _linear_slice(attn_mod.qkv, 2 * V_HIDDEN, 3 * V_HIDDEN, hidden_normed).view(b, n, V_HEADS, V_HEAD_DIM)
    cos4 = cos.unsqueeze(0).unsqueeze(2)
    sin4 = sin.unsqueeze(0).unsqueeze(2)
    q, k = cann_apply_rotary_pos_emb(q, k, cos4, sin4)
    q_bsh = q.reshape(b, n, V_HEADS * V_HEAD_DIM)
    k_bsh = k.reshape(b, n, V_HEADS * V_HEAD_DIM)
    v_bsh = v.reshape(b, n, V_HEADS * V_HEAD_DIM)
    attn_out = _CannPromptFlashAttentionBSHNonCausal.apply(
        q_bsh, k_bsh, v_bsh, int(V_HEADS), int(V_HEADS), float(attn_mod.scaling),
    )
    return _linear(attn_mod.proj, attn_out, False)


class Qwen3VLEmbeddingVisionFused(torch.nn.Module):
    """Vision tower wrapper exported for a fixed patch grid.

    The learned position embedding and the vision rotary tables only depend on the
    image grid, so they are pre-computed for `grid_thw` and registered as buffers: the
    exported model takes `pixel_values` only and the host script does not need to touch
    `pos_embed`. Attention is global over the packed patch sequence, which matches the
    upstream varlen attention for a single image per forward. The vision MLP GELU is
    emitted as Custom(FusedGeluTanhNZ) when `use_fused_gelu_tanh_nz` is set, and as a
    plain `gelu(approximate="tanh")` otherwise.

    The merger outputs are reshaped to the constant `[num_merged, V_OUT]` shape, because
    the internal `-1` reshape of the upstream merger leaves a symbolic output dimension
    that the Ascend graph engine refuses to infer.
    """

    def __init__(self, model, grid_thw, use_fused_gelu_tanh_nz=False):
        """Build the wrapper and bake the fixed-grid position and rotary tables."""
        super().__init__()
        self.use_fused_gelu_tanh_nz = bool(use_fused_gelu_tanh_nz)
        self.num_merged = int(grid_thw.prod().item()) // (V_MERGE * V_MERGE)
        visual = model.visual
        with torch.no_grad():
            pos_embed = visual.fast_pos_embed_interpolate(grid_thw).to(torch.float32)
            rotary = visual.rot_pos_emb(grid_thw).reshape(-1, V_HEAD_DIM // 2).to(torch.float32)
            emb = torch.cat((rotary, rotary), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        conv_w = visual.patch_embed.proj.weight.data.reshape(V_HIDDEN, V_IN_DIM).to(torch.float32)
        conv_b = visual.patch_embed.proj.bias.data.to(torch.float32)
        self.register_buffer("patch_w", conv_w)
        self.register_buffer("patch_b", conv_b)
        self.register_buffer("pos_embed", pos_embed)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.blocks = visual.blocks
        self.merger = visual.merger
        self.deepstack_merger_list = visual.deepstack_merger_list
        self.deepstack_idx = {layer: i for i, layer in enumerate(V_DEEPSTACK_LAYERS)}

    def forward(self, pixel_values):
        """Run the vision blocks and return the merged embedding and deepstack features."""
        hidden = torch.matmul(pixel_values, self.patch_w.t()) + self.patch_b
        hidden = hidden + self.pos_embed
        hidden = hidden.unsqueeze(0)
        ds_out = []
        for layer_num, blk in enumerate(self.blocks):
            hidden = hidden + _cann_vision_attn_forward(blk.attn, blk.norm1(hidden), self.cos, self.sin)
            hidden = hidden + _vision_mlp_forward(blk.mlp, blk.norm2(hidden), self.use_fused_gelu_tanh_nz)
            if layer_num in self.deepstack_idx:
                merger = self.deepstack_merger_list[self.deepstack_idx[layer_num]]
                ds_out.append(merger(hidden).reshape(self.num_merged, V_OUT))
        image_embeds = self.merger(hidden).reshape(self.num_merged, V_OUT)
        return image_embeds, ds_out[0], ds_out[1], ds_out[2]


class Qwen3VLEmbeddingTextImageFused(torch.nn.Module):
    """Text backbone wrapper consuming token ids, mrope positions and image features.

    `image_embeds`/`ds0`/`ds1`/`ds2` are the vision chunk outputs, each shaped
    `[num_merged, hidden]`. They are expanded to the full sequence inside the graph
    (running image-token index + gather + select), which keeps every shape static and
    avoids `NonZero`/dynamic-shape operators that fixed-shape Ascend models reject.
    `use_fused_rms_norm_nz` and `use_fused_qk_norm_rope_bsh` select the CANN fused norms
    and the fused QK-norm + RoPE attention path instead of the plain operator graphs.
    """

    def __init__(self, model, use_fused_rms_norm_nz=False, use_fused_qk_norm_rope_bsh=False):
        """Build the wrapper and pre-compute the mrope T/H/W mixing masks."""
        super().__init__()
        self.use_fused_rms_norm_nz = bool(use_fused_rms_norm_nz)
        self.use_fused_qk_norm_rope_bsh = bool(use_fused_qk_norm_rope_bsh)
        lm = model.language_model
        self.embed_tokens = lm.embed_tokens
        self.layers = lm.layers
        self.norm = lm.norm
        self.num_layers = len(self.layers)
        self.num_deepstack = len(V_DEEPSTACK_LAYERS)
        self.image_token_id = int(model.config.image_token_id)
        rotary = lm.rotary_emb
        inv_freq = rotary.inv_freq.detach().float().cpu()
        mrope_section = list(rotary.mrope_section)
        half = int(inv_freq.shape[0])
        h_mask = torch.zeros(half, dtype=torch.float32)
        w_mask = torch.zeros(half, dtype=torch.float32)
        h_mask[1:min(int(mrope_section[1]) * 3, half):3] = 1.0
        w_mask[2:min(int(mrope_section[2]) * 3, half):3] = 1.0
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("h_mask", h_mask)
        self.register_buffer("w_mask", w_mask)
        self.register_buffer("t_mask", 1.0 - h_mask - w_mask)
        self.scaling = float(getattr(rotary, "attention_scaling", None) or 1.0)
        self.half = half

    def _mrope_cos_sin(self, position_ids):
        """Build mrope cos/sin by mixing the temporal/height/width sections."""
        freqs = position_ids.float()[..., None] * self.inv_freq.view(1, 1, self.half)
        freqs_t = self.t_mask * freqs[0] + self.h_mask * freqs[1] + self.w_mask * freqs[2]
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos() * self.scaling, emb.sin() * self.scaling

    def _image_slots(self, input_ids, dtype):
        """Return the gather slot and the [batch, seq, 1] image-token mask."""
        image_mask = (input_ids == self.image_token_id).to(dtype)
        slot = torch.cumsum(image_mask, dim=1) - 1.0
        slot = (slot * image_mask).to(torch.int64)
        return slot, image_mask.unsqueeze(-1)

    def _rms_norm(self, x, norm_mod):
        """RMS-norm through Custom(FusedRmsNormNZ), or the module directly."""
        eps = _get_rmsnorm_epsilon(norm_mod)
        if self.use_fused_rms_norm_nz:
            return _CannFusedRmsNormNZ.apply(x, norm_mod.weight, eps)
        return norm_mod(x)

    def forward(self, input_ids, attention_mask, position_ids, image_embeds, ds0, ds1, ds2):
        """Run the decoder layers and return the final hidden state."""
        seq_len = input_ids.shape[1]
        cos, sin = self._mrope_cos_sin(position_ids)
        inputs_embeds = self.embed_tokens(input_ids)
        slot, mask_f = self._image_slots(input_ids, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds * (1.0 - mask_f) + image_embeds[slot] * mask_f

        bool_mask = _make_bool_causal_mask(attention_mask, seq_len, seq_len, 0)
        hidden_states = inputs_embeds
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, self.layers[0].input_layernorm)
        ds_list = [ds0, ds1, ds2]

        for i, layer in enumerate(self.layers):
            if self.use_fused_qk_norm_rope_bsh:
                attn_out, _, _ = _cann_attn_forward_bsh_fused_qkrope(
                    layer.self_attn, hidden_states, (cos, sin), bool_mask
                )
            else:
                attn_out, _, _ = _cann_attn_forward_bsh(
                    layer.self_attn, hidden_states, (cos, sin), bool_mask, True,
                )
            x = residual + attn_out
            hidden_states = self._rms_norm(x, layer.post_attention_layernorm)
            residual = x
            mlp_out = _cann_mlp_forward(layer.mlp, hidden_states, False, False)
            if i < self.num_deepstack:
                mlp_out = mlp_out + ds_list[i][slot] * mask_f
            x = residual + mlp_out
            if i < self.num_layers - 1:
                hidden_states = self._rms_norm(x, self.layers[i + 1].input_layernorm)
                residual = x
            else:
                hidden_states = self._rms_norm(x, self.norm)
        return hidden_states


def _export_onnx(model, output_path, dummy_inputs, input_names, output_names, dynamic_axes=None):
    """Export to ONNX, strip IsNaN nodes, and re-save with external data."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx_model = _remove_isnan_nodes(onnx_model)
    data_path = output_path + ".data"
    if os.path.exists(data_path):
        os.remove(data_path)
    onnx.save_model(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(output_path) + ".data",
        size_threshold=1024,
        convert_attribute=True,
    )
    _remove_stale_weight_files(out_dir or ".", output_path, start)
    print(f"Saved ONNX model to {output_path}")


def _remove_stale_weight_files(out_dir, output_path, start):
    """Delete the per-tensor weight files written by the first export pass.

    `torch.onnx.export` spills every weight of a >2 GB model into its own file next to
    the model; the re-save above packs all of them into a single `<name>.onnx.data`
    file, so those leftovers are unreferenced and only waste disk space.
    """
    keep = {os.path.basename(output_path), os.path.basename(output_path) + ".data"}
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if name in keep or not os.path.isfile(path):
            continue
        if os.path.getmtime(path) >= start:
            os.remove(path)


def _print_op_stats(onnx_path):
    """Print a breakdown of the Custom op types in the exported model."""
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    ops = Counter(n.op_type for n in onnx_model.graph.node)
    if ops.get("Custom", 0) == 0:
        print("No Custom ops found")
        return
    type_counts = Counter()
    for node in onnx_model.graph.node:
        if node.op_type != "Custom":
            continue
        for attr in node.attribute:
            if attr.name == "type":
                type_counts[attr.s.decode("utf-8")] += 1
                break
    print(f"Custom op counts: {dict(type_counts)}")


def _export_vision(wrapper, output_dir, name, num_patches, device):
    """Export the vision chunk for a fixed patch count."""
    dummy = (torch.randn(num_patches, V_IN_DIM, dtype=torch.float32, device=device),)
    out_path = os.path.join(output_dir, name)
    _export_onnx(
        wrapper,
        out_path,
        dummy,
        input_names=["pixel_values"],
        output_names=["image_embeds", "ds0", "ds1", "ds2"],
    )
    _print_op_stats(out_path)


def _export_text_image(wrapper, output_dir, name, num_merged, device, batch=1, seq=128):
    """Export the text-image chunk with dummy token, mrope and feature inputs."""
    input_ids = torch.randint(0, 1000, (batch, seq), dtype=torch.int64, device=device)
    num_image_tokens = min(num_merged, max(seq - 2, 1))
    input_ids[:, 2:2 + num_image_tokens] = wrapper.image_token_id
    image_embeds = torch.zeros(num_merged, V_OUT, dtype=torch.float32, device=device)
    dummy = (
        input_ids,
        torch.ones(batch, seq, dtype=torch.int64, device=device),
        torch.arange(seq, dtype=torch.int64, device=device)[None, None, :].repeat(3, batch, 1),
        image_embeds,
        image_embeds.clone(),
        image_embeds.clone(),
        image_embeds.clone(),
    )
    out_path = os.path.join(output_dir, name)
    _export_onnx(
        wrapper,
        out_path,
        dummy,
        input_names=["input_ids", "attention_mask", "position_ids", "image_embeds", "ds0", "ds1", "ds2"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "position_ids": {1: "batch", 2: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )
    _print_op_stats(out_path)


def _parse_args():
    """Parse the export CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL-Embedding-2B image path to ONNX vision/text chunks"
    )
    parser.add_argument("--model-id", type=str, default="./Qwen3-VL-Embedding-2B",
                        help="Local model directory or HuggingFace model id")
    parser.add_argument("--output-dir", type=str, default="./qwen3_vl_embedding_onnx",
                        help="Directory where the ONNX files are written")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device used for the export")
    parser.add_argument("--module", type=str, default="all", choices=["all", "vision", "text"],
                        help="Which artifact to produce: vision chunk, text chunk or both")
    parser.add_argument("--vision-image-size", type=int, default=1024,
                        help="Square image side used to bake the vision position/rotary tables; "
                             "the inference script must use the same image size")
    parser.add_argument("--vision-name", type=str, default="qwen3_vl_embedding_2b_vision.onnx",
                        help="File name of the vision ONNX model")
    parser.add_argument("--text-name", type=str, default="qwen3_vl_embedding_2b_text_image.onnx",
                        help="File name of the text-image ONNX model")
    parser.add_argument("--use-fused-gelu-tanh-nz", action="store_true",
                        help="Emit the vision MLP GELU(tanh) as Custom(FusedGeluTanhNZ); "
                             "off by default, the plain gelu graph is exported")
    parser.add_argument("--use-fused-rms-norm-nz", action="store_true",
                        help="Emit the text RMSNorm as Custom(FusedRmsNormNZ); "
                             "off by default, the RMSNorm module graph is exported")
    parser.add_argument("--use-fused-qk-norm-rope-bsh", action="store_true",
                        help="Emit the text QK-norm plus RoPE as Custom(FusedQKNormRopeBSH); "
                             "off by default, q_norm/k_norm and RotaryMul are exported separately")
    return parser.parse_args()


def _pick_device(device):
    """Validate the requested export device."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use --device cpu")
    return device


def _load_model(model_id, device):
    """Load the checkpoint in float32, which is required by the MindSpore Lite converter."""
    print(f"Loading model from {model_id} ...")
    kwargs = {
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "trust_remote_code": True,
    }
    try:
        model = AutoModel.from_pretrained(model_id, device_map=device, **kwargs)
    except Exception as err:  # pylint: disable=broad-except
        print(f"Loading with device_map={device} failed ({err}), falling back to a plain load")
        model = AutoModel.from_pretrained(model_id, **kwargs)
    model = model.to(device)
    return model.eval()


def _check_model_config(model):
    """Fail fast when the checkpoint does not match the fused constants of this script."""
    vision = model.config.vision_config
    text = model.config.text_config
    mismatches = []
    for name, expected in (
        ("hidden_size", V_HIDDEN),
        ("num_heads", V_HEADS),
        ("intermediate_size", V_INTER),
        ("out_hidden_size", V_OUT),
        ("spatial_merge_size", V_MERGE),
        ("patch_size", V_PATCH),
        ("temporal_patch_size", V_T_PATCH),
        ("in_channels", V_IN_CHANNELS),
    ):
        actual = getattr(vision, name, None)
        if actual is not None and int(actual) != int(expected):
            mismatches.append(f"vision.{name}={actual} (expected {expected})")
    head_dim = int(vision.hidden_size) // int(vision.num_heads)
    if head_dim != V_HEAD_DIM:
        mismatches.append(f"vision head_dim={head_dim} (expected {V_HEAD_DIM})")
    deepstack = tuple(getattr(vision, "deepstack_visual_indexes", ()))
    if deepstack != V_DEEPSTACK_LAYERS:
        mismatches.append(f"vision.deepstack_visual_indexes={deepstack} (expected {V_DEEPSTACK_LAYERS})")
    text_hidden = getattr(text, "hidden_size", None)
    if text_hidden is not None and int(text_hidden) != V_OUT:
        mismatches.append(f"text.hidden_size={text_hidden} (expected {V_OUT})")
    if mismatches:
        raise ValueError("Model config does not match this script: " + "; ".join(mismatches))
    print(f"Model config checked: vision hidden={V_HIDDEN}, head_dim={V_HEAD_DIM}, text hidden={V_OUT}")


def _make_grid_thw(model, vision_image_size, device):
    """Build the fixed (1, grid_h, grid_w) patch grid of a square image."""
    patch_size = int(model.config.vision_config.patch_size)
    merge_size = int(model.config.vision_config.spatial_merge_size)
    size = int(vision_image_size)
    if size % patch_size != 0:
        raise ValueError(f"--vision-image-size must be a multiple of patch_size={patch_size}, got {size}")
    grid_hw = size // patch_size
    if grid_hw % merge_size != 0:
        raise ValueError(
            f"--vision-image-size must be a multiple of patch_size*spatial_merge_size="
            f"{patch_size * merge_size}, got {size}"
        )
    grid_thw = torch.tensor([[1, grid_hw, grid_hw]], dtype=torch.int64, device=device)
    num_patches = int(grid_thw.prod().item())
    num_merged = num_patches // (merge_size * merge_size)
    print(f"Fixed vision grid: grid_thw={grid_thw.tolist()}, patches={num_patches}, merged tokens={num_merged}")
    return grid_thw


def main():
    """Load the checkpoint and export the requested ONNX chunks."""
    args = _parse_args()
    device = _pick_device(args.device)
    model = _load_model(args.model_id, device)
    _check_model_config(model)
    grid_thw = _make_grid_thw(model, args.vision_image_size, device)
    num_patches = int(grid_thw.prod().item())
    num_merged = num_patches // (V_MERGE * V_MERGE)
    print(
        "Fused CANN operators: "
        f"gelu_tanh_nz={args.use_fused_gelu_tanh_nz}, rms_norm_nz={args.use_fused_rms_norm_nz}, "
        f"qk_norm_rope_bsh={args.use_fused_qk_norm_rope_bsh}"
    )

    if args.module in ("all", "vision"):
        wrapper = Qwen3VLEmbeddingVisionFused(model, grid_thw, args.use_fused_gelu_tanh_nz).to(device).eval()
        _export_vision(wrapper, args.output_dir, args.vision_name, num_patches, device)
        del wrapper
        gc.collect()

    if args.module in ("all", "text"):
        wrapper = Qwen3VLEmbeddingTextImageFused(
            model, args.use_fused_rms_norm_nz, args.use_fused_qk_norm_rope_bsh
        ).to(device).eval()
        _export_text_image(wrapper, args.output_dir, args.text_name, num_merged, device)
        del wrapper
        gc.collect()

    print("Export finished.")


if __name__ == "__main__":
    main()
