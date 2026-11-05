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
"""Export Qwen3-Reranker-8B (Qwen3ForCausalLM) to ONNX with CP (Ulysses) sharding.

Two export layouts, selected by --mode:
  * whole   : single-pass whole-graph CP export (legacy; per-rank input
              slice -> Ulysses attention -> last-token logits);
  * prefix/suffix/both (default): common-prefix two-graph export (the prefix
              graph caches the shared prefix K/V once; the suffix graph
              consumes the cache and outputs the global last-token logits;
              zero-copy KV handoff at inference).

Context Parallel (Ulysses Sequence Parallel) splits the SEQUENCE, not the
weights: every rank holds the FULL model weights and only attention runs
cooperatively:
  per rank:  input slice [1, S/P] (its own sequence segment);
  attention: local q/k/v -> AllGather along seq -> full-sequence attention
             with this rank's H/P head subset -> AllGather along heads ->
             slice back to the local segment with ALL heads -> o_proj/MLP
             locally (full weights);
  lm_head:   full weights; each rank scores the LAST TOKEN OF ITS OWN segment
             and the host picks the rank that owns the global last token.

Contract: CP keeps the weights replicated, so only attention activations are
exchanged (two AllGathers per layer) and no AllReduce is emitted.  The
forward/reverse Ulysses exchanges are built from Custom(AllGather)
(HcomAllGather) plus native Transpose/Slice instead of a native all_to_all,
because the CANN 9.0.1 GE runtime registers only four Hcom collectives
(AllReduce / AllGather / Broadcast / ReduceScatter).

Layer forward and Custom ops are aligned with the TP export (prefill path):
RotaryMul (single-input), RmsNorm, PromptFlashAttention (BSND, full attrs),
SwiGlu.

Buckets: suffix 512 / 1024 / 1536 / 2048 via GE dynamicDims
(see configs/qwen3_reranker_8b_suffix.config).

Usage:
  python export_qwen3_reranker_8b_onnx_cp.py \
    --model-id /path/to/Qwen3-Reranker-8B \
    --output-dir ./output/export --device cpu --dtype fp16 --cp-size 2 \
    --mode both          # both = prefix+suffix two graphs (default); whole = whole-graph CP (legacy)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Global CP config
# ---------------------------------------------------------------------------
CP_SIZE = 1
CP_RANK = 0
QUANT_PACK = None


def _as_list_str(items):
    """Convert a list of items to a list of strings for ONNX custom op attributes."""
    return [str(x) for x in items]


def _rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    """Boolean causal + padding mask (batch, 1, q_len, k_len).

    ``attention_mask`` is the FULL-sequence mask [1, S]; the generated 4-D
    mask covers the global (q_len=k_len=S) attention inside USP.
    """
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :] == 0
    return causal | padding


# ---------------------------------------------------------------------------
# Custom ops (aligned with the TP export / qwen3_8b prefill path)
# ---------------------------------------------------------------------------
class _RotaryMulCustom(torch.autograd.Function):
    """Custom RotaryMul op (single-input): x*cos + rotate_half(x)*sin."""

    @staticmethod
    def forward(ctx, x, cos4, sin4):
        del ctx
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """Emit a single Custom(RotaryMul) node."""
        y = g.op(
            "Custom", x, cos4, sin4,
            type_s="RotaryMul",
            input_names_s=_as_list_str(["x", "r1", "r2"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0, 1, 2],
        )
        y.setType(x.type())
        return y


def rotary_mul(x, cos4, sin4):
    return _RotaryMulCustom.apply(x, cos4, sin4)


class _RmsNormCustom(torch.autograd.Function):
    """Custom RmsNorm op with two outputs (y, rstd)."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon: float):
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """Emit a Custom(RmsNorm) node with the epsilon attribute."""
        y, rstd = g.op(
            "Custom", x, gamma,
            type_s="RmsNorm",
            input_names_s=_as_list_str(["x", "gamma"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y", "rstd"]),
            output_num_i=2,
            input_index_i=[0, 1],
            epsilon_f=float(epsilon),
            outputs=2,
        )
        y.setType(x.type())
        return y, rstd


def rms_norm(x, gamma, epsilon: float = 1e-6):
    return _RmsNormCustom.apply(x, gamma, float(epsilon))[0]


def _rms_norm_layer(norm_mod, x):
    """RMSNorm with the full (unsharded) weight -- CP replicates weights."""
    return rms_norm(x, norm_mod.weight)


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom PromptFlashAttention (full-sequence, BSND)."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, sparse_mode, inner_precise,
                pre_tokens, next_tokens):
        """Eager fallback: dense attention reproducing the PFA semantics."""
        del ctx, sparse_mode, inner_precise, pre_tokens, next_tokens
        layout = str(input_layout).upper()
        q = query
        k, v = key, value
        if layout in ("BSND", "SBND"):
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        if 0 < num_key_value_heads < num_heads:
            rep = num_heads // num_key_value_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        else:
            q_len, k_len = attn.shape[-2], attn.shape[-1]
            ar_q = torch.arange(q_len, device=attn.device)
            ar_k = torch.arange(k_len, device=attn.device)
            causal = ar_k[None, :] > ar_q[:, None]
            causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
            attn = attn.masked_fill(causal, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, sparse_mode, inner_precise,
                 pre_tokens, next_tokens):
        """Emit a Custom(PromptFlashAttention) node with the full attribute set."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(3)
        y = g.op(
            "Custom", *base_inputs,
            type_s="PromptFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
            optional_input_names_s=_as_list_str(["atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=base_index,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            pre_tokens_i=int(pre_tokens),
            next_tokens_i=int(next_tokens),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            sparse_mode_i=int(sparse_mode),
            inner_precise_i=int(inner_precise),
        )
        y.setType(query.type())
        return y


def prompt_flash_attention(query, key, value, atten_mask, num_heads, scale_value,
                           input_layout, num_key_value_heads, sparse_mode=0,
                           inner_precise=1, pre_tokens=214748647, next_tokens=0):
    return _PromptFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(sparse_mode), int(inner_precise),
        int(pre_tokens), int(next_tokens))


class _SwiGluCustom(torch.autograd.Function):
    """Fused SiLU-gate activation; output dim halved (must be explicit in setType)."""

    @staticmethod
    def forward(ctx, x, dim: int):
        del ctx
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """Emit a Custom(SwiGlu) node and halve the target dim in its type."""
        y = g.op(
            "Custom", x,
            type_s="SwiGlu",
            input_names_s=_as_list_str(["x"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0],
            dim_i=int(dim),
        )
        # Halve the target dim in the declared output type (from sizes(), NOT
        # .dim() -- symbolic input is a torch._C.Value whose .dim() is
        # unreliable; sizes() gives [int|None|str] per rank).
        try:
            x_type = x.type()
            if not hasattr(x_type, "sizes"):
                return y
            sizes = list(x_type.sizes())
            if sizes is None:
                return y
            x_rank = len(sizes)
            d = int(dim)
            if d < 0:
                d += x_rank
            new_sizes = []
            for i, s in enumerate(sizes):
                val = s if isinstance(s, int) else None
                if i == d and val is not None:
                    val = val // 2
                new_sizes.append(val)
            y.setType(x_type.with_sizes(new_sizes))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


def swiglu(x, dim=-1):
    return _SwiGluCustom.apply(x, int(dim))


class _SwigluDualCustom(torch.autograd.Function):
    """Fused dual-input SwiGLU (Custom -> GE SwigluDual): act = gate*sigmoid(gate)*up.

    Semantics == silu(gate) * up, matching the MLP act_fn(gate_proj) * up_proj.
    Output shape mirrors the gate input (N=inter, NOT halved -- unlike the
    single-input SwiGlu over the 2N concat)."""

    @staticmethod
    def forward(ctx, gate, up):
        del ctx
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def symbolic(g, gate, up):
        """Emit a Custom(SwigluDual) node; output keeps the gate shape."""
        y = g.op(
            "Custom", gate, up,
            type_s="SwigluDual",
            input_names_s=_as_list_str(["gate", "up"]),
            output_names_s=_as_list_str(["act"]),
            output_num_i=1,
        )
        # Output keeps the gate shape (not halved); dims may be None for graph inputs.
        try:
            sizes = list(gate.type().sizes())
            if sizes:
                y.setType(gate.type().with_sizes(sizes))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


def swiglu_dual(gate, up):
    return _SwigluDualCustom.apply(gate, up)


# ---------------------------------------------------------------------------
# CP communication: Custom(AllGather) lowered to GE HcomAllGather
# ---------------------------------------------------------------------------
class _AllGatherCustom(torch.autograd.Function):
    """Custom(AllGather) along dim 0 -> GE HcomAllGather.

    HcomAllGather concatenates the per-rank inputs along dimension 0.  The
    eager fallback repeats the local tensor (trace only); the real gather
    happens at GE runtime, so the declared output shape must be
    ``[dim0 * CP_SIZE, *rest]``.

    Same attribute contract as the TP AllReduce custom op (see
    qwen3_8b/export_qwen3_8b_onnx.py): fusion must be 0 for the converter's
    ConvertHcomFusionId to inject the ``group`` attr, otherwise GE's
    hccl_graph_optimizer aborts with "get attr group failed".
    """

    @staticmethod
    def forward(ctx, x):
        del ctx
        return x.repeat(CP_SIZE, *[1] * (x.dim() - 1))

    @staticmethod
    def symbolic(g, x):
        """Emit a Custom(AllGather) node gathering along dim 0."""
        y = g.op(
            "Custom", x,
            type_s="AllGather",
            input_names_s=_as_list_str(["x"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["y"]),
            output_num_i=1,
            input_index_i=[0],
            group_s="hccl_world_group",
            rank_size_i=int(CP_SIZE),
            fusion_i=0,
        )
        try:
            x_type = x.type()
            sizes = list(x_type.sizes()) if hasattr(x_type, "sizes") else None
            if sizes:
                new_sizes = list(sizes)
                if isinstance(new_sizes[0], int):
                    new_sizes[0] = new_sizes[0] * CP_SIZE
                y.setType(x_type.with_sizes(new_sizes))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


def allgather_dim0(x):
    """AllGather along dim 0 across the CP group (no-op when CP_SIZE == 1)."""
    if CP_SIZE <= 1:
        return x
    return _AllGatherCustom.apply(x)


def _gather_seq_dim(x):
    """Ulysses forward exchange: [B, S/P, H, D] -> [B, S, H, D].

    Equivalent to the forward all_to_all (scatter heads, gather seq):
    AllGather along the SEQUENCE dim.  Implemented as permute(seq->0) +
    Custom(AllGather) + permute back, because HcomAllGather only gathers
    along dimension 0.
    """
    if CP_SIZE <= 1:
        return x
    x_t = x.permute(1, 0, 2, 3).contiguous()      # [S/P, B, H, D]
    x_t = allgather_dim0(x_t)                     # [S, B, H, D]
    return x_t.permute(1, 0, 2, 3).contiguous()   # [B, S, H, D]


def _gather_head_dim(x):
    """Ulysses reverse exchange: [B, S, H/P, D] -> [B, S, H, D].

    Equivalent to the reverse all_to_all (scatter seq, gather heads):
    AllGather along the HEAD dim.  After this, every rank holds the full
    attention output for ALL heads of the full sequence; the caller slices
    back its own sequence segment.
    """
    if CP_SIZE <= 1:
        return x
    x_t = x.permute(2, 0, 1, 3).contiguous()      # [H/P, B, S, D]
    x_t = allgather_dim0(x_t)                     # [H, B, S, D]
    return x_t.permute(1, 2, 0, 3).contiguous()   # [B, S, H, D]


def _slice_local_seq(x):
    """Keep only this rank's sequence segment: [B, S, H, D] -> [B, S/P, H, D]."""
    if CP_SIZE <= 1:
        return x
    s = x.shape[1]
    s_local = s // CP_SIZE
    return x[:, CP_RANK * s_local:(CP_RANK + 1) * s_local, :, :]


# ---------------------------------------------------------------------------
# A8W8 quant support (aligned with the TP export: AscendQuant + QBMV3).
# Quantization is off by default (--quant-mode=off): no --quant-dir = plain fp16.
# CP replicates weights -> quantized linears run at FULL weight (no slicing,
# no AllReduce needed; o_proj/down stay row-local with replicated weights).
# down_proj / lm_head / embeddings are NOT in the AMCT pack (fp16).
# ---------------------------------------------------------------------------
def _load_quant_pack(quant_dir, quant_format="auto"):
    """Load an A8W8 weight pack. Two on-disk layouts are supported:

      * ``amct``        -- AMCT npy pack: manifest.json + <module>__<field>.npy
      * ``safetensors`` -- single safetensors shard plus its index:
                           w8a8_quant.bin + w8a8_quant.bin.index.json

    Both normalise to the same in-memory shape, so everything downstream
    (_quant_for / _quant_proj) is layout-agnostic:
        pack["model.layers.0.self_attn.q_proj"] = {"w_int8": Tensor, ...}
    """
    global QUANT_PACK
    if not quant_dir:
        QUANT_PACK = None
        return
    pack_dir = Path(quant_dir)
    fmt = quant_format
    if fmt == "auto":
        # The safetensors writer also drops a manifest.json copy next to the
        # shard (kept for per-tensor shape/dtype reference), so an index json
        # is the stronger signal: an AMCT npy pack never has one.
        if list(pack_dir.glob("*.index.json")):
            fmt = "safetensors"
        elif (pack_dir / "manifest.json").exists():
            fmt = "amct"
        else:
            raise SystemExit(
                f"[quant] cannot detect pack format in {quant_dir}: found "
                "neither *.index.json (safetensors) nor manifest.json (amct)")
    if fmt == "amct":
        raw = _quant_raw_amct(pack_dir)
    elif fmt == "safetensors":
        raw = _quant_raw_safetensors(pack_dir)
    else:
        raise SystemExit(f"[quant] unknown --quant-format {fmt!r}")

    # Identical tensor conversion for both layouts keeps the two paths consistent.
    pack = {}
    for name, d in raw.items():
        t = {}
        for k, v in d.items():
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(np.ascontiguousarray(v))
            if k in ("w_int8", "bias", "deq_scale"):
                t[k] = v
            elif k == "scale_factor":
                t[k] = v.to(torch.float16).squeeze(0)
            else:
                t[k] = v.float()
        pack[name] = t
    QUANT_PACK = pack
    print(f"[quant] loaded {len(pack)} quantized modules from {quant_dir} "
          f"(format={fmt})")


def _quant_raw_amct(pack_dir):
    """manifest.json + per-tensor npy files (AMCT native pack)."""
    with open(pack_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    raw = {}
    for key, meta in manifest.items():
        name, field = key.split("::")
        raw.setdefault(name, {})[field] = np.load(pack_dir / meta["path"])
    return raw


def _quant_raw_safetensors(pack_dir):
    """<shard>.index.json + safetensors shard(s) (pack_to_safetensors.py).

    weight_map keys use "<module>::<field>" (same grammar as manifest.json);
    a dotted "<module>.<field>" grammar is also accepted.
    """
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit(
            "[quant] the safetensors package is required to read a "
            f"safetensors weight pack: {exc}") from exc
    index_files = sorted(pack_dir.glob("*.index.json"))
    if len(index_files) != 1:
        raise SystemExit(
            f"[quant] expected exactly one *.index.json in {pack_dir}, got "
            f"{[p.name for p in index_files]}")
    with open(index_files[0], encoding="utf-8") as f:
        index = json.load(f)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"[quant] {index_files[0].name}: missing/empty weight_map")

    by_shard = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    raw = {}
    for shard in sorted(by_shard):
        shard_path = pack_dir / shard
        if not shard_path.exists():
            raise SystemExit(
                f"[quant] shard {shard!r} referenced by {index_files[0].name} "
                f"is missing from {pack_dir}")
        tensors = load_file(str(shard_path))
        for key in by_shard[shard]:
            if key not in tensors:
                raise SystemExit(
                    f"[quant] {shard}: tensor {key!r} listed in "
                    f"{index_files[0].name} but absent from the shard")
            if "::" in key:
                name, field = key.split("::", 1)
            else:
                name, field = key.rsplit(".", 1)
            raw.setdefault(name, {})[field] = tensors[key]
    print(f"[quant] shards={sorted(by_shard)} index={index_files[0].name}")
    return raw


def _quant_for(name):
    if not QUANT_PACK:
        return None
    return QUANT_PACK.get(name)


class _AscendQuantCustom(torch.autograd.Function):
    """Custom(AscendQuant): fp16 -> int8; GE scale is multiplicative."""

    @staticmethod
    def forward(_ctx, x, scale, offset):
        xq = torch.clamp(torch.round(x.float() / scale) + offset, -128, 127)
        return xq.to(torch.int8)

    @staticmethod
    def symbolic(g, x, scale, offset):
        y = g.op("Custom", x,
                 type_s="AscendQuant",
                 input_names_s=_as_list_str(["x"]),
                 output_names_s=_as_list_str(["y"]),
                 output_num_i=1, input_index_i=[0],
                 scale_f=float(1.0 / scale), offset_f=float(offset),
                 dst_type_i=2, dtype_i=3)
        return y.setType(x.type().with_dtype(torch.int8))


class _QBMCV3Custom(torch.autograd.Function):
    """Custom(QuantBatchMatmulV3): fused int8 matmul + bias + dequant -> fp16."""

    @staticmethod
    def forward(_ctx, x1, x2, _scale_u64, _offset_f32, bias_i32):
        y = x1.float() @ x2.float()
        y = y * 1.0
        if bias_i32 is not None:
            y = y + bias_i32.float()
        return y.to(torch.float16)

    @staticmethod
    def symbolic(g, x1, x2, scale_u64, offset_f32, bias_i32):
        """Emit a Custom(QuantBatchMatmulV3) node for the static A8W8 path."""
        y = g.op("Custom", x1, x2, scale_u64, offset_f32, bias_i32,
                 type_s="QuantBatchMatmulV3",
                 input_names_s=_as_list_str(["x1", "x2", "scale", "offset", "bias"]),
                 output_names_s=_as_list_str(["y"]),
                 dtype_i=10)
        try:
            x1_t = x1.type()
            x2_t = x2.type()
            s1 = list(x1_t.sizes()) if hasattr(x1_t, "sizes") else None
            s2 = list(x2_t.sizes()) if hasattr(x2_t, "sizes") else None
            if s1 is not None and s2 is not None and len(s1) >= 2 and len(s2) >= 2:
                new_sizes = list(s1[:-1]) + [s2[-1]]
                y.setType(x1_t.with_sizes(new_sizes).with_dtype(torch.float16))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


# ---------------------------------------------------------------------------
# Dynamic per-token A8W8
#
# Differs from the static path only in how activations are quantized:
#   static:  x*sf -> AscendQuant(calibrated scale/offset) -> QBMV3(scale=uint64 deq_scale)
#   dynamic: x*sf -> DynamicQuant(per-token scale) -> QBMV3(scale=fp32 w_scale,
#                                                      pertoken_scale=...)
# Weight side is unchanged (w_int8 is [K,N]); the SmoothQuant factor sf stays on both paths.
#
# Constraints (a wrong value fails with a misleading error):
#   * g.op() interprets the _s/_i/_f suffixes (onnx.helper.make_node does not)
#   * output_names must use the operator IR names: DynamicQuant -> ["y", "scale"]
#   * the per-token scale must be plain float32 w_scale; uint64 fails TBE compilation
# ---------------------------------------------------------------------------
class _DynamicQuantCustom(torch.autograd.Function):
    """Custom(DynamicQuant): fp16 -> (int8, per-token fp32 scale), symmetric quantization."""

    @staticmethod
    def forward(_ctx, x):
        xf = x.float()
        s = (xf.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-12)
        xq = torch.clamp(torch.round(xf / s), -128, 127).to(torch.int8)
        return xq, s.squeeze(-1).to(torch.float32)

    @staticmethod
    def symbolic(g, x):
        """Emit a Custom(DynamicQuant) node with int8 values and fp32 scale."""
        y, sc = g.op("Custom", x,
                     type_s="DynamicQuant",
                     input_names_s=_as_list_str(["x"]),
                     output_names_s=_as_list_str(["y", "scale"]),
                     output_num_i=2, input_index_i=[0],
                     dst_type_i=2,
                     outputs=2)          # multi-output declared by keyword (same as the production RmsNorm)
        y.setType(x.type().with_dtype(torch.int8))
        sc_t = x.type().with_dtype(torch.float32)
        try:
            s = list(x.type().sizes())
            if s is not None and len(s) >= 2:
                sc_t = sc_t.with_sizes(s[:-1])
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        sc.setType(sc_t)
        return y, sc


class _QBMCV3PTokenCustom(torch.autograd.Function):
    """Custom(QuantBatchMatmulV3) with a dynamic per-token activation scale."""

    @staticmethod
    def forward(_ctx, x1, x2, _scale_f32, _offset_f32, bias_i32, _pt_scale):
        y = x1.float() @ x2.float()
        if bias_i32 is not None:
            y = y + bias_i32.float()
        return y.to(torch.float16)

    @staticmethod
    def symbolic(g, x1, x2, scale_f32, offset_f32, bias_i32, pt_scale):
        """Emit a Custom(QuantBatchMatmulV3) node with a per-token scale input."""
        # IR input order: x1, x2, scale, offset, bias, pertoken_scale
        y = g.op("Custom", x1, x2, scale_f32, offset_f32, bias_i32, pt_scale,
                 type_s="QuantBatchMatmulV3",
                 input_names_s=_as_list_str(
                     ["x1", "x2", "scale", "offset", "bias", "pertoken_scale"]),
                 output_names_s=_as_list_str(["y"]),
                 dtype_i=10)
        try:
            x1_t = x1.type()
            x2_t = x2.type()
            s1 = list(x1_t.sizes()) if hasattr(x1_t, "sizes") else None
            s2 = list(x2_t.sizes()) if hasattr(x2_t, "sizes") else None
            if s1 is not None and s2 is not None and len(s1) >= 2 and len(s2) >= 2:
                y.setType(x1_t.with_sizes(list(s1[:-1]) + [s2[-1]])
                          .with_dtype(torch.float16))
        except (RuntimeError, TypeError, ValueError, AttributeError):
            pass
        return y


# Quantization switch (--quant-mode; default off = plain fp16 export, no weight pack):
#   off     (default) no quantization: q/k/v/o/gate/up/down all use fp16
#   dyn     dynamic per-token A8W8 (requires --quant-dir)
#   static  static per-tensor A8W8 (requires --quant-dir)
# main() overrides this from the --quant-mode argument.
QUANT_MODE = "off"

# down_proj quantization switch:
#   auto (default) = quantize when the pack has that down_proj, else fp16
#   1              = force quantization (error when the pack lacks it)
#   0              = force fp16
QUANT_DOWN_PROJ = os.environ.get("DSH_QUANT_DOWN_PROJ", "auto").lower()
# Concat+SwiGLU -> SwigluDual fusion switch (needs an mslite build with the
# swiglu_dual op):
#   0 (default) = fall back to ConcatD + built-in SwiGlu
#   1           = SwigluDual consumes gate/up directly, removing the 24576-wide
#                 ConcatD.  Validated only on the quantized path (gate/up from
#                 QBMV3); on plain fp16 it is wrong and corrupts layer1+ KV.
FUSE_SWIGLU = os.environ.get("DSH_FUSE_SWIGLU", "0").lower() == "1"


def _bias_or_zeros(bias, w_int8, device):
    """Return zeros(int32) when the weight pack has no bias.

    Passing None in a g.op input slot makes torch.onnx.export **segfault**
    (SIGSEGV, not a catchable exception).  Omitting the input is also wrong:
    pertoken_scale would then land in the IR bias slot.

    bias is int32 with length = output dim = w_int8.shape[1] ([in, out] layout).
    """
    if bias is None:
        return torch.zeros(int(w_int8.shape[-1]), dtype=torch.int32,
                           device=device)
    return bias.to(device)


def _quant_proj_dyn(x, qd):
    """Dynamic per-token quantized linear (full CP weight, no slicing)."""
    w = qd["w_int8"]
    sf = qd["scale_factor"]
    wscale = qd["w_scale"]                 # fp32 [n]; the dynamic path uses w_scale
    bias = qd.get("bias")
    _dev = x.device
    w = w.to(_dev).to(torch.int8)
    sf = sf.to(_dev)
    wscale = wscale.to(_dev).to(torch.float32)
    bias = _bias_or_zeros(bias, w, _dev)
    in_shape = x.shape
    x = x * sf                              # SmoothQuant smoothing (identical on both paths)
    x2d = x.reshape(-1, in_shape[-1])       # per-token is per-row
    xq, pt_scale = _DynamicQuantCustom.apply(x2d)
    offset_zeros = torch.zeros(wscale.shape, dtype=torch.float32, device=_dev)
    y = _QBMCV3PTokenCustom.apply(xq, w, wscale, offset_zeros, bias, pt_scale)
    return y.reshape(*in_shape[:-1], -1)


def _quant_proj_static(x, qd):
    """Static per-tensor quantized linear (kept for comparison; full CP weight)."""
    w = qd["w_int8"]
    sf = qd["scale_factor"]
    scale_d = float(qd["scale_d"][0]) if qd["scale_d"].dim() else float(qd["scale_d"])
    offset = float(qd["act_offset"][0]) if qd["act_offset"].dim() else float(qd["act_offset"])
    bias = qd.get("bias")
    deq = qd["deq_scale"]
    _dev = x.device
    w = w.to(_dev)
    sf = sf.to(_dev)
    deq = deq.to(_dev)
    bias = _bias_or_zeros(bias, w, _dev)
    in_shape = x.shape
    x = x * sf
    xq = _AscendQuantCustom.apply(x, scale_d, offset)
    offset_zeros = torch.zeros(deq.shape, dtype=torch.float32)
    y = _QBMCV3Custom.apply(xq.reshape(-1, in_shape[-1]), w.to(torch.int8),
                            deq, offset_zeros, bias)
    return y.reshape(*in_shape[:-1], -1)


def _quant_proj(x, qd):
    """Quantized linear dispatcher: --quant-mode=dyn | static; off raises (unreachable)."""
    if QUANT_MODE == "off":
        raise SystemExit(
            "[quant] --quant-mode=off 但加载了权重包：开量化请设置 "
            "--quant-mode=dyn|static（默认关闭，直接 fp16 导出）")
    if QUANT_MODE == "static":
        return _quant_proj_static(x, qd)
    return _quant_proj_dyn(x, qd)


# ---------------------------------------------------------------------------
# Layer forward (CP / Ulysses)
# ---------------------------------------------------------------------------
def _compute_qkv(attn_mod, hidden_states, layer_idx=0):
    """QKV projection with FULL weights (CP replicates weights).

    Quantized path: q/k/v run separately through AscendQuant+QBMV3 at full
    weight; otherwise the original fused fp16 linear is kept.
    """
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    hidden_shape = (*input_shape, -1, head_dim)

    pfx = "model.layers.%d.self_attn" % layer_idx
    qd_q, qd_k, qd_v = (_quant_for(pfx + ".q_proj"), _quant_for(pfx + ".k_proj"),
                        _quant_for(pfx + ".v_proj"))
    if qd_q and qd_k and qd_v:
        q = _quant_proj(hidden_states, qd_q)
        k = _quant_proj(hidden_states, qd_k)
        v = _quant_proj(hidden_states, qd_v)
        qkv = torch.cat([q, k, v], dim=-1)
        # AMCT pack stores weights as [in, out]: output dim is shape[1].
        q_out = int(qd_q["w_int8"].shape[1])
        kv_out = int(qd_k["w_int8"].shape[1])
    else:
        qkv = torch.nn.functional.linear(
            hidden_states,
            torch.cat([attn_mod.q_proj.weight, attn_mod.k_proj.weight,
                       attn_mod.v_proj.weight], dim=0),
            torch.cat([attn_mod.q_proj.bias, attn_mod.k_proj.bias,
                       attn_mod.v_proj.bias], dim=0)
            if attn_mod.q_proj.bias is not None else None)
        q_out = int(attn_mod.q_proj.weight.shape[0])
        kv_out = int(attn_mod.k_proj.weight.shape[0])
    query = qkv[..., :q_out].view(hidden_shape)
    key = qkv[..., q_out:q_out + kv_out].view(hidden_shape)
    value = qkv[..., q_out + kv_out:].view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query = _rms_norm_layer(attn_mod.q_norm, query)
    if hasattr(attn_mod, "k_norm"):
        key = _rms_norm_layer(attn_mod.k_norm, key)
    return query, key, value, input_shape


def _run_usp_attention(query, key, value, attention_mask, num_heads,
                       num_kv_heads, scaling):
    """Ulysses attention: gather seq -> head-sharded global PFA -> gather heads.

    ``query/key/value`` are the LOCAL segment [1, S/P, H, D]; the FULL-seq
    ``attention_mask`` [1, S] builds the global (S, S) flash mask.
    """
    # Head counts come from the tensor shapes; the args only keep the caller signature.
    del num_heads, num_kv_heads
    # Ulysses forward exchange: local seq -> global seq; the head subset is taken
    # after the gather, by slicing.
    q_full = _gather_seq_dim(query)   # [1, S, H, D]
    k_full = _gather_seq_dim(key)
    v_full = _gather_seq_dim(value)

    # Each rank computes H/P heads of the global attention.  The Q-head and
    # KV-head subsets are indexed SEPARATELY: GQA has fewer KV heads, so the
    # rank's KV slice is [r*Hkv/P, (r+1)*Hkv/P), not the Q indices.
    h = q_full.shape[2]
    kv_h = k_full.shape[2]
    h_local = h // CP_SIZE
    kv_local = kv_h // CP_SIZE
    hs, he = CP_RANK * h_local, (CP_RANK + 1) * h_local
    kvs, kve = CP_RANK * kv_local, (CP_RANK + 1) * kv_local
    q_heads = q_full[:, :, hs:he, :]
    k_heads = k_full[:, :, kvs:kve, :]
    v_heads = v_full[:, :, kvs:kve, :]
    n_kv_pfa = kv_local

    q_len, k_len = q_heads.shape[1], k_heads.shape[1]
    flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
    attn_output = prompt_flash_attention(
        q_heads, k_heads, v_heads, atten_mask=flash_mask,
        num_heads=int(h_local), scale_value=float(scaling),
        input_layout="BSND", num_key_value_heads=int(n_kv_pfa),
        sparse_mode=0, inner_precise=1)

    # Ulysses reverse exchange: gather all heads of the global sequence,
    # then slice back to this rank's segment.
    out_full = _gather_head_dim(attn_output)     # [1, S, H, D]
    return _slice_local_seq(out_full)            # [1, S/P, H, D]


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask, layer_idx=0):
    """Self-attention block: QKV -> RoPE -> CP (Ulysses) PFA -> o_proj."""
    query, key, value, input_shape = _compute_qkv(attn_mod, hidden_states, layer_idx)
    query = rotary_mul(query, cos4, sin4)
    key = rotary_mul(key, cos4, sin4)
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    attn_output = _run_usp_attention(query, key, value, attention_mask,
                                     num_heads, num_kv_heads,
                                     getattr(attn_mod, "scaling",
                                             1.0 / (attn_mod.head_dim ** 0.5)))
    out = attn_output.reshape(*input_shape, -1)
    # o_proj with the FULL weight (CP replicates weights; no AllReduce).
    qd_o = _quant_for("model.layers.%d.self_attn.o_proj" % layer_idx)
    if qd_o:
        return _quant_proj(out, qd_o)
    return torch.nn.functional.linear(out, attn_mod.o_proj.weight,
                                      attn_mod.o_proj.bias)


def _run_mlp(layer, hidden_states, layer_idx=0):
    """SwiGLU MLP: gate/up (quant or fp16) -> SwiGlu -> down_proj."""
    mlp = layer.mlp
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
        pfx = "model.layers.%d.mlp" % layer_idx
        qd_g, qd_u = _quant_for(pfx + ".gate_proj"), _quant_for(pfx + ".up_proj")
        if qd_g and qd_u:
            gate = _quant_proj(hidden_states, qd_g)
            up = _quant_proj(hidden_states, qd_u)
        else:
            gate = torch.nn.functional.linear(hidden_states, mlp.gate_proj.weight,
                                              mlp.gate_proj.bias)
            up = torch.nn.functional.linear(hidden_states, mlp.up_proj.weight,
                                            mlp.up_proj.bias)
        if FUSE_SWIGLU:
            # SwigluDual accepts only fp16: cast the unquantized (fp32) branch first
            if gate.dtype != torch.float16:
                gate = gate.half()
            if up.dtype != torch.float16:
                up = up.half()
            act = swiglu_dual(gate, up)
        else:
            act = swiglu(torch.cat([gate, up], dim=-1), dim=-1)
        # down_proj: switch DSH_QUANT_DOWN_PROJ = auto|1|0 (see the constant above)
        qd_d = _quant_for(pfx + ".down_proj")
        if QUANT_DOWN_PROJ == "1" or (QUANT_DOWN_PROJ == "auto" and qd_d is not None):
            if qd_d is None:
                raise SystemExit(
                    "[quant] DSH_QUANT_DOWN_PROJ=1 但权重包里没有 "
                    f"{pfx}.down_proj（当前包只量化了 gate/up/q/k/v/o）")
            return _quant_proj(act, qd_d)
        if QUANT_DOWN_PROJ not in ("auto", "0", "1"):
            raise SystemExit(
                f"[quant] DSH_QUANT_DOWN_PROJ 取值非法: {QUANT_DOWN_PROJ!r}"
                "（应为 auto/1/0）")
        # down_proj stays fp16 (full weight, CP-replicated)
        return torch.nn.functional.linear(act, mlp.down_proj.weight, mlp.down_proj.bias)
    return mlp(hidden_states)


class Qwen3RerankerCP(torch.nn.Module):
    """Reranker forward, CP (Ulysses) sharded, single pass, last-token logits out.

    Each rank receives its own sequence segment (input_ids/position_ids are
    [1, S/P]) plus the FULL attention_mask [1, S].  The output is the
    last-token-of-my-segment logits [1, 1, vocab]; the host side picks the
    rank that owns the global last token (with left padding that is always
    the highest rank with a non-zero mask, i.e. normally rank CP_SIZE-1).
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.config = model.config

    def forward(self, input_ids, attention_mask, position_ids):
        """Run the CP-sharded backbone once and return this segment's logits."""
        # input_ids / position_ids: this rank's segment [1, S/P]
        # attention_mask: FULL sequence [1, S] (host feeds the global mask)
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out = _text_attn_forward(layer.self_attn, hidden_states, cos4, sin4,
                                          attention_mask, i)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, hidden_states, i)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        # Last token of MY segment; the host picks the owning rank.
        last_hidden = hidden_states[:, -1:, :]
        logits = torch.nn.functional.linear(last_hidden, self.lm_head.weight,
                                            self.lm_head.bias)
    # fmt: off
        return logits.float()
    # fmt: on


# ---------------------------------------------------------------------------
# Export driver
# ---------------------------------------------------------------------------
def _create_dummy_inputs(tokenizer, seq_len, cp_rank, cp_size):
    """Dummy inputs: rank's segment of a full-seq sample + the global mask."""
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets "
        "the requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    p_ids = tokenizer.encode(prefix, add_special_tokens=False)
    s_ids = tokenizer.encode(suffix, add_special_tokens=False)
    body = "<Instruct>: test instruction\n<Query>: test query\n<Document>: test document"
    b_ids = tokenizer.encode(body, add_special_tokens=False)
    ids = (p_ids + b_ids + s_ids)[:seq_len]
    pad = seq_len - len(ids)
    if pad < 0:
        pad = 0
        ids = ids[:seq_len]
    ids = ids + [tokenizer.pad_token_id or tokenizer.eos_token_id] * pad
    attn = [1] * (seq_len - pad) + [0] * pad

    # split the sequence into per-rank segments (contiguous, in order)
    seg = seq_len // cp_size
    ids_local = ids[cp_rank * seg:(cp_rank + 1) * seg]
    pos_local = list(range(cp_rank * seg, (cp_rank + 1) * seg))
    return (torch.tensor([ids_local], dtype=torch.long),
            torch.tensor([attn], dtype=torch.long),       # FULL mask on every rank
            torch.tensor([pos_local], dtype=torch.long))


def _export_rank(model, tokenizer, out_dir, seq_len):
    """Export this rank's single-pass ONNX graph and post-process it."""
    reranker = Qwen3RerankerCP(model, model.lm_head).eval()
    dummy = _create_dummy_inputs(tokenizer, seq_len, CP_RANK, CP_SIZE)
    rank_dir = os.path.join(out_dir, "rank%d" % CP_RANK)
    os.makedirs(rank_dir, exist_ok=True)
    out_path = os.path.join(rank_dir, "qwen3_reranker_8b_rank%d.onnx" % CP_RANK)

    # input_ids / position_ids are per-rank segments [1, S/P];
    # attention_mask is the full sequence [1, S].
    seg_len = seq_len // CP_SIZE
    dynamic = {"input_ids": {1: "seq_local"},
               "attention_mask": {1: "seq_full"},
               "position_ids": {1: "seq_local"},
               "logits": {1: "last"}}
    print(f"Exporting rank{CP_RANK} (full_seq={seq_len}, local_seq={seg_len}) -> {out_path}",
          flush=True)
    # dynamo=False selects the legacy (TorchScript) exporter: torch >= 2.9
    # defaults to the dynamo exporter, which cannot emit the Custom-node graph
    # the ascend_oriented converter expects.
    with torch.no_grad():
        export_kwargs = {
            "export_params": True,
            "opset_version": 18,
            "do_constant_folding": True,
            "input_names": ["input_ids", "attention_mask", "position_ids"],
            "output_names": ["logits"],
            "dynamic_axes": dynamic,
        }
        try:
            torch.onnx.export(reranker, dummy, str(out_path), dynamo=False,
                              **export_kwargs)
        except TypeError:
            # torch < 2.7 has no `dynamo` kwarg (legacy exporter is default)
            torch.onnx.export(reranker, dummy, str(out_path), **export_kwargs)
    print(f"rank{CP_RANK} exported", flush=True)
    _set_allow_nz_on_matmul(str(out_path))
    _inline_postproc(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Post-export postproc (aligned with the TP export)
# ---------------------------------------------------------------------------
def _set_allow_nz_on_matmul(onnx_path):
    """Set allow_nz=true on MatMul nodes so GE may use FRACTAL_NZ.

    Aligned with qwen3_8b export (IsAllowNzMatmul in ops_kernel_common.cc).
    Header-only rewrite: initializers keep data_location=EXTERNAL untouched.
    """
    from onnx import helper
    m = onnx.load(onnx_path, load_external_data=False)
    added = 0
    for node in m.graph.node:
        if node.op_type in ("MatMul", "MatMulV2", "BatchMatMulV2"):
            if any(a.name == "allow_nz" for a in node.attribute):
                continue
            node.attribute.append(helper.make_attribute("allow_nz", 1))
            added += 1
    if added:
        onnx.save(m, onnx_path)
    print(f"[allow_nz] {onnx_path}: set allow_nz=true on {added} MatMul nodes")


def _read_external_tensor(t, base_dir):
    """Manually read an external-data tensor (bypasses numpy_helper checker)."""
    loc, offset, length = "", 0, 0
    for ed in t.external_data:
        if ed.key == "location":
            loc = ed.value
        elif ed.key == "offset":
            offset = int(ed.value)
        elif ed.key == "length":
            length = int(ed.value)
    with open(os.path.join(base_dir, loc), "rb") as f:
        f.seek(offset)
        raw = f.read(length if length > 0 else -1)
    dt = {onnx.TensorProto.INT64: np.int64, onnx.TensorProto.INT32: np.int32,
          onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16,
          onnx.TensorProto.INT8: np.int8, onnx.TensorProto.UINT64: np.uint64}
    return np.frombuffer(raw, dtype=dt[t.data_type]).reshape(list(t.dims))


def _postproc_scale_uint64(m, base_dir):
    """Rewrite QBMV3 deq_scale int64 Constant -> uint64 initializer.

    torch.onnx.export has no uint64; the GE kernel needs uint64
    ((16384<<32)|f32_bits). Aligned with qwen3_8b export.
    """
    g = m.graph
    prod = {n.output[0]: n for n in g.node if len(n.output) > 0}
    init_names = {i.name for i in g.initializer}
    n_fixed = 0
    for n in list(g.node):
        if n.op_type == "Custom" and any(
                a.name == "type" and a.s == b"QuantBatchMatmulV3" for a in n.attribute):
            if len(n.input) < 3:
                continue
            scale_name = n.input[2]
            const_node = prod.get(scale_name)
            if const_node is None or const_node.op_type != "Constant":
                continue
            t = const_node.attribute[0].t
            if t.data_type != onnx.TensorProto.INT64:
                continue
            if t.data_location == onnx.TensorProto.EXTERNAL:
                val = _read_external_tensor(t, base_dir).astype(np.uint64)
            else:
                val = onnx.numpy_helper.to_array(t).astype(np.uint64)
            init = onnx.helper.make_tensor(scale_name, onnx.TensorProto.UINT64,
                                            list(val.shape), val.tolist())
            g.node.remove(const_node)
            if scale_name not in init_names:
                g.initializer.append(init)
                init_names.add(scale_name)
            n_fixed += 1
    return n_fixed


def _inline_postproc(path):
    """Post-export: QBMV3 deq_scale int64 -> uint64 (raw protobuf, bypass checker)."""
    import time as _time
    m = onnx.ModelProto()
    for _ in range(10):
        try:
            with open(str(path), "rb") as f:
                m.ParseFromString(f.read())
            break
        except Exception:
            _time.sleep(1)
    else:
        raise RuntimeError(f"inline postproc: onnx parse failed after retries: {path}")
    n = _postproc_scale_uint64(m, os.path.dirname(os.path.abspath(str(path))))
    with open(str(path), "wb") as f:
        f.write(m.SerializeToString())
    if n:
        print(f"  [postproc-inline] {n} QBMM scales -> uint64 @ {path}")



# ---------------------------------------------------------------------------
# Global config (set by main per rank)
# ---------------------------------------------------------------------------
CP_SIZE = 1
CP_RANK = 0
PREFIX_LEN = 128
SUFFIX_LEN = 384

# Prefix-graph attention op: "pfa" = canonical PromptFlashAttention (default),
# "innerpfa" = InnerPromptFlashAttention (qwen3_8b / suffix style).  The GE dnt
# dynamic path mis-executes canonical PFA here (KV layer1+ corrupt).
PREFIX_ATTN = os.environ.get("PREFIX_ATTN", "pfa")

# Debug mode: export extra intermediate tensors of layer 0 for pinpointing
# the first node where GE dynamic path diverges from static reference.
# Enable with DEBUG_OUTPUT=1.
DEBUG_OUTPUT = os.environ.get("DEBUG_OUTPUT", "0") == "1"


def _prefix_bucket():
    """Pad the real prefix length up to the next 128-aligned bucket.

    The 310P PromptFlashAttention tiling only accepts 128-aligned equal-length
    masked attention; the real prefix P is padded with mask=0 rows in the
    graph, and the host truncates the KV cache back to P rows afterwards.
    """
    b = max(128, -(-PREFIX_LEN // 128) * 128)
    return b


def _set_allow_nz_raw(path):
    """Set allow_nz=true on MatMul nodes WITHOUT re-saving via onnx.save.

    onnx.load + onnx.save rewrites external-data location metadata, which
    corrupts the many externalized Constant weights of a quantized graph
    (converter then reads whole files -> size mismatch).  Raw protobuf edit
    keeps every external tensor untouched (same trick as _inline_postproc).
    """
    import time as _time
    m = onnx.ModelProto()
    for _ in range(10):
        try:
            with open(str(path), "rb") as f:
                m.ParseFromString(f.read())
            break
        except Exception:
            _time.sleep(1)
    else:
        raise RuntimeError(f"allow_nz: onnx parse failed: {path}")
    added = 0
    for node in m.graph.node:
        if node.op_type in ("MatMul", "MatMulV2", "BatchMatMulV2"):
            if any(a.name == "allow_nz" for a in node.attribute):
                continue
            a = node.attribute.add()
            a.name = "allow_nz"
            a.i = 1
            added += 1
    if added:
        with open(str(path), "wb") as f:
            f.write(m.SerializeToString())
    print(f"[allow_nz-raw] {path}: set allow_nz on {added} MatMul nodes")


class _InnerPFACustom(torch.autograd.Function):
    """InnerPromptFlashAttention (qwen3_8b / TP-split signature).

    Mask slot index 4 (pse_shift slot 3 left unset); native GQA via
    num_key_value_heads (no manual KV expansion); supports Qs != Kvs.
    """

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads, scale_value,
                input_layout, num_key_value_heads, sparse_mode, inner_precise,
                pre_tokens, next_tokens):
        """Eager fallback: dense attention reproducing InnerPFA semantics."""
        del ctx, sparse_mode, inner_precise, pre_tokens, next_tokens
        layout = str(input_layout).upper()
        q = query
        k, v = key, value
        if layout in ("BSND", "SBND"):
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
        if 0 < num_key_value_heads < num_heads:
            rep = num_heads // num_key_value_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * float(scale_value)
        if atten_mask is not None:
            m = atten_mask.to(torch.bool)
            if m.dim() == 4 and m.shape[1] == 1:
                m = m.expand(attn.shape[0], attn.shape[1], m.shape[2], m.shape[3])
            attn = attn.masked_fill(m, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads, scale_value,
                 input_layout, num_key_value_heads, sparse_mode, inner_precise,
                 pre_tokens, next_tokens):
        """Emit a Custom(InnerPromptFlashAttention) node."""
        base_inputs = [query, key, value]
        base_index = [0, 1, 2]
        if atten_mask is not None:
            base_inputs.append(atten_mask)
            base_index.append(4)  # pse_shift slot 3 left unset
        y = g.op(
            "Custom", *base_inputs,
            type_s="InnerPromptFlashAttention",
            input_names_s=_as_list_str(["query", "key", "value", "pse_shift", "atten_mask"]),
            optional_input_names_s=_as_list_str(["pse_shift", "atten_mask"]),
            output_names_s=_as_list_str(["attention_out"]),
            output_num_i=1,
            input_index_i=base_index,
            num_heads_i=int(num_heads),
            scale_value_f=float(scale_value),
            pre_tokens_i=int(pre_tokens),
            next_tokens_i=int(next_tokens),
            input_layout_s=str(input_layout),
            num_key_value_heads_i=int(num_key_value_heads),
            sparse_mode_i=int(sparse_mode),
            inner_precise_i=int(inner_precise),
        )
        y.setType(query.type())
        return y


def _inner_pfa(query, key, value, atten_mask, num_heads, scale_value,
               input_layout, num_key_value_heads, sparse_mode=0,
               inner_precise=1, pre_tokens=214748647, next_tokens=0):
    return _InnerPFACustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout),
        int(num_key_value_heads), int(sparse_mode), int(inner_precise),
        int(pre_tokens), int(next_tokens))


# ---------------------------------------------------------------------------
# Shared per-layer pieces (full replicated weights, quant aware)
# ---------------------------------------------------------------------------
def _o_proj_full(out, attn_mod, layer_idx):
    """o_proj with the FULL weight (CP replicates; no AllReduce)."""
    qd_o = _quant_for("model.layers.%d.self_attn.o_proj" % layer_idx)
    if qd_o:
        return _quant_proj(out, qd_o)
    return torch.nn.functional.linear(out, attn_mod.o_proj.weight,
                                      attn_mod.o_proj.bias)


def _qkv_rotated(attn_mod, h, cos4, sin4, layer_idx):
    """Chunk q/k/v with rotary applied (full heads, quant aware)."""
    query, key, value, _ = _compute_qkv(attn_mod, h, layer_idx)
    query = rotary_mul(query, cos4, sin4)
    key = rotary_mul(key, cos4, sin4)
    return query, key, value


# ---------------------------------------------------------------------------
# Prefix graph (canonical CP structure, prefix NOT split across ranks)
# ---------------------------------------------------------------------------
class Qwen3RerankerCPPrefix(torch.nn.Module):
    """CP prefix graph: the FULL shared prefix on every rank -> K/V caches.

    Same per-layer structure as the CP whole graph, with the row
    (sequence) split removed:
      * every rank feeds the full prefix span  ids/mask/pos [1, P];
      * per layer: local full q/k/v (replicated weights) -> head subset
        (16 q / 4 kv for CP_SIZE=2) -> PromptFlashAttention over the full
        prefix (equal-length P x P, bool causal mask, exactly like the whole
        graph) -> AllGather the head subset back to 32 heads -> o_proj/MLP;
      * outputs the per-layer rotary'd K/V of the FULL prefix (all 8 kv
        heads) stacked as [36, 1, P, 8, head_dim] (identical on both ranks;
        the suffix graphs slice their own kv subset per layer).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.config = model.config

    def forward(self, input_ids, attention_mask, position_ids):
        """Run the full prefix on every rank and emit per-layer K/V caches."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        n_heads = int(self.config.num_attention_heads)
        kv_heads = int(self.config.num_key_value_heads)
        scaling = float(getattr(self.model.layers[0].self_attn, "scaling",
                                1.0 / (self.model.layers[0].self_attn.head_dim ** 0.5)))
        present_k, present_v = [], []
        debug = {}
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            h = _rms_norm_layer(layer.input_layernorm, hidden_states)
            query, key, value, _ = _compute_qkv(layer.self_attn, h, i)
            query = rotary_mul(query, cos4, sin4)
            key = rotary_mul(key, cos4, sin4)
            # cache the FULL rotary'd K/V of every prefix row (all kv heads)
            present_k.append(key.contiguous())
            present_v.append(value.contiguous())
            # Prefix graph: full-head attention, no head split and no AllGather.
            # Every rank computes all 32 q-heads / 8 kv-heads locally, so there is
            # NO inter-rank communication in this graph.
            if PREFIX_ATTN == "innerpfa":
                flash_mask = _make_poskey_mask(attention_mask, position_ids, 0)
                attn_out = _inner_pfa(
                    query, key, value, atten_mask=flash_mask,
                    num_heads=int(n_heads), scale_value=scaling,
                    input_layout="BSND", num_key_value_heads=int(kv_heads),
                    sparse_mode=0, inner_precise=1)
            else:
                flash_mask = _make_flash_attn_mask(attention_mask,
                                                      query.shape[1], key.shape[1], 0)
                attn_out = prompt_flash_attention(
                    query, key, value, atten_mask=flash_mask,
                    num_heads=n_heads, scale_value=scaling,
                    input_layout="BSND", num_key_value_heads=kv_heads,
                    sparse_mode=0, inner_precise=1)
            if DEBUG_OUTPUT and i == 0:
                debug["l0_input_embeds"] = inputs_embeds.contiguous()
                debug["l0_h_rmsnorm"] = h.contiguous()
                debug["l0_q_rotary"] = query.contiguous()
                debug["l0_k_rotary"] = key.contiguous()
                debug["l0_v"] = value.contiguous()
                debug["l0_attn_out"] = attn_out.contiguous()
            hidden_states = residual + _o_proj_full(
                attn_out.reshape(*h.shape[:-1], -1), layer.self_attn, i)
            if DEBUG_OUTPUT and i == 0:
                debug["l0_hidden_after_o_proj"] = hidden_states.contiguous()
            residual = hidden_states
            h = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, h, i)
            if DEBUG_OUTPUT and i == 0:
                debug["l0_hidden_after_mlp"] = hidden_states.contiguous()
        k_out = torch.cat([t.unsqueeze(0) for t in present_k], dim=0)
        v_out = torch.cat([t.unsqueeze(0) for t in present_v], dim=0)
        if DEBUG_OUTPUT:
            # flat tuple for ONNX export compatibility
            return (k_out, v_out,
                    debug["l0_input_embeds"],
                    debug["l0_h_rmsnorm"],
                    debug["l0_q_rotary"],
                    debug["l0_k_rotary"],
                    debug["l0_v"],
                    debug["l0_attn_out"],
                    debug["l0_hidden_after_o_proj"],
                    debug["l0_hidden_after_mlp"])
        return k_out, v_out


# ---------------------------------------------------------------------------
# Position-table causal+padding float mask (suffix graph).
#
# The host feeds the PADDED full suffix span on every rank (input_ids /
# position_ids length S, attention_mask length prefix_bucket + S).  The CP
# split happens INSIDE the graph; row indices therefore do not imply
# positions, so the causal mask is built from an explicit position table:
#   keys_pos = concat( arange(prefix_bucket), position_ids )   (P+S)
#   q_pos    = position_ids                                    (S)
#   block j from row i  <=>  keys_pos[j] > q_pos[i]
# Padding is expressed with attention_mask (0 columns are never attended).
# This is the TP-infer convention: the host right-aligns the real request
# inside the bucket (leading padding in the suffix span), so the LAST row of
# the last CP chunk is always the real final token.
# ---------------------------------------------------------------------------
def _make_poskey_mask(attention_mask, position_ids, past_len):
    """(causal_f + padding_f) > 0.5  [B, 1, q_len, k_len]."""
    ar_p = torch.arange(past_len, device=attention_mask.device)
    keys_pos = torch.cat([ar_p, position_ids.reshape(-1)], dim=0)  # [P+S]
    q_pos = position_ids.reshape(-1)                                # [S]
    causal = (keys_pos[None, :] > q_pos[:, None]).to(torch.float32)
    causal = causal[None, None, :, :]
    padding = 1.0 - attention_mask[:, None, None, :].to(torch.float32)
    return (causal + padding) > 0.5


# ---------------------------------------------------------------------------
# Suffix graph (canonical CP structure; CP split happens INSIDE the graph)
# ---------------------------------------------------------------------------
class Qwen3RerankerCPSuffix(torch.nn.Module):
    """CP suffix graph.

    Host contract (TP-infer style):
      * input_ids / position_ids [1, S] = the PADDED full suffix span, the
        SAME tensor is sent to every rank; the graph slices this rank's
        S/CP rows itself;
      * attention_mask [1, prefix_bucket + S]: keys of the full span (prefix
        bucket columns + suffix columns; 0 = padding);
      * position_ids carry the REAL global positions (suffix real tokens are
        P..P+real_len-1, right-aligned inside the bucket; padding rows are
        masked out and may carry any position);
      * the real final token is the last row of the last CP chunk, whose
        owner is rank CP_SIZE-1 (its graph output is used).

    Per layer (canonical):
      this rank's chunk rows are AllGathered into the full S rows, head
      subset (16 q / 4 kv) attends the cached prefix K/V + gathered suffix
      K/V (InnerPromptFlashAttention, float position-table mask), head subset
      is gathered back to 32 heads over the full S rows and sliced back to
      this rank's chunk; o_proj / MLP run chunk-local.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model.model
        self.lm_head = lm_head
        self.config = model.config

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run this rank's suffix chunk against the cached prefix K/V."""
        seg = input_ids.shape[1] // CP_SIZE
        # in-graph CP split: this rank's contiguous S/CP rows of the padded span
        ids = input_ids[:, CP_RANK * seg:(CP_RANK + 1) * seg]
        pos = position_ids[:, CP_RANK * seg:(CP_RANK + 1) * seg]
        inputs_embeds = self.model.embed_tokens(ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, pos)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin
        hidden_states = inputs_embeds
        n_heads = int(self.config.num_attention_heads)
        kv_heads = int(self.config.num_key_value_heads)
        h_local = n_heads // CP_SIZE
        kv_local = kv_heads // CP_SIZE
        hs, he = CP_RANK * h_local, (CP_RANK + 1) * h_local
        kvs, kve = CP_RANK * kv_local, (CP_RANK + 1) * kv_local
        scaling = float(getattr(self.model.layers[0].self_attn, "scaling",
                                1.0 / (self.model.layers[0].self_attn.head_dim ** 0.5)))
        past_len = past_key_cache.shape[2]
        for i, layer in enumerate(self.model.layers):
            residual = hidden_states
            h = _rms_norm_layer(layer.input_layernorm, hidden_states)
            query, key, value, _ = _compute_qkv(layer.self_attn, h, i)
            query = rotary_mul(query, cos4, sin4)
            key = rotary_mul(key, cos4, sin4)
            # canonical row exchange: local suffix chunk -> full S rows
            q_full = _gather_seq_dim(query)      # [1, S, 32, D]
            k_suf = _gather_seq_dim(key)         # [1, S, 8, D]
            v_suf = _gather_seq_dim(value)
            # per-layer prefix cache slice -> this rank's kv subset
            pk = past_key_cache[i][:, :, kvs:kve, :]   # [1, P, 4, D]
            pv = past_value_cache[i][:, :, kvs:kve, :]
            # head-subset queries and keys (non-equal lengths, like TP suffix)
            q_h = q_full[:, :, hs:he, :]               # [1, S, 16, D]
            k_h = torch.cat([pk, k_suf[:, :, kvs:kve, :]], dim=1)  # [1,P+S,4,D]
            v_h = torch.cat([pv, v_suf[:, :, kvs:kve, :]], dim=1)
            flash_mask = _make_poskey_mask(attention_mask, position_ids,
                                           past_len)
            attn_out = _inner_pfa(
                q_h, k_h, v_h, atten_mask=flash_mask,
                num_heads=int(h_local), scale_value=scaling,
                input_layout="BSND", num_key_value_heads=int(kv_local),
                sparse_mode=0, inner_precise=1)          # [1, S, 16, D]
            # canonical reverse exchange: all heads of the full S rows, then
            # slice back to this rank's chunk
            out_full = _gather_head_dim(attn_out)     # [1, S, 32, D]
            out_chunk = _slice_local_seq(out_full)    # [1, S/CP, 32, D]
            hidden_states = residual + _o_proj_full(
                out_chunk.reshape(*h.shape[:-1], -1), layer.self_attn, i)
            residual = hidden_states
            h = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = residual + _run_mlp(layer, h, i)
        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        last_hidden = hidden_states[:, -1:, :]
        logits = torch.nn.functional.linear(last_hidden, self.lm_head.weight,
                                            self.lm_head.bias)
        return logits.float()


# ---------------------------------------------------------------------------
# Export driver
# ---------------------------------------------------------------------------
def _dummy_ids(seq_len):
    rng = np.random.default_rng(0)
    return torch.from_numpy(
        rng.integers(3, 2000, (1, seq_len), dtype=np.int64))


def _export_prefix(model, out_dir, rank):
    """Export one rank's prefix graph (K/V caches) and post-process it."""
    graph = Qwen3RerankerCPPrefix(model).eval()
    bucket = _prefix_bucket()
    ids = _dummy_ids(bucket)
    ids[:, PREFIX_LEN:] = 0
    mask = torch.ones(1, bucket, dtype=torch.long)
    mask[:, PREFIX_LEN:] = 0
    pos = torch.arange(bucket, dtype=torch.long).view(1, -1)
    rank_dir = os.path.join(out_dir, "rank%d" % rank, "prefix")
    os.makedirs(rank_dir, exist_ok=True)
    out_path = os.path.join(rank_dir,
                            "qwen3_reranker_8b_prefix_rank%d.onnx" % rank)
    dynamic = {"input_ids": {1: "seq_local"},
               "attention_mask": {1: "seq_prefix"},
               "position_ids": {1: "seq_local"},
               "k_out": {2: "seq_prefix"},
               "v_out": {2: "seq_prefix"}}
    if DEBUG_OUTPUT:
        dynamic.update({
            "l0_input_embeds": {1: "seq_prefix"},
            "l0_h_rmsnorm": {1: "seq_prefix"},
            "l0_q_rotary": {1: "seq_prefix"},
            "l0_k_rotary": {1: "seq_prefix"},
            "l0_v": {1: "seq_prefix"},
            "l0_attn_out": {1: "seq_prefix"},
            "l0_hidden_after_o_proj": {1: "seq_prefix"},
            "l0_hidden_after_mlp": {1: "seq_prefix"},
        })
    with torch.no_grad():
        kw = {"export_params": True, "opset_version": 18,
              "do_constant_folding": True,
              "input_names": ["input_ids", "attention_mask", "position_ids"],
              "output_names": (["k_out", "v_out",
                                "l0_input_embeds", "l0_h_rmsnorm",
                                "l0_q_rotary", "l0_k_rotary", "l0_v",
                                "l0_attn_out", "l0_hidden_after_o_proj",
                                "l0_hidden_after_mlp"] if DEBUG_OUTPUT
                               else ["k_out", "v_out"]),
              "dynamic_axes": dynamic}
        try:
            torch.onnx.export(graph, (ids, mask, pos), str(out_path),
                              dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(graph, (ids, mask, pos), str(out_path), **kw)
    print(f"[prefix] rank{rank} exported -> {out_path}", flush=True)
    _set_allow_nz_raw(out_path)
    _inline_postproc(out_path)
    return out_path


def _export_suffix(model, out_dir, rank):
    """Export one rank's suffix graph (prefix KV in, last-token logits out)."""
    # Host contract: input_ids / position_ids are the FULL (padded) suffix
    # span of length SUFFIX_LEN on every rank; the graph splits it internally.
    total = _prefix_bucket() + SUFFIX_LEN   # 128-aligned key span for the graph
    graph = Qwen3RerankerCPSuffix(model, model.lm_head).eval()
    ids = _dummy_ids(SUFFIX_LEN)
    mask = torch.ones(1, total, dtype=torch.long)
    pos = torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN,
                       dtype=torch.long).view(1, -1)
    kv_dtype = next(model.parameters()).dtype
    cache = torch.zeros(36, 1, _prefix_bucket(), 8,
                        int(model.config.head_dim or 128), dtype=kv_dtype)
    rank_dir = os.path.join(out_dir, "rank%d" % rank, "suffix")
    os.makedirs(rank_dir, exist_ok=True)
    out_path = os.path.join(rank_dir,
                            "qwen3_reranker_8b_suffix_rank%d.onnx" % rank)
    dynamic = {"input_ids": {1: "seq_local"},
               "attention_mask": {1: "seq_total"},
               "position_ids": {1: "seq_local"},
               "past_key_cache": {2: "seq_prefix"},
               "past_value_cache": {2: "seq_prefix"},
               "logits": {1: "last"}}
    with torch.no_grad():
        kw = {"export_params": True, "opset_version": 18,
              "do_constant_folding": True,
              "input_names": ["input_ids", "attention_mask", "position_ids",
                              "past_key_cache", "past_value_cache"],
              "output_names": ["logits"],
              "dynamic_axes": dynamic}
        try:
            torch.onnx.export(graph, (ids, mask, pos, cache, cache),
                              str(out_path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(graph, (ids, mask, pos, cache, cache),
                              str(out_path), **kw)
    print(f"[suffix] rank{rank} exported -> {out_path}", flush=True)
    _set_allow_nz_raw(out_path)
    _inline_postproc(out_path)
    return out_path



def main():
    global CP_SIZE, CP_RANK, PREFIX_LEN, SUFFIX_LEN, QUANT_MODE
    parser = argparse.ArgumentParser(
        description="Export Qwen3-Reranker-8B CP (Ulysses) ONNX")
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--quant-dir", type=str, default=None,
                        help="A8W8 weight pack dir (AMCT npy, or safetensors "
                             "w8a8_quant.bin + .index.json). Empty = fp16.")
    parser.add_argument("--quant-format", type=str, default="auto",
                        choices=["auto", "amct", "safetensors"],
                        help="Weight pack layout; auto = detect from contents.")
    parser.add_argument("--quant-mode", choices=["off", "dyn", "static"], default="off",
                        help="Quantization mode: off (default, plain fp16) / dyn "
                             "(per-token A8W8) / static (per-tensor A8W8). "
                             "dyn and static require --quant-dir.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512,
                        help="whole 图（--mode whole）的 dummy 全序列长度，须为 cp-size 倍数")
    parser.add_argument("--prefix-len", type=int, default=128,
                        help="公共前缀桶（128 对齐；--mode prefix/both）")
    parser.add_argument("--suffix-len", type=int, default=384,
                        help="后缀 dummy 长度（--mode suffix/both）")
    parser.add_argument("--mode", choices=["whole", "prefix", "suffix", "both"],
                        default="both",
                        help="whole=全图 CP（旧版单 pass）；"
                             "prefix/suffix/both=公共前缀两张图（默认）")
    args = parser.parse_args()

    CP_SIZE = args.cp_size
    QUANT_MODE = args.quant_mode
    if QUANT_MODE != "off" and not args.quant_dir:
        raise SystemExit(
            f"--quant-mode={QUANT_MODE} requires --quant-dir (A8W8 weight pack); "
            "use --quant-mode=off for a plain fp16 export.")

    dtypes = {"fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtypes[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch_dtype, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False,
                                              trust_remote_code=True)

    _load_quant_pack(args.quant_dir, args.quant_format)

    cfg = model.config
    if cfg.num_attention_heads % CP_SIZE != 0 or cfg.num_key_value_heads % CP_SIZE != 0:
        raise SystemExit(
            f"CP size {CP_SIZE} must divide both num_attention_heads "
            f"({cfg.num_attention_heads}) and num_key_value_heads "
            f"({cfg.num_key_value_heads}) for Ulysses head sharding.")
    if args.mode == "whole":
        if args.seq_len % CP_SIZE != 0:
            raise SystemExit(
                f"seq-len ({args.seq_len}) must be a multiple of cp-size ({CP_SIZE}); "
                "the host pads every request to a bucket that satisfies this.")
    else:
        if args.prefix_len % CP_SIZE or args.suffix_len % CP_SIZE:
            raise SystemExit("prefix/suffix len must be a multiple of cp-size")
    print(f"[cp] heads/rank = {cfg.num_attention_heads // CP_SIZE}, "
          f"kv_heads/rank = {cfg.num_key_value_heads // CP_SIZE}")

    if args.mode == "whole":
        for rank in range(CP_SIZE):
            CP_RANK = rank
            _export_rank(model, tokenizer, args.output_dir, args.seq_len)
    else:
        PREFIX_LEN = args.prefix_len
        SUFFIX_LEN = args.suffix_len
        for rank in range(CP_SIZE):
            CP_RANK = rank
            if args.mode in ("prefix", "both"):
                _export_prefix(model, args.output_dir, rank)
            if args.mode in ("suffix", "both"):
                _export_suffix(model, args.output_dir, rank)
    print("ALL_RANKS_EXPORTED")


if __name__ == "__main__":
    main()
