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
Export DeepSeek-OCR-2 (deepseek-ai/DeepSeek-OCR-2) to ONNX.

DeepSeek-OCR-2 is a DeepSeek-VL-V2 style multimodal OCR model: a SAM-ViT-B
vision encoder + a 24-layer Qwen2 decoder-as-encoder + MLP projector, feeding a
12-layer DeepSeek-V2 Mixture-of-Experts (64 routed experts, top-6, +2 shared)
language model. ``use_mla=False`` so attention is standard MHA with rotate-half
RoPE.

The model is exported into three ONNX sub-graphs:

  1. Vision encoder : (global image, local crops) -> image embeddings
  2. LLM prefill    : full prompt -> logits + KV cache
  3. LLM decode      : single token + KV cache -> logits + updated KV cache

Compatibility notes
-------------------
The checkpoint ships ``trust_remote_code`` modeling pinned to transformers
4.46. This script runs under transformers 5.x by shimming two removed names
(``LlamaFlashAttention2``, ``is_torch_fx_available``) and setting the
DeepSeek-V2 config defaults the 5.x config loader skips.
"""

import argparse
import gc
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass


# ---------------------------------------------------------------------------
# transformers 5.x compatibility shims (so the 4.46-pinned custom code imports).
# ---------------------------------------------------------------------------


def _apply_transformers_shims():
    """Shim removed names so DeepSeek-V2 custom code imports under transformers 5.x."""
    import transformers.models.llama.modeling_llama as llama
    if not hasattr(llama, "LlamaFlashAttention2"):
        class LlamaFlashAttention2(llama.LlamaAttention):
            """Compatibility shim: DeepSeek-V2 selects this for the mha_flash path."""

        llama.LlamaFlashAttention2 = LlamaFlashAttention2
    import transformers.utils.import_utils as iu
    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda *a, **k: False
        import transformers.utils as u
        u.is_torch_fx_available = iu.is_torch_fx_available


_DSV2_DEFAULTS = dict(
    attention_dropout=0.0, rms_norm_eps=1e-6, rope_theta=10000.0, norm_topk_prob=False,
    routed_scaling_factor=1.0, aux_loss_alpha=0.001, seq_aux=True, hidden_act="silu",
    initializer_range=0.02, pretraining_tp=1, use_cache=True, attention_bias=False,
    use_mla=False, scoring_func="softmax", moe_layer_freq=1, ep_size=1, topk_method="greedy",
    tie_word_embeddings=False, rope_scaling=None)


def _patch_config(cfg):
    """Set DeepSeek-V2 config defaults the transformers 5.x loader skips."""
    for key, val in _DSV2_DEFAULTS.items():
        if not hasattr(cfg, key) or getattr(cfg, key) is None:
            setattr(cfg, key, val)
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = cfg.eos_token_id
    cfg._attn_implementation = "eager"
    return cfg


def _load_model(model_dir, dtype=torch.float16):
    """Load DeepSeek-OCR-2 (real weights) with transformers 5.x shims + config defaults."""
    _apply_transformers_shims()
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    _patch_config(cfg)
    parent = str(Path(model_dir).parent)
    pkg_name = Path(model_dir).name
    if parent not in sys.path:
        sys.path.insert(0, parent)
    pkg = importlib.import_module(f"{pkg_name}.modeling_deepseekocr2")
    # Load real weights via from_pretrained (auto_map -> DeepseekOCR2ForCausalLM) at the
    # requested dtype. torchvision (used only by model.infer()) is shimmed above.
    model = pkg.DeepseekOCR2ForCausalLM.from_pretrained(
        model_dir, config=cfg, trust_remote_code=True, torch_dtype=dtype,
        low_cpu_mem_usage=True)
    model.eval()
    return model, pkg


def _move_modules(model, device, dtype):
    """Move the sub-modules used by the export wrappers to the target device/dtype."""
    for mod in (model.model.embed_tokens, model.model.norm, model.lm_head):
        mod.to(device=device, dtype=dtype)
    for layer in model.model.layers:
        layer.to(device=device, dtype=dtype)
        if hasattr(layer.self_attn, "rotary_emb"):
            layer.self_attn.rotary_emb.to(device=device)


# ---------------------------------------------------------------------------
# Custom CANN operators.
# ---------------------------------------------------------------------------


def _as_list_str(items):
    """Convert items to a list of string representations for ONNX attrs."""
    return [str(x) for x in items]


class _CustomRmsNorm(torch.autograd.Function):
    """Custom RMSNorm operator for ONNX export."""

    @staticmethod
    def forward(ctx, x, gamma, epsilon_f):
        """Compute RMSNorm (fp32 accumulator) returning y and rstd."""
        del ctx
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon_f))
        y = (x_fp32 * rstd).to(dtype=x.dtype) * gamma.to(dtype=x.dtype)
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon_f):
        """Export a Custom node for RMSNorm."""
        y, rstd = g.op("Custom", x, gamma, outputs=2, type_s="RmsNorm",
                       epsilon_f=float(epsilon_f),
                       input_names_s=_as_list_str(["x", "gamma"]),
                       optional_input_names_s=_as_list_str([]),
                       output_names_s=_as_list_str(["y", "rstd"]),
                       output_num_i=2, input_index_i=[0, 1])
        y.setType(x.type())
        return y, rstd


def _rotate_half(x):
    """Standard rotate-half (first/second half split) for RoPE."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class _CustomRotaryMul(torch.autograd.Function):
    """Custom rotary multiplication (rotate-half) for ONNX export."""

    @staticmethod
    def forward(ctx, x, cos, sin):
        """Apply rotate-half rotary embedding."""
        del ctx
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        return x * cos + _rotate_half(x) * sin

    @staticmethod
    def symbolic(g, x, cos, sin):
        """Export a Custom node for RotaryMul."""
        y = g.op("Custom", x, cos, sin, type_s="RotaryMul",
                 input_names_s=_as_list_str(["x", "r1", "r2"]),
                 output_names_s=_as_list_str(["y"]),
                 output_num_i=1, input_index_i=[0, 1, 2])
        y.setType(x.type())
        return y


class _CustomSwiGlu(torch.autograd.Function):
    """Custom SwiGLU operator for ONNX export (silu(a) * b)."""

    @staticmethod
    def forward(ctx, x, dim_i):
        """Split along dim_i and return silu(first) * second."""
        del ctx
        a, b = torch.chunk(x, 2, dim=int(dim_i))
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim_i):
        """Export a Custom node for SwiGlu."""
        y = g.op("Custom", x, type_s="SwiGlu", dim_i=int(dim_i),
                 input_names_s=_as_list_str(["x"]), output_names_s=_as_list_str(["y"]))
        y.setType(x.type())
        return y




def _attn_dims(attn):
    """Derive (num_heads, num_kv_heads, head_dim) from projection weights (5.x LlamaAttention lacks num_heads attr)."""
    head_dim = getattr(attn, "head_dim", None) or 128
    num_heads = attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
    return num_heads, num_kv_heads, head_dim

def _expand_kv(key, value, num_heads, num_kv_heads):
    """Expand GQA key/value to num_heads via repeat_interleave on dim=1."""
    if num_kv_heads == num_heads:
        return key, value
    rep = num_heads // num_kv_heads
    return key.repeat_interleave(rep, dim=1), value.repeat_interleave(rep, dim=1)


class _CustomPromptFlashAttention(torch.autograd.Function):
    """Custom prompt flash attention (prefill) for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Fallback full attention for tracing."""
        del ctx, input_layout_s
        key, value = _expand_kv(key, value, int(num_heads_i), int(num_kv_heads_i))
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i, num_kv_heads_i,
                 scale_value_f, input_layout_s):
        """Export a Custom node for prompt flash attention."""
        y = g.op("Custom", query, key, value, atten_mask, type_s="PromptFlashAttention",
                 num_heads_i=int(num_heads_i), num_key_value_heads_i=int(num_kv_heads_i),
                 scale_value_f=float(scale_value_f), input_layout_s=str(input_layout_s),
                 pre_tokens_i=2147483647, next_tokens_i=0, sparse_mode_i=0, inner_precise_i=1,
                 input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                 optional_input_names_s=_as_list_str(["atten_mask"]),
                 output_names_s=_as_list_str(["attention_out"]))
        y.setType(query.type())
        return y


class _CustomIncreFlashAttention(torch.autograd.Function):
    """Custom incremental flash attention (decode) for ONNX export."""

    @staticmethod
    def forward(ctx, query, key, value, atten_mask, num_heads_i, num_kv_heads_i,
                scale_value_f, input_layout_s):
        """Fallback single-step attention for tracing."""
        del ctx, input_layout_s
        key, value = _expand_kv(key, value, int(num_heads_i), int(num_kv_heads_i))
        scores = torch.matmul(query, key.transpose(-2, -1)) * float(scale_value_f)
        if atten_mask is not None:
            scores = scores.masked_fill(atten_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probs, value)

    @staticmethod
    def symbolic(g, query, key, value, atten_mask, num_heads_i, num_kv_heads_i,
                 scale_value_f, input_layout_s):
        """Export a Custom node for incremental flash attention."""
        y = g.op("Custom", query, key, value, atten_mask, type_s="IncreFlashAttention",
                 num_heads_i=int(num_heads_i), num_key_value_heads_i=int(num_kv_heads_i),
                 scale_value_f=float(scale_value_f), input_layout_s=str(input_layout_s),
                 block_size_i=0, inner_precise_i=1,
                 input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                 optional_input_names_s=_as_list_str(["atten_mask"]),
                 output_names_s=_as_list_str(["attention_out"]))
        y.setType(query.type())
        return y


class _CustomScatter(torch.autograd.Function):
    """Custom scatter-update for fixed-length KV cache update."""

    @staticmethod
    def forward(ctx, var, indices, updates, axis_i):
        """Scatter updates into var at indices along axis_i."""
        del ctx
        bsz = var.shape[0]
        idx = indices.view(bsz, 1, 1, 1).to(dtype=torch.int64)
        idx = idx.expand(bsz, var.shape[1], 1, var.shape[3])
        return var.scatter(int(axis_i), idx, updates.to(dtype=var.dtype))

    @staticmethod
    def symbolic(g, var, indices, updates, axis_i):
        """Export a Custom node for scatter cache update."""
        y = g.op("Custom", var, indices, updates, type_s="Scatter", reduce_s="update",
                 axis_i=int(axis_i), input_names_s=_as_list_str(["var", "indices", "updates"]),
                 output_names_s=_as_list_str(["var"]))
        y.setType(var.type())
        return y


# ---------------------------------------------------------------------------
# Shared forward helpers (RMSNorm, MLP, MoE flatten, attention, rotary).
# ---------------------------------------------------------------------------


def _rms_norm(norm_mod, x):
    """Apply RMSNorm via the Custom op."""
    eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6))
    y, _ = _CustomRmsNorm.apply(x, norm_mod.weight, float(eps))
    return y


def _dense_mlp(mlp, x):
    """Run a dense DeepSeek-V2 MLP (gate_proj/up_proj/down_proj + SwiGlu)."""
    gate = mlp.gate_proj(x)
    up = mlp.up_proj(x)
    return mlp.down_proj(_CustomSwiGlu.apply(torch.cat([gate, up], dim=-1), -1))


def _stack_expert_weight(experts, attr):
    """Stack an expert projection weight across all experts into (E, out, in)."""
    return torch.stack([getattr(e, attr).weight for e in experts], dim=0)


def _moe_forward(moe, hidden):
    """Run a flattened (traceable) MoE: compute all experts, gather top-k, add shared."""
    num_experts = len(moe.experts)
    top_k = int(moe.gate.top_k)
    routed_scaling = float(getattr(moe.gate, "routed_scaling_factor", 1.0))
    norm_topk = bool(getattr(moe.gate, "norm_topk_prob", False))

    gate_w = moe.gate.weight  # (num_experts, hidden)
    scores = F.linear(hidden, gate_w).softmax(dim=-1, dtype=torch.float32).to(hidden.dtype)
    topk_weight, topk_idx = torch.topk(scores, top_k, dim=-1)  # (B,S,top_k)
    if norm_topk:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weight = topk_weight * routed_scaling

    # Compute every expert via explicit matmul (NOT einsum — converter can't handle it).
    g_w = _stack_expert_weight(moe.experts, "gate_proj")  # (E, inter, H)
    u_w = _stack_expert_weight(moe.experts, "up_proj")    # (E, inter, H)
    d_w = _stack_expert_weight(moe.experts, "down_proj").transpose(-1, -2)  # (E, inter, H)
    E, inter_dim, H = g_w.shape
    # gate/up: (B,S,H) @ w.T -> (B,S,E*inter) -> permute to (E, B*S, inter)
    g_all = F.linear(hidden, g_w.reshape(E * inter_dim, H))
    u_all = F.linear(hidden, u_w.reshape(E * inter_dim, H))
    bs = g_all.shape[0] * g_all.shape[1]
    g = g_all.view(bs, E, inter_dim).permute(1, 0, 2)       # (E, B*S, inter)
    u_out = u_all.view(bs, E, inter_dim).permute(1, 0, 2)
    act = F.silu(g) * u_out                                  # (E, B*S, inter)
    # down: bmm (E,B*S,inter) @ (E,inter,H) -> (E, B*S, H)
    per_expert = torch.bmm(act, d_w)
    bsz, seq_len = hidden.shape[0], hidden.shape[1]
    per_expert = per_expert.view(E, bsz, seq_len, H)        # (E, B, S, H)

    # Gather selected experts and weight.
    idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, hidden.shape[-1])  # (B,S,top_k,H)
    gathered = torch.gather(per_expert.permute(1, 2, 0, 3), 2, idx)     # (B,S,top_k,H)
    routed = (gathered * topk_weight.unsqueeze(-1)).sum(dim=2)          # (B,S,H)
    if hasattr(moe, "shared_experts") and moe.shared_experts is not None:
        routed = routed + _dense_mlp(moe.shared_experts, hidden)
    return routed


def _layer_forward(layer, hidden, cos, sin, bool_mask, is_moe):
    """Run one DeepSeek-V2 decoder layer (dense or MoE)."""
    residual = hidden
    hidden = _rms_norm(layer.input_layernorm, hidden)
    attn_out, pk, pv = _attn_forward(layer.self_attn, hidden, cos, sin, bool_mask)
    hidden = residual + attn_out
    residual = hidden
    hidden = _rms_norm(layer.post_attention_layernorm, hidden)
    mlp = _moe_forward(layer.mlp, hidden) if is_moe else _dense_mlp(layer.mlp, hidden)
    hidden = residual + mlp
    return hidden, pk, pv


def _attn_forward(attn, hidden, cos, sin, bool_mask):
    """Prefill attention (standard MHA + rotate-half RoPE + PromptFlashAttention)."""
    bsz, seq_len, _ = hidden.shape
    num_heads, num_kv_heads, _ = _attn_dims(attn)
    head_dim = attn.head_dim
    q = attn.q_proj(hidden).view(bsz, -1, num_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)
    v = attn.v_proj(hidden).view(bsz, -1, num_kv_heads, head_dim).transpose(1, 2)
    q = _CustomRotaryMul.apply(q, cos, sin)
    k = _CustomRotaryMul.apply(k, cos, sin)
    out = _CustomPromptFlashAttention.apply(
        q, k, v, bool_mask, int(num_heads), int(num_kv_heads),
        float(attn.scaling), "BNSD")
    out = out.transpose(1, 2).reshape(bsz, -1, num_heads * head_dim).contiguous()
    return attn.o_proj(out), k, v


def _make_prefill_mask(attention_mask, q_len, kv_len):
    """Boolean causal + padding mask (True == masked out) for prefill."""
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(kv_len, device=attention_mask.device)
    causal = ar_k[None, :] > ar_q[:, None]
    pad = attention_mask.to(torch.bool).logical_not()
    return causal[None, None, :, :] | pad[:, None, None, :]


def _make_decode_mask(attention_mask, cache_pos, max_seq_len):
    """Boolean mask (True == masked out) for single-step decode."""
    bsz = attention_mask.shape[0]
    pos = cache_pos.view(bsz, 1).to(dtype=torch.int64)
    ar_k = torch.arange(max_seq_len, device=attention_mask.device).view(1, -1)
    causal = ar_k > pos
    pad = attention_mask.to(torch.bool).logical_not()
    return causal[:, None, None, :] | pad[:, None, None, :]


# ---------------------------------------------------------------------------
# Vision encoder wrapper (SAM ViT-B + Qwen2 decoder + MLP projector).
# The Qwen2 decoder's custom 4D mask is precomputed for the fixed token layout.
# ---------------------------------------------------------------------------


def _vision_decoder_mask(n_tokens, device, dtype):
    """Build the fixed (1,1,2n,2n) mask for the Qwen2 vision decoder.

    First ``n`` tokens are image (non-causal), last ``n`` are queries (causal).
    Allowed: img-img, qry-img, qry-causal(qry). All else masked.
    """
    n = n_tokens
    full = 2 * n
    min_val = torch.finfo(dtype).min
    img = torch.arange(n)
    qry = torch.arange(n, full)
    mask = torch.full((full, full), min_val, dtype=dtype, device=device)
    mask[img[:, None], img] = 0.0          # image tokens attend to all image tokens (non-causal)
    mask[qry[:, None], img] = 0.0          # query tokens attend to all image tokens
    for i in range(n):                      # query tokens attend causally to earlier queries
        mask[qry[i], qry[: i + 1]] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


class DeepseekVisionWrapper(torch.nn.Module):
    """Wrap the DeepSeek-OCR-2 vision tower (SAM + Qwen2 decoder + projector).

    Exports a fixed configuration: one global 1024x1024 view + ``n_crops`` local
    768x768 crops. Outputs image embeddings (n_crops*144 + 256 + 1, 1280).
    """

    def __init__(self, model, n_crops):
        """Cache vision modules, fixed crop count and the view separator."""
        super().__init__()
        self.sam = model.model.sam_model
        self.qwen2 = model.model.qwen2_model
        self.projector = model.model.projector
        self.n_crops = int(n_crops)
        self.register_buffer("view_seperator", model.model.view_seperator.detach(), persistent=False)

    def _run_pipeline(self, images):
        """Run SAM -> Qwen2 decoder (precomputed mask) -> projector for a batch."""
        sam_out = self.sam(images)                       # (B,896,h,w)
        flat = sam_out.flatten(2).transpose(1, 2)        # (B, n_query, 896)
        bs, n_query, _ = flat.shape
        if n_query == 144:
            queries = self.qwen2.query_768.weight.unsqueeze(0).expand(bs, -1, -1)
        else:
            queries = self.qwen2.query_1024.weight.unsqueeze(0).expand(bs, -1, -1)
        combined = torch.cat([flat, queries], dim=1)     # (B, 2*n_query, 896)
        mask = _vision_decoder_mask(n_query, flat.device, flat.dtype).expand(bs, 1, -1, -1)
        # Run the underlying Qwen2 layers directly with the precomputed mask.
        hidden = combined
        for layer in self.qwen2.model.model.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            attn = layer.self_attn
            num_heads, num_kv_heads, head_dim = _attn_dims(attn)
            q = attn.q_proj(hidden).view(bs, -1, num_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hidden).view(bs, -1, num_kv_heads, head_dim).transpose(1, 2)
            v = attn.v_proj(hidden).view(bs, -1, num_kv_heads, head_dim).transpose(1, 2)
            cos, sin = layer.self_attn.rotary_emb(hidden, position_ids=None)
            q = _CustomRotaryMul.apply(q, cos, sin)
            k = _CustomRotaryMul.apply(k, cos, sin)
            out = _CustomPromptFlashAttention.apply(
                q, k, v, mask < 0, int(num_heads), int(num_kv_heads),
                float(attn.scaling), "BNSD")
            out = attn.o_proj(out.transpose(1, 2).reshape(bs, -1, attn.hidden_size))
            hidden = residual + out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.qwen2.model.model.norm(hidden)
        return self.projector(hidden[:, n_query:, :])    # (B, n_query, 1280)

    def forward(self, global_image, crops):
        """Run global + local crop pipelines and concatenate image embeddings."""
        global_feat = self._run_pipeline(global_image).reshape(-1, 1280)   # (256,1280)
        crop_feat = self._run_pipeline(crops).reshape(-1, 1280)            # (n_crops*144,1280)
        sep = self.view_seperator.unsqueeze(0)                             # (1,1280)
        return torch.cat([crop_feat, global_feat, sep], dim=0)             # (n_crops*144+257,1280)


# ---------------------------------------------------------------------------
# LLM prefill / decode wrappers.
# ---------------------------------------------------------------------------


class DeepseekLlmPrefill(torch.nn.Module):
    """DeepSeek-OCR-2 LLM prefill wrapper: prompt -> logits + KV cache."""

    def __init__(self, model, image_token_id, first_k_dense_replace):
        """Initialize prefill wrapper with embed, layers, norm, lm_head."""
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.image_token_id = int(image_token_id)
        self.first_k_dense_replace = int(first_k_dense_replace)
        head_dim = self.layers[0].self_attn.head_dim
        self.register_buffer(
            "inv_freq",
            (1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))),
            persistent=False)

    def _rope(self, position_ids, dtype):
        """Compute rotate-half cos/sin for the given positions."""
        seq_len = position_ids.shape[1]
        # Use matmul instead of torch.outer (which exports as Einsum i,j->ij)
        freqs = position_ids[0].float().unsqueeze(-1) * self.inv_freq  # (seq, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                      # (seq, head_dim)
        cos = emb.cos()[None, None, :, :].to(dtype)
        sin = emb.sin()[None, None, :, :].to(dtype)
        return cos, sin

    def forward(self, input_ids, attention_mask, position_ids, image_embeds):
        """Run prefill: embed -> scatter image embeds -> layers -> logits + kv."""
        inputs_embeds = self.embed_tokens(input_ids)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask, image_embeds.to(dtype=inputs_embeds.dtype))
        cos, sin = self._rope(position_ids, inputs_embeds.dtype)
        bool_mask = _make_prefill_mask(attention_mask, input_ids.shape[1], input_ids.shape[1])
        hidden = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            hidden, pk, pv = _layer_forward(layer, hidden, cos, sin, bool_mask,
                                            i >= self.first_k_dense_replace)
            present.append(pk)
            present.append(pv)
        hidden = _rms_norm(self.norm, hidden)
        logits = self.lm_head(hidden)
        return logits, torch.stack(present, dim=0)


class DeepseekLlmDecode(torch.nn.Module):
    """DeepSeek-OCR-2 LLM decode wrapper: single token + KV cache -> logits + kv."""

    def __init__(self, model, max_seq_len, first_k_dense_replace):
        """Initialize decode wrapper with fixed KV cache length."""
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.max_seq_len = int(max_seq_len)
        self.first_k_dense_replace = int(first_k_dense_replace)
        head_dim = self.layers[0].self_attn.head_dim
        self.register_buffer(
            "inv_freq",
            (1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))),
            persistent=False)

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, cache_pos):
        """Run decode: embed -> layers (scatter cache) -> logits + kv."""
        inputs_embeds = self.embed_tokens(input_ids)
        bsz = input_ids.shape[0]
        cache_pos = cache_pos.view(bsz).to(dtype=torch.int64)
        # Use elementwise mul instead of outer (avoids Einsum)
        pos_val = position_ids.reshape(-1).to(torch.float32)  # (1,)
        freqs = pos_val.unsqueeze(-1) * self.inv_freq  # (1, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :].to(inputs_embeds.dtype)
        sin = emb.sin()[None, None, :, :].to(inputs_embeds.dtype)
        bool_mask = _make_decode_mask(attention_mask, cache_pos, self.max_seq_len)
        hidden = inputs_embeds
        present = []
        for i, layer in enumerate(self.layers):
            pk_cache = past_key_values[2 * i]
            pv_cache = past_key_values[2 * i + 1]
            hidden, pk_cache, pv_cache = self._decode_layer(
                layer, hidden, cos, sin, bool_mask, cache_pos, pk_cache, pv_cache,
                i >= self.first_k_dense_replace)
            present.append(pk_cache)
            present.append(pv_cache)
        hidden = _rms_norm(self.norm, hidden)
        return self.lm_head(hidden), torch.stack(present, dim=0)

    def _decode_layer(self, layer, hidden, cos, sin, bool_mask, cache_pos,
                      key_cache, value_cache, is_moe):
        """Run one decode layer with fixed KV cache + scatter update."""
        residual = hidden
        hidden = _rms_norm(layer.input_layernorm, hidden)
        attn = layer.self_attn
        bsz = hidden.shape[0]
        num_heads, num_kv_heads, head_dim = _attn_dims(attn)
        q = attn.q_proj(hidden).view(bsz, 1, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(hidden).view(bsz, 1, num_kv_heads, head_dim).transpose(1, 2)
        q = _CustomRotaryMul.apply(q, cos, sin)
        k = _CustomRotaryMul.apply(k, cos, sin)
        key_cache = _CustomScatter.apply(key_cache, cache_pos, k, 2)
        value_cache = _CustomScatter.apply(value_cache, cache_pos, v, 2)
        out = _CustomIncreFlashAttention.apply(
            q, key_cache, value_cache, bool_mask, int(num_heads), int(num_kv_heads),
            float(attn.scaling), "BNSD")
        out = attn.o_proj(out.transpose(1, 2).reshape(bsz, 1, -1))
        hidden = residual + out
        residual = hidden
        hidden = _rms_norm(layer.post_attention_layernorm, hidden)
        mlp = _moe_forward(layer.mlp, hidden) if is_moe else _dense_mlp(layer.mlp, hidden)
        hidden = residual + mlp
        return hidden, key_cache, value_cache


# ---------------------------------------------------------------------------
# Export routines.
# ---------------------------------------------------------------------------


def export_vision(model, output_dir, device, n_crops, dtype):
    """Export the vision tower (global + fixed crops) to ONNX."""
    wrapper = DeepseekVisionWrapper(model, n_crops).to(device).eval()
    dummy_global = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
    dummy_crops = torch.randn(n_crops, 3, 768, 768, device=device, dtype=dtype)
    vision_path = Path(output_dir) / "deepseek_ocr_2_vision.onnx"
    print(f"Exporting Vision tower to {vision_path} (n_crops={n_crops})...")
    from torch.onnx import utils as onnx_utils
    with torch.no_grad():
        onnx_utils.export(wrapper, (dummy_global, dummy_crops), str(vision_path),
                          input_names=["global_image", "crops"],
                          output_names=["image_embeds"],
                          opset_version=14, do_constant_folding=True)
    print("Vision tower exported successfully.")


def export_llm(model, output_dir, device, kv_cache_len, dummy_seq, dtype):
    """Export LLM prefill + decode sub-graphs to ONNX."""
    cfg = model.config
    image_token_id = 128815
    first_k_dense = int(getattr(cfg, "first_k_dense_replace", 1))
    num_layers = len(model.model.layers)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    _move_modules(model, device, dtype)
    prefill = DeepseekLlmPrefill(model, image_token_id, first_k_dense).to(device).eval()
    decode = DeepseekLlmDecode(model, kv_cache_len, first_k_dense).to(device).eval()

    dummy_ids = torch.randint(0, 1000, (1, dummy_seq), dtype=torch.int64, device=device)
    dummy_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_pos = torch.arange(dummy_seq, dtype=torch.int64, device=device).unsqueeze(0)
    dummy_img = torch.randn(16, cfg.hidden_size, device=device, dtype=dtype)
    prefill_path = Path(output_dir) / "deepseek_ocr_2_llm_prefill.onnx"
    print(f"Exporting LLM prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(prefill, (dummy_ids, dummy_mask, dummy_pos, dummy_img), str(prefill_path),
                          input_names=["input_ids", "attention_mask", "position_ids", "image_embeds"],
                          output_names=["logits", "present_key_values"],
                          opset_version=18, do_constant_folding=True, dynamo=False)
    print("LLM prefill exported successfully.")

    step_ids = torch.randint(0, 1000, (1, 1), dtype=torch.int64, device=device)
    step_mask = torch.ones(1, kv_cache_len, dtype=torch.int64, device=device)
    step_pos = torch.tensor([[8]], dtype=torch.int64, device=device)
    dummy_past = torch.zeros(2 * num_layers, 1, cfg.num_key_value_heads, kv_cache_len, head_dim,
                             dtype=dtype, device=device)
    cache_pos = torch.tensor([8], dtype=torch.int64, device=device)
    decode_path = Path(output_dir) / "deepseek_ocr_2_llm_decode.onnx"
    print(f"Exporting LLM decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(decode, (step_ids, step_mask, step_pos, dummy_past, cache_pos),
                          str(decode_path),
                          input_names=["input_ids", "attention_mask", "position_ids",
                                       "past_key_values", "cache_pos"],
                          output_names=["logits", "present_key_values"],
                          opset_version=18, do_constant_folding=True, dynamo=False)
    print("LLM decode exported successfully.")


def _clear_cache():
    """Release torch caches and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export DeepSeek-OCR-2 to ONNX")
    parser.add_argument("--model-id", type=str, required=True, help="Local model directory")
    parser.add_argument("--output-dir", type=str, default="./deepseek_ocr_2_onnx")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n-crops", type=int, default=2,
                        help="Fixed number of local crops (controls image token count).")
    parser.add_argument("--kv-cache-len", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision export (LLM-only).")
    return parser.parse_args()


def main():
    """Load model and export vision + LLM prefill/decode ONNX sub-graphs."""
    args = _parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model_id} ({args.dtype})...")
    model, _ = _load_model(args.model_id, dtype)
    model.eval()

    if not args.skip_vision:
        export_vision(model, output_dir, args.device, int(args.n_crops), dtype)
        _clear_cache()
        del model.model.sam_model
        del model.model.qwen2_model
        del model.model.projector
        _clear_cache()
    export_llm(model, output_dir, args.device, int(args.kv_cache_len), dummy_seq=8, dtype=dtype)
    print(f"\nExport finished. Files saved in {args.output_dir}")


if __name__ == "__main__":
    main()
