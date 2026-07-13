"""Export Qwen3-ASR 1.7B audio encoder + text prefill/decode to ONNX for MindSpore Lite.

Optimized export: separate prefill/decode graphs with KV cache and Ascend custom op
fusion (RotaryMul / RmsNorm / PromptFlashAttention / IncreFlashAttention / SwiGlu /
Scatter). Converter config enables plugin_custom_ops=All unconditionally.
"""

import argparse
import gc
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

KV_CACHE_LEN = 1024
EXPORT_DTYPE = "fp32"

try:
    import torch._dynamo

    torch._dynamo.disable()
except Exception:
    pass

try:
    from torch.fx import wrap as _fx_wrap

    _fx_wrap("rotary_mul")
    _fx_wrap("apply_rotary_pos_emb_custom")
    _fx_wrap("rms_norm")
    _fx_wrap("incre_flash_attention")
    _fx_wrap("prompt_flash_attention")
    _fx_wrap("swiglu")
    _fx_wrap("scatter")
except Exception:
    pass

# Compatibility shim: qwen_asr 0.0.6 uses `@check_model_inputs()` (factory form),
# but transformers>=4.56 made it `check_model_inputs(func)` (direct decorator).
# Patch before importing qwen_asr so its module-level class defs succeed.
try:
    import transformers.utils.generic as _tug
    if not getattr(_tug.check_model_inputs, "_compat_patched", False):
        _orig_check_model_inputs = _tug.check_model_inputs

        def _compat_check_model_inputs(func=None):
            if func is not None:
                return _orig_check_model_inputs(func)

            def _deco(f):
                return _orig_check_model_inputs(f)

            return _deco

        _compat_check_model_inputs._compat_patched = True
        _tug.check_model_inputs = _compat_check_model_inputs
except Exception:
    pass

# qwen_asr imports `nagisa` (Japanese tokenizer) only for the forced-aligner path,
# which is unused here. Stub it so the package imports without the extra dep.
try:
    import nagisa  # noqa: F401  # pylint: disable=unused-import
except Exception:
    if "nagisa" not in sys.modules:
        sys.modules["nagisa"] = types.ModuleType("nagisa")

try:
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig,
        Qwen3ASRForConditionalGeneration,
    )
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
        Qwen3ASRThinkerConfig,
    )
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer
except ImportError as e:
    print(f"Error: required package not found: {e}", file=sys.stderr)
    sys.exit(1)


# Compat: transformers >=5.7 runs token-id validation inside PretrainedConfig
# __init__, which calls get_text_config() -> self.thinker_config.get_text_config().
# qwen_asr 0.0.6 sets self.thinker_config AFTER super().__init__(), so validation
# hits AttributeError. Reorder to construct thinker_config first.
def _qwen3asr_config_init(self, thinker_config=None, support_languages=None, **kwargs):
    if thinker_config is None:
        thinker_config = {}
    self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
    self.support_languages = support_languages
    super(Qwen3ASRConfig, self).__init__(**kwargs)


Qwen3ASRConfig.__init__ = _qwen3asr_config_init

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)


# ---------------------------------------------------------------------------
# Audio encoder spec helpers (unchanged from original export)
# ---------------------------------------------------------------------------

def _audio_token_len_from_feat_frames(n_frames: int) -> int:
    input_lengths_leave = n_frames % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (n_frames // 100) * 13
    return int(output_lengths)


@dataclass(frozen=True)
class _AudioOeSpec:
    n_mels: int
    n_frames: int
    chunk_size: int
    chunks: int
    aftercnn_per_chunk: int
    aftercnn_total: int
    window_aftercnn: int
    cu_seqlens: Tuple[int, ...]


def _build_audio_spec(
    n_mels: int = 128,
    n_frames: int = 3000,
    n_window: int = 100,
    n_window_infer: int = 400,
):
    """Compute the audio encoder chunking/cu_seqlens spec for ONNX export."""
    chunk_size = int(n_window * 2)
    if n_frames % chunk_size != 0:
        raise ValueError(
            "n_frames must be divisible by chunk_size, got "
            f"n_frames={n_frames}, chunk_size={chunk_size}"
        )
    chunks = int(n_frames // chunk_size)
    aftercnn_per_chunk = _audio_token_len_from_feat_frames(chunk_size)
    aftercnn_total = int(aftercnn_per_chunk * chunks)
    ratio = int(n_window_infer // (n_window * 2))
    window_aftercnn = int(aftercnn_per_chunk * ratio)
    if aftercnn_total % window_aftercnn == 0:
        parts = [window_aftercnn] * (aftercnn_total // window_aftercnn)
    else:
        parts = [window_aftercnn] * (aftercnn_total // window_aftercnn) + [
            aftercnn_total % window_aftercnn
        ]
    cu = [0]
    for p in parts:
        cu.append(cu[-1] + p)
    return _AudioOeSpec(
        n_mels=n_mels,
        n_frames=n_frames,
        chunk_size=chunk_size,
        chunks=chunks,
        aftercnn_per_chunk=aftercnn_per_chunk,
        aftercnn_total=aftercnn_total,
        window_aftercnn=window_aftercnn,
        cu_seqlens=tuple(int(x) for x in cu),
    )


class Qwen3AsrAudioEncoderOnnx(torch.nn.Module):
    """Wraps the Qwen3-ASR audio tower for ONNX export (mel features -> audio embeddings)."""

    def __init__(self, audio_tower: torch.nn.Module, spec: _AudioOeSpec):
        super().__init__()
        self.audio_tower = audio_tower
        self.spec = spec

        self.register_buffer(
            "_cu_seqlens",
            torch.tensor(spec.cu_seqlens, dtype=torch.int32),
            persistent=False,
        )
        attn = torch.full(
            (spec.aftercnn_total, spec.aftercnn_total),
            fill_value=torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        for i in range(len(spec.cu_seqlens) - 1):
            s = spec.cu_seqlens[i]
            e = spec.cu_seqlens[i + 1]
            attn[s:e, s:e] = 0.0
        self.register_buffer(
            "_attn_mask_4d",
            attn[None, None, :, :],
            persistent=False,
        )

        if hasattr(self.audio_tower, "config") and hasattr(
            self.audio_tower.config, "_attn_implementation"
        ):
            self.audio_tower.config._attn_implementation = "eager"

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run mel features through conv front-end + transformer tower -> audio embeddings."""
        x = input_features[:, :, : self.spec.n_frames]
        x = x.reshape(1, self.spec.n_mels, self.spec.chunks, self.spec.chunk_size)
        x = x.permute(0, 2, 1, 3).reshape(
            self.spec.chunks,
            self.spec.n_mels,
            self.spec.chunk_size,
        )

        x = x.unsqueeze(1)
        x = torch.nn.functional.gelu(self.audio_tower.conv2d1(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d2(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d3(x))
        b, c, f, t = x.size()
        x = (
            x.permute(0, 3, 1, 2)
            .contiguous()
            .view(b, t, c * f)
        )
        x = self.audio_tower.conv_out(x)

        pos = self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :]
        pos = pos.unsqueeze(0).to(x.dtype)
        x = x + pos
        x = x.reshape(-1, x.shape[-1]).contiguous()

        cu_seqlens = self._cu_seqlens.to(x.device)
        attn_mask_4d = self._attn_mask_4d.to(x.device, x.dtype)
        for layer in self.audio_tower.layers:
            x = layer(x, cu_seqlens=cu_seqlens, attention_mask=attn_mask_4d)[0]

        x = self.audio_tower.ln_post(x)
        x = self.audio_tower.proj1(x)
        x = self.audio_tower.act(x)
        x = self.audio_tower.proj2(x)
        return x.unsqueeze(0)


# ---------------------------------------------------------------------------
# Ascend custom op classes (ported from qwen3_1.7b, unchanged semantics)
# ---------------------------------------------------------------------------

def _as_list_str(items):
    return [str(x) for x in items]


def _rotate_half(x):
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class _RotaryMulCustom(torch.autograd.Function):
    """Custom rotary multiplication for ONNX export."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(ctx, x, cos4, sin4):
        return (x * cos4) + (_rotate_half(x) * sin4)

    @staticmethod
    def symbolic(g, x, cos4, sin4):
        """Emit a Custom ONNX op `RotaryMul` for Ascend fusion."""
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
    """Custom RmsNorm for ONNX export."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(ctx, x, gamma, epsilon: float):
        x_fp32 = x.to(torch.float32)
        var = (x_fp32 * x_fp32).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + float(epsilon))
        y = (x_fp32 * rstd).to(x.dtype) * gamma
        return y, rstd

    @staticmethod
    def symbolic(g, x, gamma, epsilon: float):
        """Emit Custom ONNX op `RmsNorm` (returns normalized y and rstd)."""
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
    return _RmsNormCustom.apply(x, gamma, float(epsilon))


def _make_flash_attn_mask(attention_mask, q_len, k_len, past_len):
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    return (causal | padding).to(torch.bool)


class _PromptFlashAttentionCustom(torch.autograd.Function):
    """Custom PromptFlashAttention for prefill ONNX export."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(
        ctx, query, key, value, atten_mask,
        num_heads, scale_value, input_layout, num_key_value_heads,
        sparse_mode, inner_precise, pre_tokens, next_tokens,
    ):
        """Reference attention used to validate the custom op (not exported as-is)."""
        layout = str(input_layout).upper()
        if layout == "BSH":
            B, S, H_q = query.shape
            head_dim = H_q // int(num_heads)
            q = query.view(B, S, int(num_heads), head_dim).transpose(1, 2)
            k = key.view(B, S, int(num_key_value_heads), head_dim).transpose(1, 2)
            v = value.view(B, S, int(num_key_value_heads), head_dim).transpose(1, 2)
            bsh_input = True
        else:
            q, k, v = query, key, value
            if layout in ("BSND", "SBND"):
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
            bsh_input = False
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
        elif int(sparse_mode) in (2, 3):
            q_len = attn.shape[-2]
            k_len = attn.shape[-1]
            ar_q = torch.arange(q_len, device=attn.device)
            ar_k = torch.arange(k_len, device=attn.device)
            causal = ar_k[None, :] > ar_q[:, None]
            causal = causal[None, None, :, :].expand(attn.shape[0], attn.shape[1], q_len, k_len)
            attn = attn.masked_fill(causal, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        if bsh_input:
            out = out.transpose(1, 2).reshape(B, S, H_q)
        elif layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(
        g, query, key, value, atten_mask,
        num_heads, scale_value, input_layout, num_key_value_heads,
        sparse_mode, inner_precise, pre_tokens, next_tokens,
    ):
        """Emit Custom ONNX op `PromptFlashAttention` for prefill (Ascend)."""
        # pylint: disable=unused-argument
        # Attribute set matches jina_reranker_v3 (verified working on 310P3).
        # sparse_mode/pre_tokens/next_tokens are intentionally NOT emitted —
        # jina's working config omits them, and emitting them caused NaN on 310P3.
        if atten_mask is None:
            y = g.op(
                "Custom", query, key, value,
                type_s="PromptFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_key_value_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                inner_precise_i=int(inner_precise),
            )
        else:
            y = g.op(
                "Custom", query, key, value, atten_mask,
                type_s="PromptFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                num_heads_i=int(num_heads),
                num_key_value_heads_i=int(num_key_value_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                inner_precise_i=int(inner_precise),
            )
        y.setType(query.type())
        return y


def prompt_flash_attention(
    query, key, value, atten_mask,
    num_heads, scale_value, input_layout, num_key_value_heads,
    sparse_mode=0, inner_precise=1, pre_tokens=214748647, next_tokens=0,
):
    return _PromptFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout), int(num_key_value_heads),
        int(sparse_mode), int(inner_precise), int(pre_tokens), int(next_tokens),
    )


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention for decode ONNX export."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(
        ctx, query, key, value, atten_mask,
        num_heads, scale_value, input_layout, num_key_value_heads,
        block_size, inner_precise,
    ):
        """Reference attention used to validate the custom op (not exported as-is)."""
        q, k, v = query, key, value
        layout = str(input_layout).upper()
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
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        if layout in ("BSND", "SBND"):
            out = out.permute(0, 2, 1, 3)
        return out

    @staticmethod
    def symbolic(
        g, query, key, value, atten_mask,
        num_heads, scale_value, input_layout, num_key_value_heads,
        block_size, inner_precise,
    ):
        """Emit Custom ONNX op `IncreFlashAttention` for decode (Ascend)."""
        if atten_mask is None:
            y = g.op(
                "Custom", query, key, value,
                type_s="IncreFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                output_num_i=1,
                input_index_i=[0, 1, 2],
                num_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                num_key_value_heads_i=int(num_key_value_heads),
                block_size_i=int(block_size),
                inner_precise_i=int(inner_precise),
            )
        else:
            y = g.op(
                "Custom", query, key, value, atten_mask,
                type_s="IncreFlashAttention",
                input_names_s=_as_list_str(["query", "key", "value", "atten_mask"]),
                optional_input_names_s=_as_list_str(["atten_mask"]),
                output_names_s=_as_list_str(["attention_out"]),
                output_num_i=1,
                input_index_i=[0, 1, 2, 3],
                num_heads_i=int(num_heads),
                scale_value_f=float(scale_value),
                input_layout_s=str(input_layout),
                num_key_value_heads_i=int(num_key_value_heads),
                block_size_i=int(block_size),
                inner_precise_i=int(inner_precise),
            )
        y.setType(query.type())
        return y


def incre_flash_attention(
    query, key, value, atten_mask,
    num_heads, scale_value, input_layout, num_key_value_heads,
    block_size=0, inner_precise=1,
):
    return _IncreFlashAttentionCustom.apply(
        query, key, value, atten_mask,
        int(num_heads), float(scale_value), str(input_layout), int(num_key_value_heads),
        int(block_size), int(inner_precise),
    )


class _SwiGluCustom(torch.autograd.Function):
    """Custom SwiGlu for ONNX export."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(ctx, x, dim: int):
        d = int(dim)
        if d < 0:
            d = x.dim() + d
        a, b = torch.chunk(x, 2, dim=d)
        return F.silu(a) * b

    @staticmethod
    def symbolic(g, x, dim: int):
        """Emit Custom ONNX op `SwiGlu` (SiLU-gated linear unit fusion)."""
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
        y.setType(x.type())
        return y


def swiglu(x, dim: int = -1):
    return _SwiGluCustom.apply(x, int(dim))


class _ScatterCustom(torch.autograd.Function):
    """Custom Scatter for KV cache update in decode."""

    @staticmethod
    # pylint: disable=unused-argument
    def forward(ctx, var, indices, updates, reduce: str, axis: int):
        """Reference scatter-update for KV cache (validates the custom op)."""
        if str(reduce) != "update":
            raise RuntimeError("Only reduce='update' is supported.")
        ax = int(axis)
        if ax < 0:
            ax = var.dim() + ax
        if var.dim() != 4 or ax != 2:
            raise RuntimeError("Only 4D var with axis=-2/2 is supported.")
        bsz, num_heads, _, _ = var.shape
        pos = indices
        if pos.dim() == 2 and pos.shape[-1] == 1:
            pos = pos.squeeze(-1)
        pos = pos.to(torch.long).view(bsz)
        upd = updates
        if upd.dim() == 4 and upd.shape[2] == 1:
            upd = upd[:, :, 0, :]
        out = var.clone()
        b = torch.arange(bsz, device=out.device).view(bsz, 1).expand(bsz, num_heads)
        h = torch.arange(num_heads, device=out.device).view(1, num_heads).expand(bsz, num_heads)
        s = pos.view(bsz, 1).expand(bsz, num_heads)
        out[b, h, s, :] = upd
        return out

    @staticmethod
    def symbolic(g, var, indices, updates, reduce: str, axis: int):
        """Emit Custom ONNX op `Scatter` for in-place KV cache update."""
        y = g.op(
            "Custom", var, indices, updates,
            type_s="Scatter",
            input_names_s=_as_list_str(["var", "indices", "updates"]),
            optional_input_names_s=_as_list_str([]),
            output_names_s=_as_list_str(["var"]),
            output_num_i=1,
            input_index_i=[0, 1, 2],
            reduce_s=str(reduce),
            axis_i=int(axis),
        )
        y.setType(var.type())
        return y


def scatter(var, indices, updates, reduce: str = "update", axis: int = -2):
    return _ScatterCustom.apply(var, indices, updates, str(reduce), int(axis))


# ---------------------------------------------------------------------------
# Layer forward helpers (ported from qwen3_1.7b, adapted for Qwen3-ASR thinker)
# ---------------------------------------------------------------------------

def _rms_norm_layer(norm_mod, x):
    gamma = norm_mod.weight
    eps = getattr(norm_mod, "variance_epsilon", 1e-6)
    y, _ = rms_norm(x, gamma, eps)
    return y


def _make_additive_causal_mask(attention_mask, q_len, k_len, past_len, dtype):
    mask_value = torch.finfo(dtype).min
    ar_q = torch.arange(q_len, device=attention_mask.device)
    ar_k = torch.arange(k_len, device=attention_mask.device)
    causal = ar_k[None, :] > (past_len + ar_q[:, None])
    causal = causal.to(dtype) * mask_value
    causal = causal[None, None, :, :].expand(attention_mask.shape[0], 1, q_len, k_len)
    padding = (1.0 - attention_mask.to(dtype)) * mask_value
    padding = padding[:, None, None, :]
    return causal + padding


# pylint: disable=protected-access
def _compute_qkv(attn_mod, hidden_states):
    """Fuse q/k/v projections into a single Linear (returns q,k,v and split sizes)."""
    q_w = attn_mod.q_proj.weight
    k_w = attn_mod.k_proj.weight
    v_w = attn_mod.v_proj.weight
    q_b = getattr(attn_mod.q_proj, "bias", None)
    k_b = getattr(attn_mod.k_proj, "bias", None)
    v_b = getattr(attn_mod.v_proj, "bias", None)
    w = torch.cat([q_w, k_w, v_w], dim=0)
    b = None if q_b is None else torch.cat([q_b, k_b, v_b], dim=0)
    q_out_features = int(q_w.shape[0])
    kv_out_features = int(k_w.shape[0])
    qkv = F.linear(hidden_states, w, b)
    q_lin = qkv[..., :q_out_features]
    k_lin = qkv[..., q_out_features: q_out_features + kv_out_features]
    v_lin = qkv[..., q_out_features + kv_out_features:]
    return q_lin, k_lin, v_lin, q_out_features, kv_out_features


def _compute_prefill_attention(query_states, key_states, value_states, attention_mask,
                               num_heads, num_kv_heads, scaling, input_shape):
    """Run prefill attention via PromptFlashAttention; returns (out, k, v) transposed."""
    q_len, k_len = query_states.shape[-3], key_states.shape[-3]
    orig_dtype = query_states.dtype
    flash_mask = _make_flash_attn_mask(attention_mask, q_len, k_len, 0)
    q_3d = query_states.reshape(*input_shape, -1).to(torch.float16)
    k_3d = key_states.reshape(*input_shape, -1).to(torch.float16)
    v_3d = value_states.reshape(*input_shape, -1).to(torch.float16)
    attn_output = prompt_flash_attention(
        q_3d, k_3d, v_3d, flash_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BSH", num_key_value_heads=num_kv_heads,
        sparse_mode=0, inner_precise=1,
    ).to(orig_dtype)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    return attn_output, key_states, value_states


def _compute_decode_attention(query_states, key_states, value_states, attention_mask,
                              num_heads, num_kv_heads, scaling, input_shape):
    pad_mask = attention_mask[:, None, None, :].to(torch.bool).logical_not()
    attn_output = incre_flash_attention(
        query_states, key_states, value_states, pad_mask,
        num_heads=num_heads, scale_value=float(scaling),
        input_layout="BNSD", num_key_value_heads=num_kv_heads, inner_precise=1,
    )
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
    return attn_output


def _text_attn_forward(attn_mod, hidden_states, cos4, sin4, attention_mask,
                       cache_pos, past_key, past_value):
    """Attention forward shared by prefill (past_key=None) and decode (past_key set)."""
    input_shape = hidden_states.shape[:-1]
    head_dim = attn_mod.head_dim
    num_heads = attn_mod.config.num_attention_heads
    num_kv_heads = attn_mod.config.num_key_value_heads
    hidden_shape = (*input_shape, -1, head_dim)
    scaling = getattr(attn_mod, "scaling", 1.0 / (head_dim ** 0.5))

    q_lin, k_lin, v_lin, _, _ = _compute_qkv(attn_mod, hidden_states)

    query_states = q_lin.view(hidden_shape)
    key_states = k_lin.view(hidden_shape)
    if hasattr(attn_mod, "q_norm"):
        query_states = _rms_norm_layer(attn_mod.q_norm, query_states)
    if hasattr(attn_mod, "k_norm"):
        key_states = _rms_norm_layer(attn_mod.k_norm, key_states)
    value_states = v_lin.view(hidden_shape)

    if past_key is not None:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

    query_states = rotary_mul(query_states, cos4, sin4)
    key_states = rotary_mul(key_states, cos4, sin4)

    if past_key is not None:
        pos = cache_pos[:, -1]
        key_states = scatter(past_key, pos, key_states, reduce="update", axis=-2)
        value_states = scatter(past_value, pos, value_states, reduce="update", axis=-2)
        attn_output = _compute_decode_attention(
            query_states, key_states, value_states, attention_mask,
            num_heads, num_kv_heads, scaling, input_shape,
        )
    else:
        attn_output, key_states, value_states = _compute_prefill_attention(
            query_states, key_states, value_states, attention_mask,
            num_heads, num_kv_heads, scaling, input_shape,
        )

    attn_output = attn_mod.o_proj(attn_output)
    return attn_output, key_states, value_states


# pylint: disable=protected-access
def _mlp_gate_up_linear(mlp_mod, x):
    """Fuse gate/up projections into a single Linear; returns (gate, up) split."""
    gate_w = mlp_mod.gate_proj.weight
    up_w = mlp_mod.up_proj.weight
    gate_b = getattr(mlp_mod.gate_proj, "bias", None)
    up_b = getattr(mlp_mod.up_proj, "bias", None)
    w = torch.cat([gate_w, up_w], dim=0)
    b = None if gate_b is None else torch.cat([gate_b, up_b], dim=0)
    y = F.linear(x, w, b)
    gate_out_features = int(gate_w.shape[0])
    gate = y[..., :gate_out_features]
    up = y[..., gate_out_features:]
    return gate, up


def _mlp_forward(mlp_mod, x):
    if hasattr(mlp_mod, "gate_proj") and hasattr(mlp_mod, "up_proj") and hasattr(mlp_mod, "down_proj"):
        gate, up = _mlp_gate_up_linear(mlp_mod, x)
        sw = swiglu(torch.cat([gate, up], dim=-1), dim=-1)
        return mlp_mod.down_proj(sw)
    return mlp_mod(x)


# ---------------------------------------------------------------------------
# ASR text Prefill / Decode wrappers (embed_tokens + audio scatter + lm_head
# all in-graph; host only does tokenizer + argmax)
# ---------------------------------------------------------------------------

class Qwen3AsrTextPrefill(torch.nn.Module):
    """Prefill with embed_tokens + audio scatter + lm_head inside the graph.

    Inputs: input_ids, audio_features, attention_mask, position_ids.
    Outputs: logits (last valid position only), present_k, present_v.
    """

    def __init__(self, thinker):
        super().__init__()
        self.model = thinker.model
        self.embed_tokens = thinker.model.embed_tokens
        self.lm_head = thinker.lm_head
        self.audio_token_id = int(thinker.config.audio_token_id)
        if hasattr(self.model, "config") and hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"

    # pylint: disable=protected-access
    def forward(self, input_ids, audio_features, attention_mask, position_ids):
        """Run one prefill step: embed + audio scatter + transformer + lm_head at last pos."""
        inputs_embeds = self.embed_tokens(input_ids)
        # Place audio_features at the audio_token_id positions. The chat template
        # expands <|audio_pad|> to audio_token_len (390) copies, matching
        # audio_features.shape[1]. We avoid torch.masked_scatter here because it
        # exports to ScatterND+NonZero, which CANN's batch fusion folds into a
        # cyclic Batch_* sub-graph. gather+where emits pure CANN-friendly ops.
        hidden = inputs_embeds.shape[-1]
        audio_pos_mask = input_ids == self.audio_token_id  # [1, seq] bool
        # Ascend CumSum rejects bool; cast to int32 first.
        audio_idx = audio_pos_mask.to(torch.int32).cumsum(dim=1).long() - 1
        audio_idx = audio_idx.clamp(min=0)  # avoid negative at non-audio positions
        audio_gathered = torch.gather(
            audio_features, 1,
            audio_idx.unsqueeze(-1).expand(-1, -1, hidden),
        )  # [1, seq, hidden]
        audio_mask_full = audio_pos_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(audio_mask_full, audio_gathered, inputs_embeds)

        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(2) if cos.dim() == 3 else cos
        sin4 = sin.unsqueeze(2) if sin.dim() == 3 else sin

        hidden_states = inputs_embeds
        present_k = []
        present_v = []
        for layer in self.model.layers:
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4,
                attention_mask, None, None, None,
            )
            pk = torch.cat(
                [pk, pk.new_zeros(pk.shape[0], pk.shape[1], KV_CACHE_LEN, pk.shape[3])],
                dim=2,
            )[:, :, :KV_CACHE_LEN, :]
            pv = torch.cat(
                [pv, pv.new_zeros(pv.shape[0], pv.shape[1], KV_CACHE_LEN, pv.shape[3])],
                dim=2,
            )[:, :, :KV_CACHE_LEN, :]
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = _mlp_forward(layer.mlp, hidden_states)
            hidden_states = residual + hidden_states
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)

        # Apply lm_head only at the LAST valid prompt position. The host argmaxes.
        # attention_mask is [1, seq]; last_idx is the index of the final 1.
        # last_valid_idx = sum(attention_mask) - 1; shape [1]
        last_valid_idx = attention_mask.sum(dim=1, keepdim=True).long() - 1  # [1, 1]
        # Gather hidden at last valid position: result [1, 1, hidden]
        gather_idx = last_valid_idx.unsqueeze(-1).expand(-1, 1, hidden_states.shape[-1])
        last_hidden = hidden_states.gather(1, gather_idx)  # [1, 1, hidden]
        last_logits = self.lm_head(last_hidden)  # [1, 1, vocab]
        return last_logits, present_k, present_v


class Qwen3AsrTextDecode(torch.nn.Module):
    """Decode with embed_tokens + lm_head inside the graph.

    Inputs: input_ids, attention_mask, position_ids, past_key_cache, past_value_cache.
    Outputs: logits [1,1,vocab], present_k, present_v.
    """

    def __init__(self, thinker):
        super().__init__()
        self.model = thinker.model
        self.embed_tokens = thinker.model.embed_tokens
        self.lm_head = thinker.lm_head
        if hasattr(self.model, "config") and hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"

    # pylint: disable=protected-access
    def forward(self, input_ids, attention_mask, position_ids,
                past_key_cache, past_value_cache):
        """Run one decode step: embed + transformer + scatter KV update + lm_head."""
        inputs_embeds = self.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(inputs_embeds, position_ids)
        cos4 = cos.unsqueeze(1)
        sin4 = sin.unsqueeze(1)

        hidden_states = inputs_embeds
        present_k = []
        present_v = []
        past_k_layers = past_key_cache.unbind(0)
        past_v_layers = past_value_cache.unbind(0)
        text_position_ids = position_ids[0]

        for i, layer in enumerate(self.model.layers):
            pk_in = past_k_layers[i]
            pv_in = past_v_layers[i]
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.input_layernorm, hidden_states)
            attn_out, pk, pv = _text_attn_forward(
                layer.self_attn, hidden_states, cos4, sin4,
                attention_mask, text_position_ids, pk_in, pv_in,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = _rms_norm_layer(layer.post_attention_layernorm, hidden_states)
            hidden_states = _mlp_forward(layer.mlp, hidden_states)
            hidden_states = residual + hidden_states
            present_k.append(pk)
            present_v.append(pv)

        hidden_states = _rms_norm_layer(self.model.norm, hidden_states)
        present_k = torch.stack(present_k, dim=0)
        present_v = torch.stack(present_v, dim=0)
        logits = self.lm_head(hidden_states)  # [1, 1, vocab]
        return logits, present_k, present_v


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _ensure_chat_template(tokenizer, model_path):
    if getattr(tokenizer, "chat_template", None):
        return
    p = os.path.join(model_path, "chat_template.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        tpl = data.get("chat_template")
        if isinstance(tpl, str) and tpl.strip() != "":
            tokenizer.chat_template = tpl


def _get_kv_cache_config(thinker):
    text_cfg = thinker.model.config
    num_layers = text_cfg.num_hidden_layers
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim = getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads)
    hidden_size = text_cfg.hidden_size
    return num_layers, num_kv_heads, head_dim, hidden_size


def _prepare_output_paths(output_dir):
    """Create per-sub-model output dirs and return their ONNX paths."""
    audio_dir = Path(output_dir) / "audio_encoder"
    prefill_dir = Path(output_dir) / "prefill"
    decode_dir = Path(output_dir) / "decode"
    for d in (audio_dir, prefill_dir, decode_dir):
        d.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "qwen3_asr_audio_encoder_fp32.onnx"
    suffix = EXPORT_DTYPE
    prefill_path = prefill_dir / f"qwen3_asr_text_prefill_{suffix}.onnx"
    decode_path = decode_dir / f"qwen3_asr_text_decode_{suffix}.onnx"
    return audio_path, prefill_path, decode_path


def _export_audio_encoder(audio_encoder, audio_onnx_path, input_features, opset):
    """Export the audio encoder torch module to ONNX."""
    print(f"Exporting audio encoder to {audio_onnx_path}...")
    with torch.no_grad():
        torch.onnx.export(
            audio_encoder,
            (input_features,),
            str(audio_onnx_path),
            input_names=["input_features"],
            output_names=["audio_features"],
            opset_version=int(opset),
            dynamo=False,
            do_constant_folding=True,
        )
    print("Audio encoder exported.")


def _create_prefill_dummy_inputs(device, seq_len, audio_token_len, hidden_size):
    """Dummy inputs for Qwen3AsrTextPrefill.

    input_ids has exactly audio_token_len occurrences of audio_token_id so that
    masked_scatter's source length matches the True count at runtime.
    """
    dummy_seq = int(seq_len)
    audio_id, pad_id = 151676, 0
    ids = torch.full((1, dummy_seq), pad_id, dtype=torch.int64, device=device)
    # Place audio_token_len audio tokens contiguously starting at position 2
    # (leave 2 template tokens before; sum(attention_mask) = dummy_seq so last_idx = seq-1).
    start = 2
    ids[0, start:start + int(audio_token_len)] = audio_id
    dummy_attention_mask = torch.ones(1, dummy_seq, dtype=torch.int64, device=device)
    dummy_position_ids = torch.arange(dummy_seq, dtype=torch.int64, device=device).view(1, 1, -1).expand(3, 1, -1)
    dummy_audio = torch.randn(1, int(audio_token_len), int(hidden_size), dtype=torch.float32, device=device)
    return ids, dummy_audio, dummy_attention_mask, dummy_position_ids


def _export_prefill_onnx(prefill, prefill_path, dummy_inputs, opset):
    """Export Qwen3AsrTextPrefill. audio_features is static (390), not dynamic."""
    print(f"Exporting text prefill to {prefill_path}...")
    with torch.no_grad():
        torch.onnx.export(
            prefill,
            dummy_inputs,
            str(prefill_path),
            input_names=["input_ids", "audio_features", "attention_mask", "position_ids"],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "position_ids": {1: "batch", 2: "seq"},
                "logits": {0: "batch"},
                "present_key_cache": {1: "batch"},
                "present_value_cache": {1: "batch"},
            },
            external_data=True,
        )
    print("Text prefill exported.")


def _create_decode_dummy_inputs(device, num_layers, num_kv_heads, head_dim):
    dummy_past_len = int(KV_CACHE_LEN)
    dummy_input_ids = torch.zeros(1, 1, dtype=torch.int64, device=device)
    dummy_attention_mask = torch.ones(1, dummy_past_len, dtype=torch.int64, device=device)
    dummy_position_ids = torch.tensor([[[dummy_past_len - 1]]], dtype=torch.int64, device=device).expand(3, 1, 1)
    dummy_k = torch.zeros(num_layers, 1, num_kv_heads, dummy_past_len, head_dim, dtype=torch.float32, device=device)
    dummy_v = torch.zeros(num_layers, 1, num_kv_heads, dummy_past_len, head_dim, dtype=torch.float32, device=device)
    return dummy_input_ids, dummy_attention_mask, dummy_position_ids, dummy_k, dummy_v


def _export_decode_onnx(decode, decode_path, dummy_inputs, opset):
    """Export Qwen3AsrTextDecode. input_ids replaces inputs_embeds."""
    print(f"Exporting text decode to {decode_path}...")
    with torch.no_grad():
        torch.onnx.export(
            decode,
            dummy_inputs,
            str(decode_path),
            input_names=[
                "input_ids", "attention_mask", "position_ids",
                "past_key_cache", "past_value_cache",
            ],
            output_names=["logits", "present_key_cache", "present_value_cache"],
            opset_version=int(opset),
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "position_ids": {1: "batch"},
                "logits": {0: "batch"},
                "past_key_cache": {1: "batch"},
                "past_value_cache": {1: "batch"},
                "present_key_cache": {1: "batch"},
                "present_value_cache": {1: "batch"},
            },
            external_data=True,
        )
    print("Text decode exported.")


def export_qwen3_asr(model, feature_extractor, output_dir, opset, device):
    """Export audio encoder + prefill + decode to ONNX under output_dir."""
    thinker = model.thinker
    audio_token_id = int(thinker.config.audio_token_id)
    num_layers, num_kv_heads, head_dim, hidden_size = _get_kv_cache_config(thinker)
    kv_dtype = next(model.parameters()).dtype
    print(f"  num_layers={num_layers}, num_kv_heads={num_kv_heads}, head_dim={head_dim}, "
          f"hidden_size={hidden_size}, audio_token_id={audio_token_id}, kv_dtype={kv_dtype}")
    print(f"  KV_CACHE_LEN={KV_CACHE_LEN}")

    audio_path, prefill_path, decode_path = _prepare_output_paths(output_dir)

    # --- Audio encoder ---
    spec = _build_audio_spec(
        n_mels=int(getattr(feature_extractor, "feature_size", 128)),
        n_frames=int(getattr(feature_extractor, "nb_max_frames", 3000)),
        n_window=int(getattr(thinker.audio_tower, "n_window", 100)),
        n_window_infer=int(getattr(thinker.audio_tower, "n_window_infer", 400)),
    )
    audio_token_len = spec.aftercnn_total
    audio_encoder = Qwen3AsrAudioEncoderOnnx(thinker.audio_tower, spec).float()

    wav = np.zeros((int(getattr(feature_extractor, "n_samples", 480000)),), dtype=np.float32)
    fe = feature_extractor(wav, sampling_rate=16000, return_attention_mask=True)
    input_features = torch.tensor(fe["input_features"], dtype=torch.float32).to(device)

    _export_audio_encoder(audio_encoder, audio_path, input_features, opset)

    # --- embed_tokens + lm_head live in the prefill/decode graphs (gather +
    # lm_head). Host no longer needs them. Sanity check tied weights.
    embed_tokens = thinker.model.embed_tokens
    if hasattr(thinker, "lm_head") and thinker.lm_head.weight is not embed_tokens.weight:
        if not torch.equal(thinker.lm_head.weight.detach(), embed_tokens.weight.detach()):
            print("WARNING: thinker.lm_head.weight != thinker.model.embed_tokens.weight; "
                  "the in-graph path will store both as separate constants.")

    # --- Prefill / Decode wrappers (single graph, all layers) ---
    thinker.model.config._attn_implementation = "eager"

    prefill_dummies = _create_prefill_dummy_inputs(
        device, seq_len=audio_token_len + 32, audio_token_len=audio_token_len,
        hidden_size=hidden_size,
    )
    decode_dummies = _create_decode_dummy_inputs(
        device, num_layers, num_kv_heads, head_dim,
    )
    prefill = Qwen3AsrTextPrefill(thinker).to(device).eval()
    decode = Qwen3AsrTextDecode(thinker).to(device).eval()

    with torch.no_grad():
        _ = prefill(*prefill_dummies)
        _ = decode(*decode_dummies)

    _export_prefill_onnx(prefill, prefill_path, prefill_dummies, opset)
    _export_decode_onnx(decode, decode_path, decode_dummies, opset)

    print(f"\nExport finished. Files under {output_dir}/{{audio_encoder,prefill,decode}}/")


def _parse_args_and_config():
    """Parse CLI args and propagate KV_CACHE_LEN / EXPORT_DTYPE to module globals."""
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR 1.7B to ONNX (separate prefill/decode graphs)")
    parser.add_argument("--model-path", type=str, default="./Qwen3-ASR-1.7B")
    parser.add_argument("--output-dir", type=str, default="./onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--kv-cache-len", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    args = parser.parse_args()

    global KV_CACHE_LEN
    KV_CACHE_LEN = int(args.kv_cache_len)
    global EXPORT_DTYPE
    EXPORT_DTYPE = str(args.dtype)
    return args


def _load_and_export(args):
    """Load Qwen3-ASR from args.model_path and run the full ONNX export pipeline."""
    _ensure_dir(args.output_dir)
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    device = torch.device(args.device)

    print(f"\nLoading model {args.model_path} (dtype={args.dtype})...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, fix_mistral_regex=True)
    _ensure_chat_template(tokenizer, args.model_path)

    export_qwen3_asr(model, feature_extractor, args.output_dir, args.opset, str(device))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = _parse_args_and_config()
    _load_and_export(args)


if __name__ == "__main__":
    main()
