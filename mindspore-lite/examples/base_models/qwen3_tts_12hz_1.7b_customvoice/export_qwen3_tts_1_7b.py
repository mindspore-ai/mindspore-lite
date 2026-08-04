"""Export Qwen3-TTS ONNX models (talker KV, speech decoder, code predictor) in one entry script."""

from __future__ import annotations

import argparse
import inspect
import importlib
import json
import os
import sys
import types
from collections import Counter
from functools import lru_cache
from typing import Any
import numpy as np
import torch.nn.functional as F

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _save_ref_io(
    model: Any,
    example_inputs: tuple,
    input_names: list[str],
    output_names: list[str],
    out_path: str,
) -> None:
    """Save model reference inputs and outputs as .npz for MindIR precision validation.

    Files are saved next to the ONNX with suffix ``_ref_io.npz``.
    Each input/output is stored as ``{name}`` key in the archive.
    Tensors keep their original dtype (bf16 stays bf16 via ml_dtypes, fp32 stays fp32).
    """
    torch_mod = _require_module("torch")
    try:
        ml_dtypes = _import_module("ml_dtypes")
        bf16 = getattr(ml_dtypes, "bfloat16", None)
    except (ImportError, ModuleNotFoundError):
        bf16 = None

    ref_path = out_path.replace(".onnx", "_ref_io.npz")
    if not ref_path.endswith("_ref_io.npz"):
        ref_path = out_path + "_ref_io.npz"

    save_dict: dict[str, np.ndarray] = {}

    def _tensor_to_np(t: Any) -> np.ndarray:
        """Convert a torch tensor to numpy, keeping original dtype."""
        if not isinstance(t, torch_mod.Tensor):
            return np.asarray(t)
        dt = t.dtype
        # BFloat16: numpy doesn't support bf16 directly, use ml_dtypes
        if dt == torch_mod.bfloat16:
            arr = t.detach().cpu().to(torch_mod.float32).numpy()
            if bf16 is not None:
                arr = arr.astype(bf16)
            return arr
        # Other dtypes: keep as-is
        return t.detach().cpu().numpy()

    # Save inputs
    with torch_mod.no_grad():
        for name, inp in zip(input_names, example_inputs):
            save_dict[f"in_{name}"] = _tensor_to_np(inp)

        # Compute outputs
        outputs = model(*example_inputs)
        if isinstance(outputs, torch_mod.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)

        for name, out in zip(output_names, outputs):
            save_dict[f"out_{name}"] = _tensor_to_np(out)

    np.savez(ref_path, **save_dict)
    print(f"  [ref_io] saved reference I/O to {ref_path}")

@lru_cache(maxsize=None)
def _import_module(name: str) -> Any:
    """Import a module by name with caching."""
    return importlib.import_module(name)


def _optional_module(name: str) -> Any | None:
    """Import a module if available, otherwise return None."""
    try:
        return _import_module(name)
    except (ImportError, ModuleNotFoundError):
        return None


def _require_module(name: str) -> Any:
    """Import a required module or raise a clear runtime error."""
    mod = _optional_module(name)
    if mod is None:
        raise RuntimeError(f"Missing dependency: {name!r}.")
    return mod


_TORCH = _optional_module("torch")
_ONNX = _optional_module("onnx")

if _TORCH is None:

    class _AutogradFunction:
        pass

    class _NNModule:
        pass

else:
    _AutogradFunction = _TORCH.autograd.Function
    _NNModule = _TORCH.nn.Module

onnx_helper = getattr(_ONNX, "helper", None)
onnx_numpy_helper = getattr(_ONNX, "numpy_helper", None)
torch = _TORCH
onnx = _ONNX


class CustomRoatryMul(_AutogradFunction):
    """Custom rotary multiplication operator for ONNX export."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        del ctx
        dim = int(x.shape[-1])
        half = dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos) + (rotated * sin)

    @staticmethod
    def symbolic(g: torch.Graph, x, cos, sin):
        return g.op(
            "Custom",
            x,
            cos,
            sin,
            input_names_s=["x", "r1", "r2"],
            output_names_s=["y"],
            type_s="RotaryMul",
        )


class _IncreFlashAttentionCustom(torch.autograd.Function):
    """Custom IncreFlashAttention for ONNX export."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        atten_mask,
        num_heads: int,
        scale_value: float,
        input_layout: str,
        num_key_value_heads: int,
        block_size: int,
        inner_precise: int,
    ):
        """Incremental flash attention forward (fallback to manual matmul)."""
        del ctx, block_size, inner_precise
        q = query
        k = key
        v = value
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
        g,
        query,
        key,
        value,
        atten_mask,
        num_heads: int,
        scale_value: float,
        input_layout: str,
        num_key_value_heads: int,
        block_size: int,
        inner_precise: int,
    ):
        """ONNX symbolic for IncreFlashAttention custom op."""
        if atten_mask is None:
            y = g.op(
                "Custom",
                query,
                key,
                value,
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
                "Custom",
                query,
                key,
                value,
                atten_mask,
                type_s="IncreFlashAttention",
                input_names_s=["query", "key", "value", "atten_mask"],
                optional_input_names_s=["atten_mask"],
                output_names_s=["attention_out"],
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
    query,
    key,
    value,
    atten_mask,
    num_heads: int,
    scale_value: float,
    input_layout: str,
    num_key_value_heads: int,
    block_size: int = 0,
    inner_precise: int = 1,
):
    """Incremental flash attention using custom ONNX op."""
    return _IncreFlashAttentionCustom.apply(
        query,
        key,
        value,
        atten_mask,
        int(num_heads),
        float(scale_value),
        str(input_layout),
        int(num_key_value_heads),
        int(block_size),
        int(inner_precise),
    )


class CustomScatterUpdate(_AutogradFunction):
    """Custom scatter update operator for KV cache update."""

    @staticmethod
    def forward(
        ctx: Any,
        var: torch.Tensor,
        indices: torch.Tensor,
        updates: torch.Tensor,
        axis: int = 0,
    ):
        """Update cache position(s) along the given axis.
        axis=2 → BNSD [B, n_kv, S, D];  axis=1 → BSND [B, S, n_kv, D]."""
        del ctx
        ax = int(axis)
        cache_total = int(var.size(ax))
        pos = indices.to(torch.int64)
        if pos.dim() == 0:
            pos = pos.unsqueeze(0)
        bsz = int(var.size(0))
        if pos.numel() < bsz:
            pos = pos.expand(bsz)
        e = torch.nn.functional.one_hot(pos, num_classes=cache_total).to(
            var.dtype)  # [bsz, S]
        if ax == 2:   # BNSD [B, n_kv, S, D]
            e_mul = e.view(bsz, 1, cache_total, 1)
            old = (var * e_mul).sum(dim=2, keepdim=True)
        elif ax == 1:  # BSND [B, S, n_kv, D]
            e_mul = e.view(bsz, cache_total, 1, 1)
            old = (var * e_mul).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported axis={ax} for ScatterUpdate")
        return var + e_mul * (updates.to(var.dtype) - old.to(var.dtype))

    @staticmethod
    def symbolic(g: torch.Graph, var, indices, updates, axis: int = 0):
        return g.op(
            "Custom",
            var,
            indices,
            updates,
            input_index_i=[0, 1, 2],
            input_names_s=["var", "indices", "updates"],
            optional_input_names_s=[],
            output_names_s=["var"],
            type_s="Scatter",
            reduce_s="update",
            axis_i=int(axis),
        )


def _kv_cache_update(
    past: torch.Tensor,
    update: torch.Tensor,
    cache_pos: torch.Tensor,
    *,
    use_custom: bool,
) -> torch.Tensor:
    """Update KV cache at cache_pos and return the updated tensor."""
    if bool(use_custom):
        indices = _to_cache_indices(cache_pos, past.device)
        return CustomScatterUpdate.apply(past, indices, update, 2)

    pos = cache_pos.to(torch.int64).reshape(-1, 1, 1, 1)
    index = pos.expand(update.size(0), update.size(1), 1, update.size(3))
    result = past.clone()
    result.scatter_(2, index, update.to(past.dtype))
    return result


def _to_cache_indices(cache_pos: Any, device: Any) -> torch.Tensor:
    """Convert cache_pos to a 1D int64 tensor with batch-sized indices."""
    if torch.is_tensor(cache_pos):
        return cache_pos.reshape(-1).to(torch.int64)
    return torch.tensor([int(cache_pos)], dtype=torch.int64, device=device)


class CustomSwiGlu(_AutogradFunction):
    """Custom SwiGLU activation operator for ONNX export."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, dim: int = -1):
        """Compute SwiGLU activation for the last dimension."""
        del ctx
        dim = int(dim)
        split = int(x.shape[dim]) // 2
        a, b = torch.split(x, [split, split], dim=dim)
        return torch.nn.functional.silu(a) * b

    @staticmethod
    def symbolic(g: torch.Graph, x, dim: int = -1):
        return g.op(
            "Custom",
            x,
            input_index_i=[0],
            input_names_s=["x"],
            optional_input_names_s=[],
            output_names_s=["y"],
            type_s="SwiGlu",
            dim_i=int(dim),
        )


def _rotary_mul_plain(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = int(x.shape[-1])
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


def _prepare_mrope_cos_sin(
    cos,
    sin,
    mrope_section,
    mrope_interleaved: bool = False,
    unsqueeze_dim: int = 1,
):
    """Prepare multimodal RoPE cos/sin tensors for talker attention."""
    if mrope_interleaved:
        dim = int(cos.shape[-1])
        modality_num = int(len(mrope_section))
        cos_half = cos[..., : dim // 2]
        sin_half = sin[..., : dim // 2]

        cos_t = cos_half[0].clone()
        sin_t = sin_half[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg_idx = i
            end_idx = int(n) * modality_num
            cos_src = cos_half[i, ..., beg_idx:end_idx:modality_num]
            sin_src = sin_half[i, ..., beg_idx:end_idx:modality_num]
            cos_t[..., beg_idx:end_idx:modality_num] = cos_src
            sin_t[..., beg_idx:end_idx:modality_num] = sin_src

        cos = torch.cat([cos_t] * 2, dim=-1).unsqueeze(int(unsqueeze_dim))
        sin = torch.cat([sin_t] * 2, dim=-1).unsqueeze(int(unsqueeze_dim))
        return cos, sin

    mrope_section = mrope_section * 2
    cos_parts = [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))]
    sin_parts = [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))]
    cos = torch.cat(cos_parts, dim=-1).unsqueeze(int(unsqueeze_dim))
    sin = torch.cat(sin_parts, dim=-1).unsqueeze(int(unsqueeze_dim))
    return cos, sin


def _ensure_torchaudio_stub() -> None:
    """Provide a minimal torchaudio stub when torchaudio is unavailable."""
    try:
        _import_module("torchaudio")
        return
    except (ImportError, OSError):
        ta = types.ModuleType("torchaudio")
        compliance = types.ModuleType("torchaudio.compliance")
        kaldi = types.ModuleType("torchaudio.compliance.kaldi")
        compliance.kaldi = kaldi
        ta.compliance = compliance
        sys.modules["torchaudio"] = ta
        sys.modules["torchaudio.compliance"] = compliance
        sys.modules["torchaudio.compliance.kaldi"] = kaldi


def _export_onnx(
    model: Any,
    example_inputs: tuple,
    out_path: str,
    opset: int,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict | None = None,
    allow_custom_ops: bool = False,
    do_constant_folding: bool = True,
) -> None:
    """Export a torch module to an ONNX file."""
    torch_mod = _require_module("torch")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    export_kwargs: dict[str, Any] = {
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": int(opset),
        "do_constant_folding": bool(do_constant_folding),
    }
    if dynamic_axes:
        export_kwargs["dynamic_axes"] = dynamic_axes
    if "dynamo" in inspect.signature(torch_mod.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    if allow_custom_ops:
        export_kwargs["operator_export_type"] = torch_mod.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
    torch_mod.onnx.export(model, example_inputs, out_path, **export_kwargs)
    _save_ref_io(model, example_inputs, input_names, output_names, out_path)


# ──────────────────────────────────────────────
# PTQ INT8 MatMul Quantization Infrastructure
# ──────────────────────────────────────────────


def _onnx_cast_to_i_from_dtype(dtype: torch.dtype) -> int:
    """Map torch dtype to ONNX Cast to_i attribute value."""
    if dtype == torch.float16:
        return 10
    if dtype == torch.float32:
        return 1
    if dtype == torch.bfloat16:
        return 16
    return 1


class _QuantLinearSymInt8(torch.autograd.Function):
    """Quantized linear layer: forward runs int8-simulated/FP32, symbolic builds ONNX.
    x_scale can be float (per-tensor) or tensor [in_features] (per-channel)."""

    _SIMULATE = os.environ.get("PTQ_INT8_SIMULATE", "0") == "1"

    @staticmethod
    def forward(ctx, x, weight_fp, bias_fp, x_scale, w_q, correction,
                w_scale_mean: float, smooth_scale, out_to_i: int):
        """Forward pass: run plain FP32 linear, or the int8-simulated quantized path."""
        del ctx, out_to_i
        if not _QuantLinearSymInt8._SIMULATE:
            return torch.nn.functional.linear(x, weight_fp, bias_fp)

        if smooth_scale is not None:
            x = x / smooth_scale.to(device=x.device, dtype=x.dtype)

        # Per-tensor 激活量化（与 ONNX AscendQuant + QuantBatchMatmul 路径对齐）
        pc_x = not isinstance(x_scale, (int, float))
        if pc_x:
            x_cmb = float(x_scale.detach().cpu().numpy().max())
        else:
            x_cmb = float(x_scale)
        ascend_scale = 1.0 / max(x_cmb, 1e-8)
        x_int8 = torch.clamp(torch.round(x * ascend_scale), -127, 127).to(torch.int8)
        combined_scale = x_cmb * float(w_scale_mean)
        y = x_int8.to(torch.int32) @ w_q.to(torch.int32).T
        y = y.to(torch.float32) * combined_scale
        if correction is not None:
            y = y * correction.to(device=y.device, dtype=y.dtype)
        if bias_fp is not None:
            y = y + bias_fp.to(device=y.device, dtype=y.dtype)
        return y.to(x.dtype)

    @staticmethod
    def symbolic(g, x, weight_fp, bias_fp, x_scale, w_q, correction,
                 w_scale_mean: float, smooth_scale, out_to_i: int):
        """Build the ONNX symbolic graph for the quantized int8 linear path."""
        del weight_fp
        import struct as _struct

        if smooth_scale is not None:
            x = g.op("Div", x, smooth_scale)

        pc_x = not isinstance(x_scale, (int, float))
        if pc_x and isinstance(x_scale, torch.Tensor):
            # Per-channel: forward 中 torch.Tensor 可读取值
            x_scale_np = x_scale.detach().cpu().numpy().astype(np.float32)
            ascend_np = (1.0 / np.maximum(x_scale_np, 1e-8)).astype(np.float32)
            scale_t = torch.from_numpy(ascend_np)
            scale_c = g.op("Constant", value_t=scale_t)
            x_cmb = float(x_scale_np.mean())
            x_i8 = g.op("Custom", x, scale_c,
                        type_s="AscendQuant",
                        input_names_s=["x", "scale"],
                        optional_input_names_s=[],
                        output_names_s=["y"], output_num_i=1,
                        input_index_i=[0, 1],
                        src_t_i=1, dst_t_i=3, offset_f=0.0)
        elif pc_x:
            # Per-channel 在 symbolic 中为 torch._C.Value，不可读，fallback per-tensor
            x_cmb = float(w_scale_mean)
            ascend_scale = 1.0
            x_i8 = g.op("Custom", x,
                        type_s="AscendQuant",
                        input_names_s=["x"], optional_input_names_s=[],
                        output_names_s=["y"], output_num_i=1, input_index_i=[0],
                        src_t_i=1, dst_t_i=3, scale_f=float(ascend_scale), offset_f=0.0)
        else:
            x_cmb = float(x_scale)
            ascend_scale = 1.0 / max(x_cmb, 1e-8)
            x_i8 = g.op("Custom", x,
                        type_s="AscendQuant",
                        input_names_s=["x"], optional_input_names_s=[],
                        output_names_s=["y"], output_num_i=1,
                        input_index_i=[0],
                        src_t_i=1, dst_t_i=3,
                        scale_f=float(ascend_scale), offset_f=0.0)

        combined_scale = float(x_cmb) * float(w_scale_mean)
        scale_bits = _struct.unpack("<I", _struct.pack("<f", combined_scale))[0]
        scale_tensor = torch.tensor([int(scale_bits)], dtype=torch.int64)
        scale_const = g.op("Constant", value_t=scale_tensor)

        op_inputs = [x_i8, w_q, scale_const]
        y = g.op("Custom", *op_inputs,
                 type_s="QuantBatchMatmul",
                 input_names_s=["x1", "x2", "scale", "offset", "bias", "pertoken_scale"],
                 optional_input_names_s=["offset", "bias", "pertoken_scale"],
                 output_names_s=["y"], output_num_i=1,
                 input_index_i=[0, 1, 2],
                 transpose_x1_s="false", transpose_x2_s="true", dtype_i=1)

        if correction is not None:
            y = g.op("Mul", y, correction)
        if bias_fp is not None:
            y = g.op("Add", y, bias_fp)
        y = g.op("Cast", y, to_i=int(out_to_i))
        y.setType(x.type())
        return y


def quant_linear_symmetric_int8(
    x, weight_fp, bias_fp,
    x_scale, w_q, w_scale,
    smooth_scale=None,
):
    """Quantized linear wrapper: forward runs FP32, symbolic builds int8 quant path.

    Args:
        x: 输入张量
        weight_fp: 原始 FP weight（forward 阶段使用）
        bias_fp: 偏置（None 时会自动创建零偏置）
        x_scale: 激活 scale（float per-tensor 或 tensor [in_features] per-channel）
        w_q: int8 量化后的 weight
        w_scale: weight scale（[out_features] per-channel）
        smooth_scale: 可选 SmoothQuant 平滑因子 [in_features]
    """
    if bias_fp is None:
        bias_fp = x.new_zeros((weight_fp.shape[0],))
    out_to_i = _onnx_cast_to_i_from_dtype(x.dtype)

    if smooth_scale is not None:
        smooth_scale = smooth_scale.to(x.dtype)

    # Weight correction: w_scale / w_scale.max → 用于 QBM per-channel scale
    per_channel_w = isinstance(w_scale, torch.Tensor) and w_scale.dim() == 1
    if per_channel_w:
        w_scale_np = w_scale.detach().cpu().numpy()
        w_scale_agg = float(w_scale_np.max())
        correction = (w_scale_np / w_scale_agg).astype(np.float32)
        correction = torch.from_numpy(correction)
        w_scale_ref = w_scale_agg
    else:
        w_scale_ref = float(w_scale) if isinstance(w_scale, torch.Tensor) else float(w_scale)
        correction = None

    # x_scale 处理：若为 per-channel tensor 则保持 tensor，否则转 float
    per_channel_x = isinstance(x_scale, torch.Tensor) and x_scale.dim() == 1
    if not per_channel_x:
        x_scale = float(x_scale)
    # per-channel tensor 保持原样传给 _QuantLinearSymInt8

    return _QuantLinearSymInt8.apply(
        x, weight_fp, bias_fp, x_scale, w_q,
        correction, w_scale_ref, smooth_scale, out_to_i,
    )


@torch.no_grad()
def quantize_weight_symmetric_int8(weight: torch.Tensor, clip_ratio: float = 0.0):
    """Quantize a weight tensor to per-channel symmetric int8.

    Args:
        weight: FP32 weight [out_features, in_features]
        clip_ratio: 可选离群值裁剪比例 (0 = no clip)

    Returns:
        w_q: int8 量化权重
        w_scale: per-channel scale [out_features]
    """
    w = weight.detach()

    # 可选：裁剪离群值
    if clip_ratio > 0 and w.numel() > 100 and w.dim() >= 2:
        flat = w.abs().reshape(w.shape[0], -1)
        k = max(1, int(flat.shape[1] * clip_ratio))
        thresholds, _ = flat.topk(k, dim=1)
        threshold = thresholds[:, -1:]
        w = w.clamp(-threshold, threshold)

    maxabs = w.abs().max(dim=1, keepdim=True).values.clamp_min(1e-8)
    w_scale = maxabs / 127.0
    w_q = torch.clamp(torch.round(w / w_scale), -127, 127).to(torch.int8)
    return w_q, w_scale.view(-1)


def _load_calib_data_jsonl(path: str, max_samples: int = 32, max_decode_steps: int = 32):
    """Load calibration data from JSONL file.

    Returns list of dicts with keys: ``input`` (numpy tensors), ``meta``, ``sequence``.
    """
    records = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if len(records) >= max_samples:
                break
    # 转换 numpy
    result = []
    for rec in records:
        inp = {}
        for name, val in rec.get("input", {}).items():
            inp[name] = np.asarray(val, dtype=np.float32 if "embed" in name else np.int64)
        meta = rec.get("meta", {})
        seq = meta.get("sequence", [])
        if len(seq) > max_decode_steps:
            seq = seq[:max_decode_steps]
        result.append({"input": inp, "meta": meta, "sequence": seq})
    return result


def _attach_observers(module: torch.nn.Module, name: str, param_store: dict) -> None:
    """Attach MinMaxObserver to a linear module for activation range collection.

    Args:
        module: 目标模块 (nn.Module)
        name: 模块名称键
        param_store: 存储校准参数的字典（``_ptq_params``）
    """
    if name not in param_store:
        param_store[name] = {}
    from torch.ao.quantization.observer import MinMaxObserver
    obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    param_store[name]["_obs"] = obs
    param_store[name]["_act_per_ch_max"] = None

    def _pre_hook(mod, args, hook_name=name, store=param_store):
        del mod
        # 防御性检查：若 _obs 已被 _compute_quant_params 清理，则跳过
        if "_obs" not in store.get(hook_name, {}):
            return
        x = args[0].detach()
        store[hook_name]["_obs"](x)
        # 同时收集 per-channel 最大值（用于 SmoothQuant）
        if x.dim() >= 2:
            flat = x.reshape(-1, x.shape[-1])
            per_ch = flat.abs().max(dim=0).values
            current = store[hook_name].get("_act_per_ch_max")
            if current is None:
                store[hook_name]["_act_per_ch_max"] = per_ch
            else:
                store[hook_name]["_act_per_ch_max"] = torch.maximum(current, per_ch)

    module.register_forward_pre_hook(_pre_hook)

def _save_ptq_params(ptq_params: dict, save_path: str) -> None:
    """Save PTQ params (int8 weights, scales) to a file for precision comparison."""
    import torch as _torch
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
    # 过滤掉不需要的字段，只保留推理所需的数据
    clean = {}
    for name, params in ptq_params.items():
        entry = {}
        for k in ("x_scale", "w_q", "w_scale", "smooth_scale"):
            if k in params:
                entry[k] = params[k]
        if entry:
            clean[name] = entry
    _torch.save(clean, save_path)


def _compute_quant_params(param_store: dict, smooth_alpha: float = 0.65, weight_clip_ratio: float = 0.01) -> None:
    """Compute quantization parameters from collected observer data.

    Args:
        param_store: ``_ptq_params`` dict with observer data
        smooth_alpha: SmoothQuant alpha (0=纯 weight, 1=纯激活, <=0 禁用)
        weight_clip_ratio: Weight 离群值裁剪比例
    """
    for _, params in param_store.items():
        obs = params.get("_obs")
        if obs is None:
            continue

        w_fp = params.get("weight_fp")
        if w_fp is None:
            continue

        act_per_ch = params.get("_act_per_ch_max")

        # ── SmoothQuant：计算 smooth_scale 并调整 activation scale ──
        smooth_scale = None
        if act_per_ch is not None and 0 < smooth_alpha < 1.0:
            max_w_per_col = w_fp.abs().max(dim=0).values.clamp_min(1e-8)
            s = (act_per_ch ** smooth_alpha) / (max_w_per_col ** (1.0 - smooth_alpha))
            s = s.clamp(1e-4, 1e4)
            smooth_scale = s
            params["smooth_scale"] = s
            # 平滑后的 per-channel 激活 scale（反量化时用 per-tensor P99 分位数）
            x_scaled_per_ch = act_per_ch / smooth_scale.clamp_min(1e-8)
            # Per-tensor 激活 scale（平滑后 per-channel 的 P99 分位数 → 对离群值更鲁棒）
            flat_pc = x_scaled_per_ch.flatten()
            k = max(1, int(flat_pc.numel() * 0.001))
            p99 = flat_pc.kthvalue(flat_pc.numel() - k).values
            params["x_scale"] = float((p99 / 127.0).item())
        else:
            maxabs = torch.maximum(obs.max_val.abs(), obs.min_val.abs()).clamp_min(1e-8)
            params["x_scale"] = float((maxabs / 127.0).cpu().item())

        # ── Weight 量化（应用 smooth_scale 到 weight） ──
        if smooth_scale is not None:
            w_to_quant = w_fp * smooth_scale.unsqueeze(0)  # [out, in] * [1, in]
        else:
            w_to_quant = w_fp
        w_q, w_scale = quantize_weight_symmetric_int8(w_to_quant, clip_ratio=weight_clip_ratio)
        params["w_q"] = w_q
        params["w_scale"] = w_scale

        # 清理 observer
        params.pop("_obs", None)
        params.pop("_act_per_ch_max", None)


def _cleanup_observers(module: torch.nn.Module) -> None:
    """Remove forward pre-hooks from the top-level module only.

    Sub-module hooks are left in place but become no-ops after _obs is
    popped from params — the _pre_hook checks for _obs existence defensively.
    Recursive cleanup is avoided to prevent removing HuggingFace internal hooks.
    """
    getattr(module, "_forward_pre_hooks").clear()


def _calibrate_talker_wrappers(
    prefill_wrapper, step_wrapper, calib_samples: list[dict],
    device: str = "cpu", smooth_alpha: float = 0.65,
    weight_clip_ratio: float = 0.01,
) -> None:
    """Run PTQ calibration: forward pass with calibration data to collect activation statistics.

    Args:
        prefill_wrapper: TalkerPrefillKVWrapper with enable_ptq=True
        step_wrapper: TalkerStepKVWrapper with enable_ptq=True
        calib_samples: 校准样本列表
        device: 设备
        smooth_alpha: SmoothQuant alpha
        weight_clip_ratio: Weight 离群值裁剪比例
    """
    ptq_params = getattr(step_wrapper, "_ptq_params")
    _ = prefill_wrapper  # prefill 也参与校准（挂载了 observer）

    for sample in calib_samples:
        inp_embeds = sample["input"]["inputs_embeds"]
        attn_mask = sample["input"]["attention_mask"]
        inp_embeds_t = torch.from_numpy(inp_embeds).to(device)
        attn_mask_t = torch.from_numpy(attn_mask).to(device)

        # Prefill 前向（收集激活分布）
        with torch.no_grad():
            _, _, past_k, past_v, prompt_len = prefill_wrapper(
                inp_embeds_t, attn_mask_t,
            )

        cache_base = int(prompt_len[0].item())
        seq = sample.get("sequence", [])

        # Decode 前向（收集激活分布）
        for step_idx, token_id in enumerate(seq):
            token_t = torch.tensor([[token_id]], device=device, dtype=torch.long)
            step_embed = step_wrapper.talker.get_input_embeddings()(token_t)
            cache_pos = torch.tensor([cache_base + step_idx], dtype=torch.int64, device=device)
            position_ids = torch.full((3, 1, 1), cache_base + step_idx, dtype=torch.int64, device=device)
            with torch.no_grad():
                step_wrapper(step_embed, past_k, past_v, position_ids, cache_pos)

    # 计算量化参数
    _compute_quant_params(ptq_params, smooth_alpha=smooth_alpha, weight_clip_ratio=weight_clip_ratio)
    # 保存 PTQ 参数供精度对比使用
    ptq_save_path = os.environ.get("PTQ_PARAMS_SAVE_PATH", "")
    if ptq_save_path:
        _save_ptq_params(ptq_params, str(ptq_save_path))
    _cleanup_observers(prefill_wrapper)
    _cleanup_observers(step_wrapper)


def _load_qwen3_tts_model(model_path: str, dtype: torch.dtype):
    """Load the Qwen3-TTS model via transformers auto classes."""
    _ensure_torchaudio_stub()
    transformers = _require_module("transformers")
    auto_config = getattr(transformers, "AutoConfig")
    auto_model = getattr(transformers, "AutoModel")
    cfg_mod = _require_module("qwen_tts.core.models.configuration_qwen3_tts")
    model_mod = _require_module("qwen_tts.core.models.modeling_qwen3_tts")
    qwen3_tts_config = getattr(cfg_mod, "Qwen3TTSConfig")
    qwen3_tts_model = getattr(model_mod, "Qwen3TTSForConditionalGeneration")
    auto_config.register("qwen3_tts", qwen3_tts_config)
    auto_model.register(qwen3_tts_config, qwen3_tts_model)
    return auto_model.from_pretrained(model_path, dtype=dtype).eval()


def _rewrite_talker_prefill_kv_to_fixed_len(src_onnx: str, fixed_len: int = 512) -> None:
    """Rewrite prefill KV cache tensors to a fixed cache length."""
    fixed_len = int(fixed_len)
    model = onnx.load(src_onnx, load_external_data=True)
    graph = model.graph

    def _make_i64(name: str, values, dims):
        return onnx_helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.INT64,
            dims=dims,
            vals=values,
        )

    def _make_f32(name: str, value: float):
        return onnx_helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=[],
            vals=[float(value)],
        )

    def _rename_output_tensor(old: str, new: str):
        found = False
        for node in graph.node:
            for idx, out_name in enumerate(node.output):
                if out_name == old:
                    node.output[idx] = new
                    found = True
        return found

    def _fix_one(kv_name: str):
        raw = f"{kv_name}__raw"
        if not _rename_output_tensor(kv_name, raw):
            raise RuntimeError(f"Cannot find producer output for {kv_name}")

        starts_name = f"{kv_name}__starts"
        ends_name = f"{kv_name}__ends"
        axes_name = f"{kv_name}__axes"
        graph.initializer.extend(
            [
                _make_i64(starts_name, [0], [1]),
                _make_i64(ends_name, [fixed_len], [1]),
                _make_i64(axes_name, [3], [1]),
            ]
        )

        slice_out = f"{kv_name}__slice"
        graph.node.append(
            onnx_helper.make_node(
                "Slice",
                inputs=[raw, starts_name, ends_name, axes_name],
                outputs=[slice_out],
                name=f"{kv_name}__Slice",
            )
        )

        shape_out = f"{kv_name}__shape"
        graph.node.append(
            onnx_helper.make_node(
                "Shape",
                inputs=[slice_out],
                outputs=[shape_out],
                name=f"{kv_name}__Shape",
            )
        )

        axis3_name = f"{kv_name}__axis3"
        graph.initializer.append(_make_i64(axis3_name, [3], []))
        gather_out = f"{kv_name}__seq"
        graph.node.append(
            onnx_helper.make_node(
                "Gather",
                inputs=[shape_out, axis3_name],
                outputs=[gather_out],
                name=f"{kv_name}__GatherSeq",
                axis=0,
            )
        )

        fixed_name = f"{kv_name}__fixed"
        graph.initializer.append(_make_i64(fixed_name, [fixed_len], []))
        sub_out = f"{kv_name}__padlen"
        graph.node.append(
            onnx_helper.make_node(
                "Sub",
                inputs=[fixed_name, gather_out],
                outputs=[sub_out],
                name=f"{kv_name}__Sub",
            )
        )

        unsq_out = f"{kv_name}__padlen_unsq"
        unsq_axes = f"{kv_name}__unsq_axes"
        graph.initializer.append(_make_i64(unsq_axes, [0], [1]))
        graph.node.append(
            onnx_helper.make_node(
                "Unsqueeze",
                inputs=[sub_out, unsq_axes],
                outputs=[unsq_out],
                name=f"{kv_name}__Unsqueeze",
            )
        )

        pads_begin = f"{kv_name}__pads_begin"
        pads_end_prefix = f"{kv_name}__pads_end_prefix"
        pads_end_suffix = f"{kv_name}__pads_end_suffix"
        graph.initializer.extend(
            [
                _make_i64(pads_begin, [0, 0, 0, 0, 0], [5]),
                _make_i64(pads_end_prefix, [0, 0, 0], [3]),
                _make_i64(pads_end_suffix, [0], [1]),
            ]
        )

        pads_out = f"{kv_name}__pads"
        graph.node.append(
            onnx_helper.make_node(
                "Concat",
                inputs=[pads_begin, pads_end_prefix, unsq_out, pads_end_suffix],
                outputs=[pads_out],
                name=f"{kv_name}__ConcatPads",
                axis=0,
            )
        )

        zero_f32 = f"{kv_name}__zero_f32"
        graph.initializer.append(_make_f32(zero_f32, 0.0))
        graph.node.append(
            onnx_helper.make_node(
                "Pad",
                inputs=[slice_out, pads_out, zero_f32],
                outputs=[kv_name],
                name=f"{kv_name}__Pad",
                mode="constant",
            )
        )

    _fix_one("past_k")
    _fix_one("past_v")

    for out in graph.output:
        if out.name == "past_k" and len(out.type.tensor_type.shape.dim) >= 4:
            out.type.tensor_type.shape.dim[3].dim_value = fixed_len
        if out.name == "past_v" and len(out.type.tensor_type.shape.dim) >= 4:
            out.type.tensor_type.shape.dim[3].dim_value = fixed_len

    data_name = os.path.splitext(os.path.basename(src_onnx))[0] + ".data"
    dst_dir = os.path.dirname(os.path.abspath(src_onnx))
    os.makedirs(dst_dir, exist_ok=True)
    dst_data = os.path.join(dst_dir, data_name)
    if os.path.exists(src_onnx):
        os.remove(src_onnx)
    if os.path.exists(dst_data):
        os.remove(dst_data)

    onnx.save_model(
        model,
        src_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )


def _causal_mask_2d(
    attn_2d: torch.Tensor,
    window_size: int | None,
    mask_value: float = -1e4,
) -> torch.Tensor:
    """Build a causal attention mask from a 2D padding mask."""
    b, s = attn_2d.shape
    device = attn_2d.device
    i = torch.arange(s, device=device).view(1, 1, s, 1)
    j = torch.arange(s, device=device).view(1, 1, 1, s)
    allowed = j <= i
    if window_size is not None and int(window_size) > 0:
        allowed = allowed & (j > (i - int(window_size)))
    key_ok = attn_2d.view(b, 1, 1, s).to(torch.bool)
    allowed = allowed & key_ok
    zero = torch.zeros((b, 1, s, s), device=device, dtype=torch.float32)
    neg = torch.full((b, 1, s, s), float(mask_value), device=device, dtype=torch.float32)
    return torch.where(allowed, zero, neg)

def _repeat_kv(x: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return x
    return x.repeat_interleave(num_key_value_groups, dim=1)


class TalkerPrefillKVWrapper(_NNModule):
    """Talker prefill wrapper exporting logits/hidden and KV cache."""

    def __init__(
        self,
        talker,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        mask_value: float = -1e4,
        use_ascend_fused_ops: bool = False,
        use_custom_rope: bool = False,
        enable_ptq: bool = False,
        ptq_skip_layers: str = "",
    ):
        super().__init__()
        self.talker = talker
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = int(num_attention_heads // num_key_value_heads)
        self.scaling = float(head_dim) ** -0.5
        self.mask_value = float(mask_value)
        self.use_ascend_fused_ops = bool(use_ascend_fused_ops)
        self.allow_custom_rope = bool(use_custom_rope)
        self._enable_ptq = bool(enable_ptq)
        self._ptq_params: dict[str, Any] = {}
        self._ptq_skip_set = set(s.strip() for s in str(ptq_skip_layers).split(",") if s.strip())
        self._ptq_skip_layers: set[int] = set()
        for s in self._ptq_skip_set:
            if s.startswith("layer."):
                try:
                    self._ptq_skip_layers.add(int(s.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        try:
            setattr(self.talker.config, "_attn_implementation", "eager")
            setattr(self.talker.model.config, "_attn_implementation", "eager")
        except (AttributeError, TypeError):
            pass
        self._mlp_gateup_w: list[torch.Tensor] = []
        self._mlp_gateup_b: list[torch.Tensor | None] = []
        layers = list(self.talker.model.layers)
        for li, layer in enumerate(layers):
            mlp = layer.mlp
            w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).detach()
            self.register_buffer(f"_mlp_gateup_w_{li}", w)
            self._mlp_gateup_w.append(getattr(self, f"_mlp_gateup_w_{li}"))
            bg = mlp.gate_proj.bias
            if bg is None:
                self._mlp_gateup_b.append(None)
            else:
                b = torch.cat([bg, mlp.up_proj.bias], dim=0).detach()
                self.register_buffer(f"_mlp_gateup_b_{li}", b)
                self._mlp_gateup_b.append(getattr(self, f"_mlp_gateup_b_{li}"))

        # PTQ: 存储 weight 引用、挂载 observer
        if self._enable_ptq:
            for li, layer in enumerate(layers):
                skip_li = li in self._ptq_skip_layers
                attn = layer.self_attn
                qkv_w = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0).detach()
                if not skip_li and "qkv" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.qkv", {})["weight_fp"] = qkv_w
                if not skip_li and "o_proj" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.o_proj", {})["weight_fp"] = attn.o_proj.weight.detach()
                gu_w = self._mlp_gateup_w[li]
                if not skip_li and "gate_up" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.gate_up", {})["weight_fp"] = gu_w
                if not skip_li and "down_proj" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.down_proj", {})[
                        "weight_fp"
                    ] = layer.mlp.down_proj.weight.detach()
                # 挂载 observer（校准用）
                # QKV: hook on input_layernorm (fused F.linear bypasses q_proj module)
                if not skip_li and "qkv" not in self._ptq_skip_set:
                    _attach_observers(layer.input_layernorm, f"layer.{li}.qkv", self._ptq_params)
                if not skip_li and "o_proj" not in self._ptq_skip_set:
                    _attach_observers(attn.o_proj, f"layer.{li}.o_proj", self._ptq_params)
                if not skip_li and "gate_up" not in self._ptq_skip_set:
                    # 挂载在 post_attention_layernorm 而非 layer.mlp，
                    # 因为校准时 _mlp 直接调 F.linear 不会触发 layer.mlp 的 hook
                    _attach_observers(layer.post_attention_layernorm, f"layer.{li}.gate_up", self._ptq_params)
                if not skip_li and "down_proj" not in self._ptq_skip_set:
                    _attach_observers(layer.mlp.down_proj, f"layer.{li}.down_proj", self._ptq_params)
            # codec_head
            if "codec_head" not in self._ptq_skip_set:
                self._ptq_params.setdefault("codec_head", {})["weight_fp"] = self.talker.codec_head.weight.detach()
                _attach_observers(self.talker.codec_head, "codec_head", self._ptq_params)

    def _attention_prefill(
        self,
        attn_mod,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask_4d: torch.Tensor,
        layer_idx: int = 0,
    ):
        """Compute attention for the prefill graph and return (out, k, v)."""
        b, s, _ = hidden_states.shape

        # ── QKV projection (可选 PTQ int8 量化路径) ──
        qkv_name = f"layer.{layer_idx}.qkv"
        qkv_params = self._ptq_params.get(qkv_name, {}) if self._enable_ptq else {}
        if qkv_params and "w_q" in qkv_params:
            qkv = quant_linear_symmetric_int8(
                hidden_states, qkv_params["weight_fp"], None,
                qkv_params.get("x_scale_per_ch", qkv_params["x_scale"]), qkv_params["w_q"], qkv_params["w_scale"],
                smooth_scale=qkv_params.get("smooth_scale"),
            )
        else:
            qkv_w = torch.cat([attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight], dim=0)
            qkv = torch.nn.functional.linear(hidden_states, qkv_w)

        q_out = int(self.num_attention_heads * self.head_dim)
        kv_out = int(self.num_key_value_heads * self.head_dim)
        q_raw, k_raw, v_raw = torch.split(qkv, [q_out, kv_out, kv_out], dim=-1)
        q = q_raw.view(b, s, self.num_attention_heads, self.head_dim)
        q = self._rms_norm_custom(attn_mod.q_norm, q).transpose(1, 2)
        k = k_raw.view(b, s, self.num_key_value_heads, self.head_dim)
        k = self._rms_norm_custom(attn_mod.k_norm, k).transpose(1, 2)
        v = v_raw.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.transpose(1, 2)

        if self.allow_custom_rope:
            q = CustomRoatryMul.apply(q, cos, sin)
            k = CustomRoatryMul.apply(k, cos, sin)
        else:
            q = _rotary_mul_plain(q, cos, sin)
            k = _rotary_mul_plain(k, cos, sin)

        k_for_attn = _repeat_kv(k, self.num_key_value_groups)
        v_for_attn = _repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q, k_for_attn.transpose(-2, -1))
        scores = scores * self.scaling
        scores = scores + attention_mask_4d
        probs = torch.softmax(scores, dim=-1).to(v_for_attn.dtype)
        out = torch.matmul(probs, v_for_attn)
        out = out.transpose(1, 2).contiguous().reshape(b, s, -1)

        # ── O projection (可选 PTQ int8 量化路径) ──
        o_name = f"layer.{layer_idx}.o_proj"
        o_params = self._ptq_params.get(o_name, {}) if self._enable_ptq else {}
        if o_params and "w_q" in o_params:
            out = quant_linear_symmetric_int8(
                out, o_params["weight_fp"], None,
                o_params.get("x_scale_per_ch", o_params["x_scale"]), o_params["w_q"], o_params["w_scale"],
                smooth_scale=o_params.get("smooth_scale"),
            )
        else:
            out = attn_mod.o_proj(out)
        return out, k, v

    def _rms_norm_custom(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        return norm_mod(x)

    def _mlp(
        self,
        mlp_mod,
        x: torch.Tensor,
        gateup_w: torch.Tensor | None = None,
        gateup_b: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute MLP, optionally using a fused SwiGLU path or PTQ int8 quant path."""
        if not self.use_ascend_fused_ops:
            return mlp_mod(x)

        # ── Gate/Up projection (可选 PTQ int8 量化路径) ──
        gu_name = f"layer.{layer_idx}.gate_up"
        gu_params = self._ptq_params.get(gu_name, {}) if self._enable_ptq else {}
        if gu_params and "w_q" in gu_params:
            gate_up = quant_linear_symmetric_int8(
                x, gu_params["weight_fp"], None,
                gu_params.get("x_scale_per_ch", gu_params["x_scale"]), gu_params["w_q"], gu_params["w_scale"],
                smooth_scale=gu_params.get("smooth_scale"),
            )
        else:
            w = (
                gateup_w
                if gateup_w is not None
                else torch.cat([mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight], dim=0)
            )
            bias = gateup_b
            if bias is None and gateup_w is None:
                bg = mlp_mod.gate_proj.bias
                if bg is not None:
                    bias = torch.cat([bg, mlp_mod.up_proj.bias], dim=0)
            gate_up = torch.nn.functional.linear(x, w, bias)
        y = CustomSwiGlu.apply(gate_up, -1)

        # ── Down projection (可选 PTQ int8 量化路径) ──
        dp_name = f"layer.{layer_idx}.down_proj"
        dp_params = self._ptq_params.get(dp_name, {}) if self._enable_ptq else {}
        if dp_params and "w_q" in dp_params:
            y = quant_linear_symmetric_int8(
                y, dp_params["weight_fp"], None,
                dp_params.get("x_scale_per_ch", dp_params["x_scale"]), dp_params["w_q"], dp_params["w_scale"],
                smooth_scale=dp_params.get("smooth_scale"),
            )
        else:
            y = mlp_mod.down_proj(y)
        return y

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Run talker prefill and return (logits_last, hidden_last, past_k, past_v, prompt_len)."""
        b, s, _ = inputs_embeds.shape
        device = inputs_embeds.device
        cache_position = torch.arange(s, device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, b, -1)

        position_embeddings = self.talker.model.rotary_emb(inputs_embeds, position_ids)
        cos_raw, sin_raw = position_embeddings
        attn0 = self.talker.model.layers[0].self_attn
        cos, sin = _prepare_mrope_cos_sin(
            cos_raw,
            sin_raw,
            attn0.rope_scaling["mrope_section"],
            mrope_interleaved=bool(attn0.rope_scaling["interleaved"]),
            unsqueeze_dim=1,
        )
        window = getattr(self.talker.model.config, "sliding_window", None)
        mask_value = float(torch.finfo(torch.float32).min)
        attn4d = _causal_mask_2d(attention_mask.to(torch.int64), window_size=window, mask_value=mask_value)

        layers = list(self.talker.model.layers)

        residual = inputs_embeds
        hidden_states = residual
        k_layers = []
        v_layers = []
        for li, layer in enumerate(layers):
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            attn_out, k, v = self._attention_prefill(layer.self_attn, hidden_states, cos, sin, attn4d, layer_idx=li)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_out = self._mlp(layer.mlp, hidden_states, self._mlp_gateup_w[li], self._mlp_gateup_b[li], layer_idx=li)
            hidden_states = residual + mlp_out
            residual = hidden_states

            k_layers.append(k)
            v_layers.append(v)

        hidden_states = self.talker.model.norm(hidden_states)
        # ── Codec head (可选 PTQ int8 量化路径) ──
        ch_params = self._ptq_params.get("codec_head", {}) if self._enable_ptq else {}
        if ch_params and "w_q" in ch_params:
            logits = quant_linear_symmetric_int8(
                hidden_states, ch_params["weight_fp"], None,
                ch_params.get("x_scale_per_ch", ch_params["x_scale"]), ch_params["w_q"], ch_params["w_scale"],
                smooth_scale=ch_params.get("smooth_scale"),
            )
        else:
            logits = self.talker.codec_head(hidden_states)

        prompt_len = attention_mask.to(torch.int64).sum(dim=1)
        one_i = prompt_len.new_tensor(1)
        prompt_len = torch.maximum(prompt_len, one_i)
        idx = (prompt_len - 1).view(b, 1, 1)
        logits_last = logits.gather(1, idx.expand(b, 1, logits.shape[-1]))[:, 0, :]
        hidden_last = hidden_states.gather(1, idx.expand(b, 1, hidden_states.shape[-1]))

        past_k = torch.stack(k_layers, dim=0)
        past_v = torch.stack(v_layers, dim=0)
        return logits_last, hidden_last, past_k, past_v, prompt_len


class TalkerStepKVWrapper(_NNModule):
    """Talker step wrapper exporting logits/hidden and updated KV cache."""

    def __init__(
        self,
        talker,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        cache_len: int = 512,
        mask_value: float = -1e4,
        use_ascend_fused_ops: bool = False,
        use_custom_rope: bool = False,
        enable_ptq: bool = False,
        ptq_skip_layers: str = "",
    ):
        super().__init__()
        self.talker = talker
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = int(num_attention_heads // num_key_value_heads)
        self.scaling = float(head_dim) ** -0.5
        self.cache_len = int(cache_len)
        self.mask_value = float(mask_value)
        self.use_ascend_fused_ops = bool(use_ascend_fused_ops)
        self.allow_custom_rope = bool(use_custom_rope)
        self._enable_ptq = bool(enable_ptq)
        self._ptq_params: dict[str, Any] = {}
        self._ptq_skip_set = set(s.strip() for s in str(ptq_skip_layers).split(",") if s.strip())
        self._ptq_skip_layers: set[int] = set()
        for s in self._ptq_skip_set:
            if s.startswith("layer."):
                try:
                    self._ptq_skip_layers.add(int(s.split(".")[1]))
                except (IndexError, ValueError):
                    pass
        try:
            setattr(self.talker.config, "_attn_implementation", "eager")
            setattr(self.talker.model.config, "_attn_implementation", "eager")
        except (AttributeError, TypeError):
            pass
        self._qkv_w: list[torch.Tensor] = []
        self._qkv_b: list[torch.Tensor | None] = []
        self._mlp_gateup_w: list[torch.Tensor] = []
        self._mlp_gateup_b: list[torch.Tensor | None] = []
        layers = list(self.talker.model.layers)
        for li, layer in enumerate(layers):
            attn = layer.self_attn
            qkv_w = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0).detach()
            self.register_buffer(f"_qkv_w_{li}", qkv_w)
            self._qkv_w.append(getattr(self, f"_qkv_w_{li}"))
            bq = attn.q_proj.bias
            if bq is None:
                self._qkv_b.append(None)
            else:
                qkv_b = torch.cat([bq, attn.k_proj.bias, attn.v_proj.bias], dim=0).detach()
                self.register_buffer(f"_qkv_b_{li}", qkv_b)
                self._qkv_b.append(getattr(self, f"_qkv_b_{li}"))
            mlp = layer.mlp
            w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).detach()
            self.register_buffer(f"_mlp_gateup_w_{li}", w)
            self._mlp_gateup_w.append(getattr(self, f"_mlp_gateup_w_{li}"))
            bg = mlp.gate_proj.bias
            if bg is None:
                self._mlp_gateup_b.append(None)
            else:
                b = torch.cat([bg, mlp.up_proj.bias], dim=0).detach()
                self.register_buffer(f"_mlp_gateup_b_{li}", b)
                self._mlp_gateup_b.append(getattr(self, f"_mlp_gateup_b_{li}"))
        # PTQ: 存储 weight 引用、挂载 observer
        if self._enable_ptq:
            for li, layer in enumerate(layers):
                skip_li = li in self._ptq_skip_layers
                attn = layer.self_attn
                if not skip_li and "qkv" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.qkv", {})["weight_fp"] = self._qkv_w[li]
                if not skip_li and "o_proj" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.o_proj", {})["weight_fp"] = attn.o_proj.weight.detach()
                if not skip_li and "gate_up" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.gate_up", {})["weight_fp"] = self._mlp_gateup_w[li]
                if not skip_li and "down_proj" not in self._ptq_skip_set:
                    self._ptq_params.setdefault(f"layer.{li}.down_proj", {})[
                        "weight_fp"
                    ] = layer.mlp.down_proj.weight.detach()
                # QKV: hook on input_layernorm (fused F.linear bypasses q_proj module)
                if not skip_li and "qkv" not in self._ptq_skip_set:
                    _attach_observers(layer.input_layernorm, f"layer.{li}.qkv", self._ptq_params)
                if not skip_li and "o_proj" not in self._ptq_skip_set:
                    _attach_observers(attn.o_proj, f"layer.{li}.o_proj", self._ptq_params)
                if not skip_li and "gate_up" not in self._ptq_skip_set:
                    # 挂载在 post_attention_layernorm 而非 layer.mlp
                    _attach_observers(layer.post_attention_layernorm, f"layer.{li}.gate_up", self._ptq_params)
                if not skip_li and "down_proj" not in self._ptq_skip_set:
                    _attach_observers(layer.mlp.down_proj, f"layer.{li}.down_proj", self._ptq_params)
            if "codec_head" not in self._ptq_skip_set:
                self._ptq_params.setdefault("codec_head", {})["weight_fp"] = self.talker.codec_head.weight.detach()
                _attach_observers(self.talker.codec_head, "codec_head", self._ptq_params)

    def _rms_norm(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        return norm_mod(x)

    def _rms_norm_custom(self, norm_mod, x: torch.Tensor) -> torch.Tensor:
        return norm_mod(x)

    def _mlp(
        self,
        mlp_mod,
        x: torch.Tensor,
        gateup_w: torch.Tensor | None = None,
        gateup_b: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute MLP, optionally using a fused SwiGLU or PTQ int8 quant path."""
        if not self.use_ascend_fused_ops:
            return mlp_mod(x)

        # ── Gate/Up projection (可选 PTQ int8 量化路径) ──
        gu_name = f"layer.{layer_idx}.gate_up"
        gu_params = self._ptq_params.get(gu_name, {}) if self._enable_ptq else {}
        if gu_params and "w_q" in gu_params:
            gate_up = quant_linear_symmetric_int8(
                x, gu_params["weight_fp"], None,
                gu_params.get("x_scale_per_ch", gu_params["x_scale"]), gu_params["w_q"], gu_params["w_scale"],
                smooth_scale=gu_params.get("smooth_scale"),
            )
        else:
            w = (
                gateup_w
                if gateup_w is not None
                else torch.cat([mlp_mod.gate_proj.weight, mlp_mod.up_proj.weight], dim=0)
            )
            bias = gateup_b
            if bias is None and gateup_w is None:
                bg = mlp_mod.gate_proj.bias
                if bg is not None:
                    bias = torch.cat([bg, mlp_mod.up_proj.bias], dim=0)
            gate_up = torch.nn.functional.linear(x, w, bias)
        y = CustomSwiGlu.apply(gate_up, -1)

        # ── Down projection (可选 PTQ int8 量化路径) ──
        dp_name = f"layer.{layer_idx}.down_proj"
        dp_params = self._ptq_params.get(dp_name, {}) if self._enable_ptq else {}
        if dp_params and "w_q" in dp_params:
            y = quant_linear_symmetric_int8(
                y, dp_params["weight_fp"], None,
                dp_params.get("x_scale_per_ch", dp_params["x_scale"]), dp_params["w_q"], dp_params["w_scale"],
                smooth_scale=dp_params.get("smooth_scale"),
            )
        else:
            y = mlp_mod.down_proj(y)
        return y

    def _attention_step(
        self,
        attn_mod,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_pad_4d: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        cache_pos: torch.Tensor,
        qkv_w: torch.Tensor | None = None,
        qkv_b: torch.Tensor | None = None,
        layer_idx: int = 0,
    ):
        """Compute attention for one step and update KV cache."""
        q_out = int(self.num_attention_heads * self.head_dim)
        kv_out = int(self.num_key_value_heads * self.head_dim)

        # ── QKV projection (可选 PTQ int8 量化路径) ──
        qkv_name = f"layer.{layer_idx}.qkv"
        qkv_params = self._ptq_params.get(qkv_name, {}) if self._enable_ptq else {}
        if qkv_params and "w_q" in qkv_params:
            qkv = quant_linear_symmetric_int8(
                hidden_states, qkv_params["weight_fp"], None,
                qkv_params.get("x_scale_per_ch", qkv_params["x_scale"]), qkv_params["w_q"], qkv_params["w_scale"],
                smooth_scale=qkv_params.get("smooth_scale"),
            )
        else:
            w = (
                qkv_w
                if qkv_w is not None
                else torch.cat(
                    [attn_mod.q_proj.weight, attn_mod.k_proj.weight, attn_mod.v_proj.weight],
                    dim=0,
                )
            )
            bias = qkv_b
            if bias is None and qkv_w is None:
                bq = attn_mod.q_proj.bias
                if bq is not None:
                    bias = torch.cat([bq, attn_mod.k_proj.bias, attn_mod.v_proj.bias], dim=0)
            qkv = torch.nn.functional.linear(hidden_states, w, bias)
        q_raw, k_raw, v_raw = torch.split(qkv, [q_out, kv_out, kv_out], dim=-1)
        q = self._rms_norm_custom(attn_mod.q_norm, q_raw.reshape(-1, self.num_attention_heads, 1, self.head_dim))
        k = self._rms_norm_custom(attn_mod.k_norm, k_raw.reshape(-1, self.num_key_value_heads, 1, self.head_dim))
        v = v_raw.reshape(-1, self.num_key_value_heads, 1, self.head_dim)

        if self.use_ascend_fused_ops:
            q = CustomRoatryMul.apply(q, cos, sin)
            k = CustomRoatryMul.apply(k, cos, sin)
        else:
            q = _rotary_mul_plain(q, cos, sin)
            k = _rotary_mul_plain(k, cos, sin)

        k_full = _kv_cache_update(past_k, k, cache_pos, use_custom=bool(self.use_ascend_fused_ops))
        v_full = _kv_cache_update(past_v, v, cache_pos, use_custom=bool(self.use_ascend_fused_ops))

        if self.use_ascend_fused_ops:
            q_bnsd = q.contiguous()
            k_bnsd = k_full.contiguous()
            v_bnsd = v_full.contiguous()
            if key_pad_4d is not None:
                attn_mask = key_pad_4d != 0
            else:
                bsz = int(k_bnsd.shape[0])
                cache_total = int(k_bnsd.shape[2])
                attn_mask = torch.zeros(
                    (bsz, 1, 1, cache_total),
                    device=hidden_states.device,
                    dtype=torch.bool,
                )
            out = incre_flash_attention(
                    q_bnsd, k_bnsd, v_bnsd, attn_mask,
                    num_heads=self.num_attention_heads, scale_value=float(self.scaling),
                    input_layout="BNSD", num_key_value_heads=int(self.num_key_value_heads), inner_precise=1,
                )
            if int(out.dim()) == 4:
                out = out.transpose(1, 2).reshape(-1, 1, self.num_attention_heads * self.head_dim)
            out = attn_mod.o_proj(out)
            return out, k_full, v_full

        if int(self.num_key_value_groups) > 1:
            k_full_rep = k_full.repeat_interleave(int(self.num_key_value_groups), dim=1)
            v_full_rep = v_full.repeat_interleave(int(self.num_key_value_groups), dim=1)
        else:
            k_full_rep = k_full
            v_full_rep = v_full

        scores = torch.matmul(q, k_full_rep.transpose(-2, -1))
        scores = scores * self.scaling
        if key_pad_4d is not None:
            scores = scores + key_pad_4d
        probs = torch.softmax(scores, dim=-1).to(v_full_rep.dtype)
        out = torch.matmul(probs, v_full_rep)
        out = out.transpose(1, 2).reshape(-1, 1, self.num_attention_heads * self.head_dim)

        # ── O projection (可选 PTQ int8 量化路径) ──
        o_name = f"layer.{layer_idx}.o_proj"
        o_params = self._ptq_params.get(o_name, {}) if self._enable_ptq else {}
        if o_params and "w_q" in o_params:
            out = quant_linear_symmetric_int8(
                out, o_params["weight_fp"], None,
                o_params.get("x_scale_per_ch", o_params["x_scale"]), o_params["w_q"], o_params["w_scale"],
                smooth_scale=o_params.get("smooth_scale"),
            )
        else:
            out = attn_mod.o_proj(out)
        return out, k_full, v_full

    def forward(
        self,
        step_embed: torch.Tensor,
        past_k: torch.Tensor,
        past_v: torch.Tensor,
        position_ids_step: torch.Tensor,
        cache_len: torch.Tensor,
    ):
        """Run one talker step and return logits/hidden plus updated KV cache."""
        position_embeddings = self.talker.model.rotary_emb(step_embed, position_ids_step)
        cos_raw, sin_raw = position_embeddings
        attn0 = self.talker.model.layers[0].self_attn
        cos, sin = _prepare_mrope_cos_sin(
            cos_raw,
            sin_raw,
            attn0.rope_scaling["mrope_section"],
            mrope_interleaved=bool(attn0.rope_scaling["interleaved"]),
            unsqueeze_dim=1,
        )
        mask_value = float(torch.finfo(torch.float32).min)
        window = getattr(self.talker.model.config, "sliding_window", None)
        cache_total = int(past_k.size(3))
        cache_pos = torch.clamp(cache_len.to(torch.int64), min=0, max=cache_total - 1)
        kv_idx = torch.arange(cache_total, device=step_embed.device, dtype=torch.int64).view(1, cache_total)
        allow = kv_idx <= cache_pos.view(-1, 1)
        if window is not None and int(window) > 0:
            start = (cache_pos - int(window) + 1).view(-1, 1)
            allow = allow & (kv_idx >= start)
        key_pad_4d = torch.full(
            (allow.size(0), 1, 1, cache_total),
            mask_value,
            device=step_embed.device,
            dtype=torch.float32,
        )
        key_pad_4d = key_pad_4d.masked_fill(
            allow.view(allow.size(0), 1, 1, cache_total),
            0.0,
        ).to(step_embed.dtype)

        layers = list(self.talker.model.layers)
        residual = step_embed
        hidden_states = residual
        past_k_layers = torch.unbind(past_k, dim=0)
        past_v_layers = torch.unbind(past_v, dim=0)
        k_layers = []
        v_layers = []
        for li, layer in enumerate(layers):
            residual = hidden_states
            hidden_states = self._rms_norm(layer.input_layernorm, hidden_states)
            attn_out, k_new, v_new = self._attention_step(
                layer.self_attn,
                hidden_states,
                cos,
                sin,
                key_pad_4d,
                past_k_layers[li],
                past_v_layers[li],
                cache_pos=cache_pos,
                qkv_w=self._qkv_w[li],
                qkv_b=self._qkv_b[li],
                layer_idx=li,
            )
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self._rms_norm(layer.post_attention_layernorm, hidden_states)
            mlp_out = self._mlp(layer.mlp, hidden_states, self._mlp_gateup_w[li], self._mlp_gateup_b[li], layer_idx=li)
            hidden_states = residual + mlp_out
            residual = hidden_states

            k_layers.append(k_new)
            v_layers.append(v_new)

        hidden_states = self._rms_norm(self.talker.model.norm, residual)
        # ── Codec head (可选 PTQ int8 量化路径) ──
        ch_params = self._ptq_params.get("codec_head", {}) if self._enable_ptq else {}
        if ch_params and "w_q" in ch_params:
            logits = quant_linear_symmetric_int8(
                hidden_states, ch_params["weight_fp"], None,
                ch_params.get("x_scale_per_ch", ch_params["x_scale"]), ch_params["w_q"], ch_params["w_scale"],
                smooth_scale=ch_params.get("smooth_scale"),
            )
        else:
            logits = self.talker.codec_head(hidden_states)
        logits_last = logits[:, 0, :]
        past_k_out = torch.stack(k_layers, dim=0)
        past_v_out = torch.stack(v_layers, dim=0)
        return logits_last, hidden_states, past_k_out, past_v_out


def export_talker_kv(
    model_path: str,
    output_dir: str,
    opset: int,
    dtype: str,
    export_seq_len: int,
    device: str = "cpu",
    ascend_fused_ops: bool = False,
    custom_rope: bool = False,
    enable_ptq: bool = False,
    ptq_calib_data: str = "",
    ptq_skip_layers: str = "",
    smooth_alpha: float = 0.65,
    weight_clip_ratio: float = 0.01,
) -> None:
    """Export talker prefill/step ONNX models.

    When ``enable_ptq=True``, loads calibration data from ``ptq_calib_data``,
    runs PTQ int8 calibration, and exports ONNX with quantized MatMul
    (AscendQuant + QuantBatchMatmul custom ops in the graph).
    """
    dt = _torch_dtype(dtype)
    os.makedirs(output_dir, exist_ok=True)
    prefill_dir = os.path.join(output_dir, "prefill")
    step_dir = os.path.join(output_dir, "step")
    os.makedirs(prefill_dir, exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)
    model = _load_qwen3_tts_model(model_path, dtype=dt).to(device)
    talker = model.talker
    info = _collect_talker_kv_export_info(model, talker)

    enable_ptq = bool(enable_ptq)
    ptq_wrapper = None
    prefill_wrap_calib = None

    if enable_ptq:
        if not ptq_calib_data or not os.path.isfile(str(ptq_calib_data)):
            print(f"  [PTQ] Calibration data not found: {ptq_calib_data!r}, skipping PTQ.")
            enable_ptq = False
        else:
            print(f"  [PTQ] Loading calibration data from {ptq_calib_data} ...")
            calib_samples = _load_calib_data_jsonl(ptq_calib_data, max_samples=4, max_decode_steps=32)
            print(f"  [PTQ] Loaded {len(calib_samples)} calibration samples.")

            # 创建 step wrapper 用于校准
            ptq_wrapper = TalkerStepKVWrapper(
                talker=talker,
                num_attention_heads=int(info["num_attention_heads"]),
                num_key_value_heads=int(info["num_key_value_heads"]),
                head_dim=int(info["head_dim"]),
                cache_len=512,
                use_ascend_fused_ops=bool(ascend_fused_ops),
                use_custom_rope=bool(custom_rope),
                enable_ptq=True,
                ptq_skip_layers=ptq_skip_layers,
            ).eval().to(device)

            prefill_wrap_calib = TalkerPrefillKVWrapper(
                talker=talker,
                num_attention_heads=int(info["num_attention_heads"]),
                num_key_value_heads=int(info["num_key_value_heads"]),
                head_dim=int(info["head_dim"]),
                use_ascend_fused_ops=bool(ascend_fused_ops),
                use_custom_rope=bool(custom_rope),
                enable_ptq=True,
                ptq_skip_layers=ptq_skip_layers,
            ).eval().to(device)

            print(f"  [PTQ] Running calibration (smooth_alpha={smooth_alpha}) ...")
            _calibrate_talker_wrappers(
                prefill_wrap_calib, ptq_wrapper, calib_samples,
                device=device, smooth_alpha=smooth_alpha,
                weight_clip_ratio=weight_clip_ratio,
            )
            n_quant = sum(
                1 for v in getattr(ptq_wrapper, "_ptq_params").values() if "w_q" in v
            )
            print(f"  [PTQ] Calibration done: {n_quant}/{4 * int(info['num_layers']) + 1} linear layers quantized.")

    _export_talker_prefill_kv_onnx(
        talker=talker,
        output_dir=prefill_dir,
        opset=int(opset),
        dt=dt,
        export_seq_len=int(export_seq_len),
        device=str(device),
        num_attention_heads=int(info["num_attention_heads"]),
        num_key_value_heads=int(info["num_key_value_heads"]),
        head_dim=int(info["head_dim"]),
        ascend_fused_ops=bool(ascend_fused_ops),
        custom_rope=bool(custom_rope),
    )
    _export_talker_step_kv_onnx(
        talker=talker,
        output_dir=step_dir,
        opset=int(opset),
        dt=dt,
        device=str(device),
        ascend_fused_ops=bool(ascend_fused_ops),
        custom_rope=bool(custom_rope),
        ptq_wrapper=ptq_wrapper,
    )


def _collect_talker_kv_export_info(model: Any, talker: Any) -> dict[str, int]:
    num_layers = int(talker.model.config.num_hidden_layers)
    hidden_size = int(talker.config.hidden_size)
    num_attention_heads = int(talker.config.num_attention_heads)
    num_key_value_heads = int(talker.config.num_key_value_heads)
    head_dim = int(getattr(talker.config, "head_dim", hidden_size // num_attention_heads))
    vocab_size = int(talker.config.vocab_size)
    eos_id = int(model.config.talker_config.codec_eos_token_id)
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "eos_id": eos_id,
    }


def _export_talker_prefill_kv_onnx(
    *,
    talker: Any,
    output_dir: str,
    opset: int,
    dt: Any,
    export_seq_len: int,
    device: str,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    ascend_fused_ops: bool,
    custom_rope: bool,
) -> None:
    """Export talker_prefill.onnx and apply post-export graph rewrites."""
    hidden_size = int(talker.config.hidden_size)
    prefill_wrap = TalkerPrefillKVWrapper(
        talker=talker,
        num_attention_heads=int(num_attention_heads),
        num_key_value_heads=int(num_key_value_heads),
        head_dim=int(head_dim),
        use_ascend_fused_ops=bool(ascend_fused_ops),
        use_custom_rope=bool(custom_rope),
    ).eval()
    prefill_wrap = prefill_wrap.to(device)
    ex_inputs_embeds = torch.zeros((1, int(export_seq_len), hidden_size), dtype=dt, device=device)
    ex_attn = torch.zeros((1, int(export_seq_len)), dtype=torch.int64, device=device)
    prefill_onnx = os.path.join(output_dir, "talker_prefill.onnx")
    _export_onnx(
        prefill_wrap,
        (ex_inputs_embeds, ex_attn),
        prefill_onnx,
        opset=int(opset),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits_last", "hidden_last", "past_k", "past_v", "prompt_len"],
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "prompt_len"},
            "attention_mask": {0: "batch", 1: "prompt_len"},
            "hidden_last": {0: "batch"},
            "logits_last": {0: "batch"},
            "past_k": {1: "batch"},
            "past_v": {1: "batch"},
            "prompt_len": {0: "batch"},
        },
        allow_custom_ops=bool(ascend_fused_ops or custom_rope),
    )
    _rewrite_talker_prefill_kv_to_fixed_len(prefill_onnx, fixed_len=512)


def _export_talker_step_kv_onnx(
    *,
    talker: Any,
    output_dir: str,
    opset: int,
    dt: Any,
    device: str,
    ascend_fused_ops: bool,
    custom_rope: bool,
    ptq_wrapper: Any = None,
) -> None:
    """Export talker_step.onnx.

    When ``ptq_wrapper`` is provided (after PTQ calibration), its
    :attr:`_ptq_params` (quantised int8 weights + scales) are reused.
    """
    num_layers = int(talker.model.config.num_hidden_layers)
    hidden_size = int(talker.config.hidden_size)
    num_attention_heads = int(talker.config.num_attention_heads)
    num_key_value_heads = int(talker.config.num_key_value_heads)
    head_dim = int(getattr(talker.config, "head_dim", hidden_size // num_attention_heads))

    if ptq_wrapper is not None:
        # 用校准后的 wrapper（包含量化参数）
        step_wrap = ptq_wrapper
    else:
        step_wrap = TalkerStepKVWrapper(
            talker=talker,
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(head_dim),
            cache_len=512,
            use_ascend_fused_ops=bool(ascend_fused_ops),
            use_custom_rope=bool(custom_rope),
        ).eval()
    step_wrap = step_wrap.to(device)
    ex_step_embed = torch.zeros((1, 1, int(hidden_size)), dtype=dt, device=device)
    ex_past_k = torch.zeros(
        (int(num_layers), 1, int(num_key_value_heads), 512, int(head_dim)),
        dtype=dt,
        device=device,
    )
    ex_past_v = torch.zeros(
        (int(num_layers), 1, int(num_key_value_heads), 512, int(head_dim)),
        dtype=dt,
        device=device,
    )
    ex_pos = torch.zeros((3, 1, 1), dtype=torch.int64, device=device)
    ex_cache_len = torch.zeros((1, 1), dtype=torch.int64, device=device)
    step_onnx = os.path.join(output_dir, "talker_step.onnx")
    _export_onnx(
        step_wrap,
        (ex_step_embed, ex_past_k, ex_past_v, ex_pos, ex_cache_len),
        step_onnx,
        opset=int(opset),
        input_names=["step_embed", "past_k", "past_v", "position_ids_step", "cache_len"],
        output_names=["logits_last", "hidden_last", "past_k_out", "past_v_out"],
        dynamic_axes={
            "step_embed": {0: "batch"},
            "past_k": {1: "batch"},
            "past_v": {1: "batch"},
            "position_ids_step": {1: "batch"},
            "cache_len": {0: "batch"},
            "logits_last": {0: "batch"},
            "hidden_last": {0: "batch"},
            "past_k_out": {1: "batch"},
            "past_v_out": {1: "batch"},
        },
        allow_custom_ops=bool(ascend_fused_ops or custom_rope or ptq_wrapper is not None),
    )


def _count_control_flow_nodes(onnx_path: str) -> dict[str, int]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    counter = Counter(n.op_type for n in model.graph.node)
    return {
        "If": int(counter.get("If", 0)),
        "Loop": int(counter.get("Loop", 0)),
        "Scan": int(counter.get("Scan", 0)),
    }


def _strip_talker_prefill_control_flow(output_dir: str) -> dict[str, Any]:
    """Strip control flow nodes from talker_prefill.onnx when possible."""
    prefill_path = os.path.join(str(output_dir), "talker_prefill.onnx")
    before = _count_control_flow_nodes(prefill_path)
    if max(before.values(), default=0) <= 0:
        return {"before": before, "after": before, "changed": False}

    conv = _require_module("convert_talker_onnx_to_mindir")
    preprocess = getattr(conv, "preprocess_prefill_onnx")
    info = preprocess(prefill_path, prefill_path)
    after = _count_control_flow_nodes(prefill_path)
    return {"before": before, "after": after, "changed": True, "preprocess": info}


def export_talker_kv_onnx(
    model_path: str,
    output_dir: str,
    opset: int = 17,
    dtype: str = "float32",
    export_seq_len: int = 512,
    device: str = "cpu",
    export_custom_ops: bool = False,
    ascend_fused_ops: bool = False,
    strip_control_flow: bool = True,
    enable_ptq: bool = False,
    ptq_calib_data: str = "",
    ptq_skip_layers: str = "",
    smooth_alpha: float = 0.65,
    weight_clip_ratio: float = 0.01,
) -> None:
    """Export talker KV ONNX models.

    When ``enable_ptq=True``, runs PTQ int8 calibration and exports
    ONNX with quantized MatMul (AscendQuant + QuantBatchMatmul).
    """
    export_talker_kv(
        model_path=model_path,
        output_dir=output_dir,
        opset=int(opset),
        dtype=str(dtype),
        export_seq_len=int(export_seq_len),
        device=str(device),
        ascend_fused_ops=bool(ascend_fused_ops),
        custom_rope=bool(export_custom_ops),
        enable_ptq=bool(enable_ptq),
        ptq_calib_data=str(ptq_calib_data),
        ptq_skip_layers=str(ptq_skip_layers),
        smooth_alpha=float(smooth_alpha),
        weight_clip_ratio=float(weight_clip_ratio),
    )
    if strip_control_flow:
        _strip_talker_prefill_control_flow(output_dir=os.path.join(str(output_dir), "prefill"))


def _postprocess_speech_decoder_onnx(output_path: str) -> None:
    """Patch exported speech_decoder.onnx for better converter/runtime compatibility."""
    onnx_mod = _require_module("onnx")
    helper = onnx_mod.helper
    tensor_proto = onnx_mod.TensorProto

    model = onnx_mod.load(output_path, load_external_data=True)
    graph = model.graph

    counter = 0

    def _unique(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter}"

    def _make_const_int64(values: list[int], name_prefix: str):
        out = _unique(f"{name_prefix}_out")
        tensor = helper.make_tensor(
            name=_unique(f"{name_prefix}_tensor"),
            data_type=tensor_proto.INT64,
            dims=[len(values)],
            vals=values,
        )
        node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[out],
            name=_unique(f"{name_prefix}_const"),
            value=tensor,
        )
        return node, out

    def _attr_int(node, key: str) -> int | None:
        for attr in node.attribute:
            if attr.name == key:
                return int(attr.i)
        return None

    def _cast_conv_weights_to_fp32(node):
        """Cast Conv/ConvTranspose weight and bias initializers from bf16 to fp32."""
        for inp_idx in (1, 2):  # 1=weight, 2=bias (optional)
            if inp_idx >= len(node.input):
                continue
            inp_name = node.input[inp_idx]
            if not inp_name:
                continue
            for init in graph.initializer:
                if init.name == inp_name and init.data_type == tensor_proto.BFLOAT16:
                    arr = onnx_mod.numpy_helper.to_array(init)
                    arr_fp32 = arr.astype(np.float32)
                    init_new = onnx_mod.numpy_helper.from_array(arr_fp32, name=inp_name)
                    init.CopyFrom(init_new)
                    break

    new_nodes = []
    for node in list(graph.node):
        if node.op_type in ("IsNaN", "IsNan"):
            inp = node.input[0]
            out = node.output[0]
            equal_out = _unique("equal_out")
            equal_node = helper.make_node(
                "Equal",
                inputs=[inp, inp],
                outputs=[equal_out],
                name=_unique("equal"),
            )
            not_node = helper.make_node(
                "Not",
                inputs=[equal_out],
                outputs=[out],
                name=_unique("not"),
            )
            new_nodes.extend((equal_node, not_node))
            continue

        if node.op_type == "ConvTranspose":
            attrs = []
            for attr in node.attribute:
                if attr.name in ("pads", "output_padding"):
                    vals = list(attr.ints)
                    if vals and all(int(v) == 0 for v in vals):
                        continue
                attrs.append(attr)
            if len(attrs) != len(node.attribute):
                node.ClearField("attribute")
                node.attribute.extend(attrs)
            new_nodes.append(node)
            continue

        if node.op_type == "Conv":
            _cast_conv_weights_to_fp32(node)
            new_nodes.append(node)
            continue

        if node.op_type == "Shape":
            start = _attr_int(node, "start")
            end = _attr_int(node, "end")
            full_end = 9223372036854775807
            start_val = 0 if start is None else int(start)
            end_val = full_end if end is None else int(end)
            if start_val == 0 and end_val == full_end:
                new_nodes.append(node)
                continue

            shape_full_out = _unique("shape_full")
            orig_out = node.output[0]
            node.output[0] = shape_full_out
            attrs = [a for a in node.attribute if a.name not in ("start", "end")]
            if len(attrs) != len(node.attribute):
                node.ClearField("attribute")
                node.attribute.extend(attrs)
            new_nodes.append(node)

            starts_node, starts_out = _make_const_int64([start_val], "slice_starts")
            ends_node, ends_out = _make_const_int64([end_val], "slice_ends")
            axes_node, axes_out = _make_const_int64([0], "slice_axes")
            steps_node, steps_out = _make_const_int64([1], "slice_steps")
            slice_node = helper.make_node(
                "Slice",
                inputs=[shape_full_out, starts_out, ends_out, axes_out, steps_out],
                outputs=[orig_out],
                name=_unique("slice"),
            )
            new_nodes.extend((starts_node, ends_node, axes_node, steps_node, slice_node))
            continue

        new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    onnx_mod.checker.check_model(model)
    onnx_mod.save(model, output_path)


def _patch_causal_convnet_pad_to_cat(root: Any) -> int:
    """Patch tokenizer causal convnet padding op to use explicit concat instead of pad."""
    patched = 0

    def _cat_pad_forward(self: Any, hidden_state: Any):
        extra_padding = getattr(self, "_get_extra_padding_for_conv1d")(hidden_state)
        left_pad = int(getattr(self, "padding", 0) or 0)
        right_pad = int(extra_padding or 0)
        batch, channels, _ = hidden_state.shape
        if left_pad > 0:
            left = hidden_state.new_zeros((batch, channels, left_pad))
            hidden_state = torch.cat((left, hidden_state), dim=-1)
        if right_pad > 0:
            right = hidden_state.new_zeros((batch, channels, right_pad))
            hidden_state = torch.cat((hidden_state, right), dim=-1)
        return self.conv(hidden_state).contiguous()

    for mod in root.modules():
        if mod.__class__.__name__ != "Qwen3TTSTokenizerV2CausalConvNet":
            continue
        mod.forward = types.MethodType(_cat_pad_forward, mod)
        patched += 1
    return patched


def _patch_causal_transconvnet_trim_to_gather(root: Any) -> int:
    """Patch tokenizer causal transconvnet trimming to use index_select (gather-like) slicing."""
    patched = 0

    def _gather_trim_forward(self: Any, hidden_state: Any):
        out = self.conv(hidden_state)
        left_pad = int(getattr(self, "left_pad", 0) or 0)
        right_pad = int(getattr(self, "right_pad", 0) or 0)
        if left_pad != 0 or right_pad != 0:
            end = out.shape[-1] - right_pad
            idx = torch.arange(left_pad, end, device=out.device)
            out = out.index_select(-1, idx)
        return out.contiguous()

    for mod in root.modules():
        if mod.__class__.__name__ != "Qwen3TTSTokenizerV2CausalTransConvNet":
            continue
        mod.forward = types.MethodType(_gather_trim_forward, mod)
        patched += 1
    return patched


class SpeechTokenizerV2DecoderOnnxWrapper(_NNModule):
    """Speech tokenizer V2 decoder wrapper for exporting an ONNX-friendly forward."""

    def __init__(self, decoder_model: Any) -> None:
        """Create wrapper with the given decoder_model."""
        super().__init__()
        self.decoder_model = decoder_model

    @staticmethod
    def _causal_mask(seq_len: Any, window: int, dtype: Any, device: Any) -> Any:
        """Build causal attention mask (optionally sliding window) as additive mask."""
        mask_value = torch.finfo(torch.float32).min
        idx = torch.arange(seq_len, device=device)
        q = idx.view(-1, 1)
        kv = idx.view(1, -1)
        allow = kv <= q
        if int(window) > 0:
            start = q - int(window) + 1
            allow = allow & (kv >= start)
        zero = torch.zeros((), device=device, dtype=torch.float32)
        neg = torch.full((), float(mask_value), device=device, dtype=torch.float32)
        mask = torch.where(allow, zero, neg).unsqueeze(0).unsqueeze(0)
        return mask.to(dtype)

    def forward(self, codes: Any) -> Any:
        """Decode codes into waveform using the tokenizer decoder subgraph."""
        hidden = self.decoder_model.quantizer.decode(codes)
        hidden = self.decoder_model.pre_conv(hidden).transpose(1, 2)

        t = torch.onnx.operators.shape_as_tensor(hidden)[1]
        device = hidden.device
        mask_full = self._causal_mask(seq_len=t, window=0, dtype=hidden.dtype, device=device)
        sliding = int(getattr(self.decoder_model.pre_transformer.config, "sliding_window", 0) or 0)
        mask_sliding = self._causal_mask(seq_len=t, window=sliding, dtype=hidden.dtype, device=device)
        attn_mask = {"full_attention": mask_full, "sliding_attention": mask_sliding}

        hidden = self.decoder_model.pre_transformer(inputs_embeds=hidden, attention_mask=attn_mask).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.decoder_model.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder_model.decoder:
            wav = block(wav)
        return torch.clamp(wav, torch.tensor(-1.0, dtype=wav.dtype, device=wav.device),
                                torch.tensor(1.0, dtype=wav.dtype, device=wav.device))


def export_speech_decoder_onnx(
    model_path: str,
    output_dir: str,
    opset: int = 17,
    dtype: str = "float32",
    device: str = "cpu",
    example_seq_len: int = 100,
) -> str:
    """Export speech decoder ONNX model."""
    torch_mod = _require_module("torch")
    qwen_tts = _require_module("qwen_tts")
    model_cls = getattr(qwen_tts, "Qwen3TTSModel")
    model = model_cls.from_pretrained(
        model_path,
        device_map=str(device),
        dtype=_torch_dtype(dtype),
    )
    speech_tokenizer = model.model.speech_tokenizer
    decoder_model = speech_tokenizer.model.decoder
    decoder_model.eval()
    _patch_causal_convnet_pad_to_cat(decoder_model)
    _patch_causal_transconvnet_trim_to_gather(decoder_model)
    # Patch EuclideanCodebook.decode to pre-compute and cache embedding,
    # avoiding dtype-mismatched Clip ops (bf16 input vs fp32 epsilon)
    # in the ONNX graph that break converter_lite.
    try:
        euclidean_codebook = _require_module(
            "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
        ).EuclideanCodebook

        def _cached_decode(self, codes):
            cached = getattr(self, "_cached_embedding", None)
            if cached is None:
                with torch.no_grad():
                    cached = (
                        self.embedding_sum
                        / self.cluster_usage.clamp(min=self.epsilon)[:, None]
                    )
                    setattr(self, "_cached_embedding", cached)
            quantized = torch.nn.functional.embedding(codes, cached)
            return quantized

        euclidean_codebook.decode = _cached_decode
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError):
        pass
    try:
        setattr(decoder_model.pre_transformer.config, "_attn_implementation", "eager")
    except (AttributeError, TypeError):
        pass
    wrapper = SpeechTokenizerV2DecoderOnnxWrapper(decoder_model).eval()

    os.makedirs(output_dir, exist_ok=True)
    num_quantizers = int(decoder_model.config.num_quantizers)
    codebook_size = int(decoder_model.config.codebook_size)
    codes = torch_mod.randint(
        low=0,
        high=codebook_size,
        size=(1, num_quantizers, int(example_seq_len)),
        device=torch_mod.device("cpu"),
    )

    output_path = os.path.join(output_dir, "speech_decoder.onnx")
    with torch_mod.no_grad():
        _ = wrapper(codes)
        torch_mod.onnx.export(
            wrapper,
            (codes,),
            output_path,
            input_names=["codes"],
            output_names=["wav"],
            opset_version=int(opset),
            do_constant_folding=True,
            training=torch_mod.onnx.TrainingMode.EVAL,
            dynamo=False,
            dynamic_axes={
                "codes": {0: "batch_size", 2: "seq_len"},
                "wav": {0: "batch_size", 2: "output_seq_len"},
            },
        )
    _postprocess_speech_decoder_onnx(output_path)
    _save_ref_io(wrapper, (codes,), ["codes"], ["wav"], output_path)
    return output_path


class CodePredictorFlashWrapper(_NNModule):
    """Code predictor wrapper that uses Custom FA/IFA attention ops.

    Replaces the standard HuggingFace transformer forward with manual
    layer loop using CustomPromptFlashAttention (prefill) and
    CustomIncreFlashAttention (decode steps).  KV cache is managed as
    plain tensors (not DynamicCache).
    """

    def __init__(
        self, code_predictor: Any, *,
        use_ascend_fused_ops: bool = False,
        use_custom_rope: bool = False,
        ifa_layout: str = "BNSD",
    ) -> None:
        super().__init__()
        self.code_predictor = code_predictor
        self.projection = code_predictor.small_to_mtp_projection
        self.transformer = code_predictor.model
        self.layers = list(code_predictor.model.layers)
        self.final_norm = code_predictor.model.norm
        self.codec_embedding = code_predictor.model.codec_embedding

        c = code_predictor.config
        mc = code_predictor.model.config
        self.hidden_size = int(c.hidden_size)
        self.num_heads = int(c.num_attention_heads)
        self.num_kv_heads = int(c.num_key_value_heads)
        self.num_layers = int(mc.num_hidden_layers)
        self.num_code_groups = int(c.num_code_groups)
        self.head_dim = int(getattr(code_predictor.model.config, "head_dim", c.hidden_size // c.num_attention_heads))
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = float(self.head_dim ** -0.5)

        # LM head weights  [num_groups, H, V]  (pre-transposed so no transpose at runtime)
        self.register_buffer("lm_head_weights_t",
            torch.stack([h.weight for h in code_predictor.lm_head], dim=0)
                .transpose(1, 2).contiguous())

        self.use_ascend_fused_ops = bool(use_ascend_fused_ops)
        self.allow_custom_rope = bool(use_custom_rope)
        layout = str(ifa_layout).strip().upper()
        if layout not in ("BSND", "BNSD"):
            layout = "BNSD"
        self.ifa_layout = layout

    # ── RoPE (standard 2D) ──────────────────────────────────────
    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor,
                    unsqueeze_dim: int = 1,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """2D RoPE.  cos/sin: [B, S, D] → unsqueeze to match layout.
        unsqueeze_dim=1 → [B,1,S,D] (BNSD).  unsqueeze_dim=2 → [B,S,1,D] (BSND)."""
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)
        if self.allow_custom_rope:
            return (CustomRoatryMul.apply(q, cos_u, sin_u),
                    CustomRoatryMul.apply(k, cos_u, sin_u))
        return (_rotary_mul_plain(q, cos_u, sin_u),
                _rotary_mul_plain(k, cos_u, sin_u))

    # ── Prefill attention  (CustomPromptFlashAttention) ─────────
    def _attention_prefill(self, layer, hidden_states: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor,
                           attn_mask: torch.Tensor, layer_idx: int = 0
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prefill attention and return (out, k, v) in BSND cache layout."""
        del layer_idx
        b, s, _ = hidden_states.shape
        attn = layer.self_attn

        qkv = torch.nn.functional.linear(
            hidden_states,
            torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
        q_raw, k_raw, v_raw = torch.split(
            qkv, [self.num_heads * self.head_dim,
                  self.num_kv_heads * self.head_dim,
                  self.num_kv_heads * self.head_dim], dim=-1)

        q = attn.q_norm(q_raw.view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
        k = attn.k_norm(k_raw.view(b, s, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v_raw.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q, k, cos, sin)
        k_rep = _repeat_kv(k, self.num_key_value_groups)
        v_rep = _repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scaling
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1).to(v_rep.dtype)
        out = torch.matmul(probs, v_rep)
        out = out.transpose(1, 2).contiguous().reshape(b, s, -1)
        out = attn.o_proj(out)
        # Return k,v in BSND for cache consistency
        return out, k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

    # ── Decode-step attention  (CustomIncreFlashAttention) ──────
    def _attention_step(self, layer, hidden_states: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor,
                        past_k: torch.Tensor, past_v: torch.Tensor,
                        cache_pos: torch.Tensor, layer_idx: int = 0
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute a single decode-step attention and update the KV cache."""
        del layer_idx
        b, s, _ = hidden_states.shape  # s == 1
        attn = layer.self_attn

        qkv = torch.nn.functional.linear(
            hidden_states,
            torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
        q_raw, k_raw, v_raw = torch.split(
            qkv, [self.num_heads * self.head_dim,
                  self.num_kv_heads * self.head_dim,
                  self.num_kv_heads * self.head_dim], dim=-1)

        if not self.use_ascend_fused_ops:
            # ── Standard path: BNSD layout (with transposes) ──
            q = attn.q_norm(q_raw.view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(k_raw.view(b, s, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = v_raw.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, cos, sin)

            # Cache is BSND → convert to BNSD for standard attention
            pk_bnsd = past_k.transpose(1, 2).contiguous()
            pv_bnsd = past_v.transpose(1, 2).contiguous()
            cache_total = int(pk_bnsd.size(2))
            cp = torch.clamp(cache_pos.to(torch.int64), min=0, max=cache_total - 1)
            idx = _to_cache_indices(cp, q.device)
            k_full = CustomScatterUpdate.apply(pk_bnsd, idx, k, 2)
            v_full = CustomScatterUpdate.apply(pv_bnsd, idx, v, 2)

            mask_val = float(torch.finfo(torch.float32).min)
            kv_idx = torch.arange(cache_total, device=q.device,
                                  dtype=torch.int64).view(1, cache_total)
            allow = kv_idx <= cp.view(-1, 1)
            if allow.size(0) != q.size(0):
                allow = allow.expand(q.size(0), cache_total)
            allow4d = allow[:, None, None, :]
            pad4d = torch.where(
                allow4d,
                torch.zeros_like(allow4d, dtype=torch.float32),
                torch.full_like(allow4d, mask_val, dtype=torch.float32),
            ).to(q.dtype)

            k_rep = _repeat_kv(k_full, self.num_key_value_groups)
            v_rep = _repeat_kv(v_full, self.num_key_value_groups)
            scores = torch.matmul(q, k_rep.transpose(-2, -1)) * self.scaling
            scores = scores + pad4d
            probs = torch.softmax(scores, dim=-1).to(v_rep.dtype)
            out = torch.matmul(probs, v_rep)
            out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
            out = attn.o_proj(out)
            # Return in BSND
            return out, k_full.transpose(1, 2).contiguous(), v_full.transpose(1, 2).contiguous()

        if self.ifa_layout == "BNSD":
            # ── IFA path: BNSD layout ──
            q = attn.q_norm(q_raw.view(b, s, self.num_heads, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(k_raw.view(b, s, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = v_raw.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope(q, k, cos, sin, unsqueeze_dim=1)

            # Cache is BSND → convert to BNSD for ScatterUpdate on axis=2
            pk_bnsd = past_k.transpose(1, 2).contiguous()   # [B, n_kv, S, D]
            pv_bnsd = past_v.transpose(1, 2).contiguous()
            cache_total = int(pk_bnsd.size(2))               # BNSD: seq dim=2
            cp = torch.clamp(cache_pos.to(torch.int64), min=0, max=cache_total - 1)
            idx = _to_cache_indices(cp, q.device)
            k_full = CustomScatterUpdate.apply(pk_bnsd, idx, k, 2)
            v_full = CustomScatterUpdate.apply(pv_bnsd, idx, v, 2)

            mask_val = float(torch.finfo(torch.float32).min)
            kv_idx = torch.arange(cache_total, device=q.device,
                                  dtype=torch.int64).view(1, cache_total)
            allow = kv_idx <= cp.view(-1, 1)
            if allow.size(0) != q.size(0):
                allow = allow.expand(q.size(0), cache_total)
            allow4d = allow[:, None, None, :]
            pad4d = torch.where(
                allow4d,
                torch.zeros_like(allow4d, dtype=torch.float32),
                torch.full_like(allow4d, mask_val, dtype=torch.float32),
            ).to(q.dtype)
            mask_bool = pad4d != 0

            out = incre_flash_attention(
                q, k_full, v_full, mask_bool,
                self.num_heads, self.scaling, "BNSD",
                self.num_kv_heads, block_size=16, inner_precise=0)
            if out.dim() == 4:
                out = out.transpose(1, 2).reshape(b, s, self.num_heads * self.head_dim)
            out = attn.o_proj(out)
            # Return in BSND (convert BNSD cache back to BSND)
            return out, k_full.transpose(1, 2).contiguous(), v_full.transpose(1, 2).contiguous()

        # ── IFA path: BSND layout (no transposes) ──
        q_bs = attn.q_norm(q_raw.view(b, s, self.num_heads, self.head_dim))
        k_bs = attn.k_norm(k_raw.view(b, s, self.num_kv_heads, self.head_dim))
        v_bs = v_raw.view(b, s, self.num_kv_heads, self.head_dim)
        q_bs, k_bs = self._apply_rope(q_bs, k_bs, cos, sin, unsqueeze_dim=2)

        # Cache is BSND [B, S, n_kv, D]; k_bs is [B, 1, n_kv, D]
        cache_total = int(past_k.size(1))              # BSND: seq dim=1
        cp = torch.clamp(cache_pos.to(torch.int64), min=0, max=cache_total - 1)
        idx = _to_cache_indices(cp, q_bs.device)
        k_full = CustomScatterUpdate.apply(past_k, idx, k_bs, 1)
        v_full = CustomScatterUpdate.apply(past_v, idx, v_bs, 1)

        mask_val = float(torch.finfo(torch.float32).min)
        kv_idx = torch.arange(cache_total, device=q_bs.device,
                              dtype=torch.int64).view(1, cache_total)
        allow = kv_idx <= cp.view(-1, 1)
        if allow.size(0) != q_bs.size(0):
            allow = allow.expand(q_bs.size(0), cache_total)
        allow4d = allow[:, None, None, :]
        pad4d = torch.where(
            allow4d,
            torch.zeros_like(allow4d, dtype=torch.float32),
            torch.full_like(allow4d, mask_val, dtype=torch.float32),
        ).to(q_bs.dtype)
        mask_bool = pad4d != 0

        out = incre_flash_attention(
            q_bs, k_full, v_full, mask_bool,
            self.num_heads, self.scaling, "BSND",
            self.num_kv_heads, block_size=16, inner_precise=0)
        if out.dim() == 4:
            out = out.reshape(b, s, self.num_heads * self.head_dim)
        out = attn.o_proj(out)
        # Return in BSND (already in correct format)
        return out, k_full, v_full

    # ── Single decode step ──────────────────────────────────────
    def _step(
        self, inputs_embeds: torch.Tensor,
        generation_step: torch.Tensor,
        past_k: torch.Tensor | None,
        past_v: torch.Tensor | None,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one decode step: attention, MLP, LM head and greedy sampling."""
        proj_in = self.projection(inputs_embeds)          # [B, S, 1024]
        _, s, _ = proj_in.shape
        device = proj_in.device

        # Position embeddings (standard 2D RoPE)
        pos_ids = cache_position.unsqueeze(0).to(device)  # [1, S]
        cos, sin = self.transformer.rotary_emb(proj_in, pos_ids)  # [B, S, D]

        # Causal mask for prefill
        if past_k is None:
            mask_val = float(torch.finfo(torch.float32).min)
            attn_mask = torch.full((1, 1, s, s), mask_val, device=device, dtype=torch.float32)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        else:
            attn_mask = None

        hs = proj_in
        k_list, v_list = [], []

        for li, layer in enumerate(self.layers):
            # Pre-norm + attention
            residual = hs
            hs = layer.input_layernorm(hs)

            if past_k is None:
                attn_out, kn, vn = self._attention_prefill(
                    layer, hs, cos, sin, attn_mask, layer_idx=li)
            else:
                attn_out, kn, vn = self._attention_step(
                    layer, hs, cos, sin,
                    past_k[li], past_v[li], cache_position, layer_idx=li)

            hs = residual + attn_out

            # Post-norm + MLP (SwiGLU)
            residual = hs
            hs = layer.post_attention_layernorm(hs)
            hs = layer.mlp(hs)
            hs = residual + hs

            k_list.append(kn)
            v_list.append(vn)

        # Final norm
        hs = self.final_norm(hs)
        last_hidden = hs[:, -1, :]                         # [B, H]

        # Update KV cache tensors
        if past_k is None:
            # Pre-allocate cache with capacity for all future decode steps
            cp_len = int(inputs_embeds.shape[1])
            total_steps = int(self.num_code_groups) - 1    # 15
            cache_total = cp_len + total_steps
            # k_list items are [B, S, n_kv, D] (BSND)
            k_stacked = torch.stack(k_list, dim=0)           # [L, B, S, n_kv, D]
            v_stacked = torch.stack(v_list, dim=0)
            pad_sz = cache_total - cp_len
            if pad_sz > 0:
                new_past_k = F.pad(k_stacked, (0, 0, 0, 0, 0, pad_sz, 0, 0, 0, 0))
                new_past_v = F.pad(v_stacked, (0, 0, 0, 0, 0, pad_sz, 0, 0, 0, 0))
            else:
                new_past_k, new_past_v = k_stacked, v_stacked
        else:
            # Use scattered KV from each layer (_attention_step returns updated cache)
            new_past_k = torch.stack(k_list, dim=0)
            new_past_v = torch.stack(v_list, dim=0)

        # LM head  (pre-transposed weights → no transpose at runtime)
        weights = torch.index_select(self.lm_head_weights_t, 0, generation_step)  # [1, H, V]
        logits = torch.matmul(last_hidden.unsqueeze(1), weights)[:, 0]  # [B, V]

        # ── Sampling (greedy: argmax, top-p=1.0 and temp=1.0 are constants) ──
        ntok = torch.argmax(logits, dim=-1, keepdim=True).reshape(-1).long()
        return ntok, new_past_k, new_past_v

    # ── Full generate loop (15 unrolled steps) ──────────────────
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the unrolled generate loop and return stacked codec ids."""
        token_list: list[torch.Tensor] = []
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        device = inputs_embeds.device
        for step_idx in range(int(self.num_code_groups) - 1):
            gs = torch.tensor([step_idx], dtype=torch.int64, device=device)

            if step_idx == 0:
                cp = torch.arange(inputs_embeds.shape[1], device=device)
                ntok, past_k, past_v = self._step(
                    inputs_embeds, gs, None, None, cp)
            else:
                cp = torch.zeros_like(inputs_embeds[:, 0, 0], dtype=torch.int64) + (
                    inputs_embeds.shape[1] + step_idx - 1
                )
                ntok, past_k, past_v = self._step(
                    next_code_hidden, gs, past_k, past_v, cp)

            token_list.append(ntok)
            next_code_hidden = self.codec_embedding[step_idx](ntok).unsqueeze(1)

        return torch.stack(token_list, dim=1)


class GenerateProcessAndStepEmbedWrapperFlash(_NNModule):
    """FA/IFA version: uses CodePredictorFlashWrapper instead of GenerateProcessWrapper."""

    def __init__(self, code_predictor: Any, *,
                 use_ascend_fused_ops: bool = False,
                 use_custom_rope: bool = False,
                 ifa_layout: str = "BNSD") -> None:
        super().__init__()
        self.gen = CodePredictorFlashWrapper(
            code_predictor,
            use_ascend_fused_ops=bool(use_ascend_fused_ops),
            use_custom_rope=bool(use_custom_rope),
            ifa_layout=str(ifa_layout),
        )
        self.codec_embedding = code_predictor.model.codec_embedding

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        next_id: torch.Tensor,
        last_id_hidden: torch.Tensor,
        trailing_step: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated codec ids and the accumulated step embedding."""
        sequences = self.gen(inputs_embeds).to(torch.long)
        next_id = next_id.to(torch.long)
        codec_ids = torch.cat([next_id, sequences], dim=1)

        acc = last_id_hidden
        for i in range(int(sequences.shape[1])):
            e = self.codec_embedding[i](sequences[:, i]).to(acc.dtype).unsqueeze(1)
            acc = acc + e
        step_embed = acc + trailing_step.to(acc.dtype)
        return codec_ids, step_embed


def _torch_dtype(name: str) -> torch.dtype:
    torch_mod = _require_module("torch")
    val = (name or "float32").strip().lower()
    if val in ("float16", "fp16"):
        return torch_mod.float16
    if val in ("bfloat16", "bf16"):
        return torch_mod.bfloat16
    return torch_mod.float32


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for exporting Qwen3-TTS ONNX models."""
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS ONNX models in one shot.")
    parser.add_argument("--model_path", type=str, default="../Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--output_root", type=str, default="./onnx_models")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--talker_export_seq_len", type=int, default=32)
    parser.add_argument("--speech_example_seq_len", type=int, default=16)
    parser.add_argument("--code_predictor_ifa_layout", type=str, default="BNSD",
                        choices=["BNSD", "BSND"],
                        help="IncreFlashAttention layout for code predictor. "
                             "BNSD (default) or BSND.")
    # PTQ int8 MatMul 量化参数
    parser.add_argument("--ptq-calib-data", type=str, default="./quant_calib.jsonl",
                        help="Path to calibration data JSONL for PTQ int8 quantization.")
    parser.add_argument("--ptq-smooth-alpha", type=float, default=0.65,
                        help="SmoothQuant alpha for PTQ (0=纯 weight, 1=纯激活).")
    parser.add_argument("--ptq-weight-clip-ratio", type=float, default=0.01,
                        help="Weight outlier clip ratio for PTQ.")
    parser.add_argument(
        "--ptq-skip-layers",
        type=str,
        default=(
            "layer.0,layer.1,layer.2,layer.3,layer.4,layer.5,layer.6,layer.7,layer.8,"
            "layer.9,layer.10,layer.11,layer.16,layer.17,layer.18,layer.19,layer.20,"
            "layer.21,layer.22,layer.23,layer.24,layer.25,layer.26,layer.27"
        ),
        help="Comma-separated layer types to skip from quantization. "
             "Default skips qkv + mlp to stabilize hidden-state accumulation.",
    )
    parser.add_argument("--disable-ptq", action="store_true",
                        help="Disable PTQ int8 MatMul quantization (enabled by default when calib data present).")
    return parser.parse_args(argv)


def _export_talker_if_enabled(args: argparse.Namespace, output_root: str) -> None:
    """Export talker models."""
    out_dir = output_root
    disable_ptq = bool(args.disable_ptq)
    ptq_calib = str(args.ptq_calib_data)
    enable_ptq = not disable_ptq and bool(ptq_calib) and os.path.isfile(str(ptq_calib))

    export_talker_kv_onnx(
        model_path=str(args.model_path),
        output_dir=out_dir,
        opset=int(args.opset),
        dtype=str(args.dtype),
        export_seq_len=int(args.talker_export_seq_len),
        device="cpu",
        export_custom_ops=True,
        ascend_fused_ops=True,
        strip_control_flow=True,
        enable_ptq=enable_ptq,
        ptq_calib_data=ptq_calib,
        ptq_skip_layers=str(args.ptq_skip_layers),
        smooth_alpha=float(args.ptq_smooth_alpha),
        weight_clip_ratio=float(args.ptq_weight_clip_ratio),
    )
    print(os.path.join(out_dir, "talker_prefill.onnx"))
    print(os.path.join(out_dir, "talker_step.onnx"))


def _export_speech_if_enabled(args: argparse.Namespace, output_root: str) -> None:
    """Export speech decoder."""
    out_dir = output_root
    out = export_speech_decoder_onnx(
        model_path=str(args.model_path),
        output_dir=out_dir,
        opset=int(args.opset),
        dtype=str(args.dtype),
        device="cpu",
        example_seq_len=int(args.speech_example_seq_len),
    )
    print(out)


def _load_talker_configs(model_path: str) -> tuple[Any, Any, int]:
    """Load talker config objects for code predictor export."""
    cfg_mod = _require_module("qwen_tts.core.models.configuration_qwen3_tts")
    modeling_mod = _require_module("qwen_tts.core.models.modeling_qwen3_tts")
    talker_cfg_cls = getattr(modeling_mod, "Qwen3TTSTalkerConfig")
    code_cfg_cls = getattr(cfg_mod, "Qwen3TTSTalkerCodePredictorConfig")

    config_path = os.path.join(str(model_path), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        full_config = json.load(f)
    talker_config_dict = full_config["talker_config"]
    code_predictor_config = code_cfg_cls(**talker_config_dict["code_predictor_config"])
    talker_config = talker_cfg_cls(**talker_config_dict)
    talker_hidden_size = int(talker_config.hidden_size)
    return talker_config, code_predictor_config, talker_hidden_size


def _filter_checkpoint_state_dict(checkpoint: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Filter safetensors state_dict by prefix and strip wrappers like module."""
    filtered: dict[str, Any] = {}
    for key, value in checkpoint.items():
        k = str(key)
        if k.startswith("module."):
            k = k[len("module.") :]
        if k.startswith(prefix):
            filtered[k[len(prefix) :]] = value
    return filtered


def _load_code_predictor_weights(ckpt_path: str, prefix: str) -> dict[str, Any]:
    """Load safetensors checkpoint and return filtered code predictor weights."""
    safetensors_torch = _require_module("safetensors.torch")
    load_file = getattr(safetensors_torch, "load_file")
    checkpoint = load_file(str(ckpt_path))
    filtered = _filter_checkpoint_state_dict(checkpoint, prefix)
    if not filtered:
        sample = ", ".join(list(checkpoint.keys())[:5])
        message = (
            f"No code_predictor weights matched prefix {prefix!r}. "
            f"checkpoint={str(ckpt_path)!r}, sample_keys=[{sample}]"
        )
        raise RuntimeError(message)
    return filtered


def _export_generate_process_onnx(
    wrapper: Any,
    output_path: str,
    *,
    opset: int,
    export_custom_ops: bool,
    talker_hidden_size: int,
    dtype: str = "bfloat16",
) -> None:
    """Export generate_process ONNX."""
    torch_mod = _require_module("torch")
    export_dtype = _torch_dtype(dtype)
    batch_size = int(os.environ.get("EXPORT_BATCH_SIZE", "1"))
    initial_seq_len = 2
    inputs_embeds = torch_mod.randn(batch_size, initial_seq_len, int(talker_hidden_size),
                                    dtype=export_dtype)
    next_id = torch_mod.zeros((batch_size, 1), dtype=torch_mod.int64)
    last_id_hidden = torch_mod.randn(batch_size, 1, int(talker_hidden_size),
                                     dtype=export_dtype)
    trailing_step = torch_mod.randn(batch_size, 1, int(talker_hidden_size),
                                    dtype=export_dtype)

    with torch_mod.no_grad():
        _ = wrapper(inputs_embeds, next_id, last_id_hidden, trailing_step)
        export_kwargs: dict[str, Any] = {}
        if "dynamo" in inspect.signature(torch_mod.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        if bool(export_custom_ops):
            export_kwargs["operator_export_type"] = torch_mod.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
        torch_mod.onnx.export(
            wrapper,
            (inputs_embeds, next_id, last_id_hidden, trailing_step),
            str(output_path),
            input_names=["inputs_embeds", "next_id", "last_id_hidden", "trailing_step"],
            output_names=["codec_ids", "step_embed"],
            opset_version=int(opset),
            do_constant_folding=True,
            training=torch_mod.onnx.TrainingMode.EVAL,
            dynamic_axes={
                "inputs_embeds": {0: "batch"},
                "next_id": {0: "batch"},
                "last_id_hidden": {0: "batch"},
                "trailing_step": {0: "batch"},
                "codec_ids": {0: "batch"},
                "step_embed": {0: "batch"},
            },
            **export_kwargs,
        )
    _save_ref_io(
        wrapper,
        (inputs_embeds, next_id, last_id_hidden, trailing_step),
        ["inputs_embeds", "next_id", "last_id_hidden", "trailing_step"],
        ["codec_ids", "step_embed"],
        str(output_path),
    )


def _export_code_predictor_if_enabled(args: argparse.Namespace, output_root: str) -> None:
    """Export code predictor generate_process."""
    out_dir = output_root
    ckpt = os.path.join(str(args.model_path), "model.safetensors")
    prefix = "talker.code_predictor."
    talker_config, code_predictor_config, talker_hidden_size = _load_talker_configs(str(args.model_path))
    modeling_mod = _require_module("qwen_tts.core.models.modeling_qwen3_tts")
    code_model_cls = getattr(modeling_mod, "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration")
    code_predictor = code_model_cls(code_predictor_config, talker_config).eval()

    filtered_checkpoint = _load_code_predictor_weights(ckpt, prefix)
    code_predictor.load_state_dict(filtered_checkpoint, strict=True)
    # Convert to target dtype for ONNX export
    code_predictor = code_predictor.to(_torch_dtype(args.dtype))
    code_predictor.eval()

    try:
        setattr(code_predictor.model.config, "_attn_implementation", "eager")
    except (AttributeError, TypeError):
        pass

    ifa_layout = str(getattr(args, "code_predictor_ifa_layout", "BNSD")).strip().upper()
    if ifa_layout not in ("BNSD", "BSND"):
        ifa_layout = "BNSD"
    wrapper = GenerateProcessAndStepEmbedWrapperFlash(
        code_predictor,
        use_ascend_fused_ops=True,
        use_custom_rope=True,
        ifa_layout=ifa_layout,
    ).eval()

    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "generate_process.onnx")
    _export_generate_process_onnx(
        wrapper,
        output_path,
        opset=int(args.opset),
        export_custom_ops=True,
        talker_hidden_size=int(talker_hidden_size),
        dtype=str(args.dtype),
    )
    print(output_path)


def main(argv: list[str] | None = None) -> int:
    """Entry point for exporting Qwen3-TTS ONNX models."""
    args = _parse_args(argv)
    output_root = os.path.abspath(args.output_root)
    os.makedirs(output_root, exist_ok=True)

    _export_talker_if_enabled(args, output_root)
    _export_speech_if_enabled(args, output_root)
    _export_code_predictor_if_enabled(args, output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
