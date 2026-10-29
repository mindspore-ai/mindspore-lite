# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""GPTQ tensor loading and reference dequantization for Qwen3.5 MoE."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def cube_qweight(tensors: dict[str, np.ndarray]) -> np.ndarray:
    """Return signed INT4 in Kirin widening-cast/Cube feed order.

    GPTQ stores unsigned nibbles in [K/8,N]. Symmetric Qwen3.5 GPTQ uses a
    zero point of 8, so flipping each nibble's sign bit preserves q - zero.
    Kirin's INT4-to-FP16 widening cast emits the low nibbles of 256 bytes,
    followed by their high nibbles, while P0..P3 select every fourth nibble.
    ND2NZ lays activation K as even values followed by odd values. Therefore
    byte[k,row] pairs row[2*k] in its low nibble with row[2*k+1] in its high
    nibble. Each part then matches the activation's physical K order directly.
    The returned tensor is [N/4,K/2] INT32.
    """
    qweight = tensors["qweight"]
    qzeros = tensors["qzeros"].view(np.uint32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    zeros = ((qzeros[..., None] >> shifts) & np.uint32(0xF)).reshape(
        qzeros.shape[0], qzeros.shape[1] * 8
    )
    if not np.all(zeros == 8):
        raise ValueError("Cube GPTQ layout requires symmetric zero points equal to 8")
    quantized_nk = unpack_int4(qweight, axis=0).T.astype(np.uint8, copy=False)
    n, k = quantized_nk.shape
    if k % 128:
        raise ValueError(f"Cube GPTQ K must align to 128: {(n, k)}")
    signed_nibbles = quantized_nk ^ np.uint8(0x8)
    if n % 4:
        raise ValueError(f"Cube GPTQ N must align to 4: {(n, k)}")
    groups = signed_nibbles.reshape(n // 4, 4, k // 128, 128)
    groups = groups.transpose(0, 2, 3, 1)
    packed_bytes = groups[:, :, 0::2, :] | (
        groups[:, :, 1::2, :] << np.uint8(4)
    )
    return np.ascontiguousarray(packed_bytes).view(np.int32).reshape(n // 4, k // 2)


@dataclass(frozen=True)
class Quantization:
    bits: int
    group_size: int
    symmetric: bool


class IndexedSafetensors:
    """Reads individual tensors without materializing a model or shard set."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.resolve()
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing safetensors index: {index_path}")
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self._contexts = ExitStack()
        self._shards = {}

    def close(self):
        """Release the lazily opened shard mappings."""
        self._contexts.close()
        self._shards.clear()

    def __del__(self):
        contexts = getattr(self, "_contexts", None)
        if contexts is not None:
            contexts.close()

    def tensor(self, name: str) -> np.ndarray:
        shard_name = self.weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"tensor is absent from checkpoint index: {name}")
        if shard_name not in self._shards:
            shard = self.model_dir / shard_name
            self._shards[shard_name] = self._contexts.enter_context(safe_open(str(shard), framework="np"))
        return np.asarray(self._shards[shard_name].get_tensor(name))


def text_config(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    return config.get("text_config", config)


def quantization(model_dir: Path) -> Quantization:
    """Validate and return the checkpoint GPTQ quantization configuration."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    settings = config.get("quantization_config", {})
    result = Quantization(
        bits=int(settings.get("bits", 0)),
        group_size=int(settings.get("group_size", 0)),
        symmetric=bool(settings.get("sym", False)),
    )
    if result.bits != 4 or result.group_size != 128 or not result.symmetric:
        raise ValueError(
            "only symmetric GPTQ INT4 with group_size=128 is supported; "
            f"checkpoint has {result}"
        )
    return result


def expert_prefix(layer: int, expert: int, projection: str) -> str:
    if layer < 0 or expert < 0 or projection not in PROJECTIONS:
        raise ValueError("invalid layer, expert, or projection")
    return (
        f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}"
    )


def load_projection(
    checkpoint: IndexedSafetensors, layer: int, expert: int, projection: str
) -> dict[str, np.ndarray]:
    """Read the packed tensors belonging to one GPTQ projection."""
    prefix = expert_prefix(layer, expert, projection)
    tensors = {
        suffix: checkpoint.tensor(f"{prefix}.{suffix}")
        for suffix in ("qweight", "qzeros", "scales", "g_idx")
    }
    g_idx = tensors["g_idx"]
    expected = np.arange(g_idx.size, dtype=np.int32) // 128
    if g_idx.dtype != np.int32 or not np.array_equal(g_idx, expected):
        raise ValueError(
            f"{prefix}.g_idx is not the contiguous desc_act=false layout"
        )
    return tensors


def unpack_int4(values: np.ndarray, axis: int) -> np.ndarray:
    """Unpacks eight little-endian INT4 lanes from every int32 value."""
    if values.dtype != np.int32:
        raise TypeError(f"packed GPTQ tensor must be int32, got {values.dtype}")
    unsigned = values.view(np.uint32)
    shifts = np.arange(0, 32, 4, dtype=np.uint32)
    expanded = np.expand_dims(unsigned, axis=axis + 1)
    shape = [1] * expanded.ndim
    shape[axis + 1] = 8
    lanes = (expanded >> shifts.reshape(shape)) & np.uint32(0xF)
    output_shape = list(values.shape)
    output_shape[axis] *= 8
    return lanes.reshape(output_shape).astype(np.int8)


def dequantize_projection(tensors: dict[str, np.ndarray]) -> np.ndarray:
    """Returns an [N,K] FP32 matrix matching x @ weight.T."""
    qweight = tensors["qweight"]
    qzeros = tensors["qzeros"]
    scales = tensors["scales"]
    g_idx = tensors["g_idx"]
    if qweight.ndim != 2 or qzeros.ndim != 2 or scales.ndim != 2:
        raise ValueError("GPTQ qweight, qzeros, and scales must be rank-2")

    # qweight is [K/8,N]. qzeros packs eight output channels in [G,N/8].
    quantized = unpack_int4(qweight, axis=0)
    zeros = unpack_int4(qzeros, axis=1)
    k, n = quantized.shape
    if scales.shape != zeros.shape or scales.shape[1] != n or g_idx.shape != (k,):
        raise ValueError(
            "inconsistent GPTQ shapes: "
            f"qweight={qweight.shape}, qzeros={qzeros.shape}, "
            f"scales={scales.shape}, g_idx={g_idx.shape}"
        )
    groups = g_idx.astype(np.int64, copy=False)
    if groups.min(initial=0) < 0 or groups.max(initial=0) >= scales.shape[0]:
        raise ValueError("g_idx references a group outside scales")
    weight_kn = (
        quantized.astype(np.float32)
        - zeros[groups].astype(np.float32)
    ) * scales[groups].astype(np.float32)
    return np.ascontiguousarray(weight_kn.T)


def expert_forward(
    hidden: np.ndarray, projections: dict[str, dict[str, np.ndarray]]
) -> np.ndarray:
    """Reference Qwen expert: down(silu(gate) * up), accumulated in FP32."""
    if hidden.dtype not in (np.float16, np.float32) or hidden.shape[-1] != 2048:
        raise ValueError("hidden must be FP16/FP32 with K=2048")
    x = hidden.astype(np.float32)
    gate = x @ dequantize_projection(projections["gate_proj"]).T
    up = x @ dequantize_projection(projections["up_proj"]).T
    silu = gate / (1.0 + np.exp(-gate))
    return (silu * up) @ dequantize_projection(projections["down_proj"]).T


def expert_intermediates(
    hidden: np.ndarray, projections: dict[str, dict[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    """Return FP32 reference values at each Expert projection boundary."""
    if hidden.dtype not in (np.float16, np.float32) or hidden.shape[-1] != 2048:
        raise ValueError("hidden must be FP16/FP32 with K=2048")
    x = hidden.astype(np.float32)
    gate = x @ dequantize_projection(projections["gate_proj"]).T
    up = x @ dequantize_projection(projections["up_proj"]).T
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    silu = gate * sigmoid
    gated = silu * up
    output = gated @ dequantize_projection(projections["down_proj"]).T
    return {
        "gate": gate,
        "up": up,
        "gate_sigmoid": sigmoid,
        "gate_silu": silu,
        "gated": gated,
        "output": output,
    }
