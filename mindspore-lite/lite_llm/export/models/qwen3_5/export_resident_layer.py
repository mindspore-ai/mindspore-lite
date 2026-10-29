# Adapted from HMOS-Device-AI converter/qwen3.5-35b-a3b; see source_provenance.json.
#!/usr/bin/env python3
"""Export one Qwen3.5 MoE resident layer as an FP16 custom-op ONNX POC.

Routed experts are intentionally outside this graph.  The graph produces the
post-attention expert input, routing logits, and shared-expert contribution so
the runtime can page the selected GPTQ experts independently.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, helper


HIDDEN = 2048
CACHE = 8192
LINEAR_HEADS = 32
LINEAR_DIM = 128
FULL_Q_HEADS = 16
FULL_KV_HEADS = 2
FULL_DIM = 256
ROTARY_DIM = 64
EPSILON = np.float32(1.0e-6)


@dataclass(frozen=True)
class SafeRecord:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


class SafeTensorReader:
    """Random-access safetensors reader that materializes only one tensor."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, tuple[int, dict]] = {}

    def record(self, name: str) -> SafeRecord:
        """Find the safetensors shard and metadata for a named tensor."""
        shard_name = self.weight_map[name]
        if shard_name not in self.headers:
            shard = self.model_dir / shard_name
            with shard.open("rb") as stream:
                header_length = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_length))
            self.headers[shard_name] = (8 + header_length, header)
        data_base, header = self.headers[shard_name]
        info = header[name]
        begin, end = info["data_offsets"]
        return SafeRecord(
            self.model_dir / shard_name,
            info["dtype"],
            tuple(int(dim) for dim in info["shape"]),
            data_base + int(begin),
            int(end) - int(begin),
        )

    def load(self, name: str, target: str = "fp16") -> np.ndarray:
        """Read a named tensor and convert it to the requested representation."""
        record = self.record(name)
        with record.shard.open("rb") as stream:
            stream.seek(record.offset)
            raw = stream.read(record.length)
        if len(raw) != record.length:
            raise EOFError(f"short read for {name}")
        dtypes = {
            "BF16": ml_dtypes.bfloat16,
            "F16": np.float16,
            "F32": np.float32,
            "I32": np.int32,
        }
        if record.dtype not in dtypes:
            raise ValueError(f"unsupported source dtype for {name}: {record.dtype}")
        value = np.frombuffer(raw, dtype=dtypes[record.dtype]).reshape(record.shape)
        if target == "fp16":
            return value.astype(np.float16)
        if target == "fp32":
            return value.astype(np.float32)
        raise ValueError(target)


def pack_nz(weight: np.ndarray) -> np.ndarray:
    n, k = weight.shape
    if n % 16 or k % 16:
        raise ValueError(f"NZ requires N/K multiples of 16, got {weight.shape}")
    return np.ascontiguousarray(
        weight.T.reshape(k // 16, 16, n // 16, 16).transpose(2, 0, 1, 3)
    )


class ExternalWeights:
    """Append graph weights to one external tensor data file."""
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("wb")
        self.inputs: list[onnx.ValueInfoProto] = []
        self.input_files: dict[str, str] = {}
        self.layout: dict[str, dict] = {}

    def close(self) -> None:
        self.stream.close()

    def add(self, name: str, value: np.ndarray,
            logical_shape: Sequence[int] | None = None) -> str:
        """Append a tensor and return its ONNX external-data reference."""
        raw = np.ascontiguousarray(value).tobytes()
        padding = (-self.stream.tell()) % 64
        if padding:
            self.stream.write(b"\0" * padding)
        offset = self.stream.tell()
        self.stream.write(raw)
        shape = tuple(int(dim) for dim in (logical_shape or value.shape))
        self.inputs.append(vi(name, shape))
        filename = f"weight_{name}.bin"
        (self.path.parent / filename).write_bytes(raw)
        self.input_files[name] = filename
        self.layout[name] = {"offset": offset, "length": len(raw),
                             "logical_shape": list(shape), "dtype": "float16"}
        return name


class Graph:
    """Build ONNX nodes, constants, and uniquely named intermediate tensors."""
    def __init__(self, weights: ExternalWeights):
        self.weights = weights
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self.counter = 0

    def unique(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}_{self.counter}"

    def node(self, op: str, inputs: Iterable[str], output: str | None = None,
             outputs: Sequence[str] | None = None, **attrs) -> str | list[str]:
        names = list(outputs) if outputs is not None else [output or self.unique(op.lower())]
        self.nodes.append(helper.make_node(op, list(inputs), names, **attrs))
        return names if outputs is not None else names[0]

    def const(self, value, dtype, stem: str = "const") -> str:
        name = self.unique(stem)
        self.initializers.append(
            onnx.numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)
        )
        return name

    def reshape(self, value: str, shape: Sequence[int], output: str | None = None) -> str:
        return self.node("Reshape", [value, self.const(shape, np.int64, "shape")], output)

    def transpose(self, value: str, perm: Sequence[int], output: str | None = None) -> str:
        return self.node("Transpose", [value], output, perm=list(perm))

    def unsqueeze(self, value: str, axes: Sequence[int], output: str | None = None) -> str:
        return self.node("Unsqueeze", [value, self.const(axes, np.int64, "axes")], output)


def vi(name: str, shape: Sequence[int | str]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT16, list(shape))


def tensor_weight(reader: SafeTensorReader, external: ExternalWeights,
                  source_name: str, graph_name: str, nz: bool = False,
                  direct_fp32_bits: bool = False) -> str:
    """Read and optionally pack a model tensor for the exported graph."""
    if direct_fp32_bits:
        value = reader.load(source_name, "fp32")
        return external.add(graph_name, value, [value.size * 2])
    value = reader.load(source_name, "fp16")
    logical_shape = value.shape
    if nz:
        value = pack_nz(value)
    return external.add(graph_name, value, logical_shape)


def combined_weight(reader: SafeTensorReader, external: ExternalWeights,
                    source_names: Sequence[str], graph_name: str,
                    nz: bool = False) -> str:
    """Concatenate independent output projections without changing dtype."""
    values = [reader.load(name, "fp16") for name in source_names]
    if any(value.ndim != 2 for value in values):
        raise ValueError(f"combined projection must be rank-2: {source_names}")
    if len({value.shape[1] for value in values}) != 1:
        raise ValueError(f"combined projection K mismatch: {source_names}")
    value = np.concatenate(values, axis=0)
    packed = pack_nz(value) if nz else value
    return external.add(graph_name, packed, value.shape)


def add_common_moe(g: Graph, reader: SafeTensorReader, external: ExternalWeights,
                   layer: int, expert_input: str,
                   fuse_independent_projections: bool = False) -> tuple[str, str]:
    """Append router and shared-expert projections to the layer graph."""
    base = f"model.language_model.layers.{layer}.mlp"
    router_w = tensor_weight(reader, external, f"{base}.gate.weight", "router_w", nz=True)
    if fuse_independent_projections:
        gate_up_w = combined_weight(
            reader, external,
            [f"{base}.shared_expert.gate_proj.weight",
             f"{base}.shared_expert.up_proj.weight"],
            "shared_gate_up_w", nz=True)
    else:
        gate_w = tensor_weight(reader, external,
                               f"{base}.shared_expert.gate_proj.weight",
                               "shared_gate_w", nz=True)
        up_w = tensor_weight(reader, external,
                             f"{base}.shared_expert.up_proj.weight",
                             "shared_up_w", nz=True)
    down_w = tensor_weight(reader, external, f"{base}.shared_expert.down_proj.weight",
                           "shared_down_w")
    # Cube kernels require a 16-column output block. The generic ND MatMul
    # accepts N=1 at compile time but returns zero for decoder M=1 on
    # Kirin9030, so pad this projection offline and discard the extra lanes.
    score_value = reader.load(f"{base}.shared_expert_gate.weight", "fp16")
    score_padded = np.zeros((16, score_value.shape[1]), dtype=np.float16)
    score_padded[0] = score_value[0]
    score_w = external.add("shared_score_w", pack_nz(score_padded),
                           score_padded.shape)
    router = g.node("MatMulFp16Nz", [expert_input, router_w], "router_logits")
    if fuse_independent_projections:
        gate_up = g.node("MatMulFp16Nz", [expert_input, gate_up_w],
                         "shared_gate_up")
        gate, up = g.node(
            "Split", [gate_up, g.const([512, 512], np.int64, "split")],
            outputs=["shared_gate", "shared_up"], axis=2)
    else:
        gate = g.node("MatMulFp16Nz", [expert_input, gate_w], "shared_gate")
        up = g.node("MatMulFp16Nz", [expert_input, up_w], "shared_up")
    sigmoid = g.node("Sigmoid", [gate])
    activated = g.node("Mul", [gate, sigmoid])
    product = g.node("Mul", [activated, up])
    down = g.node("MatMulFp16", [product, down_w], "shared_down")
    score_padded = g.node("MatMulFp16Nz", [expert_input, score_w],
                          "shared_score_padded")
    score = g.node(
        "Slice",
        [score_padded,
         g.const([0], np.int64, "slice_start"),
         g.const([1], np.int64, "slice_end"),
         g.const([2], np.int64, "slice_axis"),
         g.const([1], np.int64, "slice_step")],
        "shared_score")
    score = g.node("Sigmoid", [score])
    shared = g.node("Mul", [down, score], "shared_expert_output")
    return router, shared


def build_linear(g: Graph, reader: SafeTensorReader, external: ExternalWeights,
                 layer: int, sequence: int, dynamic_sequence: bool = False,
                 fuse_independent_projections: bool = False,
                 fuse_residual_rms: bool = False) -> tuple[list, list, dict]:
    """Build a resident linear-attention layer and its state outputs."""
    base = f"model.language_model.layers.{layer}"
    attn = f"{base}.linear_attn"
    seq = "seq_len" if dynamic_sequence else sequence
    reshape_seq = -1 if dynamic_sequence else sequence
    inputs = [vi("hidden_states", [1, seq, HIDDEN]),
              vi("output_zero", [1]),
              vi("past_conv_state", [1, 8192, 4]),
              vi("past_recurrent_state_bits", [1, 32, 128, 256])]
    in_norm = tensor_weight(reader, external, f"{base}.input_layernorm.weight", "in_norm")
    post_norm = tensor_weight(reader, external, f"{base}.post_attention_layernorm.weight", "post_norm")
    if fuse_independent_projections:
        qkv_z_w = combined_weight(
            reader, external,
            [f"{attn}.in_proj_qkv.weight", f"{attn}.in_proj_z.weight"],
            "qkv_z_w", nz=True)
        a_b_w = combined_weight(
            reader, external,
            [f"{attn}.in_proj_a.weight", f"{attn}.in_proj_b.weight"],
            "a_b_w")
    else:
        qkv_w = tensor_weight(reader, external,
                              f"{attn}.in_proj_qkv.weight", "qkv_w", nz=True)
        z_w = tensor_weight(reader, external,
                            f"{attn}.in_proj_z.weight", "z_w", nz=True)
        a_w = tensor_weight(reader, external, f"{attn}.in_proj_a.weight", "a_w")
        b_w = tensor_weight(reader, external, f"{attn}.in_proj_b.weight", "b_w")
    conv_w = tensor_weight(reader, external, f"{attn}.conv1d.weight", "conv_w")
    dt_bias = tensor_weight(reader, external, f"{attn}.dt_bias", "dt_bias")
    a_log = tensor_weight(reader, external, f"{attn}.A_log", "a_log_bits", direct_fp32_bits=True)
    gated_norm = tensor_weight(reader, external, f"{attn}.norm.weight", "gated_norm")
    out_w = tensor_weight(reader, external, f"{attn}.out_proj.weight", "out_w", nz=True)

    x = g.node("RMSNorm", ["hidden_states", in_norm], "normalized_hidden")
    if fuse_independent_projections:
        qkv_z = g.node("MatMulFp16Nz", [x, qkv_z_w], "projected_qkv_z")
        qkv, z_flat = g.node(
            "Split", [qkv_z, g.const([8192, 4096], np.int64, "split")],
            outputs=["projected_qkv", "projected_z"], axis=2)
    else:
        qkv = g.node("MatMulFp16Nz", [x, qkv_w], "projected_qkv")
    conv, _ = g.node(
        "CausalConv1dSiLU", [qkv, "past_conv_state", conv_w],
        outputs=["convolved_qkv", "present_conv_state"])
    q, k, v = g.node(
        "Split", [conv, g.const([2048, 2048, 4096], np.int64, "split")],
        outputs=["q16", "k16", "v32"], axis=2)
    q = g.reshape(q, [1, reshape_seq, 16, 128])
    k = g.reshape(k, [1, reshape_seq, 16, 128])
    v = g.reshape(v, [1, reshape_seq, 32, 128])
    if fuse_independent_projections:
        z = g.reshape(z_flat, [1, reshape_seq, 32, 128])
        a_b = g.node("MatMulFp16", [x, a_b_w], "a_b")
        a, b = g.node(
            "Split", [a_b, g.const([32, 32], np.int64, "split")],
            outputs=["a", "b"], axis=2)
    else:
        z = g.reshape(g.node("MatMulFp16Nz", [x, z_w]),
                      [1, reshape_seq, 32, 128])
        a = g.node("MatMulFp16", [x, a_w], "a")
        b = g.node("MatMulFp16", [x, b_w], "b")
    core, _ = g.node(
        "GatedDeltaNet", [q, k, v, a, b, dt_bias, a_log,
                          "past_recurrent_state_bits"],
        outputs=["delta_core", "present_recurrent_state_bits"])
    core = g.node("GatedRMSNorm", [core, z, gated_norm], "gated_core")
    core = g.reshape(core, [1, reshape_seq, 4096])
    mixed = g.node("MatMulFp16Nz", [core, out_w], "attention_output")
    if fuse_residual_rms:
        mixed = g.node("Add", [mixed, "output_zero"], "attention_output_with_zero")
        expert_input, residual = g.node(
            "AddRMSNorm", ["hidden_states", mixed, post_norm],
            outputs=["expert_input", "attention_residual"])
    else:
        residual = g.node("Add", ["hidden_states", mixed], "attention_residual_internal")
        # OMG/NNRT cannot expose this internal residual together with cache
        # outputs reliably unless NetOutput receives a dedicated buffer.
        g.node("Add", [residual, "output_zero"], "attention_residual")
        expert_input = g.node("RMSNorm", [residual, post_norm], "expert_input")
    add_common_moe(
        g, reader, external, layer, expert_input,
        fuse_independent_projections)
    outputs = [vi("attention_residual", [1, seq, HIDDEN]),
               vi("expert_input", [1, seq, HIDDEN]),
               vi("router_logits", [1, seq, 256]),
               vi("shared_expert_output", [1, seq, HIDDEN]),
               vi("present_conv_state", [1, 8192, 4]),
               vi("present_recurrent_state_bits", [1, 32, 128, 256])]
    return inputs, outputs, {
        "cache_aliases": {"present_conv_state": "past_conv_state",
                          "present_recurrent_state_bits": "past_recurrent_state_bits"},
        "outputs": [item.name for item in outputs],
        "fused_independent_projections": fuse_independent_projections,
        "fused_residual_rms": fuse_residual_rms,
    }


def build_full(g: Graph, reader: SafeTensorReader, external: ExternalWeights,
               layer: int, sequence: int, dynamic_sequence: bool = False,
               fuse_independent_projections: bool = False,
               fuse_residual_rms: bool = False) -> tuple[list, list, dict]:
    """Build a resident full-attention layer and its KV outputs."""
    base = f"model.language_model.layers.{layer}"
    attn = f"{base}.self_attn"
    seq = "seq_len" if dynamic_sequence else sequence
    reshape_seq = -1 if dynamic_sequence else sequence
    inputs = [vi("hidden_states", [1, seq, HIDDEN]),
              vi("output_zero", [1]),
              vi("rope_cos", [1, seq, ROTARY_DIM]),
              vi("rope_sin", [1, seq, ROTARY_DIM]),
              vi("kvcache_mask", [1, 1, seq, CACHE]),
              vi("past_key", [1, FULL_KV_HEADS, CACHE, FULL_DIM]),
              vi("past_value", [1, FULL_KV_HEADS, CACHE, FULL_DIM])]
    in_norm = tensor_weight(reader, external, f"{base}.input_layernorm.weight", "in_norm")
    post_norm = tensor_weight(reader, external, f"{base}.post_attention_layernorm.weight", "post_norm")
    if fuse_independent_projections:
        qkv_w = combined_weight(
            reader, external,
            [f"{attn}.q_proj.weight", f"{attn}.k_proj.weight",
             f"{attn}.v_proj.weight"],
            "qkv_w", nz=True)
    else:
        q_w = tensor_weight(reader, external, f"{attn}.q_proj.weight", "q_w", nz=True)
        k_w = tensor_weight(reader, external, f"{attn}.k_proj.weight", "k_w", nz=True)
        v_w = tensor_weight(reader, external, f"{attn}.v_proj.weight", "v_w", nz=True)
    o_w = tensor_weight(reader, external, f"{attn}.o_proj.weight", "o_w", nz=True)
    q_norm = tensor_weight(reader, external, f"{attn}.q_norm.weight", "q_norm")
    k_norm = tensor_weight(reader, external, f"{attn}.k_norm.weight", "k_norm")

    x = g.node("RMSNorm", ["hidden_states", in_norm], "normalized_hidden")
    if fuse_independent_projections:
        qkv = g.node("MatMulFp16Nz", [x, qkv_w], "projected_qkv")
        qgate_flat, k_flat, v_flat = g.node(
            "Split", [qkv, g.const([8192, 512, 512], np.int64, "split")],
            outputs=["projected_qgate", "projected_k", "projected_v"], axis=2)
        qgate = g.reshape(qgate_flat,
                          [1, reshape_seq, FULL_Q_HEADS, FULL_DIM * 2])
    else:
        qgate = g.reshape(g.node("MatMulFp16Nz", [x, q_w]),
                          [1, reshape_seq, FULL_Q_HEADS, FULL_DIM * 2])
    q, gate = g.node(
        "Split", [qgate, g.const([FULL_DIM, FULL_DIM], np.int64, "split")],
        outputs=["q", "attention_gate"], axis=3)
    q = g.node("RMSNorm", [q, q_norm], "normalized_q")
    if fuse_independent_projections:
        k = g.reshape(k_flat, [1, reshape_seq, FULL_KV_HEADS, FULL_DIM])
    else:
        k = g.reshape(g.node("MatMulFp16Nz", [x, k_w]),
                      [1, reshape_seq, FULL_KV_HEADS, FULL_DIM])
    k = g.node("RMSNorm", [k, k_norm], "normalized_k")
    if fuse_independent_projections:
        v = g.reshape(v_flat, [1, reshape_seq, FULL_KV_HEADS, FULL_DIM])
    else:
        v = g.reshape(g.node("MatMulFp16Nz", [x, v_w]),
                      [1, reshape_seq, FULL_KV_HEADS, FULL_DIM])
    q, k = g.node("RopeEmbedding", [q, k, "rope_cos", "rope_sin"],
                  outputs=["rope_q", "rope_k"])
    attended, _, _ = g.node(
        "FullAttention", [q, k, v, gate, "kvcache_mask", "past_key", "past_value"],
        outputs=["attention_context", "present_key", "present_value"])
    mixed = g.node("MatMulFp16Nz", [attended, o_w], "attention_output")
    if fuse_residual_rms:
        mixed = g.node("Add", [mixed, "output_zero"], "attention_output_with_zero")
        expert_input, residual = g.node(
            "AddRMSNorm", ["hidden_states", mixed, post_norm],
            outputs=["expert_input", "attention_residual"])
    else:
        residual = g.node("Add", ["hidden_states", mixed], "attention_residual_internal")
        g.node("Add", [residual, "output_zero"], "attention_residual")
        expert_input = g.node("RMSNorm", [residual, post_norm], "expert_input")
    add_common_moe(
        g, reader, external, layer, expert_input,
        fuse_independent_projections)
    outputs = [vi("attention_residual", [1, seq, HIDDEN]),
               vi("expert_input", [1, seq, HIDDEN]),
               vi("router_logits", [1, seq, 256]),
               vi("shared_expert_output", [1, seq, HIDDEN]),
               vi("present_key", [1, FULL_KV_HEADS, CACHE, FULL_DIM]),
               vi("present_value", [1, FULL_KV_HEADS, CACHE, FULL_DIM])]
    return inputs, outputs, {
        "cache_aliases": {"present_key": "past_key", "present_value": "past_value"},
        "outputs": [item.name for item in outputs],
        "fused_independent_projections": fuse_independent_projections,
        "fused_residual_rms": fuse_residual_rms,
    }


def rms(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    x = value.astype(np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(variance + EPSILON) *
            (np.float32(1.0) + weight.astype(np.float32))).astype(np.float16)


def linear(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.matmul(value.astype(np.float32), weight.astype(np.float32).T).astype(np.float16)


def silu(value: np.ndarray) -> np.ndarray:
    x = value.astype(np.float32)
    return (x / (np.float32(1.0) + np.exp(-x))).astype(np.float16)


def common_golden(reader: SafeTensorReader, layer: int, expert_input: np.ndarray) -> dict[str, np.ndarray]:
    """Compute shared router and expert reference outputs."""
    base = f"model.language_model.layers.{layer}.mlp"
    router = linear(expert_input, reader.load(f"{base}.gate.weight"))
    gate = linear(expert_input, reader.load(f"{base}.shared_expert.gate_proj.weight"))
    up = linear(expert_input, reader.load(f"{base}.shared_expert.up_proj.weight"))
    product = np.multiply(silu(gate), up, dtype=np.float16)
    down = linear(product, reader.load(f"{base}.shared_expert.down_proj.weight"))
    score = linear(expert_input, reader.load(f"{base}.shared_expert_gate.weight"))
    score = (np.float32(1.0) / (np.float32(1.0) + np.exp(-score.astype(np.float32)))).astype(np.float16)
    shared = np.multiply(down, score, dtype=np.float16)
    return {"router_logits": router, "shared_expert_output": shared}


def l2_fp16(value: np.ndarray) -> np.ndarray:
    square = np.multiply(value, value, dtype=np.float16)
    total = np.sum(square, axis=-1, keepdims=True, dtype=np.float16)
    return np.multiply(value, np.reciprocal(np.sqrt(total + np.float16(1.0e-6)),
                                             dtype=np.float16), dtype=np.float16)


def linear_golden(reader: SafeTensorReader, layer: int, sequence: int,
                  inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute reference outputs for the linear-attention layer."""
    base = f"model.language_model.layers.{layer}"
    attn = f"{base}.linear_attn"
    hidden = inputs["hidden_states"]
    x = rms(hidden, reader.load(f"{base}.input_layernorm.weight"))
    mixed = linear(x, reader.load(f"{attn}.in_proj_qkv.weight"))
    weight = reader.load(f"{attn}.conv1d.weight")[:, 0]
    state = inputs["past_conv_state"]
    combined = np.concatenate((state, mixed.transpose(0, 2, 1)), axis=2)
    present_conv = combined[:, :, sequence:sequence + 4].copy()
    convolved = np.empty_like(mixed)
    for token in range(sequence):
        window = combined[0, :, token + 1:token + 5].astype(np.float32)
        value = np.sum(window * weight.astype(np.float32), axis=1, dtype=np.float32)
        convolved[0, token] = (value / (1.0 + np.exp(-value))).astype(np.float16)
    q16, k16, v = np.split(convolved, [2048, 4096], axis=2)
    q = np.repeat(q16.reshape(1, sequence, 16, 128), 2, axis=2)
    k = np.repeat(k16.reshape(1, sequence, 16, 128), 2, axis=2)
    v = v.reshape(1, sequence, 32, 128)
    a = linear(x, reader.load(f"{attn}.in_proj_a.weight"))
    b = linear(x, reader.load(f"{attn}.in_proj_b.weight"))
    dt = reader.load(f"{attn}.dt_bias").astype(np.float32)
    a_log = reader.load(f"{attn}.A_log", "fp32")
    qf = l2_fp16(q).astype(np.float32) / np.sqrt(np.float32(128.0))
    kf = l2_fp16(k).astype(np.float32)
    vf = v.astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-b.astype(np.float32)))).astype(np.float16).astype(np.float32)
    at = a.astype(np.float32) + dt
    decay = -(np.maximum(at, 0.0) + np.log1p(np.exp(-np.abs(at)))) * np.exp(a_log)
    recurrent = inputs["past_recurrent_state_bits"].view(np.float32).reshape(1, 32, 128, 128).copy()
    core = np.empty((1, sequence, 32, 128), np.float16)
    for token in range(sequence):
        recurrent *= np.exp(decay[:, token], dtype=np.float32)[..., None, None]
        prediction = np.sum(recurrent * kf[:, token, :, :, None], axis=-2, dtype=np.float32)
        delta = (vf[:, token] - prediction) * beta[:, token, :, None]
        recurrent += kf[:, token, :, :, None] * delta[..., None, :]
        core[:, token] = np.sum(recurrent * qf[:, token, :, :, None], axis=-2,
                                dtype=np.float32).astype(np.float16)
    z = linear(x, reader.load(f"{attn}.in_proj_z.weight")).reshape(1, sequence, 32, 128)
    cf = core.astype(np.float32)
    variance = np.mean(cf * cf, axis=-1, keepdims=True, dtype=np.float32)
    norm_w = reader.load(f"{attn}.norm.weight").astype(np.float32)
    gated = z.astype(np.float32) / (1.0 + np.exp(-z.astype(np.float32)))
    core = (cf / np.sqrt(variance + EPSILON) * norm_w * gated).astype(np.float16)
    mixed = linear(core.reshape(1, sequence, 4096), reader.load(f"{attn}.out_proj.weight"))
    residual = np.add(hidden, mixed, dtype=np.float16)
    expert_input = rms(residual, reader.load(f"{base}.post_attention_layernorm.weight"))
    result = {"attention_residual": residual, "expert_input": expert_input,
              "present_conv_state": present_conv,
              "present_recurrent_state_bits": recurrent.view(np.float16).reshape(1, 32, 128, 256)}
    result.update(common_golden(reader, layer, expert_input))
    return result


def rotary_tables(sequence: int, valid: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(valid, valid + sequence, dtype=np.float32)
    inverse = np.float32(1.0) / np.power(
        np.float32(10_000_000.0),
        np.arange(0, ROTARY_DIM, 2, dtype=np.float32) / np.float32(ROTARY_DIM))
    frequency = positions[:, None] * inverse[None]
    table = np.concatenate((frequency, frequency), axis=-1)
    return np.cos(table)[None].astype(np.float16), np.sin(table)[None].astype(np.float16)


def apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    headed = value.transpose(0, 2, 1, 3)
    output = headed.copy()
    rotary = headed[..., :ROTARY_DIM]
    rotated = np.concatenate((-rotary[..., ROTARY_DIM // 2:],
                              rotary[..., :ROTARY_DIM // 2]), axis=-1)
    output[..., :ROTARY_DIM] = np.add(
        np.multiply(rotary, cos[:, None], dtype=np.float16),
        np.multiply(rotated, sin[:, None], dtype=np.float16), dtype=np.float16)
    return output


def full_golden(reader: SafeTensorReader, layer: int, sequence: int,
                inputs: dict[str, np.ndarray], valid: int) -> dict[str, np.ndarray]:
    """Compute reference outputs for the full-attention layer."""
    base = f"model.language_model.layers.{layer}"
    attn = f"{base}.self_attn"
    hidden = inputs["hidden_states"]
    x = rms(hidden, reader.load(f"{base}.input_layernorm.weight"))
    qgate = linear(x, reader.load(f"{attn}.q_proj.weight")).reshape(
        1, sequence, FULL_Q_HEADS, FULL_DIM * 2)
    q, gate = np.split(qgate, 2, axis=3)
    q = rms(q, reader.load(f"{attn}.q_norm.weight"))
    k = linear(x, reader.load(f"{attn}.k_proj.weight")).reshape(
        1, sequence, FULL_KV_HEADS, FULL_DIM)
    k = rms(k, reader.load(f"{attn}.k_norm.weight"))
    v = linear(x, reader.load(f"{attn}.v_proj.weight")).reshape(
        1, sequence, FULL_KV_HEADS, FULL_DIM)
    q = apply_rope(q, inputs["rope_cos"], inputs["rope_sin"])
    k = apply_rope(k, inputs["rope_cos"], inputs["rope_sin"])
    present_k = inputs["past_key"].copy()
    present_v = inputs["past_value"].copy()
    present_k[:, :, valid:valid + sequence] = k
    present_v[:, :, valid:valid + sequence] = v.transpose(0, 2, 1, 3)
    key_limit = ((valid + sequence + 127) // 128) * 128
    keys = np.repeat(present_k[:, :, :key_limit], 8, axis=1)
    values = np.repeat(present_v[:, :, :key_limit], 8, axis=1)
    scores = np.matmul(q.astype(np.float32), keys.transpose(0, 1, 3, 2).astype(np.float32))
    scores = scores.astype(np.float16).astype(np.float32) / np.float32(16.0)
    scores += inputs["kvcache_mask"][..., :key_limit].astype(np.float32)
    scores -= np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(scores, dtype=np.float32)
    probability = (probability / np.sum(probability, axis=-1, keepdims=True,
                                        dtype=np.float32)).astype(np.float16)
    context = np.matmul(probability.astype(np.float32), values.astype(np.float32)).astype(np.float16)
    context = context.transpose(0, 2, 1, 3)
    gate_sigmoid = (1.0 / (1.0 + np.exp(-gate.astype(np.float32)))).astype(np.float16)
    context = np.multiply(context, gate_sigmoid, dtype=np.float16).reshape(1, sequence, 4096)
    mixed = linear(context, reader.load(f"{attn}.o_proj.weight"))
    residual = np.add(hidden, mixed, dtype=np.float16)
    expert_input = rms(residual, reader.load(f"{base}.post_attention_layernorm.weight"))
    result = {"attention_residual": residual, "expert_input": expert_input,
              "present_key": present_k, "present_value": present_v}
    result.update(common_golden(reader, layer, expert_input))
    return result


def make_test_data(reader: SafeTensorReader, layer: int, layer_type: str,
                   sequence: int, output_dir: Path, manifest: dict) -> None:
    """Write graph inputs and reference outputs for layer validation."""
    rng = np.random.default_rng(355000 + layer * 100 + sequence)
    hidden = rng.normal(0.0, 0.25, (1, sequence, HIDDEN)).astype(np.float16)
    inputs: dict[str, np.ndarray] = {
        "hidden_states": hidden,
        "output_zero": np.zeros((1,), np.float16),
    }
    if layer_type == "linear_attention":
        inputs["past_conv_state"] = rng.normal(0.0, 0.04, (1, 8192, 4)).astype(np.float16)
        recurrent = rng.normal(0.0, 0.001, (1, 32, 128, 128)).astype(np.float32)
        inputs["past_recurrent_state_bits"] = recurrent.view(np.float16).reshape(1, 32, 128, 256)
        golden = linear_golden(reader, layer, sequence, inputs)
    else:
        valid = 63 if sequence == 1 else 0
        cos, sin = rotary_tables(sequence, valid)
        inputs["rope_cos"], inputs["rope_sin"] = cos, sin
        inputs["past_key"] = np.zeros((1, 2, CACHE, FULL_DIM), np.float16)
        inputs["past_value"] = np.zeros((1, 2, CACHE, FULL_DIM), np.float16)
        if valid:
            inputs["past_key"][:, :, :valid] = rng.normal(
                0.0, 0.08, (1, 2, valid, FULL_DIM)).astype(np.float16)
            inputs["past_value"][:, :, :valid] = rng.normal(
                0.0, 0.08, (1, 2, valid, FULL_DIM)).astype(np.float16)
        mask = np.full((1, 1, sequence, CACHE), np.float16(-65504.0), np.float16)
        for token in range(sequence):
            mask[:, :, token, :valid + token + 1] = np.float16(0.0)
        inputs["kvcache_mask"] = mask
        golden = full_golden(reader, layer, sequence, inputs, valid)
    input_files, output_specs = {}, {}
    for name, value in inputs.items():
        filename = f"input_{name}.bin"
        value.tofile(output_dir / filename)
        input_files[name] = filename
    tolerances = {
        "attention_residual": 0.12, "expert_input": 0.15,
        "router_logits": 0.18, "shared_expert_output": 0.12,
        "present_conv_state": 0.02, "present_recurrent_state_bits": 0.006,
        "present_key": 0.03, "present_value": 0.02,
    }
    for name, value in golden.items():
        filename = f"golden_{name}.bin"
        value.tofile(output_dir / filename)
        dtype = "float32_bits" if name == "present_recurrent_state_bits" else "float16"
        output_specs[name] = {"golden": filename, "dtype": dtype,
                              "atol": tolerances[name], "relative_l2": 0.04,
                              "cosine": 0.999}
    manifest["inputs"].update(input_files)
    manifest["output_specs"] = output_specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sequence", type=int, choices=(1, 64), required=True)
    parser.add_argument("--dynamic-sequence", action="store_true",
                        help="Export a dynamic sequence axis for an S1/S64 OMC")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-test-data", action="store_true")
    parser.add_argument(
        "--fuse-independent-projections", action="store_true",
        help="POC: concatenate independent FP16 projections and split outputs")
    parser.add_argument(
        "--fuse-residual-rms", action="store_true",
        help="Fuse the FP16 residual add and following RMSNorm")
    args = parser.parse_args()
    config = json.loads((args.model_dir / "config.json").read_text())["text_config"]
    if args.layer < 0 or args.layer >= int(config["num_hidden_layers"]):
        parser.error("layer index out of range")
    if args.dynamic_sequence and args.with_test_data:
        parser.error("--dynamic-sequence cannot generate fixed-shape test data")
    layer_type = config["layer_types"][args.layer]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.onnx"
    data_path = args.output_dir / "resident_weights.bin"
    reader = SafeTensorReader(args.model_dir)
    external = ExternalWeights(data_path)
    graph = Graph(external)
    try:
        if layer_type == "linear_attention":
            inputs, outputs, metadata = build_linear(
                graph, reader, external, args.layer, args.sequence,
                args.dynamic_sequence, args.fuse_independent_projections,
                args.fuse_residual_rms)
        elif layer_type == "full_attention":
            inputs, outputs, metadata = build_full(
                graph, reader, external, args.layer, args.sequence,
                args.dynamic_sequence, args.fuse_independent_projections,
                args.fuse_residual_rms)
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")
    finally:
        external.close()
    model_graph = helper.make_graph(
        graph.nodes, f"Qwen35MoEResidentLayer{args.layer}",
        [*inputs, *external.inputs], outputs, initializer=graph.initializers)
    model = helper.make_model(
        model_graph, producer_name="HMOS-Device-AI/qwen3.5-35b-a3b",
        opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    model.doc_string = "Resident attention/router/shared-expert layer; routed GPTQ experts are paged by the runtime."
    onnx.save_model(model, model_path)
    manifest = {
        "schema_version": 1, "layer": args.layer, "layer_type": layer_type,
        "sequence": "dynamic" if args.dynamic_sequence else args.sequence,
        "gears": [1, 64] if args.dynamic_sequence else [args.sequence],
        "onnx": model_path.name, "model": "model.omc",
        "weight_file": data_path.name, "weight_bytes": data_path.stat().st_size,
        "weight_layout": external.layout, "inputs": dict(external.input_files),
        **metadata,
    }
    if args.with_test_data:
        make_test_data(reader, args.layer, layer_type, args.sequence,
                       args.output_dir, manifest)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"ONNX: {model_path}")
    print(f"weights: {data_path} ({data_path.stat().st_size / 1024 / 1024:.2f} MiB)")
    print(f"layer={args.layer} type={layer_type} sequence={args.sequence}")


if __name__ == "__main__":
    main()
