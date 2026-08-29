# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Export quantized model weights for the NNRT custom-op contract.

Merges the former ``quantize`` / ``packing`` / ``quant_config`` modules:

* ``QuantizationConfig`` / ``ModelConfig`` / ``LiteTurboConfig`` — quantization
  and model-shape configuration consumed by the exporters.
* Pure-Python (NumPy) weight packing kernels (bit-exact ports of the C++
  reference kernels for the Kirin NPU ``MsQuant*`` custom operators).
* ``apply_quant`` / ``quantize_linear_ops`` — graph-level quantization
  (weights + tied lm_head) producing the NNRT-compatible ONNX.
"""

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

from utils.onnx_postprocess import duplicate_shared_initializers


# ─── quantization configuration ────────────────────────────────────────────

class QuantizationConfig:
    """Quantization method configuration for embedding or decoder layers."""

    def __init__(self, quant_method):
        if quant_method is None:
            self.is_quant = False
            self.bits = 0
            self.group_size = 0
            return

        if quant_method not in ["W4A8", "W2A16", "W4A16"]:
            raise ValueError(
                f"`quant_method` should be one of [W4A8, W2A16, W4A16], but is {quant_method}"
            )

        self.is_quant = True
        self.quant_method = quant_method
        if quant_method == "W4A8":
            self.bits = 4
            self.group_size = 128
        elif quant_method == "W2A16":
            self.bits = 2
            self.group_size = 32
        elif quant_method == "W4A16":
            self.bits = 4
            self.group_size = 32

    def asdict(self):
        if self.is_quant:
            return {
                "quant_method": self.quant_method,
                "bits": self.bits,
                "group_size": self.group_size,
            }
        return {}

    def __repr__(self):
        config = f"<is_quant: {self.is_quant}"
        if hasattr(self, "quant_method"):
            config += (
                f", quant_method: {self.quant_method}, bits: {self.bits}, "
                f"group_size: {self.group_size}"
            )
        config += ">"
        return config


@dataclass
class ModelConfig:
    """Model shape + quantization context consumed by the quantizer."""

    max_length: int
    chunk_size: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    eos_id: int
    embedding_quant: QuantizationConfig
    decoder_quant: QuantizationConfig


@dataclass
class LiteTurboConfig:
    """Sampling / runtime defaults emitted alongside the exported graph.

    The current NNRT executor is greedy (argmax on device); these fields are kept
    for future sampling work and for parity with the reference exporter.
    """

    max_length: int
    chunk_size: int
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    eos_id: int
    scale_gp_size: int = 32
    embedding_quant: bool = False
    do_sample: bool = True
    temperature: float = 0.3
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 1.0
    diversity_penalty: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    random_seed: int = 42

    def asdict(self):
        return {
            "max_length": self.max_length,
            "chunk_size": self.chunk_size,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "eos_id": self.eos_id,
            "scale_gp_size": self.scale_gp_size,
            "embedding_quant": self.embedding_quant,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "typical_p": self.typical_p,
            "diversity_penalty": self.diversity_penalty,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "random_seed": self.random_seed,
        }


# ─── weight packing kernels (NumPy, bit-exact ports of the C++ reference) ──

def _fp32_to_fp16(f):
    """Faithful port of ggml's ``ggml_compute_fp32_to_fp16`` (round-to-nearest-even).

    We reproduce the exact bit manipulation rather than relying on numpy's
    ``float16`` cast so the scale bytes stay bit-identical to the C++ reference
    for all inputs, including subnormals and signed zero.
    """
    f = np.ascontiguousarray(f, dtype=np.float32)
    w = f.view(np.uint32)

    scale_to_inf = np.array([0x77800000], dtype=np.uint32).view(np.float32)[0]  # 0x1.0p+112
    scale_to_zero = np.array([0x08800000], dtype=np.uint32).view(np.float32)[0]  # 0x1.0p-110

    base = (np.abs(f) * scale_to_inf) * scale_to_zero
    base = base.astype(np.float32, copy=False)

    shl1_w = (w + w).astype(np.uint32)
    sign = w & np.uint32(0x80000000)
    bias = shl1_w & np.uint32(0xFF000000)
    bias = np.where(bias < np.uint32(0x71000000), np.uint32(0x71000000), bias).astype(np.uint32)

    tmp_bits = ((bias >> np.uint32(1)) + np.uint32(0x07800000)).astype(np.uint32)
    base = (tmp_bits.view(np.float32) + base).astype(np.float32, copy=False)

    bits = base.view(np.uint32)
    exp_bits = (bits >> np.uint32(13)) & np.uint32(0x00007C00)
    mantissa_bits = bits & np.uint32(0x00000FFF)
    nonsign = (exp_bits + mantissa_bits).astype(np.uint32)

    result = np.bitwise_or(
        sign >> np.uint32(16),
        np.where(shl1_w > np.uint32(0xFF000000), np.uint32(0x7E00), nonsign).astype(np.uint32),
    )
    return result.astype(np.uint16)


def _signed_max_per_group(xg):
    """Value with the largest absolute magnitude per group (first on ties).

    Mirrors the C++ reduction (``max`` is the signed value whose ``fabsf`` is
    strictly largest; all-zero groups keep the initial ``+0.0f``).
    """
    abs_x = np.abs(xg)
    idx = np.argmax(abs_x, axis=-1)
    max_vals = np.take_along_axis(xg, idx[..., None], axis=-1)[..., 0]
    return np.where(np.abs(max_vals) != 0, max_vals, np.float32(0.0))


def _inverse_scale(max_vals, divisor):
    """Compute ``scale = max / -divisor`` and ``id = scale ? 1/scale : 0``."""
    scale = (max_vals / np.float32(-divisor)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.float32(1.0) / scale
    id_ = np.where(scale != 0, inv, np.float32(0.0)).astype(np.float32)
    return scale, id_


# ---------------------------------------------------------------------------
# W4A16: MsQuant4N0Group32 (group 32, planar "V1" layout, fp16 scales)
# ---------------------------------------------------------------------------
def _q4_n0_v1(x, n, d):
    """Port of ``quantize_Q4_N_0_V1_reference``.

    ``x`` is the transposed weight ``[N, K]`` flattened row-major; ``n=N``,
    ``d=K``. Returns the packed ``uint8`` buffer: ``n*d/2`` weight bytes
    followed by ``n*d/32`` fp16 scales (``n*d/16`` bytes).
    """
    qk = 32
    nb = n * d // qk
    xg = x.reshape(nb, qk)

    max_vals = _signed_max_per_group(xg)
    scale, id_ = _inverse_scale(max_vals, 8)

    x0 = xg[:, :16] * id_[:, None]
    x1 = xg[:, 16:] * id_[:, None]
    xi0 = np.minimum(15, (x0 + np.float32(8.5)).astype(np.int64))
    xi1 = np.minimum(15, (x1 + np.float32(8.5)).astype(np.int64))

    packed = (xi0.astype(np.uint8) & 0x0F) | ((xi1.astype(np.uint8) & 0x0F) << 4)
    weight_bytes = packed.reshape(-1)

    scale_bytes = _fp32_to_fp16(scale).view(np.uint8).reshape(-1)
    return np.concatenate([weight_bytes, scale_bytes])


# ---------------------------------------------------------------------------
# W4A8: MsQuant4N0Group128 (group 128, NZ fractal layout, fp32 scales)
# ---------------------------------------------------------------------------
def _q4_n0_nz(x, n, d):
    """Port of ``quantize_row_q4_n_0_nz_reference``.

    ``x`` is the transposed weight ``[N, K]`` flattened; ``n=N``, ``d=K``.
    Returns the packed ``uint8`` buffer: weight bytes in Ascend NZ fractal
    order followed by fp32 scales.
    """
    qk = 128
    n_factor_q = 64
    fractal_n = 16
    fractal_d = 32
    nb_row = d // qk

    aligned_n = (n + fractal_n - 1) // fractal_n * fractal_n

    xg = x.reshape(n, nb_row, qk)
    max_vals = _signed_max_per_group(xg)  # [n, nb_row]
    scale, id_ = _inverse_scale(max_vals, 8)  # [n, nb_row]

    x_scaled = xg * id_[:, :, None]  # [n, nb_row, 128]
    x_even = x_scaled[..., 0::2]  # [n, nb_row, 64]
    x_odd = x_scaled[..., 1::2]  # [n, nb_row, 64]

    xi0 = np.minimum(15, (x_even + np.float32(8.5)).astype(np.int64)) - 8
    xi1 = np.minimum(15, (x_odd + np.float32(8.5)).astype(np.int64)) - 8
    nib0 = xi0.astype(np.uint8) & 0x0F
    nib1 = xi1.astype(np.uint8) & 0x0F
    packed = nib0 | (nib1 << 4)  # [n, nb_row, 64] -> [n, d//2]

    # NZ fractal: for each 64-row block, layout rows as [d/64, aligned_rows, 32].
    weight_blocks = []
    for nr_n in range(0, n, n_factor_q):
        nr_loop = min(n_factor_q, n - nr_n)
        aligned_nr = (nr_loop + fractal_n - 1) // fractal_n * fractal_n
        block_rows = packed[nr_n:nr_n + nr_loop].reshape(nr_loop, d // 2)
        if nr_loop < aligned_nr:
            block_rows = np.concatenate(
                [block_rows, np.zeros((aligned_nr - nr_loop, d // 2), dtype=np.uint8)], axis=0
            )
        block = block_rows.reshape(aligned_nr, d // 64, fractal_d).transpose(1, 0, 2)
        weight_blocks.append(block.reshape(-1))

    weight_bytes = np.concatenate(weight_blocks) if weight_blocks else np.zeros(0, dtype=np.uint8)

    nb = aligned_n * nb_row
    scale_flat = scale.astype(np.float32).T.reshape(-1)  # [nb_row, n] -> j-major
    if scale_flat.size < nb:
        scale_flat = np.concatenate([scale_flat, np.zeros(nb - scale_flat.size, dtype=np.float32)])
    scale_bytes = scale_flat.view(np.uint8).reshape(-1)

    return np.concatenate([weight_bytes, scale_bytes])


# ---------------------------------------------------------------------------
# W2A16: MsQuant2N0Group32 (group 32, fp16 scales)
# ---------------------------------------------------------------------------
def _q2_n0(x, n, d):
    """Port of ``quantize_Q2_N_0_reference``.

    ``x`` is the transposed weight ``[N, K]`` flattened; ``n=N``, ``d=K``.
    Returns the packed ``uint8`` buffer: ``n*d/4`` weight bytes followed by
    ``n*d/32`` fp16 scales (``n*d/16`` bytes).
    """
    qk = 32
    nb = n * d // qk
    xg = x.reshape(nb, qk)

    max_vals = _signed_max_per_group(xg)
    scale, id_ = _inverse_scale(max_vals, 2)

    xs = (xg * id_[:, None]).reshape(nb, 4, 8)  # element z*8+j
    xii = np.minimum(3, (xs + np.float32(2.5)).astype(np.int64)).astype(np.uint8)  # [nb, 4, 8]

    packed = xii[:, 0, :] | (xii[:, 1, :] << 2) | (xii[:, 2, :] << 4) | (xii[:, 3, :] << 6)
    weight_bytes = packed.reshape(-1)  # [nb, 8] -> nb*8

    scale_bytes = _fp32_to_fp16(scale).view(np.uint8).reshape(-1)
    return np.concatenate([weight_bytes, scale_bytes])


# ---------------------------------------------------------------------------
# Public wrappers: identical signatures/return dtypes to the original ctypes
# wrappers, minus the .so loading.
# ---------------------------------------------------------------------------
def _ceil(x, y):
    return (x + y - 1) // y * y


def quantize_weight_g32_4bit_nd(weight):
    """W4A16 weight quantize, 4 bits, symmetric, group size=32 (planar layout)."""
    k, n = weight.shape  # [K, N]
    x = weight.T.astype(np.float32).reshape(-1)  # [N, K] flattened
    out = _q4_n0_v1(x, n, k)
    length = _ceil(n, 16) * (k // 2 + k // 32 * 2)
    return _pad_to_length(out, length).view(np.uint8)


def quantize_weight_g128_4bit_nz(weight):
    """W4A8 weight quantize, 4 bits, symmetric, group size=128."""
    k, n = weight.shape  # [K, N]
    x = weight.T.astype(np.float32).reshape(-1)  # [N, K] flattened
    out = _q4_n0_nz(x, n, k)
    length = _ceil(n, 16) * (k // 2 + k // 128 * 4)
    return _pad_to_length(out, length).view(np.uint8)


def quantize_weight_g32_2bit_nd(weight):
    """W2A16 weight quantization, group size=32."""
    k, n = weight.shape  # [K, N]
    x = weight.T.astype(np.float32).reshape(-1)  # [N, K] flattened
    out = _q2_n0(x, n, k)
    length = _ceil(n, 16) * (k // 4 + k // 32 * 2)
    return _pad_to_length(out, length)


def _pad_to_length(out, length):
    if out.size < length:
        out = np.concatenate([out, np.zeros(length - out.size, dtype=np.uint8)])
    return out


# ─── graph-level quantization ──────────────────────────────────────────────

def custom_op_infer_shape(graph, chunk_size, max_seq_len, num_kv_heads, num_q_heads, dim, is_prefill):
    """Infer shapes for the custom operators before quantization."""
    graph_output = {vi.name: vi for vi in graph.output}

    for node in graph.node:
        if node.op_type == "MsScatterND":
            if node.attribute[0].name == "layout" and node.attribute[0].s.lower() == "bsnd":
                shape = [1, max_seq_len, num_kv_heads, dim]
            else:
                shape = [1, num_kv_heads, max_seq_len, dim]
            value_info = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT16, shape)
            if node.output[0] in graph_output:
                output = graph_output[node.output[0]]
                output.type.tensor_type.shape.CopyFrom(value_info.type.tensor_type.shape)
            else:
                graph.value_info.append(value_info)
        elif node.op_type in ["MsRmsNorm", "MsAddRmsNorm"]:
            shape = [1, chunk_size, num_q_heads * dim] if is_prefill else [1, 1, num_q_heads * dim]
            for _, output_name in enumerate(node.output):
                value_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT16, shape)
                graph.value_info.append(value_info)
        elif node.op_type == "MsRotaryPosEmb":
            rope_q_shape = [1, num_q_heads, chunk_size, dim] if is_prefill else [1, num_q_heads, 1, dim]
            rope_k_shape = [1, num_kv_heads, chunk_size, dim] if is_prefill else [1, num_kv_heads, 1, dim]

            rope_q_value_info = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT16, rope_q_shape)
            rope_k_value_info = helper.make_tensor_value_info(node.output[1], TensorProto.FLOAT16, rope_k_shape)
            graph.value_info.append(rope_q_value_info)
            graph.value_info.append(rope_k_value_info)
        elif node.op_type == "MsGroupMatmul":
            if node.attribute[0].name == "trans_b" and node.attribute[0].s.lower() == "true":
                shape = [1, num_q_heads, chunk_size, max_seq_len] if is_prefill else [1, num_q_heads, 1, max_seq_len]
            else:
                shape = [1, num_q_heads, chunk_size, dim] if is_prefill else [1, num_q_heads, 1, dim]
            value_info = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT16, shape)
            graph.value_info.append(value_info)


def load_q2_constant():
    return np.array(
        [
            0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
            32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55,
            64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87,
            96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119,
            8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31,
            40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63,
            72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95,
            104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127,
        ],
        dtype=np.uint8,
    )


def get_shape_info(graph):
    """Collect shape info from graph value_info."""
    shape_info = {}
    for value_info in graph.value_info:
        tensor_type = value_info.type.tensor_type
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(dim.dim_param)
        if len(shape) != 0:
            shape_info[value_info.name] = shape
    return shape_info


def quant_node_4bit_gp32(shape_info, origin_node, initializers):
    """W4A16 quantization (MsQuant4N0Group32)."""
    new_node_list = []
    new_initializer_list = []
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape
    quantized_weight = quantize_weight_g32_4bit_nd(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(quantized_weight, quant_weight_name)
    new_initializer_list.append(quant_weight_init)

    input_name = origin_node.input[0]
    if input_name not in shape_info:
        raise ValueError(f"Can not get input shape of {origin_node.name}.")

    quant_linear_node = helper.make_node(
        "MsQuant4N0Group32",
        inputs=[input_name, quant_weight_name],
        outputs=[origin_node.output[0]],
        input1_shape=f"{weight_shape[0]},{weight_shape[1]}",
        name=origin_node.name + "_quant",
        domain="custom",
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def quant_node_4bit(shape_info, origin_node, initializers):
    """W4A8 quantization (MsQuant4N0Group128)."""
    new_node_list = []
    new_initializer_list = []
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape
    quantized_weight = quantize_weight_g128_4bit_nz(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(quantized_weight, quant_weight_name)
    new_initializer_list.append(quant_weight_init)

    input_name = origin_node.input[0]
    quant_act_name = input_name + "_quant"
    quant_act_node = helper.make_node(
        "MsFloatCastInt", inputs=[input_name], outputs=[quant_act_name], name=quant_act_name, domain="custom"
    )
    new_node_list.append(quant_act_node)

    if input_name not in shape_info:
        raise ValueError(f"Can not get input shape of {origin_node.name}.")

    quant_linear_node = helper.make_node(
        "MsQuant4N0Group128",
        inputs=[quant_act_name, quant_weight_name],
        outputs=[origin_node.output[0]],
        input1_shape=f"{weight_shape[0]},{weight_shape[1]}",
        name=origin_node.name + "_quant",
        domain="custom",
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def quant_node_2bit(shape_info, origin_node, initializers, q2_v):
    """W2A16 quantization (MsQuant2N0Group32)."""
    new_node_list = []
    new_initializer_list = []
    weight_name = origin_node.input[1]
    weight_init = initializers[weight_name]
    weight_data = onnx.numpy_helper.to_array(weight_init)
    weight_shape = weight_data.shape

    quantized_weight = quantize_weight_g32_2bit_nd(weight_data)
    quant_weight_name = weight_name + "_quant"
    quant_weight_init = onnx.numpy_helper.from_array(
        np.frombuffer(quantized_weight.tobytes() + q2_v.tobytes(), np.uint8), quant_weight_name
    )
    new_initializer_list.append(quant_weight_init)

    input_name = origin_node.input[0]
    if input_name not in shape_info:
        raise ValueError(f"Can not get input shape of {origin_node.name}.")

    quant_linear_node = helper.make_node(
        "MsQuant2N0Group32",
        inputs=[origin_node.input[0], quant_weight_name],
        outputs=[origin_node.output[0]],
        input1_shape=f"{weight_shape[0]},{weight_shape[1]}",
        name=origin_node.name + "_quant",
        domain="custom",
    )
    new_node_list.append(quant_linear_node)
    return new_node_list, new_initializer_list


def _find_lmhead_matmul(graph):
    """Locate the lm_head MatMul node.

    Matches by node name first (torch export naming ``/lm_head/MatMul``, and the
    quantized ``MsQuant*`` variant keeps that name with a ``_quant`` suffix), then
    by the weight initializer name containing ``lm_head`` (robust to renaming).
    """
    initializer_names = {init.name for init in graph.initializer}
    lmhead_quant_ops = {"MsQuant4N0Group32", "MsQuant4N0Group128", "MsQuant2N0Group32"}
    for node in graph.node:
        if node.op_type not in ["MatMul", *lmhead_quant_ops]:
            continue
        if "/lm_head/MatMul" in node.name or "lm_head" in node.name:
            return node
        if len(node.input) > 1 and node.input[1] in initializer_names and "lm_head" in node.input[1]:
            return node
    return None


def apply_shared_weight(model, is_quant=False):
    """Replace the tied lm_head weight with the ``embedding_weight`` graph input.

    Inserts the input at index 6 so the final input order is the NNRT contract:
    [valid_seq_len, lmhead_idx, rope_cos, rope_sin, inputs_embeds, attention_mask,
    embedding_weight] + past_key_i/past_val_i.
    """
    graph = model.graph
    node = _find_lmhead_matmul(graph)
    if node is None:
        raise ValueError("lm_head MatMul node not found; cannot apply shared embedding weight")

    weight_name = node.input[1]
    for init in graph.initializer:
        if init.name != weight_name:
            continue

        weight_data = onnx.numpy_helper.to_array(init)

        if is_quant:
            embedding_input = helper.make_tensor_value_info("embedding_weight", TensorProto.UINT8, weight_data.shape)
        else:
            embedding_input = helper.make_tensor_value_info(
                "embedding_weight", TensorProto.FLOAT16, weight_data.T.shape
            )

        node_input = "embedding_weight"
        if not is_quant:
            transpose_node = helper.make_node(
                "Transpose",
                inputs=["embedding_weight"],
                outputs=["embedding_weight_transpose"],
                perm=[1, 0],
                name="embedding_weight_transpose",
            )
            node_input = "embedding_weight_transpose"
            graph.node.append(transpose_node)

        node.input[1] = node_input
        graph.input.insert(6, embedding_input)
        graph.initializer.remove(init)
        return
    raise ValueError(f"lm_head weight initializer '{weight_name}' not found")


def quantize_linear_ops(model, embedding_quant_config, decoder_quant_config):
    """Quantize decoder Linear weights and the lm_head (tied embedding)."""
    graph = model.graph

    shape_info = get_shape_info(graph)

    new_initializers = []
    new_nodes = []

    linear_nodes = []
    lmhead_node = _find_lmhead_matmul(graph)
    for node in graph.node:
        if node.op_type not in ["MatMul"] or node is lmhead_node:
            continue
        linear_nodes.append(node)

    initializers = {init.name: init for init in graph.initializer}
    q2_v = None
    if getattr(embedding_quant_config, "quant_method", None) == "W2A16" or getattr(
        decoder_quant_config, "quant_method", None
    ) == "W2A16":
        q2_v = load_q2_constant()

    if lmhead_node is None:
        raise ValueError("lm_head MatMul node not found for quantization")

    if embedding_quant_config.is_quant:
        if embedding_quant_config.quant_method == "W4A8":
            new_node_list, new_initializer_list = quant_node_4bit(shape_info, lmhead_node, initializers)
        elif embedding_quant_config.quant_method == "W2A16":
            new_node_list, new_initializer_list = quant_node_2bit(shape_info, lmhead_node, initializers, q2_v)
        elif embedding_quant_config.quant_method == "W4A16":
            new_node_list, new_initializer_list = quant_node_4bit_gp32(shape_info, lmhead_node, initializers)
        else:
            raise RuntimeError(f"quant method: {embedding_quant_config.quant_method} not supported")
        new_nodes.extend(new_node_list)
        new_initializers.extend(new_initializer_list)

    for node in linear_nodes:
        if node.input[1] not in initializers:
            # Dynamic MatMul (e.g. attention score products): keep as-is.
            new_nodes.append(node)
            continue
        if decoder_quant_config.quant_method == "W4A8":
            new_node_list, new_initializer_list = quant_node_4bit(shape_info, node, initializers)
        elif decoder_quant_config.quant_method == "W2A16":
            new_node_list, new_initializer_list = quant_node_2bit(shape_info, node, initializers, q2_v)
        elif decoder_quant_config.quant_method == "W4A16":
            new_node_list, new_initializer_list = quant_node_4bit_gp32(shape_info, node, initializers)
        else:
            raise RuntimeError(f"quant method: {decoder_quant_config.quant_method} not supported")
        new_nodes.extend(new_node_list)
        new_initializers.extend(new_initializer_list)

    linear_ids = {id(n) for n in linear_nodes}
    lmhead_id = id(lmhead_node) if lmhead_node is not None else None
    for node in graph.node:
        if id(node) in linear_ids:
            continue
        # The original lm_head MatMul is replaced by its MsQuant node when quantized.
        if lmhead_id is not None and id(node) == lmhead_id and embedding_quant_config.is_quant:
            continue
        new_nodes.append(node)

    for ori_init in graph.initializer:
        is_quanted = any(ori_init.name + "_quant" == new_init.name for new_init in new_initializers)
        if not is_quanted:
            new_initializers.append(ori_init)

    new_graph = helper.make_graph(new_nodes, graph.name, graph.input, graph.output, new_initializers)
    new_model = helper.make_model(new_graph, producer_name=model.producer_name)
    new_model.opset_import[0].version = 18

    apply_shared_weight(new_model, is_quant=embedding_quant_config.is_quant)

    # Remove redundant MsFloatCastInt nodes and duplicate shared initializers.
    from onnxslim import slim

    new_model = slim(new_model)
    duplicate_shared_initializers(new_model)
    return new_model


def infer_shape(model, chunk_size, max_seq_len, num_kv_heads, num_q_heads, dim, is_prefill=True):
    custom_op_infer_shape(model.graph, chunk_size, max_seq_len, num_kv_heads, num_q_heads, dim, is_prefill)
    model = shape_inference.infer_shapes(model)
    return model


def apply_quant(input_model, output_model, model_config: ModelConfig):
    """Quantize the exported model in place (weights + lm_head) and save."""
    model = onnx.load(input_model)

    model = infer_shape(
        model,
        model_config.chunk_size,
        model_config.max_length,
        model_config.num_attention_heads,
        model_config.num_key_value_heads,
        model_config.hidden_size // model_config.num_key_value_heads,
    )

    model = quantize_linear_ops(model, model_config.embedding_quant, model_config.decoder_quant)

    onnx.save(model, output_model)
    return output_model
