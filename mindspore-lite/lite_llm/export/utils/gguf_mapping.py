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
"""GGUF tensor loading, including private Q3_0 and compact phase4 Q4_0 packing."""

import logging
from typing import Optional

import numpy as np
import onnx
from gguf.quants import GGMLQuantizationType, dequantize

from mslite_llm_quant.q3_0 import BLOCK_BYTES as Q3_0_BLOCK_BYTES, BLOCK_SIZE as Q3_0_BLOCK_SIZE
from mslite_llm_quant.q3_0 import GGUF_TYPE as PRIVATE_Q3_0_TYPE, register_private_q3_0_type
from utils import load_custom_op

from utils.quantization import QuantType

logger = logging.getLogger(__name__)

LEGACY_Q3_TYPE1024 = 1024
LEGACY_Q3_TYPE1024_ERROR = ("legacy private type-1024 Q3 GGUF is no longer supported; "
                            "re-quantize from HF with mslite-llm-quantize-gguf --quant-type q3_0")


def open_gguf_reader(path):
    """Register the private format before parsing, leaving parse errors intact."""
    from gguf import GGUFReader

    register_private_q3_0_type()
    if not isinstance(GGUFReader, type):
        return GGUFReader(path)

    class CheckedReader(GGUFReader):
        """Reject legacy decoder records before gguf resolves their type."""

        def _build_tensors(self, start_offs, fields):
            for field in fields:
                name = bytes(field.parts[1]).decode("utf-8")
                if (name.startswith("blk.") and name.endswith(".weight")
                        and not name.endswith("_norm.weight") and int(field.parts[4][0]) == LEGACY_Q3_TYPE1024):
                    raise ValueError(LEGACY_Q3_TYPE1024_ERROR)
            return super()._build_tensors(start_offs, fields)

    return CheckedReader(path)


def detect_gguf_quantization(reader):
    """Detect uniform Q3_0/Q4_0 decoder weights; embedding stays Q4_0."""
    decoder_types = {
        int(tensor.tensor_type) for tensor in reader.tensors
        if tensor.name.startswith("blk.") and tensor.name.endswith(".weight")
        and not tensor.name.endswith("_norm.weight")
    }
    if LEGACY_Q3_TYPE1024 in decoder_types:
        raise ValueError(LEGACY_Q3_TYPE1024_ERROR)
    embeddings = [tensor for tensor in reader.tensors if tensor.name == "token_embd.weight"]
    if decoder_types == {PRIVATE_Q3_0_TYPE}:
        if len(embeddings) != 1 or int(embeddings[0].tensor_type) != int(GGMLQuantizationType.Q4_0):
            raise ValueError("Q3_0 requires Q4_0 token_embd.weight")
        return QuantType.Q3_0, QuantType.Q4_0
    if decoder_types == {int(GGMLQuantizationType.Q4_0)}:
        return QuantType.Q4_0, QuantType.Q4_0
    raise ValueError("GGUF decoder projection weights must all use Q4_0 or private Q3_0")


def rearrange_q3_0_g32(data):
    """Repack GGUF canonical blocks via MsQuant3N0Group32.offline_reorder."""
    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        raise ValueError("GGUF Q3_0 rows must be a UINT8 ndarray")
    if data.ndim != 2 or data.shape[1] % Q3_0_BLOCK_BYTES:
        raise ValueError("Expected GGUF Q3_0 rows [N, K/32 * 14]")
    shape = [data.shape[1] // Q3_0_BLOCK_BYTES * Q3_0_BLOCK_SIZE, data.shape[0]]
    flat = np.ascontiguousarray(data).reshape(-1)
    adapter = load_custom_op("MsQuant3N0Group32")
    _, blob = adapter.offline_reorder([None, flat], shape)
    return np.ascontiguousarray(blob.detach().cpu().numpy() if hasattr(blob, "detach") else blob)


def create_new_initializer(name, weight):
    """Create an ONNX initializer tensor from a numpy weight array."""
    return onnx.numpy_helper.from_array(weight, name)


def rearrange_q4_0_g32(data):
    """Repack GGUF rows into signed int4 NZF without requantizing their scales.

    Each input row holds K/32 blocks of fp16 scale + 16 split-nibble bytes.
    GGUF represents weights as (unsigned_q - 8) * scale; NZF stores the
    two's-complement int4 value and the original per-row fp16 scale bits.
    """
    quant_op_class = load_custom_op("MsQuant4N0Group32")

    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        raise ValueError("GGUF Q4_0 rows must be a UINT8 ndarray")
    if data.ndim != 2 or data.shape[1] % 18:
        raise ValueError("Expected GGUF Q4_0 rows [N, K/32 * 18]")
    return quant_op_class.repack_q4_0_to_nzf(data, (data.shape[1] // 18 * 32, data.shape[0]))


def convert_embedding_weight(data, tensor_type, embedding_quantize_config: Optional[QuantType]):
    """Convert a GGUF embedding tensor to the selected runtime representation.

    Q4_0 already has the same group-32 quantization contract as W4A16 and only
    needs its blocks rearranged. Other GGUF formats must first be dequantized.
    """
    if embedding_quantize_config == QuantType.Q4_0:
        if tensor_type == GGMLQuantizationType.Q4_0:
            return rearrange_q4_0_g32(data)
        quant_op_class = load_custom_op("MsQuant4N0Group32")

        fp32 = dequantize(data, tensor_type)
        return quant_op_class.quantize_weight_g32_4bit(fp32.T)

    if embedding_quantize_config is None:
        if tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
            return data.astype(np.float16)
        return dequantize(data, tensor_type).astype(np.float16)

    raise ValueError(
        f"embedding_quantize_config {embedding_quantize_config} not supported (q4_0/FP16)"
    )


def load_file_from_tensors(tensors, embedding_weight_save_path,
                           decoder_quantize_config: Optional[QuantType],
                           embedding_quantize_config: Optional[QuantType]):
    """Read GGUF tensors using quantization types normalized by the model loader."""
    name2weight = {}
    for tensor_item in tensors:
        name = tensor_item.name
        # fp16 layer norms: attn_norm / ffn_norm / per-head q_norm / k_norm.
        # output_norm.weight is handled separately below (F32 as well).
        if name.endswith("bias") or (name.endswith("_norm.weight") and name != "output_norm.weight"):
            name2weight[name] = tensor_item.data.astype(np.float16)
        elif name == "output.weight":
            # The skeleton is tied (apply_shared_weight): the lm_head consumes
            # embedding_weight == token_embd.weight.  Ignore the separate Q8_0
            # output tensor that community GGUFs emit for higher-precision logits.
            logger.info("Skipping %s (tied skeleton uses token_embd.weight for lm_head)", name)
        elif "weight" in name and name not in ("token_embd.weight", "output_norm.weight"):
            if decoder_quantize_config in (QuantType.Q4_0, QuantType.Q3_0):
                expected = PRIVATE_Q3_0_TYPE if decoder_quantize_config == QuantType.Q3_0 else 2
                if int(tensor_item.tensor_type) != expected:
                    raise ValueError(f"GGUF tensor {name} does not use {decoder_quantize_config}")
                repack = rearrange_q3_0_g32 if decoder_quantize_config == QuantType.Q3_0 else rearrange_q4_0_g32
                name2weight[name] = repack(tensor_item.data)
            elif decoder_quantize_config is None:
                name2weight[name] = tensor_item.data
            else:
                raise ValueError(f"decoder_quantize_config {decoder_quantize_config} not supported")
        elif name == "token_embd.weight":
            name2weight[name] = convert_embedding_weight(
                tensor_item.data, tensor_item.tensor_type, embedding_quantize_config
            )
            name2weight[name].tofile(embedding_weight_save_path)
        elif name == "output_norm.weight":
            name2weight[name] = tensor_item.data.astype(np.float16)
        else:
            raise ValueError(f"Unexpected GGUF tensor {name}")
    return name2weight
