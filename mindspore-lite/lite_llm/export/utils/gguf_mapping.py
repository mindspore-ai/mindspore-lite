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
"""GGUF Q4_0 / FP16 tensor helpers shared by model weight injectors.

Model-agnostic: block rearrangement for the ``MsQuant4N0Group32`` g32 NZF
layout, initializer creation, and ``load_file_from_tensors`` which reads the
raw GGUF tensors into a name->weight map (biases/norms as fp16, quantized
weights rearranged, embedding saved separately).

Example:
    from utils.gguf_mapping import load_file_from_tensors, rearrange_q4_0_g32
"""

import logging
from typing import Optional

import numpy as np
import onnx
from gguf.quants import GGMLQuantizationType, dequantize

from utils import ensure_custom_ops

from utils.quantization import QuantType

logger = logging.getLogger(__name__)


def create_new_initializer(name, weight):
    """Create an ONNX initializer tensor from a numpy weight array."""
    return onnx.numpy_helper.from_array(weight, name)


def rearrange_q4_0_g32(data):
    """Repack GGUF rows into signed int4 NZF without requantizing their scales.

    Each input row holds K/32 blocks of fp16 scale + 16 split-nibble bytes.
    GGUF represents weights as (unsigned_q - 8) * scale; NZF stores the
    two's-complement int4 value and the original per-row fp16 scale bits.
    """
    ensure_custom_ops()
    from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32  # pylint: disable=import-outside-toplevel

    if not isinstance(data, np.ndarray) or data.dtype != np.uint8:
        raise ValueError("GGUF Q4_0 rows must be a UINT8 ndarray")
    if data.ndim != 2 or data.shape[1] % 18:
        raise ValueError("Expected GGUF Q4_0 rows [N, K/32 * 18]")
    return MsQuant4N0Group32.repack_q4_0_to_nzf(data, (data.shape[1] // 18 * 32, data.shape[0]))


def convert_embedding_weight(data, tensor_type, embedding_quantize_config: Optional[QuantType]):
    """Convert a GGUF embedding tensor to the selected runtime representation.

    Q4_0 already has the same group-32 quantization contract as W4A16 and only
    needs its blocks rearranged. Other GGUF formats must first be dequantized.
    """
    if embedding_quantize_config == QuantType.Q4_0:
        if tensor_type == GGMLQuantizationType.Q4_0:
            return rearrange_q4_0_g32(data)
        ensure_custom_ops()
        from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32  # pylint: disable=import-outside-toplevel

        fp32 = dequantize(data, tensor_type)
        return MsQuant4N0Group32.quantize_weight_g32_4bit(fp32.T)

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
            if decoder_quantize_config == QuantType.Q4_0:
                name2weight[name] = rearrange_q4_0_g32(tensor_item.data)
            elif decoder_quantize_config is None:
                name2weight[name] = tensor_item.data
            else:
                raise ValueError(f"decoder_quantize_config {decoder_quantize_config} not supported (q4_0/FP16)")
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
