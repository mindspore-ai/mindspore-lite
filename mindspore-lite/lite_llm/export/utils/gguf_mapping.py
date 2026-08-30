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

Model-agnostic: block rearrangement for the ``MsQuant4N0Group32`` g32 planar
layout, initializer creation, and ``load_file_from_tensors`` which reads the
raw GGUF tensors into a name->weight map (biases/norms as fp16, quantized
weights rearranged, embedding saved separately).

Example:
    from utils.gguf_mapping import load_file_from_tensors, rearrange_q4_0_g32
"""

import logging

import numpy as np
import onnx

logger = logging.getLogger(__name__)

def create_new_initializer(name, weight):
    return onnx.numpy_helper.from_array(weight, name)


def rearrange_q4_0_g32(data):
    """Rearrange GGUF Q4_0 blocks into the g32 planar layout.

    GGUF Q4_0 block (32 weights): 2-byte fp16 scale + 16-byte int4 nibbles.
    ``MsQuant4N0Group32`` expects all weight bytes first, then all fp16 scales
    (matching ``packing.quantize_weight_g32_4bit_nd``).
    """
    data_flatten = np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)
    block_bytes = 2 + 32 // 2  # 2 scale + 16 qweight = 18 bytes per block
    num_block = data_flatten.shape[0] // block_bytes
    final_data = np.zeros((num_block * block_bytes,), dtype=np.uint8)
    scale_start = num_block * 16
    for block_id in range(num_block):
        scale_raw = data_flatten[block_id * block_bytes: block_id * block_bytes + 2]
        qweight = data_flatten[block_id * block_bytes + 2: (block_id + 1) * block_bytes]
        # GGUF Q4_0 nibbles are already SPLIT (byte j = elem j | elem j+16),
        # matching the douyin MsQuant4N0Group32 kernel. Only reorder blocks
        # (weights first, scales after).
        final_data[block_id * 16: (block_id + 1) * 16] = qweight
        final_data[scale_start + block_id * 2] = scale_raw[0]
        final_data[scale_start + block_id * 2 + 1] = scale_raw[1]
    return final_data



def load_file_from_tensors(tensors, embedding_weight_save_path, decoder_quantize_config, embedding_quantize_config):
    """Read GGUF tensors, rearrange Q4_0 weights, and save the embedding weight."""
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
            if decoder_quantize_config == "W4A16":
                name2weight[name] = rearrange_q4_0_g32(tensor_item.data)
            elif decoder_quantize_config == "FP16":
                name2weight[name] = tensor_item.data
            else:
                raise ValueError(f"decoder_quantize_config {decoder_quantize_config} not supported (W4A16/FP16)")
        elif name == "token_embd.weight":
            if embedding_quantize_config == "W4A16":
                name2weight[name] = rearrange_q4_0_g32(tensor_item.data)
            elif embedding_quantize_config == "FP16":
                name2weight[name] = tensor_item.data
            else:
                raise ValueError(f"embedding_quantize_config {embedding_quantize_config} not supported (W4A16/FP16)")
            name2weight[name].tofile(embedding_weight_save_path)
        elif name == "output_norm.weight":
            name2weight[name] = tensor_item.data.astype(np.float16)
        else:
            raise ValueError(f"Unexpected GGUF tensor {name}")
    return name2weight
