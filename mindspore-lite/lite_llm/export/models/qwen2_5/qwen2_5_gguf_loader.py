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
"""Inject GGUF Q4_0 or private Q3_0 decoder weights into a Qwen2.5-0.5B skeleton.

Q4_0 embedding/lm_head payloads retain the compact phase4 NZF layout.
Q3_0 decoder payloads use the source MsQuant3N0Group32 adapter's planes.

Example:
    from models.qwen2_5.qwen2_5_gguf_loader import gguf_loader

    gguf_loader(gguf_path=..., onnx_input_path=..., onnx_output_path=...,
                embedding_weight_save_path=...)
"""

from typing import Optional
import logging

import numpy as np
import onnx
from onnxslim import slim

from utils.onnx_postprocess import _save_onnx, duplicate_shared_initializers
from utils.gguf_mapping import create_new_initializer, load_file_from_tensors, open_gguf_reader

from utils.quantization import QuantType

logger = logging.getLogger(__name__)

# Shared by the metadata-only skeleton and the tensor-shape checks below.
QWEN2_5_ARCH = {
    "num_hidden_layers": 24,
    "intermediate_size": 4864,
    "max_position_embeddings": 32768,
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
}

# ONNX node name -> GGUF tensor name (quantized decoder matmuls).
QUANT_MATMUL_MAP = {
    "/model/layers.{}/self_attn/q_proj/MatMul_quant": "blk.{}.attn_q.weight",
    "/model/layers.{}/self_attn/k_proj/MatMul_quant": "blk.{}.attn_k.weight",
    "/model/layers.{}/self_attn/v_proj/MatMul_quant": "blk.{}.attn_v.weight",
    "/model/layers.{}/self_attn/o_proj/MatMul_quant": "blk.{}.attn_output.weight",
    "/model/layers.{}/mlp/gate_proj/MatMul_quant": "blk.{}.ffn_gate.weight",
    "/model/layers.{}/mlp/up_proj/MatMul_quant": "blk.{}.ffn_up.weight",
    "/model/layers.{}/mlp/down_proj/MatMul_quant": "blk.{}.ffn_down.weight",
}

# ONNX node name -> GGUF tensor name (fp16 biases + layer norms).
FP16_WEIGHT_MAP = {
    "/model/layers.{}/self_attn/q_proj/Add": "blk.{}.attn_q.bias",
    "/model/layers.{}/self_attn/k_proj/Add": "blk.{}.attn_k.bias",
    "/model/layers.{}/self_attn/v_proj/Add": "blk.{}.attn_v.bias",
    "/model/layers.{}/input_layernorm/MsRmsNorm": "blk.{}.attn_norm.weight",
    "/model/layers.{}/post_attention_layernorm/MsRmsNorm": "blk.{}.ffn_norm.weight",
}

# ONNX node name -> GGUF tensor name (final model norm).
MODEL_WEIGHT_MAP = {"/model/norm/MsRmsNorm": "output_norm.weight"}


def _validate_tensor_shapes(tensors, layers):
    """Check logical GGUF dimensions without loading or dequantizing payloads."""
    hidden = QWEN2_5_ARCH["hidden_size"]
    intermediate = QWEN2_5_ARCH["intermediate_size"]
    head_dim = hidden // QWEN2_5_ARCH["num_attention_heads"]
    kv_width = head_dim * QWEN2_5_ARCH["num_key_value_heads"]
    embedding_shape = (hidden, QWEN2_5_ARCH["vocab_size"])
    # GGUF matrices declare [input, output], not the stored numpy row order.
    layer_shapes = {
        "attn_q.weight": (hidden, hidden),
        "attn_k.weight": (hidden, kv_width),
        "attn_v.weight": (hidden, kv_width),
        "attn_output.weight": (hidden, hidden),
        "ffn_gate.weight": (hidden, intermediate),
        "ffn_up.weight": (hidden, intermediate),
        "ffn_down.weight": (intermediate, hidden),
        "attn_q.bias": (hidden,),
        "attn_k.bias": (kv_width,),
        "attn_v.bias": (kv_width,),
        "attn_norm.weight": (hidden,),
        "ffn_norm.weight": (hidden,),
    }
    required_shapes = {
        f"blk.{layer}.{name}": shape
        for layer in range(layers) for name, shape in layer_shapes.items()
    }
    required_shapes.update({
        "token_embd.weight": embedding_shape,
        "output_norm.weight": (hidden,),
    })
    tensor_shapes = {tensor.name: tuple(tensor.shape) for tensor in tensors}
    # Tied models omit output.weight; validate it too when a GGUF includes it.
    if "output.weight" in tensor_shapes:
        required_shapes["output.weight"] = embedding_shape
    for name, expected in required_shapes.items():
        actual = tensor_shapes.get(name)
        if actual is None:
            raise ValueError(f"Missing required GGUF tensor {name}")
        if actual != expected:
            raise ValueError(
                f"GGUF tensor {name} has logical shape {actual}, expected {expected}"
            )


def load_quantized_weight(model, weights, layers=24):
    """Replace each quantized MatMul weight with its packed GGUF payload."""
    nodes = {node.name: node for node in model.graph.node}

    new_initializers = []
    for i in range(layers):
        for onnx_layer_name, gguf_weight_name in QUANT_MATMUL_MAP.items():
            onnx_layer_name = onnx_layer_name.format(i)
            node = nodes[onnx_layer_name]
            onnx_weight_name = node.input[1]
            gguf_weight_name = gguf_weight_name.format(i)
            gguf_weight = np.frombuffer(weights[gguf_weight_name].tobytes(), dtype=np.uint8)
            new_initializers.append(create_new_initializer(onnx_weight_name, gguf_weight))
            logger.debug("Load %s -> %s", gguf_weight_name, onnx_weight_name)
    return new_initializers


def load_decode_fp16_weight(model, weights, layers=24):
    """Inject fp16 biases and layer-norm weights into the decoder."""
    nodes = {node.name: node for node in model.graph.node}

    new_initializers = []
    for i in range(layers):
        for onnx_layer_name, gguf_weight_name in FP16_WEIGHT_MAP.items():
            onnx_layer_name = onnx_layer_name.format(i)
            node = nodes[onnx_layer_name]
            if node.op_type in ("MsRmsNorm",):
                onnx_weight_name = node.input[1]
            elif node.op_type in ("Add",):
                onnx_weight_name = node.input[0]
            elif node.op_type in ("MsAddRmsNorm",):
                onnx_weight_name = node.input[2]
            else:
                raise ValueError(f"Unsupported GGUF bias/norm node {onnx_layer_name}: {node.op_type}")

            gguf_weight_name = gguf_weight_name.format(i)
            gguf_weight = weights.get(gguf_weight_name)
            if gguf_weight is None:
                raise ValueError(f"Missing required GGUF tensor {gguf_weight_name}")
            new_initializers.append(create_new_initializer(onnx_weight_name, gguf_weight))
            logger.debug("Load %s -> %s", gguf_weight_name, onnx_weight_name)
    return new_initializers


def load_model_fp16_weight(model, weights):
    """Inject the final model norm weight."""
    nodes = {node.name: node for node in model.graph.node}

    new_initializers = []
    for onnx_layer_name, gguf_weight_name in MODEL_WEIGHT_MAP.items():
        node = nodes[onnx_layer_name]
        if node.op_type not in ("MsAddRmsNorm", "MsRmsNorm"):
            raise ValueError(f"Unsupported GGUF model norm node {onnx_layer_name}: {node.op_type}")
        onnx_weight_name = node.input[2] if node.op_type == "MsAddRmsNorm" else node.input[1]
        gguf_weight = weights.get(gguf_weight_name)
        if gguf_weight is None:
            raise ValueError(f"Missing required GGUF tensor {gguf_weight_name}")
        new_initializers.append(create_new_initializer(onnx_weight_name, gguf_weight))
    return new_initializers



def load_weight(model, weights, layers=24, decoder_quantize_config: Optional[QuantType] = QuantType.Q4_0):
    """Inject decoder quantized weights + fp16 norm/bias + model norm into the skeleton."""
    if decoder_quantize_config in (QuantType.Q4_0, QuantType.Q3_0):
        quant_weight = load_quantized_weight(model, weights, layers)
    elif decoder_quantize_config is None:
        quant_weight = load_decode_fp16_weight(model, weights, layers)
    else:
        raise ValueError(f"decoder_quantize_config {decoder_quantize_config} not supported")

    decode_fp16_weight = load_decode_fp16_weight(model, weights, layers)
    model_fp16_weight = load_model_fp16_weight(model, weights)

    new_initializers = quant_weight + decode_fp16_weight + model_fp16_weight

    # Keep any initializer not replaced above (e.g. constants).
    for init in model.graph.initializer:
        if not any(init.name == new_init.name for new_init in new_initializers):
            new_initializers.append(init)

    new_graph = onnx.helper.make_graph(
        model.graph.node, model.graph.name, model.graph.input, model.graph.output, new_initializers
    )
    new_model = onnx.helper.make_model(new_graph, producer_name=model.producer_name)
    new_model.opset_import[0].version = 18

    new_model = slim(new_model)
    duplicate_shared_initializers(new_model)
    return new_model


def gguf_loader(
    gguf_path,
    onnx_input_path,
    onnx_output_path,
    embedding_weight_save_path,
    layers=24,
    embedding_quantize_config=QuantType.Q4_0,
    decoder_quantize_config=QuantType.Q4_0,
):
    """Load GGUF decoder weights into the skeleton and save the result."""
    embedding_quantize_config = QuantType.parse(embedding_quantize_config)
    decoder_quantize_config = QuantType.parse(decoder_quantize_config)
    reader = open_gguf_reader(gguf_path)
    _validate_tensor_shapes(reader.tensors, layers)
    name2weight = load_file_from_tensors(
        reader.tensors, embedding_weight_save_path, decoder_quantize_config, embedding_quantize_config
    )
    model = onnx.load(onnx_input_path)
    model = load_weight(model, name2weight, layers, decoder_quantize_config)
    _save_onnx(model, onnx_output_path)
    logger.info("Saved quantized model to %s", onnx_output_path)
