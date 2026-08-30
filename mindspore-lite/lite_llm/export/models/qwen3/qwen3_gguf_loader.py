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
"""Inject GGUF Q4_0 weights into a MiniMind-3 (Qwen3 dense) ONNX skeleton.

Model-specific half of the GGUF weight injection: the ONNX node name -> GGUF
tensor name maps and the replacement orchestration.  The model-agnostic
Q4_0 block rearrangement lives in ``utils.gguf_mapping``.

Qwen3 differences vs the Qwen2.5 loader:

* no q/k/v projection biases (``attention_bias=False``) — the ``Add`` bias
  entries are dropped;
* per-head Q/K RMSNorm weights (``q_norm`` / ``k_norm``, F32 ``[head_dim]``)
  are injected into the extra ``MsRmsNorm`` nodes.

The ONNX skeleton must already carry the quantized ``MatMul_quant`` nodes
(exported with ``--decoder-quant W4A16`` from ``qwen3_exporter``); this module
replaces their placeholder quantized weights with the real Q4_0 weights read
from the GGUF file, rearranged into the g32 planar layout expected by
``MsQuant4N0Group32``.

Example:
    from models.qwen3.qwen3_gguf_loader import gguf_loader

    gguf_loader(gguf_path=..., onnx_input_path=..., onnx_output_path=...,
                embedding_weight_save_path=...)
"""

import logging

import numpy as np
import onnx
from gguf import GGUFReader
from onnxslim import slim

from utils.onnx_postprocess import duplicate_shared_initializers
from utils.gguf_mapping import create_new_initializer, load_file_from_tensors

logger = logging.getLogger(__name__)

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

# ONNX node name -> GGUF tensor name (fp16 layer norms; Qwen3 has no biases).
FP16_WEIGHT_MAP = {
    "/model/layers.{}/self_attn/q_norm/MsRmsNorm": "blk.{}.attn_q_norm.weight",
    "/model/layers.{}/self_attn/k_norm/MsRmsNorm": "blk.{}.attn_k_norm.weight",
    "/model/layers.{}/input_layernorm/MsRmsNorm": "blk.{}.attn_norm.weight",
    "/model/layers.{}/post_attention_layernorm/MsRmsNorm": "blk.{}.ffn_norm.weight",
}

# ONNX node name -> GGUF tensor name (final model norm).
MODEL_WEIGHT_MAP = {"/model/norm/MsRmsNorm": "output_norm.weight"}


def load_q4_weight(model, weights, layers=8):
    """Replace each ``MatMul_quant`` weight with the rearranged GGUF Q4_0 weight."""
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


def load_decode_fp16_weight(model, weights, layers=8):
    """Inject fp16 layer-norm weights (incl. per-head q/k norm) into the decoder."""
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
                logger.warning("%s not supported (%s)", onnx_layer_name, node.op_type)
                continue

            gguf_weight_name = gguf_weight_name.format(i)
            gguf_weight = weights.get(gguf_weight_name)
            if gguf_weight is None:
                logger.warning("Missing GGUF tensor %s", gguf_weight_name)
                continue
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
            logger.warning("%s not supported (%s)", onnx_layer_name, node.op_type)
            continue
        onnx_weight_name = node.input[2] if node.op_type == "MsAddRmsNorm" else node.input[1]
        gguf_weight = weights.get(gguf_weight_name)
        if gguf_weight is None:
            logger.warning("Missing GGUF tensor %s", gguf_weight_name)
            continue
        new_initializers.append(create_new_initializer(onnx_weight_name, gguf_weight))
    return new_initializers


def load_weight(model, weights, layers=8, decoder_quantize_config="W4A16"):
    """Inject decoder quantized weights + fp16 norms + model norm into the skeleton."""
    if decoder_quantize_config == "W4A16":
        quant_weight = load_q4_weight(model, weights, layers)
    elif decoder_quantize_config == "FP16":
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
    layers=8,
    embedding_quantize_config="W4A16",
    decoder_quantize_config="W4A16",
):
    """Load GGUF Q4_0 weights into the ONNX skeleton and save the result."""
    reader = GGUFReader(gguf_path)
    name2weight = load_file_from_tensors(
        reader.tensors, embedding_weight_save_path, decoder_quantize_config, embedding_quantize_config
    )
    model = onnx.load(onnx_input_path)
    model = load_weight(model, name2weight, layers, decoder_quantize_config)
    onnx.save(model, onnx_output_path)
    logger.info("Saved quantized model to %s", onnx_output_path)
