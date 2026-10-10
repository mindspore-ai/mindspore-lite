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
"""Verbatim-shared middle of the family export pipelines (Qwen2.5 / Qwen3 / MiniCPM).

The three family exporters are deliberate mirror images: same steps, same
order, only the family bits (wrapper class, architecture checks, config
fields, filenames) differ in their own modules.  This module holds the shared
steps so each family keeps its per-model function small:

* ``collect_kv_names`` / ``make_input_placeholders`` — the NNRT graph I/O
  contract names and the placeholder tensors traced by ``torch.onnx.export``;
* ``finalize_onnx_graph`` — slim + add-rmsnorm fusion + shared-initializer
  duplication (the family logs the completion line with its own logger);
* ``ExportArtifacts`` / ``dump_json`` — artifact path bundle and the indented
  config-JSON writer;
* ``ExportQuantRequest`` / ``build_quant_request`` / ``prepare_weights_for_compile``
  — the post-export weight step: quantize the graph (weights + tied lm_head)
  or, without quantization, insert the shared embedding weight graph input.
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import onnx
import torch
from onnxslim import slim

from utils.onnx_postprocess import (
    _save_onnx,
    duplicate_shared_initializers,
    fuse_add_rmsnorm,
)
from utils.export_quant import ModelConfig, QuantizationConfig
from utils.export_quant import apply_quant, apply_shared_weight

# Tracing device/dtype shared by the family exporters (fp16 on CPU), mirrored
# from the family modules' own constants.
DEVICE = "cpu"
dtype = torch.float16

# The fixed non-KV input names of the NNRT graph contract; the interleaved
# per-layer past_key_i/past_val_i names are appended per model.
NNRT_INPUT_NAMES = [
    "valid_seq_len",
    "lmhead_idx",
    "rope_cos",
    "rope_sin",
    "inputs_embeds",
    "attention_mask",
]


def collect_kv_names(num_layers):
    """Interleaved per-layer KV input/output names of the NNRT contract."""
    kv_names = [(f"past_key_{i}", f"past_val_{i}") for i in range(num_layers)]
    out_kv_names = [(f"out_key_{i}", f"out_val_{i}") for i in range(num_layers)]
    return [name for kv in kv_names for name in kv], [name for kv in out_kv_names for name in kv]


@dataclass
class NnrtTraceSpec:
    """Shape knobs of the traced graph (placeholder tensor construction)."""

    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    chunk_size: int


def make_input_placeholders(spec: NnrtTraceSpec) -> tuple:
    """Build the NNRT contract placeholder tensors for ``torch.onnx.export``."""
    valid_seq_len = torch.tensor([0], dtype=torch.int32).to(DEVICE)
    lmhead_idx = torch.tensor([0], dtype=torch.int32).to(DEVICE)

    rope_cos = torch.zeros((1, spec.chunk_size, spec.head_dim), device=DEVICE, dtype=dtype)
    rope_sin = torch.zeros((1, spec.chunk_size, spec.head_dim), device=DEVICE, dtype=dtype)

    past_key_or_value = torch.zeros(
        (1, spec.num_kv_heads, spec.max_seq_len, spec.head_dim), device=DEVICE, dtype=dtype
    )
    past_key_values = [[past_key_or_value] * 2] * spec.num_layers

    inputs_embeds = torch.zeros((1, spec.chunk_size, spec.hidden_size), device=DEVICE, dtype=dtype)
    attention_mask = torch.zeros(1, 1, spec.chunk_size, spec.max_seq_len, dtype=dtype)

    return (
        None,  # input_ids (not an input: embedding lookup is CPU-side)
        valid_seq_len,
        lmhead_idx,
        rope_cos,
        rope_sin,
        inputs_embeds,
        attention_mask,
        past_key_values,
    )


def finalize_onnx_graph(model_path):
    """Slim, fuse add-rmsnorm and duplicate shared initializers, in place."""
    new_model = slim(model_path, skip_fusion_patterns=["FusionGemm"])
    _save_onnx(new_model, model_path)

    fuse_add_rmsnorm(model_path, model_path)

    new_model = onnx.load(model_path)
    duplicate_shared_initializers(new_model)
    _save_onnx(new_model, model_path)


@dataclass
class ExportArtifacts:
    """Output artifact paths produced by one family ``export_*`` run."""

    onnx: str
    embedding: str
    rope_cos: str
    rope_sin: str
    mask: str
    config: str


def dump_json(path: str, payload: Any) -> None:
    """Write ``payload`` as indented JSON (the exporters' config format)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@dataclass
class ExportQuantRequest:
    """Inputs of the post-export weight step (quantize vs shared weight)."""

    max_length: int
    chunk_size: int
    embedding_quant: QuantizationConfig
    decoder_quant: QuantizationConfig
    # Some architectures use a projected head width that differs from
    # hidden_size / num_attention_heads; 0 matches the ModelConfig default.
    head_dim: int = 0


def build_quant_request(embedding_quant, decoder_quant, max_length, chunk_size, head_dim=0):
    """Build the quant request from the raw quant-method strings."""
    return ExportQuantRequest(
        max_length=max_length,
        chunk_size=chunk_size,
        embedding_quant=QuantizationConfig(embedding_quant),
        decoder_quant=QuantizationConfig(decoder_quant),
        head_dim=head_dim,
    )


def prepare_weights_for_compile(onnx_path: str, arch_config, request: ExportQuantRequest) -> str:
    """Prepare the exported graph's weights for omc compilation; returns the effective ONNX path.

    Quantized configs rewrite the graph via apply_quant into ``<name>_quant.onnx``
    (the .omc is compiled from the quantized graph); otherwise the embedding
    weight is externalized into a runtime-injected graph input.
    """
    if request.embedding_quant.is_quant or request.decoder_quant.is_quant:
        model_config = ModelConfig(
            max_length=request.max_length,
            chunk_size=request.chunk_size,
            vocab_size=arch_config.vocab_size,
            hidden_size=arch_config.hidden_size,
            num_attention_heads=arch_config.num_attention_heads,
            num_key_value_heads=arch_config.num_key_value_heads,
            eos_id=arch_config.eos_token_id,
            embedding_quant=request.embedding_quant,
            decoder_quant=request.decoder_quant,
            head_dim=request.head_dim,
        )
        path, name = os.path.split(onnx_path)
        name, ext = os.path.splitext(name)
        quant_model_path = os.path.join(path, name + "_quant" + ext)
        apply_quant(onnx_path, quant_model_path, model_config)
        return quant_model_path  # the .omc is compiled from the quantized graph
    model = onnx.load(onnx_path)
    apply_shared_weight(model)  # inserts embedding_weight input at index 6
    _save_onnx(model, onnx_path)
    return onnx_path
