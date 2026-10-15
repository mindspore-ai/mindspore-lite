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
"""Qwen3 model admission and exported architecture regression tests."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "export"))

from transformers import Qwen3Config  # pylint: disable=wrong-import-position
from models.qwen3.qwen3_exporter import Qwen3Onnx, _set_static_activation_scales  # pylint: disable=wrong-import-position
from utils.export_quant import QuantizationConfig  # pylint: disable=wrong-import-position


def _qwen3_06b_config():
    """Mirror the original Qwen3-0.6B HF architecture without loading weights."""
    return Qwen3Config(
        hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
        num_attention_heads=16, num_key_value_heads=8, head_dim=128,
        vocab_size=151936, max_position_embeddings=40960, rope_theta=1000000,
        rms_norm_eps=1e-6, tie_word_embeddings=True, eos_token_id=151645,
    )


def test_qwen3_06b_architecture():
    """Keep explicit head dimension and all layers in the packaged config."""
    exporter = Qwen3Onnx()
    exporter.config = _qwen3_06b_config()
    exporter._validate_config()  # pylint: disable=protected-access
    exporter.model = SimpleNamespace(config=exporter.config)
    quant = QuantizationConfig("S16S4")
    config = exporter.build_config(1024, 64, quant, quant, "qwen3")
    assert config["architecture"] == {
        "num_layers": 28, "hidden_size": 1024, "intermediate_size": 3072,
        "num_heads": 16, "num_kv_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "max_position_embeddings": 40960,
        "rope_theta": 1000000.0, "norm_eps": 1e-6, "tie_word_embeddings": 1,
    }
    assert config["generation"]["stop_token_ids"] == [151645]
    assert config["npu"]["embedding_format"] == "S16S4"
    assert "activation_scale" not in config["npu"]


@pytest.mark.parametrize("geometry,decoder_scale,head_scale,embedding_format", [
    ((1024, 3072, 16, 8), 32.0, 1024.0, "S16S4"),
    ((2560, 9728, 32, 8), 128.0, 128.0, "S16S4"),
    ((768, 2048, 8, 4), 1024.0, 1024.0, "S16S4"),
])
def test_model_activation_ranges(geometry, decoder_scale, head_scale, embedding_format):
    """Keep decoder ranges separate from the runtime's shared-head encoding."""
    config = SimpleNamespace(model_type="qwen3", num_hidden_layers=36 if head_scale == 128 else 28, **dict(zip(
        ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"), geometry)))
    embedding = QuantizationConfig("s16s4")
    decoder = QuantizationConfig("s16s4")
    _set_static_activation_scales(config, embedding, decoder)
    assert decoder.activation_scale == decoder_scale
    assert embedding.activation_scale == head_scale
    assert embedding.embedding_format.value == embedding_format


@pytest.mark.parametrize("layers", [8, 28, 36])
def test_supported_qwen3_layers(layers):
    """Retain the existing MiniMind and Qwen3-4B admission paths."""
    exporter = Qwen3Onnx()
    exporter.config = _qwen3_06b_config()
    exporter.config.num_hidden_layers = layers
    exporter._validate_config()  # pylint: disable=protected-access


@pytest.mark.parametrize("attribute,value,message", [
    ("num_hidden_layers", 27, "layer count"),
    ("tie_word_embeddings", False, "tied"),
    ("use_sliding_window", True, "Sliding-window"),
    ("num_experts", 8, "MoE"),
])
def test_unsupported_qwen3_config(attribute, value, message):
    """Adding 28 layers preserves the dense tied-embedding restrictions."""
    exporter = Qwen3Onnx()
    exporter.config = _qwen3_06b_config()
    setattr(exporter.config, attribute, value)
    with pytest.raises(ValueError, match=message):
        exporter._validate_config()  # pylint: disable=protected-access
