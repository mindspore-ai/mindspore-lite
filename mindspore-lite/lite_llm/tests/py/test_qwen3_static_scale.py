"""Calibrated Qwen3 scale policy and shared embedding version selection."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "export"))

qwen3_exporter = importlib.import_module("models.qwen3.qwen3_exporter")
_set_static_activation_scales = getattr(qwen3_exporter, "_set_static_activation_scales")
QuantizationConfig = importlib.import_module("utils.export_quant").QuantizationConfig


@pytest.mark.parametrize("architecture,scale,encoding", [
    ((2560, 9728, 32, 8), 128, "S16S4"),
    ((768, 2048, 8, 2), 1024, "S16S4"),
    ((4096, 12288, 32, 8), 1024, "S16S4"),
])
def test_static_scale_is_scoped_to_qwen3_4b(architecture, scale, encoding):
    """Select the calibrated scale only for supported Qwen3 architectures."""
    config = SimpleNamespace(model_type="qwen3", num_hidden_layers=36 if scale == 128 else 28, **dict(zip(
        ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"), architecture)))
    embedding, decoder = QuantizationConfig("s16s4"), QuantizationConfig("s16s4")
    _set_static_activation_scales(config, embedding, decoder)
    assert embedding.activation_scale == decoder.activation_scale == scale
    assert embedding.embedding_format == encoding
    assert "activation_scale" not in embedding.asdict()
    q4 = QuantizationConfig("q4_0")
    _set_static_activation_scales(config, q4, decoder)
    assert q4.embedding_format == "W4A16"
    assert "activation_scale" not in q4.asdict()


def test_unquantized_embedding_format_is_fp16():
    assert QuantizationConfig(None).embedding_format == "FP16"


def test_shared_embedding_format_is_independent_of_scale():
    assert QuantizationConfig("s16s4", 256).embedding_format == "S16S4"


@pytest.mark.parametrize("model_type,layers,scale", [
    ("qwen3", 28, 1024),
    ("qwen3", 36, 128),
    ("qwen3", 32, 1024),
    ("other", 36, 1024),
])
def test_shared_head_policy_uses_model_name_and_layer_count(model_type, layers, scale):
    """Select the shared-head multiplier from the model family and layer count."""
    config = SimpleNamespace(model_type=model_type, num_hidden_layers=layers,
                             hidden_size=768, intermediate_size=2048,
                             num_attention_heads=8, num_key_value_heads=2)
    embedding, decoder = QuantizationConfig("s16s4"), QuantizationConfig("s16s4")
    _set_static_activation_scales(config, embedding, decoder)
    assert embedding.activation_scale == scale
