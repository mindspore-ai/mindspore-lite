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
"""Qwen2.5 metadata geometry, artifact names and weight-source validation."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from transformers import Qwen2Config

from models.qwen2_5.qwen2_5_exporter import (
    QWEN2_5_NEUTRAL_NAMES, Qwen2Onnx, _canonical_model_scale, qwen2_5_artifact_stem, resolve_qwen2_5_artifact_names,
    validate_attention_shapes, validate_quant_capability, validate_qwen2_5_geometry,
    validate_qwen2_5_source_layers,
)
from models.qwen2_5.qwen2_5_gguf_loader import read_gguf_geometry, resolve_gguf_layers
from utils.quantization import QuantType

# Public Qwen2.5 geometries, used only as test inputs; the product code derives
# them from config.json / GGUF metadata and enumerates no size anywhere.
SIZE_GEOMETRIES = {
    "0.5b": (24, 896, 4864, 14, 2, 64),
    "1.5b": (28, 1536, 8960, 12, 2, 128),
    "3b": (36, 2048, 11008, 16, 2, 128),
}


def _geometry(size):
    """Build the derived-geometry mapping the validators consume."""
    layers, hidden, intermediate, heads, kv_heads, head_dim = SIZE_GEOMETRIES[size]
    return {"num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": intermediate,
            "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
            "vocab_size": 151936, "max_position_embeddings": 32768}


def _write_gguf_header(path, size, with_key_length=True, omit=None):
    """Write a tensorless GGUF carrying one size's architecture metadata."""
    gguf = pytest.importorskip("gguf")
    layers, hidden, intermediate, heads, kv_heads, head_dim = SIZE_GEOMETRIES[size]
    writer = gguf.GGUFWriter(str(path), "qwen2")
    try:
        adders = {
            "block_count": lambda: writer.add_block_count(layers),
            "embedding_length": lambda: writer.add_embedding_length(hidden),
            "feed_forward_length": lambda: writer.add_feed_forward_length(intermediate),
            "head_count": lambda: writer.add_head_count(heads),
            "head_count_kv": lambda: writer.add_head_count_kv(kv_heads),
        }
        for key, adder in adders.items():
            if key != omit:
                adder()
        if with_key_length:
            writer.add_key_length(head_dim)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


@pytest.mark.parametrize("size", ["0.5b", "1.5b", "3b"])
@pytest.mark.parametrize("with_key_length", [True, False])
def test_read_gguf_geometry(tmp_path, size, with_key_length):
    """Derive every size hyper-parameter from GGUF metadata, head_dim included."""
    from utils.gguf_mapping import open_gguf_reader

    path = tmp_path / f"{size}.gguf"
    _write_gguf_header(path, size, with_key_length)
    geometry = read_gguf_geometry(open_gguf_reader(path).fields)
    layers, hidden, intermediate, heads, kv_heads, head_dim = SIZE_GEOMETRIES[size]
    assert geometry["num_hidden_layers"] == layers
    assert (geometry["hidden_size"], geometry["intermediate_size"]) == (hidden, intermediate)
    assert (geometry["num_attention_heads"], geometry["num_key_value_heads"]) == (heads, kv_heads)
    assert geometry["head_dim"] == head_dim
    assert resolve_gguf_layers(geometry) == layers
    assert resolve_gguf_layers(geometry, layers) == layers
    with pytest.raises(ValueError, match=f"requested layers=7, GGUF metadata block_count={layers}"):
        resolve_gguf_layers(geometry, 7)


def test_read_gguf_geometry_missing_field(tmp_path):
    """Name the missing metadata key instead of falling back to a size table."""
    from utils.gguf_mapping import open_gguf_reader

    path = tmp_path / "no-ffn.gguf"
    _write_gguf_header(path, "1.5b", omit="feed_forward_length")
    with pytest.raises(ValueError, match=r"qwen2.feed_forward_length must be a positive integer, got None"):
        read_gguf_geometry(open_gguf_reader(path).fields)


@pytest.mark.parametrize("size", ["0.5b", "1.5b", "3b"])
@pytest.mark.parametrize("quant_type", [QuantType.Q4_0, QuantType.Q3_0])
def test_quant_capability_accepts_every_size(size, quant_type):
    """Q4_0 and Q3_0 accept all three sizes derived from metadata."""
    validate_quant_capability(_geometry(size), quant_type)
    validate_quant_capability(_geometry(size), None)


@pytest.mark.parametrize("size", ["0.5b", "1.5b", "3b"])
def test_quant_capability_rejects_s16s4_unaligned_k(size):
    """s16s4 needs K%512; report the role, the actual K and the divisor."""
    with pytest.raises(ValueError, match=r"K=\d+ must be divisible by 512 \(S16S4_K_ALIGNMENT\)"):
        validate_quant_capability(_geometry(size), QuantType.S16S4)


def test_quant_capability_rejects_unaligned_intermediate():
    """An intermediate size that breaks the output fractal alignment is named."""
    geometry = dict(_geometry("3b"), intermediate_size=11000)
    with pytest.raises(ValueError, match=r"ffn_gate.weight N=11000 must be divisible by 16"):
        validate_quant_capability(geometry, QuantType.Q4_0)


@pytest.mark.parametrize("layers,hidden,intermediate,heads,dim", [
    (28, 1536, 8960, 12, 128), (24, 896, 4864, 14, 64),
    (1, 896, 4864, 14, 64), (30, 896, 4864, 14, 64),
])
def test_geometry(layers, hidden, intermediate, heads, dim):
    """Accept supported configurations with varying geometry."""
    config = Qwen2Config(num_hidden_layers=layers, hidden_size=hidden, intermediate_size=intermediate,
                         num_attention_heads=heads, num_key_value_heads=2, vocab_size=151936,
                         max_position_embeddings=32768)
    assert validate_qwen2_5_geometry(config) == {
        "num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": intermediate,
        "num_attention_heads": heads, "num_key_value_heads": 2, "head_dim": dim,
        "vocab_size": 151936, "max_position_embeddings": 32768,
    }


@pytest.mark.parametrize("change,field", [
    ({"hidden_size": 840, "num_attention_heads": 12}, "head_dim"),
    ({"num_key_value_heads": 3}, "num_key_value_heads"),
    ({"model_type": "other"}, "model_type"),
])
def test_invalid_geometry(change, field):
    """Reject invalid architecture geometry."""
    config = SimpleNamespace(model_type="qwen2", num_hidden_layers=1, hidden_size=896,
                             intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2,
                             vocab_size=100, max_position_embeddings=32768)
    for key, value in change.items():
        setattr(config, key, value)
    with pytest.raises(ValueError, match=field):
        validate_qwen2_5_geometry(config)


def test_attention_shapes():
    """Enforce attention alignment constraints."""
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_attention_shapes(1000, 64, 64)
    with pytest.raises(ValueError, match="chunk_size"):
        validate_attention_shapes(1024, 24, 64)
    validate_attention_shapes(1024, 1, 64)
    validate_attention_shapes(1024, 64, 64)


@pytest.mark.parametrize("label,size", [("0.5B", "0b5"), ("1.5B", "1b5"), ("3B", "3b"), ("7B", "7b")])
def test_labels(label, size):
    """Normalize model size tags and artifact stems."""
    assert _canonical_model_scale(label) == label.lower()
    assert qwen2_5_artifact_stem(label) == f"qwen2_5_{size}"


def test_artifact_names(tmp_path):
    """Resolve bounded model size tags in directory names."""
    directory = tmp_path / "Qwen2.5-1.5B-Instruct"
    directory.mkdir()
    assert resolve_qwen2_5_artifact_names(directory) == {
        "onnx_name": "qwen2_5_1b5.onnx", "quant_name": "qwen2_5_1b5_quant.onnx",
        "gguf_name": "qwen2_5_1b5_gguf.onnx", "model_name": "qwen2.5-1.5b",
    }
    assert resolve_qwen2_5_artifact_names(tmp_path / "unknown") == QWEN2_5_NEUTRAL_NAMES
    assert resolve_qwen2_5_artifact_names(tmp_path / "qwen2_5_1b5") == QWEN2_5_NEUTRAL_NAMES
    assert resolve_qwen2_5_artifact_names(tmp_path / "Qwen2.5-0.5B-Instruct") == {
        "onnx_name": "qwen2_5_0b5.onnx", "quant_name": "qwen2_5_0b5_quant.onnx",
        "gguf_name": "qwen2_5_0b5_gguf.onnx", "model_name": "qwen2.5-0.5b",
    }


def test_source_layers(tmp_path):
    """Validate source counts and contiguous layer indices."""
    with pytest.raises(ValueError, match="cannot determine weight layer count"):
        validate_qwen2_5_source_layers(tmp_path, 28)
    weights = {f"model.layers.{i}.self_attn.q_proj.weight": "model.safetensors" for i in range(28)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    validate_qwen2_5_source_layers(tmp_path, 28)
    with pytest.raises(ValueError, match="config layers=30, weight layers=28"):
        validate_qwen2_5_source_layers(tmp_path, 30)
    weights.pop("model.layers.27.self_attn.q_proj.weight")
    weights["model.layers.28.self_attn.q_proj.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    with pytest.raises(ValueError, match=r"config layers=28, weight layers=28.*missing indices=\[27\].*\[28\]"):
        validate_qwen2_5_source_layers(tmp_path, 28)


@pytest.mark.parametrize("count", [28, 24])
def test_source_layers_unindexed_shards(tmp_path, count):
    """Combine layer keys from all unindexed safetensors shards."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    for shard, indices in enumerate((range(14), range(14, count))):
        weights = {f"model.layers.{i}.weight": np.zeros((2, 2), dtype=np.float32) for i in indices}
        save_file(weights, str(tmp_path / f"model-{shard}.safetensors"))
    if count == 28:
        validate_qwen2_5_source_layers(tmp_path, 28)
        with pytest.raises(ValueError, match="config layers=30, weight layers=28"):
            validate_qwen2_5_source_layers(tmp_path, 30)
    else:
        with pytest.raises(ValueError, match="config layers=28, weight layers=24"):
            validate_qwen2_5_source_layers(tmp_path, 28)


def test_source_layers_pytorch_bin_only(tmp_path):
    """Reject a directory with only unsupported PyTorch weights."""
    (tmp_path / "pytorch_model.bin").touch()
    with pytest.raises(ValueError, match="cannot determine weight layer count"):
        validate_qwen2_5_source_layers(tmp_path, 28)


@pytest.mark.parametrize("with_gguf_size_meta", [True, False])
def test_gguf_source_and_names(tmp_path, with_gguf_size_meta):
    """Read GGUF layers and resolve size from metadata or filename."""
    gguf = pytest.importorskip("gguf")
    path = tmp_path / "qwen2.5-1.5b-instruct-fp16.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen2")
    writer.add_block_count(2)
    if with_gguf_size_meta:
        writer.add_string("general." + "size" + "_label", "1.5B")
    for index in range(2):
        writer.add_tensor(f"blk.{index}.attn_q.weight", np.zeros((4, 4), np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    validate_qwen2_5_source_layers(path, 2)
    with pytest.raises(ValueError, match="config layers=3, weight layers=2"):
        validate_qwen2_5_source_layers(path, 3)
    assert resolve_qwen2_5_artifact_names(path) == {
        "onnx_name": "qwen2_5_1b5.onnx", "quant_name": "qwen2_5_1b5_quant.onnx",
        "gguf_name": "qwen2_5_1b5_gguf.onnx", "model_name": "qwen2.5-1.5b",
    }


def test_default_model_name():
    """Use a nonempty default model name before loading."""
    assert Qwen2Onnx().model_name == "qwen2.5"
