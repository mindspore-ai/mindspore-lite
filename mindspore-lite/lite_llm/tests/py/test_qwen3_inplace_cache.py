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
"""Regression coverage for fused attention with writable packed KV caches."""

import importlib
from pathlib import Path
import sys

import pytest

onnx = pytest.importorskip("onnx")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "export"))

lower_inplace_cache = importlib.import_module("models.qwen3.inplace_cache").lower_inplace_cache


def _graph():
    """Build the smallest graph containing the fused cache contract."""
    helper = onnx.helper
    inputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, [1, 2, 8, 32])
              for name in ["q", "k", "v", "cos", "sin", "past_key_0", "past_val_0", "index", "mask"]]
    nodes = [
        helper.make_node("MsTransRopeScatterNDUpdate", [value.name for value in inputs[:8]],
                         ["rope_q", "out_key_0", "out_val_0"], domain="custom"),
        helper.make_node("MsAttention", ["rope_q", "out_key_0", "out_val_0", "mask", "scale", "index"],
                         ["logits"], domain="custom"),
    ]
    outputs = [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, [1, 2, 8, 32])
               for name in ["logits", "out_key_0", "out_val_0"]]
    scale = helper.make_tensor("scale", onnx.TensorProto.FLOAT16, [1], [0.125])
    return helper.make_model(helper.make_graph(nodes, "cache", inputs, outputs, [scale]),
                             opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)])


def test_cache_inputs_remain_persistent_and_ordered():
    model = _graph()
    lower_inplace_cache(model, 1)
    onnx.checker.check_model(model)
    fused, attention = model.graph.node
    assert list(fused.output) == ["rope_q"]
    assert onnx.helper.get_attribute_value(fused.attribute[0]) == b"NZ"
    assert list(attention.input) == ["rope_q", "past_key_0", "past_val_0", "mask", "scale", "index"]
    assert [value.name for value in model.graph.output] == ["logits"]


@pytest.mark.parametrize("input_index", [0, 1, 2, 5])
def test_unordered_cache_read_is_rejected_without_mutating_graph(input_index):
    model = _graph()
    model.graph.node[1].input[input_index] = "q"
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="cache consumers|not ordered"):
        lower_inplace_cache(model, 1)
    assert model.SerializeToString() == before


@pytest.mark.parametrize("chunk_size", [1, 16])
def test_lowering_accepts_traced_qwen3_graph(tmp_path, chunk_size):
    """Exercise real Torch export, slimming and fusion before cache lowering."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from models.qwen3.qwen3_exporter import Qwen3Onnx
    from utils.export_quant import infer_shape
    from onnxslim import slim

    config = transformers.Qwen3Config(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, tie_word_embeddings=True,
    )
    exporter = Qwen3Onnx()
    exporter.config = config
    with torch.random.fork_rng(devices=[]):
        exporter.model = transformers.Qwen3ForCausalLM(config).half().eval()
    exporter.num_layers = 2
    exporter.num_kv_heads = 2
    exporter.hidden_size = 128
    path = tmp_path / "tiny.onnx"
    exporter.export(str(path), max_seq_len=64, chunk_size=chunk_size)
    # Quantization also runs slimming after RMSNorm fusion to restore node order.
    model = slim(onnx.load(path), no_shape_infer=True)
    model = infer_shape(model, chunk_size, 64, 2, 4, 32)
    lower_inplace_cache(model, 2)
    onnx.checker.check_model(model)
    assert [output.name for output in model.graph.output] == ["logits"]
    assert len([node for node in model.graph.node if node.op_type == "MsTransRopeScatterNDUpdate"]) == 2
    assert len([node for node in model.graph.node if node.op_type == "MsAttention"]) == 2
    assert not any(node.op_type in ("MsGroupMatmul", "MsAddSoftmax") for node in model.graph.node)
    for node in model.graph.node:
        if node.op_type == "MsTransRopeScatterNDUpdate":
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            assert attrs["storage_layout"] == b"NZ"
        if node.op_type == "MsAttention":
            assert node.input[5] == "valid_seq_len"
            values = {value.name: value for value in model.graph.value_info}
            mask = values[node.input[3]].type.tensor_type
            assert [dim.dim_value for dim in mask.shape.dim] == [chunk_size, 64]
            output = values[node.output[0]].type.tensor_type
            assert [dim.dim_value for dim in output.shape.dim] == [1, 4, chunk_size, 32]


def test_extra_packed_query_consumer_is_rejected():
    model = _graph()
    model.graph.node.append(onnx.helper.make_node("Identity", ["rope_q"], ["unexpected"]))
    before = model.SerializeToString()
    with pytest.raises(ValueError, match="packed query consumers"):
        lower_inplace_cache(model, 1)
    assert model.SerializeToString() == before
