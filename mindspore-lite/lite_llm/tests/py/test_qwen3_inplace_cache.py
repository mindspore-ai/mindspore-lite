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

_inplace_cache = importlib.import_module("models.qwen3.inplace_cache")
lower_inplace_cache = _inplace_cache.lower_inplace_cache
SHADOW_OPS = _inplace_cache.SHADOW_OPS


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
    fused_before = [node for node in model.graph.node
                    if node.op_type == "MsTransRopeScatterNDUpdate"]
    assert all(len(node.output) == 1 for node in fused_before)
    assert sum(node.op_type == "ScatterND" for node in model.graph.node) == 4
    assert len(model.graph.output) == 5
    lower_inplace_cache(model, 2)
    onnx.checker.check_model(model)
    assert [output.name for output in model.graph.output] == ["logits"]
    assert len([node for node in model.graph.node if node.op_type == "MsTransRopeScatterNDUpdate"]) == 2
    assert len([node for node in model.graph.node if node.op_type == "MsAttention"]) == 2
    assert not any(node.op_type in ("MsGroupMatmul", "MsAddSoftmax", "ScatterND") for node in model.graph.node)
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


def _traced_cache_module(torch, num_heads, head_dim):
    """Build the smallest Torch module that writes a KV cache with ``index_put``."""
    from mslite_llm_ops import MsAttention, MsTransRopeScatterNDUpdate

    class TinyCacheLayer(torch.nn.Module):
        """One fused RoPE + in-place cache write feeding fused attention."""

        def forward(self, q, k, v, cos, sin, past_key_0, past_val_0, valid_seq_len, mask):
            """Mirror the Qwen3 wrapper: fused write, then attention on the caches."""
            query = MsTransRopeScatterNDUpdate.apply(
                q, k, v, cos, sin, past_key_0, past_val_0, valid_seq_len,
                num_heads, head_dim,
            )
            scale = query.new_full((1,), head_dim ** -0.5)
            logits = MsAttention.apply(
                query, past_key_0, past_val_0, mask, scale, valid_seq_len,
            )
            return logits, past_key_0, past_val_0

    return TinyCacheLayer()


def test_traced_index_put_cache_update_is_lowered(tmp_path):
    """Lower a cache update whose ScatterND indices come from Torch ``index_put``.

    Exporting ``cache[:, :, idx:idx + seq, :].copy_(...)`` makes Torch build the
    ScatterND indices with its constant broadcast idiom
    (``ConstantOfShape``/``Equal``/``Where``/``Expand``).  The hand-built ONNX
    fixtures above only use plain shape arithmetic, so they cannot catch a
    ``SHADOW_OPS`` set that omits those operators; such a gap rejected every
    real Qwen3 export with ``Unexpected cache update operator: Where``.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("mslite_llm_ops")

    batch, seq_len, kv_len = 1, 4, 16
    num_heads, num_kv_heads, head_dim = 4, 2, 32
    module = _traced_cache_module(torch, num_heads, head_dim)
    args = (
        torch.zeros(batch, seq_len, num_heads, head_dim, dtype=torch.float16),
        torch.zeros(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16),
        torch.zeros(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16),
        torch.zeros(1, seq_len, head_dim, dtype=torch.float16),
        torch.zeros(1, seq_len, head_dim, dtype=torch.float16),
        torch.zeros(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16),
        torch.zeros(batch, num_kv_heads, kv_len, head_dim, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(seq_len, kv_len, dtype=torch.float16),
    )
    path = tmp_path / "traced_cache.onnx"
    torch.onnx.export(
        module, args, str(path), opset_version=18, dynamo=False,
        input_names=["q", "k", "v", "rope_cos", "rope_sin",
                     "past_key_0", "past_val_0", "valid_seq_len", "mask"],
        output_names=["logits", "out_key_0", "out_val_0"],
    )
    model = onnx.load(path)
    scatters = [node for node in model.graph.node if node.op_type == "ScatterND"]
    assert scatters, "Torch no longer lowers the cache write to ScatterND"

    lower_inplace_cache(model, 1)

    assert [value.name for value in model.graph.output] == ["logits"]
    assert not any(node.op_type == "ScatterND" for node in model.graph.node)
    fused = [node for node in model.graph.node
             if node.op_type == "MsTransRopeScatterNDUpdate"]
    attention = [node for node in model.graph.node if node.op_type == "MsAttention"]
    assert len(fused) == 1 and len(attention) == 1
    assert list(attention[0].input[1:3]) == ["past_key_0", "past_val_0"]


def test_shadow_ops_cover_the_torch_index_put_index_idiom():
    """Pin the operator set Torch emits for the ScatterND indices of a cache write."""
    assert {"ConstantOfShape", "Equal", "Where", "Expand"} <= SHADOW_OPS
