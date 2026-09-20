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
"""Tests for the Qwen2.5-0.5B NNRT exporter and its packaging path.

The exporter lives in the flat ``lite_llm/export/`` tree (``mslite_llm_export.py``
+ ``models/`` + ``utils/``); tests import it by path, mirroring
``test_export_entrypoint.py``.

Requires the qwen2.5 export extras (``pip install -r requirements.txt``)
for the ONNX-graph tests; the config/quant-config tests run without torch/onnx.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from utils.export_quant import QuantizationConfig  # noqa: E402

# onnx is an optional dependency: importorskip must run first so the
# config-only tests still collect when onnx is absent; the import below
# therefore intentionally stays after it instead of at the module top.
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402,H2305


def _make_lmhead_graph():
    """A minimal graph with an lm_head MatMul consuming a weight initializer.

    Mirrors the NNRT input contract: 6 non-embedding inputs precede the
    ``embedding_weight`` input so ``apply_shared_weight`` can insert at index 6.
    """
    weight = helper.make_tensor("lm_head.weight", TensorProto.FLOAT16, [4, 8], [0.0] * 32)
    hidden = helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 1, 8])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, 1, 4])
    matmul = helper.make_node(
        "MatMul",
        inputs=["hidden", "lm_head.weight"],
        outputs=["logits"],
        name="model/lm_head/MatMul",
    )
    graph = helper.make_graph(
        [matmul],
        "g",
        [hidden],
        [logits],
        [weight],
    )
    # Contract inputs [valid_seq_len, lmhead_idx, rope_cos, rope_sin, inputs_embeds,
    # attention_mask] precede the 7th non-KV input (embedding_weight, inserted later).
    for i, name in enumerate(
        ["valid_seq_len", "lmhead_idx", "rope_cos", "rope_sin", "inputs_embeds", "attention_mask"]
    ):
        graph.input.insert(i, helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1]))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model


def test_quantization_config():
    """Test that W4 quant schemes parse and unsupported schemes raise."""
    assert QuantizationConfig(None).is_quant is False
    cfg = QuantizationConfig("W4A16")
    assert cfg.is_quant and cfg.bits == 4 and cfg.group_size == 32
    cfg = QuantizationConfig("W4A8")
    assert cfg.group_size == 128
    with pytest.raises(ValueError):
        QuantizationConfig("W8A8")


def test_apply_shared_weight_inserts_embedding_input_at_index_6():
    """apply_shared_weight inserts the embedding input at the contract index."""
    from utils.export_quant import apply_shared_weight

    model = _make_lmhead_graph()
    apply_shared_weight(model)

    names = [vi.name for vi in model.graph.input]
    assert "embedding_weight" in names
    # NNRT contract: embedding_weight sits at index 6 of the 7 non-KV inputs.
    assert names.index("embedding_weight") == 6
    # The lm_head MatMul now consumes the transposed graph input, and the
    # weight initializer is gone.
    assert not any(init.name == "lm_head.weight" for init in model.graph.initializer)
    lm_head = next(n for n in model.graph.node if "lm_head" in n.name)
    assert lm_head.input[1] == "embedding_weight_transpose"


def test_quantized_lmhead_and_embedding_share_compact_graph_contract():
    """The quant pass and tied input expose the same live compact blob."""
    from utils.export_quant import apply_shared_weight, quant_node_4bit_gp32
    from utils.omc_compiler import embedding_weight_elems

    weight = np.tile(np.arange(-8, 8, dtype=np.float16), (48, 2)).T
    initializer = onnx.numpy_helper.from_array(weight, "lm_head.weight")
    original = helper.make_node("MatMul", ["hidden", "lm_head.weight"], ["logits"], name="lm_head/MatMul")
    nodes, initializers = quant_node_4bit_gp32({"hidden": [1, 1, 32]}, original, {initializer.name: initializer})
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1]) for name in
              ("valid_seq_len", "lmhead_idx", "rope_cos", "rope_sin", "hidden", "attention_mask")]
    model = helper.make_model(helper.make_graph(nodes, "quant", inputs, [], initializers))
    payload = onnx.numpy_helper.to_array(initializers[0])
    assert payload.dtype == np.uint8
    assert payload.shape == (48 * 32 // 32 * 18,)
    np.testing.assert_array_equal(payload[48 * 32 // 2:].view("<f2"), np.ones(48, dtype=np.float16))
    apply_shared_weight(model, is_quant=True)
    embedding = model.graph.input[6]
    assert embedding.name == "embedding_weight"
    assert embedding.type.tensor_type.elem_type == TensorProto.UINT8
    assert [dim.dim_value for dim in embedding.type.tensor_type.shape.dim] == [
        embedding_weight_elems(48, 32, "W4A16")
    ]
    node = model.graph.node[0]
    assert (node.domain, node.op_type, list(node.input)) == (
        "custom", "MsQuant4N0Group32", ["hidden", "embedding_weight"]
    )
    assert helper.get_attribute_value(node.attribute[0]) == b"32,48"
    assert not model.graph.initializer


@pytest.mark.parametrize("is_prefill", [True, False])
def test_custom_matmul_shapes_decode_string_attributes(is_prefill):
    """QK uses cache length, while PV uses head dimension, including decode."""
    from utils.export_quant import custom_op_infer_shape

    seq_len = 8 if is_prefill else 1
    qk = helper.make_node(
        "MsGroupMatmul", ["query", "key"], ["scores"], domain="custom", trans_b="True"
    )
    pv = helper.make_node(
        "MsGroupMatmul", ["scores", "value"], ["context"], domain="custom", trans_b="False"
    )
    graph = helper.make_graph(
        [qk, pv], "attention", [],
        [helper.make_tensor_value_info("context", TensorProto.FLOAT16, [1])],
        value_info=[helper.make_tensor_value_info("scores", TensorProto.FLOAT16, [1])],
    )
    for _ in range(2):
        custom_op_infer_shape(graph, 8, 64, 2, 4, 16, is_prefill)

    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in (*graph.value_info, *graph.output)
    }
    assert shapes == {"scores": [1, 4, seq_len, 64], "context": [1, 4, seq_len, 16]}
    assert [value.name for value in graph.value_info] == ["scores"]


def test_custom_norm_preserves_per_head_key_shape():
    """Qwen3 key RMSNorm must not acquire flattened query hidden dimensions."""
    from utils.export_quant import custom_op_infer_shape

    node = helper.make_node("MsRmsNorm", ["key", "weight"], ["normalized"], domain="custom")
    graph = helper.make_graph(
        [node], "key_norm",
        [helper.make_tensor_value_info("key", TensorProto.FLOAT16, [1, 8, 2, 16])],
        [helper.make_tensor_value_info("normalized", TensorProto.FLOAT16, [1, 8, 2, 16])],
    )
    custom_op_infer_shape(graph, 8, 64, 2, 4, 16, True)
    assert [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim] == [1, 8, 2, 16]
    assert not graph.value_info


def test_quantized_model_retains_shape_and_model_metadata(tmp_path):
    """The saved quantized graph keeps custom shapes and its original model contract."""
    from utils.export_quant import quantize_linear_ops

    norm = helper.make_node(
        "MsRmsNorm", ["hidden", "norm.weight"], ["normalized"], domain="custom", epsilon=1e-6
    )
    lmhead = helper.make_node(
        "MatMul", ["normalized", "lm_head.weight"], ["logits"], name="/lm_head/MatMul"
    )
    graph = helper.make_graph(
        [norm, lmhead], "metadata",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 2, 32])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, 2, 16])],
        [
            onnx.numpy_helper.from_array(np.ones(32, dtype=np.float16), "norm.weight"),
            onnx.numpy_helper.from_array(np.ones((32, 16), dtype=np.float16), "lm_head.weight"),
        ],
        value_info=[helper.make_tensor_value_info("normalized", TensorProto.FLOAT16, [1, 2, 32])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("custom", 1)],
        producer_name="export-regression", producer_version="2", domain="lite_llm",
        model_version=7, doc_string="Quantized model metadata regression",
    )
    model.ir_version = 8
    helper.set_model_props(model, {"architecture": "qwen3"})
    quantized = quantize_linear_ops(model, QuantizationConfig("W4A16"), QuantizationConfig("W4A16"))
    path = tmp_path / "quantized.onnx"
    onnx.save(quantized, str(path))
    restored = onnx.load(str(path))

    onnx.checker.check_model(restored)
    assert restored.ir_version == 8
    assert {(opset.domain, opset.version) for opset in restored.opset_import} == {("", 17), ("custom", 1)}
    assert (restored.producer_name, restored.producer_version, restored.domain, restored.model_version) == (
        "export-regression", "2", "lite_llm", 7
    )
    assert restored.doc_string == "Quantized model metadata regression"
    assert {prop.key: prop.value for prop in restored.metadata_props} == {"architecture": "qwen3"}
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in restored.graph.value_info
    }
    assert shapes["normalized"] == [1, 2, 32]


def test_fuse_add_rmsnorm():
    """Add -> MsRmsNorm gets fused into MsAddRmsNorm."""
    from utils.onnx_postprocess import fuse_add_rmsnorm

    rms = helper.make_node(
        "MsRmsNorm",
        name="layer0/rmsnorm",
        inputs=["residual_add", "norm.weight"],
        outputs=["norm_out"],
        epsilon=1e-6,
    )
    add = helper.make_node(
        "Add",
        name="layer0/add",
        inputs=["hidden", "residual"],
        outputs=["residual_add"],
    )
    out = helper.make_tensor_value_info("norm_out", TensorProto.FLOAT16, [1, 128, 8])
    graph = helper.make_graph([add, rms], "g", [], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8

    path = str(Path(__file__).parent / "tmp_fuse_add_rmsnorm.onnx")
    onnx.save(model, path)
    fused = fuse_add_rmsnorm(path, path)
    os.remove(path)

    assert any(n.op_type == "MsAddRmsNorm" for n in fused.graph.node)
    assert not any(n.op_type == "MsRmsNorm" for n in fused.graph.node)


def _make_contract_model(num_layers=2, embedding_quant=False):
    """Build a synthetic ONNX matching the NNRT contract (names/order only)."""

    input_names = [
        "valid_seq_len",
        "lmhead_idx",
        "rope_cos",
        "rope_sin",
        "inputs_embeds",
        "attention_mask",
        "embedding_weight",
    ]
    for i in range(num_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_val_{i}")
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"out_key_{i}")
        output_names.append(f"out_val_{i}")

    emb_dtype = TensorProto.UINT8 if embedding_quant else TensorProto.FLOAT16
    inputs = [helper.make_tensor_value_info(name, emb_dtype if name == "embedding_weight" else TensorProto.FLOAT16, [1])
              for name in input_names]
    outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1]) for name in output_names]
    graph = helper.make_graph([], "g", inputs, outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    return model, input_names


def test_validate_contract_ok():
    """Test that a synthetic contract-conforming model passes validation."""
    from utils.onnx_postprocess import validate_contract

    model, _ = _make_contract_model(num_layers=2)
    path = str(Path(__file__).parent / "tmp_contract.onnx")
    onnx.save(model, path)
    try:
        assert validate_contract(path, num_layers=2, embedding_quant=False) is True
    finally:
        os.remove(path)


def test_validate_contract_quant_dtype():
    """validate_contract rejects an embedding dtype that contradicts config."""
    from utils.onnx_postprocess import validate_contract

    model, _ = _make_contract_model(num_layers=1, embedding_quant=True)
    path = str(Path(__file__).parent / "tmp_contract_quant.onnx")
    onnx.save(model, path)
    try:
        assert validate_contract(path, num_layers=1, embedding_quant=True) is True
        with pytest.raises(ValueError):
            validate_contract(path, num_layers=1, embedding_quant=False)  # dtype mismatch
    finally:
        os.remove(path)


def test_validate_contract_rejects_bad_order():
    """validate_contract rejects input order that violates the NNRT contract."""
    from utils.onnx_postprocess import validate_contract

    model, input_names = _make_contract_model(num_layers=1)
    # Swap the first two inputs to break the contract order.
    model.graph.input[0].name, model.graph.input[1].name = input_names[1], input_names[0]
    path = str(Path(__file__).parent / "tmp_contract_bad.onnx")
    onnx.save(model, path)
    try:
        with pytest.raises(ValueError):
            validate_contract(path, num_layers=1, embedding_quant=False)
    finally:
        os.remove(path)
