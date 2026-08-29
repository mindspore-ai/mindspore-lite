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

import pytest

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from utils.export_quant import QuantizationConfig  # noqa: E402

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


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


def test_apply_shared_weight_quant_input_is_uint8():
    """Quantized embedding input follows the NNRT UINT8 contract."""
    from utils.export_quant import apply_shared_weight

    model = _make_lmhead_graph()
    apply_shared_weight(model, is_quant=True)

    names = [vi.name for vi in model.graph.input]
    assert names.index("embedding_weight") == 6
    emb = model.graph.input[6]
    assert emb.type.tensor_type.elem_type == TensorProto.UINT8


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


def test_build_config_emits_manifest_fragment():
    """Qwen2Onnx.build_config emits the packager-consumable manifest fragment."""
    from models.qwen2_5.qwen2_5_exporter import Qwen2Onnx

    class FakeArch:
        """Minimal arch stand-in exposing the config fields Qwen2Onnx reads."""

        _name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
        num_hidden_layers = 24
        hidden_size = 896
        intermediate_size = 4864
        num_attention_heads = 14
        num_key_value_heads = 2
        vocab_size = 151936
        max_position_embeddings = 32768
        rope_theta = 1000000.0
        rms_norm_eps = 1e-6
        tie_word_embeddings = True
        eos_token_id = 151645

    class FakeModel:
        config = FakeArch()

    exporter = Qwen2Onnx.__new__(Qwen2Onnx)
    exporter.model = FakeModel()
    exporter.config = FakeArch()

    config = exporter.build_config(1024, 128, QuantizationConfig(None), QuantizationConfig(None))
    assert config["architecture"]["num_kv_heads"] == 2
    assert config["architecture"]["head_dim"] == 64
    assert config["generation"]["stop_token_ids"] == [151645]
    assert config["npu"] == {"max_length": 1024, "chunk_size": 128, "embedding_quant": None, "decoder_quant": None}
    assert config["sampling"]["eos_id"] == 151645


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
