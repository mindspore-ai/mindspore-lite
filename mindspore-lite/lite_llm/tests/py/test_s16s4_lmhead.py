"""Regression checks for S16S4 head quantization and shared graph inputs."""

import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "export"))
from utils.export_quant import (  # pylint: disable=wrong-import-position
    quantize_linear_ops,
    QuantizationConfig,
)  # pylint: disable=wrong-import-position
from utils.omc_compiler import build_omg_command  # pylint: disable=wrong-import-position
from utils.quantization import QuantType  # pylint: disable=wrong-import-position


@pytest.mark.parametrize("name,expected,group_size", [
    ("q4_0", QuantType.Q4_0, 32),
    ("W4A16", QuantType.Q4_0, 32),
    ("s16s4", QuantType.S16S4, 128),
    ("S16S4", QuantType.S16S4, 128),
    (QuantType.Q4_0, QuantType.Q4_0, 32),
    (QuantType.S16S4, QuantType.S16S4, 128),
])
def test_quantization_names(name, expected, group_size):
    """Legacy names normalize to canonical algorithms and serialized names."""
    import json

    config = QuantizationConfig(name)
    assert config.quant_method is expected
    assert config.group_size == group_size
    assert json.loads(json.dumps(config.asdict()))["quant_method"] == expected.value
    assert json.loads(json.dumps(expected.embedding_format)) == (
        "S16S4_NZ_V1" if expected is QuantType.S16S4 else "W4A16"
    )


@pytest.mark.parametrize("decoder_quant", ["W4A16", "S16S4", QuantType.Q4_0, QuantType.S16S4])
@pytest.mark.parametrize("embedding_quant", [None, "W4A16", "S16S4", QuantType.Q4_0, QuantType.S16S4])
def test_lmhead_follows_embedding_quant_config(embedding_quant, decoder_quant):
    """Decoder quantization must not override the embedding/head format."""
    x = helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, 1, 128])
    y = helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, 1, 256])
    weights = [
        numpy_helper.from_array(np.ones((128, 128), np.float16), "decoder.weight"),
        numpy_helper.from_array(np.ones((128, 256), np.float16), "head.weight"),
    ]
    nodes = [
        helper.make_node("MatMul", ["inputs_embeds", "decoder.weight"], ["hidden"], name="/decoder/MatMul"),
        helper.make_node("MatMul", ["hidden", "head.weight"], ["logits"], name="/lm_head/MatMul"),
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        [x],
        [y],
        weights,
        value_info=[x, helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 1, 128])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    quant = quantize_linear_ops(model, QuantizationConfig(embedding_quant), QuantizationConfig(decoder_quant))
    onnx.checker.check_model(quant)
    op_types = {None: "MatMul", QuantType.Q4_0: "MsQuant4N0Group32", QuantType.S16S4: "MsMatmulS16S4"}
    head = next(node for node in quant.graph.node if "lm_head" in node.name and node.op_type in op_types.values())
    assert head.op_type == op_types[QuantType.parse(embedding_quant)]
    decoder = next(node for node in quant.graph.node if "/decoder/" in node.name and node.op_type in op_types.values())
    assert decoder.op_type == op_types[QuantType.parse(decoder_quant)]
    if QuantType.parse(embedding_quant) is QuantType.S16S4:
        initializers = {i.name: i for i in quant.graph.initializer}
        assert all(name in initializers for name in head.input[1:])
        assert numpy_helper.to_array(initializers[head.input[2]]).shape == (256, 64)
    assert "embedding_weight" not in {i.name for i in quant.graph.input}


def test_omg_contract_shared_weight():
    """The compiler always includes the shared embedding input."""
    config = {
        "vocab_size": 151936,
        "num_layers": 36,
        "hidden_size": 2560,
        "num_kv_heads": 8,
        "num_heads": 32,
        "head_dim": 128,
    }
    cmd = build_omg_command(
        "in.onnx", "out", config, 1024, (1, 64), "W4A16", omg="/omg"
    )
    shapes = next((arg.split("=", 1)[1] for arg in cmd if arg.startswith("--input_shape="))).split(";")
    assert len(shapes) == 72 + 7
    assert shapes[7].startswith("past_key_0:")
    assert any(s.startswith("embedding_weight:") for s in shapes)


@pytest.mark.parametrize("embedding_format", ["W4A16", "S16S4_NZ_V1"])
def test_graph_contract_shared_embedding_and_kv_order(tmp_path, embedding_format):
    """Require shared inputs and interleaved KV order for both formats."""
    from utils.onnx_postprocess import validate_contract

    names = ["valid_seq_len", "lmhead_idx", "rope_cos", "rope_sin", "inputs_embeds", "attention_mask"]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1]) for name in names]
    is_s16s4 = embedding_format == "S16S4_NZ_V1"
    dtype = TensorProto.INT8 if is_s16s4 else TensorProto.UINT8
    inputs.append(helper.make_tensor_value_info("embedding_weight", dtype, [128]))
    if is_s16s4:
        inputs.append(helper.make_tensor_value_info("embedding_scale", TensorProto.INT8, [128]))
    fixed_inputs = len(inputs)
    for name in ["past_key_0", "past_val_0"]:
        inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2, 64, 128]))
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, 1, 256])]
    outputs += [
        helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2, 64, 128])
        for name in ["out_key_0", "out_val_0"]
    ]
    model = helper.make_model(helper.make_graph([], "contract", inputs, outputs))
    path = tmp_path / "contract.onnx"
    onnx.save(model, path)
    args = {
        "embedding_quant": True,
        "embedding_format": embedding_format,
    }
    assert validate_contract(str(path), 1, **args)
    model.graph.input[fixed_inputs].name = "past_val_0"
    onnx.save(model, path)
    with pytest.raises(ValueError, match="inputs"):
        validate_contract(str(path), 1, **args)

    model.graph.input[fixed_inputs].name = "past_key_0"
    del model.graph.input[6:fixed_inputs]
    onnx.save(model, path)
    with pytest.raises(ValueError, match="inputs"):
        validate_contract(str(path), 1, **args)


@pytest.mark.parametrize("hidden,vocab", [(128, 256), (256, 384)])
def test_shared_s16s4_asset_matches_head(tmp_path, hidden, vocab):
    """Shared inputs replace head constants without changing packed bytes."""
    from utils.export_quant import quant_node_s16s4
    from utils.export_quant import apply_shared_weight
    from utils.msl_pack import build_manifest, manifest_to_kv
    weights = numpy_helper.from_array(np.ones((hidden, vocab), np.float16), "head.weight")
    nodes, constants = quant_node_s16s4(
        helper.make_node("MatMul", ["inputs_embeds", weights.name], ["y"], name="/lm_head/MatMul"),
        {weights.name: weights},
    )
    expected = numpy_helper.to_array(constants[1]).tobytes() + numpy_helper.to_array(constants[3]).tobytes()
    names = ["valid_seq_len", "lmhead_idx", "rope_cos", "rope_sin", "inputs_embeds", "attention_mask"]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1]) for name in names]
    model = helper.make_model(helper.make_graph(nodes, "shared", inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 1, vocab])], constants),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("custom", 1)])
    asset = tmp_path / "embedding.bin"
    embedding_asset = apply_shared_weight(model)
    assert embedding_asset.quant_type == QuantType.S16S4
    assert embedding_asset.logical_shape == (vocab, hidden)
    assert len(embedding_asset.buffers) == 2
    assert not asset.exists()
    embedding_asset.save(asset)
    onnx.checker.check_model(model)
    assert asset.read_bytes() == expected
    assert [value.name for value in model.graph.input[6:]] == ["embedding_weight", "embedding_scale"]
    head = next(node for node in model.graph.node if node.op_type == "MsMatmulS16S4")
    remaining = {value.name for value in model.graph.initializer}
    assert head.input[1] in remaining and head.input[3] in remaining
    assert head.input[2] not in remaining and head.input[4] not in remaining
    manifest = build_manifest("test", {"vocab_size": vocab, "hidden_size": hidden}, {"max_length": 64, "chunk_size": 1,
        "embedding_quant": True, "scale_gp_size": 128, "embedding_format": "S16S4_NZ_V1"}, {}, "model.omc")
    assert manifest_to_kv(manifest)["npu.embedding_format"] == "S16S4_NZ_V1"


def test_shared_s16s4_compiler_shapes():
    """The shared head adds two ranked inputs before the KV caches."""
    config = {"vocab_size": 151936, "num_layers": 36, "hidden_size": 2560,
              "num_kv_heads": 8, "num_heads": 32, "head_dim": 128}
    cmd = build_omg_command("in", "out", config, 1024, (1, 64), "S16S4",
                            platform="kirin9030", omg="/omg")
    shapes = next(arg for arg in cmd if arg.startswith("--input_shape="))
    assert "embedding_weight:151936,1280;embedding_scale:1187,20,128,8;past_key_0" in shapes


def test_compiler_rejects_unsupported_quant_target():
    """Direct compilation must reject an unsupported kernel before invoking omg."""
    with pytest.raises(ValueError, match="kirin9030"):
        build_omg_command("in", "out", {}, 1024, (1, 64), "s16s4",
                          platform="kirin9020", omg="/omg")


@pytest.mark.parametrize("quant_type", [QuantType.Q4_0, QuantType.W4A8])
def test_packed_embedding_matches_compiler_size(quant_type):
    """Compiler shapes must account for the actual packed weights and scales."""
    from utils.export_quant import quantize_weight_g128_4bit_nz
    from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32
    from utils.omc_compiler import embedding_weight_elems

    weight = np.arange(256 * 128, dtype=np.float32).reshape(256, 128) / 1024
    pack = (MsQuant4N0Group32.quantize_weight_g32_4bit
            if quant_type is QuantType.Q4_0 else quantize_weight_g128_4bit_nz)
    assert pack(weight).nbytes == embedding_weight_elems(128, 256, quant_type)


def test_preset_cannot_override_group_size():
    """A caller cannot silently request a group size unsupported by the kernel."""
    from dataclasses import FrozenInstanceError
    from utils.quantization import get_quant_preset

    preset = get_quant_preset("W4A16")
    assert preset is get_quant_preset("q4_0")
    with pytest.raises(FrozenInstanceError):
        preset.group_size = 128
