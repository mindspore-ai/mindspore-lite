"""Host-side checks for the Kirin 9030 MatmulS16S4 materializer."""

import sys
from pathlib import Path

import numpy as np
import pytest

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

onnx = pytest.importorskip("onnx")
from onnx import helper, numpy_helper  # pylint: disable=wrong-import-position

from utils.export_quant import (  # pylint: disable=wrong-import-position
    pack_s16s4_nz,
    quantize_weight_s16s4_rtn128,
    unpack_s16s4_nz,
)
from utils.export_quant import QuantizationConfig  # pylint: disable=wrong-import-position
from utils.export_quant import quant_node_s16s4  # pylint: disable=wrong-import-position


@pytest.mark.parametrize("k_dim,n_dim", [(768, 384), (2432, 768)])
def test_minimind_signed_s4_pack_round_trip(k_dim, n_dim):
    rng = np.random.default_rng(k_dim + n_dim)
    logical = rng.integers(-8, 8, size=(k_dim, n_dim), dtype=np.int8)
    packed = pack_s16s4_nz(logical)
    assert packed.dtype == np.int8
    assert packed.shape == (n_dim, k_dim // 2)
    np.testing.assert_array_equal(unpack_s16s4_nz(packed, k_dim, n_dim), logical)


def test_rtn128_materializer_shapes_and_fixpipe_records():
    """Validate materialized weights and FixPipe scale records."""
    rng = np.random.default_rng(7)
    weight = rng.normal(0.0, 0.1, size=(768, 384)).astype(np.float16)
    result = quantize_weight_s16s4_rtn128(weight)

    assert result["x_scale"].dtype == np.float16
    assert result["x_scale"].tolist() == [1024.0]
    assert result["w"].shape == (384, 384)
    assert result["bias"].shape == (384,)
    assert result["w_scale"].shape == (3, 6, 128, 8)

    records = result["w_scale"].view(np.float32).reshape(3, 6, 128, 2)
    effective = records[..., 0].transpose(1, 0, 2).reshape(6, 384)
    np.testing.assert_array_equal(records[..., 1], 0.0)
    np.testing.assert_allclose(
        effective, result["scales"].astype(np.float32) / 1024.0, rtol=0, atol=0
    )
    recovered = unpack_s16s4_nz(result["w"], 768, 384).T
    np.testing.assert_array_equal(recovered, result["weight_s4"])


def test_zero_weight_is_deterministic_and_finite():
    result = quantize_weight_s16s4_rtn128(
        np.zeros((768, 128), dtype=np.float16)
    )
    assert not np.any(result["w"])
    assert not np.any(result["bias"])
    assert not np.any(result["w_scale"].view(np.float32))


def test_s16s4_quant_config_is_group128():
    config = QuantizationConfig("S16S4")
    assert config.is_quant
    assert config.bits == 4
    assert config.group_size == 128


def test_quant_node_has_five_input_abi_and_attrs():
    """Check the five-input S16S4 node contract."""
    weight = np.ones((768, 128), dtype=np.float16)
    initializer = numpy_helper.from_array(weight, "decoder.weight")
    matmul = helper.make_node(
        "MatMul", ["hidden", initializer.name], ["out"], name="decoder/MatMul"
    )
    nodes, initializers = quant_node_s16s4(
        matmul, {initializer.name: initializer}
    )
    node = next(node for node in nodes if node.op_type == "MsMatmulS16S4")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    assert node.op_type == "MsMatmulS16S4"
    assert node.domain == "custom"
    assert len(node.input) == 5
    assert attrs == {"group_size": 128, "w_shape": [768, 128]}
    arrays = {init.name: onnx.numpy_helper.to_array(init) for init in initializers}
    assert arrays["decoder.weight_quant"].shape == (128, 384)
    assert arrays["decoder.weight_s16s4_w_scale"].shape == (1, 6, 128, 8)


@pytest.mark.parametrize("shape", [(1, 1, 768), (1, 64, 2560)])
def test_dynamic_activation_prevents_saturation(shape):
    """Execute exported standard ops on outliers, zero rows and ordinary rows."""
    from onnx.reference import ReferenceEvaluator
    from utils.export_quant import _s16s4_dynamic_activation

    node = helper.make_node("MsMatmulS16S4", ["x", "xs", "w", "b", "ws"], ["y"],
                            name="projection", domain="custom")
    nodes, constants = _s16s4_dynamic_activation(node)
    outputs = [
        helper.make_tensor_value_info(node.input[0], onnx.TensorProto.FLOAT16, shape),
        helper.make_tensor_value_info(nodes[-1].input[1], onnx.TensorProto.FLOAT16, [*shape[:-1], 1]),
    ]
    graph = helper.make_graph(nodes[:-2], "normalize", [
        helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, shape)
    ], outputs, constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    rng = np.random.default_rng(19)
    x = rng.normal(size=shape).astype(np.float16)
    x[0, 0, 0], x[0, 0, 1] = 1444, -660
    normalized, scale = evaluator.run(None, {"x": x})
    assert np.max(np.abs(normalized)) <= 16
    assert np.isfinite(normalized * np.float16(1024)).all()
    restored = np.rint((normalized * np.float16(1024)).astype(np.float32)) / 1024 * scale
    reference = x.astype(np.float32)
    assert np.linalg.norm(restored - reference) / np.linalg.norm(reference) < 0.002
    if shape[1] > 1:
        np.testing.assert_array_equal(normalized[:, 1:], x[:, 1:])
    zero, zero_scale = evaluator.run(None, {"x": np.zeros(shape, np.float16)})
    np.testing.assert_array_equal(zero, 0)
    np.testing.assert_array_equal(zero_scale, 1)
