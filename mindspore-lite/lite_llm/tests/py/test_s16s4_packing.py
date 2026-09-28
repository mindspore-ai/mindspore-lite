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
        matmul, {initializer.name: initializer}, activation_scale=128.0
    )
    assert len(nodes) == 1
    node = next(node for node in nodes if node.op_type == "MsMatmulS16S4")
    assert node.input[0] == "hidden" and list(node.output) == ["out"]
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    assert node.op_type == "MsMatmulS16S4"
    assert node.domain == "custom"
    assert len(node.input) == 5
    assert attrs == {"group_size": 128, "w_shape": [768, 128]}
    arrays = {init.name: onnx.numpy_helper.to_array(init) for init in initializers}
    assert arrays["decoder.weight_s16s4_x_scale"].tolist() == [128.0]
    assert arrays["decoder.weight_quant"].shape == (128, 384)
    assert arrays["decoder.weight_s16s4_w_scale"].shape == (1, 6, 128, 8)


@pytest.mark.parametrize("activation_scale", [16.0, 32.0, 128.0, 512.0, 1024.0])
def test_static_scale_updates_bias_and_fixpipe_together(activation_scale):
    """Changing activation range preserves weights and compensation algebra."""
    weight = np.random.default_rng(37).normal(0, 0.1, (256, 128)).astype(np.float16)
    baseline = quantize_weight_s16s4_rtn128(weight)
    result = quantize_weight_s16s4_rtn128(weight, activation_scale)
    np.testing.assert_array_equal(result["w"], baseline["w"])
    np.testing.assert_array_equal(result["scales"], baseline["scales"])
    assert result["x_scale"].tolist() == [activation_scale]
    np.testing.assert_allclose(result["w_scale"].view(np.float32) * activation_scale,
                               baseline["w_scale"].view(np.float32) * 1024, rtol=0, atol=0)
    np.testing.assert_allclose(result["bias"].astype(np.float32) * activation_scale,
                               baseline["bias"].astype(np.float32) * 1024, rtol=0.001, atol=0.001)


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, 65536, 1e-12, 1.0001])
def test_static_scale_rejects_invalid_fp16_values(scale):
    with pytest.raises(ValueError, match="activation scale"):
        quantize_weight_s16s4_rtn128(np.ones((128, 128), np.float16), scale)


def test_static_scale_rejects_correction_bias_overflow():
    """A representable multiplier must also produce finite FP16 compensation."""
    with pytest.raises(ValueError, match="correction bias overflow"):
        quantize_weight_s16s4_rtn128(np.ones((128, 128), np.float16), 0.125)
