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
"""Independent scalar checks for compact phase4 Q4_0 and canonical quantization."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf.quants import GGMLQuantizationType, dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "export"))

from utils import ensure_custom_ops  # pylint: disable=wrong-import-position
from utils.gguf_mapping import rearrange_q4_0_g32  # pylint: disable=wrong-import-position

ensure_custom_ops()
from torch_custom.ms_quant4_n0_group32 import MsQuant4N0Group32  # pylint: disable=wrong-import-position


def _gguf_blocks(n, k):
    """GGUF rows with varying nibble lanes and signed per-row/group scales."""
    groups = k // 32
    row = np.arange(n)[:, None, None]
    group = np.arange(groups)[None, :, None]
    lane = np.arange(16)[None, None, :]
    low = (row * 3 + group * 5 + lane) % 16
    high = (row * 7 + group * 3 + lane * 5 + 1) % 16
    scales = ((1 + np.arange(n)[:, None] * 3 + np.arange(groups)[None, :]) / 64).astype("<f2")
    scales[::2] *= -1
    blocks = np.empty((n, groups, 18), dtype=np.uint8)
    blocks[..., :2] = scales[..., None].view(np.uint8)
    blocks[..., 2:] = (low | (high << 4)).astype(np.uint8)
    return blocks.reshape(n, groups * 18), scales


def _decode_scalar(payload, n, k):
    """Use byte addresses, independently of the vectorized cell transforms."""
    assert payload.dtype == np.uint8
    assert payload.shape == (n * k // 32 * 18,)
    decoded = np.empty((n, k), dtype=np.float32)
    scale_bits = np.empty((n, k // 32), dtype="<u2")
    for row in range(n):
        n0 = row // 64 * 64
        nc = min(64, n - n0)
        for col in range(k):
            k0 = col // 1024 * 1024
            kc = min(1024, k - k0)
            cell_start = n0 * k // 2 + nc * k0 // 2
            lane = row % 16 * 16 + col % 16
            pair = lane // 2
            fractal = ((col - k0) // 16 * (nc // 16) + (row - n0) // 16) * 128
            address = cell_start + fractal + 4 * (pair % 32) + pair // 32
            nibble = (int(payload[address]) >> (4 * (lane % 2))) & 15
            signed = nibble - 16 if nibble >= 8 else nibble
            scale_start = n * k // 2 + n0 * k // 16 + nc * k0 // 16
            scale_address = scale_start + ((row - n0) * (kc // 32) + (col - k0) // 32) * 2
            bits = int(payload[scale_address]) | int(payload[scale_address + 1]) << 8
            scale_bits[row, col // 32] = bits
            scale = np.array([bits], dtype="<u2").view("<f2")[0]
            decoded[row, col] = np.float32(signed) * np.float32(scale)
    return decoded, scale_bits


@pytest.mark.parametrize("n,k", [(16, 32), (48, 896), (64, 1024), (80, 1056)])
@pytest.mark.parametrize("source", ["gguf", "blocks"])
def test_nzf_matches_gguf_dequantization_and_scale_bits(n, k, source):
    """Live N/K tails, all nibble phases and signed scales remain bit exact."""
    rows, scales = _gguf_blocks(n, k)
    if source == "gguf":
        payload = rearrange_q4_0_g32(rows)
    else:
        payload = MsQuant4N0Group32.repack_q4_0_to_nzf(rows, (k, n))
    actual, scale_bits = _decode_scalar(payload, n, k)
    expected = dequantize(rows, GGMLQuantizationType.Q4_0)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(scale_bits, scales.view(np.uint16))
    eager = MsQuant4N0Group32.dequantize_weight_g32_4bit(payload, (k, n)).T
    np.testing.assert_array_equal(eager.view(np.uint16), expected.astype(np.float16).view(np.uint16))


@pytest.mark.parametrize("source", ["gguf", "blocks"])
def test_nzf_preserves_every_fp16_scale_bit_pattern(source):
    """Repacking cannot canonicalize signed zero, subnormals or NaN payloads."""
    scale_bits = np.arange(65536, dtype="<u2").reshape(2048, 32)
    blocks = np.full((2048, 32, 18), 0x88, dtype=np.uint8)
    blocks[..., :2] = scale_bits[..., None].view(np.uint8)
    if source == "gguf":
        payload = rearrange_q4_0_g32(blocks.reshape(2048, 32 * 18))
    else:
        payload = MsQuant4N0Group32.repack_q4_0_to_nzf(blocks, (1024, 2048))
    np.testing.assert_array_equal(payload[2048 * 1024 // 2:].view("<u2"), scale_bits.reshape(-1))
    assert not np.any(payload[:2048 * 1024 // 2])


def _quantize_scalar(weight):
    """Literal llama.cpp block reference, without vector reductions or packing."""
    k, n = weight.shape
    result = np.empty((n, k // 32, 18), dtype=np.uint8)
    for row in range(n):
        for group in range(k // 32):
            values = weight[group * 32:(group + 1) * 32, row].astype(np.float32)
            vmax = np.float32(0)
            for value in values:
                if abs(value) > abs(vmax):
                    vmax = value
            scale = np.float32(vmax / np.float32(-8))
            reciprocal = np.float32(1) / scale if scale != 0 else np.float32(0)
            result[row, group, :2] = np.array([scale], dtype="<f2").view(np.uint8)
            codes = []
            for value in values:
                shifted = np.float32(np.float32(value * reciprocal) + np.float32(8.5))
                codes.append(min(15, int(shifted)))
            for lane in range(16):
                result[row, group, lane + 2] = codes[lane] | codes[lane + 16] << 4
    return result.reshape(-1)


def test_canonical_float_quantization_uses_first_extremum_and_fp32_codes():
    """Tied signs, halfway rounding, FP16 scale rounding and underflow differ."""
    rows = np.tile(np.linspace(-8, 7, 32, dtype=np.float32), (16, 1))
    rows[0, :4] = [8, -8, 0.5, -0.5]
    rows[1, :4] = [-8, 8, 0.5, -0.5]
    rows[2] = np.float32(-0.0)
    rows[3] *= np.float32(1.0003)
    rows[3, 1] = np.float32(0.5001)
    rows[4] *= np.float32(1e-9)
    expected = _quantize_scalar(rows.T)
    blocks = MsQuant4N0Group32.quantize_q4_0_blocks(rows.T)
    np.testing.assert_array_equal(blocks, expected)
    blob = MsQuant4N0Group32.quantize_weight_g32_4bit(rows.T)
    decoded, scale_bits = _decode_scalar(blob, 16, 32)
    reference = dequantize(expected.reshape(16, 18), GGMLQuantizationType.Q4_0)
    np.testing.assert_array_equal(decoded.view(np.uint32), reference.view(np.uint32))
    assert scale_bits[0, 0] == 0xBC00
    assert scale_bits[1, 0] == 0x3C00
    assert scale_bits[2, 0] == 0x8000


def test_zero_quantization_has_zero_codes_and_negative_zero_scales():
    """Zero groups follow canonical Q4_0 even in both compact tail dimensions."""
    payload = MsQuant4N0Group32.quantize_weight_g32_4bit(np.zeros((1056, 80), dtype=np.float16))
    decoded, scale_bits = _decode_scalar(payload, 80, 1056)
    assert np.isfinite(decoded).all()
    assert not np.any(decoded)
    assert np.all(scale_bits == 0x8000)
    assert not np.any(payload[:80 * 1056 // 2])


@pytest.mark.parametrize("representation", ["bytes", "readonly", "strided"])
def test_repack_accepts_byte_streams_without_mutation(representation):
    """Noncontiguous and immutable ingress must produce identical live cells."""
    rows, _ = _gguf_blocks(80, 1056)
    expected = MsQuant4N0Group32.repack_q4_0_to_nzf(rows, (1056, 80))
    storage = np.full((80, rows.shape[1] * 2), 0xCD, dtype=np.uint8)
    storage[:, ::2] = rows
    variants = {"bytes": rows.tobytes(), "readonly": rows, "strided": storage[:, ::2]}
    rows.flags.writeable = False
    storage.flags.writeable = False
    actual = MsQuant4N0Group32.repack_q4_0_to_nzf(variants[representation], (1056, 80))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(storage[:, ::2], rows)
    assert np.all(storage[:, 1::2] == 0xCD)


def test_quantize_accepts_readonly_strided_floating_weights():
    """HF transpose/view inputs must not be mutated or require contiguous K."""
    storage = np.arange(64 * 32, dtype=np.float32).reshape(64, 32) / np.float32(511)
    storage.flags.writeable = False
    weight = storage[::2, ::2]
    original = storage.copy()
    blocks = MsQuant4N0Group32.quantize_q4_0_blocks(weight)
    np.testing.assert_array_equal(blocks, _quantize_scalar(weight))
    np.testing.assert_array_equal(storage, original)


@pytest.mark.parametrize("shape", [(0, 16), (32, 0), (31, 16), (33, 16), (32, 1), (32, 17),
                                  (32, 1.5), (32.0, 16), (True, 16), (32, False), (32,), "32,16"])
def test_repack_rejects_invalid_logical_dimensions(shape):
    with pytest.raises(ValueError):
        MsQuant4N0Group32.repack_q4_0_to_nzf(bytes(288), shape)


@pytest.mark.parametrize("shape", [(0, 16), (32, 0), (31, 16), (32, 17), (32,), (32, 16, 1)])
def test_quantize_rejects_invalid_matrix_shapes(shape):
    with pytest.raises(ValueError):
        MsQuant4N0Group32.quantize_weight_g32_4bit(np.zeros(shape, dtype=np.float16))


@pytest.mark.parametrize("data", [bytes(287), bytes(289), np.zeros(288, np.float32), [0] * 288])
@pytest.mark.parametrize("method", ["repack_q4_0_to_nzf", "dequantize_weight_g32_4bit"])
def test_weight_ingress_rejects_wrong_dtype_or_byte_length(data, method):
    with pytest.raises(ValueError):
        getattr(MsQuant4N0Group32, method)(data, (32, 16))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_quantize_rejects_nonfinite_values(value):
    weight = np.zeros((32, 16), dtype=np.float32)
    weight[5, 7] = value
    with pytest.raises(ValueError):
        MsQuant4N0Group32.quantize_q4_0_blocks(weight)


@pytest.mark.parametrize("dtype", [np.uint8, np.int32, np.complex64])
def test_quantize_rejects_nonfloating_matrices(dtype):
    with pytest.raises(ValueError):
        MsQuant4N0Group32.quantize_q4_0_blocks(np.zeros((32, 16), dtype=dtype))


@pytest.mark.parametrize("shape", [(18,), (1, 1, 18), (16, 19), (16, 17)])
def test_rearrange_rejects_non_block_rows(shape):
    with pytest.raises(ValueError):
        rearrange_q4_0_g32(np.zeros(shape, np.uint8))


def test_eager_consumes_compact_tail_weights_and_rejects_padded_blob():
    """The real operator evaluates compact weights, not old padded/S1 bytes."""
    rows, _ = _gguf_blocks(48, 32)
    blob = rearrange_q4_0_g32(rows)
    identity = torch.eye(32, dtype=torch.float16)
    result = MsQuant4N0Group32.apply(identity, torch.from_numpy(blob), (32, 48))
    expected = dequantize(rows, GGMLQuantizationType.Q4_0).T.astype(np.float16)
    torch.testing.assert_close(result, torch.from_numpy(expected), rtol=0, atol=0)
    with pytest.raises(ValueError):
        MsQuant4N0Group32.apply(identity, torch.zeros(36864, dtype=torch.uint8), (32, 48))


class _CompactMatmul(torch.nn.Module):
    """Export through the actual autograd symbolic rather than a fake graph."""

    def forward(self, hidden, weight):
        return MsQuant4N0Group32.apply(hidden, weight, "32,48")


def test_symbolic_preserves_custom_op_inputs_shape_and_output(tmp_path):
    """Compact packing must not change the ONNX operator/attribute contract."""
    import onnx  # pylint: disable=import-outside-toplevel

    rows, _ = _gguf_blocks(48, 32)
    blob = MsQuant4N0Group32.repack_q4_0_to_nzf(rows, (32, 48))
    path = tmp_path / "compact.onnx"
    torch.onnx.export(
        _CompactMatmul(), (torch.ones((1, 32), dtype=torch.float16), torch.from_numpy(blob)),
        str(path), input_names=["hidden", "weight"], output_names=["result"], opset_version=18, dynamo=False,
    )
    graph = onnx.load(path).graph
    node = next(node for node in graph.node if node.op_type == "MsQuant4N0Group32")
    assert node.domain == "custom"
    assert list(node.input) == ["hidden", "weight"]
    assert onnx.helper.get_attribute_value(node.attribute[0]) == b"32,48"
    assert graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim] == [1, 48]
