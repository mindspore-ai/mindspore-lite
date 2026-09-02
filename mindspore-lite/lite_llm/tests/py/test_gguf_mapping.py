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
"""Tests for type-aware GGUF embedding conversion."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("gguf")

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from gguf.quants import GGMLQuantizationType  # noqa: E402
from utils import gguf_mapping  # noqa: E402
from utils.export_quant import quantize_weight_g32_4bit_nd  # noqa: E402


def _embedding(data, tensor_type):
    return SimpleNamespace(
        name="token_embd.weight", data=data, tensor_type=tensor_type
    )


def test_q4_embedding_preserves_direct_rearrangement(tmp_path):
    """Q4_0 keeps the existing byte-preserving rearrangement path."""
    scale = np.array([0.5], dtype=np.float16).view(np.uint8)
    qweight = np.arange(16, dtype=np.uint8)
    q4_block = np.concatenate((scale, qweight))
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(q4_block, GGMLQuantizationType.Q4_0)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="W4A16",
    )

    expected = gguf_mapping.rearrange_q4_0_g32(q4_block)
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    np.testing.assert_array_equal(np.fromfile(output_path, dtype=np.uint8), expected)


def test_fp16_embedding_is_requantized_to_w4a16(tmp_path):
    """A non-Q4 embedding matches host dequantize-then-quantize conversion."""
    fp16 = np.linspace(-1.0, 1.0, num=16 * 32, dtype=np.float16).reshape(16, 32)
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(fp16, GGMLQuantizationType.F16)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="W4A16",
    )

    expected = quantize_weight_g32_4bit_nd(fp16.T)
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    np.testing.assert_array_equal(np.fromfile(output_path, dtype=np.uint8), expected)


def test_quantized_embedding_is_dequantized_for_fp16(monkeypatch, tmp_path):
    """Quantized GGUF embeddings are materialized as FP16 when requested."""
    fp32 = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw = np.arange(8, dtype=np.uint8)
    monkeypatch.setattr(
        gguf_mapping,
        "dequantize",
        lambda data, tensor_type: fp32
        if data is raw and tensor_type == GGMLQuantizationType.Q8_0
        else None,
    )
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(raw, GGMLQuantizationType.Q8_0)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="FP16",
    )

    expected = fp32.astype(np.float16)
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    written = np.fromfile(output_path, dtype=np.float16).reshape(3, 4)
    np.testing.assert_array_equal(written, expected)
