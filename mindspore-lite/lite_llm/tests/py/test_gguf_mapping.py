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

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from gguf.quants import GGMLQuantizationType, dequantize  # noqa: E402
from utils import gguf_mapping  # noqa: E402


def _embedding(data, tensor_type):
    return SimpleNamespace(
        name="token_embd.weight", data=data, tensor_type=tensor_type
    )


def test_q4_embedding_preserves_direct_rearrangement(tmp_path):
    """Q4_0 preserves scales and signed values in phase4 wire format."""
    scale = np.array([0.5], dtype=np.float16).view(np.uint8)
    qweight = np.arange(16, dtype=np.uint8)
    q4_block = np.tile(np.concatenate((scale, qweight)), (16, 1))
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(q4_block, GGMLQuantizationType.Q4_0),
         SimpleNamespace(name="blk.0.attn_q.weight", data=q4_block, tensor_type=GGMLQuantizationType.Q4_0)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="W4A16",
    )

    # Repeated rows occupy both live K16 fractals, with no N64/K1024 padding.
    expected = np.empty(288, dtype=np.uint8)
    pairs = np.array([0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76], dtype=np.uint8)
    expected[:128] = np.tile(np.repeat(pairs, 4), 4)
    expected[128:256] = 0x88
    expected[256:] = np.tile(scale, 16)
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    np.testing.assert_array_equal(weights["blk.0.attn_q.weight"], expected)
    np.testing.assert_array_equal(np.fromfile(output_path, dtype=np.uint8), expected)


def test_fp16_embedding_is_requantized_to_w4a16(tmp_path):
    """A non-Q4 embedding matches host dequantize-then-quantize conversion."""
    fp16 = np.tile(np.arange(-8, 8, dtype=np.float16), (16, 2))
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(fp16, GGMLQuantizationType.F16)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="W4A16",
    )

    expected = np.empty(288, dtype=np.uint8)
    pairs = np.array([0x98, 0xBA, 0xDC, 0xFE, 0x10, 0x32, 0x54, 0x76], dtype=np.uint8)
    expected[:128] = np.tile(np.repeat(pairs, 4), 4)
    expected[128:256] = np.tile(np.repeat(pairs, 4), 4)
    expected[256:] = np.tile(np.array([1.0], dtype="<f2").view(np.uint8), 16)
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    np.testing.assert_array_equal(np.fromfile(output_path, dtype=np.uint8), expected)


def test_quantized_embedding_is_dequantized_for_fp16(tmp_path):
    """Quantized GGUF embeddings are materialized as FP16 when requested."""
    scales = np.array([0.5, -0.25, 2.0], dtype="<f2")
    values = (np.arange(3 * 32).reshape(3, 32) - 48).astype(np.int8)
    raw = np.empty((3, 34), dtype=np.uint8)
    raw[:, :2] = scales[:, None].view(np.uint8)
    raw[:, 2:] = values.view(np.uint8)
    output_path = tmp_path / "embedding.bin"

    weights = gguf_mapping.load_file_from_tensors(
        [_embedding(raw, GGMLQuantizationType.Q8_0)],
        output_path,
        decoder_quantize_config="W4A16",
        embedding_quantize_config="FP16",
    )

    expected = dequantize(raw, GGMLQuantizationType.Q8_0).astype(np.float16)
    np.testing.assert_array_equal(expected, values.astype(np.float16) * scales[:, None])
    np.testing.assert_array_equal(weights["token_embd.weight"], expected)
    written = np.fromfile(output_path, dtype=np.float16).reshape(3, 32)
    np.testing.assert_array_equal(written, expected)
