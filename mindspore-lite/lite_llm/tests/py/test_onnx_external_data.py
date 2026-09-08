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
"""Tests for ONNX external tensor data handling."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, numpy_helper  # pylint: disable=wrong-import-position

EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(EXPORT_DIR))

from utils import export_quant  # pylint: disable=wrong-import-position


def test_quantization_loads_external_payload_after_shape_inference(monkeypatch, tmp_path):
    """Verify that quantization loads external payloads after shape inference."""
    tensor = numpy_helper.from_array(np.arange(8, dtype=np.float32), name="weight")
    graph = helper.make_graph([], "external", [], [], [tensor])
    model = helper.make_model(graph)
    input_path = tmp_path / "input.onnx"
    onnx.save_model(
        model, str(input_path), save_as_external_data=True, all_tensors_to_one_file=True,
        location="input.onnx.data", size_threshold=0,
    )

    def fake_infer(candidate, *_args, **_kwargs):
        assert any(
            initializer.data_location == onnx.TensorProto.EXTERNAL
            for initializer in candidate.graph.initializer
        )
        assert not candidate.graph.initializer[0].raw_data
        return candidate

    def fake_quantize(candidate, *_args, **_kwargs):
        assert candidate.graph.initializer[0].raw_data
        return candidate

    saved = []
    monkeypatch.setattr(export_quant, "infer_shape", fake_infer)
    monkeypatch.setattr(export_quant, "quantize_linear_ops", fake_quantize)
    monkeypatch.setattr(export_quant, "_save_onnx", lambda candidate, path: saved.append((candidate, path)))
    config = SimpleNamespace(
        chunk_size=1, max_length=1, num_attention_heads=1,
        num_key_value_heads=1, hidden_size=1, embedding_quant=None, decoder_quant=None,
    )

    output_path = tmp_path / "output.onnx"
    assert export_quant.apply_quant(str(input_path), str(output_path), config) == str(output_path)
    assert saved and saved[0][1] == str(output_path)
