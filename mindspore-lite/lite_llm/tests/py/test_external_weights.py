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
"""Tests for opt-in external OMC weights."""

from pathlib import Path
import sys

import pytest

pytest.importorskip("onnx")

EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(EXPORT_DIR))

from utils import msl_pack, omc_compiler  # pylint: disable=wrong-import-position


def _architecture():
    return {
        "num_layers": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 16,
        "vocab_size": 64,
        "max_position_embeddings": 128,
    }


def test_omg_external_weight_flag_and_output_contract(monkeypatch, tmp_path):
    """Verify that external weights are requested from OMG only when enabled."""
    calls = []
    monkeypatch.setattr(omc_compiler.subprocess, "run", lambda command, check: calls.append((command, check)))
    monkeypatch.setattr(omc_compiler, "resolve_omg", lambda: "omg")
    output = tmp_path / "model"

    result = omc_compiler.compile_omc(
        "input.onnx", _architecture(), omc_path=str(output),
        save_external_weights=True,
    )

    assert calls[0][1] is True
    assert "--save_weights_as_external_data=true" in calls[0][0]
    assert result == str(output / "model.omc")


def test_external_weight_package_is_opt_in(tmp_path):
    """Verify that an external weight is packaged only when explicitly supplied."""
    files = {}
    for name in ("model.omc", "vocab.bin", "embedding.bin", "cos.bin", "sin.bin", "mask.bin"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        files[name] = str(path)
    external = tmp_path / "SubGraph_0.weight"
    external.write_bytes(b"external-weights")

    common = {
        "omc_path": files["model.omc"], "vocab_path": files["vocab.bin"],
        "embedding_path": files["embedding.bin"], "rope_cos": files["cos.bin"],
        "rope_sin": files["sin.bin"], "attention_mask": files["mask.bin"],
        "architecture": _architecture(),
        "npu_config": {"max_length": 128, "chunk_size": 32, "embedding_quant": False},
        "generation_policy": {}, "package_name": "test",
    }

    embedded = tmp_path / "embedded.msl"
    msl_pack.build_single_file_msl(output_path=str(embedded), **common)
    embedded_dir = tmp_path / "embedded"
    embedded_kv = msl_pack.unpack(str(embedded), str(embedded_dir))
    assert "npu.om_weight_dir" not in embedded_kv
    assert not (embedded_dir / "SubGraph_0.weight").exists()

    split = tmp_path / "split.msl"
    msl_pack.build_single_file_msl(
        output_path=str(split), external_weight_path=str(external), **common
    )
    split_dir = tmp_path / "split"
    split_kv = msl_pack.unpack(str(split), str(split_dir))
    assert split_kv["npu.om_weight_dir"] == "weights"
    assert (split_dir / "SubGraph_0.weight").read_bytes() == b"external-weights"
