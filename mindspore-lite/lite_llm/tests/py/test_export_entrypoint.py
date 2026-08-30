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
"""Tests for the one-click export entry point (``export/mslite_llm_export.py``).

The flat export scripts live in the ``export/`` tree; tests import them by file
path.  The heavy pipeline steps (skeleton export / omg / mspacker) require a DDK
environment and a real model, so only the pure interface logic is exercised here.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("gguf")

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"


@pytest.fixture(scope="module")
def export_module():
    spec = importlib.util.spec_from_file_location(
        "mslite_llm_export", str(_EXPORT_DIR / "mslite_llm_export.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_model_kind(export_module,  # pylint: disable=redefined-outer-name
    tmp_path):
    """Model-kind detection: HF dir, GGUF file and unknown are distinct."""
    hf_dir = tmp_path / "model_dir"
    hf_dir.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")
    txt = tmp_path / "model.txt"
    txt.write_text("")

    assert export_module.detect_model_kind(str(hf_dir)) == "hf"
    assert export_module.detect_model_kind(str(gguf)) == "gguf"

    with pytest.raises(ValueError):
        export_module.detect_model_kind(str(txt))
    with pytest.raises(ValueError):
        export_module.detect_model_kind(str(tmp_path / "missing"))


def test_detect_model_type_hf(export_module,  # pylint: disable=redefined-outer-name
    tmp_path):
    """An HF dir with config.json reports model type from its arch name."""
    hf_dir = tmp_path / "model_dir"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(
        '{"model_type": "qwen2", "num_hidden_layers": 24}'
    )
    assert export_module.detect_model_type(str(hf_dir)) == "qwen2_5"

    (hf_dir / "config.json").write_text(
        '{"model_type": "qwen2", "num_hidden_layers": 36}'
    )
    with pytest.raises(ValueError, match="unsupported model size"):
        export_module.detect_model_type(str(hf_dir))

    (hf_dir / "config.json").write_text(
        '{"model_type": "llama", "num_hidden_layers": 24}'
    )
    with pytest.raises(ValueError, match="unsupported model architecture"):
        export_module.detect_model_type(str(hf_dir))


def test_parser_interface(export_module,  # pylint: disable=redefined-outer-name
    tmp_path):
    """CLI parser defaults and custom overrides wire up correctly."""
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")

    args = export_module.build_parser().parse_args(
        ["--target", "kirin9020", "--model", str(gguf), "--output", "out.msl"]
    )
    assert args.target == "kirin9020"
    assert args.max_length == 1024
    assert args.chunk_size == 64

    # --target has a default; the one-click invocation needs only model+output.
    args = export_module.build_parser().parse_args(
        ["--model", str(gguf), "--output", "out.msl"]
    )
    assert args.target == "kirin9020"

    args = export_module.build_parser().parse_args(
        [
            "--target", "kirin9020",
            "--model", str(gguf),
            "--output", "out.msl",
            "--max-length", "2048",
            "--chunk-size", "256",
            "--verbose",
        ]
    )
    assert args.max_length == 2048
    assert args.chunk_size == 256
    assert args.verbose


def test_unsupported_target_rejected(export_module,  # pylint: disable=redefined-outer-name
    tmp_path):
    """Unknown --target values must be rejected by the parser."""
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")

    with pytest.raises(ValueError, match="not supported"):
        export_module.main(["--target", "foo", "--model", str(gguf), "--output", "out.msl"])
