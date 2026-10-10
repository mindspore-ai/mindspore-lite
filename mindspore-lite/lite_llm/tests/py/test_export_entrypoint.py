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
import os
from pathlib import Path

import pytest


_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"


@pytest.fixture(scope="module")
def export_module():
    """Load the export entry point module from export/ by file path."""
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


def test_run_pipeline_step_order_and_paths_flow(export_module,  # pylint: disable=redefined-outer-name
    tmp_path, monkeypatch):
    """run_pipeline runs the four steps in order, threading one shared StepPaths."""
    work_dir = str(tmp_path)
    args = export_module.build_parser().parse_args(
        ["--model", str(tmp_path / "model"), "--output", str(tmp_path / "out.msl")]
    )
    monkeypatch.setattr(export_module, "detect_model_kind", lambda model: "hf")
    monkeypatch.setattr(export_module, "detect_model_type", lambda model: "qwen2_5")

    calls = []
    path_ids = []

    def fake_skeleton(args, mt, work_dir, model_kind, paths):
        assert args.model == str(tmp_path / "model")
        assert mt is export_module.MODEL_TYPES["qwen2_5"]
        assert work_dir == str(tmp_path)
        assert model_kind == "hf"
        calls.append(("skeleton", paths.onnx_path))
        path_ids.append(id(paths))
        paths.onnx_path = os.path.join(work_dir, "skeleton.onnx")
        paths.embedding_bin = os.path.join(work_dir, "embedding.bin")
        paths.embedding_quant = None

    def fake_compile(args, work_dir, use_external_weights, paths):  # pylint: disable=unused-argument
        assert use_external_weights is True  # qwen2_5 bundles external weights
        calls.append(("omc", paths.onnx_path))
        path_ids.append(id(paths))
        paths.omc_path = os.path.join(work_dir, "model.omc")
        paths.architecture = {"num_layers": 24}

    def fake_tokenizer(args, mt, work_dir, paths):  # pylint: disable=unused-argument
        calls.append(("tokenizer", paths.omc_path))
        path_ids.append(id(paths))
        paths.vocab_path = os.path.join(work_dir, "vocab.bin")
        paths.generation_policy = {"stop_token_ids": [1]}

    def fake_package(args, work_dir, use_external_weights, paths):  # pylint: disable=unused-argument
        assert use_external_weights is True
        assert paths.architecture == {"num_layers": 24}
        assert paths.generation_policy == {"stop_token_ids": [1]}
        calls.append(("package", paths.vocab_path))
        path_ids.append(id(paths))
        return args.output

    monkeypatch.setattr(export_module, "_export_skeleton", fake_skeleton)
    monkeypatch.setattr(export_module, "_compile_omc", fake_compile)
    monkeypatch.setattr(export_module, "_export_tokenizer", fake_tokenizer)
    monkeypatch.setattr(export_module, "_package_msl", fake_package)

    result = export_module.run_pipeline(args, work_dir)

    assert result == str(tmp_path / "out.msl")
    # Steps run in pipeline order, each observing the previous step's artifact.
    assert calls == [
        ("skeleton", ""),
        ("omc", os.path.join(work_dir, "skeleton.onnx")),
        ("tokenizer", os.path.join(work_dir, "model.omc")),
        ("package", os.path.join(work_dir, "vocab.bin")),
    ]
    # One shared StepPaths instance flows through all four steps.
    assert len(set(path_ids)) == 1


def test_run_pipeline_model_kind_drives_skeleton_and_weight_mode(
    export_module,  # pylint: disable=redefined-outer-name
    tmp_path, monkeypatch):
    """gguf/hf detection reaches the skeleton step; external weights follow the model type."""
    args = export_module.build_parser().parse_args(
        ["--model", str(tmp_path / "model"), "--output", str(tmp_path / "out.msl")]
    )
    skeleton_seen = []
    weight_modes = []

    monkeypatch.setattr(
        export_module, "_export_skeleton",
        lambda args, mt, work_dir, model_kind, paths: skeleton_seen.append(
            (model_kind, mt["model_name"])))
    monkeypatch.setattr(
        export_module, "_compile_omc",
        lambda args, work_dir, use_external_weights, paths: weight_modes.append(
            ("omc", use_external_weights)))
    monkeypatch.setattr(
        export_module, "_export_tokenizer",
        lambda args, mt, work_dir, paths: None)
    monkeypatch.setattr(
        export_module, "_package_msl",
        lambda args, work_dir, use_external_weights, paths: (
            weight_modes.append(("package", use_external_weights)), args.output)[1])

    monkeypatch.setattr(export_module, "detect_model_kind", lambda model: "gguf")
    monkeypatch.setattr(export_module, "detect_model_type", lambda model: "qwen2_5")
    export_module.run_pipeline(args, str(tmp_path))

    monkeypatch.setattr(export_module, "detect_model_kind", lambda model: "hf")
    monkeypatch.setattr(export_module, "detect_model_type", lambda model: "qwen3")
    export_module.run_pipeline(args, str(tmp_path))

    assert skeleton_seen == [("gguf", "qwen2.5-0.5b"), ("hf", "minimind-3-qwen3")]
    # qwen2_5 exports external decoder weights; qwen3 does not.
    assert weight_modes == [("omc", True), ("package", True), ("omc", False), ("package", False)]
