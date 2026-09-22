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
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"


@pytest.fixture(scope="module", name="export_module")
def export_module_fixture():
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
    assert export_module.detect_model_type(str(hf_dir)) == ("qwen2_5", 24)

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

    (hf_dir / "config.json").write_text(
        '{"model_type": "qwen3", "num_hidden_layers": 8}'
    )
    assert export_module.detect_model_type(str(hf_dir)) == ("qwen3", 8)

    (hf_dir / "config.json").write_text(
        '{"model_type": "qwen3", "num_hidden_layers": 36}'
    )
    assert export_module.detect_model_type(str(hf_dir)) == ("qwen3", 36)

    (hf_dir / "config.json").write_text(
        '{"model_type": "qwen3", "num_hidden_layers": 12}'
    )
    with pytest.raises(ValueError, match="unsupported model size"):
        export_module.detect_model_type(str(hf_dir))


def test_qwen3_structural_validation(export_module):  # pylint: disable=redefined-outer-name
    """Both supported sizes use structural validation rather than MiniMind dimensions."""
    qwen3_cls = export_module.export_qwen3.__globals__["Qwen3Onnx"]

    def valid_config(layers):
        return SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=layers,
            hidden_act="silu",
            use_sliding_window=False,
            num_experts=0,
            tie_word_embeddings=True,
        )

    for layers in (8, 36):
        exporter = qwen3_cls()
        exporter.config = valid_config(layers)
        exporter._validate_config()  # pylint: disable=protected-access

    invalid = valid_config(36)
    invalid.num_experts = 8
    exporter.config = invalid
    with pytest.raises(ValueError, match="MoE"):
        exporter._validate_config()  # pylint: disable=protected-access

    invalid = valid_config(36)
    invalid.tie_word_embeddings = False
    exporter.config = invalid
    with pytest.raises(ValueError, match="tied"):
        exporter._validate_config()  # pylint: disable=protected-access


def test_qwen3_hf_load_disables_remote_code(export_module, monkeypatch, tmp_path):  # pylint: disable=redefined-outer-name
    """HF-directory loading uses the built-in transformers Qwen3 implementation."""
    qwen3_globals = export_module.export_qwen3.__globals__
    qwen3_cls = qwen3_globals["Qwen3Onnx"]
    calls = {}
    config = SimpleNamespace(
        model_type="qwen3",
        num_hidden_layers=8,
        hidden_act="silu",
        use_sliding_window=False,
        num_experts=0,
        tie_word_embeddings=True,
        hidden_size=2560,
        num_key_value_heads=8,
    )

    class FakeModel:
        def __init__(self, model_config):
            self.config = model_config

        def eval(self):
            return self

    def load_config(path, **kwargs):
        del path
        calls["config"] = kwargs
        return config

    def load_model(path, **kwargs):
        del path
        calls["model"] = kwargs
        return FakeModel(kwargs["config"])

    monkeypatch.setattr(qwen3_globals["AutoConfig"], "from_pretrained", load_config)
    monkeypatch.setattr(qwen3_globals["AutoModelForCausalLM"], "from_pretrained", load_model)
    monkeypatch.setattr(qwen3_globals["AutoTokenizer"], "from_pretrained", lambda path: object())

    qwen3_cls().load(str(tmp_path), layers=36)
    assert calls["config"]["trust_remote_code"] is False
    assert calls["model"]["trust_remote_code"] is False
    assert config.num_hidden_layers == 36


@pytest.mark.parametrize("case", [
    (family, layers, kind, quant_type)
    for family, layers, quant_types in [
        ("qwen3", 8, ("q4_0", "s16s4")),
        ("qwen3", 36, ("q4_0", "s16s4")),
        ("qwen2_5", 24, ("q4_0",)),
        ("minicpm", 40, ("q4_0",)),
    ]
    for kind in ("gguf", "hf")
    for quant_type in quant_types
])
def test_model_quantization_pipeline(export_module, monkeypatch, tmp_path, case):  # pylint: disable=redefined-outer-name
    """The selected mode controls export, weight injection and package metadata."""
    family, layers, kind, quant_type = case
    quant = export_module.QuantType(quant_type)
    model_info = export_module.MODEL_TYPES[family]
    calls = {}
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    def fake_exporter(**kwargs):
        calls["export_layers"] = kwargs["layers"]
        calls["decoder_quant"] = kwargs["decoder_quant"]
        calls["embedding_quant"] = kwargs["embedding_quant"]
        (work_dir / model_info["quant_name" if quant else "onnx_name"]).write_bytes(b"onnx")
        (work_dir / "architecture.json").write_text(json.dumps({"vocab_size": 128, "hidden_size": 128}))

    def fake_loader(**kwargs):
        calls["loader_layers"] = kwargs["layers"]

    def fake_compile(**kwargs):
        calls["onnx_path"] = kwargs["onnx_path"]
        calls["compile_quant"] = kwargs["embedding_quant"]
        calls["platform"] = kwargs["platform"]
        calls["save_external_weights"] = kwargs["save_external_weights"]
        output_dir = work_dir / "model"
        output_dir.mkdir()
        path = output_dir / "model.omc"
        path.write_bytes(b"omc")
        (output_dir / "SubGraph_0.weight").write_bytes(b"external-weights")
        return str(path)

    def fake_tokenizer(**_kwargs):
        tokenizer_dir = work_dir / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "generation_policy.json").write_text("{}")
        vocab = tokenizer_dir / "vocab.bin"
        vocab.write_bytes(b"vocab")
        return str(vocab)

    monkeypatch.setattr(export_module, "detect_model_kind", lambda _model: kind)
    monkeypatch.setattr(export_module, "detect_model_type", lambda _model: (family, layers))
    monkeypatch.setitem(model_info, "exporter", fake_exporter)
    monkeypatch.setitem(model_info, "gguf_loader", fake_loader)
    monkeypatch.setattr(export_module, "compile_omc", fake_compile)
    monkeypatch.setattr(export_module, "export_tokenizer", fake_tokenizer)
    def fake_pack(**kwargs):
        calls["npu_config"] = kwargs["npu_config"]
        calls["embedding_path"] = kwargs["embedding_path"]
        calls["external_weight_path"] = kwargs["external_weight_path"]
        return kwargs["output_path"]

    monkeypatch.setattr(export_module, "build_single_file_msl", fake_pack)

    args = SimpleNamespace(
        model="model.gguf",
        output=str(tmp_path / "qwen3.msl"),
        max_length=1024,
        chunk_size=64,
        target="kirin9030" if quant_type == "s16s4" else "kirin9020",
        quant_type=quant_type,
    )
    assert export_module.run_pipeline(args, str(work_dir)) == args.output
    assert calls["export_layers"] == layers
    assert calls["decoder_quant"] == quant
    assert calls["embedding_quant"] == quant
    inject_gguf = quant_type == "q4_0" and kind == "gguf"
    assert calls.get("loader_layers") == (layers if inject_gguf else None)
    assert calls["compile_quant"] == quant
    assert calls["onnx_path"] == str(work_dir / model_info["gguf_name" if inject_gguf else "quant_name"])
    assert calls["embedding_path"] == str(work_dir / ("embedding_weight.bin" if inject_gguf else "embedding_quant.bin"))
    assert calls["npu_config"]["embedding_format"] == ("S16S4_NZ_V1" if quant_type == "s16s4" else "W4A16")
    assert calls["npu_config"]["scale_gp_size"] == (128 if quant_type == "s16s4" else 32)
    assert calls["npu_config"].get("q4_0_weight_layout") == (
        "q4_0_nzf_compact_phase4" if quant_type == "q4_0" else None
    )
    assert calls["platform"] == args.target
    assert calls["save_external_weights"] == (family in ("qwen3", "qwen2_5"))



def test_external_weight_policy_preserves_qwen_families(export_module):  # pylint: disable=redefined-outer-name
    """Every supported Qwen size preserves external decoder weights."""
    qwen3 = export_module.MODEL_TYPES["qwen3"]
    qwen2_5 = export_module.MODEL_TYPES["qwen2_5"]

    assert qwen3["external_weights"] is True
    assert qwen2_5["external_weights"] is True


def test_qwen3_family_artifact_names(export_module):  # pylint: disable=redefined-outer-name
    """Generated Qwen3 artifacts use family names for every supported size."""
    qwen3 = export_module.MODEL_TYPES["qwen3"]
    assert qwen3["onnx_name"] == "qwen3.onnx"
    assert qwen3["quant_name"] == "qwen3_quant.onnx"
    assert qwen3["gguf_name"] == "qwen3_gguf.onnx"
    assert qwen3["model_name"] == "qwen3"


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
    assert args.quant_type == "q4_0"

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

    args = export_module.build_parser().parse_args(
        ["--target", "kirin9030", "--model", str(gguf), "--output", "out.msl"]
    )
    assert args.target == "kirin9030"


def test_unsupported_target_rejected(export_module,  # pylint: disable=redefined-outer-name
    tmp_path):
    """Unknown --target values must be rejected by the parser."""
    gguf = tmp_path / "model.gguf"
    gguf.write_text("")

    with pytest.raises(ValueError, match="not supported"):
        export_module.main(["--target", "foo", "--model", str(gguf), "--output", "out.msl"])


@pytest.mark.parametrize("layers", [8, 36])
def test_s16s4_requires_9030(export_module, monkeypatch, tmp_path, layers):  # pylint: disable=redefined-outer-name
    """Reject unsupported targets before loading an S16S4 model."""
    monkeypatch.setattr(export_module, "detect_model_kind", lambda _: "hf")
    monkeypatch.setattr(export_module, "detect_model_type", lambda _: ("qwen3", layers))
    args = SimpleNamespace(model="unused", target="kirin9020", quant_type="s16s4")
    with pytest.raises(ValueError, match="kirin9030"):
        export_module.run_pipeline(args, str(tmp_path))


def test_decoder_quant_option_removed(export_module):  # pylint: disable=redefined-outer-name
    """The old decoder-only switch is not a supported alias for quant_type."""
    with pytest.raises(SystemExit):
        export_module.build_parser().parse_args(
            ["--model", "model.gguf", "--output", "out.msl", "--decoder-quant", "S16S4"]
        )


@pytest.mark.parametrize("option", ["--quant-type", "--quant_type"])
@pytest.mark.parametrize("quant_type", ["q4_0", "s16s4"])
def test_quant_type_option(export_module, option, quant_type):
    """Both CLI spellings select one of the supported quantization modes."""
    args = export_module.build_parser().parse_args(
        ["--model", "model.gguf", "--output", "out.msl", option, quant_type]
    )
    assert args.quant_type == quant_type


def test_unknown_quant_type_rejected(export_module):
    """Reject unsupported modes at argument parsing time."""
    with pytest.raises(SystemExit):
        export_module.build_parser().parse_args(
            ["--model", "model.gguf", "--output", "out.msl", "--quant-type", "int8"]
        )


@pytest.mark.parametrize("family,layers", [("qwen2_5", 24), ("minicpm", 40)])
def test_unsupported_model_quant_type(export_module, monkeypatch, tmp_path, family, layers):
    """Reject S16S4 when the model exporter lacks the shared-weight contract."""
    monkeypatch.setattr(export_module, "detect_model_kind", lambda _: "hf")
    monkeypatch.setattr(export_module, "detect_model_type", lambda _: (family, layers))
    args = SimpleNamespace(model="unused", target="kirin9030", quant_type="s16s4")
    with pytest.raises(ValueError, match="not supported by"):
        export_module.run_pipeline(args, str(tmp_path))
