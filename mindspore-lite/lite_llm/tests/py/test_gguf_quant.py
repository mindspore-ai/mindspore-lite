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
"""Independent HF-to-GGUF CLI, pure quantization, and consumer contracts."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import pytest
from safetensors.numpy import save_file

from utils import gguf_mapping
from utils.quantization import QuantType
from mslite_llm_quant import gguf_quant


@pytest.fixture(name="tiny_hf")
def tiny_hf_fixture(tmp_path):
    """A complete miniature tied Qwen2 layer with local tokenizer/checkpoint."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from transformers import PreTrainedTokenizerFast, Qwen2Config

    config = Qwen2Config(
        vocab_size=16, hidden_size=64, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=4, tie_word_embeddings=True,
    )
    config.save_pretrained(tmp_path)
    backend = Tokenizer(BPE(vocab={"<unk>": 0, "a": 1, "b": 2, "<eos>": 3},
                            merges=[], unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>",
                                        unk_token="<unk>", pad_token="<eos>")
    tokenizer.save_pretrained(tmp_path)
    rng = np.random.default_rng(41)
    tensors = {}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensors[f"model.layers.0.self_attn.{proj}.weight"] = rng.normal(size=(64, 64)).astype(np.float32)
    for proj in ("gate_proj", "up_proj", "down_proj"):
        tensors[f"model.layers.0.mlp.{proj}.weight"] = rng.normal(size=(64, 64)).astype(np.float32)
    for proj in ("q_proj", "k_proj", "v_proj"):
        tensors[f"model.layers.0.self_attn.{proj}.bias"] = np.zeros(64, dtype=np.float32)
    for norm in ("input_layernorm", "post_attention_layernorm"):
        tensors[f"model.layers.0.{norm}.weight"] = np.ones(64, dtype=np.float32)
    tensors["model.norm.weight"] = np.ones(64, dtype=np.float32)
    tensors["model.embed_tokens.weight"] = rng.normal(size=(16, 64)).astype(np.float32)
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].copy()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path, config, tensors


@pytest.mark.parametrize("quant_type,wire_type,block_bytes", [("q4_0", 2, 18), ("q3_0", 41, 14)])
def test_full_hf_to_pure_gguf_and_loader(tiny_hf, monkeypatch, quant_type, wire_type, block_bytes):
    """Real HF files -> CLI -> real GGUF -> type detection and runtime packing."""
    model, config, tensors = tiny_hf
    # Keep the production Q3 architecture guard; scale its supported fixture
    # down to a complete tiny layer so this test does not allocate 0.5B weights.
    monkeypatch.setattr(gguf_quant, "SUPPORTED_MODEL", {
        name: getattr(config, name) for name in gguf_quant.SUPPORTED_MODEL
    })
    output = model / f"pure-{quant_type}.gguf"
    assert gguf_quant.main(["--model", str(model), "--output", str(output),
                           "--quant-type", quant_type]) == 0
    reader = gguf_mapping.open_gguf_reader(output)
    quant = QuantType(quant_type)
    assert gguf_mapping.detect_gguf_quantization(reader) == (quant, QuantType.Q4_0)
    assert {int(t.tensor_type) for t in reader.tensors} == {0, 2, wire_type}
    assert not any(int(t.tensor_type) in (3, 14) for t in reader.tensors)  # Q4_1 / Q6_K
    decoder = [t for t in reader.tensors if t.name.startswith("blk.") and t.data.ndim == 2]
    assert len(decoder) == 7
    for tensor in decoder:
        assert int(tensor.tensor_type) == wire_type
        assert tuple(tensor.shape) == (64, 64)
        assert tensor.data.shape == (64, 2 * block_bytes)
    assert not any(t.name == "output.weight" for t in reader.tensors)
    for tensor in reader.tensors:
        if tensor.data.ndim == 1:
            assert int(tensor.tensor_type) == 0  # F32 norms and biases
    embedding = model / "embedding.bin"
    packed = gguf_mapping.load_file_from_tensors(reader.tensors, embedding, quant, QuantType.Q4_0)
    assert embedding.is_file() and embedding.stat().st_size > 0
    assert len(packed) == len(reader.tensors)
    assert packed["blk.0.attn_q.weight"].size == 64 * 64 // 32 * block_bytes
    if quant_type == "q4_0":
        from gguf.quants import quantize
        q = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
        np.testing.assert_array_equal(q.data, quantize(
            tensors["model.layers.0.self_attn.q_proj.weight"], gguf.GGMLQuantizationType.Q4_0))


def test_q4_untied_qwen3_keeps_pure_output(tiny_hf):
    """Qwen3's explicit output and per-head norms must not trigger mixed types."""
    model, _, tensors = tiny_hf
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    config.update(model_type="qwen3", tie_word_embeddings=False, head_dim=16)
    config_path.write_text(json.dumps(config))
    tensors["model.layers.0.self_attn.q_norm.weight"] = np.ones(16, dtype=np.float32)
    tensors["model.layers.0.self_attn.k_norm.weight"] = np.ones(16, dtype=np.float32)
    save_file(tensors, str(model / "model.safetensors"))
    output = model / "qwen3-pure.gguf"
    gguf_quant.export_gguf(model, output, "q4_0")
    reader = gguf_mapping.open_gguf_reader(output)
    by_name = {t.name: t for t in reader.tensors}
    assert int(by_name["output.weight"].tensor_type) == 2
    assert int(by_name["blk.0.attn_q_norm.weight"].tensor_type) == 0
    assert int(by_name["blk.0.attn_k_norm.weight"].tensor_type) == 0
    assert {int(t.tensor_type) for t in reader.tensors} == {0, 2}
    assert gguf_mapping.detect_gguf_quantization(reader) == (QuantType.Q4_0, QuantType.Q4_0)


@pytest.mark.parametrize("quant_type", ["q4_0", "q3_0"])
def test_standalone_cli_dispatch(monkeypatch, quant_type):
    """The independent command calls only its writer, without export/DDK work."""
    calls = []
    monkeypatch.setattr(gguf_quant, "export_gguf", lambda *args: calls.append(args))
    assert gguf_quant.main(["--model", "local", "--output", "pure.gguf",
                           "--quant-type", quant_type]) == 0
    assert calls == [("local", "pure.gguf", quant_type)]


@pytest.mark.parametrize("value", ["msl_q3_0", "q4_1", "q6_k"])
def test_standalone_cli_rejects_unsupported_types(value):
    """Only the supported pure quantization modes are exposed."""
    with pytest.raises(SystemExit):
        gguf_quant.build_parser().parse_args(["--model", "local", "--output", "out",
                                             "--quant-type", value])


def test_standalone_default_is_q4():
    """The standalone command defaults to standard Q4_0."""
    args = gguf_quant.build_parser().parse_args(["--model", "local", "--output", "out"])
    assert args.quant_type == "q4_0"


def test_q4_unsupported_family_rejected():
    """Unsupported HF architecture names fail before conversion."""
    with pytest.raises(ValueError, match="dense Qwen2/Qwen3"):
        gguf_quant.validate_config(SimpleNamespace(model_type="llama"), "q4_0")


@pytest.mark.parametrize("invalid", ["shape", "nan"])
def test_q4_invalid_weights_fail_explicitly(tiny_hf, invalid):
    """Public HF conversion rejects malformed or non-finite checkpoint weights."""
    model, _, _ = tiny_hf
    data = np.zeros((16, 33 if invalid == "shape" else 64), dtype=np.float32)
    if invalid == "nan":
        data[0, 0] = np.nan
    save_file({"model.layers.0.self_attn.q_proj.weight": data}, str(model / "model.safetensors"))
    with pytest.raises(ValueError, match="divisible by 32|non-finite"):
        gguf_quant.export_gguf(model, model / "bad.gguf", "q4_0")


LITE_LLM_DIR = Path(__file__).resolve().parents[2]


def _export_top_level_names():
    """Top-level import names that the export tree installs (packages and modules)."""
    export_dir = LITE_LLM_DIR / "export"
    names = {"export"}
    names.update(path.parent.name for path in export_dir.glob("*/__init__.py"))
    names.update(path.stem for path in export_dir.glob("*.py"))
    return names


def _imported_roots(tree):
    """Yield (line, top-level name) for every absolute import in a module AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module.split(".")[0]


def test_quant_package_never_imports_export():
    """quant is the lower layer: export may import it, never the reverse."""
    own = gguf_quant.__name__.split(".")[0]
    forbidden = _export_top_level_names() - {own}
    assert {"utils", "models", "mslite_llm_export"} <= forbidden
    # quant/tests/ is not packaged; cross-tool contract tests there may use export.
    quant_dir = LITE_LLM_DIR / "quant"
    sources = sorted(path for path in quant_dir.rglob("*.py")
                     if "tests" not in path.relative_to(quant_dir).parts)
    assert sources
    violations = [
        f"{path.relative_to(LITE_LLM_DIR)}:{line} imports {root}"
        for path in sources
        for line, root in _imported_roots(ast.parse(path.read_text(encoding="utf-8")))
        if root in forbidden
    ]
    assert not violations, violations
