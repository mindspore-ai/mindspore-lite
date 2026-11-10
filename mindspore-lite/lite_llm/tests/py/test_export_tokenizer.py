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
"""Golden tests for the chat template IR compiler.

Parity reference is plain Jinja2 — the same engine HF's ``apply_chat_template``
uses — so byte-for-byte equivalence with HF rendering follows from the same
template string. The C++ interpreter golden tests (tests/ut/) use the same IR
bytes embedded in a fixed program.
"""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mslite_llm_quant
import pytest

jinja2 = pytest.importorskip("jinja2")

# utils is registered by tests/py/conftest.py before collection; the import
# must stay below importorskip: utils/export_tokenizer.py imports jinja2 at
# module load, so a missing jinja2 should skip instead of erroring collection.
from utils.export_tokenizer import (  # pylint: disable=wrong-import-position
    _bpe_merges,
    MAGIC,
    VERSION,
    UnsupportedTemplateError,
    compile_chat_template_ir,
    render_ir,
)

QWEN_TEMPLATE = (
    "{%- if tools %}\n"
    "    {{- '<|im_start|>system\\n' }}\n"
    "{%- endif %}\n"
    "{%- for message in messages %}\n"
    "    {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|im_start|>assistant\\n' }}\n"
    "{%- endif %}\n"
)


def _jinja_render(messages, add_generation_prompt):
    env = jinja2.Environment()
    return env.from_string(QWEN_TEMPLATE).render(
        messages=messages, add_generation_prompt=add_generation_prompt
    )


_ROLE_ID = {"system": 0, "user": 1, "assistant": 2}


def _msgs(*tuples):
    """Return (HF string-role form, IR int-role form) for role/content tuples."""
    hf = [{"role": role, "content": content} for role, content in tuples]
    ir = [{"role": _ROLE_ID[role], "content": content} for role, content in tuples]
    return hf, ir


@pytest.mark.parametrize(
    "messages, add_generation_prompt",
    [
        (({"role": "user", "content": "hi"},), True),
        (({"role": "user", "content": "hi"},), False),
        (({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}), True),
        (({"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello!"}), False),
        (({"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}), True),
        (({"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}), False),
        (({"role": "system", "content": "sys"}, {"role": "user", "content": "u1"},
          {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}), True),
        ((), True),
        ((), False),
    ],
)
def test_ir_renders_identically_to_jinja2(messages, add_generation_prompt):
    """The IR renderer must match Jinja2 output for representative prompts."""
    ir = compile_chat_template_ir(QWEN_TEMPLATE)
    int_form = [
        {"role": _ROLE_ID[m["role"]], "content": m["content"]} for m in messages
    ]
    assert render_ir(ir, int_form, add_generation_prompt) == _jinja_render(
        list(messages), add_generation_prompt
    )


def test_ir_header():
    """Test that the IR blob starts with the magic and version header."""
    ir = compile_chat_template_ir(QWEN_TEMPLATE)
    assert len(ir) >= 5
    magic, version = ir[0:4], ir[4]
    assert int.from_bytes(magic, "little") == MAGIC
    assert version == VERSION


def test_tools_branch_folded_false():
    """Test that the {% if tools %} branch folds to false like Jinja2."""
    # {% if tools %} folds to false: a system message in the conversation still
    # renders (it goes through the loop), but no implicit system prefix is
    # injected — exactly what Jinja2 does with tools absent.
    ir = compile_chat_template_ir(QWEN_TEMPLATE)
    int_form = [{"role": 1, "content": "hi"}]
    rendered = render_ir(ir, int_form, False)
    assert rendered == _jinja_render([{"role": "user", "content": "hi"}], False)
    assert "<|im_start|>system\n" not in rendered


@pytest.mark.parametrize(
    "template",
    [
        "{% set x = 1 %}{{ x }}",                      # set
        "{{ message['content'] | trim }}",             # filter
        "{% if add_generation_prompt %}a{% else %}b{% endif %}",  # else on addgen
        "{% if chat %}a{% endif %}",                   # unknown condition
        "{% for item in other %}{{ item }}{% endfor %}",  # unknown iterable
        ("{% for message in messages %}{{ message['role'] }}{{ message['content'] }}{% endfor %}"
         "{% call x() %}y{% endcall %}"),                          # call
        "{{ 1 + 2 }}",                                 # non-string const
    ],
)
def test_unsupported_syntax_raises(template):
    """Unsupported template syntax must raise UnsupportedTemplateError."""
    with pytest.raises(UnsupportedTemplateError):
        compile_chat_template_ir(template)


def test_compiler_rejects_message_index_access():
    """Test that positional message access (message[0]) is rejected."""
    # message[0] (positional access) is not in the v1 subset.
    with pytest.raises(UnsupportedTemplateError):
        compile_chat_template_ir(
            "{% for message in messages %}{{ message[0] }}{% endfor %}"
        )


@pytest.mark.parametrize("merges", [["a b", "ab c"], [["a", "b"], ["ab", "c"]]])
def test_fast_bpe_merges_preserve_rank(merges):
    """Fast tokenizers store merge rules in backend JSON, in rank order."""
    backend = SimpleNamespace(to_str=lambda: json.dumps({"model": {"type": "BPE", "merges": merges}}))
    assert _bpe_merges(SimpleNamespace(backend_tokenizer=backend)) == ["a b", "ab c"]


def test_explicit_bpe_merges_take_precedence():
    """Keep the existing slow/GGUF merge source when present."""
    assert _bpe_merges(SimpleNamespace(merges=["b c", "a b"])) == ["b c", "a b"]


def test_bpe_merges_file(tmp_path):
    """Read the legacy merges.txt path and ignore its header."""
    path = tmp_path / "merges.txt"
    path.write_text("#version: 0.2\na b\n\nab c\n", encoding="utf-8")
    assert _bpe_merges(SimpleNamespace(merges_file=path)) == ["a b", "ab c"]


@pytest.mark.parametrize("merges", [["abc"], [["a", "b", "c"]], [["a", 1]], [["", "b"]]])
def test_invalid_bpe_merges_rejected(merges):
    """Malformed rules must fail during export instead of corrupting vocab.bin."""
    with pytest.raises(ValueError, match="BPE merge"):
        _bpe_merges(SimpleNamespace(merges=merges))


@pytest.fixture(name="hf_policy_tokenizer")
def hf_policy_tokenizer_fixture(tmp_path):
    """Tiny HF metadata using the existing Qwen chat-template fixture."""
    vocab = {"a": 0, "b": 1, "<eos>": 2, "<end>": 3, "<suppress>": 4}
    tokenizer = SimpleNamespace(
        get_vocab=lambda: vocab,
        get_added_vocab=lambda: {token: index for token, index in vocab.items() if index >= 2},
        all_special_tokens=["<eos>", "<end>", "<suppress>"],
        bos_token_id=None, eos_token_id=2, pad_token_id=2, chat_template=QWEN_TEMPLATE,
    )
    (tmp_path / "tokenizer.json").write_text(json.dumps({"model": {"merges": []}}), encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps({"eos_token_id": 1}), encoding="utf-8")
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 3, 2], "suppress_tokens": [4, 4]}), encoding="utf-8"
    )
    return tokenizer


def _write_policy_gguf(path, tokenizer):
    """Exercise the HF metadata writer with a real canonical type41 tensor."""
    gguf = pytest.importorskip("gguf")
    import numpy as np
    from mslite_llm_quant.gguf_quant import _add_tokenizer
    from mslite_llm_quant.q3_0 import register_private_q3_0_type

    q3_type = register_private_q3_0_type()
    writer = gguf.GGUFWriter(str(path), "qwen2")
    try:
        _add_tokenizer(writer, SimpleNamespace(vocab_size=5), tokenizer, path.parent)
        writer.add_tensor("blk.0.attn_q.weight", np.zeros((16, 14), dtype=np.uint8), raw_dtype=q3_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


@pytest.mark.parametrize("eos_ids,suppress_ids", [([2, 3, 2], [4, 4]), ([], [])])
def test_gguf_generation_policy_roundtrip(tmp_path, hf_policy_tokenizer, eos_ids, suppress_ids):
    """GGUF->MSL's exported policy preserves all EOS/suppress IDs, including empty lists."""
    from utils.export_tokenizer import export_generation_policy, export_tokenizer

    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": eos_ids, "suppress_tokens": suppress_ids}), encoding="utf-8"
    )
    hf_policy = export_generation_policy(hf_policy_tokenizer, tmp_path, tmp_path / "hf-policy.json")
    path = tmp_path / "policy.gguf"
    _write_policy_gguf(path, hf_policy_tokenizer)
    output_dir = tmp_path / "msl-tokenizer"
    vocab = export_tokenizer(path, output_dir)
    expected = {"stop_token_ids": list(dict.fromkeys(eos_ids)),
                "suppress_token_ids": list(dict.fromkeys(suppress_ids))}
    assert json.loads(hf_policy.read_text(encoding="utf-8")) == expected
    assert json.loads((output_dir / "generation_policy.json").read_text(encoding="utf-8")) == expected
    assert Path(vocab).is_file()


@pytest.mark.parametrize("generation_config,match", [
    ({"eos_token_id": [2, -1]}, "non-negative integer"),
    ({"eos_token_id": [2], "suppress_tokens": ["4"]}, "non-negative integer"),
    ({"eos_token_id": [2, 3], "suppress_tokens": [3]}, "must not overlap"),
])
def test_gguf_writer_validates_generation_policy(tmp_path, hf_policy_tokenizer, generation_config, match):
    """The GGUF writer uses the same normalization and conflict checks as direct HF export."""
    (tmp_path / "generation_config.json").write_text(json.dumps(generation_config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        _write_policy_gguf(tmp_path / "invalid.gguf", hf_policy_tokenizer)


def test_type41_tokenizer_export_in_fresh_process(tmp_path, hf_policy_tokenizer):
    """Standalone tokenizer export registers private Q3 before parsing a tensor-bearing file."""
    path = tmp_path / "fresh-process.gguf"
    _write_policy_gguf(path, hf_policy_tokenizer)
    output_dir = tmp_path / "standalone"
    # The source tree's quant/ directory installs as mslite_llm_quant; register it
    # by location in the child, as tests/py/conftest.py does for this process.
    package = mslite_llm_quant.__name__
    script = (
        "import importlib.util\n"
        "import sys\n"
        "import gguf\n"
        "assert 41 not in {int(value) for value in gguf.GGMLQuantizationType}\n"
        "spec = importlib.util.spec_from_file_location(sys.argv[3], sys.argv[4] + '/__init__.py',"
        " submodule_search_locations=[sys.argv[4]])\n"
        "sys.modules[sys.argv[3]] = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(sys.modules[sys.argv[3]])\n"
        "from utils.export_tokenizer import export_tokenizer\n"
        "export_tokenizer(sys.argv[1], sys.argv[2])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(output_dir), package, mslite_llm_quant.__path__[0]],
        cwd=Path(__file__).resolve().parents[2] / "export",
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "vocab.bin").is_file()
    assert json.loads((output_dir / "generation_policy.json").read_text(encoding="utf-8")) == {
        "stop_token_ids": [2, 3], "suppress_token_ids": [4],
    }
