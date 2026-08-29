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

import sys
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

_EXPORT_DIR = Path(__file__).resolve().parents[2] / "export"
sys.path.insert(0, str(_EXPORT_DIR))

# pylint: disable=wrong-import-position  # export/ added to sys.path above
from utils.export_tokenizer import (  # noqa: E402
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
    ir = compile_chat_template_ir(QWEN_TEMPLATE)
    assert len(ir) >= 5
    magic, version = ir[0:4], ir[4]
    assert int.from_bytes(magic, "little") == MAGIC
    assert version == VERSION


def test_tools_branch_folded_false():
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
        "{% for message in messages %}{{ message['role'] }}{{ message['content'] }}{% endfor %}{% call x() %}y{% endcall %}",  # call
        "{{ 1 + 2 }}",                                 # non-string const
    ],
)
def test_unsupported_syntax_raises(template):
    """Unsupported template syntax must raise UnsupportedTemplateError."""
    with pytest.raises(UnsupportedTemplateError):
        compile_chat_template_ir(template)


def test_compiler_rejects_message_index_access():
    # message[0] (positional access) is not in the v1 subset.
    with pytest.raises(UnsupportedTemplateError):
        compile_chat_template_ir(
            "{% for message in messages %}{{ message[0] }}{% endfor %}"
        )
