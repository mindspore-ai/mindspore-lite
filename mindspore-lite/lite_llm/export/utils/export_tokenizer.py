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
"""Export a tokenizer to ``vocab.bin`` (MSLT binary) + policy JSON.

Produces the exact byte layout consumed by ``lite_llm/src/tokenizer/tokenizer.cpp``
(``BPECodec`` / ``SentencePieceCodec``) and the packager-consumable
``generation_policy.json``.  Qwen2.5 uses a ``vocab.json`` + ``merges.txt`` BPE
tokenizer; SentencePiece is kept as a fallback for LLaMA-style tokenizers.

Also hosts the chat-template -> restricted IR compiler: Jinja2 templates are
compiled to a small binary instruction stream at export time; the runtime
interprets the IR with no Jinja dependency.
"""

import json
import logging
import os
import struct
from pathlib import Path

from jinja2 import Environment, nodes

logger = logging.getLogger(__name__)

MSLT_MAGIC = 0x4D534C54
# v2: custom chat template payload switched from raw Jinja to restricted IR
# Old runtimes reject v2 packages at load (tokenizer.cpp kVersion).
MSLT_VERSION = 2

CODEC_BPE = 0
CODEC_SENTENCEPIECE = 1

CHAT_TEMPLATE_LEGACY = 0  # type field: byte-layout only; the custom template carries the payload

LEGACY_STOP_TOKENS = ("<|endoftext|>", "<|im_end|>", "</s>")
LEGACY_SUPPRESSED_TOKENS = ("<|im_start|>",)

# ─── chat template -> restricted IR (v1) ──────────────────────────────────
# IR byte schema (v1)::

#     header: u32 magic 0x4D534C49 ("MSLI") + u8 version 1
#     body:   opcodes, terminated by IR_END
#
#     opcode  payload             semantics
#     ------  -------             ---------
#     0x01    u32 len + bytes     EMIT_CONST: output the constant bytes
#     0x02    -                   EMIT_ROLE: output the current message role
#     0x03    -                   EMIT_CONTENT: output the current message content
#     0x04    -                   LOOP_MESSAGES_START: begin per-message loop
#     0x05    -                   LOOP_MESSAGES_END: repeat body for next message
#     0x06    -                   IF_ADD_GENERATION_PROMPT_START
#     0x07    -                   IF_END
#     0x08    -                   IR_END
#
# Runtime state: EMIT_ROLE / EMIT_CONTENT apply to the innermost loop frame's
# current message; the ``add_generation_prompt`` flag comes from the API. The
# ``{% if tools %}`` branch is folded to false at compile time (the v1 API has
# no tools input).

MAGIC = 0x4D534C49
VERSION = 1

EMIT_CONST = 0x01
EMIT_ROLE = 0x02
EMIT_CONTENT = 0x03
LOOP_MESSAGES_START = 0x04
LOOP_MESSAGES_END = 0x05
IF_ADD_GENERATION_PROMPT_START = 0x06
IF_END = 0x07
IR_END = 0x08

_ROLE_NAMES = {0: "system", 1: "user", 2: "assistant"}


class UnsupportedTemplateError(ValueError):
    """The template uses Jinja syntax outside the v1 IR subset."""


def compile_chat_template_ir(template):
    """Compile a Jinja2 chat template to v1 IR bytes. Raises on unsupported syntax."""
    ast = Environment().parse(template)
    out = bytearray()
    _compile_nodes(ast.body, out)
    out.append(IR_END)
    return struct.pack("<IB", MAGIC, VERSION) + bytes(out)


def render_ir(ir_bytes, messages, add_generation_prompt=False):
    """Reference interpreter for v1 IR bytes (used by tests / debugging).

    ``messages`` is a list of ``{"role": <int 0-2>, "content": str}`` dicts.
    """
    magic, version = struct.unpack_from("<IB", ir_bytes, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(f"bad IR header: magic={magic:#x} version={version}")

    pos = 5
    size = len(ir_bytes)
    out = []
    frames = []  # [{is_loop, body_start, msg_index}]

    while pos < size:
        op = ir_bytes[pos]
        pos += 1
        if op == EMIT_CONST:
            ln = struct.unpack_from("<I", ir_bytes, pos)[0]
            pos += 4
            out.append(ir_bytes[pos:pos + ln].decode("utf-8"))
            pos += ln
        elif op == EMIT_ROLE:
            out.append(_ROLE_NAMES[_current_message(frames, messages)["role"]])
        elif op == EMIT_CONTENT:
            out.append(_current_message(frames, messages)["content"])
        elif op == LOOP_MESSAGES_START:
            if not messages:
                pos = _skip_to_loop_end(ir_bytes, pos)
            else:
                frames.append({"is_loop": True, "body_start": pos, "msg_index": 0})
        elif op == LOOP_MESSAGES_END:
            frame = frames[-1]
            frame["msg_index"] += 1
            if frame["msg_index"] < len(messages):
                pos = frame["body_start"]
            else:
                frames.pop()
        elif op == IF_ADD_GENERATION_PROMPT_START:
            if not add_generation_prompt:
                pos = _skip_to_if_end(ir_bytes, pos)
        elif op == IF_END:
            pass
        elif op == IR_END:
            break
        else:
            raise ValueError(f"bad IR opcode {op:#x} at {pos - 1}")
    return "".join(out)


def _current_message(frames, messages):
    for frame in reversed(frames):
        if frame["is_loop"]:
            return messages[frame["msg_index"]]
    raise ValueError("EMIT_ROLE/EMIT_CONTENT outside a message loop")


def _skip_to_if_end(data, pos):
    """Advance past a false IF block, parsing nested IFs and CONST payloads."""
    depth = 1
    size = len(data)
    while pos < size:
        op = data[pos]
        pos += 1
        if op == EMIT_CONST:
            ln = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + ln
        elif op == IF_ADD_GENERATION_PROMPT_START:
            depth += 1
        elif op == IF_END:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError("unbalanced IF in IR")


def _skip_to_loop_end(data, pos):
    """Advance past a loop body (empty messages), parsing nested loops and CONST payloads."""
    depth = 1
    size = len(data)
    while pos < size:
        op = data[pos]
        pos += 1
        if op == EMIT_CONST:
            ln = struct.unpack_from("<I", data, pos)[0]
            pos += 4 + ln
        elif op == LOOP_MESSAGES_START:
            depth += 1
        elif op == LOOP_MESSAGES_END:
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError("unbalanced LOOP in IR")


def _compile_nodes(body, out):
    for node in body:
        _compile_node(node, out)


def _compile_node(node, out):
    """Emit IR for a single template AST node."""
    if isinstance(node, nodes.Output):
        for child in node.nodes:
            _compile_node(child, out)
    elif isinstance(node, nodes.Add):
        _compile_node(node.left, out)
        _compile_node(node.right, out)
    elif isinstance(node, nodes.Const):
        if not isinstance(node.value, str):
            raise UnsupportedTemplateError(
                f"non-string constant in chat template: {node.value!r}"
            )
        _emit_const(out, node.value)
    elif isinstance(node, nodes.Getitem):
        _compile_member(node, out)
    elif isinstance(node, nodes.For):
        _compile_for(node, out)
    elif isinstance(node, nodes.If):
        _compile_if(node, out)
    else:
        raise UnsupportedTemplateError(
            f"unsupported Jinja node in chat template: {type(node).__name__}"
        )


def _emit_const(out, value):
    data = value.encode("utf-8")
    out.append(EMIT_CONST)
    out += struct.pack("<I", len(data))
    out += data


def _compile_member(node, out):
    """Compile a member-expression node (only ``message`` is supported)."""
    if not (isinstance(node.node, nodes.Name) and node.node.name == "message"):
        raise UnsupportedTemplateError(
            "only message['role'] / message['content'] access is supported"
        )
    if isinstance(node.arg, nodes.Const):
        if node.arg.value == "role":
            out.append(EMIT_ROLE)
            return
        if node.arg.value == "content":
            out.append(EMIT_CONTENT)
            return
    raise UnsupportedTemplateError(f"unsupported message member: {node.arg!r}")


def _compile_for(node, out):
    if not (isinstance(node.iter, nodes.Name) and node.iter.name == "messages"):
        raise UnsupportedTemplateError("only {% for message in messages %} is supported")
    if node.else_ or node.recursive:
        raise UnsupportedTemplateError("loop else/recursive is unsupported")
    out.append(LOOP_MESSAGES_START)
    _compile_nodes(node.body, out)
    out.append(LOOP_MESSAGES_END)


def _compile_if(node, out):
    """Compile an if/else node; ``elif`` chains are unsupported."""
    if node.elif_:
        raise UnsupportedTemplateError("elif is unsupported")
    test = node.test
    if isinstance(test, nodes.Name):
        if test.name == "add_generation_prompt":
            if node.else_:
                raise UnsupportedTemplateError(
                    "add_generation_prompt if with else is unsupported"
                )
            out.append(IF_ADD_GENERATION_PROMPT_START)
            _compile_nodes(node.body, out)
            out.append(IF_END)
            return
        if test.name == "tools":
            # tools is not an API input in v1: the branch never runs; an else
            # branch, if present, always runs.
            if node.else_:
                _compile_nodes(node.else_, out)
            return
    raise UnsupportedTemplateError(f"unsupported if condition: {test!r}")


# ─── tokenizer export ──────────────────────────────────────────────────────

def _read_optional_json(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_token_ids(value, field):
    """Normalise a token-id value into a sorted de-duplicated list."""
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    token_ids = []
    for token_id in values:
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{field} must contain non-negative integer token IDs")
        if token_id not in token_ids:
            token_ids.append(token_id)
    return token_ids


def _policy_tokens_present(tokenizer, candidates):
    if not hasattr(tokenizer, "get_vocab"):
        return list(candidates)
    vocab = tokenizer.get_vocab()
    return [token for token in candidates if token in vocab]


def _write_custom_chat_template(file_obj, tokenizer):
    """Serialize the tokenizer's chat template into the .msl payload."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        file_obj.write(struct.pack("<I", 0))
        return
    try:
        ir = compile_chat_template_ir(template)
    except UnsupportedTemplateError as exc:
        raise ValueError(
            f"chat template cannot be compiled to restricted IR: {exc}"
        ) from exc
    file_obj.write(struct.pack("<I", len(ir)))
    file_obj.write(ir)


def _write_special_token_policy(file_obj, tokenizer):
    """Write the legacy stop/suppressed token policy flags."""
    stop_tokens = _policy_tokens_present(tokenizer, LEGACY_STOP_TOKENS)
    suppressed_tokens = _policy_tokens_present(tokenizer, LEGACY_SUPPRESSED_TOKENS)

    file_obj.write(struct.pack("<I", len(stop_tokens)))
    for token in stop_tokens:
        token_bytes = token.encode("utf-8")
        file_obj.write(struct.pack("<I", len(token_bytes)))
        file_obj.write(token_bytes)

    file_obj.write(struct.pack("<I", len(suppressed_tokens)))
    for token in suppressed_tokens:
        token_bytes = token.encode("utf-8")
        file_obj.write(struct.pack("<I", len(token_bytes)))
        file_obj.write(token_bytes)


def _write_header(file_obj, codec, vocab_size, tokenizer):
    """Write the tokenizer payload header (BOS/EOS/PAD ids)."""
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1

    file_obj.write(struct.pack("<I", MSLT_MAGIC))
    file_obj.write(struct.pack("<I", MSLT_VERSION))
    file_obj.write(struct.pack("<I", codec))
    file_obj.write(struct.pack("<I", vocab_size))
    file_obj.write(struct.pack("<i", bos_id))
    file_obj.write(struct.pack("<i", eos_id))
    file_obj.write(struct.pack("<i", pad_id))
    file_obj.write(struct.pack("<i", unk_id))
    # Legacy chat template type: always 0; the custom template written after
    # the vocab carries the actual pinned template.
    file_obj.write(struct.pack("<I", CHAT_TEMPLATE_LEGACY))


def _export_bpe(tokenizer, vocab_path):
    """Export vocab.json + merges.txt BPE tokenizer (Qwen2.5)."""
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    merges = list(getattr(tokenizer, "merges", None) or [])
    if not merges:
        merges_file = getattr(tokenizer, "merges_file", None)
        if merges_file:
            merges_path = Path(merges_file)
            if not merges_path.exists():
                merges_path = Path(tokenizer.vocab_file).parent / "merges.txt"
            if merges_path.exists():
                for line in merges_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    merges.append(stripped)

    with open(vocab_path, "wb") as f:
        _write_header(f, CODEC_BPE, len(sorted_vocab), tokenizer)

        for token_str, token_id in sorted_vocab:
            token_bytes = token_str.encode("utf-8")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)
            f.write(struct.pack("<I", token_id))

        f.write(struct.pack("<I", len(merges)))
        for merge in merges:
            merge_bytes = merge.encode("utf-8")
            f.write(struct.pack("<I", len(merge_bytes)))
            f.write(merge_bytes)

        _write_custom_chat_template(f, tokenizer)
        _write_special_token_policy(f, tokenizer)

    logger.info("BPE vocab exported to %s (%d tokens, %d merges)", vocab_path, len(sorted_vocab), len(merges))


def _export_sentencepiece(tokenizer, vocab_path):
    """Export SentencePiece tokenizer (LLaMA-style fallback)."""
    sp_model = None
    if hasattr(tokenizer, "sp_model") and tokenizer.sp_model is not None:
        sp_model = tokenizer.sp_model
    elif hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file is not None:
        import sentencepiece as spm

        sp_model = spm.SentencePieceProcessor(model_file=tokenizer.vocab_file)
    else:
        raise ValueError("Tokenizer does not have a SentencePiece model")

    vocab_size = sp_model.get_piece_size()

    with open(vocab_path, "wb") as f:
        _write_header(f, CODEC_SENTENCEPIECE, vocab_size, tokenizer)

        for idx in range(vocab_size):
            piece = sp_model.id_to_piece(idx)
            score = sp_model.get_score(idx)
            piece_bytes = piece.encode("utf-8")
            f.write(struct.pack("<I", len(piece_bytes)))
            f.write(piece_bytes)
            f.write(struct.pack("<I", idx))
            f.write(struct.pack("<f", score))

        sp_proto = sp_model.serialized_model_proto()
        f.write(struct.pack("<I", len(sp_proto)))
        f.write(sp_proto)
        _write_custom_chat_template(f, tokenizer)
        _write_special_token_policy(f, tokenizer)

    logger.info("SentencePiece vocab exported to %s (%d tokens)", vocab_path, vocab_size)


def export_generation_policy(tokenizer, model_dir, output_path):
    """Export ``generation_policy.json`` (stop/suppress token IDs)."""
    generation_config = _read_optional_json(Path(model_dir) / "generation_config.json")
    model_config = _read_optional_json(Path(model_dir) / "config.json")

    eos_token_ids = generation_config.get("eos_token_id")
    if eos_token_ids is None:
        eos_token_ids = model_config.get("eos_token_id")
    if eos_token_ids is None:
        eos_token_ids = getattr(tokenizer, "eos_token_id", None)

    stop_token_ids = _normalize_token_ids(eos_token_ids, "eos_token_id")
    suppress_token_ids = _normalize_token_ids(generation_config.get("suppress_tokens"), "suppress_tokens")

    conflict = sorted(set(stop_token_ids) & set(suppress_token_ids))
    if conflict:
        raise ValueError(f"Stop and suppress token IDs must not overlap: {conflict}")

    policy = {"stop_token_ids": stop_token_ids, "suppress_token_ids": suppress_token_ids}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    logger.info("Generation policy exported to %s", output_path)
    return output_path


class _GGUFTokenizerAdapter:
    """Minimal tokenizer facade over GGUF tokenizer metadata.

    Reuses transformers 4.57 ``load_gguf_checkpoint`` (the same parser that
    builds the GGUF model config), so the token order/ids match the dequantized
    model exactly.  Exposes only the attributes ``_export_bpe`` needs.
    """

    def __init__(self, gguf_path):
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

        info = load_gguf_checkpoint(gguf_path, return_tensors=False)
        tokenizer = info["tokenizer"]
        tokenizer_config = info.get("tokenizer_config", {}) or {}

        self.tokens = list(tokenizer.get("tokens", []))
        self.merges = list(tokenizer.get("merges", None) or [])
        self.bos_token_id = tokenizer.get("bos_token_id")
        self.eos_token_id = tokenizer.get("eos_token_id")
        self.pad_token_id = tokenizer.get("pad_token_id")
        self.unk_token_id = tokenizer.get("unk_token_id")
        self.chat_template = tokenizer_config.get("chat_template")
        self._vocab = {token: i for i, token in enumerate(self.tokens)}

    def get_vocab(self):
        return dict(self._vocab)


def export_tokenizer(model_dir, output_dir, chat_template=None):
    """Export ``vocab.bin`` + ``generation_policy.json`` for a Qwen2.5 model.

    ``model_dir`` may be a HF directory or a ``.gguf`` file (transformers 4.57
    rebuilds the tokenizer from GGUF metadata).  Returns the ``vocab.bin`` path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = str(model_dir)
    if os.path.isfile(model_dir) and model_dir.endswith(".gguf"):
        tokenizer = _GGUFTokenizerAdapter(model_dir)
        if chat_template is not None:
            tokenizer.chat_template = chat_template
        if not tokenizer.chat_template:
            raise ValueError(
                "GGUF has no chat template metadata; the chat template must be pinned "
                "at export time and the runtime rejects template-less packages"
            )
        vocab_path = output_dir / "vocab.bin"
        _export_bpe(tokenizer, vocab_path)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if chat_template is not None:
            tokenizer.chat_template = chat_template
        if not tokenizer.chat_template:
            raise ValueError(
                "HF model has no chat template in tokenizer_config.json; the chat "
                "template must be pinned at export time and the runtime "
                "rejects template-less packages"
            )

        vocab_path = output_dir / "vocab.bin"

        if hasattr(tokenizer, "vocab_file") and tokenizer.vocab_file is not None:
            vocab_file = Path(tokenizer.vocab_file)
            if vocab_file.suffix == ".json":
                _export_bpe(tokenizer, vocab_path)
            elif vocab_file.suffix == ".model":
                _export_sentencepiece(tokenizer, vocab_path)
            else:
                logger.warning("Unknown vocab file format %s; assuming BPE", vocab_file.suffix)
                _export_bpe(tokenizer, vocab_path)
        elif hasattr(tokenizer, "get_vocab"):
            _export_bpe(tokenizer, vocab_path)
        else:
            raise ValueError("Tokenizer has no recognizable vocab format")

    export_generation_policy(tokenizer, model_dir, str(output_dir / "generation_policy.json"))
    return str(vocab_path)
