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
"""Offline Hugging Face/C++ parity for Qwen byte-level BPE packages."""
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "export"))
from utils.export_tokenizer import _export_bpe, _GGUFTokenizerAdapter  # pylint: disable=wrong-import-position

TEXTS = [
    "Hello", "加拿大的首都是哪里？", "17加25等于多少？/no_think",
    "I'M testing.\n\n1234567890", "  hello\tworld\r\n\n", "end   ",
    "<think>\n\n</think>\n\n", "你好，世界！ Hello_world",
    "Русский Ελληνικά café e\u0301 😀\n", "\t\tword \u2003 word",
    "'RE 'Ve 'LL 'D 'S 'T 'M", "!?\r\n\nnext", "١٢٣ Ⅷ²",
]
PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@pytest.fixture(scope="module", name="cpp_probe")
def make_cpp_probe(tmp_path_factory):
    """Build the production tokenizer with a small command-line test driver."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    output = tmp_path_factory.mktemp("qwen-probe") / "probe"
    sources = [ROOT / "tests/data/tokenizer_probe.cc"] + [
        ROOT / "src/tokenizer" / name for name in
        ("tokenizer.cc", "bpe_codec.cc", "sentencepiece_codec.cc", "chat_template.cc")]
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(ROOT / "src"),
                    "-I" + str(ROOT / "include"), *map(str, sources), "-o", str(output)], check=True)
    return output


@pytest.fixture(scope="module", name="qwen_fixture")
def make_qwen_fixture(tmp_path_factory):
    """Train a tiny local BPE using HF's Qwen pre-tokenization rules."""
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        tokenizers.pre_tokenizers.Split(tokenizers.Regex(PATTERN), behavior="isolated"),
        tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    backend.decoder = tokenizers.decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=420, special_tokens=special,
                                           initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator(TEXTS, trainer)
    hf = transformers.Qwen2TokenizerFast(tokenizer_object=backend)
    path = tmp_path_factory.mktemp("qwen-vocab") / "vocab.bin"
    _export_bpe(hf, path)
    return hf, path


@pytest.mark.parametrize("text", TEXTS)
def test_qwen_ids_and_decoding(cpp_probe, qwen_fixture, tmp_path, text):
    """Match HF token IDs and lossless decoding across Unicode and boundaries."""
    hf, vocab = qwen_fixture
    input_path = tmp_path / "text.txt"
    input_path.write_text(text, encoding="utf-8")
    result = subprocess.run([str(cpp_probe), str(vocab), str(input_path)], check=True, capture_output=True)
    encoded, suppressed, decoded = result.stdout.decode("utf-8").split("\n", 2)
    assert list(map(int, encoded.split())) == hf.encode(text, add_special_tokens=False)
    assert decoded == text
    assert hf.convert_tokens_to_ids("<think>") not in list(map(int, suppressed.split()))
    assert hf.convert_tokens_to_ids("</think>") not in list(map(int, suppressed.split()))


def _rules_offset(data):
    """Skip the vocabulary and merges to locate codec-2 metadata."""
    offset = 36
    vocab_count = struct.unpack_from("<I", data, 12)[0]
    for _ in range(vocab_count):
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 8 + size
    merge_count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    for _ in range(merge_count):
        size = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + size
    return offset


@pytest.mark.parametrize("damage", ["truncated", "empty", "range", "added_id"])
def test_invalid_qwen_rules_rejected(cpp_probe, qwen_fixture, tmp_path, damage):
    """Reject malformed mandatory metadata after a valid vocabulary and merges."""
    _, vocab = qwen_fixture
    data = bytearray(vocab.read_bytes())
    offset = _rules_offset(data)
    if damage == "truncated":
        data = data[:offset + 2]
    elif damage == "empty":
        struct.pack_into("<I", data, offset, 0)
    elif damage == "range":
        struct.pack_into("<I", data, offset + 4, 0x110000)
    else:
        count = struct.unpack_from("<I", data, offset)[0]
        struct.pack_into("<I", data, offset + 8 + count * 12, 0xFFFFFFFF)
    broken = tmp_path / "broken.bin"
    broken.write_bytes(data)
    text = tmp_path / "text.txt"
    text.write_text("Hello", encoding="utf-8")
    result = subprocess.run([str(cpp_probe), str(broken), str(text)], check=False, capture_output=True)
    assert result.returncode == 3


def test_unsupported_qwen_pre_tokenizer_rejected(qwen_fixture, tmp_path):
    """Do not label modified Qwen tokenization as the supported codec."""
    hf, _ = qwen_fixture
    backend = tokenizers.Tokenizer.from_str(hf.backend_tokenizer.to_str())
    changed = transformers.Qwen2TokenizerFast(tokenizer_object=backend)
    changed.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
    with pytest.raises(ValueError, match="pre-tokenizer pattern"):
        _export_bpe(changed, tmp_path / "vocab.bin")


def test_original_qwen_nfc_configuration_exports(qwen_fixture, tmp_path):
    """Original Qwen checkpoints declare NFC even for already-normalized input."""
    hf, _ = qwen_fixture
    backend = tokenizers.Tokenizer.from_str(hf.backend_tokenizer.to_str())
    backend.normalizer = tokenizers.normalizers.NFC()
    original = transformers.Qwen2TokenizerFast(tokenizer_object=backend)
    _export_bpe(original, tmp_path / "vocab.bin")
    assert (tmp_path / "vocab.bin").stat().st_size > 36


def _write_gguf_tokenizer(path, hf, *, architecture="qwen3", pre_tokenizer="qwen2", omit_types=False):
    """Serialize a real, small GGUF tokenizer without model tensors."""
    gguf = pytest.importorskip("gguf")
    backend = json.loads(hf.backend_tokenizer.to_str())
    vocab = hf.get_vocab()
    tokens = sorted(vocab, key=vocab.get)
    added_ids = {token["id"] for token in backend["added_tokens"]}
    kinds = [
        gguf.TokenType.USER_DEFINED if token in ("<think>", "</think>") else
        gguf.TokenType.CONTROL if index in added_ids else gguf.TokenType.NORMAL
        for index, token in enumerate(tokens)
    ]
    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_tokenizer_model("gpt2")
    if pre_tokenizer is not None:
        writer.add_tokenizer_pre(pre_tokenizer)
    writer.add_token_list(tokens)
    if not omit_types:
        writer.add_token_types(kinds)
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in backend["model"]["merges"]]
    writer.add_token_merges(merges)
    writer.add_eos_token_id(hf.eos_token_id)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture(scope="module", name="gguf_fixture")
def make_gguf_fixture(qwen_fixture, tmp_path_factory):
    """Exercise the production adapter against actual GGUF metadata."""
    hf, _ = qwen_fixture
    root = tmp_path_factory.mktemp("qwen-gguf")
    _write_gguf_tokenizer(root / "tokenizer.gguf", hf)
    adapter = _GGUFTokenizerAdapter(root / "tokenizer.gguf")
    vocab = root / "vocab.bin"
    _export_bpe(adapter, vocab)
    assert adapter.eos_token_id == hf.eos_token_id
    assert adapter.get_vocab() == hf.get_vocab()
    return hf, vocab


@pytest.mark.parametrize("text", TEXTS)
def test_gguf_qwen_ids_and_decoding(cpp_probe, gguf_fixture, tmp_path, text):
    """GGUF control/user-defined tokens and Qwen splitting must match HF."""
    test_qwen_ids_and_decoding(cpp_probe, gguf_fixture, tmp_path, text)


@pytest.mark.parametrize("pre_tokenizer", [None, "llama-bpe"])
def test_unknown_qwen_gguf_rules_rejected(qwen_fixture, tmp_path, pre_tokenizer):
    """Do not silently fall back to legacy BPE for unknown Qwen metadata."""
    path = tmp_path / "unknown.gguf"
    _write_gguf_tokenizer(path, qwen_fixture[0], pre_tokenizer=pre_tokenizer)
    with pytest.raises(ValueError, match="pre-tokenizer"):
        _GGUFTokenizerAdapter(path)


def test_missing_qwen_gguf_token_types_rejected(qwen_fixture, tmp_path):
    """Missing added-token classifications must not produce a broken package."""
    path = tmp_path / "missing-types.gguf"
    _write_gguf_tokenizer(path, qwen_fixture[0], omit_types=True)
    with pytest.raises(ValueError, match="token types"):
        _GGUFTokenizerAdapter(path)


def test_non_qwen_gguf_retains_legacy_codec(qwen_fixture, tmp_path):
    """The Qwen metadata extension must not reclassify other BPE tokenizers."""
    path = tmp_path / "legacy.gguf"
    _write_gguf_tokenizer(path, qwen_fixture[0], architecture="llama", pre_tokenizer="llama-bpe")
    vocab = tmp_path / "vocab.bin"
    _export_bpe(_GGUFTokenizerAdapter(path), vocab)
    assert struct.unpack_from("<I", vocab.read_bytes(), 8)[0] == 0
