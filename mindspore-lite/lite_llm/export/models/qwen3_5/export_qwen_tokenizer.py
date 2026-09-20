#!/usr/bin/env python3
"""Export a Hugging Face fast tokenizer to the small tokenizer.txt format.

The exporter intentionally reads tokenizer.json directly.  This makes it usable
in a deployment image that does not contain PyTorch or transformers.  Qwen 3
(and Qwen-family fast tokenizers) use a byte-level BPE model, which is the
format implemented by the C++ runtime in this directory.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import struct
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

TEXT_MAGIC = "QWEN_BPE_TOKENIZER_V1"
BINARY_MAGIC = b"QWENBPE2"
BINARY_VERSION = 2

SEC_BYTE_TOKEN_IDS = 1
SEC_LETTER_RANGES = 2
SEC_MARK_RANGES = 3
SEC_NUMBER_RANGES = 4
SEC_WHITESPACE_RANGES = 5
SEC_NFD_INDEX = 6
SEC_NFD_VALUES = 7
SEC_CCC = 8
SEC_COMPOSITION = 9
SEC_MERGES = 10
SEC_ADDED_TOKEN_INDEX = 11
SEC_ADDED_TOKEN_BYTES = 12
SEC_TOKEN_INDEX = 13
SEC_TOKEN_BYTES = 14
SEC_TOKEN_OFFSETS = 15

TOKEN_SPECIAL = 1 << 0
TOKEN_ADDED = 1 << 1


def b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def find_byte_level(node: Any) -> Dict[str, Any] | None:
    """Find the ByteLevel configuration within a tokenizer pipeline."""
    if isinstance(node, dict):
        if node.get("type") == "ByteLevel":
            return node
        for value in node.values():
            result = find_byte_level(value)
            if result is not None:
                return result
    elif isinstance(node, list):
        for value in node:
            result = find_byte_level(value)
            if result is not None:
                return result
    return None


def find_split_regex(node: Any) -> str:
    """Find the tokenizer split expression within a pipeline."""
    if isinstance(node, dict):
        if node.get("type") == "Split":
            pattern = node.get("pattern")
            if isinstance(pattern, dict) and isinstance(pattern.get("Regex"), str):
                return pattern["Regex"]
        for value in node.values():
            result = find_split_regex(value)
            if result:
                return result
    elif isinstance(node, list):
        for value in node:
            result = find_split_regex(value)
            if result:
                return result
    return ""


def unicode_ranges(category_prefix: str) -> List[Tuple[int, int]]:
    """Collect contiguous Unicode ranges matching the requested categories."""
    ranges: List[Tuple[int, int]] = []
    begin = -1
    previous = -1
    for codepoint in range(0x110000):
        matches = unicodedata.category(chr(codepoint)).startswith(category_prefix)
        if matches and begin < 0:
            begin = previous = codepoint
        elif matches:
            previous = codepoint
        elif begin >= 0:
            ranges.append((begin, previous))
            begin = previous = -1
    if begin >= 0:
        ranges.append((begin, previous))
    return ranges


def whitespace_ranges() -> List[Tuple[int, int]]:
    points = [codepoint for codepoint in range(0x110000) if chr(codepoint).isspace()]
    ranges: List[Tuple[int, int]] = []
    for codepoint in points:
        if ranges and codepoint == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], codepoint)
        else:
            ranges.append((codepoint, codepoint))
    return ranges


def nfc_tables() -> Tuple[List[Tuple[int, List[int]]], List[Tuple[int, int]], List[Tuple[int, int, int]]]:
    """Build canonical decomposition, combining class, and composition tables."""
    decompositions: List[Tuple[int, List[int]]] = []
    combining_classes: List[Tuple[int, int]] = []
    compositions: List[Tuple[int, int, int]] = []
    for codepoint in range(0x110000):
        char = chr(codepoint)
        combining = unicodedata.combining(char)
        if combining:
            combining_classes.append((codepoint, combining))
        # Hangul decomposition/composition is handled algorithmically in C++.
        if not 0xAC00 <= codepoint <= 0xD7A3:
            normalized = unicodedata.normalize("NFD", char)
            values = [ord(value) for value in normalized]
            if values != [codepoint]:
                decompositions.append((codepoint, values))
        raw = unicodedata.decomposition(char)
        if raw and not raw.startswith("<"):
            values = [int(value, 16) for value in raw.split()]
            if len(values) == 2 and unicodedata.normalize("NFC", "".join(map(chr, values))) == char:
                compositions.append((values[0], values[1], codepoint))
    return decompositions, combining_classes, compositions


def get_tokenizer_json(model_dir: Path) -> Path:
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Download a fast tokenizer (tokenizer.json) "
            "from the Hugging Face model directory first."
        )
    return path


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def read_special_tokens(model_dir: Path, tokenizer: Dict[str, Any]) -> Dict[int, str]:
    """Return added special tokens, without requiring transformers."""
    result: Dict[int, str] = {}
    for item in tokenizer.get("added_tokens", []):
        if not isinstance(item, dict):
            continue
        token = item.get("content")
        token_id = item.get("id")
        if isinstance(token, str) and isinstance(token_id, int):
            result[token_id] = token

    # tokenizer_config.json is useful for tokenizers whose special-token
    # declarations are not marked in tokenizer.json.
    config_path = model_dir / "tokenizer_config.json"
    if config_path.is_file():
        try:
            config = load_json(config_path)
            added = config.get("added_tokens_decoder", {})
            if isinstance(added, dict):
                for raw_id, item in added.items():
                    if not isinstance(item, dict):
                        continue
                    token_id = int(raw_id)
                    token = item.get("content")
                    if isinstance(token, str):
                        result[token_id] = token
        except (ValueError, TypeError, json.JSONDecodeError):
            # tokenizer.json remains the source of truth if the optional config
            # is incomplete or uses a format unknown to this exporter.
            pass
    return result


def validate_added_tokens(model_dir: Path, tokenizer: Dict[str, Any]) -> None:
    """Validate added token IDs and their supported matching options."""
    items = list(tokenizer.get("added_tokens", []))
    config_path = model_dir / "tokenizer_config.json"
    if config_path.is_file():
        decoder = load_json(config_path).get("added_tokens_decoder", {})
        if isinstance(decoder, dict):
            items.extend(decoder.values())
    for item in items:
        if not isinstance(item, dict):
            continue
        if any(bool(item.get(name, False)) for name in
               ("single_word", "lstrip", "rstrip", "normalized")):
            raise ValueError(f"unsupported Qwen AddedToken flags: {item!r}")


def validate_qwen_regex(pattern: str) -> None:
    required = (
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)", r"\p{L}", r"\p{N}",
        r"\s+(?!\S)", r"[\r\n]",
    )
    if not pattern or any(fragment not in pattern for fragment in required):
        raise ValueError(f"unsupported Qwen Split regex: {pattern!r}")
    if r"|\p{N}|" not in pattern and r"|\p{N}{1,3}|" not in pattern:
        raise ValueError(f"unsupported Qwen numeric Split profile: {pattern!r}")


def extract_merges(model: Dict[str, Any], vocab: Dict[str, int]) -> List[Tuple[str, str, int]]:
    """Normalize tokenizer merge records while preserving rank order."""
    raw_merges = model.get("merges", [])
    result: List[Tuple[str, str, int]] = []
    for rank, item in enumerate(raw_merges):
        left: str
        right: str
        if isinstance(item, str):
            pieces = item.split(" ", 1)
            if len(pieces) != 2:
                raise ValueError(f"Invalid BPE merge at rank {rank}: {item!r}")
            left, right = pieces
        elif isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            left, right = item
        else:
            raise ValueError(f"Invalid BPE merge at rank {rank}: {item!r}")
        if left + right not in vocab:
            # A valid tokenizer may contain an intermediate merge which is not
            # explicitly present in some old tokenizer.json files.  The runtime
            # can still merge it, but emitting the warning makes broken exports
            # easy to diagnose.
            print(
                f"warning: merge {rank} produces {left + right!r}, which is not in vocab",
                file=sys.stderr,
            )
        result.append((left, right, rank))
    return result


def find_config_token_id(model_dir: Path, tokenizer: Dict[str, Any], vocab: Dict[str, int], name: str) -> int:
    """Resolve a configured special token to its vocabulary ID."""
    candidates: List[str] = []
    model = tokenizer.get("model", {})
    if isinstance(model, dict) and isinstance(model.get(name), str):
        candidates.append(model[name])
    config_path = model_dir / "tokenizer_config.json"
    if config_path.is_file():
        try:
            config = load_json(config_path)
            value = config.get(name)
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict) and isinstance(value.get("content"), str):
                candidates.append(value["content"])
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return int(next((vocab[token] for token in candidates if token in vocab), -1))


def find_unk_id(tokenizer: Dict[str, Any], vocab: Dict[str, int]) -> int:
    model = tokenizer.get("model", {})
    if isinstance(model, dict) and isinstance(model.get("unk_token"), str):
        return int(vocab.get(model["unk_token"], -1))
    for name in ("unk_token", "<|endoftext|>", "<unk>"):
        if name in vocab:
            return vocab[name]
    return -1


def load_export_data(model_dir: Path) -> Dict[str, Any]:
    """Load and validate the tokenizer tables needed by both output formats."""
    tokenizer_path = get_tokenizer_json(model_dir)
    tokenizer = load_json(tokenizer_path)
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        model_type = model.get("type") if isinstance(model, dict) else None
        raise ValueError(
            f"Only fast byte-level BPE tokenizers are supported, got model type {model_type!r}"
        )
    raw_vocab = model.get("vocab")
    if not isinstance(raw_vocab, dict):
        raise ValueError("tokenizer.json model.vocab is missing or is not an object")
    vocab: Dict[str, int] = {}
    for token, token_id in raw_vocab.items():
        if not isinstance(token, str) or not isinstance(token_id, int):
            raise ValueError("model.vocab must map strings to integer ids")
        vocab[token] = token_id

    byte_level = find_byte_level(tokenizer.get("pre_tokenizer")) or {}
    split_regex = find_split_regex(tokenizer.get("pre_tokenizer"))
    validate_qwen_regex(split_regex)
    validate_added_tokens(model_dir, tokenizer)
    decoder = tokenizer.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
        raise ValueError(f"unsupported Qwen decoder: {decoder!r}")
    normalizer = tokenizer.get("normalizer")
    normalizer_type = normalizer.get("type", "") if isinstance(normalizer, dict) else ""
    if normalizer_type not in ("", "NFC"):
        raise ValueError(f"Unsupported tokenizer normalizer: {normalizer_type!r}")
    special_tokens = read_special_tokens(model_dir, tokenizer)
    merges = extract_merges(model, vocab)
    all_ids = list(vocab.values()) + list(special_tokens)
    max_id = max(all_ids, default=-1)

    category_ranges = {name: unicode_ranges(name) for name in ("L", "M", "N")}
    category_ranges["S"] = whitespace_ranges()
    decompositions: List[Tuple[int, List[int]]] = []
    combining_classes: List[Tuple[int, int]] = []
    compositions: List[Tuple[int, int, int]] = []
    if normalizer_type == "NFC":
        decompositions, combining_classes, compositions = nfc_tables()

    all_token_ids = dict(vocab)
    all_token_ids.update({token: token_id for token_id, token in special_tokens.items()})

    return {
        "tokenizer_path": tokenizer_path,
        "tokenizer": tokenizer,
        "model": model,
        "vocab": vocab,
        "byte_level": byte_level,
        "split_regex": split_regex,
        "normalizer_type": normalizer_type,
        "special_tokens": special_tokens,
        "merges": merges,
        "vocab_size": max_id + 1,
        "unk_id": find_unk_id(tokenizer, vocab),
        "bos_id": find_config_token_id(model_dir, tokenizer, all_token_ids, "bos_token"),
        "eos_id": find_config_token_id(model_dir, tokenizer, all_token_ids, "eos_token"),
        "ranges": category_ranges,
        "decompositions": decompositions,
        "combining_classes": combining_classes,
        "compositions": compositions,
    }


def export_text(model_dir: Path, output: Path) -> None:
    """Write the legacy text tokenizer representation."""
    data = load_export_data(model_dir)
    metadata = {
        "format": TEXT_MAGIC,
        "model_type": "BPE",
        "vocab_size": str(data["vocab_size"]),
        "unk_id": str(data["unk_id"]),
        "bos_id": str(data["bos_id"]),
        "eos_id": str(data["eos_id"]),
        "byte_level": "1",
        "add_prefix_space": str(bool(data["byte_level"].get("add_prefix_space", False))).lower(),
        "trim_offsets": str(bool(data["byte_level"].get("trim_offsets", True))).lower(),
        "use_regex": str(bool(data["byte_level"].get("use_regex", True))).lower(),
        "split_regex": data["split_regex"],
        "normalizer": data["normalizer_type"],
        "unicode_version": unicodedata.unidata_version,
        "source": str(data["tokenizer_path"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        file.write(TEXT_MAGIC + "\n")
        for key, value in metadata.items():
            file.write(f"META\t{b64(key)}\t{b64(value)}\n")
        for name, ranges in data["ranges"].items():
            for begin, finish in ranges:
                file.write(f"RANGE\t{name}\t{begin}\t{finish}\n")
        for codepoint, values in data["decompositions"]:
            file.write("NFD\t" + str(codepoint) + "\t" + "\t".join(map(str, values)) + "\n")
        for codepoint, value in data["combining_classes"]:
            file.write(f"CCC\t{codepoint}\t{value}\n")
        for first, second, result in data["compositions"]:
            file.write(f"COMPOSE\t{first}\t{second}\t{result}\n")
        for token, token_id in sorted(data["vocab"].items(), key=lambda pair: pair[1]):
            file.write(f"TOKEN\t{token_id}\t{b64(token)}\n")
        for token_id, token in sorted(data["special_tokens"].items()):
            file.write(f"SPECIAL\t{token_id}\t{b64(token)}\n")
        for left, right, rank in data["merges"]:
            file.write(f"MERGE\t{b64(left)}\t{b64(right)}\t{rank}\n")
    print(f"exported text tokenizer to {output}")


def bytes_to_unicode() -> Dict[int, str]:
    """Map each byte to its reversible ByteLevel Unicode representation."""
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(0xA1, 0xAC + 1))
    values += list(range(0xAE, 0xFF + 1))
    chars = values[:]
    missing = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            chars.append(256 + missing)
            missing += 1
    return dict(zip(values, map(chr, chars)))


def align8(buffer: bytearray) -> None:
    buffer.extend(b"\0" * ((-len(buffer)) & 7))


def binary_merge_rows(data, vocab):
    """Convert string merge pairs to sorted, unique binary records."""
    binary_merges = []
    for left, right, rank in data["merges"]:
        if left not in vocab or right not in vocab:
            raise ValueError(f"merge rank {rank} references a token absent from vocab")
        result_token = left + right
        if result_token not in vocab:
            raise ValueError(f"merge rank {rank} result {result_token!r} is absent from vocab")
        left_id, right_id = vocab[left], vocab[right]
        key = (left_id << 32) | right_id
        binary_merges.append((key, vocab[result_token], rank))
    binary_merges.sort(key=lambda item: item[0])
    if any(binary_merges[i - 1][0] == binary_merges[i][0] for i in range(1, len(binary_merges))):
        raise ValueError("duplicate BPE pair in merges")

    return binary_merges


def added_token_metadata(model_dir, data):
    """Merge tokenizer and configuration metadata for added tokens."""
    tokenizer_added = {
        item["id"]: item for item in data["tokenizer"].get("added_tokens", [])
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    config_path = model_dir / "tokenizer_config.json"
    if config_path.is_file():
        config_added = load_json(config_path).get("added_tokens_decoder", {})
        if isinstance(config_added, dict):
            for raw_id, item in config_added.items():
                if isinstance(item, dict):
                    tokenizer_added[int(raw_id)] = item
    return tokenizer_added


def append_decode_sections(model_dir, data, vocab, sections):
    """Append compact token offsets and UTF-8 bytes for streaming decode."""
    id_to_token: List[str | None] = [None] * data["vocab_size"]
    token_flags = [0] * data["vocab_size"]
    for token, token_id in vocab.items():
        id_to_token[token_id] = token
    tokenizer_added = added_token_metadata(model_dir, data)
    for token_id, token in data["special_tokens"].items():
        id_to_token[token_id] = token
        item = tokenizer_added.get(token_id, {})
        token_flags[token_id] |= TOKEN_ADDED
        if item.get("special", False):
            token_flags[token_id] |= TOKEN_SPECIAL
    if any(token is None for token in id_to_token):
        missing = [i for i, token in enumerate(id_to_token) if token is None][:10]
        raise ValueError(f"non-contiguous Qwen vocabulary IDs: {missing}")
    all_token_bytes = bytearray()
    token_offsets: List[int] = []
    for token_id, token in enumerate(id_to_token):
        assert token is not None
        encoded = token.encode("utf-8")
        if len(all_token_bytes) >= (1 << 28):
            raise ValueError("Qwen token byte table exceeds compact offset limit")
        token_offsets.append(len(all_token_bytes) | (token_flags[token_id] << 28))
        all_token_bytes.extend(encoded)
    token_offsets.append(len(all_token_bytes))
    sections.append((SEC_TOKEN_OFFSETS, 4, struct.pack(
        f"<{len(token_offsets)}I", *token_offsets)))
    sections.append((SEC_TOKEN_BYTES, 1, bytes(all_token_bytes)))


def binary_byte_ids(vocab):
    """Resolve every ByteLevel base token before writing binary sections."""
    mapping = bytes_to_unicode()
    byte_ids = []
    for byte in range(256):
        token = mapping[byte]
        if token not in vocab:
            raise ValueError(f"ByteLevel base token {token!r} for byte {byte} is absent from vocab")
        byte_ids.append(vocab[token])

    return byte_ids


def export_binary(model_dir: Path, output: Path, full: bool = False) -> None:
    """Write the compact binary tokenizer with an optional decode table."""
    data = load_export_data(model_dir)
    vocab: Dict[str, int] = data["vocab"]
    byte_ids = binary_byte_ids(vocab)
    binary_merges = binary_merge_rows(data, vocab)
    nfd_values: List[int] = []
    nfd_index = []
    for codepoint, values in data["decompositions"]:
        nfd_index.append((codepoint, len(nfd_values), len(values)))
        nfd_values.extend(values)

    added_bytes = bytearray()
    added_index = []
    for token_id, token in sorted(data["special_tokens"].items()):
        encoded = token.encode("utf-8")
        added_index.append((token_id, len(added_bytes), len(encoded), 0))
        added_bytes.extend(encoded)

    sections: List[Tuple[int, int, bytes]] = []
    sections.append((SEC_BYTE_TOKEN_IDS, 4, struct.pack("<256i", *byte_ids)))
    for section_type, name in (
        (SEC_LETTER_RANGES, "L"), (SEC_MARK_RANGES, "M"),
        (SEC_NUMBER_RANGES, "N"), (SEC_WHITESPACE_RANGES, "S"),
    ):
        values = data["ranges"][name]
        sections.append((section_type, 8, b"".join(struct.pack("<II", *item) for item in values)))
    sections.append((SEC_NFD_INDEX, 12, b"".join(struct.pack("<III", *item) for item in nfd_index)))
    sections.append((SEC_NFD_VALUES, 4, struct.pack(f"<{len(nfd_values)}I", *nfd_values)))
    sections.append((SEC_CCC, 8, b"".join(struct.pack("<II", *item) for item in data["combining_classes"])))
    composition_rows = sorted(((first << 32) | second, result) for first, second, result in data["compositions"])
    sections.append((SEC_COMPOSITION, 12, b"".join(struct.pack("<QI", *item) for item in composition_rows)))
    sections.append((SEC_ADDED_TOKEN_INDEX, 16, b"".join(struct.pack("<iIII", *item) for item in added_index)))
    sections.append((SEC_ADDED_TOKEN_BYTES, 1, bytes(added_bytes)))
    if full:
        append_decode_sections(model_dir, data, vocab, sections)
    # Keep merges last so the C++ loader can release the raw sorted rows after
    # constructing its open-addressing hash.
    sections.append((SEC_MERGES, 16, b"".join(struct.pack("<QII", *item) for item in binary_merges)))

    section_count = len(sections)
    header_size = 48 + section_count * 16
    flags = (1 if data["byte_level"].get("add_prefix_space", False) else 0)
    flags |= 2 if data["normalizer_type"] == "NFC" else 0
    flags |= 4 if "\\p{M}" in data["split_regex"] else 0
    number_chunk_size = 3 if "\\p{N}{1,3}" in data["split_regex"] else 1
    output_bytes = bytearray(b"\0" * header_size)
    directory = []
    for section_type, stride, payload in sections:
        align8(output_bytes)
        offset = len(output_bytes)
        output_bytes.extend(payload)
        count = len(payload) // stride
        directory.append((section_type, offset, count, stride))

    struct.pack_into(
        "<8sIIIiiiiIII", output_bytes, 0, BINARY_MAGIC, BINARY_VERSION,
        header_size, flags, data["vocab_size"], data["unk_id"], data["bos_id"],
        data["eos_id"], number_chunk_size, section_count, 0,
    )
    for index, entry in enumerate(directory):
        struct.pack_into("<IIII", output_bytes, 48 + index * 16, *entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(output_bytes)
    print(
        f"exported binary tokenizer: vocab_size={data['vocab_size']}, "
        f"merges={len(binary_merges)}, added_tokens={len(added_index)}, "
        f"size={len(output_bytes)} bytes -> {output}"
    )


def export(model_dir: Path, output: Path, output_format: str = "auto") -> None:
    if output_format == "auto":
        output_format = "binary" if output.suffix.lower() in (".bin", ".qtok") else "text"
    if output_format == "binary":
        export_binary(model_dir, output)
    elif output_format == "text":
        export_text(model_dir, output)
    else:
        raise ValueError(f"unknown format: {output_format}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Hugging Face model directory containing tokenizer.json",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("tokenizer.bin"), help="output tokenizer file"
    )
    parser.add_argument("--format", choices=("auto", "binary", "text"), default="auto")
    parser.add_argument("--full", action="store_true", help="include id-to-token decode tables")
    args = parser.parse_args()
    try:
        if args.full and args.format in ("auto", "binary"):
            export_binary(args.model_dir.expanduser(), args.output.expanduser(), full=True)
        else:
            export(args.model_dir.expanduser(), args.output.expanduser(), args.format)
    except (OSError, ValueError, KeyError, TypeError) as error:  # friendly CLI error, with a non-zero status
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
