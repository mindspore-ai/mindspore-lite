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
"""msl_pack — pack/unpack single-file ``.msl`` (MindSpore Lite model) format v1.

Single-file, self-contained model container: header + KV metadata +
resource table + data region.  There is no external ``manifest.json``:
metadata travels inside the file as key-value entries.

Byte layout (little-endian), mirrors the C++ schema consumed by the
runtime (``src/manifest/msl_format.h``):

    MslHeader (24 bytes)
        magic[4]          ".MSL"
        version     u32   1
        kv_count    u32
        resource_count u32
        alignment   u32   payload offset alignment (default 4096)
        reserved    u32

    KV region: kv_count entries, each:
        key_len    u32
        key        UTF-8 bytes
        type       u32
        value_len  u32
        value      type-encoded bytes

    Resource table: resource_count entries, each 88 bytes:
        name[64]   fixed, NUL padded
        offset     u64   absolute file offset of the payload
        size       u64
        access     u32   0 = mmap, 1 = read
        reserved   u32

    Data region: payloads, each starting at an offset aligned to
    ``alignment`` and laid out consecutively by ``size``.

KV value types (v1 closed set; unknown types are rejected on unpack):
    bool / uint32 / uint64 / float32 / string / string[]

Extensibility: adding a *key* does not bump the version (readers skip
unknown keys); adding a *value type* or changing the layout requires a
new ``version`` (readers reject unknown versions).

Also maps the export artifacts onto the v1 KV schema (see
``build_manifest`` / ``manifest_to_kv``): the .omc graph is mmap'd at
runtime (access 0); tokenizer/rope/mask assets are read (access 1); the
quantized embedding is mmap'd too.
"""

import argparse
import json
import logging
import os
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

MAGIC = b".MSL"
VERSION = 1
HEADER_SIZE = 24
ENTRY_SIZE = 88
NAME_MAX = 64
DEFAULT_ALIGNMENT = 4096

# ─── KV value types (v1 closed set) ────────────────────────────────────────
TYPE_BOOL = 0
TYPE_UINT32 = 1
TYPE_UINT64 = 2
TYPE_FLOAT32 = 3
TYPE_STRING = 4
TYPE_STRING_ARRAY = 5

# Access modes (mspacker semantics).
ACCESS_MMAP = 0
ACCESS_READ = 1


class MslPackError(Exception):
    """Raised for any malformed input or format violation."""


# ─── value encoding ────────────────────────────────────────────────────────

def encode_value(value_type: int, value: Any) -> bytes:
    """Encode a Python value into its v1 wire format."""
    if value_type == TYPE_BOOL:
        if not isinstance(value, bool):
            raise MslPackError(f"expected bool, got {type(value).__name__}")
        return b"\x01" if value else b"\x00"
    if value_type == TYPE_UINT32:
        if not isinstance(value, int) or isinstance(value, bool):
            raise MslPackError(f"expected uint32, got {type(value).__name__}")
        if not 0 <= value <= 0xFFFFFFFF:
            raise MslPackError(f"uint32 out of range: {value}")
        return struct.pack("<I", value)
    if value_type == TYPE_UINT64:
        if not isinstance(value, int) or isinstance(value, bool):
            raise MslPackError(f"expected uint64, got {type(value).__name__}")
        if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
            raise MslPackError(f"uint64 out of range: {value}")
        return struct.pack("<Q", value)
    if value_type == TYPE_FLOAT32:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise MslPackError(f"expected float32, got {type(value).__name__}")
        return struct.pack("<f", float(value))
    if value_type == TYPE_STRING:
        if not isinstance(value, str):
            raise MslPackError(f"expected string, got {type(value).__name__}")
        return value.encode("utf-8")
    if value_type == TYPE_STRING_ARRAY:
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise MslPackError("expected string[] (list of str)")
        payload = bytearray()
        payload += struct.pack("<I", len(value))
        for item in value:
            encoded = item.encode("utf-8")
            payload += struct.pack("<I", len(encoded))
            payload += encoded
        return bytes(payload)
    raise MslPackError(f"unknown KV value type: {value_type}")


def decode_value(value_type: int, raw: bytes) -> Any:
    """Decode a v1 wire-format value back into a Python object."""
    if value_type == TYPE_BOOL:
        if len(raw) != 1:
            raise MslPackError("bool value must be 1 byte")
        return raw != b"\x00"
    if value_type == TYPE_UINT32:
        if len(raw) != 4:
            raise MslPackError("uint32 value must be 4 bytes")
        return struct.unpack("<I", raw)[0]
    if value_type == TYPE_UINT64:
        if len(raw) != 8:
            raise MslPackError("uint64 value must be 8 bytes")
        return struct.unpack("<Q", raw)[0]
    if value_type == TYPE_FLOAT32:
        if len(raw) != 4:
            raise MslPackError("float32 value must be 4 bytes")
        return struct.unpack("<f", raw)[0]
    if value_type == TYPE_STRING:
        return raw.decode("utf-8")
    if value_type == TYPE_STRING_ARRAY:
        count = struct.unpack_from("<I", raw, 0)[0]
        pos = 4
        items = []
        for _ in range(count):
            if pos + 4 > len(raw):
                raise MslPackError("string[] truncated")
            item_len = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            if pos + item_len > len(raw):
                raise MslPackError("string[] element truncated")
            items.append(raw[pos:pos + item_len].decode("utf-8"))
            pos += item_len
        return items
    raise MslPackError(f"unknown KV value type: {value_type}")


def infer_type(value: Any) -> int:
    """Map a Python value to its v1 KV type (used by the convenience API)."""
    if isinstance(value, bool):
        return TYPE_BOOL
    if isinstance(value, int):
        return TYPE_UINT64 if value > 0xFFFFFFFF else TYPE_UINT32
    if isinstance(value, float):
        return TYPE_FLOAT32
    if isinstance(value, str):
        return TYPE_STRING
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return TYPE_STRING_ARRAY
    raise MslPackError(f"cannot infer KV type for {type(value).__name__}")


# ─── resource name validation (v1: full relative path incl. subdirs) ───────

def validate_resource_name(name: str) -> None:
    """Entry names are full relative paths (e.g. ``npu_offline/x.omc``).

    The old C++ mspacker stored only basenames while the runtime looked up
    full paths — a format inconsistency fixed in v1: both sides use the
    full relative path.  Reject path traversal (``..``), backslashes and
    control characters.
    """
    encoded = name.encode("utf-8")
    if not name or len(encoded) >= NAME_MAX:
        raise MslPackError(f"resource name must be 1..{NAME_MAX - 1} bytes: {name!r}")
    for ch in name:
        code = ord(ch)
        if ch == "\\" or code < 0x20 or code == 0x7F:
            raise MslPackError(f"resource name contains forbidden character {code:#x}: {name!r}")
    for segment in name.split("/"):
        if segment in ("", ".", ".."):
            raise MslPackError(f"resource name has invalid path segment: {name!r}")


# ─── pack ──────────────────────────────────────────────────────────────────

def pack(output_path: str, kv: Dict[str, Any], resources: Iterable[Tuple[str, str, int]],
         alignment: int = DEFAULT_ALIGNMENT) -> str:
    """Write a v1 ``.msl`` file.

    Args:
        output_path: destination file path.
        kv: metadata key -> Python value (type inferred per entry).
        resources: iterable of ``(name, file_path, access)`` triples.
        alignment: payload offset alignment (must be > 0).
    """
    if alignment <= 0:
        raise MslPackError(f"alignment must be > 0, got {alignment}")
    entries = list(resources)
    for name, _, access in entries:
        validate_resource_name(name)
        if access not in (ACCESS_MMAP, ACCESS_READ):
            raise MslPackError(f"invalid access mode {access} for {name!r}")

    # Encode the KV region.
    kv_entries = []
    for key, value in kv.items():
        key_bytes = key.encode("utf-8")
        value_type = infer_type(value)
        value_bytes = encode_value(value_type, value)
        kv_entries.append((key_bytes, value_type, value_bytes))
    kv_region = bytearray()
    for key_bytes, value_type, value_bytes in kv_entries:
        kv_region += struct.pack("<I", len(key_bytes))
        kv_region += key_bytes
        kv_region += struct.pack("<II", value_type, len(value_bytes))
        kv_region += value_bytes

    # Layout: header | KV region | resource table | data region.
    table_offset = HEADER_SIZE + len(kv_region)
    data_offset = table_offset + ENTRY_SIZE * len(entries)

    header = struct.pack("<4sIIIII", MAGIC, VERSION, len(kv_entries), len(entries), alignment, 0)

    table = bytearray()
    payloads = bytearray()
    file_pos = data_offset
    for name, path, access in entries:
        with open(path, "rb") as f:
            data = f.read()
        aligned = ((file_pos + alignment - 1) // alignment) * alignment
        payloads += b"\x00" * (aligned - file_pos)
        table += _encode_entry(name, aligned, len(data), access)
        payloads += data
        file_pos = aligned + len(data)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(kv_region)
        f.write(table)
        f.write(payloads)
    return output_path


def _encode_entry(name: str, offset: int, size: int, access: int) -> bytes:
    name_bytes = name.encode("utf-8")
    return name_bytes + b"\x00" * (NAME_MAX - len(name_bytes)) + struct.pack("<QQII", offset, size, access, 0)


# ─── unpack ────────────────────────────────────────────────────────────────

def unpack(msl_path: str, out_dir: str, emit_kv: Optional[str] = None) -> Dict[str, Any]:
    """Parse a v1 ``.msl`` file, write every resource under ``out_dir``.

    Returns the KV metadata dict.  Unknown KV keys are skipped (forward
    compatibility); unknown types and versions are rejected.  ``emit_kv``,
    if given, additionally writes the metadata as JSON to that path.
    """
    with open(msl_path, "rb") as f:
        data = f.read()
    return unpack_bytes(data, out_dir, emit_kv)


def unpack_bytes(data: bytes, out_dir: str, emit_kv: Optional[str] = None) -> Dict[str, Any]:
    """Parse an in-memory v1 ``.msl`` buffer, extracting its resources."""
    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        raise MslPackError("not a .msl file (bad magic)")
    version, kv_count, resource_count, alignment, _ = struct.unpack_from("<IIIII", data, 4)
    if version != VERSION:
        raise MslPackError(f"unsupported .msl version {version} (this packer understands {VERSION})")
    if alignment <= 0:
        raise MslPackError(f"invalid alignment {alignment}")

    pos = HEADER_SIZE
    kv: Dict[str, Any] = {}
    for _ in range(kv_count):
        if pos + 4 > len(data):
            raise MslPackError("KV region truncated")
        key_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + key_len + 8 > len(data):
            raise MslPackError("KV key truncated")
        key = data[pos:pos + key_len].decode("utf-8")
        pos += key_len
        value_type, value_len = struct.unpack_from("<II", data, pos)
        pos += 8
        if pos + value_len > len(data):
            raise MslPackError(f"KV value truncated for {key!r}")
        # Unknown value types are a layout contract violation: reject.
        kv[key] = decode_value(value_type, data[pos:pos + value_len])
        pos += value_len

    table_offset = pos
    if table_offset + ENTRY_SIZE * resource_count > len(data):
        raise MslPackError("resource table out of range")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(resource_count):
        base = table_offset + ENTRY_SIZE * i
        name_bytes = data[base:base + NAME_MAX].split(b"\x00", 1)[0]
        name = name_bytes.decode("utf-8")
        offset, size, access, _ = struct.unpack_from("<QQII", data, base + NAME_MAX)
        if offset > len(data) or size > len(data) - offset:
            raise MslPackError(f"resource {name!r} range out of file bounds")
        if access not in (ACCESS_MMAP, ACCESS_READ):
            raise MslPackError(f"resource {name!r} has invalid access mode {access}")
        payload = data[offset:offset + size]
        target = os.path.normpath(os.path.join(out_dir, name))
        if not target.startswith(os.path.normpath(out_dir) + os.sep) and target != os.path.normpath(out_dir):
            raise MslPackError(f"resource name escapes output dir: {name!r}")
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(target, "wb") as f:
            f.write(payload)

    if emit_kv is not None:
        with open(emit_kv, "w", encoding="utf-8") as f:
            json.dump(kv, f, indent=2, sort_keys=True)
    return kv


# ─── manifest (export metadata dict) -> KV ────────────────────────────────
# v1 KV key list; the C++ runtime mirrors this in msl_format.h.  Adding a
# key here does NOT bump the version (readers skip unknown keys).  Only
# keys with runtime consumers are listed (verified against nnrt_backend/
# llm.cpp): architecture params, NPU config, resource assets and the eos
# token id (NNRTBackend reads stop_token_ids.front() as eos_id).

ARCH_U32_KEYS = {
    "num_layers": "arch.num_layers",
    "hidden_size": "arch.hidden_size",
    "intermediate_size": "arch.intermediate_size",
    "num_heads": "arch.num_heads",
    "num_kv_heads": "arch.num_kv_heads",
    "head_dim": "arch.head_dim",
    "vocab_size": "arch.vocab_size",
    "max_position_embeddings": "arch.max_position_embeddings",
    "tie_word_embeddings": "arch.tie_word_embeddings",
}
ARCH_F32_KEYS = {
    "rope_theta": "arch.rope_theta",
    "norm_eps": "arch.norm_eps",
}
ASSET_KEYS = {
    "tokenizer": "asset.tokenizer",
    "embedding": "asset.embedding",
    "embedding_fp16": "asset.embedding_fp16",
    "rope_sin": "asset.rope_sin",
    "rope_cos": "asset.rope_cos",
    "attention_mask": "asset.attention_mask",
}
NPU_U32_KEYS = {
    "max_length": "npu.max_length",
    "chunk_size": "npu.chunk_size",
    "scale_gp_size": "npu.scale_gp_size",
}
NPU_BOOL_KEYS = {
    "embedding_quant": "npu.embedding_quant",
}


def manifest_to_kv(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Map an export manifest dict (the ``build_manifest`` shape) to v1 KV.

    Redundant sections with no runtime consumer are intentionally dropped:
    ``generation_policy`` (the tokenizer reads stop/suppress token strings
    from the embedded vocab.bin policy segment) and ``graph_io`` (the
    runtime contract check derives I/O from num_layers + NNRT descs).
    The one generation value the runtime does consume — the eos token id
    (NNRTBackend eos_id) — is kept as ``gen.eos_token_id``.
    """
    kv: Dict[str, Any] = {}
    if "model_name" in manifest:
        kv["model.name"] = manifest["model_name"]
    if "version" in manifest:
        kv["model.version"] = manifest["version"]
    if "format_version" in manifest:
        kv["model.format_version"] = manifest["format_version"]
    pipeline = manifest.get("pipeline_config") or {}
    if "precision" in pipeline:
        kv["model.dtype"] = pipeline["precision"]

    arch = manifest.get("architecture") or {}
    for src, dst in ARCH_U32_KEYS.items():
        if src in arch:
            kv[dst] = int(arch[src])
    for src, dst in ARCH_F32_KEYS.items():
        if src in arch:
            kv[dst] = float(arch[src])

    litert = manifest.get("litert") or {}
    if isinstance(litert.get("prefill"), str):
        kv["litert.prefill.path"] = litert["prefill"]
    if "prefill_seq_len" in litert:
        kv["litert.prefill.seq_len"] = int(litert["prefill_seq_len"])
    decode = litert.get("decode")
    if isinstance(decode, str):
        kv["litert.decode.path"] = decode
    elif isinstance(decode, dict):
        for src, dst in (("path", "litert.decode.path"),
                         ("dynamic_past_len", "litert.decode.dynamic_past_len"),
                         ("past_len", "litert.decode.past_len"),
                         ("max_past_len", "litert.decode.max_past_len")):
            if src in decode:
                kv[dst] = decode[src]
    variants = litert.get("decode_variants")
    if variants is not None:
        kv["litert.decode_variants"] = json.dumps(variants, separators=(",", ":"))

    for src, dst in ASSET_KEYS.items():
        if src in (manifest.get("assets") or {}):
            kv[dst] = manifest["assets"][src]
    npu = manifest.get("npu") or {}
    for src, dst in NPU_U32_KEYS.items():
        if src in npu:
            kv[dst] = int(npu[src])
    for src, dst in NPU_BOOL_KEYS.items():
        if src in npu:
            kv[dst] = bool(npu[src])
    generation = manifest.get("generation") or {}
    stop_ids = generation.get("stop_token_ids") or []
    if stop_ids:
        kv["gen.eos_token_id"] = int(stop_ids[0])
    return kv


logger = logging.getLogger(__name__)


# ─── CLI (pack_info.cfg DSL compatible with mspacker_tool) ─────────────────

def parse_config(config_text: str, base_dir: str) -> Tuple[str, Dict[str, Any], List[Tuple[str, str, int]]]:
    """Parse the pack_info.cfg DSL.

    Lines: ``OUTPUT_PATH:<path>``, ``ENTRY:<relpath> <access>``,
    ``KV:<key> <type> <value>`` (type in
    bool/uint32/uint64/float32/string/string[], string[] as JSON array),
    ``KVJSON:<manifest.json>`` (auto-mapped via :func:`manifest_to_kv`).
    """
    output_path: Optional[str] = None
    kv: Dict[str, Any] = {}
    resources: List[Tuple[str, str, int]] = []
    for raw_line in config_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("OUTPUT_PATH:"):
            output_path = line[len("OUTPUT_PATH:"):]
        elif line.startswith("ENTRY:"):
            rest = line[len("ENTRY:"):].split()
            if len(rest) != 2:
                raise MslPackError(f"ENTRY expects '<name> <access>': {line!r}")
            name, access = rest[0], int(rest[1])
            resources.append((name, _join(base_dir, name), access))
        elif line.startswith("KVJSON:"):
            manifest_path = _join(base_dir, line[len("KVJSON:"):])
            with open(manifest_path, encoding="utf-8") as f:
                kv.update(manifest_to_kv(json.load(f)))
        elif line.startswith("KV:"):
            rest = line[len("KV:"):].split(maxsplit=2)
            if len(rest) != 3:
                raise MslPackError(f"KV expects '<key> <type> <value>': {line!r}")
            key, type_name, raw_value = rest
            kv[key] = _parse_kv_scalar(type_name, raw_value)
        else:
            raise MslPackError(f"unrecognized config line: {line!r}")
    if output_path is None:
        raise MslPackError("OUTPUT_PATH is required")
    return output_path, kv, resources


def _join(base_dir: str, name: str) -> str:
    if os.path.isabs(name) or not base_dir:
        return name
    return os.path.join(base_dir, name)


def _parse_kv_scalar(type_name: str, raw: str) -> Any:
    """Parse a scalar config value for a given v1 KV type name."""
    if type_name == "bool":
        return raw.lower() in ("1", "true", "yes")
    if type_name == "uint32":
        return int(raw)
    if type_name == "uint64":
        return int(raw)
    if type_name == "float32":
        return float(raw)
    if type_name == "string":
        return raw
    if type_name == "string[]":
        return json.loads(raw)
    raise MslPackError(f"unknown KV type {type_name!r}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="msl_pack",
        description="Pack/unpack single-file .msl (v1) model containers.")
    parser.add_argument("-d", metavar="CONFIG", help="pack according to pack_info.cfg")
    parser.add_argument("-g", metavar="PACK_FILE", help="unpack into the current directory")
    parser.add_argument("--out-dir", default=".", help="unpack destination (default: current dir)")
    args = parser.parse_args(argv)

    if args.d and args.g:
        parser.error("choose exactly one of -d / -g")
    if args.d:
        base_dir = _dirname(args.d)
        with open(args.d, encoding="utf-8") as f:
            output_path, kv, resources = parse_config(f.read(), base_dir)
        pack(output_path, kv, resources)
        print(f"packed {output_path} ({len(resources)} resources, {len(kv)} KV entries)")
        return 0
    if args.g:
        unpack(args.g, args.out_dir)
        print(f"unpacked {args.g} into {args.out_dir}")
        return 0
    parser.error("need -d or -g")
    return 1  # unreachable (parser.error exits)


def _dirname(path: str) -> str:
    return os.path.dirname(path)




# ─── export artifact -> v1 KV schema ───────────────────────────────────────

def build_manifest(package_name, architecture, npu_config, generation_policy, omc_name):
    """Build the export manifest dict consumed by ``manifest_to_kv``."""
    manifest = {
        "format_version": "1.0",
        "model_name": package_name,
        "version": "1.0.0",
        "pipeline_config": {"model_type": "llm", "precision": "fp16"},
        "architecture": dict(architecture),
        "litert": {
            "precision": "fp16",
            "prefill": f"npu_offline/{omc_name}",
        },
        "assets": {
            "tokenizer": "vocab/vocab.bin",
            "embedding": "assets/embedding_quant.bin",
            "rope_cos": "assets/rope_cos.bin",
            "rope_sin": "assets/rope_sin.bin",
            "attention_mask": "assets/attention_mask.bin",
        },
        "npu": {
            "max_length": int(npu_config["max_length"]),
            "chunk_size": int(npu_config["chunk_size"]),
            "embedding_quant": bool(npu_config["embedding_quant"]),
            "scale_gp_size": int(npu_config.get("scale_gp_size", 32)),
        },
    }
    if generation_policy:
        manifest["generation"] = dict(generation_policy)
    return manifest


def build_single_file_msl(omc_path, vocab_path, embedding_path, rope_cos, rope_sin,
                          attention_mask, architecture, npu_config, generation_policy,
                          package_name, output_path):
    """Pack the export artifacts into a single-file ``.msl`` (v1).

    The .omc graph is mmap'd at runtime (access 0); tokenizer/rope/mask
    assets are read (access 1); the quantized embedding is mmap'd too.
    """
    omc_name = os.path.basename(omc_path)
    manifest = build_manifest(package_name, architecture, npu_config, generation_policy, omc_name)
    kv = manifest_to_kv(manifest)

    resources = [
        (f"npu_offline/{omc_name}", omc_path, ACCESS_MMAP),
        ("vocab/vocab.bin", vocab_path, ACCESS_READ),
        ("assets/embedding_quant.bin", embedding_path, ACCESS_MMAP),
        ("assets/rope_cos.bin", rope_cos, ACCESS_READ),
        ("assets/rope_sin.bin", rope_sin, ACCESS_READ),
        ("assets/attention_mask.bin", attention_mask, ACCESS_READ),
    ]

    logger.info("packing %d resources, %d KV entries -> %s", len(resources), len(kv), output_path)
    pack(output_path, kv, resources)
    return output_path


if __name__ == "__main__":
    sys.exit(main())
