#!/usr/bin/env python3
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
"""Regenerate the v1 format golden artifacts.

Produces ``golden_v1.msl`` (a deterministic, minimal-but-complete .msl
covering all six KV value types, aligned payloads, access=0/1 and
subdirectory resource names) plus ``golden_v1.expected.json`` (the
expected layout for tests).

Re-running this script must reproduce byte-identical ``golden_v1.msl``;
the Python test asserts that, and the C++ test validates the file
against the layout contract.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "export"))
# pylint: disable=wrong-import-position  # export/ added to sys.path above
from utils import msl_pack as mp

HERE = os.path.dirname(os.path.abspath(__file__))
MSL_PATH = os.path.join(HERE, "golden_v1.msl")
EXPECTED_PATH = os.path.join(HERE, "golden_v1.expected.json")

# Deterministic payload bytes so the golden file is reproducible.
def payload(size: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(size))


RESOURCES = [
    ("npu_offline/x.omc", mp.ACCESS_MMAP, 70000),
    ("assets/embedding_quant.bin", mp.ACCESS_MMAP, 12345),
    ("vocab/vocab.bin", mp.ACCESS_READ, 3000),
    ("a.bin", mp.ACCESS_READ, 1),
]

KV = {
    # one entry per v1 KV type
    "model.name": "qwen2.5-0.5b",                  # string
    "model.dtype": "fp16",                         # string
    "arch.num_layers": 24,                         # uint32
    "arch.rope_theta": 10000.0,                    # float32
    "arch.norm_eps": 1e-6,                         # float32 (lossy roundtrip, bytes are golden)
    "npu.max_length": 1024,                        # uint32
    "npu.embedding_quant": True,                   # bool
    "gen.eos_token_id": 151643,                    # uint32 (NNRTBackend eos_id)
    "litert.decode_variants": '[{"past_len":1,"path":"npu_offline/x.omc"}]',  # string (JSON payload)
    "string.array": ["a", "bb", "ccc"],            # string[] (extra key: unknown to runtime, skipped)
}


def build_expected() -> dict:
    """Rebuild the golden payload and record offsets for the expected file."""
    # Replicate the pack layout to record offsets for the expected file.
    kv_region = bytearray()
    kv_meta = []
    for key, value in KV.items():
        key_bytes = key.encode("utf-8")
        value_type = mp.infer_type(value)
        value_bytes = mp.encode_value(value_type, value)
        kv_region += struct_pack_u32(len(key_bytes)) + key_bytes
        kv_region += struct_pack_u32(value_type) + struct_pack_u32(len(value_bytes)) + value_bytes
        kv_meta.append({"key": key, "type": value_type, "value_len": len(value_bytes)})

    data_offset = mp.HEADER_SIZE + len(kv_region)
    file_pos = data_offset
    entries = []
    for name, access, size in RESOURCES:
        aligned = ((file_pos + mp.DEFAULT_ALIGNMENT - 1) // mp.DEFAULT_ALIGNMENT) * mp.DEFAULT_ALIGNMENT
        entries.append({"name": name, "offset": aligned, "size": size, "access": access})
        file_pos = aligned + size
    return {
        "magic": list(b".MSL"),
        "version": mp.VERSION,
        "alignment": mp.DEFAULT_ALIGNMENT,
        "kv": kv_meta,
        "resources": entries,
        "file_size": file_pos,
    }


def struct_pack_u32(v: int) -> bytes:
    import struct
    return struct.pack("<I", v)


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="golden_gen_") as tmp:
        paths = []
        for name, access, size in RESOURCES:
            path = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(payload(size))
            paths.append((name, path, access))
        mp.pack(MSL_PATH, KV, paths)

    expected = build_expected()
    with open(EXPECTED_PATH, "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)
    print(f"wrote {MSL_PATH} ({os.path.getsize(MSL_PATH)} bytes) and {EXPECTED_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
