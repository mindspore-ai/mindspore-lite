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
"""Error-path tests for ``msl_pack.unpack_bytes`` (malformed .msl buffers).

Every case starts from a valid buffer produced by ``pack()`` (golden
round-trip construction) and corrupts exactly one region, so the asserted
message pinpoints which parse stage (_parse_kv_region /
_extract_resources / _extract_resource) rejected the input.
"""

import re
import struct

import pytest

from utils import msl_pack as mp

# Fixed inputs: two resources exercising both access modes and a
# sub-directory name; KV mixes the value kinds the parser must decode.
KV = {
    "model.name": "tiny",
    "arch.num_layers": 2,
    "npu.embedding_quant": True,
}
RESOURCES = [
    ("npu_offline/x.omc", mp.ACCESS_MMAP, b"OMC-PAYLOAD"),
    ("a.bin", mp.ACCESS_READ, b"tail"),
]


def _packed_bytes(tmp_path):
    """Round-trip helper: pack KV/RESOURCES and return the .msl bytes."""
    entries = []
    for name, access, payload in RESOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        entries.append((name, str(path), access))
    out = tmp_path / "valid.msl"
    mp.pack(str(out), KV, entries)
    return out.read_bytes()


def _table_offset():
    """File offset of the resource table for the fixed KV above."""
    return mp.HEADER_SIZE + len(mp._encode_kv_region(KV)[1])  # pylint: disable=protected-access


def test_header_truncated_rejected(tmp_path):
    """A buffer shorter than the 24-byte header is rejected up front."""
    data = _packed_bytes(tmp_path)
    assert len(data) > 10
    with pytest.raises(mp.MslPackError, match=re.escape("not a .msl file (bad magic)")):
        mp.unpack_bytes(data[:10], str(tmp_path / "out"))


def test_kv_key_length_out_of_bounds_rejected(tmp_path):
    """A KV key length field pointing past EOF fails the region bounds check."""
    data = bytearray(_packed_bytes(tmp_path))
    struct.pack_into("<I", data, mp.HEADER_SIZE, 0xFFFFFF00)
    with pytest.raises(mp.MslPackError, match=re.escape("KV key truncated")):
        mp.unpack_bytes(bytes(data), str(tmp_path / "out"))


def test_resource_table_truncated_rejected(tmp_path):
    """A file cut inside the resource table fails the table range check."""
    data = _packed_bytes(tmp_path)
    cut = _table_offset() + mp.ENTRY_SIZE * len(RESOURCES) - 10
    with pytest.raises(mp.MslPackError, match=re.escape("resource table out of range")):
        mp.unpack_bytes(data[:cut], str(tmp_path / "out"))


def test_data_region_truncated_rejected(tmp_path):
    """Dropping the last payload byte puts the final resource out of file bounds."""
    data = _packed_bytes(tmp_path)
    with pytest.raises(
        mp.MslPackError,
        match=re.escape("resource 'a.bin' range out of file bounds"),
    ):
        mp.unpack_bytes(data[:-1], str(tmp_path / "out"))


def test_kv_unknown_value_type_rejected(tmp_path):
    """An unknown KV type code is a layout contract violation and is rejected."""
    data = bytearray(_packed_bytes(tmp_path))
    first_key = next(iter(KV))
    type_offset = mp.HEADER_SIZE + 4 + len(first_key.encode("utf-8"))
    struct.pack_into("<I", data, type_offset, 99)
    with pytest.raises(mp.MslPackError, match=re.escape("unknown KV value type: 99")):
        mp.unpack_bytes(bytes(data), str(tmp_path / "out"))


def test_resource_name_escape_rejected(tmp_path):
    """A table entry whose name escapes out_dir ('../x') is rejected on extract."""
    data = bytearray(_packed_bytes(tmp_path))
    base = _table_offset()
    data[base:base + mp.NAME_MAX] = b"../x" + b"\x00" * (mp.NAME_MAX - 4)
    with pytest.raises(
        mp.MslPackError,
        match=re.escape("resource name escapes output dir: '../x'"),
    ):
        mp.unpack_bytes(bytes(data), str(tmp_path / "out"))
