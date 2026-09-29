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
"""Golden tests for the .msl v1 format.

``tests/data/golden_v1.msl`` is the authoritative byte layout: the C++
test validates it from the runtime side, this test validates that
``msl_pack.py`` can (a) reproduce it byte-identically from the same
inputs, and (b) decode it back to the expected metadata/resources.
"""

import importlib.util
import json
import os
import re
import struct
import tempfile

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "..", "data")

# msl_pack is stdlib-only; load it straight from the export tree by file path
# (same pattern as tests/py/test_export_entrypoint.py) — no sys.path mutation.
_MSL_PACK = os.path.join(TESTS_DIR, "..", "..", "export", "utils", "msl_pack.py")
_spec = importlib.util.spec_from_file_location("msl_pack", _MSL_PACK)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

GOLDEN_MSL = os.path.join(DATA_DIR, "golden_v1.msl")
GOLDEN_EXPECTED = os.path.join(DATA_DIR, "golden_v1.expected.json")

KV = {
    "model.name": "qwen2.5-0.5b",
    "model.dtype": "fp16",
    "arch.num_layers": 24,
    "arch.rope_theta": 10000.0,
    "arch.norm_eps": 1e-6,
    "npu.max_length": 1024,
    "npu.embedding_quant": True,
    "gen.eos_token_id": 151643,
    "litert.decode_variants": '[{"past_len":1,"path":"npu_offline/x.omc"}]',
    "string.array": ["a", "bb", "ccc"],
}
RESOURCES = [
    ("npu_offline/x.omc", mp.ACCESS_MMAP, 70000),
    ("assets/embedding_quant.bin", mp.ACCESS_MMAP, 12345),
    ("vocab/vocab.bin", mp.ACCESS_READ, 3000),
    ("a.bin", mp.ACCESS_READ, 1),
]


def _payload(size: int) -> bytes:
    return bytes((i * 7 + 3) % 256 for i in range(size))


def test_golden_reproducible():
    """Re-packing the same inputs must produce byte-identical output."""
    with tempfile.TemporaryDirectory(prefix="golden_repro_") as tmp:
        paths = []
        for name, access, size in RESOURCES:
            path = os.path.join(tmp, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(_payload(size))
            paths.append((name, path, access))
        rebuilt = os.path.join(tmp, "rebuilt.msl")
        mp.pack(rebuilt, KV, paths)
        with open(GOLDEN_MSL, "rb") as f:
            golden = f.read()
        with open(rebuilt, "rb") as f:
            produced = f.read()
        assert produced == golden, "msl_pack.py no longer reproduces golden_v1.msl byte-identically"


def test_golden_expected_json_matches():
    """The checked-in expected file must reflect the actual golden bytes."""
    with open(GOLDEN_EXPECTED, encoding="utf-8") as f:
        expected = json.load(f)
    with open(GOLDEN_MSL, "rb") as f:
        data = f.read()
    assert data[:4] == bytes(expected["magic"])
    version, kv_count, res_count, alignment, _ = struct.unpack_from("<IIIII", data, 4)
    assert version == expected["version"]
    assert kv_count == expected["kv_count"] if "kv_count" in expected else kv_count == len(expected["kv"])
    assert res_count == len(expected["resources"])
    assert alignment == expected["alignment"]
    assert len(data) == expected["file_size"]


def test_golden_unpack():
    """Decoding the golden file yields the original metadata and resources."""
    with tempfile.TemporaryDirectory(prefix="golden_unpack_") as tmp:
        got_kv = mp.unpack(GOLDEN_MSL, tmp)
        assert len(got_kv) == len(KV)
        for key, value in KV.items():
            assert mp.encode_value(mp.infer_type(got_kv[key]), got_kv[key]) == mp.encode_value(
                mp.infer_type(value), value), key
        for name, _, size in RESOURCES:
            with open(os.path.join(tmp, name), "rb") as f:
                payload = f.read()
            assert payload == _payload(size), name


def test_pack_rejects_invalid_resource_names():
    """pack() refuses path-escaping and overlong resource names up front."""
    with tempfile.TemporaryDirectory(prefix="golden_badname_") as tmp:
        path = os.path.join(tmp, "a.bin")
        with open(path, "wb") as f:
            f.write(b"x")
        out = os.path.join(tmp, "o.msl")
        # Note: duplicate names are NOT among pack()'s v1 validations (the
        # table allows them; last one wins on unpack) — the enforced name
        # rules are path safety and the 1..63-byte length budget.
        with pytest.raises(
            mp.MslPackError,
            match=re.escape("resource name has invalid path segment: '../esc.bin'"),
        ):
            mp.pack(out, KV, [("../esc.bin", path, mp.ACCESS_READ)])
        overlong = "q" * mp.NAME_MAX
        expected = f"resource name must be 1..{mp.NAME_MAX - 1} bytes: {overlong!r}"
        with pytest.raises(mp.MslPackError, match=re.escape(expected)):
            mp.pack(out, KV, [(overlong, path, mp.ACCESS_READ)])


def test_pack_rejects_invalid_access_mode():
    """pack() refuses resource access modes outside {0 mmap, 1 read}."""
    with tempfile.TemporaryDirectory(prefix="golden_access_") as tmp:
        path = os.path.join(tmp, "x.omc")
        with open(path, "wb") as f:
            f.write(b"omc")
        with pytest.raises(
            mp.MslPackError,
            match=re.escape("invalid access mode 7 for 'npu_offline/x.omc'"),
        ):
            mp.pack(os.path.join(tmp, "o.msl"), KV, [("npu_offline/x.omc", path, 7)])
