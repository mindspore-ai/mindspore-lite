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
"""Unit tests for the OMC kernel-bin parse and CUBE padding cleanup.

The default, binary-free coverage is synthetic OMC blobs plus a replay of the golden geometry
from ``omc_kernel_bins_golden.json``; byte-level regression of real ``.omc`` files is left to
the opt-in ``MSL_OMC_GOLDEN_DIR`` tests below and to the gate Smoke ST.
"""

import hashlib
import json
import logging
import os
import struct

import pytest

from utils import omc_compiler
from utils import omc_kernel_bins as kb

GOLDEN_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "omc_kernel_bins_golden.json")
BLACKLISTED = struct.pack("<I", 0xF0C00001)
# Bytes that live outside the kernel-bin table in a real OMC and that the old
# fixed 80-byte window overwrote: a TLV head and the weight index JSON.
TLV_HEAD = b"\xcd\xab\xcd\xab" + struct.pack("<I", 64)
WEIGHT_JSON = b'{"WeightInfoLists":[{"WeightBlockInfo":[{"hash":1,"offset":0,"size":512}]}]}'


def _elf64(name: str, length: int) -> bytes:
    """ELF64 LE image of exactly ``length`` bytes (``e_shoff + e_shentsize * e_shnum``)."""
    body_len = length - 64 - 128
    body = name.encode() + b"\x00"
    body += bytes((i * 7 + 1) & 0xFF for i in range(body_len - len(body)))
    hdr = bytearray(64)
    hdr[0:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", hdr, 16, 1, 0x1029, 1)
    struct.pack_into("<Q", hdr, 0x28, 64 + len(body))  # e_shoff
    struct.pack_into("<H", hdr, 0x34, 64)  # e_ehsize
    struct.pack_into("<HH", hdr, 0x3A, 64, 2)  # e_shentsize, e_shnum
    return bytes(hdr) + body + bytes(128)


def _kernel(name: str, pad: int, pad_fill: int = 0x5A):
    """A CUBE kernel: ELF followed by ``pad`` dirty bytes up to a 256-B boundary."""
    elf = _elf64(name, 768 - pad)
    return (kb.CUBE_MAGIC, name, elf + bytes([pad_fill]) * pad)


def _cube(name: str, formal_len: int, pad: int):
    """A CUBE kernel with an explicit formal ELF length followed by ``pad`` dirty bytes."""
    return (kb.CUBE_MAGIC, name, _elf64(name, formal_len) + bytes([0x5A]) * pad)


def _builtin(name: str, size: int = 512):
    """A builtin executor bin: not ELF, never touched."""
    return (kb.BUILTIN_MAGIC, name, bytes((0x11 + i) & 0xFF for i in range(size)))


def _omc(kernels, prefix=b"OMCHEAD" * 9, suffix=TLV_HEAD + WEIGHT_JSON):
    """Assemble ``prefix + kernel-bin table + suffix``; return ``(bytes, blob offsets)``."""
    records = b""
    for magic, name, blob in kernels:
        name_bytes = name.encode() + b"\x00"
        records += struct.pack("<IQIII", 1, 24 + len(name_bytes), magic, len(blob), 0) + name_bytes
    blobs = b"".join(blob for _, _, blob in kernels)
    table = records + struct.pack("<II", 2, len(blobs) + 8)
    offsets, cursor = [], len(prefix) + len(table)
    for _, _, blob in kernels:
        offsets.append(cursor)
        cursor += len(blob)
    return prefix + table + blobs + suffix, offsets


def _sample():
    """Pads of 8/88/168 bytes around builtins, like the d1101 layout."""
    return [
        _kernel("te_msquant4n0group32_gen_a", 8),
        _builtin("executor_add_eltwise_half_nd"),
        _kernel("te_msscatternd_gen_b", 88),
        _kernel("te_msgroupmatmul_gen_c", 168),
        _builtin("executor_reshape_nd", 256),
    ]


def _pads(data):
    return [kbin.pad for kbin in kb.parse_kernel_bins(data) if kbin.is_cube]


def test_parse_gives_exact_blob_and_pad_intervals():
    """Blob offsets follow record order and pads end exactly at blob + size."""
    data, offsets = _omc(_sample())
    bins = kb.parse_kernel_bins(data)
    assert [kbin.blob_off for kbin in bins] == offsets
    assert [kbin.is_cube for kbin in bins] == [True, False, True, True, False]
    assert [kbin.pad[1] - kbin.pad[0] for kbin in bins if kbin.is_cube] == [8, 88, 168]
    assert all(kbin.pad is None for kbin in bins if not kbin.is_cube)


def test_zero_clears_only_padding():
    """Regression for the old 80-B window (B/C/D): builtins, TLV head and JSON keep their bytes."""
    data, _ = _omc(_sample())
    out, stats = kb.zero_cube_pads(data)
    pads = _pads(data)
    for start, end in pads:
        assert out[start:end] == bytes(end - start)
    cursor = 0
    for start, end in pads:
        assert out[cursor:start] == data[cursor:start]
        cursor = end
    assert out[cursor:] == data[cursor:]
    assert out.endswith(TLV_HEAD + WEIGHT_JSON)
    assert stats.cube_kernels == 3 and stats.pad_bytes == 264 and stats.cleared_bytes == 264


def test_g_control_last_pad_word_is_cleared():
    """G: a hit in the last word of a 168-B pad (beyond any 80-B window) is scanned and zeroed."""
    data, _ = _omc(_sample())
    start, end = _pads(data)[2]
    assert end - start == 168
    dirty = data[:end - 4] + BLACKLISTED + data[end:]
    out, stats = kb.zero_cube_pads(dirty)
    assert len(stats.hits_before) == 1 and "pad164:0xf0c00001" in stats.hits_before[0]
    assert out == kb.zero_cube_pads(data)[0]


def test_h_control_word_after_blob_is_not_scanned_or_changed():
    """H: the first word after a CUBE blob belongs to the builtin bin; leave it and do not flag it."""
    data, _ = _omc(_sample())
    bins = kb.parse_kernel_bins(data)
    after = bins[0].blob_end
    assert bins[1].blob_off == after and not bins[1].is_cube
    dirty = data[:after] + BLACKLISTED + data[after + 4:]
    out, stats = kb.zero_cube_pads(dirty)
    assert stats.hits_before == ()
    assert out[after:after + 4] == BLACKLISTED


def test_hit_inside_elf_body_warns(caplog):
    """A blacklisted word in kernel code cannot be fixed by the export: warn and continue."""
    data, offsets = _omc(_sample())
    body = offsets[2] + 128
    dirty = data[:body] + BLACKLISTED + data[body + 4:]
    with caplog.at_level(logging.WARNING, logger=kb.__name__):
        out, stats = kb.zero_cube_pads(dirty)
    assert out[body:body + 4] == BLACKLISTED
    for start, end in _pads(dirty):
        assert out[start:end] == bytes(end - start)
    assert stats.body_hit_kernels == 1 and stats.body_hits == 1
    warnings = [rec for rec in caplog.records if rec.name == kb.__name__
                and rec.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "inside kernel ELF code" in text
    assert hex(body) in text and "0xf0c00001" in text


def test_body_hits_in_two_kernels_warn_with_stats(caplog):
    """Each affected kernel gets exactly one warning, and the stats count every word."""
    data, offsets = _omc(_sample())
    first = offsets[0] + 128
    second, third = offsets[2] + 128, offsets[2] + 132
    dirty = data
    for off in (first, second, third):
        dirty = dirty[:off] + BLACKLISTED + dirty[off + 4:]
    with caplog.at_level(logging.WARNING, logger=kb.__name__):
        out, stats = kb.zero_cube_pads(dirty)
    assert stats.body_hit_kernels == 2 and stats.body_hits == 3
    assert out[first:first + 4] == BLACKLISTED
    assert out[second:second + 4] == BLACKLISTED
    assert out[third:third + 4] == BLACKLISTED
    warnings = [rec for rec in caplog.records if rec.name == kb.__name__
                and rec.levelno == logging.WARNING]
    assert len(warnings) == 2


def test_padding_hit_after_cleanup_fails_closed():
    """The padding re-scan still fails closed; a body-only hit is accepted by the helper."""
    data, offsets = _omc(_sample())
    pads = [(kbin, kbin.pad) for kbin in kb.parse_kernel_bins(data) if kbin.is_cube]
    start, _ = pads[1][1]
    dirty_pad = data[:start + 4] + BLACKLISTED + data[start + 8:]
    with pytest.raises(kb.OmcFormatError, match="still matches the device ISA blacklist"):
        kb._assert_pads_clean(dirty_pad, pads)  # pylint: disable=protected-access
    body = offsets[2] + 128
    dirty_body = data[:body] + BLACKLISTED + data[body + 4:]
    kb._assert_pads_clean(dirty_body, pads)  # pylint: disable=protected-access


def _corrupt_blob_len(data, offsets):
    del offsets
    table_end = data.find(struct.pack("<I", 2), data.find(b"executor_reshape_nd"))
    return data[:table_end + 4] + struct.pack("<I", 1) + data[table_end + 8:]


def _corrupt_size_alignment(data, offsets):
    """Shift 4 bytes from the first record's size to the next one; blob_len still adds up."""
    del offsets
    first = data.find(struct.pack("<I", kb.CUBE_MAGIC))
    second = data.find(struct.pack("<I", kb.BUILTIN_MAGIC), first)
    out = bytearray(data)
    for rec, delta in ((first, 4), (second, -4)):
        size, = struct.unpack_from("<I", data, rec + 4)
        struct.pack_into("<I", out, rec + 4, size + delta)
    return bytes(out)


def _corrupt_elf_magic(data, offsets):
    return data[:offsets[0]] + b"\x00ELF" + data[offsets[0] + 4:]


def _stray_elf_outside_table(data, offsets):
    del offsets
    return data + _elf64("stray", 256)


def _no_table(data, offsets):
    del data, offsets
    return b"\x00" * 4096


def _truncate_mid_first_blob(data, offsets):
    """Cut the file in the middle of the first blob: the table survives, the blobs do not."""
    return data[:offsets[0] + 256]


def _truncate_after_descriptors(data, offsets):
    """Cut right after the descriptor sequence, inside the blob header, so blobs are incomplete."""
    return data[:offsets[0] - 4]


def _corrupt_elf_class(data, offsets):
    """Set e_ident[EI_CLASS] = 1 (ELF32) in the first CUBE blob."""
    return data[:offsets[0] + 4] + b"\x01" + data[offsets[0] + 5:]


def _corrupt_elf_data(data, offsets):
    """Set e_ident[EI_DATA] = 2 (big endian) in the first CUBE blob."""
    return data[:offsets[0] + 5] + b"\x02" + data[offsets[0] + 6:]


@pytest.mark.parametrize("corrupt, message", [
    (_corrupt_blob_len, "blob_len"),
    (_corrupt_size_alignment, "not a positive multiple of 256"),
    (_corrupt_elf_magic, "not a parseable ELF64"),
    (_corrupt_elf_class, "not a parseable ELF64"),
    (_corrupt_elf_data, "not a parseable ELF64"),
    (_stray_elf_outside_table, "parseable ELF64 image"),
    (_truncate_mid_first_blob, "no kernel-bin descriptor table found"),
    (_truncate_after_descriptors, "no kernel-bin descriptor table found"),
    (_no_table, "no kernel-bin descriptor table"),
])
def test_layout_violations_fail_closed(corrupt, message):
    """Any broken invariant raises instead of falling back to a guessed window."""
    data, offsets = _omc(_sample())
    with pytest.raises(kb.OmcFormatError, match=message):
        kb.zero_cube_pads(corrupt(data, offsets))


def test_name_beyond_backtrack_window_fails_closed():
    """G13: a name past the 256-B back-track window must fail closed, not drop the hidden record."""
    control, _ = _omc([_builtin("te_" + "n" * 200), _kernel("te_msquant4n0group32_gen_a", 8)])
    assert len(kb.parse_kernel_bins(control)) == 2
    data, _ = _omc([_builtin("te_" + "n" * 300), _kernel("te_msquant4n0group32_gen_a", 8)])
    with pytest.raises(kb.OmcFormatError, match="declares blob_len"):
        kb.parse_kernel_bins(data)


def test_size_must_equal_align256_of_elf():
    """A record size one alignment step too large means the pad would be >= 256 B: refuse."""
    magic, name, blob = _kernel("te_msrmsnorm_gen_d", 32)
    data, _ = _omc([(magic, name, blob + bytes(256))])
    with pytest.raises(kb.OmcFormatError, match="align256"):
        kb.zero_cube_pads(data)


def test_diff_outside_pads_is_detected():
    """The post-write guard rejects any byte change outside the padding intervals."""
    data, _ = _omc(_sample())
    pads = _pads(data)
    changed = bytearray(data)
    changed[pads[0][1]] ^= 0xFF  # first byte after the first pad
    with pytest.raises(kb.OmcFormatError, match="outside CUBE padding"):
        kb._assert_diff_within_pads(data, bytes(changed), pads)  # pylint: disable=protected-access


def test_sanitize_file_rewrites_atomically(tmp_path, caplog):
    """The file is replaced in one step and the log names the hit that was cleared."""
    data, _ = _omc(_sample())
    start, end = _pads(data)[1]
    path = tmp_path / "model.omc"
    path.write_bytes(data[:start + 4] + BLACKLISTED + data[start + 8:])
    with caplog.at_level(logging.INFO, logger=kb.__name__):
        stats = kb.sanitize_omc_kernel_pads(str(path))
    assert path.read_bytes()[start:end] == bytes(end - start)
    assert not os.path.exists(str(path) + ".pads.tmp")
    assert stats.hits_before and "pad4:0xf0c00001" in caplog.text


def test_sanitize_file_keeps_original_on_bad_layout(tmp_path):
    """Fail closed without touching the file when the layout is unknown."""
    path = tmp_path / "model.omc"
    path.write_bytes(b"\x00" * 1024)
    with pytest.raises(kb.OmcFormatError):
        kb.sanitize_omc_kernel_pads(str(path))
    assert path.read_bytes() == b"\x00" * 1024


def test_sanitize_file_warns_on_body_hit_and_still_zeroes_pads(tmp_path, caplog):
    """A body hit is a warning: the file is rewritten and the info log reports the count."""
    data, offsets = _omc(_sample())
    body = offsets[0] + 128
    path = tmp_path / "model.omc"
    path.write_bytes(data[:body] + BLACKLISTED + data[body + 4:])
    with caplog.at_level(logging.INFO, logger=kb.__name__):
        stats = kb.sanitize_omc_kernel_pads(str(path))
    written = path.read_bytes()
    assert written[body:body + 4] == BLACKLISTED
    for start, end in _pads(written):
        assert written[start:end] == bytes(end - start)
    assert stats.body_hits == 1
    assert "1 CUBE kernel(s) with 1 blacklist hit(s) inside ELF code" in caplog.text


def _config():
    return {"vocab_size": 64, "hidden_size": 128, "num_hidden_layers": 1,
            "num_key_value_heads": 1, "num_attention_heads": 2}


def test_compile_omc_zeroes_pads_of_omg_output(monkeypatch, tmp_path):
    """compile_omc runs the cleanup on the file omg produced."""
    data, _ = _omc(_sample())

    def fake_run(command, check, env):
        del check, env
        out = next(arg for arg in command if arg.startswith("--output=")).split("=", 1)[1]
        with open(out + ".omc", "wb") as handle:
            handle.write(data)

    monkeypatch.setattr(omc_compiler, "resolve_omg", lambda: "omg")
    monkeypatch.setattr(omc_compiler.subprocess, "run", fake_run)
    result = omc_compiler.compile_omc(str(tmp_path / "m.onnx"), _config(), max_seq_len=64,
                                      chunk_sizes=(32,), embedding_quant=None)
    with open(result, "rb") as handle:
        assert handle.read() == kb.zero_cube_pads(data)[0]


def test_compile_omc_fails_on_unparsable_omg_output(monkeypatch, tmp_path):
    """An OMC whose layout is unknown fails the export instead of shipping it unchecked."""
    def fake_run(command, check, env):
        del check, env
        out = next(arg for arg in command if arg.startswith("--output=")).split("=", 1)[1]
        with open(out + ".omc", "wb") as handle:
            handle.write(b"\x00" * 512)

    monkeypatch.setattr(omc_compiler, "resolve_omg", lambda: "omg")
    monkeypatch.setattr(omc_compiler.subprocess, "run", fake_run)
    with pytest.raises(kb.OmcFormatError):
        omc_compiler.compile_omc(str(tmp_path / "m.onnx"), _config(), max_seq_len=64,
                                 chunk_sizes=(32,), embedding_quant=None)


def _golden():
    with open(GOLDEN_JSON, encoding="utf-8") as handle:
        return {k: v for k, v in json.load(handle).items() if not k.startswith("_")}


def _contiguous_runs(kernels):
    """Split golden kernels at every gap between blobs: a real OMC starts a new table there."""
    runs = [[]]
    for kbin in kernels:
        if runs[-1] and kbin[1] != runs[-1][-1][1] + runs[-1][-1][2]:
            runs.append([])
        runs[-1].append(kbin)
    return runs


def _synthetic_omc(kernels):
    """OMC with the golden blob offsets, sizes and pads, synthetic names and dirty fillers."""
    chunks, cursor, index = [], 0, 0
    for run in _contiguous_runs(kernels):
        blobs = []
        for magic, blob_off, size, elf_end in run:
            name = f"synth_{'cube' if magic == kb.CUBE_MAGIC else 'builtin'}_{index:03d}"
            index += 1
            if elf_end is None:
                blobs.append(_builtin(name, size))
            else:
                blobs.append(_cube(name, elf_end - blob_off, size - (elf_end - blob_off)))
        records = b""
        for magic, name, blob in blobs:
            name_bytes = name.encode() + b"\x00"
            rec_len = 24 + len(name_bytes)
            records += struct.pack("<IQIII", 1, rec_len, magic, len(blob), 0) + name_bytes
        table_start = run[0][1] - len(records) - 8
        assert table_start >= cursor, f"golden run at {run[0][1]} leaves no room for its records"
        chunks.append(bytes([0x5A]) * (table_start - cursor) + records)
        chunks.append(struct.pack("<II", 2, sum(len(blob) for _, _, blob in blobs) + 8))
        chunks.extend(blob for _, _, blob in blobs)
        cursor = run[-1][1] + run[-1][2]
    chunks.append(TLV_HEAD + WEIGHT_JSON)
    return b"".join(chunks)


@pytest.mark.parametrize("label", sorted(_golden()))
def test_golden_intervals_of_real_omg_outputs(label):
    """Real DDK 6.1.1 outputs: intervals and the device-verified sanitized bytes (opt-in)."""
    golden_dir = os.environ.get("MSL_OMC_GOLDEN_DIR")
    path = os.path.join(golden_dir or "", f"{label}.omc")
    if not golden_dir or not os.path.isfile(path):
        pytest.skip("set MSL_OMC_GOLDEN_DIR to a directory with the real <label>.omc files")
    expected = _golden()[label]
    with open(path, "rb") as handle:
        data = handle.read()
    assert hashlib.sha256(data).hexdigest() == expected["sha256"]
    got = [[kbin.magic, kbin.blob_off, kbin.size, kbin.elf_end] for kbin in kb.parse_kernel_bins(data)]
    assert got == expected["kernels"]
    out, _ = kb.zero_cube_pads(data)
    assert hashlib.sha256(out).hexdigest() == expected["sanitized_sha256"]


@pytest.mark.parametrize("label", sorted(_golden()))
def test_golden_geometry_of_synthetic_omc(label):
    """Replay the golden geometry from JSON, so pads/sizes stay covered without any .omc file."""
    expected = _golden()[label]["kernels"]
    data = _synthetic_omc(expected)
    bins = kb.parse_kernel_bins(data)
    assert [[kbin.magic, kbin.blob_off, kbin.size, kbin.elf_end] for kbin in bins] == expected
    expected_pads = [size - (elf_end - blob_off) for _, blob_off, size, elf_end in expected
                     if elf_end is not None]
    assert [kbin.pad[1] - kbin.pad[0] for kbin in bins if kbin.is_cube] == expected_pads
    assert all(any(data[kbin.pad[0]:kbin.pad[1]]) for kbin in bins if kbin.is_cube)
    out, stats = kb.zero_cube_pads(data)
    assert len(out) == len(data)
    assert stats.cube_kernels == len(expected_pads) and stats.pad_bytes == sum(expected_pads)
    assert stats.cleared_bytes == stats.pad_bytes
    for kbin in bins:
        body_end = kbin.pad[0] if kbin.is_cube else kbin.blob_end
        assert out[kbin.blob_off:body_end] == data[kbin.blob_off:body_end]
        if kbin.is_cube:
            assert out[kbin.pad[0]:kbin.pad[1]] == bytes(kbin.pad[1] - kbin.pad[0])
