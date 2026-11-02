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
"""Structural parsing of the OMC kernel-bin table and CUBE padding cleanup.

Background
----------
OpenHarmony 6.1+ ``NpuGraphExecutorKernelBinMgmt::CheckISALimitForCustomOP``
scans every registered custom (``'CUBE'``) kernel-bin word by word over
``[blob, blob + size)`` and rejects the model when a word matches the Cube ISA
blacklist rule ``(word & 0xFFC00003) == 0xF0C00001``.  ``size`` in the OMC
kernel descriptor is the formal ELF length rounded up to 256 bytes, so the
alignment padding after the ELF is scanned as if it were instructions.  The
fused OMG path leaves that padding filled with uninitialized high-entropy
data (the single-operator path writes zeros).  The risk is computed from the
padding word count and the mask, not measured on real devices: a typical 0.5B
W4A16 artifact has about 464 padding words (26 CUBE kernels, 1856 B of
alignment padding in total), each word matches the rule with probability
1/4096 under random data, so such an artifact is refused on device with ``hit
insecure code`` with a probability of about 11%.

Neither OMG nor the device can be changed here, so the export zeroes exactly
that padding.  The padding interval is derived from the kernel-bin table, not
guessed from a fixed window: a fixed window both misses the longer paddings
(up to 168 bytes) and overruns the shorter ones into live model data (builtin
non-ELF kernel code, TLV headers, the ``WeightInfoLists`` index).

Layout (reverse engineered from DDK 6.1.1 artifacts; no official spec)::

    N times: [u32 tag=1][u64 rec_len = 24 + len(name) + 1]
             [u32 magic][u32 size][u32 reserved=0][name\\0]
    then:    [u32 tag=2][u32 blob_len = sum(size) + 8]
    then:    N blobs, in record order, each exactly ``size`` bytes
    CUBE blob: [formal ELF, length L][padding of size - L zero-able bytes]

``magic`` is ``'CUBE'`` (0x43554245) for compiled tensor-engine kernels, which are ELF64,
and 0xABCEED50 for the builtin ``executor_*`` bins, which are not ELF and are
skipped by the device check.  Builtin blobs are never modified.

Supported range
---------------
The layout is reverse engineered from DDK 6.1.1 OMG artifacts and has no
official specification.  Only DDK 6.1.1 with ``platform=kirin9020`` has been
verified against real samples; other DDK versions and platforms are
unverified because no samples are available.  The only caller is
``utils.omc_compiler.compile_omc``, which sanitizes OMG output for a graph
whose ``platform`` defaults to ``kirin9020``, so non-Kirin export paths are
unaffected.

Because the layout is reverse engineered it may change with a DDK upgrade, so
every structural invariant is checked and a violation raises
:class:`OmcFormatError` instead of silently falling back to a weaker cleanup
(fail closed).  There is no switch - no environment variable and no option -
that skips or downgrades the cleanup: when the layout assumption does not
hold, the whole export fails.  A blacklisted word inside the ELF code itself
is not a structural violation: the export cannot sanitize kernel code, and
such words were observed in valid kirin9030 ``te_msmatmuls16s4`` kernels (168
words in 12 of 26 CUBE kernels of a Qwen3-4B S16S4 export), so it is logged as
a warning with per-kernel stats and the export continues.  The remedy for a
structural violation is to upgrade the export tooling, or to contact the
lite_llm export maintainers to confirm the new DDK layout.

Memory and time: the cleanup reads the whole ``.omc`` into memory and, while
processing, holds both the original ``bytes`` and a ``bytearray`` copy, so the
peak is about 3x the file size.  Weights travel as external data, so a typical
``.omc`` is a few MB: a 3.5 MB artifact peaks at about 10 MB and takes about
0.04 s.
"""

import logging
import os
import struct
from typing import List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# Kernel descriptor record / blob section tags.
_TAG_RECORD = 1
_TAG_BLOB = 2
# Magic of a compiled tensor-engine kernel ("CUBE", little endian in the file) and of the
# builtin executor bins, which are not ELF and not scanned by the device.
CUBE_MAGIC = 0x43554245
BUILTIN_MAGIC = 0xABCEED50
_KNOWN_MAGICS = (CUBE_MAGIC, BUILTIN_MAGIC)
# Fixed part of a record: tag(4) + rec_len(8) + magic(4) + size(4) + reserved(4).
_REC_FIXED = 24
# Kernel blobs are padded to this alignment by OMG.
_BIN_ALIGN = 256
# Device-side blacklist rule of CheckISALimitForCustomOP.
ISA_MASK = 0xFFC00003
ISA_PATTERN = 0xF0C00001
# One-line triage hint appended to every OmcFormatError so the message says what
# to do next, not only what failed.
_DIAG_HINT = (
    "the OMC layout is only verified for DDK 6.1.1 / kirin9020, so dump the OMC around this "
    "offset and contact the lite_llm export maintainers if the DDK was upgraded"
)

_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_ELF64_EHSIZE = 64


class OmcFormatError(RuntimeError):
    """The OMC does not match the known kernel-bin layout.

    Raised instead of degrading to a partial cleanup: an unknown layout means
    the padding interval cannot be trusted, and writing into the wrong bytes
    corrupts the model (observed as sig 6 in ``ParseWeightsInfo`` on device).
    """


class KernelBin(NamedTuple):
    """One kernel-bin: its descriptor fields and its blob interval."""

    index: int
    name: str
    magic: int
    blob_off: int
    size: int
    elf_end: Optional[int]  # formal ELF end for CUBE bins, None for builtins

    @property
    def blob_end(self) -> int:
        """End offset (exclusive) of the blob, i.e. the end of the device scan."""
        return self.blob_off + self.size

    @property
    def is_cube(self) -> bool:
        """Whether the device scans this bin (``'CUBE'`` magic)."""
        return self.magic == CUBE_MAGIC

    @property
    def pad(self) -> Optional[Tuple[int, int]]:
        """``(start, end)`` of the alignment padding, or None for builtins."""
        if self.elf_end is None:
            return None
        return (self.elf_end, self.blob_end)


def _align_up(value: int, alignment: int = _BIN_ALIGN) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def elf_formal_end(buf: bytes, off: int) -> Optional[int]:
    """Absolute end of the ELF64 at ``off`` (``e_shoff + e_shentsize*e_shnum``).

    OMG packs kernel ``.o`` files with the section header table last, so this
    is the formal file length.  Returns None when ``off`` does not hold a
    little-endian ELF64 header whose tables stay inside ``buf``.
    """
    if off < 0 or off + _ELF64_EHSIZE > len(buf) or buf[off:off + 4] != _ELF_MAGIC:
        return None
    if buf[off + 4] != _ELFCLASS64 or buf[off + 5] != _ELFDATA2LSB or buf[off + 6] != 1:
        return None
    e_shoff, = struct.unpack_from("<Q", buf, off + 0x28)
    e_ehsize, = struct.unpack_from("<H", buf, off + 0x34)
    e_shentsize, e_shnum = struct.unpack_from("<HH", buf, off + 0x3A)
    if e_ehsize != _ELF64_EHSIZE or e_shentsize == 0 or e_shnum in (0, 0xFFFF):
        return None
    if e_shoff < _ELF64_EHSIZE:
        return None
    end = off + e_shoff + e_shentsize * e_shnum
    if end <= off + _ELF64_EHSIZE or end > len(buf):
        return None
    return end


def count_elf_magics(buf: bytes) -> int:
    """Number of parseable ELF64 headers in ``buf``, scanning by magic.

    Used as an independent cross-check of the structural parse: an ELF that
    the table does not describe (or the other way round) means the layout
    assumption is wrong.
    """
    found, start = 0, 0
    while True:
        off = buf.find(_ELF_MAGIC, start)
        if off < 0:
            return found
        end = elf_formal_end(buf, off)
        if end is None:
            start = off + 4
        else:
            found += 1
            start = end


def _read_record(buf: bytes, off: int) -> Optional[Tuple[int, int, int, str]]:
    """Parse one ``tag=1`` descriptor at ``off`` -> ``(rec_len, magic, size, name)``."""
    if off < 0 or off + _REC_FIXED > len(buf):
        return None
    tag, rec_len = struct.unpack_from("<IQ", buf, off)
    if tag != _TAG_RECORD or not 0 < rec_len <= len(buf) - off:
        return None
    magic, size, reserved = struct.unpack_from("<III", buf, off + 12)
    if magic not in _KNOWN_MAGICS or reserved != 0:
        return None
    name_end = buf.find(b"\x00", off + _REC_FIXED, off + int(rec_len))
    if name_end < 0 or rec_len != _REC_FIXED + (name_end - (off + _REC_FIXED)) + 1:
        return None
    name = buf[off + _REC_FIXED:name_end].decode("ascii", "replace")
    return (int(rec_len), magic, int(size), name)


def _table_start(buf: bytes, magic_off: int) -> int:
    """Walk back over contiguous descriptors to the first record of the table."""
    start = magic_off - 12
    while True:
        previous = None
        for rec_len in range(_REC_FIXED, _REC_FIXED + 256):
            candidate = start - rec_len
            if candidate < 0:
                break
            parsed = _read_record(buf, candidate)
            if parsed is not None and parsed[0] == rec_len:
                previous = candidate
                break
        if previous is None:
            return start
        start = previous


def _check_cube_blob(buf: bytes, blob_off: int, size: int, name: str) -> int:
    """Validate a CUBE blob (ELF64 with ``size == align256(ELF length)``); return the ELF end."""
    elf_end = elf_formal_end(buf, blob_off)
    if elf_end is None:
        raise OmcFormatError(
            f"kernel {name!r}: CUBE blob at {blob_off} ({hex(blob_off)}) is not a parseable ELF64 "
            f"(ELF magic, header size or section-header table is invalid); {_DIAG_HINT}"
        )
    formal_len = elf_end - blob_off
    if elf_end > blob_off + size:
        raise OmcFormatError(
            f"kernel {name!r}: formal ELF length {formal_len} ending at {elf_end} ({hex(elf_end)}) "
            f"exceeds record size {size} of the blob at {blob_off} ({hex(blob_off)}); {_DIAG_HINT}"
        )
    if size != _align_up(formal_len):
        raise OmcFormatError(
            f"kernel {name!r}: record size {size} at blob {blob_off} ({hex(blob_off)}) != "
            f"align256(formal ELF length {formal_len}) = {_align_up(formal_len)}; {_DIAG_HINT}"
        )
    if formal_len % 4 != 0:
        raise OmcFormatError(
            f"kernel {name!r}: formal ELF length {formal_len} at blob {blob_off} ({hex(blob_off)}) "
            f"is not 4-byte word aligned, so the device scan origin cannot be reproduced; "
            f"{_DIAG_HINT}"
        )
    if name.encode("ascii", "replace") not in buf[blob_off:elf_end]:
        raise OmcFormatError(
            f"kernel {name!r}: descriptor name is absent from its ELF "
            f"[{blob_off} ({hex(blob_off)}), {elf_end} ({hex(elf_end)})) "
            f"so the descriptor-to-blob mapping is wrong; {_DIAG_HINT}"
        )
    return elf_end


def _read_records(buf: bytes, start: int) -> Tuple[List[Tuple[int, int, int, str]], int]:
    """Parse the contiguous ``tag=1`` descriptors starting at ``start``.

    Returns ``(records, off)`` where ``off`` is the offset just past the last
    descriptor, i.e. the offset where the ``tag=2`` blob section must start.
    """
    records, off = [], start
    while True:
        parsed = _read_record(buf, off)
        if parsed is None:
            break
        records.append(parsed)
        off += parsed[0]
    return records, off


def _not_a_table_reason(buf: bytes, start: int) -> Optional[str]:
    """Why ``start`` does not look like a kernel-bin table, or None when it does.

    Only the weak structural shape is checked here - one or more ``tag=1``
    descriptors followed by a ``tag=2`` blob section whose blobs stay inside
    ``buf`` - so that :func:`_parse_table` still reports field-level corruption
    of a real table (blob_len, size alignment, ELF body) as corruption, while a
    ``'CUBE'`` magic that merely happens to start something table-shaped is
    reported as a probable stray hit inside weight/graph data.
    """
    records, off = _read_records(buf, start)
    count = len(records)
    total = sum(rec[2] for rec in records)
    if count == 0:
        return "no tag=1 descriptor at the back-tracked start"
    if off + 8 > len(buf):
        return (f"the {count} descriptor(s) end at {off} ({hex(off)}), leaving no room for the "
                f"8-byte blob header")
    tag, = struct.unpack_from("<I", buf, off)
    if tag != _TAG_BLOB:
        return (f"tag={_TAG_BLOB} is required after the {count} descriptor(s) ending at {off} "
                f"({hex(off)}), found tag {tag}")
    blob_off = off + 8
    if blob_off + total > len(buf):
        return (f"the blobs starting at {blob_off} ({hex(blob_off)}) need {total} byte(s), i.e. up "
                f"to {blob_off + total}, past the file end {len(buf)}")
    return None


def _parse_table(buf: bytes, start: int, first_index: int) -> Tuple[List[KernelBin], int]:
    """Parse one kernel-bin table at ``start`` -> ``(kernel bins, end offset)``."""
    records, off = _read_records(buf, start)
    if not records:
        raise OmcFormatError(
            f"kernel-bin table at {start} ({hex(start)}) holds no kernel descriptor: the tag="
            f"{_TAG_RECORD} record invariant fails at this offset; {_DIAG_HINT}"
        )
    if off + 8 > len(buf):
        raise OmcFormatError(
            f"kernel-bin table at {start} ({hex(start)}) is truncated before its blob tag: "
            f"{len(buf) - off} byte(s) left at {off} ({hex(off)}), fewer than the 8-byte tag="
            f"{_TAG_BLOB} blob header; {_DIAG_HINT}"
        )
    tag, blob_len = struct.unpack_from("<II", buf, off)
    if tag != _TAG_BLOB:
        raise OmcFormatError(
            f"kernel-bin table at {start} ({hex(start)}) is corrupt: expected tag={_TAG_BLOB} "
            f"after its {len(records)} descriptor(s) at {off} ({hex(off)}), found tag {tag}; "
            f"{_DIAG_HINT}"
        )
    total = sum(rec[2] for rec in records)
    if blob_len != total + 8:
        raise OmcFormatError(
            f"kernel-bin table at {start} ({hex(start)}) is corrupt: the blob header at {off} "
            f"({hex(off)}) declares blob_len {blob_len} but its {len(records)} descriptor(s) sum "
            f"to {total} (+8 header = {total + 8}); {_DIAG_HINT}"
        )
    blob_off = off + 8
    if blob_off + total > len(buf):
        raise OmcFormatError(
            f"kernel-bin table at {start} ({hex(start)}) is corrupt: its blobs start at {blob_off} "
            f"({hex(blob_off)}) and need {total} byte(s), i.e. up to {blob_off + total}, past the "
            f"file end {len(buf)}; {_DIAG_HINT}"
        )

    bins: List[KernelBin] = []
    cursor = blob_off
    for _, magic, size, name in records:
        if size <= 0 or size % _BIN_ALIGN != 0:
            raise OmcFormatError(
                f"kernel {name!r}: size {size} at blob {cursor} ({hex(cursor)}) is not a positive "
                f"multiple of {_BIN_ALIGN}, so its padding interval is undefined; {_DIAG_HINT}"
            )
        elf_end = _check_cube_blob(buf, cursor, size, name) if magic == CUBE_MAGIC else None
        bins.append(KernelBin(first_index + len(bins), name, magic, cursor, size, elf_end))
        cursor += size
    return bins, blob_off + total


def _skipped_note(skipped: List[str]) -> str:
    """Render the candidates that were skipped as probable stray ``'CUBE'`` magics."""
    if not skipped:
        return ""
    return (
        f"; {len(skipped)} 'CUBE' magic(s) were skipped as probable stray hits in weight/graph "
        f"data, which is where a corrupt real table would also end up: {'; '.join(skipped[:4])}"
    )


def parse_kernel_bins(buf: bytes, skipped_out: Optional[List[str]] = None) -> List[KernelBin]:
    """Parse every kernel-bin descriptor table in ``buf``.

    A ``'CUBE'`` magic is only accepted as a table location after the
    back-tracked start parses as a descriptor sequence followed by a ``tag=2``
    blob section that stays inside ``buf``.  A magic that fails that shape
    test is a probable stray ``'CUBE'`` byte sequence in weight/graph data and
    is skipped rather than parsed, so live data is never mistaken for a table;
    skipping stays fail closed because a real table that was skipped this way
    makes the CUBE/ELF cross-check below fail, and the skipped candidates are
    then listed in the error.

    Raises :class:`OmcFormatError` when any layout invariant fails, including
    the cross-check that the number of CUBE bins equals the number of ELF64
    headers found by a plain magic scan.

    When ``skipped_out`` is given, the reason for every skipped stray
    ``'CUBE'`` magic candidate is appended to it, so the successful path can
    report what was ignored (the default None keeps the call behaviour
    unchanged).
    """
    bins: List[KernelBin] = []
    skipped: List[str] = [] if skipped_out is None else skipped_out
    cube_magic_bytes = struct.pack("<I", CUBE_MAGIC)
    search, parsed_until = 0, -1
    while True:
        hit = buf.find(cube_magic_bytes, search)
        if hit < 0:
            break
        if hit <= parsed_until:
            search = hit + 4
            continue
        start = _table_start(buf, hit)
        if _read_record(buf, start) is None:
            search = hit + 4
            continue
        not_a_table = _not_a_table_reason(buf, start)
        if not_a_table is not None:
            skipped.append(
                f"magic at {hit} ({hex(hit)}), candidate table start {start} ({hex(start)}): "
                f"{not_a_table}"
            )
            search = hit + 4
            continue
        table_bins, parsed_until = _parse_table(buf, start, len(bins))
        bins.extend(table_bins)
        search = parsed_until
    if not bins:
        raise OmcFormatError(
            f"no kernel-bin descriptor table found in the {len(buf)}-byte OMC: no 'CUBE' magic is "
            f"followed by a tag={_TAG_RECORD} descriptor sequence and a tag={_TAG_BLOB} blob "
            f"section{_skipped_note(skipped)}; {_DIAG_HINT}"
        )
    cube_count = sum(1 for kbin in bins if kbin.is_cube)
    elf_count = count_elf_magics(buf)
    if cube_count != elf_count:
        raise OmcFormatError(
            f"kernel-bin table describes {cube_count} CUBE bin(s) but the OMC contains "
            f"{elf_count} parseable ELF64 image(s), so a table entry is missing or stray"
            f"{_skipped_note(skipped)}; {_DIAG_HINT}"
        )
    return bins


def scan_isa_hits(buf: bytes, start: int, end: int) -> List[Tuple[int, int]]:
    """Return ``(offset, word)`` for every blacklisted word in ``[start, end)``.

    Mirrors the device scan: unsigned 32-bit little-endian words counted from
    ``start``.  The device counts words from the blob start, so callers pass a
    blob start or an offset a multiple of 4 bytes after it (ELF lengths are
    checked to be word aligned).
    """
    hits = []
    for off in range(start, end - 3, 4):
        word, = struct.unpack_from("<I", buf, off)
        if (word & ISA_MASK) == ISA_PATTERN:
            hits.append((off, word))
    return hits


class PadCleanupStats(NamedTuple):
    """What :func:`zero_cube_pads` parsed and changed."""

    kernels: int
    cube_kernels: int
    pad_bytes: int
    cleared_bytes: int
    hits_before: Tuple[str, ...]
    skipped_candidates: int = 0
    body_hit_kernels: int = 0
    body_hits: int = 0


def _warn_body_hits(data: bytes, pads: List[Tuple[KernelBin, Tuple[int, int]]]) -> Tuple[int, int]:
    """Warn about blacklisted words inside CUBE ELF bodies; return ``(kernels, hits)``.

    A hit inside ``.text`` is not fixable on the export side, so it must not
    fail the export; one warning per affected kernel keeps it actionable.
    """
    kernels, total = 0, 0
    for kbin, pad in pads:
        hits = scan_isa_hits(data, kbin.blob_off, pad[0])
        if not hits:
            continue
        kernels += 1
        total += len(hits)
        first, word = hits[0]
        logger.warning(
            "kernel %r: %d word(s) inside kernel ELF code match the device ISA blacklist "
            "(rule (w & 0x%08x) == 0x%08x); first hit at %d (%s), +%d in blob, word 0x%08x. "
            "The export continues and the padding is still zeroed, but the export cannot "
            "change kernel code, so the device CheckISA scan may still reject the model; "
            "report it to the operator/compiler owners",
            kbin.name, len(hits), ISA_MASK, ISA_PATTERN, first, hex(first),
            first - kbin.blob_off, word,
        )
    return kernels, total


def zero_cube_pads(data: bytes) -> Tuple[bytes, PadCleanupStats]:
    """Zero the alignment padding of every CUBE kernel-bin in an OMC blob.

    Only ``[formal ELF end, blob start + size)`` of CUBE bins is written;
    builtin bins, ELF bodies and everything outside the blob section keep
    their bytes.  Blacklisted words inside an ELF body are only logged as a
    per-kernel warning (the export cannot change kernel code) and counted in
    the stats, so the export continues.  The result is verified before it is
    returned: the byte diff must stay inside the padding intervals, and the
    padding of every CUBE bin is re-scanned and must no longer match the
    device blacklist.

    Raises :class:`OmcFormatError` on any structural layout violation and on a
    blacklisted word that survives the padding cleanup.
    """
    skipped: List[str] = []
    bins = parse_kernel_bins(data, skipped)
    pads = [(kbin, kbin.pad) for kbin in bins if kbin.is_cube]

    body_hit_kernels, body_hits = _warn_body_hits(data, pads)

    hits_before = []
    out = bytearray(data)
    cleared = 0
    for kbin, pad in pads:
        for off, word in scan_isa_hits(data, pad[0], pad[1]):
            hits_before.append(f"{kbin.name}+pad{off - pad[0]}:0x{word:08x}")
        cleared += sum(1 for byte in data[pad[0]:pad[1]] if byte)
        out[pad[0]:pad[1]] = bytes(pad[1] - pad[0])

    result = bytes(out)
    _assert_diff_within_pads(data, result, [pad for _, pad in pads])
    _assert_pads_clean(result, pads)
    stats = PadCleanupStats(
        kernels=len(bins),
        cube_kernels=len(pads),
        pad_bytes=sum(pad[1] - pad[0] for _, pad in pads),
        cleared_bytes=cleared,
        hits_before=tuple(hits_before),
        skipped_candidates=len(skipped),
        body_hit_kernels=body_hit_kernels,
        body_hits=body_hits,
    )
    return result, stats


def _assert_pads_clean(result: bytes, pads: List[Tuple[KernelBin, Tuple[int, int]]]) -> None:
    """Fail closed when a blacklisted word survives in a CUBE padding interval.

    Only the padding is re-scanned: hits inside the ELF body are reported as a
    warning by :func:`_warn_body_hits` and are not the export's to fix.
    """
    for kbin, pad in pads:
        remaining = scan_isa_hits(result, pad[0], pad[1])
        if remaining:
            raise OmcFormatError(
                f"kernel {kbin.name!r} still matches the device ISA blacklist in its padding "
                "after padding cleanup at "
                f"{', '.join(f'{off} ({hex(off)}):0x{word:08x}' for off, word in remaining[:8])} "
                f"(padding [{pad[0]} ({hex(pad[0])}), {pad[1]} ({hex(pad[1])})), blob at "
                f"{kbin.blob_off} ({hex(kbin.blob_off)}), size {kbin.size}); {_DIAG_HINT}"
            )


def _assert_diff_within_pads(before: bytes, after: bytes, pads: List[Tuple[int, int]]) -> None:
    """Fail closed unless every changed byte lies in one of ``pads``."""
    if len(before) != len(after):
        raise OmcFormatError(
            f"padding cleanup changed the OMC length {len(before)} -> {len(after)}, so a write "
            f"escaped its padding interval; {_DIAG_HINT}"
        )
    cursor = 0
    for start, end in sorted(pads):
        if before[cursor:start] != after[cursor:start]:
            raise OmcFormatError(
                f"padding cleanup changed bytes outside CUBE padding, between {cursor} "
                f"({hex(cursor)}) and {start} ({hex(start)}); {_DIAG_HINT}"
            )
        cursor = end
    if before[cursor:] != after[cursor:]:
        raise OmcFormatError(
            f"padding cleanup changed bytes outside CUBE padding, after offset {cursor} "
            f"({hex(cursor)}) up to the file end {len(before)} ({hex(len(before))}); {_DIAG_HINT}"
        )


def sanitize_omc_kernel_pads(omc_path: str) -> PadCleanupStats:
    """Rewrite ``omc_path`` in place with every CUBE padding zeroed.

    The new content is written to a sibling temporary file and published with
    ``os.replace``, so an interrupted write cannot leave a truncated ``.omc``.
    """
    with open(omc_path, "rb") as handle:
        data = handle.read()
    sanitized, stats = zero_cube_pads(data)
    if sanitized != data:
        tmp_path = omc_path + ".pads.tmp"
        try:
            with open(tmp_path, "wb") as handle:
                handle.write(sanitized)
            os.replace(tmp_path, omc_path)
        except OSError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    logger.info(
        "Zeroed CUBE kernel padding in %s: %d kernel(s), %d CUBE, %d padding byte(s), "
        "%d non-zero byte(s) cleared, %d stray 'CUBE' magic(s) skipped, "
        "%d CUBE kernel(s) with %d blacklist hit(s) inside ELF code (warned, not fixable), "
        "blacklist hit(s) before cleanup: %s",
        omc_path, stats.kernels, stats.cube_kernels, stats.pad_bytes, stats.cleared_bytes,
        stats.skipped_candidates, stats.body_hit_kernels, stats.body_hits,
        ", ".join(stats.hits_before) or "none",
    )
    return stats
