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
"""fp16 / fp32 binary output comparison with NaN/Inf handling.

Compares a device-produced binary file against a golden, producing a
structured report suitable for both human and JSON consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DTYPE_MAP = {"fp16": np.float16, "fp32": np.float32}
REL_FLOOR = 1e-12  # avoid division by zero in relative error


@dataclass(frozen=True)
class Mismatch:
    index: int
    actual: float
    expected: float
    abs_error: float
    rel_error: float


@dataclass(frozen=True)
class PrecisionReport:
    passed: bool
    element_count: int
    mismatch_count: int
    max_abs_error: float
    max_rel_error: float
    samples: tuple[Mismatch, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "element_count": self.element_count,
            "mismatch_count": self.mismatch_count,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "samples": [
                {
                    "index": s.index,
                    "actual": s.actual,
                    "expected": s.expected,
                    "abs_error": s.abs_error,
                    "rel_error": s.rel_error,
                }
                for s in self.samples
            ],
        }


MAX_SAMPLES = 16


def compare_binary(
    actual_path,
    golden_path,
    dtype: str,
    element_count: int,
    abs_tol: float,
    rel_tol: float,
) -> PrecisionReport:
    """Load two raw binary files, compare element-wise, and return a report."""
    np_dtype = DTYPE_MAP.get(dtype)
    if np_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype!r}")

    itemsize = np.dtype(np_dtype).itemsize
    expected_bytes = element_count * itemsize

    actual_data = _read_binary(actual_path, expected_bytes, "actual")
    golden_data = _read_binary(golden_path, expected_bytes, "golden")

    actual = np.frombuffer(actual_data, dtype=np_dtype)
    golden = np.frombuffer(golden_data, dtype=np_dtype)

    mismatches: list[Mismatch] = []
    max_abs = 0.0
    max_rel = 0.0

    for i in range(element_count):
        a = float(actual[i])
        g = float(golden[i])

        a_nan = np.isnan(actual[i])
        g_nan = np.isnan(golden[i])

        if a_nan and g_nan:
            continue  # both NaN → match regardless of sign

        if a_nan != g_nan:
            mismatches.append(Mismatch(i, a, g, float("nan"), float("nan")))
            continue

        a_inf = np.isinf(actual[i])
        g_inf = np.isinf(golden[i])
        if a_inf or g_inf:
            if a_inf and g_inf and np.sign(a) == np.sign(g):
                continue
            mismatches.append(Mismatch(i, a, g, float("nan"), float("nan")))
            continue

        abs_err = abs(a - g)
        denom = max(abs(g), REL_FLOOR)
        rel_err = abs_err / denom

        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)

        if abs_err <= abs_tol or rel_err <= rel_tol:
            continue

        mismatches.append(Mismatch(i, a, g, abs_err, rel_err))

    return PrecisionReport(
        passed=len(mismatches) == 0,
        element_count=element_count,
        mismatch_count=len(mismatches),
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        samples=tuple(mismatches[:MAX_SAMPLES]),
    )


def _read_binary(path, expected_bytes: int, label: str) -> bytes:
    """Read binary."""
    import os
    actual_size = os.path.getsize(path)
    if actual_size != expected_bytes:
        raise ValueError(
            f"{label} byte count mismatch: expected {expected_bytes}, got {actual_size}"
        )
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected_bytes:
        raise ValueError(f"{label}: short read ({len(data)} != {expected_bytes})")
    return data
