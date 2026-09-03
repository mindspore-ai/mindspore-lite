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
"""Shared pytest hooks for the lite_llm Python test suite.

``tests/data/golden_v1.msl`` is gitignored and deterministically
regenerable (``gen_golden.py``).  Without provisioning a fresh checkout
fails the three ``test_msl_golden`` tests with ``FileNotFoundError`` —
both here and in the C++ suite.  This hook generates it once per session
when missing; an existing file is left untouched (so an externally
provisioned golden, if any, wins).  Byte-level correctness is still
guarded by the committed ``golden_v1.expected.json`` cross-check.
"""

import os
import subprocess
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, "..", "data")
GEN_GOLDEN = os.path.join(DATA_DIR, "gen_golden.py")
GOLDEN_MSL = os.path.join(DATA_DIR, "golden_v1.msl")


def pytest_sessionstart(session):
    """Generate ``golden_v1.msl`` if missing (deterministic; see gen_golden.py)."""
    del session  # hook signature; not used
    if os.path.exists(GOLDEN_MSL):
        return
    proc = subprocess.run(
        [sys.executable, GEN_GOLDEN],
        check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0 or not os.path.exists(GOLDEN_MSL):
        raise RuntimeError(
            f"auto-generating golden_v1.msl failed (rc={proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    print(f"[conftest] generated missing golden fixture: {GOLDEN_MSL}")
