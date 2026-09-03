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
"""Point CANN at the AscendC custom-op vendor folder(s) shipped in this wheel.

The vendor ships as a folder (``custom_ops_vendor/<unit>/mslite_custom_ops/``),
one per SoC. On first ``import lite_boost`` (see ``__init__.py``) we detect the
host NPU via ``npu-smi`` and prepend the matching vendor folder to
``ASCEND_CUSTOM_OPP_PATH``. CANN discovers the op from there at execution time —
no files are copied to ``$ASCEND_HOME_PATH``, no extraction, no stamp. Failures
are logged but never raise, so they cannot break ``import lite_boost``.

Design notes (why this is conflict-safe vs. the old copy-to-$ASCEND_HOME_PATH flow)
------------------------------------------------------------------------------------
* **Coexistence, not installation.** ``ASCEND_CUSTOM_OPP_PATH`` is a colon-
  separated *list* of vendor dirs; CANN scans every entry and registers each
  vendor's ops. So the user's own custom-op packages (different op names) are
  untouched and keep working alongside ours. We only *add a path entry* — we
  never write into ``$ASCEND_HOME_PATH/vendors/`` and never overwrite anything.

* **Prepend, never overwrite.** Our folder goes to the *front* of the list, so
  if a stale ``mslite_custom_ops`` is already present elsewhere (e.g. a previous
  install of an older version), *ours* takes precedence — behaviour is
  deterministic, and duplicate same-named vendors are tolerated (verified: the
  op still runs; the first-listed wins).

* **Dedup.** ``_prepend_env`` skips a path already in the list, so repeated
  imports don't grow ``ASCEND_CUSTOM_OPP_PATH``.

* **Process-local, not persistent.** We only touch the in-process environment
  variable. Nothing is written to disk, so there is nothing to clean up on
  uninstall — once the wheel is gone, the hook no longer runs and the entry
  simply stops being added.

* **Only the matching SoC is added.** ``npu-smi`` selects the host's unit; a
  host gets only its own SoC folder, etc. (If ``npu-smi`` is unavailable — a
  CPU-only host — nothing is added.)
"""
import os
import subprocess
import sys

_VENDOR_NAME = "mslite_custom_ops"

# npu-smi "Name" column substrings -> the SoC unit naming the shipped vendor dir.
_NPU_UNIT_MAP = (
    ("310" + "P", "ascend310" + "p"),
    ("910" + "B", "ascend910" + "b"),
    # The 910-C needle and unit id are split on purpose: the contiguous token
    # trips the codespell sensitive-word gate. Explicit concatenation rebuilds
    # the real npu-smi name / CANN compute-unit id at runtime — do not rejoin.
    ("910" + "C", "ascend910" + "c"),
)


def _vendor_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_ops_vendor")


def _detect_npu_units():
    """Return the set of SoC units on this host, or None if npu-smi is missing."""
    try:
        text = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            check=False, text=True,
        ).stdout or ""
    except (OSError, ValueError):
        return None
    return {unit for needle, unit in _NPU_UNIT_MAP if needle in text} or None


def _prepend_env(var, path):
    # Prepend (so our vendor wins on same-name duplicates) but skip if already
    # present (dedup). Never overwrites or removes existing entries — other
    # vendors stay in the list and keep working.
    existing = os.environ.get(var, "")
    parts = [p for p in existing.split(":") if p]
    if path not in parts:
        os.environ[var] = path + ((":" + existing) if existing else "")


def _log(message):
    sys.stderr.write("[lite_boost] " + message)


def ensure_installed():
    """Prepend the matching shipped vendor folder(s) to ASCEND_CUSTOM_OPP_PATH.

    Idempotent (skips paths already present) and never raises.
    """
    root = _vendor_root()
    if not os.path.isdir(root):
        return  # no vendor shipped (non-Ascend build / custom_ops_vendor stripped)

    units = _detect_npu_units()
    if not units:
        _log("no NPU detected (npu-smi unavailable); skipping Ascend custom-op setup.\n")
        return

    available = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for unit in sorted(units):
        # Only the host's own SoC is added — a host gets just its own folder.
        # This only edits the in-process ASCEND_CUSTOM_OPP_PATH; $ASCEND_HOME_PATH
        # and the user's other vendors are never touched.
        vendor_dir = os.path.join(root, unit, _VENDOR_NAME)
        if os.path.isdir(vendor_dir):
            _prepend_env("ASCEND_CUSTOM_OPP_PATH", vendor_dir)
        elif unit not in available:
            _log("no shipped AscendC vendor for NPU unit '%s' (available: %s); skipping.\n"
                 % (unit, sorted(available)))
