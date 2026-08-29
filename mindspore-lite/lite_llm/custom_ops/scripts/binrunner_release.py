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
"""Launch the installed BinRunner release with local-device compatibility.

The GitHub release wheel provides the complete host CLI.  Its bundled HAP is
signed for the release device list, so local development uses an independently
signed HAP and selects that bundle through ``BINAPP_BUNDLE``.  Windows hdc.exe
also returns CRLF to WSL; normalize it before the release parser compares the
``<<< END`` protocol marker.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile


def _bootstrap_hdc() -> None:
    """Expose a configured Windows ``hdc.exe`` as ``hdc`` under WSL."""
    if shutil.which("hdc"):
        return

    configured = os.environ.get("HDC_PATH", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(
            "/mnt/c/Program Files/Huawei/DevEco Studio/sdk/default/"
            "openharmony/toolchains/hdc.exe"
        ),
    ]
    hdc_executable = next(
        (candidate for candidate in candidates if candidate and candidate.is_file()),
        None,
    )
    if hdc_executable is None:
        return

    shim_dir = Path(tempfile.gettempdir()) / "hmos_binrunner_hdc"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "hdc"
    if shim.is_symlink() and shim.resolve() != hdc_executable.resolve():
        shim.unlink()
    if not shim.exists():
        shim.symlink_to(hdc_executable)
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}"


_bootstrap_hdc()

# BinRunner 1.1.1 moved configuration and hilog parsing out of ``__main__``.
# Patch those leaf modules before importing the CLI so modules using
# ``from ... import`` bind the configured values rather than the defaults.
# pylint: disable=wrong-import-position  # imports intentionally follow _bootstrap_hdc()
from binrunner import config as release_config  # noqa: E402
from binrunner import hilog as release_hilog  # noqa: E402


release_config.BUNDLE = os.environ.get("BINAPP_BUNDLE", release_config.BUNDLE)

_release_parse_output = release_hilog.parse_output


def _parse_output(output, started, report_lines, parts, run_id=""):
    normalized = output.replace("\r\n", "\n").replace("\r", "\n")
    return _release_parse_output(
        normalized, started, report_lines, parts, run_id
    )


release_hilog.parse_output = _parse_output

from binrunner import __main__ as release_cli  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(release_cli.main())
