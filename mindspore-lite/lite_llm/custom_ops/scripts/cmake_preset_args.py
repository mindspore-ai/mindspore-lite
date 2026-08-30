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
"""Emit one NUL-delimited CMake cache argument per preset variable."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} CMAKE_PRESETS SOURCE_DIR", file=sys.stderr)
        return 2

    preset_path = Path(sys.argv[1])
    source_dir = str(Path(sys.argv[2]).resolve())
    document = json.loads(preset_path.read_text(encoding="utf-8"))
    presets = document.get("configurePresets", [])
    preset = next((item for item in presets if item.get("name") == "default"), None)
    if preset is None:
        raise ValueError(f"No default configure preset in {preset_path}")

    output = sys.stdout.buffer
    for name, specification in preset.get("cacheVariables", {}).items():
        value = specification.get("value") if isinstance(specification, dict) else specification
        if isinstance(value, bool):
            value = "ON" if value else "OFF"
        value = str(value).replace("${sourceDir}", source_dir)
        output.write(f"-D{name}={value}".encode())
        output.write(b"\0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
