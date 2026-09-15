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

"""Analyze real translation units while reporting findings in changed files."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
_HEADER_SUFFIXES = {".h", ".hh", ".hpp"}
_DIAGNOSTIC = re.compile(r"^(.+):\d+:\d+: (error|warning|style|performance|portability): ")


def _changed_files(root, file_list):
    paths = {root / line for line in file_list.read_text(encoding="utf-8").splitlines()}
    return {path.resolve() for path in paths
            if path.suffix in _SOURCE_SUFFIXES | _HEADER_SUFFIXES and path.is_file()}


def _project_sources(project):
    entries = json.loads(project.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("compile_commands.json must contain an array")
    return {(project.parent / entry["directory"] / entry["file"]).resolve() for entry in entries}


def _report_diagnostics(stream, root, changed):
    for line in stream:
        match = _DIAGNOSTIC.match(line)
        # Analysis errors must remain visible even in unchanged dependencies.
        if not match or match.group(2) == "error" or (root / match.group(1)).resolve() in changed:
            sys.stderr.write(line)


def main():
    """Return cppcheck's status, keeping execution failures distinct from findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("file_list", type=Path)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    project = args.project.resolve()
    try:
        changed = _changed_files(root, args.file_list)
        sources = _project_sources(project)
        missing = {path for path in changed if path.suffix in _SOURCE_SUFFIXES} - sources
        if missing:
            raise ValueError("compilation database omits changed sources: "
                             + ", ".join(str(path) for path in sorted(missing)))
        command = [
            "cppcheck", "--enable=style", "--inline-suppr", "--error-exitcode=2",
            "--library=googletest", "--relative-paths=" + str(root),
            "--template={file}:{line}:{column}: {severity}: {message} [{id}]",
            "--project=" + str(project),
        ]
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as diagnostics:
            result = subprocess.run(command, cwd=root, stderr=diagnostics, check=False)
            diagnostics.seek(0)
            _report_diagnostics(diagnostics, root, changed)
        return result.returncode
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"cppcheck configuration error: {error}", file=sys.stderr)
        print("Set CPPCHECK_COMPILE_COMMANDS to a compilation database covering changed sources "
              "and consumers of changed headers; generate it with CMAKE_EXPORT_COMPILE_COMMANDS=ON.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
