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
"""Collect split AscendC kernels and merge their binary build configs.

An operator config may live either beside ``op_kernel/`` or inside it.  The
supported input schema is::

    {
      "shell": {"file": "ms_new_op.cpp"},
      "implement": [
        {"platform": "kirin9020", "files": ["ms_new_op_impl.cpp"]}
      ]
    }

The merged schema deliberately represents ``shell`` as a list.  JSON objects
cannot contain the repeated ``file`` keys shown in the historical design note.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_PLATFORMS = {"kirin9020", "kirin9030", "kirinx90"}
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".o"}
OP_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """Raised when a binary operator config is invalid."""


def _operator_context(config_path: Path) -> tuple[str, Path]:
    """_operator_context: helper."""
    config_path = config_path.resolve()
    if config_path.parent.name == "op_kernel":
        op_name = config_path.parent.parent.name
        source_dir = config_path.parent
    else:
        op_name = config_path.parent.name
        candidate = config_path.parent / "op_kernel"
        source_dir = candidate if candidate.is_dir() else config_path.parent
    if not OP_NAME_RE.fullmatch(op_name):
        raise ConfigError(f"invalid operator name derived from {config_path}: {op_name!r}")
    return op_name, source_dir.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"config root must be an object: {path}")
    return value


def _relative_source(source_dir: Path, value: object, field: str) -> tuple[Path, Path]:
    """_relative_source: helper."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"{field} must stay inside the operator kernel directory: {value}")
    source = (source_dir / relative).resolve()
    try:
        source.relative_to(source_dir)
    except ValueError as exc:
        raise ConfigError(f"{field} escapes the operator kernel directory: {value}") from exc
    if not source.is_file():
        raise ConfigError(f"{field} does not exist: {source}")
    return relative, source


def _copy_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            return
        raise ConfigError(f"different files map to the same destination: {destination}")
    shutil.copy2(source, destination)


def _shell_files(config: dict[str, Any], config_path: Path) -> list[str]:
    """_shell_files: helper."""
    shell = config.get("shell")
    if not isinstance(shell, dict):
        raise ConfigError(f"shell must be an object in {config_path}")
    values: list[object] = []
    if "file" in shell:
        values.append(shell["file"])
    if "files" in shell:
        files = shell["files"]
        if not isinstance(files, list):
            raise ConfigError(f"shell.files must be a list in {config_path}")
        values.extend(files)
    if not values:
        raise ConfigError(f"shell.file is required in {config_path}")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ConfigError(f"shell file names must be non-empty strings in {config_path}")
        if value not in result:
            result.append(value)
    return result


def collect_configs(
    config_paths: Sequence[Path], shell_dir: Path, impl_dir: Path
) -> dict[str, Any]:
    """Copy configured sources and return one deterministic merged config."""

    shell_dir = shell_dir.resolve()
    impl_dir = impl_dir.resolve()
    shell_records: list[dict[str, str]] = []
    implementation_records: list[dict[str, Any]] = []

    for config_path in sorted({path.resolve() for path in config_paths}, key=str):
        config = _load_json(config_path)
        _, source_dir = _operator_context(config_path)
        shell_files = _shell_files(config, config_path)
        for shell_file in shell_files:
            relative, source = _relative_source(
                source_dir, shell_file, f"{config_path}: shell.file"
            )
            if source.suffix.lower() not in SOURCE_SUFFIXES - {".o"}:
                raise ConfigError(f"unsupported shell source type: {source}")
            destination = shell_dir / relative.name
            _copy_checked(source, destination)
            shell_records.append({"file": destination.name})

        implementations = config.get("implement")
        if not isinstance(implementations, list) or not implementations:
            raise ConfigError(f"implement must be a non-empty list in {config_path}")
        default_target = Path(shell_files[0]).with_suffix(".o").name

        for index, implementation in enumerate(implementations):
            if not isinstance(implementation, dict):
                raise ConfigError(f"implement[{index}] must be an object in {config_path}")
            platform = implementation.get("platform")
            if platform not in SUPPORTED_PLATFORMS:
                supported = ", ".join(sorted(SUPPORTED_PLATFORMS))
                raise ConfigError(
                    f"unsupported platform {platform!r} in {config_path}; expected {supported}"
                )
            files = implementation.get("files")
            if not isinstance(files, list) or not files:
                raise ConfigError(f"implement[{index}].files must be a non-empty list")
            target = implementation.get("target", default_target)
            if not isinstance(target, str) or Path(target).name != target or not target.endswith(".o"):
                raise ConfigError(
                    f"implement[{index}].target must be a plain .o file name in {config_path}"
                )
            target_stem = Path(target).stem
            merged_files: list[str] = []
            for file_index, file_name in enumerate(files):
                relative, source = _relative_source(
                    source_dir,
                    file_name,
                    f"{config_path}: implement[{index}].files[{file_index}]",
                )
                if source.suffix.lower() not in SOURCE_SUFFIXES:
                    raise ConfigError(f"unsupported implementation source type: {source}")
                destination = impl_dir / target_stem / relative
                _copy_checked(source, destination)
                merged_files.append(destination.relative_to(impl_dir).as_posix())

            implementation_records.append(
                {
                    "platform": platform,
                    "files": merged_files,
                    "target": target,
                }
            )

    shell_records.sort(key=lambda item: item["file"])
    implementation_records.sort(
        key=lambda item: (item["platform"], item["target"])
    )
    return {
        "schema_version": 1,
        "shell": shell_records,
        "implement": implementation_records,
    }


def write_merged_config(config: dict[str, Any], destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def discover_configs(paths: Iterable[Path]) -> list[Path]:
    """discover_configs: helper."""
    result: list[Path] = []
    for path in paths:
        path = path.resolve()
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(path.rglob("config.json"))
        else:
            raise ConfigError(f"scan path does not exist: {path}")
    return sorted(set(result), key=str)


def _parser() -> argparse.ArgumentParser:
    """_parser: helper."""
    parser = argparse.ArgumentParser(
        description="Copy split kernel sources and merge binary operator configs."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("file", "process explicit config.json files"),
        ("scan", "recursively scan directories for config.json"),
        ("scam", "compatibility alias for the historical 'scam' typo"),
    ):
        command = subparsers.add_parser(mode, help=help_text)
        command.add_argument("paths", nargs="+", type=Path)
        command.add_argument("-s", "--shell-dir", required=True, type=Path)
        command.add_argument("-i", "--impl-dir", required=True, type=Path)
        command.add_argument(
            "--merge-json", "--merge_json", required=True, dest="merge_json", type=Path
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        configs = (
            discover_configs(args.paths)
            if args.mode in {"scan", "scam"}
            else [path.resolve() for path in args.paths]
        )
        if not configs:
            raise ConfigError("no config.json files found")
        merged = collect_configs(configs, args.shell_dir, args.impl_dir)
        write_merged_config(merged, args.merge_json)
    except ConfigError as exc:
        print(f"scripts/copy_kernel.py: error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Collected {len(merged['shell'])} shell source(s) and "
        f"{len(merged['implement'])} implementation build(s) into {args.merge_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
