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
"""Aggregate custom operators into one binary release package."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# pylint: disable=wrong-import-position  # copy_kernel resolves via scripts/
from copy_kernel import ConfigError, collect_configs, write_merged_config


OPS_ROOT = REPO_ROOT / "ascendc_ops"
WORKSPACE_TEMPLATE = REPO_ROOT / "workspace"
BUILD_ROOT = REPO_ROOT / "build" / "ms_ops_pack"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "ms_ops_pack"
OP_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
PLATFORM_ORDER = {"kirin9020": 0, "kirin9030": 1, "kirinx90": 2}


class BuildError(RuntimeError):
    """User-facing packaging failure."""


def copy_checked(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and filecmp.cmp(source, destination, shallow=False):
            return
        raise BuildError(f"source collision at {destination} (from {source})")
    shutil.copy2(source, destination)


def operator_dirs(names: Sequence[str] | None) -> list[Path]:
    """operator_dirs: helper."""
    if names is None:
        operators = sorted(
            path
            for path in OPS_ROOT.iterdir()
            if path.is_dir() and (path / "operator.json").is_file()
        )
        if not operators:
            raise BuildError(f"no operators found under {OPS_ROOT}")
        return operators

    result: list[Path] = []
    errors: list[str] = []
    for name in names:
        if not OP_NAME_RE.fullmatch(name):
            errors.append(f"{name!r}: invalid operator name")
            continue
        candidates = [
            path.resolve()
            for path in (OPS_ROOT / name,)
            if path.is_dir() and (path / "operator.json").is_file()
        ]
        if not candidates:
            errors.append(
                f"{name}: operator not found under ascendc_ops/"
            )
            continue
        if len(candidates) > 1:
            locations = ", ".join(str(path.relative_to(REPO_ROOT)) for path in candidates)
            errors.append(f"{name}: ambiguous operator name found at {locations}")
            continue
        path = candidates[0]
        if path not in result:
            result.append(path)
    if errors:
        raise BuildError("operator selection failed:\n  - " + "\n  - ".join(errors))
    return result


def binary_config(operator: Path) -> Path | None:
    for candidate in (operator / "config.json", operator / "op_kernel" / "config.json"):
        if candidate.is_file():
            return candidate
    return None


def clear(build_root: Path = BUILD_ROOT) -> None:
    """Recreate the isolated aggregate build workspace."""

    build_root = build_root.resolve()
    expected_parent = (REPO_ROOT / "build").resolve()
    if build_root.parent != expected_parent or build_root.name != "ms_ops_pack":
        raise BuildError(f"refusing to clear unexpected build path: {build_root}")
    if build_root.exists():
        shutil.rmtree(build_root)
    shutil.copytree(WORKSPACE_TEMPLATE, build_root)


def reset_aggregate_sources(build_root: Path) -> None:
    """reset_aggregate_sources: helper."""
    for relative in ("op_host", "op_kernel"):
        directory = build_root / relative
        for child in directory.iterdir():
            if child.name != "CMakeLists.txt":
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    framework = build_root / "framework"
    if framework.exists():
        shutil.rmtree(framework)
    (build_root / "op_kernel_impl").mkdir(parents=True, exist_ok=True)


def copy_host(operator: Path, build_root: Path = BUILD_ROOT) -> None:
    """Copy one operator's host and framework sources into the aggregate tree."""

    host_dir = operator / "op_host"
    if not host_dir.is_dir():
        raise BuildError(f"missing op_host directory: {operator.name}")
    for source in sorted(host_dir.iterdir()):
        if source.is_file() and source.name != "CMakeLists.txt":
            copy_checked(source, build_root / "op_host" / source.name)

    framework = operator / "framework"
    if framework.is_dir():
        for source in sorted(path for path in framework.rglob("*") if path.is_file()):
            copy_checked(source, build_root / "framework" / source.relative_to(framework))


def copy_kernel_support(
    operator: Path, build_root: Path, has_binary_config: bool
) -> None:
    """copy_kernel_support: helper."""
    kernel_dir = operator / "op_kernel"
    if not kernel_dir.is_dir():
        raise BuildError(f"missing op_kernel directory: {operator.name}")
    support_suffixes = {".h", ".hpp", ".inc", ".py"}
    for source in sorted(path for path in kernel_dir.rglob("*") if path.is_file()):
        if source.name in {"CMakeLists.txt", "config.json"}:
            continue
        dynamic_suffixes = support_suffixes | {".c", ".cc", ".cpp", ".cxx"}
        if not has_binary_config and source.suffix.lower() in dynamic_suffixes:
            copy_checked(source, build_root / "op_kernel" / source.name)
        elif has_binary_config and source.suffix.lower() in support_suffixes:
            copy_checked(source, build_root / "op_kernel" / source.name)
            copy_checked(
                source,
                build_root / "op_kernel_impl" / operator.name / source.relative_to(kernel_dir),
            )


def process_scan(operators: Sequence[Path], build_root: Path = BUILD_ROOT) -> list[str]:
    """Collect configured binary implementations and legacy dynamic kernels."""

    configs: list[Path] = []
    dynamic_only: list[str] = []
    for operator in operators:
        config = binary_config(operator)
        copy_kernel_support(operator, build_root, config is not None)
        if config is None:
            dynamic_only.append(operator.name)
        else:
            configs.append(config)

    if configs:
        try:
            merged = collect_configs(
                configs, build_root / "op_kernel", build_root / "op_kernel_impl"
            )
        except ConfigError as exc:
            raise BuildError(str(exc)) from exc
        write_merged_config(merged, build_root / "op_kernel_impl" / "config.json")
    return dynamic_only


def stage_sources(operators: Sequence[Path], build_root: Path) -> list[str]:
    """Assemble operator sources in an already-copied workspace tree."""

    reset_aggregate_sources(build_root)
    for operator in operators:
        copy_host(operator, build_root)
    return process_scan(operators, build_root)


def validate_operator_descriptor(operator: Path) -> None:
    """validate_operator_descriptor: helper."""
    descriptor = operator / "operator.json"
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BuildError(f"missing operator.json: {descriptor}") from exc
    except json.JSONDecodeError as exc:
        raise BuildError(f"invalid JSON in {descriptor}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise BuildError(f"operator.json must be a non-empty array: {descriptor}")
    names = {
        record.get("op")
        for record in value
        if isinstance(record, dict) and isinstance(record.get("op"), str)
    }
    if operator.name not in names:
        raise BuildError(
            f"operator.json does not declare {operator.name}: {descriptor}"
        )


def validate_staging(operators: Sequence[Path]) -> None:
    """Validate every operator and aggregate collisions before clearing BUILD_ROOT."""

    errors: list[str] = []
    valid: list[Path] = []
    for operator in operators:
        operator_errors: list[str] = []
        try:
            validate_operator_descriptor(operator)
        except (BuildError, OSError) as exc:
            operator_errors.append(str(exc))
        try:
            if binary_config(operator) is None:
                raise BuildError(
                    "missing op_kernel/config.json (build.config is no longer supported)"
                )
        except (BuildError, OSError) as exc:
            operator_errors.append(str(exc))
        try:
            with tempfile.TemporaryDirectory(prefix="msops-preflight-") as temporary:
                build_root = Path(temporary) / "workspace"
                shutil.copytree(WORKSPACE_TEMPLATE, build_root)
                stage_sources([operator], build_root)
        except (BuildError, ConfigError, OSError, json.JSONDecodeError) as exc:
            operator_errors.append(str(exc))
        if operator_errors:
            errors.append(f"{operator.name}: " + "; ".join(dict.fromkeys(operator_errors)))
        else:
            valid.append(operator)

    if len(valid) > 1:
        try:
            with tempfile.TemporaryDirectory(prefix="msops-preflight-") as temporary:
                build_root = Path(temporary) / "workspace"
                shutil.copytree(WORKSPACE_TEMPLATE, build_root)
                stage_sources(valid, build_root)
        except (BuildError, ConfigError, OSError, json.JSONDecodeError) as exc:
            errors.append(f"aggregate: {exc}")

    if errors:
        raise BuildError("operator preflight failed:\n  - " + "\n  - ".join(errors))


def collect_outputs(build_root: Path, output_dir: Path) -> None:
    """collect_outputs: helper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_package in output_dir.glob("custom_opp_*.run"):
        old_package.unlink()
    for package in sorted((build_root / "build_out").glob("custom_opp_*.run")):
        shutil.copy2(package, output_dir / package.name)

    tools = build_root / "build_out" / "tools"
    staged_tools = sorted(
        (build_root / "build_out" / "_CPack_Packages").glob(
            "*/External/custom_opp_*.run/tools"
        )
    )
    if staged_tools:
        tools = staged_tools[-1]
    destination = output_dir / "tools"
    if destination.exists():
        shutil.rmtree(destination)
    if tools.is_dir():
        shutil.copytree(tools, destination)


def run_package(package: Path, prefix: Path) -> None:
    subprocess.run(
        [str(package), "--", "--prefix", str(prefix.resolve())],
        check=True,
        cwd=package.parent,
    )


def parser() -> argparse.ArgumentParser:
    """parser: helper."""
    result = argparse.ArgumentParser(
        prog="./build.py",
        description=(
            "Aggregate custom operators and build one binary .run release package. "
            "With no operator selection, every operator under ascendc_ops/ is packaged."
        ),
    )
    selection = result.add_mutually_exclusive_group()
    selection.add_argument(
        "--all", action="store_true", help="package every ascendc_ops operator (default)"
    )
    selection.add_argument("--ops", nargs="+", metavar="OP", help="operators to package")
    result.add_argument("--preset", default="default", help="CMake configure preset")
    result.add_argument("--target", default="package", help="CMake build target")
    result.add_argument("--jobs", type=int, default=16, help="parallel build jobs")
    result.add_argument(
        "--install",
        metavar="DDK_PATH",
        type=Path,
        help="run the generated installer into this DDK",
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "operators",
        nargs="*",
        metavar="OP",
        help="shorthand for --ops OP [OP ...]",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.operators and (args.all or args.ops):
        argument_parser.error("positional operators cannot be combined with --all or --ops")
    if args.jobs < 1:
        argument_parser.error("--jobs must be positive")

    selected = args.ops or args.operators or None
    try:
        operators = operator_dirs(selected)
        validate_staging(operators)
        environment = os.environ.copy()
        if not environment.get("DDK_PATH"):
            if args.install is None:
                raise BuildError(
                    "DDK_PATH is not set; source tools/tools_ascendc/set_ascendc_env.sh first"
                )
            environment["DDK_PATH"] = str(args.install.resolve())
        environment["BUILD_JOBS"] = str(args.jobs)

        clear()
        dynamic_only = stage_sources(operators, BUILD_ROOT)
        if dynamic_only:
            raise BuildError(
                "operators missing op_kernel/config.json (build.config is no longer "
                "supported): " + ", ".join(dynamic_only)
            )

        merged_config = BUILD_ROOT / "op_kernel_impl" / "config.json"
        if not merged_config.is_file():
            raise BuildError(f"no merged operator config at {merged_config}")
        config = json.loads(merged_config.read_text(encoding="utf-8"))
        platforms = {record["platform"] for record in config["implement"]}
        if not platforms:
            raise BuildError("no platforms found in op_kernel_impl/config.json")

        requested = os.environ.get("ASCEND_COMPUTE_UNIT", "")
        if requested:
            explicit = {p.strip() for p in requested.split(";") if p.strip()}
            unknown = explicit - platforms
            if unknown:
                raise BuildError(
                    "ASCEND_COMPUTE_UNIT contains platforms with no operators: "
                    + ", ".join(sorted(unknown))
                )
            platforms = explicit
        environment["ASCEND_COMPUTE_UNIT"] = ";".join(
            sorted(platforms, key=PLATFORM_ORDER.__getitem__)
        )

        script = BUILD_ROOT / "build.sh"
        command = [str(script), "--preset", args.preset, "--target", args.target]
        subprocess.run(command, cwd=BUILD_ROOT, env=environment, check=True)
        collect_outputs(BUILD_ROOT, args.output_dir.resolve())

        packages = sorted(args.output_dir.resolve().glob("custom_opp_*.run"))
        if not packages:
            raise BuildError(f"no custom_opp_*.run produced under {BUILD_ROOT / 'build_out'}")
        if args.install is not None:
            run_package(packages[-1], args.install)
        print(f"Package complete: {args.output_dir.resolve()}")
        return 0
    except (BuildError, OSError, subprocess.CalledProcessError) as exc:
        print(f"build.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
