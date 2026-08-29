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
"""Common HDC lifecycle for operator device unit tests."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence
import uuid


def build_omg_environment(platform: str) -> dict[str, str]:
    """Prefer an installed custom OPP over same-named DDK implementations."""
    env = os.environ.copy()
    ddk_path = os.environ.get("DDK_PATH", "").strip()
    custom_roots = [
        Path(root)
        for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if root
    ]
    custom_libraries = [
        str(root / f"tools/platform/{platform}/lib64")
        for root in custom_roots
    ]
    custom_impls = [
        str(root / f"tools/platform/{platform}/ops/impl")
        for root in custom_roots
    ]

    library_paths = [*custom_libraries, env.get("LD_LIBRARY_PATH", "")]
    python_paths = [*custom_impls, env.get("PYTHONPATH", "")]
    if ddk_path:
        library_paths.append(f"{ddk_path}/tools/platform/{platform}/lib64")
        python_paths.append(f"{ddk_path}/tools/platform/{platform}/ops/impl")

    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, library_paths))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, python_paths))
    env["SOC_VERSION"] = platform
    return env


class TestCaseBasic:
    """Create an isolated host/device workspace for one pytest class."""

    required_env = ("REMOTE_TARGET", "MODEL_RUN_TOOLS_PATH")
    remote_root = "/data/local/tmp"
    transport = "hdc"

    @staticmethod
    def _env_flag(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def _run(
        cls,
        command: Sequence[str],
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run."""
        command = [str(part) for part in command]
        print(f"$ {shlex.join(command)}", flush=True)
        return subprocess.run(
            command,
            check=check,
            text=True,
            capture_output=capture_output,
        )

    @classmethod
    def run_hdc(
        cls,
        *arguments: str,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return cls._run(
            [*cls.hdc_prefix_args, *map(str, arguments)],
            check=check,
            capture_output=capture_output,
        )

    @classmethod
    def hdc_local_path(cls, path: str | os.PathLike[str]) -> str:
        """Translate a WSL path when the selected HDC is a Windows executable."""
        local_path = str(path)
        hdc_path = str(cls.hdc_prefix_args[0]).lower()
        wslpath = shutil.which("wslpath")
        if hdc_path.endswith(".exe") and wslpath:
            result = subprocess.run(
                [wslpath, "-w", local_path],
                check=True,
                text=True,
                capture_output=True,
            )
            return result.stdout.strip()
        return local_path

    @classmethod
    def setup_class(cls) -> None:
        """setup_class: helper."""
        cls.unique_id = uuid.uuid4().hex
        if uuid.UUID(cls.unique_id).hex != cls.unique_id:
            raise RuntimeError(f"Invalid generated UUID: {cls.unique_id!r}")

        missing = [name for name in cls.required_env if not os.environ.get(name)]
        if missing:
            raise EnvironmentError(
                "Missing required device-test environment variable(s): "
                + ", ".join(missing)
            )

        cls.remote_target = os.environ["REMOTE_TARGET"]
        cls.model_run_tools_path = os.environ["MODEL_RUN_TOOLS_PATH"]
        cls.test_perf = cls._env_flag(os.environ.get("TEST_PERF"))
        cls.data_proc_tool = os.environ.get("DATA_PROC_TOOL", "").strip()

        cls.worker_id = os.environ.get("PYTEST_XDIST_WORKER", "master")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", cls.worker_id):
            raise ValueError(f"Unsafe PYTEST_XDIST_WORKER: {cls.worker_id!r}")

        hdc_path = os.environ.get("HDC_PATH", "hdc")
        hdc_target_option = os.environ.get("HDC_TARGET_OPTION", "-s")
        cls.hdc_prefix_args = [hdc_path, hdc_target_option, cls.remote_target]
        cls.hdc_prefix = shlex.join(cls.hdc_prefix_args)

        prefix = f"MsLiteUT_{cls.unique_id}_{cls.worker_id}"
        cls.local_dir = tempfile.mkdtemp(prefix=prefix)
        cls.remote_dir = f"{cls.remote_root}/{prefix}"
        cls.run_hdc("shell", "mkdir", "-p", cls.remote_dir)

        # MODEL_RUN_TOOLS_PATH may be a command already installed on the device,
        # or a local executable that should be copied into this test workspace.
        local_runner = Path(cls.model_run_tools_path).expanduser()
        if local_runner.is_file():
            cls.remote_model_run_tools = cls.remote_path(local_runner.name)
            cls.run_hdc(
                "file",
                "send",
                cls.hdc_local_path(local_runner.resolve()),
                cls.remote_model_run_tools,
            )
            cls.run_hdc("shell", "chmod", "700", cls.remote_model_run_tools)
        else:
            cls.remote_model_run_tools = cls.model_run_tools_path

        cls.model_run_ld_library_path = os.environ.get(
            "MODEL_RUN_LD_LIBRARY_PATH",
            str(Path(cls.remote_model_run_tools).parent).replace("\\", "/"),
        )

    @classmethod
    def teardown_class(cls) -> None:
        """teardown_class: helper."""
        local_dir = getattr(cls, "local_dir", "")
        if local_dir and Path(local_dir).is_dir():
            shutil.rmtree(local_dir)

        remote_dir = getattr(cls, "remote_dir", "")
        unique_id = getattr(cls, "unique_id", "")
        safe_prefix = f"{cls.remote_root}/MsLiteUT_"
        if remote_dir.startswith(safe_prefix) and unique_id in remote_dir:
            cls.run_hdc("shell", "rm", "-rf", remote_dir, check=False)

    @classmethod
    def local_path(cls, file_name: str | os.PathLike[str]) -> Path:
        path = Path(file_name)
        return path if path.is_absolute() else Path(cls.local_dir) / path

    @classmethod
    def remote_path(cls, file_name: str | os.PathLike[str]) -> str:
        return f"{cls.remote_dir}/{Path(file_name).name}"

    def upload(
        self,
        omc: str | os.PathLike[str],
        inputs: Iterable[str | os.PathLike[str]],
    ) -> tuple[str, list[str]]:
        """Upload one OMC followed by each input file."""
        remote_files: list[str] = []
        for file_name in [omc, *inputs]:
            local_file = self.local_path(file_name)
            if not local_file.is_file():
                raise FileNotFoundError(local_file)
            remote_file = self.remote_path(local_file.name)
            self.run_hdc(
                "file", "send", self.hdc_local_path(local_file), remote_file
            )
            remote_files.append(remote_file)
        return remote_files[0], remote_files[1:]

    def download(
        self,
        remote_name: str | os.PathLike[str],
        local_file: str | os.PathLike[str],
    ) -> Path:
        """download: helper."""
        destination = self.local_path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.run_hdc(
            "file",
            "recv",
            self.remote_path(remote_name),
            self.hdc_local_path(destination),
        )
        return destination

    def display_perf(self, suffix: str) -> None:
        """Collect ``profile_<suffix>`` and print its ``*_op.csv`` files."""
        if not (self.test_perf and self.data_proc_tool):
            return
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", suffix):
            raise ValueError(f"Unsafe profile suffix: {suffix!r}")

        remote_profile = f"{self.remote_dir}/profile_{suffix}"
        local_profile = Path(self.local_dir) / f"profile_{suffix}"
        self.run_hdc("shell", "rm", "-rf", remote_profile)
        self.run_hdc("shell", "mkdir", "-p", remote_profile)

        if "{profile}" in self.data_proc_tool:
            profile_command = shlex.split(
                self.data_proc_tool.format(profile=remote_profile)
            )
        else:
            profile_command = [
                *shlex.split(self.data_proc_tool),
                f"--result_path={remote_profile}",
            ]
        self.run_hdc("shell", *profile_command)

        if local_profile.exists():
            shutil.rmtree(local_profile)
        self.run_hdc(
            "file", "recv", remote_profile, self.hdc_local_path(local_profile)
        )

        csv_files = sorted(local_profile.rglob("*_op.csv"))
        if not csv_files:
            print(f"No *_op.csv found under {local_profile}", flush=True)
            return
        self._run(["cat", *map(str, csv_files)])
