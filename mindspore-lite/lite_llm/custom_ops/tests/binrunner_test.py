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
"""BinRunner transport extension for operator device unit tests."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid

from base_test import TestCaseBasic


class BinRunnerTestCaseBasic(TestCaseBasic):
    """Run the same UT contract inside the BinRunner application sandbox."""

    binapp_files_bin = "data/storage/el2/base/haps/entry/files/bin"
    # The BinRunner RPC reports success for payloads above its effective
    # single-file limit even though the file is not committed to the app
    # sandbox.  Keep the normal, faster RPC path for ordinary operator assets
    # and use HDC's bundle-aware file transport for large model inputs.
    binrunner_push_max_bytes = 64 * 1024 * 1024
    transport = "binapp"

    @classmethod
    def run_binrunner(
        cls,
        *arguments: str,
        check: bool = True,
        capture_output: bool = False,
    ):
        return cls._run(
            [
                *cls.binrunner_command,
                "-t",
                cls.remote_target,
                "-p",
                str(cls.binrunner_port),
                *map(str, arguments),
            ],
            check=check,
            capture_output=capture_output,
        )

    @classmethod
    def setup_class(cls) -> None:
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
        # TCP 连接设备（REMOTE_TARGET=IP:port）必须用 -t；默认 -s 会让所有 hdc
        # 命令（含 file recv）报 Connect failed，错误被 capture_output 吞掉，
        # 表现为“设备端执行成功但结果文件拉不回”。
        hdc_target_option = os.environ.get("HDC_TARGET_OPTION", "-s")
        cls.hdc_prefix_args = [hdc_path, hdc_target_option, cls.remote_target]
        cls.hdc_prefix = shlex.join(cls.hdc_prefix_args)

        cls.remote_case_name = f"MsLiteUT_{cls.unique_id}_{cls.worker_id}"
        cls.local_dir = tempfile.mkdtemp(prefix=cls.remote_case_name)
        cls.remote_dir = f"@/bin/{cls.remote_case_name}"
        cls.remote_bundle_dir = f"{cls.binapp_files_bin}/{cls.remote_case_name}"

        configured_command = os.environ.get("BINRUNNER_COMMAND", "").strip()
        installed_br = shutil.which("br")
        if configured_command:
            cls.binrunner_command = shlex.split(configured_command)
        elif installed_br:
            cls.binrunner_command = [installed_br]
        else:
            # Fall back to the vendored BinRunner submodule (ADR-0015). The
            # host CLI is a package, so run it as `python -m binrunner` with
            # the submodule root on PYTHONPATH.
            default_root = (
                Path(__file__).resolve().parents[1] / "third_party" / "BinRunner"
            )
            cli_root = Path(
                os.environ.get("BINRUNNER_PATH", str(default_root))
            ).expanduser()
            if not (cli_root / "binrunner" / "__main__.py").is_file():
                raise FileNotFoundError(
                    f"Cannot find installed 'br' or BinRunner submodule at {cli_root}"
                )
            python_path = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = (
                f"{cli_root}{os.pathsep}{python_path}" if python_path else str(cli_root)
            )
            cls.binrunner_command = [
                os.environ.get("BINRUNNER_PYTHON", sys.executable),
                "-m",
                "binrunner",
            ]

        cls.binrunner_port = int(os.environ.get("BINRUNNER_PORT", "8888"))
        cls.binapp_bundle = os.environ.get("BINAPP_BUNDLE", "com.example.binrunner")
        # A previous long-running NPU command may leave the application frozen
        # by memmgr.  Recover it before the first BinRunner RPC so setup does
        # not spend a full command timeout trying to remove a stale case.
        cls._restart_app()
        cls.run_binrunner("rm", cls.remote_case_name, check=False)

        local_runner = Path(cls.model_run_tools_path).expanduser()
        if not local_runner.is_file():
            raise FileNotFoundError(
                f"BinRunner requires a local model runner: {local_runner}"
            )
        cls.run_binrunner(
            "push",
            str(local_runner.resolve()),
            f"{cls.remote_case_name}/{local_runner.name}",
        )
        cls.remote_model_run_tools = cls.remote_path(local_runner.name)

        cls.binapp_dependencies = []
        dependencies = os.environ.get("BINAPP_DEPENDENCIES", "").strip()
        for item in filter(None, dependencies.split(os.pathsep)):
            dependency = Path(item).expanduser()
            if not dependency.is_file():
                raise FileNotFoundError(f"Cannot find BinRunner dependency: {dependency}")
            cls.run_binrunner("push", str(dependency.resolve()), dependency.name)
            cls.binapp_dependencies.append(dependency.name)

    @classmethod
    def teardown_class(cls) -> None:
        local_dir = getattr(cls, "local_dir", "")
        if local_dir and Path(local_dir).is_dir():
            shutil.rmtree(local_dir)
        remote_case_name = getattr(cls, "remote_case_name", "")
        unique_id = getattr(cls, "unique_id", "")
        if remote_case_name.startswith("MsLiteUT_") and unique_id in remote_case_name:
            cls._restart_app()
            cls.run_binrunner("rm", remote_case_name, check=False)

    def upload(self, omc, inputs):
        remote_files = []
        for file_name in [omc, *inputs]:
            local_file = self.local_path(file_name)
            if not local_file.is_file():
                raise FileNotFoundError(local_file)
            push_limit = int(os.environ.get(
                "BINRUNNER_PUSH_MAX_BYTES", self.binrunner_push_max_bytes
            ))
            if local_file.stat().st_size > push_limit:
                self._restart_app()
                self.run_hdc(
                    "shell", "aa", "appdebug", "-c", check=False,
                    capture_output=True,
                )
                self.run_hdc(
                    "shell", "aa", "attach", "-b", self.binapp_bundle,
                    capture_output=True,
                )
                self.run_hdc(
                    "file", "send", "-b", self.binapp_bundle,
                    self.hdc_local_path(local_file),
                    f"{self.remote_bundle_dir}/{local_file.name}",
                    capture_output=True,
                )
            else:
                self.run_binrunner(
                    "push",
                    str(local_file),
                    f"{self.remote_case_name}/{local_file.name}",
                )
            remote_files.append(self.remote_path(local_file.name))
        return remote_files[0], remote_files[1:]

    @classmethod
    def _restart_app(cls) -> None:
        """Wake the device and restart the BinRunner app.

        The device memmgr freezes the app process during low-power state,
        making file recv hang. Restarting the app is the only reliable
        un-freeze.
        """
        for shell_args in (
            ("power-shell", "wakeup"),
            ("aa", "force-stop", cls.binapp_bundle),
            ("aa", "start", "-b", cls.binapp_bundle, "-a", "EntryAbility"),
        ):
            # Windows hdc.exe can emit a terminal status query when attached
            # to pytest's PTY. Piping control-command output avoids waiting
            # forever for a cursor-position reply that is irrelevant here.
            cls.run_hdc(
                "shell", *shell_args, check=False, capture_output=True
            )
        time.sleep(2)

    def download(self, remote_name, local_file):
        destination = self.local_path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = f"{self.remote_bundle_dir}/{Path(remote_name).name}"
        try:
            self._restart_app()
            self.run_hdc(
                "shell", "aa", "appdebug", "-c", check=False,
                capture_output=True
            )
            self.run_hdc(
                "shell", "aa", "attach", "-b", self.binapp_bundle,
                capture_output=True
            )
            self.run_hdc(
                "file",
                "recv",
                "-b",
                self.binapp_bundle,
                source,
                self.hdc_local_path(destination),
                capture_output=True,
            )
        except OSError:
            self._restart_app()
            self.run_hdc(
                "shell", "aa", "appdebug", "-c", check=False,
                capture_output=True
            )
            self.run_hdc(
                "shell", "aa", "attach", "-b", self.binapp_bundle,
                capture_output=True
            )
            self.run_hdc(
                "file",
                "recv",
                "-b",
                self.binapp_bundle,
                source,
                self.hdc_local_path(destination),
                capture_output=True,
            )
        return destination

    def display_perf(self, suffix: str) -> None:
        if self.test_perf and self.data_proc_tool:
            print("BinRunner transport does not expose model-runner profiles", flush=True)
