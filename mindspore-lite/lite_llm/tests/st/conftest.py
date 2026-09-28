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
"""ST shared fixtures: release package / model inputs / device declaration.

ST guards the shipped artifact: it always runs against the release package
(``--package``, the build.sh tarball), never the source tree. The exporter wheel
inside the package is installed to a scratch dir and both the export CLI
(``mslite-llm-export``) and ``utils.msl_pack`` are exercised from there. The
operator adapter package ``mslite_llm_ops`` must be installed by CI beforehand.

    # Full chain: package -> install wheel -> export (GGUF -> .msl) -> mslite-chat
    MSLITE_LLM_ST_DEVICE=1 pytest tests/st \
        --package=output/mindspore-lite-llm-linux-x64-0.1.0.tar.gz \
        --model-dir=/path/to/models

    # Reuse an already-built .msl (skips the export stage)
    MSLITE_LLM_ST_DEVICE=1 pytest tests/st \
        --package=output/mindspore-lite-llm-linux-x64-0.1.0.tar.gz --msl=/path/model.msl

Fixtures skip (pytest.skip) when their prerequisite is absent; failures abort
the case. The device must be declared explicitly via ``MSLITE_LLM_ST_DEVICE=1``
so a host-only run never attempts NPU inference.
"""
# pylint: disable=redefined-outer-name,unused-argument  # fixture deps/names


import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from collections import deque

import pytest

# Model parameter table: one entry per guarded model.  The export options are
# consumed by mslite-llm-export (see export/README.md).
MODELS = {
    "qwen2.5-0.5b": {
        "path": "qwen2.5-0.5b-instruct-q4_0.gguf",
        "target": "kirin9020",
        "max_length": 1024,
        "chunk_size": 64,
    },
}

EXPORT_TIMEOUT_SECONDS = 7200
EXPORT_HEARTBEAT_SECONDS = 60
EXPORT_POLL_SECONDS = 1
EXPORT_HEARTBEAT_TAIL_LINES = 5
EXPORT_FAILURE_TAIL_LINES = 200


def _is_export_milestone(line):
    """Keep Python exporter stage logs while suppressing verbose OMG details."""
    return any(f" {level} " in line for level in ("INFO", "WARNING", "ERROR", "CRITICAL"))


def _drain_export_output(reader, pending, tail):
    """Read newly appended exporter output and print only high-level stages."""
    data = reader.read()
    if not data:
        return pending, 0

    text = pending + data.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    pending = ""
    if lines and not lines[-1].endswith(("\n", "\r")):
        pending = lines.pop()

    suppressed = 0
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        tail.append(line)
        if _is_export_milestone(line):
            print(line, flush=True)
        else:
            suppressed += 1
    return pending, suppressed


def _print_export_tail(title, tail, limit):
    """Print a bounded recent-output section for heartbeat or failure diagnosis."""
    recent = list(tail)[-limit:]
    if recent:
        print(f"[ST] {title}:\n" + "\n".join(recent), flush=True)


def _run_export(cmd, env):
    """Run exporter with live milestones, rate-limited OMG output and heartbeat."""
    started = time.monotonic()
    print("[ST] Model export process started; verbose OMG output is rate-limited", flush=True)
    tail = deque(maxlen=EXPORT_FAILURE_TAIL_LINES)
    pending = ""
    suppressed = 0
    next_heartbeat = EXPORT_HEARTBEAT_SECONDS

    with tempfile.TemporaryDirectory(prefix="llm_st_export_log_") as log_dir:
        log_path = os.path.join(log_dir, "export.log")
        with open(log_path, "wb") as output, open(log_path, "rb") as reader:
            with subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env) as process:
                while True:
                    elapsed = time.monotonic() - started
                    remaining = EXPORT_TIMEOUT_SECONDS - elapsed
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        pending, count = _drain_export_output(reader, pending, tail)
                        suppressed += count
                        if pending:
                            tail.append(pending)
                        _print_export_tail("Exporter output before timeout", tail, EXPORT_FAILURE_TAIL_LINES)
                        raise subprocess.TimeoutExpired(cmd, EXPORT_TIMEOUT_SECONDS)

                    try:
                        returncode = process.wait(timeout=min(EXPORT_POLL_SECONDS, remaining))
                        finished = True
                    except subprocess.TimeoutExpired:
                        finished = False

                    pending, count = _drain_export_output(reader, pending, tail)
                    suppressed += count
                    elapsed = time.monotonic() - started
                    if not finished and elapsed >= next_heartbeat:
                        print(
                            f"[ST] Model export still running: {int(elapsed)}s; "
                            f"suppressed {suppressed} verbose line(s) since last heartbeat",
                            flush=True,
                        )
                        _print_export_tail(
                            "Recent exporter output", tail, EXPORT_HEARTBEAT_TAIL_LINES
                        )
                        suppressed = 0
                        next_heartbeat += EXPORT_HEARTBEAT_SECONDS
                    if finished:
                        break

            pending, count = _drain_export_output(reader, pending, tail)
            suppressed += count
            if pending:
                tail.append(pending)
                if _is_export_milestone(pending):
                    print(pending, flush=True)
                else:
                    suppressed += 1

        if returncode != 0:
            _print_export_tail("Export failure output tail", tail, EXPORT_FAILURE_TAIL_LINES)

    elapsed = int(time.monotonic() - started)
    print(
        f"[ST] Model export process exited: code={returncode}, elapsed={elapsed}s; "
        f"suppressed {suppressed} remaining verbose line(s)",
        flush=True,
    )
    return returncode


def pytest_addoption(parser):
    """Register the ST command-line options (package/model directory/msl)."""
    parser.addoption("--package", default=None,
                     help="release package path (.tar.gz or extracted dir); ST runs against it")
    parser.addoption("--model-dir", default=None,
                     help="root directory containing raw inputs registered in conftest.MODELS")
    parser.addoption("--br-port", default="8888",
                     help="local hdc forwarding port used by BinRunner (default: 8888)")
    parser.addoption("--msl", default=None,
                     help="pre-built .msl package; when given the export stage is skipped")


@pytest.fixture(scope="session")
def model_id(request):
    """Model id declared by a test through indirect parametrization."""
    name = getattr(request, "param", None)
    if not name:
        pytest.fail("test must declare a model id through the model_id fixture")
    if name not in MODELS:
        pytest.fail(f"model {name!r} not registered (available: {list(MODELS)})")
    return name


@pytest.fixture(scope="session")
def model_cfg(model_id):
    """Export option table for the selected model id."""
    return MODELS[model_id]


@pytest.fixture(scope="session")
def release(request, tmp_path_factory):
    """The release package under test: extract (or accept) it, locate bin/."""
    path = request.config.getoption("--package")
    if not path:
        pytest.skip("no release package given: pass --package=output/<name>.tar.gz")
    if not os.path.exists(path):
        pytest.fail(f"--package path does not exist: {path}")

    root = path
    if os.path.isfile(path):
        if not tarfile.is_tarfile(path):
            pytest.fail(f"--package is not a tar archive: {path}")
        extract_root = os.path.join(tmp_path_factory.mktemp("st_release"), "pkg")
        with tarfile.open(path) as tf:
            tf.extractall(extract_root, filter="data")
        # The release archive wraps its contents in a single top-level package
        # directory; descend into it so bin/ and tool/ resolve as before.
        entries = os.listdir(extract_root)
        root = extract_root
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_root, entries[0])):
            root = os.path.join(extract_root, entries[0])
    elif not os.path.isdir(path):
        pytest.fail(f"--package is neither a tar archive nor a directory: {path}")
    info = {"root": root}
    info["mslite_chat"] = os.path.join(root, "bin", "mslite-chat")
    tool_dir = os.path.join(root, "tool")
    names = os.listdir(tool_dir) if os.path.isdir(tool_dir) else []
    exporter_wheels = [
        os.path.join(tool_dir, name) for name in names
        if name.startswith("mslite_llm-") and name.endswith(".whl")
    ]
    if len(exporter_wheels) != 1:
        pytest.fail(
            "release package must contain exactly one mslite_llm wheel; "
            f"found {len(exporter_wheels)}"
        )
    info["wheel"] = exporter_wheels[0]
    missing = [
        key for key in ("root", "mslite_chat", "wheel")
        if info[key] is None or not os.path.exists(info[key])
    ]
    if missing:
        pytest.fail(f"release package incomplete (missing: {missing})")
    return info


@pytest.fixture(scope="session")
def installed_wheel(release, tmp_path_factory):
    """Install the packaged exporter and check CI's operator adapters."""
    install_dir = os.path.join(tmp_path_factory.mktemp("st_wheel"), "site")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q",
         "--target", install_dir, release["wheel"]],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel install failed:\n{result.stdout}\n{result.stderr}")

    check = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import mslite_llm_ops

            required = (
                "MsRmsNorm", "MsAddRmsNorm", "MsAddSoftmax", "MsGroupMatmul",
                "MsQuant4N0Group32", "MsRotaryPosEmb", "MsScatterND",
            )
            missing = [name for name in required if not hasattr(mslite_llm_ops, name)]
            if missing:
                raise SystemExit("operator adapter package is missing: " + ", ".join(missing))
            print("[ST] Export adapters: " + mslite_llm_ops.__file__)
        """)],
        capture_output=True, text=True, check=False,
        env=dict(os.environ, PYTHONPATH=install_dir),
    )
    if check.returncode != 0:
        pytest.fail(f"operator adapter check failed:\n{check.stdout}\n{check.stderr}")
    print(check.stdout.strip(), flush=True)
    return install_dir


@pytest.fixture(scope="session")
def export_cli(installed_wheel):  # pylint: disable=unused-argument
    """Invocation for the packaged export CLI from the installed wheel.

    pip --target installs do not wire up console scripts, so run the module
    directly with the install dir on PYTHONPATH (still the packaged wheel).
    """
    return [sys.executable, "-m", "mslite_llm_export"]


@pytest.fixture(scope="session")
def msl_pack(installed_wheel):  # pylint: disable=unused-argument
    """utils.msl_pack imported from the installed wheel (guards the artifact)."""
    sys.path.insert(0, installed_wheel)  # pylint: disable=wrong-import-position
    from utils import msl_pack  # pylint: disable=import-outside-toplevel

    return msl_pack


@pytest.fixture(scope="session")
def model_input(request, model_cfg):
    """Raw model path resolved from --model-dir and the model registry."""
    model_dir = request.config.getoption("--model-dir")
    if not model_dir:
        pytest.skip("no model directory given: pass --model-dir=/path/to/models")
    if not os.path.isdir(model_dir):
        pytest.fail(f"--model-dir is not a directory: {model_dir}")
    path = os.path.join(model_dir, model_cfg["path"])
    if not os.path.exists(path):
        pytest.fail(f"registered model input does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def ddk_env():
    """The DDK environment must be sourced before ST runs.

    omg (compiled during the export stage) and te_fusion (its Python plugin)
    are resolved from ``DDK_PATH``; a missing/incomplete environment fails the
    run up front instead of surfacing as a confusing omg error deep inside the
    pipeline.
    """
    ddk = os.environ.get("DDK_PATH", "").strip()
    if not ddk:
        pytest.fail(
            "DDK not configured: source the DDK env first, e.g. "
            "`source $DDK/tools/tools_ascendc/set_ascendc_env.sh`"
        )
    omg = os.path.join(ddk, "tools", "tools_omg", "omg")
    if not os.path.isfile(omg):
        pytest.fail(f"DDK omg not found under DDK_PATH: {omg}")
    return ddk


@pytest.fixture(scope="session")
def binrunner_env(request):
    """BinRunner (``br``) must be installed and reach the device.

    The inference stage executes the packaged ``mslite-chat`` on the phone via
    BinRunner's memory loader (the only non-root exec path on HarmonyOS).  An
    absent CLI or unreachable device aborts the run up front.  Returns
    ``(br_path, udid, local_port)``; with multiple devices the first one is used
    (override via ``BR_UDID``).  The local port selects the existing hdc
    forwarding rule used for file transfer.
    """
    br = shutil.which("br")
    if not br:
        pytest.fail(
            "BinRunner (br) not found on PATH; install it and set up the device "
            "(`br setup`). See the BinRunner project docs."
        )
    proc = subprocess.run(
        [br, "devices"], capture_output=True, text=True, timeout=60, check=False
    )
    devices = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not devices:
        pytest.fail(
            f"BinRunner cannot see a device (`br devices` failed):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    udid = os.environ.get("BR_UDID", "").strip() or devices[0]
    if udid not in devices:
        pytest.fail(f"BR_UDID {udid!r} not among `br devices`: {devices}")
    port_text = str(request.config.getoption("--br-port")).strip()
    try:
        port = int(port_text)
    except ValueError:
        pytest.fail(f"--br-port must be an integer: {port_text!r}")
    if not 1 <= port <= 65535:
        pytest.fail(f"--br-port must be in 1-65535: {port}")
    return br, udid, port


@pytest.fixture(scope="session")
def device_ready(binrunner_env, ddk_env, mslite_chat):
    """Real-device gate: MSLITE_LLM_ST_DEVICE=1 plus a working BinRunner/DDK.

    Unlike the other fixtures, an undeclared device is a skip (host-only run);
    once declared, a missing BinRunner or DDK setup fails loudly, and the
    packaged mslite-chat must be an AArch64 ELF (device inference needs the
    OHOS build, `build.sh -b nnrt`).
    """
    if os.environ.get("MSLITE_LLM_ST_DEVICE") != "1":
        pytest.skip("device not declared: set MSLITE_LLM_ST_DEVICE=1 to run NPU inference")
    with open(mslite_chat, "rb") as f:
        head = f.read(24)
    if len(head) != 24 or head[:4] != b"\x7fELF":
        pytest.fail(f"{mslite_chat} is not an ELF binary")
    e_machine = struct.unpack("<H", head[18:20])[0]
    if e_machine != 0xB7:  # EM_AARCH64
        pytest.fail(
            f"{mslite_chat} is not AArch64 (e_machine={e_machine:#x}); "
            "device inference needs the OHOS build (`build.sh -b nnrt`)"
        )
    return binrunner_env


@pytest.fixture(scope="session")
def msl_package(request, model_cfg, installed_wheel, tmp_path_factory, ddk_env):
    """The .msl package: reuse --msl when given, otherwise run the packaged export.

    The export stage runs the wheel's ``mslite-llm-export`` (GGUF/HF -> ONNX ->
    omg(.omc) -> .msl).  omg needs the DDK env sourced (checked up front by
    ``ddk_env``) and the Ms* custom ops installed; the first compile is slow
    (kernel cache makes reruns fast).
    """
    given = request.config.getoption("--msl")
    if given:
        if not os.path.isfile(given):
            pytest.fail(f"--msl file does not exist: {given}")
        return given

    # Export stage prerequisites are fetched lazily so --msl runs do not
    # require a raw model from --model-dir or the wheel.
    model_input = request.getfixturevalue("model_input")
    export_cli = request.getfixturevalue("export_cli")
    out = os.path.join(tmp_path_factory.mktemp("st_export"), "model.msl")
    cmd = export_cli + [
        "--model", model_input,
        "--output", out,
        "--target", model_cfg["target"],
        "--max-length", str(model_cfg["max_length"]),
        "--chunk-size", str(model_cfg["chunk_size"]),
    ]
    # Keep the DDK on PYTHONPATH: omg's te_fusion is resolved from the DDK
    # package/python dir, which must survive the wheel-first override.
    ddk_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    pythonpath = installed_wheel
    if ddk_pythonpath:
        pythonpath = installed_wheel + os.pathsep + ddk_pythonpath
    env = dict(os.environ, PYTHONPATH=pythonpath)
    print("[ST] Exporting model (GGUF/HF -> ONNX -> OMC -> MSL)", flush=True)
    try:
        returncode = _run_export(cmd, env)
    except subprocess.TimeoutExpired:
        pytest.fail(f"export pipeline timed out after {EXPORT_TIMEOUT_SECONDS}s")
    if returncode != 0:
        pytest.fail(f"export pipeline failed ({returncode}); see output tail above")
    if not os.path.isfile(out):
        pytest.fail(f"export pipeline did not produce {out}")
    print(f"[ST] Model export complete: {out}", flush=True)
    return out


@pytest.fixture(scope="session")
def mslite_chat(release):
    """mslite-chat from the release package bin/."""
    return release["mslite_chat"]
