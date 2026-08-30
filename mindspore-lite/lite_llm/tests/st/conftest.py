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
"""ST shared fixtures: release package / model input / device declaration.

ST guards the shipped artifact: it always runs against the release package
(``--package``, the build.sh tarball), never the source tree. The wheel inside
the package is installed to a scratch dir and both the export CLI
(``mslite-llm-export``) and ``utils.msl_pack`` are exercised from there.

    # Full chain: package -> install wheel -> export (GGUF -> .msl) -> mslite-chat
    MSLITE_LLM_ST_DEVICE=1 pytest tests/st \
        --package=output/mindspore-lite-llm-linux-x64-0.1.0.tar.gz --gguf=/path/model.gguf

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

import pytest

# Model parameter table: one entry per guarded model.  The export options are
# consumed by mslite-llm-export (see export/README.md).
MODELS = {
    "qwen2.5-0.5b": {
        "target": "kirin9020",
        "max_length": 1024,
        "chunk_size": 64,
    },
}


def pytest_addoption(parser):
    parser.addoption("--model", default="qwen2.5-0.5b",
                     help="model id registered in conftest.MODELS (default: qwen2.5-0.5b)")
    parser.addoption("--package", default=None,
                     help="release package path (.tar.gz or extracted dir); ST runs against it")
    parser.addoption("--gguf", default=None,
                     help="raw model input (GGUF file or HF dir); required for the export stage")
    parser.addoption("--msl", default=None,
                     help="pre-built .msl package; when given the export stage is skipped")


@pytest.fixture(scope="session")
def model_id(request):
    """Model id from --model; validated against MODELS."""
    name = request.config.getoption("--model")
    if name not in MODELS:
        pytest.skip(f"--model {name} not registered (available: {list(MODELS)})")
    return name


@pytest.fixture(scope="session")
def model_cfg(model_id):
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
    if tarfile.is_tarfile(path):
        root = os.path.join(tmp_path_factory.mktemp("st_release"), "pkg")
        with tarfile.open(path) as tf:
            tf.extractall(root, filter="data")
    info = {"root": root}
    info["mslite_chat"] = os.path.join(root, "bin", "mslite-chat")
    info["wheel"] = next(
        (os.path.join(root, "tool", n) for n in os.listdir(os.path.join(root, "tool"))
         if n.endswith(".whl")),
        None,
    )
    missing = [k for k, v in info.items() if v is None or not os.path.exists(v)]
    if missing:
        pytest.fail(f"release package incomplete (missing: {missing})")
    return info


@pytest.fixture(scope="session")
def installed_wheel(release, tmp_path_factory):
    """Install the packaged wheel into a scratch dir; returns its root."""
    install_dir = os.path.join(tmp_path_factory.mktemp("st_wheel"), "site")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-q",
         "--target", install_dir, release["wheel"]],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"wheel install failed:\n{result.stdout}\n{result.stderr}")
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
def model_input(request):
    """Raw model path (--gguf). Skipped when absent: no input, nothing to guard."""
    path = request.config.getoption("--gguf")
    if not path:
        pytest.skip("no raw model given: pass --gguf=/path/model.gguf")
    if not os.path.exists(path):
        pytest.fail(f"--gguf path does not exist: {path}")
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
def binrunner_env():
    """BinRunner (``br``) must be installed and reach the device.

    The inference stage executes the packaged ``mslite-chat`` on the phone via
    BinRunner's memory loader (the only non-root exec path on HarmonyOS).  An
    absent CLI or unreachable device aborts the run up front.  Returns
    ``(br_path, udid)``; with multiple devices the first one is used (override
    via ``BR_UDID``).
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
    return br, udid


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
    # require the raw model (--gguf) or the wheel.
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
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, check=False, env=env)
    if result.returncode != 0:
        pytest.fail(f"export pipeline failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    if not os.path.isfile(out):
        pytest.fail(f"export pipeline did not produce {out}")
    return out


@pytest.fixture(scope="session")
def mslite_chat(release):
    """mslite-chat from the release package bin/."""
    return release["mslite_chat"]
