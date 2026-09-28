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
"""ST: model-based end-to-end guard (export -> real-device inference).

One test case per guarded model; the case name carries the model id (e.g.
``test_qwen2_5_0b5_full_chain``).  Model parameters live in
``conftest.MODELS``; add a model there plus a case here.  Each case runs the
full chain in a single test:

    1. conversion stage — GGUF/HF -> ONNX -> omg(.omc) -> single-file .msl
       (skipped when ``--msl`` is given), verified by unpacking the package;
    2. inference stage — ``mslite-chat`` loads the package and generates,
       verified by the streamed output / finish reason / stats lines.

The inference stage requires a Kirin NPU device (``MSLITE_LLM_ST_DEVICE=1``);
without it the case skips after the conversion stage has been validated.
"""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

PROMPT = "你好，请介绍一下你自己"
MAX_TOKENS = "5"
DEVICE_LOG_TAIL_LINES = 300
DEVICE_LOG_PATTERN = re.compile(
    r"mslite|MSLLM|NNRT|CANN|HIAI|AI_FMK|TE_FUSION|"
    r"Executor|MODEL_LOAD|insecure",
    re.IGNORECASE,
)
BINRUNNER_LOG_PATTERN = re.compile(
    r"not readable:|resolved via push dir:|<<< exit=|push .*: (?:OK|FAIL)"
)
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w.])/(?:[^\s\"'(),:]+/?)+")
DEVICE_ID_PATTERN = re.compile(r"\b[A-Z0-9]{16}\b")
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Model-specific facts asserted against the unpacked package (KV + resources).
QWEN2_5_0B5 = {
    "num_layers": 24,
    "hidden_size": 896,
    "vocab_size": 151936,
}


def _sanitize_device_log(text, udid):
    """Remove device identifiers, paths and non-ASCII payload text from logs."""
    text = text.replace(udid, "<DEVICE>").replace(PROMPT, "<PROMPT>")
    text = DEVICE_ID_PATTERN.sub("<DEVICE>", text)
    text = IPV4_PATTERN.sub("<IP>", text)
    text = ABSOLUTE_PATH_PATTERN.sub("<PATH>", text)
    text = re.sub(r"[^\x00-\x7f]+", "<TEXT>", text)
    text = re.sub(r"not readable:.*", "executable not readable", text)
    text = re.sub(r"resolved via push dir:.*", "executable resolved from push directory", text)
    text = re.sub(r"push .*?: (OK|FAIL).*", r"push status=\1", text)
    return text


def _run_diagnostic(title, cmd, udid, timeout=60, line_filter=None):
    """Run one failure-only diagnostic without hiding the original failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[ST] {title} unavailable: {_sanitize_device_log(str(exc), udid)}"
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    lines = output.splitlines()
    if line_filter is not None:
        lines = [line for line in lines if line_filter.search(line)]
    lines = [_sanitize_device_log(line, udid) for line in lines[-DEVICE_LOG_TAIL_LINES:]]
    body = "\n".join(lines) if lines else "(no matching output)"
    return f"[ST] {title} (exit={proc.returncode}, last {len(lines)} line(s)):\n{body}"


def _check_device_files(br_cmd, udid, model_name):
    """Report only expected-file presence, without printing the directory listing."""
    try:
        proc = subprocess.run(
            br_cmd + ["ls", "@/bin"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[ST] BinRunner files unavailable: {_sanitize_device_log(str(exc), udid)}"
    output = proc.stdout + proc.stderr
    executable = "present" if "mslite-chat" in output else "missing"
    model = "present" if model_name in output else "missing"
    return f"[ST] BinRunner files (exit={proc.returncode}): executable={executable}, model={model}"


def _collect_device_diagnostics(br_cmd, udid, model_name):
    """Collect bounded and sanitized device diagnostics after a failure."""
    sections = [
        _run_diagnostic("BinRunner version", br_cmd + ["version"], udid, timeout=30),
        _check_device_files(br_cmd, udid, model_name),
    ]
    hdc = shutil.which("hdc")
    if not hdc:
        sections.append("[ST] Device hilog unavailable: hdc not found on PATH")
        return "\n".join(sections)
    sections.extend([
        _run_diagnostic(
            "BinRunner hilog",
            [hdc, "-t", udid, "shell", "hilog -x -T BinRunner"],
            udid,
            line_filter=BINRUNNER_LOG_PATTERN,
        ),
        _run_diagnostic(
            "Filtered device hilog",
            [hdc, "-t", udid, "shell", "hilog -x"],
            udid,
            timeout=90,
            line_filter=DEVICE_LOG_PATTERN,
        ),
    ])
    return "\n".join(sections)


@pytest.mark.parametrize("model_id", ["qwen2.5-0.5b"], indirect=True)
def test_qwen2_5_0b5_full_chain(
    model_id, model_cfg, msl_package, mslite_chat, device_ready, msl_pack
):
    """qwen2.5-0.5b: export (or reuse --msl) -> real-device inference.

    ``msl_pack`` is imported from the installed wheel (guards the artifact);
    ``device_ready`` fails up front when the DDK/BinRunner setup is missing or
    the packaged binary is not an AArch64 build.
    """
    # Fixture side effects gate the case (skip/fail on missing prerequisites);
    # re-assert the device contract here so it is exercised explicitly.
    assert model_id == "qwen2.5-0.5b"
    assert device_ready

    # ── Stage 1: package sanity (conversion stage already ran in msl_package) ──
    with tempfile.TemporaryDirectory(prefix="st_unpack_") as tmp:
        kv = msl_pack.unpack(msl_package, tmp)
        assert kv.get("arch.num_layers") == QWEN2_5_0B5["num_layers"], kv
        assert kv.get("arch.hidden_size") == QWEN2_5_0B5["hidden_size"], kv
        assert kv.get("arch.vocab_size") == QWEN2_5_0B5["vocab_size"], kv
        assert kv.get("npu.max_length") == model_cfg["max_length"], kv
        # The .omc entry name is whatever the exporter wrote into the manifest
        # (fixed basename "model" in the pipeline), not derived from the .msl
        # filename — read it from the manifest path to stay contract-faithful.
        omc_entry = kv.get("litert.prefill.path")
        assert omc_entry, f"litert.prefill.path missing: {kv}"
        required = [
            omc_entry,
            "vocab/vocab.bin",
            "assets/embedding_quant.bin",
            "assets/rope_cos.bin",
            "assets/rope_sin.bin",
            "assets/attention_mask.bin",
        ]
        missing = [name for name in required if not os.path.isfile(os.path.join(tmp, name))]
        assert not missing, f"missing resources in package: {missing}"

    # ── Stage 2: device inference via BinRunner (memory loader, no root) ────
    # The packaged OHOS binary and the .msl are pushed to the phone; `br run`
    # executes mslite-chat in the app sandbox and streams stdout/stderr back.
    br, udid, local_port = device_ready
    br_cmd = [br, "-t", udid, "-p", str(local_port)]
    model_name = os.path.basename(msl_package)
    push_cmds = [
        br_cmd + ["push", mslite_chat, "mslite-chat"],
        br_cmd + ["push", msl_package, model_name],
    ]
    for cmd in push_cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        if proc.returncode != 0:
            diagnostics = _collect_device_diagnostics(br_cmd, udid, model_name)
            stderr = _sanitize_device_log(proc.stderr, udid)
            pytest.fail(
                f"BinRunner push failed (exit={proc.returncode}, "
                f"stdout_bytes={len(proc.stdout.encode())}):\n{stderr}\n{diagnostics}"
            )

    run_cmd = f"mslite-chat @/bin/{model_name} {PROMPT} {MAX_TOKENS}"
    result = subprocess.run(
        br_cmd + ["run", run_cmd], capture_output=True, text=True, timeout=1800, check=False
    )
    if result.returncode != 0:
        diagnostics = _collect_device_diagnostics(br_cmd, udid, model_name)
        stderr = _sanitize_device_log(result.stderr, udid)
        pytest.fail(
            f"mslite-chat failed (exit={result.returncode}, "
            f"stdout_bytes={len(result.stdout.encode())}):\n{stderr}\n{diagnostics}"
        )
    assert "[finish reason]" in result.stdout, "no finish reason in output"
    assert "[stats]" in result.stdout, "no stats line in output"
    assert len(result.stdout) > 0, "empty generation output"
