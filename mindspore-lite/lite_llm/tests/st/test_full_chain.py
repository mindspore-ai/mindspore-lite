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
import subprocess
import tempfile

import pytest

PROMPT = "你好，请介绍一下你自己"
MAX_TOKENS = "5"

# Model-specific facts asserted against the unpacked package (KV + resources).
QWEN2_5_0B5 = {
    "num_layers": 24,
    "hidden_size": 896,
    "vocab_size": 151936,
}


def test_qwen2_5_0b5_full_chain(model_cfg, msl_package, mslite_chat, device_ready, msl_pack):
    """qwen2.5-0.5b: export (or reuse --msl) -> real-device inference.

    ``msl_pack`` is imported from the installed wheel (guards the artifact);
    ``device_ready`` fails up front when the DDK/BinRunner setup is missing or
    the packaged binary is not an AArch64 build.
    """
    # Fixture side effects gate the case (skip/fail on missing prerequisites);
    # re-assert the device contract here so it is exercised explicitly.
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
    br, udid = device_ready
    br_cmd = [br, "-t", udid]
    model_name = os.path.basename(msl_package)
    push_cmds = [
        br_cmd + ["push", mslite_chat, "mslite-chat"],
        br_cmd + ["push", msl_package, model_name],
    ]
    for cmd in push_cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        if proc.returncode != 0:
            pytest.fail(f"BinRunner push failed ({cmd}):\n{proc.stdout}\n{proc.stderr}")

    run_cmd = f"mslite-chat @/bin/{model_name} {PROMPT} {MAX_TOKENS}"
    result = subprocess.run(
        br_cmd + ["run", run_cmd], capture_output=True, text=True, timeout=1800, check=False
    )
    assert result.returncode == 0, (
        f"mslite-chat failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
    )
    assert "[finish reason]" in result.stdout, f"no finish reason in output:\n{result.stdout}"
    assert "[stats]" in result.stdout, f"no stats line in output:\n{result.stdout}"
    assert len(result.stdout) > 0, "empty generation output"
