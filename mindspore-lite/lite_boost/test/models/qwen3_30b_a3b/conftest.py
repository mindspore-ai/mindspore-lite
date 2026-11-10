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
"""Fixtures shared by the Qwen3-30B-A3B end-to-end tests.

The end-to-end cases need a bf16 Qwen3-30B-A3B checkpoint and several free
NPUs, which the default CI image does not have. Both are supplied through the
environment; when either is missing the cases skip instead of failing:

    QWEN3_30B_A3B_MODEL_DIR     Hugging Face bf16 checkpoint directory
    QWEN3_30B_A3B_TP            tensor-parallel size (default: 2)
    ASCEND_RT_VISIBLE_DEVICES   cards handed to torchrun (default: 0..TP-1)

The cases share one module per checkpoint flavour and are split by the
suite-wide ``--mode`` option (registered in ``test/conftest.py``): cases marked
``accuracy`` or ``perf`` are deselected when the other mode is requested. Each
mode has its own torchrun entry next to this file -- ``accuracy_runner.py`` and
``perf_runner.py`` -- named after the ``--mode`` value they serve.
"""
import os
import subprocess
import sys

import pytest

MODEL_DIR_ENV = "QWEN3_30B_A3B_MODEL_DIR"
TP_ENV = "QWEN3_30B_A3B_TP"
DEFAULT_TP = 2
TORCHRUN_TIMEOUT_S = 3600

_HERE = os.path.dirname(os.path.abspath(__file__))
ACCURACY_RUNNER = os.path.join(_HERE, "accuracy_runner.py")

MODE_MARKERS = frozenset({"accuracy", "perf"})


def pytest_collection_modifyitems(config, items):
    """Deselect the cases that belong to the other ``--mode`` group."""
    mode = config.getoption("--mode", default="all")
    if mode == "all":
        return
    kept, deselected = [], []
    for item in items:
        marks = {marker.name for marker in item.iter_markers()}
        if mode in marks or not marks & MODE_MARKERS:
            kept.append(item)
        else:
            deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def tp_size():
    """Return the configured tensor-parallel world size."""
    return int(os.environ.get(TP_ENV, str(DEFAULT_TP)))


def run_torchrun(entry: str, args, world_size: int, timeout: int = TORCHRUN_TIMEOUT_S,
                 env_overrides=None):
    """Launch ``entry`` with torchrun and return its stdout (raises on failure)."""
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={world_size}", entry, *args]
    env = dict(os.environ)
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES",
                   ",".join(str(i) for i in range(world_size)))
    env.update(env_overrides or {})
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, env=env, check=False)
    if proc.returncode != 0:
        tail_out, tail_err = proc.stdout[-4000:], proc.stderr[-4000:]
        raise AssertionError(
            f"torchrun failed (exit {proc.returncode})\n--- stdout ---\n"
            f"{tail_out}\n--- stderr ---\n{tail_err}")
    return proc.stdout


@pytest.fixture(scope="session")
def model_dir():
    """Return the bf16 checkpoint directory, or skip the calling test."""
    path = os.environ.get(MODEL_DIR_ENV, "")
    if not path or not os.path.isdir(path):
        pytest.skip(f"{MODEL_DIR_ENV} is not set to an existing directory")
    return path


@pytest.fixture(scope="session")
def parallel_size():
    """Return the TP world size after checking enough cards are visible."""
    world = tp_size()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible and len(visible.split(",")) < world:
        pytest.skip(f"{TP_ENV}={world} needs {world} cards in "
                    f"ASCEND_RT_VISIBLE_DEVICES (got {visible})")
    return world
