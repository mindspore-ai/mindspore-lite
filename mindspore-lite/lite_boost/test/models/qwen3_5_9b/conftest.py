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
"""Fixtures shared by the Qwen3.5-9B end-to-end tests.

The end-to-end cases need a bf16 Qwen3.5-9B checkpoint and free NPUs, which the
default CI image does not have. Both are supplied through the environment; when
either is missing the cases skip instead of failing:

    QWEN35_9B_MODEL_DIR         Hugging Face bf16 checkpoint directory
    QWEN35_9B_TP                parallel world size (default: 1)
    QWEN35_9B_MODE              multi-card scheme, tp or cp (default: tp)
    ASCEND_RT_VISIBLE_DEVICES   cards handed to torchrun (default: 0..TP-1)

The cases share one module and are split by the suite-wide ``--mode`` option
(registered in ``test/conftest.py``): cases marked ``accuracy`` or ``perf`` are
deselected when the other mode is requested. Each mode has its own torchrun
entry next to this file -- ``accuracy_runner.py`` and ``perf_runner.py`` --
named after the ``--mode`` value they serve.
"""
import os
import subprocess
import sys

import pytest

MODEL_DIR_ENV = "QWEN35_9B_MODEL_DIR"
TP_ENV = "QWEN35_9B_TP"
MODE_ENV = "QWEN35_9B_MODE"
DEFAULT_TP = 1
DEFAULT_MODE = "tp"
PARALLEL_MODES = ("tp", "cp")
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
    """Return the configured parallel world size."""
    return int(os.environ.get(TP_ENV, str(DEFAULT_TP)))


def _parallel_mode():
    """Return the configured multi-card scheme (``tp`` or ``cp``)."""
    mode = os.environ.get(MODE_ENV, DEFAULT_MODE)
    if mode not in PARALLEL_MODES:
        raise ValueError(f"{MODE_ENV} must be one of {PARALLEL_MODES}, got {mode!r}")
    return mode


def run_torchrun(entry: str, args, world_size: int, timeout: int = TORCHRUN_TIMEOUT_S):
    """Launch ``entry`` with torchrun and return its stdout (raises on failure)."""
    cmd = [sys.executable, "-m", "torch.distributed.run",
           f"--nproc_per_node={world_size}", entry, *args]
    env = dict(os.environ)
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES",
                   ",".join(str(i) for i in range(world_size)))
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
    """Return the parallel world size after checking enough cards are visible."""
    world = tp_size()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible and len(visible.split(",")) < world:
        pytest.skip(f"{TP_ENV}={world} needs {world} cards in "
                    f"ASCEND_RT_VISIBLE_DEVICES (got {visible})")
    return world


@pytest.fixture(scope="session")
def parallel_mode():
    """Return the multi-card scheme (``tp``/``cp``) the runners must use."""
    return _parallel_mode()
