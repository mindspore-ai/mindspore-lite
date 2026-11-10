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
"""Options shared by the whole lite_boost test tree.

``--mode`` lets the model test files put accuracy and performance cases in one
module while still running them separately (the end-to-end cases load a
multi-GB checkpoint, so nobody wants to pay that twice):

    python -m pytest models/qwen3_30b_a3b/test_qwen3_30b_a3b.py --mode accuracy
    python -m pytest models/qwen3_30b_a3b/test_qwen3_30b_a3b.py --mode perf

The model conftests implement the deselection with the ``accuracy`` / ``perf``
markers; ``all`` (the default) keeps the historical behaviour of running
everything pytest selects.

A model directory follows one layout so every case is discoverable and the
mode names line up with the files that implement them:

    <model>/test_<model>.py       pytest cases (marks, assertions)
    <model>/conftest.py           checkpoint / parallel-size fixtures, --mode
    <model>/accuracy_runner.py    torchrun entry for --mode accuracy
    <model>/perf_runner.py        torchrun entry for --mode perf
    <model>/*.yaml                optional static boost configs

The two runners are plain scripts: they stay runnable by hand with torchrun,
and pytest only wraps them to judge their output. New model directories are
expected to use these names.
"""

MODE_OPTION = "--mode"
MODE_CHOICES = ("all", "accuracy", "perf")
MODE_HELP = (
    "Select which Qwen3-30B-A3B cases to run: 'accuracy', 'perf', or 'all' "
    "(default). Cases without either marker are always kept."
)


def pytest_addoption(parser):
    """Register the suite-wide ``--mode`` switch."""
    parser.addoption(MODE_OPTION, action="store", default="all",
                     choices=MODE_CHOICES, help=MODE_HELP)
