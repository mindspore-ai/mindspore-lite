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
"""Module helpers for the custom-ops build tooling."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# Register the vendored torch_custom package by content instead of mutating
# sys.path; test modules keep plain ``from torch_custom.ms_* import`` lines.
# (Sibling helpers like base_test resolve via pytest's own prepend import of
# this conftest's directory.)
_TORCH_CUSTOM_DIR = REPO_ROOT / "torch_custom"
if "torch_custom" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "torch_custom", str(_TORCH_CUSTOM_DIR / "__init__.py"), submodule_search_locations=[str(_TORCH_CUSTOM_DIR)]
    )
    _torch_custom = importlib.util.module_from_spec(_spec)
    sys.modules["torch_custom"] = _torch_custom
    _spec.loader.exec_module(_torch_custom)


def pytest_addoption(parser):
    """Register the --ext-platform command line option."""
    parser.addoption(
        "--ext-platform",
        action="store",
        default=os.environ.get("EXT_PLATFORM", ""),
        help="Override the platform used by OMG for device operator tests.",
    )


@pytest.fixture
def ext_platform(request):
    """Provide the platform override passed via --ext-platform."""
    return request.config.getoption("--ext-platform")
