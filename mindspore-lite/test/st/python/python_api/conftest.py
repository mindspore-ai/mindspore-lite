# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Pytest ST configuration
"""
import os

import pytest


def pytest_configure(config):
    """
    Configure hook - runs before test collection.
    Set ASCEND_DEVICE_ID from --device_id option for both Python tests and subprocess tools.
    """
    device_ids = sorted(set(config.getoption("device_id")))
    assigned_id = device_ids[0] if device_ids else 0
    os.environ['ASCEND_DEVICE_ID'] = str(assigned_id)
    print(f"\n[pytest] assigned device_id={assigned_id}")


def pytest_addoption(parser):
    """
    pytest extra options
    """
    parser.addoption(
        "--backend",
        action="store",
        nargs="+",
        default=[],
        choices=(
            "arm_ascend310p_cloud",
            "arm_ascend310p_ge_cloud",
            "mslite_large_model_inference_arm_ascend910B",
            "arm32_cpu",
            "arm64_cpu",
        ),
        help="Only test specified backend testcases. Example: --backend arm_ascend310_cloud arm_ascend310_ge_cloud",
    )

    parser.addoption(
        "--device_id",
        action="store",
        nargs="+",
        default=[0],
        type=int,
        help="Available device ids for test, default is [0]. Example: --device_id 0 1",
    )

    parser.addoption(
        "--mindir_dir",
        action="store",
        default="",
        help="path of mindir",
    )

    parser.addoption(
        "--output_dir",
        action="store",
        default="",
        help="convert output dir",
    )

    parser.addoption(
        "--config_dir",
        action="store",
        default="",
        help="convert config dir",
    )

    parser.addoption(
        "--so_path",
        action="store",
        default="",
        help="path of mindspore_lite tools",
    )

@pytest.fixture(scope="session")
def device_id(request):
    """
    device_id fixture
    """
    return list(set(request.config.getoption("device_id")))

@pytest.fixture(scope="session")
def so_path(request):
    """
    so_path fixture
    """
    return request.config.getoption("so_path")

@pytest.fixture(scope="session")
def mindir_dir(request):
    """
    mindir_dir fixture
    """
    return request.config.getoption("mindir_dir")

@pytest.fixture(scope="session")
def output_dir(request):
    """
    output_dir fixture
    """
    return request.config.getoption("output_dir")

@pytest.fixture(scope="session")
def config_dir(request):
    """
    config_dir fixture
    """
    return request.config.getoption("config_dir")

def _parse_backend_mark(item, device_id_option):
    """
    parse backend mark from item
    """
    marker = item.get_closest_marker("backend")
    if not marker:
        return None
    supported_backends = marker.args
    require_device_num = marker.kwargs.get("require_device_num", 1)
    if not supported_backends:
        raise ValueError(f"item {item} marked with backend but no backend specified.")
    if not isinstance(require_device_num, int):
        raise TypeError(
            f"item {item} marked with require_device_num={require_device_num},"
            f" but require_device_num should be int. got {type(require_device_num)}"
        )
    if require_device_num < 0:
        raise ValueError(
            f"item {item} marked with require_device_num={require_device_num},"
            f" but require_device_num should be non-negative."
        )
    if len(device_id_option) < require_device_num:
        raise ValueError(
            f"item {item} marked with require_device_num={require_device_num},"
            f" but got device_id_option={device_id_option}, not enough devices."
        )
    return supported_backends


def pytest_collection_modifyitems(config, items):
    """
    pytest collection modifyitems hook.
    1. if a backend option is appeared, this hook will filter items that are marked with target backend
    """
    backend_types = config.getoption("backend")
    device_id_option = list(set(config.getoption("device_id")))
    if not backend_types:
        return

    selected = []
    for item in items:
        supported_backends = _parse_backend_mark(item, device_id_option)
        if not supported_backends:
            continue
        if "all" in supported_backends or any(backend in supported_backends for backend in backend_types):
            selected.append(item)

    config.hook.pytest_deselected(items=list(filter(lambda i: i not in selected, items)))
    items[:] = selected
