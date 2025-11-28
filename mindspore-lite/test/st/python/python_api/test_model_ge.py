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
Test for MindSpore Lite Model
"""

import os
import pytest
import mindspore_lite as mslite
import numpy as np

MODEL_FILE = "./ge_test_mul.mindir"
DEVICE_ID = 0


# For Model Resize
def test_python_ge_api_resize_001(config_dir):
    """
    Test model resize functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    config_file = os.path.join(config_dir, 'ge_test_mul.config')
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    context.ascend.provider = "ge"
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                          config_path=config_file)
    model.resize(model.get_inputs(), [[1, 128], [1, 128]])

    x = np.array(np.ones((1, 128))).astype(np.float32)
    res = model.predict([x, x])


def test_python_ge_api_fail_001(config_dir):
    """
    Test model path not exist functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path="xxxx", model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)

    assert "build_from_file failed, model_path does not exist!" in str(raise_info)


def test_python_ge_api_fail_002(config_dir):
    """
    Test model input data empty functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)
        model.resize(model.get_inputs(), [[1, 128], [1, 128]])

        x = np.array(np.ones((1, 128))).astype(np.float32)
        res = model.predict([])

    assert "not equal input len" in str(raise_info)


def test_python_ge_api_fail_003(config_dir):
    """
    Test model input data not set functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)

        x = np.array(np.ones((1, 128))).astype(np.float32)
        res = model.predict(model.get_inputs())

    assert "predict failed" in str(raise_info)


def test_python_ge_api_fail_004(config_dir):
    """
    Test model input data type not equal functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)
        model.resize(model.get_inputs(), [[1, 128], [1, 128]])

        x = np.array(np.ones((1, 128))).astype(np.int32)
        res = model.predict([x, x])

    assert "data type not equal" in str(raise_info)


def test_python_ge_api_fail_005(config_dir):
    """
    Test model input data shape not equal functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)
        model.resize(model.get_inputs(), [[1, 128], [1, 128]])

        x = np.array(np.ones((1, 64))).astype(np.float32)
        res = model.predict([x, x])

    assert "data size not equal" in str(raise_info)


def test_python_ge_api_fail_006(config_dir):
    """
    Test model input data number not equal functionality with GE backend.

    Args:
        config_dir (str): Path to directory containing configuration files.

    Examples:
        >>> test_python_ge_api_resize_001("/path/to/config/")
    """
    with pytest.raises(RuntimeError) as raise_info:
        config_file = os.path.join(config_dir, 'ge_test_mul.config')
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.ascend.provider = "ge"
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_path=config_file)
        model.resize(model.get_inputs(), [[1, 128], [1, 128]])

        x = np.array(np.ones((1, 128))).astype(np.float32)
        res = model.predict([x, x, x])

    assert "not equal input len" in str(raise_info)
