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

# ----------- config file -----------
# [acl_build_options]
# input_format="ND"
# input_shape="sample:2,4,-1,-1;timestep:1;encoder_hidden_states:2,77,768"
# ge.dynamicDims="64,64;96,96"

MODEL_FILE = "./sd1.5_unet.onnx_graph.mindir"
# Read device_id from environment set by conftest.py pytest_configure
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def test_python_api_func_parallel_001():
    """
    test_python_api_func_parallel_001 success
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    context.parallel.workers_num = 2
    runner = mslite.ModelParallelRunner()
    runner.build_from_file(model_path=MODEL_FILE, context=context)

    input_1 = mslite.Tensor(np.random.rand(2, 4, 64, 64).astype(np.float32))
    input_2 = mslite.Tensor(np.random.rand(1).astype(np.float32))
    input_3 = mslite.Tensor(np.random.rand(2, 77, 768).astype(np.float32))
    runner.predict([input_1, input_2, input_3])


def test_python_api_fi_parallel_003():
    """
    test_python_api_fi_parallel_003 file not exist
    """
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.parallel.workers_num = 2
        runner = mslite.ModelParallelRunner()
        runner.build_from_file(model_path="xxx", context=context)
    assert "ModelParallelRunner's build from file failed, model_path does not exist!" in str(raise_info)


def test_python_api_fi_parallel_005():
    """
    test_python_api_fi_parallel_005 incorrect input
    """
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.parallel.workers_num = 2
        runner = mslite.ModelParallelRunner()
        runner.build_from_file(model_path=MODEL_FILE, context=context)
        runner.predict([])
    assert "predict failed!" in str(raise_info.value)


def test_python_api_fi_parallel_006():
    """
    test_python_api_fi_parallel_006 predict without build
    """
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        context.parallel.workers_num = 2
        runner = mslite.ModelParallelRunner()
        inputs = runner.get_inputs()
        runner.predict(inputs)
    assert "predict failed!" in str(raise_info.value)
