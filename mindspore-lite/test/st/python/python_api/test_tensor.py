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
Test for MindSpore Lite MSTensor
"""

import pytest
import mindspore_lite as mslite
import numpy as np


# DeviceTensor
def test_runtime_general_device_tensor_func_001():
    with pytest.raises(TypeError) as e:
        tensor = mslite.Tensor(shape=[1, 2], dtype=mslite.DataType.FLOAT32, device="cpu")
        assert "now only support ascend device" in str(e.value)


def test_runtime_general_device_tensor_func_002():
    tensor = mslite.Tensor(shape=[1, 2], dtype=mslite.DataType.FLOAT32, device="ascend:0")
    assert tensor.device == "ascend:0"


def test_runtime_general_device_tensor_func_003():
    data = np.array([1, 2, 3]).astype(np.float32)
    device_tensor = mslite.Tensor(data, dtype=mslite.DataType.FLOAT32, device="ascend:0")
    assert device_tensor.device == "ascend:0"
    host_tensor = mslite.Tensor(device_tensor)  # D2H
    assert host_tensor.device == "None:-1"
    assert (device_tensor.get_data_to_numpy() == host_tensor.get_data_to_numpy()).all()

    host_tensor_1 = mslite.Tensor(data, dtype=mslite.DataType.FLOAT32)
    assert host_tensor_1.device == "None:-1"
    device_tensor_1 = mslite.Tensor(host_tensor_1, device="ascend:0")
    assert device_tensor_1.device == "ascend:0"
    assert (host_tensor_1.get_data_to_numpy() == device_tensor_1.get_data_to_numpy()).all()


# TensorAPI
def test_python_api_func_tensor_001():
    tensor = mslite.Tensor()
    assert tensor.name == ""
    assert tensor.dtype == mslite.DataType.FLOAT32
    assert tensor.shape == []
    assert tensor.format == mslite.Format.NCHW
    assert tensor.element_num == 1
    assert tensor.data_size == 4


def test_python_api_func_tensor_002():
    tensor = mslite.Tensor()
    assert tensor.name == ""
    tensor.name = "tensor1"
    assert tensor.name == "tensor1"


def test_python_api_func_tensor_003():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.INT32
    assert tensor.dtype == mslite.DataType.INT32


def test_python_api_func_tensor_004():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.FLOAT32
    tensor.shape = [16, 16]
    assert tensor.shape == [16, 16]
    assert tensor.element_num == 256
    assert tensor.data_size == 1024


def test_python_api_func_tensor_005():
    tensor = mslite.Tensor()
    tensor.format = mslite.Format.NHWC4
    assert tensor.format == mslite.Format.NHWC4


def test_python_api_func_tensor_006():
    tensor = mslite.Tensor()
    tensor.dtype = mslite.DataType.FLOAT32
    tensor.shape = [2, 3]
    in_data = np.arange(2 * 3, dtype=np.float32).reshape((2, 3))
    tensor.set_data_from_numpy(in_data)
    tensor_data = tensor.get_data_to_numpy()
    assert (tensor_data == in_data).all()


def test_python_api_fi_tensor_001():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.dtype = mslite.DataType.FLOAT32
        in_data = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
        tensor.set_data_from_numpy(in_data)
    assert "data size not equal" in str(raise_info.value)


def test_python_api_fi_tensor_002():
    with pytest.raises(RuntimeError) as raise_info:
        tensor = mslite.Tensor()
        tensor.dtype = mslite.DataType.FLOAT32
        tensor.shape = [2, 3]
        in_data = np.arange(2 * 3, dtype=np.int32).reshape((2, 3))
        tensor.set_data_from_numpy(in_data)
        assert "data type bot equal" in str(raise_info.value)


def test_python_api_fi_tensor_numpy_001():
    with pytest.raises(TypeError) as raise_info:
        mslite.Tensor("abc")
    assert "tensor must be MindSpore Lite's Tensor._tensor or numpy ndarray" in str(raise_info.value)
