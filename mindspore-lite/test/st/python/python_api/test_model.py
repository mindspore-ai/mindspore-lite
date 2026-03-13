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

import pytest
import mindspore_lite as mslite
import numpy as np

# ----------- config file -----------
# [acl_build_options]
# input_format="ND"
# input_shape="sample:2,4,-1,-1;timestep:1;encoder_hidden_states:2,77,768"
# ge.dynamicDims="64,64;96,96"

MODEL_FILE = "./sd1.5_unet.onnx_graph.mindir"
MODEL_STATIC_FILE = "./single_matmul_model.onnx.mindir"
MODEL_DYNAMIC_FILE = "./resize.onnx.mindir"
DEVICE_ID = 0


# For Model Resize
def test_python_api_resize_001():
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    model.resize(model.get_inputs(), [[2, 4, 64, 64], [1], [2, 77, 768]])


def test_python_api_fi_resize_001():
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
        model.resize(model.get_inputs(), [[10, 4, 64, 64], [1], [2, 77, 768]])
    assert "resize failed! Error is The given input shape's value != original input shape's value." \
            in str(raise_info.value)


def test_python_api_fi_resize_002():
    with pytest.raises(RuntimeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
        model.resize(model.get_inputs(), [[2, 4, -1, -1], [1], [2, 77, 768]])
    assert "resize failed! Error is Invalid shape!" in str(raise_info.value)

def test_python_api_resize_003():
    """
    test output shape after resize for Dynamic Binning Model
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    inputs = model.get_inputs()
    input_shapes = [inp.shape for inp in inputs]
    assert (input_shapes == [[2, 4, -1, -1], [1], [2, 77, 768]])
    outputs = model.get_outputs()
    output_shapes = [out.shape for out in outputs]
    assert (output_shapes == [[2,4,96,96]])
    model.resize(model.get_inputs(), [[2, 4, 64, 64], [1], [2, 77, 768]])
    inputs = model.get_inputs()
    input_shapes = [inp.shape for inp in inputs]
    assert (input_shapes == [[2, 4, 64, 64], [1], [2, 77, 768]])
    outputs = model.get_outputs()
    output_shapes = [out.shape for out in outputs]
    assert (output_shapes == [[2,4,64, 64]])

def test_python_api_resize_004():
    """
    test output shape after resize for static input model
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_STATIC_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    inputs = model.get_inputs()
    input_shapes = [inp.shape for inp in inputs]
    assert (input_shapes == [[1, 4]])
    outputs = model.get_outputs()
    output_shapes = [out.shape for out in outputs]
    assert (output_shapes == [[1,4]])

def test_python_api_resize_005():
    """
    test output shape after resize for dynamic input model
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_DYNAMIC_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    inputs = model.get_inputs()
    input_shapes = [inp.shape for inp in inputs]
    assert (input_shapes == [[-1,3,512,512]])
    outputs = model.get_outputs()
    output_shapes = [out.shape for out in outputs]
    assert output_shapes == [[-1]]
    model.resize(model.get_inputs(),[[1,3,512,512]])
    inputs = model.get_inputs()
    input_shapes = [inp.shape for inp in inputs]
    assert (input_shapes == [[1,3,512,512]])
    outputs = model.get_outputs()
    output_shapes = [out.shape for out in outputs]
    assert output_shapes == [[-1]]

def test_python_api_func_resize_random_shape_001():
    """
    test python api func resize random shape 001
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)

    # data for 64
    input_1_data_1 = np.random.rand(2, 4, 64, 64).astype(np.float32)
    input_2_data_1 = np.random.rand(1).astype(np.float32)
    input_3_data_1 = np.random.rand(2, 77, 768).astype(np.float32)
    # data for 96
    input_1_data_2 = np.random.rand(2, 4, 96, 96).astype(np.float32)
    input_2_data_2 = np.random.rand(1).astype(np.float32)
    input_3_data_2 = np.random.rand(2, 77, 768).astype(np.float32)

    # predict for shape of 64
    model.resize(model.get_inputs(), [[2, 4, 64, 64], [1], [2, 77, 768]])
    model.predict([input_1_data_1, input_2_data_1, input_3_data_1])
    # predict for shape of 96
    model.resize(model.get_inputs(), [[2, 4, 96, 96], [1], [2, 77, 768]])
    model.predict([input_1_data_2, input_2_data_2, input_3_data_2])
    # predict for shape of 64
    model.resize(model.get_inputs(), [[2, 4, 64, 64], [1], [2, 77, 768]])
    model.predict([input_1_data_1, input_2_data_1, input_3_data_1])


def test_python_api_func_resize_random_shape_002():
    """
    test python api func resize random shape 002
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)

    # data for 64
    input_1_data_1 = np.random.rand(2, 4, 64, 64).astype(np.float32)
    input_2_data_1 = np.random.rand(1).astype(np.float32)
    input_3_data_1 = np.random.rand(2, 77, 768).astype(np.float32)
    # data for 96
    input_1_data_2 = np.random.rand(2, 4, 96, 96).astype(np.float32)
    input_2_data_2 = np.random.rand(1).astype(np.float32)
    input_3_data_2 = np.random.rand(2, 77, 768).astype(np.float32)

    # predict for shape of 96
    model.resize(model.get_inputs(), [[2, 4, 96, 96], [1], [2, 77, 768]])
    model.predict([input_1_data_2, input_2_data_2, input_3_data_2])
    # predict for shape of 64
    model.resize(model.get_inputs(), [[2, 4, 64, 64], [1], [2, 77, 768]])
    model.predict([input_1_data_1, input_2_data_1, input_3_data_1])
    # predict for shape of 96
    model.resize(model.get_inputs(), [[2, 4, 96, 96], [1], [2, 77, 768]])
    model.predict([input_1_data_2, input_2_data_2, input_3_data_2])
