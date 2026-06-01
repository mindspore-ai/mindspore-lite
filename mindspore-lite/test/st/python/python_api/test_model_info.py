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
Test for MindSpore Lite Base Inference
"""

import os
import pytest
import mindspore_lite as mslite

MODEL_FILE = "./sd1.5_unet.onnx_graph.mindir"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


# ----------- config file -----------
# [acl_build_options]
# input_format="ND"
# input_shape="sample:2,4,-1,-1;timestep:1;encoder_hidden_states:2,77,768"
# ge.dynamicDims="64,64;96,96"

def test_runtime_general_model_info_func_ascend_001():
    """
    Feature: test runtime general model info
    Description: test get_model_info with input_shape
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    empty_shape = model.get_model_info("input_shape")
    assert empty_shape == ""
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    input_shape_config = model.get_model_info("input_shape")
    assert input_shape_config == "sample:2,4,-1,-1;timestep:1;encoder_hidden_states:2,77,768"
    dynamic_dums_config = model.get_model_info("dynamic_dims")
    assert dynamic_dums_config == "64,64;96,96"


def test_python_api_fi_get_model_info_001():
    with pytest.raises(TypeError) as raise_info:
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        model = mslite.Model()
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
        model.get_model_info(1)
    assert "key must be str" in str(raise_info.value)
