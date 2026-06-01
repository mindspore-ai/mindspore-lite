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
Test for MindSpore Lite Profiling of Ascend ALC
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
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def test_acl_profiling_with_config_file():
    """
    test profiling with config file
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    profiling_config = {"ascend_context": {"profiling_config_file": "./prof.json"}}
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                          config_dict=profiling_config)

    input_data_1 = np.random.rand(2, 4, 96, 96).astype(np.float32)
    input_data_2 = np.random.rand(1).astype(np.float32)
    input_data_3 = np.random.rand(2, 77, 768).astype(np.float32)

    model.resize(model.get_inputs(), [[2, 4, 96, 96], [1], [2, 77, 768]])
    model.predict([input_data_1, input_data_2, input_data_3])
    path_list = os.listdir("./profiling")
    assert len(path_list) == 1
    prof_path = os.path.join("./profiling", path_list[0])
    prof_file_list = os.listdir(prof_path)
    assert len(prof_file_list) == 2
    assert "device_" + str(DEVICE_ID) in prof_file_list
    assert "host" in prof_file_list


def test_acl_profiling_without_config_file():
    """
    test profiling without config file
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    profiling_config = {"ascend_context": {"profiling_config_file": "./xx.json"}}
    with pytest.raises(RuntimeError) as raise_info:
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                              config_dict=profiling_config)
    assert "build_from_file failed! Error is Profiling init failed, please check your file." \
           in str(raise_info.value)
