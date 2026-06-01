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

import os

import mindspore_lite as mslite
import numpy as np

MODEL_FILE_1 = "./sd1.5_unet.onnx_graph.mindir"
MODEL_FILE_2 = "./sd1.5_unet.onnx_graph.mindir"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def model_infer(model_path):
    """
    model infer func
    """
    # init mindspore lite context
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR, context)

    input_data_1 = np.random.rand(2, 4, 96, 96).astype(np.float32)
    input_data_2 = np.random.rand(1).astype(np.float32)
    input_data_3 = np.random.rand(2, 77, 768).astype(np.float32)

    model.resize(model.get_inputs(), [[2, 4, 96, 96], [1], [2, 77, 768]])
    model.predict([input_data_1, input_data_2, input_data_3])


def test_model_group_for_shared_work_space():
    """
    test shared work space
    """
    # init model group
    model_group_context = mslite.Context()
    model_group_context.target = ["ascend"]
    model_group_context.ascend.device_id = DEVICE_ID
    model_group = mslite.ModelGroup()
    model_group.add_model([MODEL_FILE_1, MODEL_FILE_2])
    model_group.cal_max_size_of_workspace(mslite.ModelType.MINDIR, model_group_context)
    # model inference
    model_infer(MODEL_FILE_1)
    model_infer(MODEL_FILE_2)


def test_model_group_for_shared_weight_space():
    """
    test shared weight space
    """
    # init model group
    model_group_context = mslite.Context()
    model_group_context.target = ["ascend"]
    model_group_context.ascend.device_id = DEVICE_ID
    model_group = mslite.ModelGroup(mslite.ModelGroupFlag.SHARE_WEIGHT_WORKSPACE)
    model_group.add_model([MODEL_FILE_1, MODEL_FILE_2])
    model_group.cal_max_size_of_workspace(mslite.ModelType.MINDIR, model_group_context)
    # model inference
    model_infer(MODEL_FILE_1)
    model_infer(MODEL_FILE_2)
