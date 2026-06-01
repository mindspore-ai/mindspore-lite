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
Test for MindSpore Lite update_weights
"""

import os
import pytest
import mindspore_lite as mslite
import numpy as np

MODEL_FILE = "./single_matmul_model.onnx.mindir"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def test_update_weight_resul_change():
    '''
    test inference result changed after update weight
    '''
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    outputs_nolora = model.predict(ms_inputs)[0].get_data_to_numpy()
    weight = np.ones((4, 4), dtype=np.float32)
    tensor = mslite.Tensor(weight)
    model.update_weights([[tensor]])
    outputs_lora = model.predict(ms_inputs)[0].get_data_to_numpy()
    assert not np.allclose(outputs_nolora, outputs_lora)

def test_update_weight_multiple_times():
    '''
    test update weight multi time
    '''
    try:
        model = mslite.Model()
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
        weight = np.ones((4, 4), dtype=np.float32)
        tensor = mslite.Tensor(weight)
        for _ in range(5):
            model.update_weights([[tensor]])
    except Exception as exc:
        raise RuntimeError("test update weight multiple times failed!") from exc

def test_update_weight_zero_copy():
    '''
    test update weight use zero copy
    '''
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    weight = np.ones((4, 4), dtype=np.float32)
    tensor = mslite.Tensor(tensor=weight, device="ascend:"+str(DEVICE_ID))
    model.update_weights([[tensor]])
    outputs_lora = model.predict(ms_inputs)[0].get_data_to_numpy()
    lora_out = np.ones((1, 4), dtype=np.float32) @ np.ones((4, 4), dtype=np.float32)
    assert np.mean(lora_out-outputs_lora) < 1e-5

def test_update_weight_precision():
    '''
    test precision after update weight
    '''
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    weight = np.ones((4, 4), dtype=np.float32)
    tensor = mslite.Tensor(weight)
    model.update_weights([[tensor]])
    outputs_lora = model.predict(ms_inputs)[0].get_data_to_numpy()
    lora_out = np.ones((1, 4), dtype=np.float32) @ np.ones((4, 4), dtype=np.float32)
    assert np.mean(lora_out-outputs_lora) < 1e-5

def test_update_weight_empty_weight():
    '''
    test update empty weight
    '''
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    with pytest.raises(RuntimeError) as e:
        model.update_weights([[]])
    assert "update weight failed! Error is Common error code" in str(e.value)

def test_update_weight_mindir(mindir_dir, so_path, output_dir, config_dir):
    '''
    test update weight for mindir model
    '''
    model_path = os.path.join(mindir_dir, "linear.mindir")
    fmk_type = "MINDIR"
    config_path = os.path.join(config_dir, "linear.mindir.config")
    output_path = os.path.join(output_dir, "linear_lite")
    cmd_string = so_path + "/tools/converter/converter/converter_lite " + \
                    " --modelFile=" + model_path + \
                    " --optimize=ascend_oriented " + \
                    " --outputFile=" + output_path + \
                    " --fmk=" + fmk_type + \
                    " --configFile=" + config_path
    ret = os.system(cmd_string)
    if ret != 0:
        raise RuntimeError("model convert failed, cmd_string is: ", cmd_string)
    try:
        model = mslite.Model()
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = DEVICE_ID
        model.build_from_file(output_path+".mindir", mslite.ModelType.MINDIR, context)
        weight = mslite.Tensor(np.random.randn(64,128).astype(np.float32))
        model.update_weights([[weight]])
    except Exception as exc:
        raise RuntimeError('update weight for mindir model failed!') from exc
