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
Test for MindSpore Lite preinference
"""

import os
import mindspore_lite as mslite
import numpy as np

DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))

def test_optimizer_01_convert_none(mindir_dir, output_dir):
    '''
    test convert opeimizer none model
    '''
    converter = mslite.converter.Converter()
    converter.save_type = mslite.ModelType.MINDIR
    converter.optimize = "none"
    converter.input_shape = {"input": [1, 4]}
    try:
        converter.convert(mslite.FmkType.ONNX, os.path.join(mindir_dir, "single_matmul_model.onnx"),
                          os.path.join(output_dir,"single_matmul_model_none"))
    except Exception as exc:
        raise RuntimeError('convert model optimize none failed!') from exc

def test_optimizer_02_convert_general(mindir_dir, output_dir):
    '''
    test convert general model
    '''
    converter = mslite.converter.Converter()
    converter.save_type = mslite.ModelType.MINDIR
    converter.optimize = "general"
    converter.input_shape = {"input": [1, 4]}
    try:
        converter.convert(mslite.FmkType.ONNX, os.path.join(mindir_dir, "single_matmul_model.onnx"),
                          os.path.join(output_dir,"single_matmul_model_general"))
    except Exception as exc:
        raise RuntimeError('convert model optimize general failed!') from exc

def test_optimizer_03_convert_mindir_lite(mindir_dir, output_dir):
    '''
    test convert mindir_lite model
    '''
    converter = mslite.converter.Converter()
    converter.save_type = mslite.ModelType.MINDIR_LITE
    converter.optimize = "ascend_oriented"
    converter.input_shape = {"input": [1, 4]}
    try:
        converter.convert(mslite.FmkType.ONNX, os.path.join(mindir_dir, "single_matmul_model.onnx"),
                          os.path.join(output_dir,"single_matmul_model_lite"))
    except Exception as exc:
        raise RuntimeError('convert model mindir_lite failed!') from exc

def test_optimizer_04_inference_none(output_dir):
    '''
    test inference optimize none model
    '''
    model_path = os.path.join(output_dir,"single_matmul_model_none.mindir")
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model.build_from_file(model_path=model_path, model_type=mslite.ModelType.MINDIR, context=context)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    outputs = model.predict(ms_inputs)[0].get_data_to_numpy()
    correct_result = np.ones((1, 4), dtype=np.float32) @ np.ones((4, 4), dtype=np.float32)
    assert np.mean(correct_result-outputs) < 1e-5

def test_optimizer_05_inference_general(output_dir):
    '''
    test inference optimize general model
    '''
    model_path = os.path.join(output_dir,"single_matmul_model_general.mindir")
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["CPU"]
    model.build_from_file(model_path=model_path, model_type=mslite.ModelType.MINDIR, context=context)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    outputs = model.predict(ms_inputs)[0].get_data_to_numpy()
    correct_result = np.ones((1, 4), dtype=np.float32) @ np.ones((4, 4), dtype=np.float32)
    assert np.mean(correct_result-outputs) < 1e-5
