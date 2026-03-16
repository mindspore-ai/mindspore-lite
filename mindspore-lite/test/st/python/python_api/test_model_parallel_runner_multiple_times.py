# Copyright 2026 Huawei Technologies Co., Ltd
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
Test for MindSpore Lite ModelParallelRunner (repeat predict many times).
"""

# pylint: disable=invalid-name
import os
import numpy as np
import mindspore_lite as mslite


def GenRandData(shape, dtype):
    size = 1
    for ele in shape:
        size *= ele
    data = np.random.rand(size).reshape(shape).astype(dtype)
    return data


class MSLiteModel:
    """Minimal wrapper for ModelParallelRunner, aligned with test_model_origin.py style."""

    def __init__(self, model_path):
        self.model_path = model_path
        context = mslite.Context()
        context.target = ["ascend"]
        # self.model = mslite.Model()
        self.model = mslite.ModelParallelRunner()
        self.model.build_from_file(model_path, context)

    def Predict(self, input_data):
        """Set inputs and run predict once, returning outputs as numpy arrays."""
        model_inputs = self.model.get_inputs()
        if len(model_inputs) != len(input_data):
            raise RuntimeError("input data size is not equal model input size")
        for i in range(len(model_inputs)):
            # ModelParallelRunner has no resize() in python API; set input tensor shape explicitly.
            model_inputs[i].shape = list(input_data[i].shape)
            model_inputs[i].set_data_from_numpy(input_data[i])
        outputs = self.model.predict(model_inputs)
        print(len(outputs))
        np_out = []
        for out in outputs:
            np_out.append(out.get_data_to_numpy())
        return np_out


def test_python_api_func_parallel_001(output_dir, mindir_dir):
    """
    Use a multi-output ONNX model from HIAI: deepaudio.onnx (3 outputs).
    Shape is from mindspore-lite/test/config_level0/models_server_inference.cfg: 5,80,80.
    """
    model_file = os.path.join(output_dir, "deepaudio.onnx.mindir") if output_dir else "deepaudio.onnx.mindir"
    if not os.path.exists(model_file) and mindir_dir:
        model_file = os.path.join(mindir_dir, "deepaudio.onnx.mindir")
    ms_model = MSLiteModel(model_file)

    dtype = np.float32
    data0 = GenRandData([5, 80, 80], dtype)
    run_times = 200
    for _ in range(run_times):
        model_outputs = ms_model.Predict([data0])
        assert len(model_outputs) == 3
