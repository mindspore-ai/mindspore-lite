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
Test bfloat16 inference.
"""
import mindspore_lite as mslite
import ml_dtypes
import numpy as np

MODEL_FILE = "matmul_bf16.onnx.mindir"


def test_python_bf16():
    """Test matmul inference with bfloat16 inputs on Ascend."""
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = 0
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
    x = np.ones((2, 4), dtype=ml_dtypes.bfloat16)
    y = np.ones((4, 3), dtype=ml_dtypes.bfloat16)
    out_np = x @ y
    inputs = model.get_inputs()
    inputs[0].set_data_from_numpy(x)
    inputs[1].set_data_from_numpy(y)
    out_mslite = model.predict(inputs)[0]
    assert inputs[0].dtype == mslite.DataType.BFLOAT16
    assert inputs[1].dtype == mslite.DataType.BFLOAT16
    assert out_mslite.dtype == mslite.DataType.BFLOAT16
    out_mslite_npy = out_mslite.get_data_to_numpy()
    err = np.mean(np.abs(out_np - out_mslite_npy) / (np.abs(out_np) + 1e-6))
    assert err < 0.01
