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
Test inference with zerocopy.
"""
import mindspore_lite as mslite

MODEL_FILE = "./single_matmul_model.onnx.mindir"
DEVICE_ID = 0


def test_zerocopy_inference():
    '''
    test zerocopy inference.
    '''
    try:
        for _ in range(2):
            context = mslite.Context()
            context.target = ["ascend"]
            context.ascend.device_id = DEVICE_ID
            model = mslite.Model()
            model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context)
            inputs = [mslite.Tensor(shape=[1,4],dtype=mslite.DataType.FLOAT32,device="ascend:"+str(DEVICE_ID))]
            outputs = [mslite.Tensor(shape=[1,4],dtype=mslite.DataType.FLOAT32,device="ascend:"+str(DEVICE_ID))]
            model.predict(inputs, outputs)
    except Exception as exc:
        raise RuntimeError("test zerocopy inference failed!") from exc
