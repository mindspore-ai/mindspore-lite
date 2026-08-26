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
import os
import mindspore_lite as mslite
import numpy as np

MODEL_FILE = "./ge_test_mul.mindir"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def test_ge_zerocopy_inference(config_dir):
    '''
    test zerocopy inference.
    '''
    config_file = os.path.join(config_dir, 'ge_test_mul_dynamic_bucket.config')
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    context.ascend.provider = "ge"
    model = mslite.Model()
    model.build_from_file(model_path=MODEL_FILE, model_type=mslite.ModelType.MINDIR, context=context,
                          config_path=config_file)
    model.resize(model.get_inputs(), [[1, 128], [1, 128]])

    x = np.random.randn(1, 128).astype(np.float32)
    res_normal = model.predict([x, x])[0].get_data_to_numpy()

    inputs = [mslite.Tensor(shape=[1,128],dtype=mslite.DataType.FLOAT32,device="ascend:"+str(DEVICE_ID)),
        mslite.Tensor(shape=[1,128],dtype=mslite.DataType.FLOAT32,device="ascend:"+str(DEVICE_ID))]
    outputs = [mslite.Tensor(shape=[1,128],dtype=mslite.DataType.FLOAT32,device="ascend:"+str(DEVICE_ID))]
    inputs[0].set_data_from_numpy(x)
    inputs[1].set_data_from_numpy(x)
    res_zerocopy = model.predict(inputs, outputs)[0].get_data_to_numpy()
    assert np.allclose(res_normal, res_zerocopy)
    res_zerocopy = model.predict(inputs, outputs)[0].get_data_to_numpy()
    assert np.allclose(res_normal, res_zerocopy)
