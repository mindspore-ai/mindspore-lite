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
Test for MindSpore Lite encrypt_and_decrypt
"""

import mindspore_lite as mslite
import numpy as np

MODEL_FILE = "./single_matmul_model.onnx"
ENCRYPT_MINDIR_MODEL_FILE = "encrypt_single_matmul_model"
DECRYPT_KEY = "30313233343637383939414243444546"
DECRYPT_MODE = "AES-GCM"
ENCRYPT_KEY = "30313233343637383939414243444546"
DEC_NUM_PARALLEL = 64

def test_convert_encrypt_and_infer_decrypt_model():
    """
    test convert encrypt and infer decrypt model
    """
    # convert and encrypt model
    converter = mslite.Converter()
    converter.input_shape = {"input": [1, 4]}
    converter.input_data_type = mslite.DataType.FLOAT32
    converter.output_data_type = mslite.DataType.FLOAT32
    converter.save_type = mslite.ModelType.MINDIR
    converter.enable_encryption = True
    converter.encrypt_key = ENCRYPT_KEY
    converter.infer = True
    converter.optimize = "general"
    converter.save_type = mslite.ModelType.MINDIR
    converter.convert(mslite.FmkType.ONNX, MODEL_FILE, ENCRYPT_MINDIR_MODEL_FILE)
    # decrypt and predict model
    model = mslite.Model()
    context = mslite.Context()
    context.target = ["cpu"]
    dec_key = bytes.fromhex(DECRYPT_KEY)
    model.build_from_file(model_path = ENCRYPT_MINDIR_MODEL_FILE + ".mindir",
                          model_type = mslite.ModelType.MINDIR, context = context,
                          dec_key = dec_key, dec_mode = DECRYPT_MODE,
                          dec_num_parallel = DEC_NUM_PARALLEL)
    np_input = np.ones((1, 4), dtype=np.float32)
    ms_inputs = model.get_inputs()
    ms_inputs[0].set_data_from_numpy(np_input)
    out = model.predict(ms_inputs)
    assert out[0].shape == [1, 4]
