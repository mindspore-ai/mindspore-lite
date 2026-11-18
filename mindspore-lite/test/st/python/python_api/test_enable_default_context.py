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
Test for the ascend_context enable_default_context option of the Ascend ACL backend.
"""

import gc
import os

import numpy as np
import mindspore_lite as mslite

MODEL_FILE = "./single_matmul_model.onnx.mindir"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))
INPUT_SHAPE = (1, 4)
CONFIG_DEFAULT_CONTEXT_ON = {"ascend_context": {"enable_default_context": "on"}}
CONFIG_DEFAULT_CONTEXT_OFF = {"ascend_context": {"enable_default_context": "off"}}


def _create_model(config_dict):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    model = mslite.Model()
    model.build_from_file(MODEL_FILE, mslite.ModelType.MINDIR, context, config_dict=config_dict)
    return model


def _predict(model, data):
    return model.predict([data])[0].get_data_to_numpy()


def test_enable_default_context_keeps_result():
    """
    enable_default_context=on must not change the inference result.
    """
    data = np.random.random(INPUT_SHAPE).astype(np.float32)
    expected = _predict(_create_model(CONFIG_DEFAULT_CONTEXT_OFF), data)
    actual = _predict(_create_model(CONFIG_DEFAULT_CONTEXT_ON), data)
    assert np.allclose(actual, expected, rtol=0, atol=1e-6)


def test_enable_default_context_shared_between_models():
    """
    Models with enable_default_context=on share the caller's default context, so
    finalizing one model must not invalidate the context used by the others.
    """
    data = np.random.random(INPUT_SHAPE).astype(np.float32)
    expected = _predict(_create_model(CONFIG_DEFAULT_CONTEXT_OFF), data)
    first = _create_model(CONFIG_DEFAULT_CONTEXT_ON)
    second = _create_model(CONFIG_DEFAULT_CONTEXT_ON)
    assert np.allclose(_predict(first, data), expected, rtol=0, atol=1e-6)

    del first
    gc.collect()
    assert np.allclose(_predict(second, data), expected, rtol=0, atol=1e-6)


def test_enable_default_context_repeated_build():
    """
    Rebuilding a model after finalizing the previous one must keep working: the
    shared default context is not destroyed by Finalize.
    """
    data = np.random.random(INPUT_SHAPE).astype(np.float32)
    expected = _predict(_create_model(CONFIG_DEFAULT_CONTEXT_OFF), data)
    for _ in range(2):
        model = _create_model(CONFIG_DEFAULT_CONTEXT_ON)
        assert np.allclose(_predict(model, data), expected, rtol=0, atol=1e-6)
        del model
        gc.collect()
