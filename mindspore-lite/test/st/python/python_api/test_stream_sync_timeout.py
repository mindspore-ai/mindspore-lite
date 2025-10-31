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
Test lite python API.
"""
import os
from collections import namedtuple
import pytest
import mindspore_lite as mslite
import numpy as np
from utils import ScopeTimeRecord, expect_error


MODEL_BASE_PATH = "."

STREAM_SYNC_TIMEOUT_CONFIG_DICT_LIMITED = {
    "ascend_context": {"timeout": "3"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_UNLIMITED = {
    "ascend_context": {"timeout": "-1"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_1 = {
    "ascend_context": {"timeout": "0"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_2 = {
    "ascend_context": {"timeout": "-2"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_3 = {
    "ascend_context": {"timeout": "-2147483648"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_4 = {
    "ascend_context": {"timeout": "-2147483649"},
}

STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_5 = {
    "ascend_context": {"timeout": "2147483648"},
}

ConfigAndWillError = namedtuple("ConfigAndWillError", ["config", "build_error", "infer_error"])


def _create_context(device_id):
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = device_id
    return context


def _run_case_with_config(model_path, device_id, model_input, model_output, config_dict, build_error, infer_error):
    """
    run case with a config
    """
    print(f"run case with config: {config_dict}, expect: {build_error} {infer_error}", flush=True)
    context = _create_context(device_id)
    model = mslite.Model()

    with expect_error(build_error):
        model.build_from_file(model_path, mslite.ModelType.MINDIR, context, config_dict=config_dict)

    with expect_error(infer_error):
        with ScopeTimeRecord() as record:
            model.predict(model_input, model_output)

    print(f"model predict time with config {config_dict}: {record.duration} ms", flush=True)


@pytest.mark.parametrize(
    "model_name, inputs, outputs",
    (
        (
            "sd1.5_unet.onnx_graph.mindir",
            (
                np.ones((2, 4, 64, 64)).astype(np.float32),
                np.ones((1,)).astype(np.float32),
                np.ones((2, 77, 768)).astype(np.float32),
            ),
            (np.ones((2, 4, 64, 64)).astype(np.float32),),
        ),
    ),
)
@pytest.mark.backend("mslite_large_model_inference_arm_ascend910B")
def test_stream_sync_timeout(model_name, inputs, outputs, device_id):
    """
    test config stream_sync_timeout
    """
    model_path = os.path.join(MODEL_BASE_PATH, model_name)

    model_input = [mslite.Tensor(tensor=i, device=f"ascend:{device_id[0]}") for i in inputs]
    model_output = [mslite.Tensor(tensor=o, device=f"ascend:{device_id[0]}") for o in outputs]

    # None * 3 for warm up
    case_list = [ConfigAndWillError(None, None, None)] * 3 + [
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_UNLIMITED, None, None),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_LIMITED, None, RuntimeError),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_1, RuntimeError, RuntimeError),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_2, RuntimeError, RuntimeError),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_3, RuntimeError, RuntimeError),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_4, RuntimeError, RuntimeError),
        ConfigAndWillError(STREAM_SYNC_TIMEOUT_CONFIG_DICT_INVALID_5, RuntimeError, RuntimeError),
    ]

    for case in case_list:
        _run_case_with_config(model_path, device_id[0], model_input, model_output, *case)
