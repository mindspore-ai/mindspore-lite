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
import subprocess
from pathlib import Path
from typing import List
import pytest
import numpy as np
import mindspore_lite as mslite

MODEL_PATH1: str = ""
DIM_IN1 = [[1, 3, 64, 64], [1, 1, 64, 64], [1, 1, 1, 1]]
DIM_OUT1 = [[1, 6, 1, 1], [1, 256, 2, 2], [1, 256, 2, 2]]

MODEL_PATH2: str = ""
DIM_IN2 = [1, 3, 512, 512]
DIM_OUT2 = [1, 3, 512, 512]

MODEL_PATH3: str = ""
DIM_IN3 = [1, 3, 512, 512]
DIM_OUT3 = [1, 3, 512, 512]

DEVICE_ID = 0
ResultList = List[List[np.ndarray]]


def _convert_onnx_to_mindir(model_name: str, so_path: Path, mindir_dir: Path, output_dir: Path) -> Path:
    '''
    convert model from onnx to mindir
    '''
    model_path = Path(mindir_dir) / model_name
    output_path = Path(output_dir) / model_name
    fmk = "ONNX"
    cmd = [
        Path(so_path) / "tools/converter/converter/converter_lite",
        "--optimize=none",
        f"--modelFile={model_path}",
        f"--outputFile={output_path}",
        f"--fmk={fmk}",
        "--saveType=MINDIR",
    ]
    subprocess.run(cmd, check=True)
    return Path(str(output_path) + ".mindir")


@pytest.fixture(scope="module", autouse=True)
def setup_model_paths(so_path, mindir_dir, output_dir):
    """
    Automatically converts raw ONNX models to MindIR format and assigns paths
    to global MODEL_PATHx variables before running tests.
    Cleans up the converted files after tests are done.
    """
    global MODEL_PATH1, MODEL_PATH2, MODEL_PATH3
    models = [
        "cloud-assistant_device_awb.onnx",
        "cloud-assistant_device_X_pipeline__facebeauty__skin_dodge_burn_gpenclean.onnx",
        "cloud-assistant_device_X_pipeline_facebeauty_TeethWhite_260000.onnx"
    ]
    MODEL_PATH1 = str(_convert_onnx_to_mindir(models[0], so_path, mindir_dir, output_dir))
    MODEL_PATH2 = str(_convert_onnx_to_mindir(models[1], so_path, mindir_dir, output_dir))
    MODEL_PATH3 = str(_convert_onnx_to_mindir(models[2], so_path, mindir_dir, output_dir))
    yield
    Path(MODEL_PATH1).unlink(missing_ok=True)
    Path(MODEL_PATH2).unlink(missing_ok=True)
    Path(MODEL_PATH3).unlink(missing_ok=True)


def _create_context(provider):
    """
    Creates a MindSpore Lite context for Ascend device with a specific provider.
    Args:
        provider (str): Backend provider, e.g., "ge" or "ge-v1".
    Returns:
        mslite.Context: Configured context object.
    """
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.device_id = DEVICE_ID
    context.ascend.provider = provider
    return context


def _create_inputs_outputs():
    np_input1 = np.random.random((1, 3, 64, 64)).astype(np.float32)
    np_input2 = np.random.random((1, 1, 64, 64)).astype(np.float32)
    np_input3 = np.random.random((1, 1, 1, 1)).astype(np.float32)
    np_output1 = np.random.random((1, 6, 1, 1)).astype(np.float32)
    np_output2 = np.random.random((1, 256, 2, 2)).astype(np.float32)
    np_output3 = np.random.random((1, 256, 2, 2)).astype(np.float32)
    return [[np_input1, np_input2, np_input3], [np_output1, np_output2, np_output3]]


def _common_functional_accuracy(outputs_type: str):
    """
    Common accuracy test for model inference with different output configurations.
    Args:
        outputs_type (str): A string of '0', '1', or '2' indicating:
            '0' = null output tensor,
            '1' = host output tensor,
            '2' = device output tensor.
    """
    model_ge_v1 = mslite.Model()
    context_ge_v1 = _create_context("ge-v1")
    model_ge_v1.build_from_file(model_path=MODEL_PATH1, model_type=mslite.ModelType.MINDIR, context=context_ge_v1)

    model_ge = mslite.Model()
    context_ge = _create_context("ge")
    model_ge.build_from_file(model_path=MODEL_PATH1, model_type=mslite.ModelType.MINDIR, context=context_ge)

    inputs_outputs = _create_inputs_outputs()

    loop = 3
    result_ge = []
    for i in range(loop):
        inputs_ge = [
            mslite.Tensor(tensor=inputs_outputs[0][0].copy(), shape=DIM_IN1[0], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[0][1].copy(), shape=DIM_IN1[1], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[0][2].copy(), shape=DIM_IN1[2], dtype=mslite.DataType.FLOAT32)
        ]
        outputs = [
            mslite.Tensor(tensor=inputs_outputs[1][0].copy(), shape=DIM_OUT1[0], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[1][1].copy(), shape=DIM_OUT1[1], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[1][2].copy(), shape=DIM_OUT1[2], dtype=mslite.DataType.FLOAT32)
        ]
        model_ge.predict(inputs_ge, outputs)
        result_ge.append([t.get_data_to_numpy() for t in outputs])

    result_ge_v1 = []
    for i in range(loop):
        outputs = []
        inputs_ge_v1 = [
            mslite.Tensor(tensor=inputs_outputs[0][0].copy(), shape=DIM_IN1[0], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[0][1].copy(), shape=DIM_IN1[1], dtype=mslite.DataType.FLOAT32,
                          device="ascend:" + str(DEVICE_ID)),
            mslite.Tensor(tensor=inputs_outputs[0][2].copy(), shape=DIM_IN1[2], dtype=mslite.DataType.FLOAT32,
                          device="ascend:" + str(DEVICE_ID))
        ]
        if outputs_type:
            for j, char in enumerate(outputs_type):
                if char == '0':
                    tensor = mslite.Tensor(shape=DIM_OUT1[j], dtype=mslite.DataType.FLOAT32)
                elif char == '1':
                    tensor = mslite.Tensor(tensor=inputs_outputs[1][j].copy(), shape=DIM_OUT1[j],
                                           dtype=mslite.DataType.FLOAT32)
                elif char == '2':
                    tensor = mslite.Tensor(tensor=inputs_outputs[1][j].copy(),
                                           shape=DIM_OUT1[j],
                                           dtype=mslite.DataType.FLOAT32,
                                           device="ascend:" + str(DEVICE_ID))
                else:
                    raise ValueError(f"invalid: '{char}',only: '0', '1', or '2'.")
                outputs.append(tensor)

        model_ge_v1.predict(inputs_ge_v1, outputs)
        result_ge_v1.append([t.get_data_to_numpy() for t in outputs])

    print("test result: ")
    for of, ob in zip(result_ge, result_ge_v1):
        for f, b in zip(of, ob):
            np.testing.assert_allclose(f, b)

    print("Common single-model accuracy verification passed.")
    return True


def test_inputs_host_device_outputs_null():
    '''
    test inputs are on host and device, outputs are []
    '''
    result = _common_functional_accuracy("")
    if result:
        print("test_inputs_host_device_outputs_data_null test success!")


def test_inputs_host_device_outputs_data_null():
    '''
    test inputs are on host and device, outputs's data is [null, null, null]
    '''
    result = _common_functional_accuracy("000")
    if result:
        print("test_inputs_host_device_outputs_data_null test success!")


def test_inputs_host_device_outputs_data_device():
    '''
    test inputs are on host and device, outputs's data is on [device, device, device]
    '''
    result = _common_functional_accuracy("222")
    if result:
        print("test_inputs_host_device_outputs_data_device test success")


def test_inputs_host_device_outputs_data_host():
    '''
    test inputs are on host and device, outputs's data is on [host, host, host]
    '''
    result = _common_functional_accuracy("111")
    if result:
        print("test_inputs_host_device_outputs_data_host test success!")


def test_inputs_host_device_outputs_host_device():
    '''
    test inputs are on host and device, outputs's data is on [host, device, device]
    '''
    result = _common_functional_accuracy("122")
    if result:
        print("test_inputs_host_device_outputs_host_device test success!")


def test_inputs_host_device_outputs_data_null_host_device():
    '''
    test inputs are on host and device, outputs's data is on [null, host, device]
    '''
    result = _common_functional_accuracy("012")
    if result:
        print("test_inputs_host_device_outputs_data_null_host_device test success!")


def test_single_model_autoregression_performance():
    '''
    test Single-Model Autoregression performance
    '''
    model = mslite.Model()
    context = _create_context("ge-v1")
    model.build_from_file(model_path=MODEL_PATH2, model_type=mslite.ModelType.MINDIR, context=context)

    np_input = np.random.random((1, 3, 512, 512)).astype(np.float32)
    np_output = np.random.random((1, 3, 512, 512)).astype(np.float32)
    input_host = mslite.Tensor(tensor=np_input, shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)

    output_device1 = mslite.Tensor(tensor=np_output, shape=DIM_IN2,
                                   dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))
    output_device2 = mslite.Tensor(tensor=np_output, shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)
    outputs_host1 = mslite.Tensor(tensor=np_output, shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)
    outputs_host2 = mslite.Tensor(tensor=np_output, shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)

    loop = 3
    for i in range(loop):
        model.predict([input_host], [output_device1])
        model.predict([output_device1], [output_device2])
    for i in range(loop):
        model.predict([input_host], [outputs_host1])
        model.predict([outputs_host1], [outputs_host2])

def test_double_model_autoregression_performance():
    '''
    test Multi-Model Performance Evaluation with Single-Model Autoregression
    '''
    model1 = mslite.Model()
    context = _create_context("ge-v1")
    model1.build_from_file(model_path=MODEL_PATH2, model_type=mslite.ModelType.MINDIR, context=context)

    model2 = mslite.Model()
    model2.build_from_file(model_path=MODEL_PATH3, model_type=mslite.ModelType.MINDIR, context=context)

    np_input = np.random.random((1, 3, 512, 512)).astype(np.float32)
    np_output = np.random.random((1, 3, 512, 512)).astype(np.float32)
    input_host = mslite.Tensor(tensor=np_input, shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)

    output_device1 = mslite.Tensor(tensor=np_output, shape=DIM_IN2,
                                   dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))
    output_device2 = mslite.Tensor(tensor=np_output, shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)
    outputs_host1 = mslite.Tensor(tensor=np_output, shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)
    outputs_host2 = mslite.Tensor(tensor=np_output, shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)

    loop = 3
    for i in range(loop):
        model1.predict([input_host], [output_device1])
        model2.predict([output_device1], [output_device2])
    for i in range(loop):
        model1.predict([input_host], [outputs_host1])
        model2.predict([outputs_host1], [outputs_host2])


def _common_functional_accuracy_mult_model(model1_path: str, model2_path: str, output_add: str):
    '''
    common accuracy verification for multiple model scenarios.
    '''
    model1_ge_v1 = mslite.Model()
    context1_ge_v1 = _create_context("ge-v1")
    model1_ge_v1.build_from_file(model_path=model1_path,
                                 model_type=mslite.ModelType.MINDIR, context=context1_ge_v1)
    model2_ge_v1 = mslite.Model()
    context2_ge_v1 = _create_context("ge-v1")
    model2_ge_v1.build_from_file(model_path=model2_path,
                                 model_type=mslite.ModelType.MINDIR, context=context2_ge_v1)

    model1_ge = mslite.Model()
    context1_ge = _create_context("ge")
    model1_ge.build_from_file(model_path=model1_path,
                              model_type=mslite.ModelType.MINDIR, context=context1_ge)
    model2_ge = mslite.Model()
    context2_ge = _create_context("ge")
    model2_ge.build_from_file(model_path=model2_path,
                              model_type=mslite.ModelType.MINDIR, context=context2_ge)

    np_input = np.random.random((1, 3, 512, 512)).astype(np.float32)
    loop = 3
    result_ge = []
    for i in range(loop):
        inputs = mslite.Tensor(tensor=np_input.copy(), shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)
        outputs_host1 = mslite.Tensor(tensor=np_input.copy(), shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)
        outputs_host2 = mslite.Tensor(tensor=np_input.copy(), shape=DIM_OUT2, dtype=mslite.DataType.FLOAT32)

        model1_ge.predict([inputs], [outputs_host1])
        model2_ge.predict([outputs_host1], [outputs_host2])
        result_ge.append([outputs_host1.get_data_to_numpy(), outputs_host2.get_data_to_numpy()])

    result_ge_v1 = []
    for i in range(loop):
        inputs = mslite.Tensor(tensor=np_input.copy(), shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)
        outputs2_host = mslite.Tensor(tensor=np_input.copy(), shape=DIM_IN2, dtype=mslite.DataType.FLOAT32)
        outputs1 = (
            mslite.Tensor(tensor=np_input.copy(),
                          shape=DIM_IN2,
                          dtype=mslite.DataType.FLOAT32,
                          device="ascend:" + str(DEVICE_ID))
            if output_add == "device"
            else mslite.Tensor(tensor=np_input.copy(),
                               shape=DIM_IN2,
                               dtype=mslite.DataType.FLOAT32)
        )
        model1_ge_v1.predict([inputs], [outputs1])
        model2_ge_v1.predict([outputs1], [outputs2_host])
        result_ge_v1.append([outputs1.get_data_to_numpy(), outputs2_host.get_data_to_numpy()])
    print("test result: ")
    for of, ob in zip(result_ge, result_ge_v1):
        for f, b in zip(of, ob):
            np.testing.assert_allclose(f, b)
    print("Common multi-model accuracy verification passed.")
    return True


def test_single_autoregressive_accuracy_device():
    '''
    test Single-model autoregression (device output) accuracy
    '''
    # Autoregression is achieved by using MODEL_PATH2 twice
    result = _common_functional_accuracy_mult_model(MODEL_PATH2, MODEL_PATH2, "device")
    if result:
        print("Single-model autoregression (device output) accuracy passed.")


def test_single_autoregressive_accuracy_host():
    '''
    test Single-model autoregression (host output) accuracy
    '''
    result = _common_functional_accuracy_mult_model(MODEL_PATH2, MODEL_PATH2, "host")
    if result:
        print("Single-model autoregression (host output) accuracy passed.")


def test_double_accuracy_device():
    '''
    test double-model (host output) accuracy
    '''
    result = _common_functional_accuracy_mult_model(MODEL_PATH2, MODEL_PATH3, "device")
    if result:
        print("The accuracy verification for dual-model device output passed.")


def test_double_accuracy_host():
    '''
    test double-model (host output) accuracy
    '''
    result = _common_functional_accuracy_mult_model(MODEL_PATH2, MODEL_PATH3, "host")
    if result:
        print("The accuracy verification for dual-model host output passed.")
