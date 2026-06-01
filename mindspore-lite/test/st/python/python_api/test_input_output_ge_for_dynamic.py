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
Test lite python API.
"""
import os
import time
import subprocess
from pathlib import Path
from typing import List
import pytest
import numpy as np
import mindspore_lite as mslite



MODEL_PATH: str = ""
DIM_IN = [[100,77], [100,16,128], [1]]
DIM_OUT = [[100,128], [100]]


DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))
ResultList = List[List[np.ndarray]]

def _convert_onnx_to_mindir(model_name: str, so_path: Path, mindir_dir: Path, output_dir: Path)-> Path:
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
    global MODEL_PATH
    models = [
        "poisson_itm_16x128_text.onnx"
    ]
    MODEL_PATH = str(_convert_onnx_to_mindir(models[0], so_path, mindir_dir, output_dir))
    yield
    Path(MODEL_PATH).unlink(missing_ok=True)


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


def _create_dynamic_inputs_outputs():
    np_input1 = np.random.random((100,77)).astype(np.int32)
    np_input2 = np.random.random((100,16,128)).astype(np.float32)
    np_input3 = np.random.random((1)).astype(np.float32)
    np_output1 = np.random.random((100,128)).astype(np.float32)
    np_output2 = np.random.random((100)).astype(np.float32)
    return [[np_input1, np_input2, np_input3], [np_output1, np_output2]]

def _common_functional_dynamic_accuracy(outputs_type: str):
    """
    Common accuracy test for dynamic model inference with different output configurations.
    Args:
        outputs_type (str): A string of '0', '1', or '2' indicating:
            '0' = null output tensor,
            '1' = host output tensor,
            '2' = device output tensor.
    """
    model_ge_v1 = mslite.Model()
    context_ge_v1 = _create_context("ge-v1")
    model_ge_v1.build_from_file(model_path=MODEL_PATH, model_type=mslite.ModelType.MINDIR, context=context_ge_v1)

    model_ge = mslite.Model()
    context_ge = _create_context("ge")
    model_ge.build_from_file(model_path=MODEL_PATH, model_type=mslite.ModelType.MINDIR, context=context_ge)

    inputs_outputs = _create_dynamic_inputs_outputs()

    loop = 3
    result_ge = []
    for i in range(loop):
        inputs_ge = [
            mslite.Tensor(tensor=inputs_outputs[0][0].copy(), shape=DIM_IN[0], dtype=mslite.DataType.INT32),
            mslite.Tensor(tensor=inputs_outputs[0][1].copy(), shape=DIM_IN[1], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[0][2].copy(), shape=DIM_IN[2], dtype=mslite.DataType.FLOAT32)
        ]
        outputs = [
            mslite.Tensor(tensor=inputs_outputs[1][0].copy(), shape=DIM_OUT[0], dtype=mslite.DataType.FLOAT32),
            mslite.Tensor(tensor=inputs_outputs[1][1].copy(), shape=DIM_OUT[1], dtype=mslite.DataType.FLOAT32),
        ]
        model_ge.predict(inputs_ge, outputs)
        result_ge.append([t.get_data_to_numpy() for t in outputs])

    result_ge_v1 = []

    for i in range(loop):
        outputs = []
        inputs_ge_v1 = [
            mslite.Tensor(tensor=inputs_outputs[0][0].copy(), shape=DIM_IN[0], dtype=mslite.DataType.INT32),
            mslite.Tensor(tensor=inputs_outputs[0][1].copy(), shape=DIM_IN[1], dtype=mslite.DataType.FLOAT32,
                          device="ascend:" + str(DEVICE_ID)),
            mslite.Tensor(tensor=inputs_outputs[0][2].copy(), shape=DIM_IN[2], dtype=mslite.DataType.FLOAT32,
                          device="ascend:" + str(DEVICE_ID))
        ]
        if outputs_type:
            for j, char in enumerate(outputs_type):
                if char == '0':
                    tensor = mslite.Tensor(shape=DIM_OUT[j], dtype=mslite.DataType.FLOAT32)
                elif char == '1':
                    tensor = mslite.Tensor(tensor=inputs_outputs[1][j].copy(), shape=DIM_OUT[j],
                                           dtype=mslite.DataType.FLOAT32)
                elif char == '2':
                    tensor = mslite.Tensor(tensor=inputs_outputs[1][j].copy(),
                                           shape=DIM_OUT[j],
                                           dtype=mslite.DataType.FLOAT32,
                                           device="ascend:" + str(DEVICE_ID))
                else:
                    raise ValueError(f"Invalid value: '{char}', only '0', '1', or '2' are allowed.")
                outputs.append(tensor)
        model_ge_v1.predict(inputs_ge_v1, outputs)
        result_ge_v1.append([t.get_data_to_numpy() for t in outputs])

    print("test result: ")
    for of, ob in zip(result_ge_v1, result_ge):
        for f, b in zip(of, ob):
            print(f"GE_v1 shape: {f.shape}, GE shape: {b.shape}")
            np.testing.assert_allclose(f, b)

    print("Common single-model accuracy verification passed.")
    return True

def test_dynamic_inputs_host_device_outputs_null():
    '''
    test inputs are on host and device, outputs are null.
    '''
    result = _common_functional_dynamic_accuracy("")
    if result:
        print("test_dynamic_inputs_host_device_outputs_data_null test success!")
def test_dynamic_inputs_host_device_outputs_device():
    '''
    test inputs are on host and device, outputs are on device.
    '''
    result = _common_functional_dynamic_accuracy("22")
    if result:
        print("test_dynamic_inputs_host_device_outputs_data_device test success!")

def test_dynamic_inputs_host_device_outputs_host():
    '''
    test inputs are on host and device, outputs are on host.
    '''
    result = _common_functional_dynamic_accuracy("11")
    if result:
        print("test_dynamic_inputs_host_device_outputs_data_host test success!")

def test_dynamic_inputs_host_device_outputs_device_host():
    '''
    test inputs are on host and device, outputs are on host and device.
    '''
    result = _common_functional_dynamic_accuracy("21")
    if result:
        print("test_dynamic_inputs_host_device_outputs_host_device test success!")

def test_dynamic_inputs_host_device_outputs_null_host():
    '''
    test inputs are on host and device, outputs are null and host.
    '''
    result = _common_functional_dynamic_accuracy("01")
    if result:
        print("test_dynamic_inputs_host_device_outputs_data_null_host_device test success!")

def test_dynamic_inputs_host_device_outputs_data_null_device():
    '''
    test inputs are on host and device, outputs are null and device.
    '''
    result = _common_functional_dynamic_accuracy("02")
    if result:
        print("test_dynamic_inputs_host_device_outputs_data_null_host_device test success!")

def test_single_dynamic_model_performance():
    '''
    test single dynamic model performance
    '''
    model = mslite.Model()
    context = _create_context("ge-v1")
    model.build_from_file(model_path=MODEL_PATH, model_type=mslite.ModelType.MINDIR, context=context)

    inputs_outputs = _create_dynamic_inputs_outputs()

    inputs_host1 = mslite.Tensor(tensor=inputs_outputs[0][0], shape=DIM_IN[0], dtype=mslite.DataType.INT32)
    inputs_host2 = mslite.Tensor(tensor=inputs_outputs[0][1], shape=DIM_IN[1], dtype=mslite.DataType.FLOAT32)
    inputs_host3 = mslite.Tensor(tensor=inputs_outputs[0][2], shape=DIM_IN[2], dtype=mslite.DataType.FLOAT32)

    inputs_device1 = mslite.Tensor(tensor=inputs_outputs[0][0], shape=DIM_IN[0],
                                   dtype=mslite.DataType.INT32, device="ascend:" + str(DEVICE_ID))
    inputs_device2 = mslite.Tensor(tensor=inputs_outputs[0][1], shape=DIM_IN[1],
                                   dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))
    inputs_device3 = mslite.Tensor(tensor=inputs_outputs[0][2], shape=DIM_IN[2],
                                   dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))

    outputs_host1 = mslite.Tensor(tensor=inputs_outputs[1][0], shape=DIM_OUT[0], dtype=mslite.DataType.FLOAT32)
    outputs_host2 = mslite.Tensor(tensor=inputs_outputs[1][1], shape=DIM_OUT[1], dtype=mslite.DataType.FLOAT32)

    outputs_device1 = mslite.Tensor(tensor=inputs_outputs[1][0], shape=DIM_OUT[0],
                                    dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))
    outputs_device2 = mslite.Tensor(tensor=inputs_outputs[1][1], shape=DIM_OUT[1],
                                    dtype=mslite.DataType.FLOAT32, device="ascend:" + str(DEVICE_ID))
    loop = 10
    time_host_ms = []
    time_device_ms = []
    predict_start1 = time.time()
    for i in range(loop):
        loop_predict_start = time.time()
        model.predict([inputs_host1, inputs_host2, inputs_host3], [outputs_host1, outputs_host2])
        loop_predict_end = time.time()
        current_time_ms = (loop_predict_end - loop_predict_start) * 1000
        print(f"Host loop {i}: predict time : {current_time_ms} ms")
        if i > 0:
            time_host_ms.append(current_time_ms)
    predict_end1 = time.time()
    print("[inputs_host_outputs_host] model predict time = ", (predict_end1 - predict_start1)*1000, " ms")

    predict_start2 = time.time()
    for i in range(loop):
        loop_predict_start = time.time()
        model.predict([inputs_device1, inputs_device2, inputs_device3],
                      [outputs_device1, outputs_device2])
        loop_predict_end = time.time()
        current_time_ms = (loop_predict_end - loop_predict_start) * 1000
        print(f"Device loop {i}: predict time : {current_time_ms:.4f} ms")
        if i >0:
            time_device_ms.append(current_time_ms)
    predict_end2 = time.time()
    print("[inputs_device_outputs_device] model predict time = ", (predict_end2 - predict_start2)*1000, " ms")

    # Performance Guardrail
    max_ratio = 1.1
    all_time_device=0
    all_time_host=0
    for i in range(loop-1):
        all_time_host += time_host_ms[i]
        all_time_device += time_device_ms[i]
    print("-" * 30)
    print(f"Total Host Time (sum of first): {all_time_host:.4f} ms")
    print(f"Total Device Time (sum of first): {all_time_device:.4f} ms")
    print(f"Average Host Time: {all_time_host/9:.4f} ms")
    print(f"Average Device Time: {all_time_device/9:.4f} ms")
    print("-" * 30)
    performance_guardrail(max_ratio, all_time_host/9, all_time_device/9,"single dynamic model performance")


def performance_guardrail(max_ratio: float, third_loop_time_host_ms: float,
                          third_loop_time_device_ms: float, test_name: str):
    """
    Performance guardrail for device-mode prediction time relative to host-mode prediction time.
    """
    print(f"\n--- Running Performance Guardrail for {test_name} ---")

    if third_loop_time_host_ms > 0:
        ratio = third_loop_time_device_ms / third_loop_time_host_ms

        print(f"Device/Host Time Ratio: {ratio:.4f} (Required Max: {max_ratio:.2f})")

        assert ratio <= max_ratio, \
            f"Performance Guard Failed for {test_name}: Device time ({third_loop_time_device_ms:.4f} ms) " \
            f"is {ratio*100:.2f}% of Host time ({third_loop_time_host_ms:.4f} ms). " \
            f"Required maximum ratio: {max_ratio*100:.0f}%."

        print(f"SUCCESS: Device time ratio is {ratio*100:.2f}%, within the required range.")
    else:
        assert False, (
            f"Guard Failed ({test_name}): Host time invalid ({third_loop_time_host_ms:.4f} ms). "
            f"Cannot calculate ratio."
        )
    print(f"--- {test_name} guardrail completed ---")
