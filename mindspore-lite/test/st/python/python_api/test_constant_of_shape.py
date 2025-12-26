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
test constant of shape
"""
from pathlib import Path
import subprocess
import numpy as np
import pytest
import mindspore_lite as mslite


@pytest.fixture(scope="module", autouse=True)
def module_setup_and_teardown_fixture(so_path, mindir_dir, output_dir):
    """
    module setup
    """
    fmk = "ONNX"
    model_path = Path(mindir_dir) / "constant_of_shape.onnx"
    output_path = Path(output_dir) / "constant_of_shape.onnx"
    optimize = "general"
    cmd = [
        Path(so_path) / "tools/converter/converter/converter_lite",
        f"--optimize={optimize}",
        f"--modelFile={model_path}",
        f"--outputFile={output_path}",
        f"--fmk={fmk}",
    ]
    subprocess.run(cmd, check=True)

    yield


@pytest.mark.backend("mslite_large_model_inference_arm_ascend910B", "arm32_cpu", "arm64_cpu")
def test_constant_of_shape(output_dir):
    """
    test constant of shape
    """
    context = mslite.Context()
    model = mslite.Model()
    context.target = ["cpu"]

    model.build_from_file(str(Path(output_dir) / "constant_of_shape.onnx.mindir"), mslite.ModelType.MINDIR, context)

    input1 = np.array([5], dtype=np.int32)
    input2 = np.array([6], dtype=np.int32)

    out1 = model.predict([input1])
    assert out1[0].shape == input1

    out2 = model.predict([input2])
    assert out2[0].shape == input2
