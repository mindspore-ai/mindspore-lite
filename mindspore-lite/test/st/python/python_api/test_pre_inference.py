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
Test for MindSpore Lite preinference
"""

import os
import subprocess
import shlex

def test_pre_inference_01_convert(mindir_dir, so_path, output_dir):
    '''
    test convert zeroshape model
    '''
    fmk_type = "ONNX"
    model_path = os.path.join(mindir_dir, "zeroshape.onnx")
    output_path = output_dir + "zeroshape"
    cmd_string = so_path + "/tools/converter/converter/converter_lite " + \
                    " --modelFile=" + model_path + \
                    " --optimize=ascend_oriented " + \
                    " --outputFile=" + output_path + \
                    " --fmk=" + fmk_type + \
                    " --inputShape=input:2,2"
    ret = os.system(cmd_string)
    if ret != 0:
        raise RuntimeError("model convert failed, cmd_string is: ", cmd_string)

def test_pre_inference_02_not_enable(so_path, output_dir):
    '''
    test not enable pre inference
    '''
    model_path = os.path.join(output_dir, "zeroshape.mindir")
    cmd_string = so_path + "/tools/benchmark/benchmark " + \
                    " --modelFile=" + model_path + \
                    " --modelType=MindIR" + \
                    " --device=Ascend"
    result = subprocess.run(shlex.split(cmd_string), shell=False, capture_output=True, text=True, check=False)
    assert "Inference error" in result.stderr

def test_pre_inference_03_enable(so_path, output_dir, config_dir):
    '''
    test enable pre inference
    '''
    model_path = os.path.join(output_dir, "zeroshape.mindir")
    config_path = os.path.join(config_dir, "pre_inference.config")
    cmd_string = so_path + "/tools/benchmark/benchmark " + \
                    " --modelFile=" + model_path + \
                    " --modelType=MindIR" + \
                    " --device=Ascend" + \
                    " --configFile=" + config_path
    result = subprocess.run(shlex.split(cmd_string), shell=False, capture_output=True, text=True, check=False)
    assert "PreBuild or PreInference failed" in result.stderr
