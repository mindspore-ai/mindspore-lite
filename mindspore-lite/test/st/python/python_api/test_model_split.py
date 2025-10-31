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
Test for MindSpore graph split
"""

import os
import mindspore_lite as mslite
import numpy as np


def test_graph_split_01_convert(mindir_dir, so_path, output_dir, config_dir):
    '''
    test model cache convert.
    '''
    fmk_type = "ONNX"
    config_file = os.path.join(config_dir, 'graph_split.config')
    model_path = os.path.join(mindir_dir, "02-seg_3.onnx")
    output_path = output_dir + "graph_split"
    cmd_string = so_path + "/tools/converter/converter/converter_lite " + \
                    " --modelFile=" + model_path + \
                    " --optimize=ascend_oriented " + \
                    " --outputFile=" + output_path + \
                    " --fmk=" + fmk_type + \
                    " --configFile=" + config_file + \
                    " --inputShape=input:1,289,289,3"
    ret = os.system(cmd_string)
    if ret != 0:
        raise RuntimeError("model convert failed, cmd_string is: ", cmd_string)

def test_graph_split_02_build(output_dir):
    '''
    test model cache build and inference.
    '''
    dtype_map = {
        mslite.DataType.FLOAT32: np.float32,
        mslite.DataType.INT32: np.int32,
        mslite.DataType.FLOAT16: np.float16,
        mslite.DataType.INT8: np.int8
    }
    context = mslite.Context()
    context.target = ["ascend"]
    context.ascend.devcie_id = 0
    try:
        runner = mslite.MultiModelRunner()
        model_path = output_dir + 'graph_split.mindir'
        runner.build_from_file(model_path, mslite.ModelType.MINDIR, context)
        execs = runner.get_model_executor()
        for exec_ in execs:
            exec_inputs = exec_.get_inputs()
            for input_ in exec_inputs:
                data = np.random.randn(*input_.shape).astype(dtype_map[input_.dtype])
                input_.set_data_from_numpy(data)
            exec_.predict(exec_inputs)
    except Exception as excep:
        raise RuntimeError('run graph split model failed!') from excep
