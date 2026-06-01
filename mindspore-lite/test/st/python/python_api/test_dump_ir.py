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
Test for MindSpore Lite Converter: DumpGraphIR
"""

import os
import pathlib
import mindspore_lite as mslite

MODEL_FILE = "./single_matmul_model.onnx"
DEVICE_ID = int(os.environ.get('ASCEND_DEVICE_ID', '0'))


def test_converter_with_dump_graph_ir_without_dump_graph():
    """
    test MindSpore Lite converter model without dump graph ir.
    """
    converter = mslite.Converter()
    converter.device_id = DEVICE_ID
    converter.input_shape = {"input": [1, 4]}
    converter.optimize = "ascend_oriented"
    converter.save_type = mslite.ModelType.MINDIR
    converter.convert(fmk_type=mslite.FmkType.ONNX, model_file=MODEL_FILE, output_file="test")
    ir_file_list = list(pathlib.Path("./").glob("*.ir"))
    assert len(ir_file_list) == 0


def test_converter_with_dump_graph_ir_with_level1():
    """
    test MindSpore Lite converter model with dump graph ir.
    """
    os.environ['MSLITE_DUMP_GRAPH_LEVEL'] = '1'
    converter = mslite.Converter()
    converter.device_id = DEVICE_ID
    converter.input_shape = {"input": [1, 4]}
    converter.optimize = "ascend_oriented"
    converter.save_type = mslite.ModelType.MINDIR
    converter.convert(fmk_type=mslite.FmkType.ONNX, model_file=MODEL_FILE, output_file="test")
    ir_file_list = list(pathlib.Path("./").glob("*.ir"))
    assert len(ir_file_list) > 0


def test_converter_with_dump_graph_ir_with_level1_and_path():
    """
    test MindSpore Lite converter model with dump graph ir.
    """
    dump_path = "./dump_graph/"
    os.environ['MSLITE_DUMP_GRAPH_LEVEL'] = '1'
    os.environ['MSLITE_DUMP_GRAPH_PATH'] = dump_path
    folder = os.path.exists(dump_path)
    if not folder:
        os.makedirs(dump_path)
    converter = mslite.Converter()
    converter.device_id = DEVICE_ID
    converter.input_shape = {"input": [1, 4]}
    converter.optimize = "ascend_oriented"
    converter.save_type = mslite.ModelType.MINDIR
    converter.convert(fmk_type=mslite.FmkType.ONNX, model_file=MODEL_FILE, output_file="test")
    ir_file_list = list(pathlib.Path(dump_path).glob("*.ir"))
    assert len(ir_file_list) > 0
