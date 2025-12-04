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
MindSpore Lite Model Predict
"""
import os
import time
import numpy as np
import mindspore_lite as mslite

WARM_UP = 3
LOOP_COUNT = 10

INDEX_MODEL_INFO_NAME_PAIR = 1
INDEX_MODEL_INFO_NAME_SIZE = 0
INDEX_MODEL_INFO_NAME = 1

INDEX_MODEL_INFO_SHAPE = 2

RES_KEY_MODEL_NAME = "model_name"
RES_KEY_BUILD_TIME = "build_time"
RES_KEY_PREDICT_TIME = "predict_time"
RES_KEY_ACCURACY_THRESHOLD = "accuracy_threshold"
RES_KEY_RESULT = "result"


def ParseModelName(config_line):
    """
    :param config_line: model info
    :return: config file path
    """
    path = config_line.split(";")[0]
    return path


def ParseModelShape(config_line):
    """
    :param config_line: model info
    :return: input name and input shape
    """
    inputs_shape = []
    inputs_name = []
    if len(config_line.split(";")) < 3:
        raise RuntimeError("model info is wrong, please check config file.")
    for input_shape_str in config_line.split(";")[INDEX_MODEL_INFO_SHAPE].split(":"):
        input_shape = [int(dim) for dim in input_shape_str.split(",")]
        inputs_shape.append(input_shape)
    inputs_name_list = config_line.split(";")[INDEX_MODEL_INFO_NAME_PAIR].split(":")[INDEX_MODEL_INFO_NAME].split(",")
    for input_name_str in inputs_name_list:
        inputs_name.append(input_name_str)
    return inputs_name, inputs_shape


class MSLiteModel():
    """
    MindSpore Lite model
    """

    # models_path, save_models_path, config_files_path, so_path, model_info, "ascend", device_id
    def __init__(self, models_path, save_models_path, config_files_path, so_path, model_info, target="ascend",
                 device_id=0):
        self.models_path = models_path
        self.save_models_path = save_models_path
        self.model_name = ParseModelName(model_info)
        self.model_path = os.path.join(models_path, self.model_name)
        self.so_path = so_path
        self.device_id = device_id
        self.target = target

        self.input_names = ParseModelShape(model_info)[0]
        self.input_shapes = ParseModelShape(model_info)[1]
        self.config_file = os.path.join(config_files_path, self.model_name + ".config")
        if len(model_info.split(" ")) < 2:
            raise RuntimeError("accuracy_threshold is wrong, please check config file.")
        self.accuracy_threshold = float(model_info.split(" ")[1])

        self.benchmark_data_file = os.path.join(self.models_path, "input_output", "output", self.model_name + ".out")
        self.in_data_file_list = []
        if len(self.input_names) > 1:
            for i in range(len(self.input_shapes)):
                self.in_data_file_list.append(
                    os.path.join(self.models_path, "input_output", "input", self.model_name + ".bin_" + str(i + 1)))
        else:
            self.in_data_file_list.append(
                os.path.join(self.models_path, "input_output", "input", self.model_name + ".bin"))

        self.mindir_path = os.path.join(save_models_path, self.model_name + ".mindir")
        if len(model_info.split(";")) < 5:
            raise RuntimeError("config info is wrong, please check config file.")
        if model_info.split(";")[4].split(" ")[0] == "large_model":
            self.mindir_path = os.path.join(save_models_path, self.model_name + "_graph.mindir")
        self.input_shape_str = ""
        if model_info.split(";")[3].split(" ")[0] == "static":
            for i,name in enumerate(self.input_names):
                input_shape_str = ""
                for shape in self.input_shapes[i]:
                    input_shape_str += str(shape) + ","
                self.input_shape_str += name + ":" + input_shape_str[:-1] + ";"
        self.input_shape_str = self.input_shape_str[:-1]

        self.predict_result = {}
        self.predict_result[RES_KEY_MODEL_NAME] = self.model_name
        self.predict_result[RES_KEY_BUILD_TIME] = "failed"
        self.predict_result[RES_KEY_PREDICT_TIME] = "failed"
        self.predict_result[RES_KEY_RESULT] = "failed"

    def CreateInputTensors(self):
        """
        create model input tensor by user's data
        """
        model_inputs = self.model.get_inputs()
        inputs = []
        if len(self.in_data_file_list) == len(model_inputs):
            for i,_ in enumerate(model_inputs):
                if model_inputs[i].dtype == mslite.DataType.FLOAT32:
                    data = np.fromfile(self.in_data_file_list[i], dtype=np.float32).reshape(self.input_shapes[i])
                elif model_inputs[i].dtype == mslite.DataType.INT32:
                    data = np.fromfile(self.in_data_file_list[i], dtype=np.int32).reshape(self.input_shapes[i])
                elif model_inputs[i].dtype == mslite.DataType.FLOAT16:
                    data = np.fromfile(self.in_data_file_list[i], dtype=np.float16).reshape(self.input_shapes[i])
                else:
                    raise RuntimeError("not support input dtype: ", model_inputs[i].dtype)
                model_inputs[i].shape = self.input_shapes[i]
                model_inputs[i].set_data_from_numpy(data)
                inputs.append(model_inputs[i])

        else:
            for i,_ in enumerate(model_inputs):
                if model_inputs[i].dtype == mslite.DataType.FLOAT32:
                    data = np.random.random(self.input_shapes[i]).astype(np.float32)
                elif model_inputs[i].dtype == mslite.DataType.INT32:
                    data = np.random.random(self.input_shapes[i]).astype(np.int32)
                elif model_inputs[i].dtype == mslite.DataType.FLOAT16:
                    data = np.random.random(self.input_shapes[i]).astype(np.float16)
                model_inputs[i].shape = self.input_shapes[i]
                model_inputs[i].set_data_from_numpy(data)
                inputs.append(model_inputs[i])
        return inputs

    def Comparison(self, outputs_tensors, benchmark_file):
        """
        for acc
        """
        all_benchmark_data = []
        num_line = 0
        with open(benchmark_file, "r", encoding='utf-8') as f:
            for line in f:
                num_line += 1
                if line[-1] == "\n":
                    line = line[:-1]
                if line[-1] == " ":
                    line = line[:-1]
                if num_line % 2 == 1:
                    continue
                data_line = line.split(" ")
                benchmark_data = []
                for data in data_line:
                    benchmark_data.append(float(data))
                all_benchmark_data.append(benchmark_data)
        if len(all_benchmark_data) != len(outputs_tensors):
            raise RuntimeError("benchmark data file is wrong.")
        all_err = 0
        for input_i,output in enumerate(outputs_tensors):
            mean_err = 0
            count = 0
            out_data = output.get_data_to_numpy().flatten()
            benchmark_data = all_benchmark_data[input_i]
            if len(benchmark_data) != out_data.size:
                raise RuntimeError("benchmark data size not equal to model output data size.")
            for i,_ in enumerate(benchmark_data):
                abs_err = np.abs(out_data[i] - benchmark_data[i])
                tolerance = 1e-10 + 1e-7 * np.abs(benchmark_data[i])
                if abs_err > tolerance:
                    if np.abs(benchmark_data[i]) < 1e-7:
                        if abs_err > 1e-5:
                            mean_err += abs_err
                            count += 1
                        else:
                            continue
                    else:
                        mean_err += abs_err / np.abs(benchmark_data[i])
                        count += 1
            all_err += (mean_err / (count + 1e-5))
        return all_err / len(outputs_tensors)

    def Convert(self):
        """
        model convert by converter_lite
        """
        if not os.path.exists(self.model_path):
            raise RuntimeError("model file not exit, model file is: ", self.model_path)
        if self.target != "ascend":
            raise RuntimeError("target is not ascend, target is ", self.target)
        fmk_type = None
        model_type = self.model_path.split(".")[-1]
        if model_type == "onnx":
            fmk_type = "ONNX"
        elif model_type == "mindir":
            fmk_type = "MINDIR"
        elif model_type == "pb":
            fmk_type = "TF"
        elif model_type == "tflite":
            fmk_type = "TFLITE"
        else:
            raise RuntimeError("model type is not onnx/mindir/pb/tflite, model type is ", model_type)
        config_file = ""
        if os.path.exists(self.config_file):
            config_file = self.config_file
        cmd_string = self.so_path + "/tools/converter/converter/converter_lite " + \
                     " --modelFile=" + self.model_path + \
                     " --optimize=ascend_oriented " + \
                     " --outputFile=" + self.save_models_path + "/" + self.model_path.split("/")[-1] + \
                     " --fmk=" + fmk_type + \
                     " --configFile=" + config_file + \
                     " --inputShape=\"" + self.input_shape_str + "\""
        ret = os.system(cmd_string)
        if ret != 0:
            raise RuntimeError("model convert failed, cmd_string is: ", cmd_string)

    def Build(self):
        """
        model build by python API
        """
        if not os.path.exists(self.mindir_path):
            raise RuntimeError("mindir file not exit, mindir path is: ", self.mindir_path)
        context = mslite.Context()
        context.target = ["ascend"]
        context.ascend.device_id = self.device_id
        self.model = mslite.Model()
        time_build_start = time.time()
        self.model.build_from_file(model_path=self.mindir_path, model_type=mslite.ModelType.MINDIR, context=context)
        time_build_end = time.time()
        build_time = (time_build_end - time_build_start) * 1000
        self.predict_result[RES_KEY_BUILD_TIME] = str(build_time)

    def Predict(self):
        """
        model predict by python API
        """
        self.model.resize(self.model.get_inputs(), self.input_shapes)
        inputs = self.CreateInputTensors()
        for _ in range(WARM_UP):
            outputs = self.model.predict(inputs)
        time_predict_start = time.time()
        for _ in range(LOOP_COUNT):
            outputs = self.model.predict(inputs)
        time_predict_end = time.time()
        time_predict = (time_predict_end - time_predict_start) * 1000 / LOOP_COUNT
        self.predict_result[RES_KEY_PREDICT_TIME] = str(time_predict)
        acc = self.Comparison(outputs, self.benchmark_data_file)
        self.predict_result[RES_KEY_ACCURACY_THRESHOLD] = str(acc * 100) + "%"
        self.predict_result[RES_KEY_RESULT] = "pass"
        if acc * 100 > self.accuracy_threshold:
            raise RuntimeError("acc is more than accuracy_threshold, acc is ", acc * 100, "%")

    def GetPredictResult(self):
        """
        return model predict result
        """
        return self.predict_result
