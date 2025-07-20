import logging
import os
import yaml
from onnx import helper
import numpy as np


current_directory = os.getcwd()
log_path = current_directory+"/op_st.log"
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logger = logging.getLogger("opst")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

if not os.path.exists(log_path):
    os.mkdir(log_path)


type_map = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    8: np.string_,
    9: np.bool_,
    10: np.float16,
    11: np.double,
    12: np.uint32,
    13: np.uint64,
    14: np.complex64,
}

non_negative_op = set(
    ["Pow", "Log", "Sqrt", "ReduceLogSum", "ReduceLog", "ConstantOfShape"])

# 应该改名叫parser


def has_key(convert_dict, key="fp16"):
    for k, v in convert_dict.items():
        if k == key and v:
            return True
    return False


class Initializer:
    def __init__(self, graph_inputs, graph_outputs, graph_initializer):
        self.initializers = graph_initializer
        self.input_tensors = graph_inputs
        self.output_tensors = graph_outputs

    def init_values(self):
        input_tensors = []
        for item in self.input_tensors:
            dims_ = []
            for j in item["dims"]:
                if j == "None":
                    dims_.append(None)
                else:
                    dims_.append(j)
            input_tensor = helper.make_tensor_value_info(
                item["name"], item["data_type"], dims_
            )
            input_tensors.append(input_tensor)
        output_tensors = []
        for item in self.output_tensors:
            dims_ = []
            for j in item["dims"]:
                if j == "None":
                    dims_.append(None)
                else:
                    dims_.append(j)
            output_tensor = helper.make_tensor_value_info(
                item["name"], item["data_type"], dims_
            )
            output_tensors.append(output_tensor)
        initializers = []
        if self.initializers != "None":
            for init in self.initializers:
                # 随机生成tensor值
                init_vals = 0
                set_value = False
                for k, v in init.items():
                    if k == "value":
                        set_value = True
                if set_value:
                    init_vals = (
                        np.array(init["value"])
                        .reshape(*init["dims"])
                        .astype(type_map[init["data_type"]])
                    )
                else:
                    init_vals = np.random.randn(*init["dims"]).astype(
                        type_map[init["data_type"]]
                    )
                initializer = helper.make_tensor(
                    name=init["name"],
                    data_type=init["data_type"],
                    dims=init["dims"],
                    vals=init_vals,  # 这里初始化为0.1，你可以根据需要调整
                )
                initializers.append(initializer)
        return input_tensors, output_tensors, initializers


# 生成配置列表
# 后续方便正交测试用例
class OnnxModelConfig:
    """
    This is config parameters for generating onnx model
    Input: yaml configure file
    Output: onnx model config list
    """

    def __init__(self, config_file):
        self.disable_all = False
        self.config_file = config_file
        with open(os.path.realpath(self.config_file), "r") as file:
            self.configs = yaml.safe_load(file)
        self.disable_all = has_key(self.configs, "disabled")
        self.op_configs_ = {}
        self.op_configs_["op_name"] = self.configs["op_name"]
        self.op_configs_["op_configs"] = self.configs["genonnx"]
        self.graph_input_names = {}
        for each_model in self.op_configs_["op_configs"]:
            each_input_name = []
            for graph_input in each_model["graph_param"]["inputs"]:
                each_input_name.append(graph_input["name"])
            self.graph_input_names[each_model["model_name"]] = each_input_name
        # for graph_input in self.op_configs_["op_configs"][0]["graph_param"]["inputs"]:
        #     self.graph_input_names.append(graph_input["name"])

    def modelconfigs(self):
        return self.op_configs_


class GenGoldConfig:
    """
    todo:
    1. 建立gold_name到input_shapes的反向hash表,方便run阶段查找输入维度
    2. 设置特殊算子输入表，标明一些输入为正数的算子
    3. 将[[128,128]]转换为X:128,128
    """

    def __init__(self, config_file):
        self.config_file = config_file
        with open(os.path.realpath(self.config_file), "r") as file:
            self.configs = yaml.safe_load(file)
        # 确保输入数据维度的长度和模型设置的inputs长度相同
        self.gold_configs_ = self.configs["gengold"]
        self.gold_input_shapes_dict = {}
        self.gold_input_name_dict = {}
        for gold_config in self.gold_configs_:
            self.gold_input_shapes_dict[gold_config["gold_name"]] = len(
                gold_config["input_shapes"]
            )
            self.gold_input_name_dict[gold_config["gold_name"]] = gold_config[
                "input_shapes"
            ]

    def goldconfigs(self):
        return self.gold_configs_


class ConvertConfig:
    """
    todo:
    1. 完成解析输入shape的函数
    """

    def __init__(self, config_file):
        self.config_file = config_file
        with open(os.path.realpath(self.config_file), "r") as file:
            self.configs = yaml.safe_load(file)
        self.convert_configs_ = self.configs["convert"]

    def convertconfigs(self):
        return self.convert_configs_


class RunConfigs:
    """
    run process config
    """

    def __init__(self, config_file):
        self.config_file = config_file
        with open(os.path.realpath(self.config_file), "r") as file:
            self.configs = yaml.safe_load(file)
        self.run_configs_ = self.configs["run"]

    def runconfigs(self):
        return self.run_configs_
