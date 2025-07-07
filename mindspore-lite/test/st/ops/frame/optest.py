import os
import copy
import subprocess
import configs as conf
import onnxruntime

import onnx
from onnx import helper

import numpy as np

class OpTest:
    def __init__(self, config_file, output_path):
        self.config_file = config_file
        self.model_configs = conf.OnnxModelConfig(self.config_file)
        self.golden_confis = conf.GenGoldConfig(self.config_file)
        self.convert_configs = conf.ConvertConfig(self.config_file)
        self.run_configs = conf.RunConfigs(self.config_file)
        self.num_success = 0
        self.test_nums = 0
        self.failed_test_name = []
        self.failed_gold_in = []
        self.ms_onnx_convert_reverse_dict = {}
        mslite_package_path = os.getenv("MSLITE_PACKAGE_PATH")

        if mslite_package_path is None:
            raise Exception(
                "Please set envion \"MSLITE_PACKAGE_PATH\" to specify the MSLite"+
                "package root path, which is necessary for MSLITE-OP-ST!"
            )

        mslite_package_path = os.path.realpath(mslite_package_path)
        conf.logger.info("use mslite package: %s", mslite_package_path)

        self.mslite_convert_path = os.path.join(
            mslite_package_path, "tools/converter/converter/converter_lite"
        )
        conf.logger.info("use mslite converter: %s", self.mslite_convert_path)
        self.mslite_benchmark_path = os.path.join(
            mslite_package_path, "tools/benchmark/benchmark"
        )
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        self.output_path = (
            self.output_path
            + "/"
            + str.lower(self.model_configs.op_configs_["op_name"])
        )
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        conf.logger.info("use mslite benchmark: %s", self.mslite_benchmark_path)

## 因为ms默认输入格式为NHWC，将onnx模型的NCHW格式输入转为NHWC格式
# 将所有标杆数据 gold_name:shapes 转换为 X1:dim1, dim2;X2:dim1, dim2...的格式
    def get_run_input_shapes(self, run_config):
        run_input_shapes_ = {}
        input_shapes = self.golden_confis.gold_input_name_dict[run_config["gold_in"]]
        assert len(input_shapes) == len(self.model_configs.graph_input_names[
            self.ms_onnx_convert_reverse_dict[run_config["in_model"]]])
        temp_input = copy.deepcopy(input_shapes)
        input_shape_str = ""
        for i in range(len(temp_input)):
            input_shape_str += self.model_configs.graph_input_names[
                self.ms_onnx_convert_reverse_dict[run_config["in_model"]]][i] + ":"
            if len(temp_input[i]) == 4:
                # benchmark ms模型转换为NHWC
                C = temp_input[i][1]
                H = temp_input[i][2]
                W = temp_input[i][3]
                temp_input[i][1] = H
                temp_input[i][2] = W
                temp_input[i][3] = C
            for l in temp_input[i]:
                input_shape_str += str(l) + ","
            input_shape_str = input_shape_str[:-1]
            input_shape_str += ";"
        run_input_shapes_[run_config["gold_in"]] = input_shape_str[:-1]
        return run_input_shapes_

    def gen_onnx(self):
        op_name = self.model_configs.op_configs_["op_name"]
        op_configs = self.model_configs.op_configs_["op_configs"]

        onnx_path = self.output_path + "/onnx_models"

        if not os.path.exists(onnx_path):
            conf.logger.debug("mkdir %s", onnx_path)
            os.mkdir(onnx_path)
        for op_config in op_configs:

            disabled = conf.has_key(op_config, key="disabled")
            if disabled:
                conf.logger.debug(
                    "disable onnx case. model name: %s", op_config['model_name']
                )
                continue

            model_name = op_config["model_name"]
            node_param = op_config["node_param"]
            node_inputs = node_param["inputs"]
            node_outputs = node_param["outputs"]
            attributes = node_param["attributes"]
            if attributes == "None":
                attributes = {}
            node = helper.make_node(
                op_name, inputs=node_inputs, outputs=node_outputs, **attributes
            )

            graph_param = op_config["graph_param"]
            graph_inputs = graph_param["inputs"]
            graph_outputs = graph_param["outputs"]
            graph_initializer = graph_param["initializer"]

            input_tensors, output_tensors, initializers = conf.Initializer(
                graph_inputs, graph_outputs, graph_initializer
            ).init_values()

            graph = helper.make_graph(
                [node],
                model_name,
                input_tensors,
                output_tensors,
                initializer=initializers,
            )
            model = helper.make_model(graph, producer_name="test", opset_imports=[helper.make_opsetid("", 18)])
            # for opset in model.opset_import:
            #     print(f"Domain: {opset.domain}, Version: {opset.version}")
            onnx.checker.check_model(model)
            model.ir_version = 8

            # 设置保存路径为当前工作目录的子目录
            onnx.save(model, onnx_path + "/" + model_name)
            conf.logger.info("save onnx model %s success", model_name)

    def gen_gold(self):
        for gold_config in self.golden_confis.gold_configs_:

            disabled = conf.has_key(gold_config, key="disabled")
            if disabled:
                conf.logger.debug(
                    "disable gengold case. gold_name: %s", gold_config['gold_name']
                )
                continue

            model = onnx.load(
                self.output_path + "/onnx_models/" + gold_config["in_model"]
            )
            onnx.checker.check_model(model)

            # conf.logger.debug(onnx.helper.printable_graph(model.graph))
            # dtypes默认fp32
            input_tensors = []
            input_feeds = {}
            output_names = []

            input_shapes = gold_config["input_shapes"]
            input_dtypes = gold_config["input_dtypes"]
            assert len(input_shapes) == len(input_dtypes)
            input_length = len(input_shapes)
            seed = 0
            np.random.seed(seed)
            # 判断非负输入的算子
            if self.model_configs.op_configs_["op_name"] in conf.non_negative_op:
                for i in range(input_length):
                    input_tensors.append(
                        np.abs(
                            np.random.randn(*input_shapes[i]).astype(
                                conf.type_map[input_dtypes[i]]
                            )
                        )
                    )
            else:
                for i in range(input_length):
                    input_tensors.append(
                        np.random.randn(*input_shapes[i]).astype(
                            conf.type_map[input_dtypes[i]]
                        )
                    )
            ort_session = onnxruntime.InferenceSession(
                self.output_path + "/onnx_models/" + gold_config["in_model"],
                providers=["CPUExecutionProvider"],
            )
            for i in range(input_length):
                input_feeds[ort_session.get_inputs()[i].name] = input_tensors[i]
            for i in range(len(ort_session.get_outputs())):
                output_names.append(ort_session.get_outputs()[i].name)
            ort_outputs = ort_session.run(
                output_names=output_names, input_feed=input_feeds
            )

            ##############Save input and output######################
            gold_path = self.output_path + "/gold_files"
            if not os.path.exists(gold_path):
                conf.logger.debug("mkdir: %s", gold_path)
                os.mkdir(gold_path)
            gold_root = gold_path + "/" + gold_config["gold_name"]
            if not os.path.exists(gold_root):
                conf.logger.debug("mkdir: %s", gold_root)
                os.mkdir(gold_root)
            gold_inputs = input_tensors
            for i, gold_input in enumerate(gold_inputs):
                # NCHW to NHWC 目前只有4Dtensor
                if len(gold_input.shape) == 4:
                    gold_input = np.transpose(gold_input, (0, 2, 3, 1))

                if gold_input.dtype == np.int64:
                    gold_input = gold_input.astype(np.int32)
                gold_input.tofile(os.path.join(gold_root, f"input_{i}.bin"))
            output_onnx_name = "model.onnx.out"
            output_file = os.path.join(
                gold_root, output_onnx_name
            )  # benchmark output data path

            with open(output_file, "w") as text_file:
                for i in range(len(ort_outputs)):
                    gold_output = ort_outputs[i]
                    if gold_output.dtype == np.int64:
                        gold_output = gold_output.astype(np.int32)
                    output_name = ort_session.get_outputs()[i].name
                    output_shape = gold_output.shape
                    conf.logger.info(
                        "saving output %d, %s, type is: %s, shape: %s", i, output_name, gold_output.dtype, output_shape
                    )

                    text_file.write(output_name + " " + str(len(output_shape)) + " ")
                    text_file.write(" ".join([str(s) for s in output_shape]))
                    text_file.write("\n")
                    if np.isnan(gold_output).any():
                        conf.logger.error("nan number: %s", output_file)
                        breakpoint()
                    for k in gold_output.flatten():
                        text_file.write(str(k) + " ")
                    text_file.write("\n")
        conf.logger.info(
            "Generate %d golddata success ", len(self.golden_confis.gold_configs_)
        )

    def convert_ops(self):
        convert_configs = self.convert_configs.convert_configs_
        ms_path = self.output_path + "/ms_models"
        if not os.path.exists(ms_path):
            conf.logger.info("mkdir: %s", ms_path)
            os.mkdir(ms_path)
        for convert_config in convert_configs:
            # 默认fp16为false
            fp16 = conf.has_key(convert_config)

            disabled = conf.has_key(convert_config, key="disabled")
            if disabled:
                conf.logger.debug(
                    "disable convert case out model: %s, in_model: %s",
                    convert_config['out_model'], convert_config['in_model']
                )
                continue
            # 动态转动态或者静态转静态，不需要设置--inputShape参数
            input_shapes = convert_config["input_shapes"]
            self.ms_onnx_convert_reverse_dict[
                convert_config["out_model"].split(".")[0]+".ms"] = convert_config["in_model"]
            in_model_path = (
                self.output_path + "/onnx_models/" + convert_config["in_model"]
            )
            out_model_path = ms_path + "/" + convert_config["out_model"].split(".")[0]
            args = [
                self.mslite_convert_path,
                "--fmk=ONNX",
                f"--modelFile={in_model_path}",
                f"--outputFile={out_model_path}",
                "--optimize=none",
            ]
            if input_shapes == "None" and fp16:
                args.append("--fp16=on")
            elif input_shapes != "None" and not fp16:
                args.append(f"--inputShape={input_shapes}")
            elif input_shapes != "None" and fp16:
                args.append(f"--inputShape={input_shapes}")
                args.append("--fp16=off")
            convert_comand = ""
            for arg in args:
                convert_comand += arg + " "
            conf.logger.info("convert command : %s", convert_comand)
            result = subprocess.run(args, capture_output=True, text=True)
            conf.logger.error(result.stderr)
            conf.logger.debug(result.stdout)
            if "SUCCESS:0" in result.stdout:
                conf.logger.info("convert model %s success", convert_config['out_model'])
            else:
                conf.logger.error("convert model %s failed", convert_config['out_model'])

    def run_models(self):
        run_configs = self.run_configs.run_configs_

        # run_input_shapes = self.get_run_input_shapes()

        for run_config in run_configs:
            in_model_path = self.output_path + "/ms_models/" + run_config["in_model"]

            gold_in_path = self.output_path + "/gold_files/" + run_config["gold_in"]

            disabled = conf.has_key(run_config, key="disabled")
            if disabled:
                conf.logger.debug(
                    "disable test case %s gold_in: %s", run_config['in_model'], run_config['gold_in']
                )
                continue

            # dtypes = type_map[run_config["dtypes"]]
            run_input_shapes = self.get_run_input_shapes(run_config)

            input_shapes = run_input_shapes[run_config["gold_in"]]
            input_length = self.golden_confis.gold_input_shapes_dict[
                run_config["gold_in"]
            ]

            gold_in_param = ""
            gold_out_param = gold_in_path + "/model.onnx.out"
            for i in range(input_length):
                gold_in_param += gold_in_path + "/" + f"input_{i}.bin, "
            args = [
                self.mslite_benchmark_path,
                f"--modelFile={in_model_path}",
                "--device=CPU",
                "--loopCount=1",
                "--warmUpLoopCount=1",
                f"--inDataFile={gold_in_param[:-1]}",
                f"--benchmarkDataFile={gold_out_param}",
                f"--inputShape={input_shapes}",
            ]
            # if input_shapes != "None":
            #     args.append(
            #         f"--inputShape={input_shapes}",
            #     )
            run_command = ""
            for arg in args:
                run_command += arg + " "
            conf.logger.info("run command : %s", run_command)
            result = subprocess.run(args, capture_output=True, text=True)
            self.test_nums += 1

            conf.logger.info(result.stdout)
            # assert "Success" in result.stdout
            if "Success" in result.stdout:
                conf.logger.info("run model %s success", run_config['in_model'])
                self.num_success += 1
            else:
                conf.logger.error("run model %s failed", run_config['in_model'])
                self.failed_test_name.append(run_config["in_model"])
                self.failed_gold_in.append(gold_in_param)

    def exec_st(self):
        if self.model_configs.disable_all:
            conf.logger.info("diasabled test: %s", self.model_configs.op_configs_["op_name"])
        else:
            self.gen_onnx()
            self.gen_gold()
            self.convert_ops()
            self.run_models()
