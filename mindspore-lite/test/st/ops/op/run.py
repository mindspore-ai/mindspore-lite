import os
import argparse
import time
import optest
import configs as conf



def run_all(output_path, package):
    current_directory = os.getcwd()
    current_directory = current_directory+"/ops/op"
    op_directories = [
        d
        for d in os.listdir(current_directory)
        if os.path.isdir(os.path.join(current_directory, d))
    ]
    total_ms_op_num = len(op_directories)
    total_test_num = 0
    total_success_num = 0
    total_failed_test_name = []
    total_failed_gold_in = []
    total_onnx_op_num = 0
    for op_ in op_directories:
        yaml_files = [
            f
            for f in os.listdir(os.path.join(current_directory, op_))
            if f.endswith(".yaml")
        ]
        total_onnx_op_num += len(yaml_files)
        for yaml_file in yaml_files:
            yaml_path = current_directory + "/" + op_ + "/" + yaml_file
            conv_test_suit = optest.OpTest(yaml_path, output_path, package)
            conv_test_suit.exec_st()
            total_test_num += conv_test_suit.test_nums
            total_success_num += conv_test_suit.num_success
            if conv_test_suit.failed_test_name != []:
                total_failed_test_name.append(set(conv_test_suit.failed_test_name))
            total_failed_gold_in.append(conv_test_suit.failed_gold_in)
    conf.logger.info("total mslite op nums: %d", total_ms_op_num)
    conf.logger.info("total onnx op nums: %d", total_onnx_op_num)
    conf.logger.info("total test num: %d", total_test_num)
    conf.logger.info("total success: %d", total_success_num)
    conf.logger.info("failed model name: %s", total_failed_test_name)
    tmp_file = os.path.join(current_directory, "../failed_tmp.txt")
    with open(os.path.realpath(tmp_file), 'w') as file:
        file.write("failed_op_name: ")
        for item in total_failed_test_name:
            file.write(f"{item}")
        file.write("\n")


def run_one(output_path, op_name, package):
    current_directory = os.getcwd()
    op_dir = os.path.join(current_directory+"/ops/op", op_name)
    yaml_files = [f for f in os.listdir(op_dir) if f.endswith(".yaml")]
    total_test_num = 0
    total_success_num = 0
    total_failed_test_name = []
    total_failed_gold_in = []
    for yaml_file in yaml_files:
        yaml_path = op_dir + "/" + yaml_file
        conv_test_suit = optest.OpTest(yaml_path, output_path, package)
        conv_test_suit.exec_st()
        total_test_num += conv_test_suit.test_nums
        total_success_num += conv_test_suit.num_success
        if conv_test_suit.failed_test_name != []:
            total_failed_test_name.append(set(conv_test_suit.failed_test_name))
        total_failed_gold_in.append(conv_test_suit.failed_gold_in)
    conf.logger.info("total test num: %d", total_test_num)
    conf.logger.info("total success: %d", total_test_num)
    conf.logger.info("failed model name: %d", total_failed_test_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run all test or one op")
    parser.add_argument("-a", "--all", action="store_true", help="是否测试全部算子")
    parser.add_argument("-o", "--output", type=str, help="输出路径", required=True)
    parser.add_argument("-n", "--name", type=str, help="测试算子名称")
    parser.add_argument("-p", "--package", type=str, help="二进制包路径")
    args = parser.parse_args()
    if args.all and (args.name is not None):
        print("Can't set -a and -n same time!")

    start_time = time.time()
    if args.all:
        run_all(args.output, args.package)
    else:
        run_one(args.output, args.name, args.package)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"cost time: {execution_time:.6f} s")
