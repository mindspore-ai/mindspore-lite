#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

source ./scripts/base_functions.sh

# Run on arm64 platform:
function Run_arm64() {
    # Unzip arm64 runtime and converter
    cd ${arm64_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-*.tar.gz || exit 1
    # $1:framework;
    cd ${arm64_path}/mindspore-lite-${version}-linux-*/ || exit 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./tools/converter/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/glog
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/dnnl
    cp tools/benchmark/benchmark ./ || exit 1
    # Run converted models:
    # $1:cfgFileList; $2:modelPath; $3:dataPath; $4:logFile; $5:resultFile; $6:platform; $7:processor; $8:phoneId;
    Run_Benchmark "${arm64_cfg_file_list[*]}" $models_path $models_path $run_arm64_log_file $run_benchmark_result_file 'arm64_cloud' 'CPU' '' $arm64_fail_not_return
}

# Example:sh run_benchmark_arm64_cloud_cpu.sh -r /home/temp_test -m /home/temp_test/models -e arm64_cloud_mindir
while getopts "r:m:e:p:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${OPTARG}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${OPTARG}"
            ;;
        p)
            arm64_fail_not_return=${OPTARG}
            echo "arm64_fail_not_return is ${OPTARG}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

basepath=$(pwd)
echo ${basepath}
arm64_path=${release_path}/linux_aarch64/cloud_fusion
cd ${arm64_path}
file_name=$(ls *-linux-*.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}
cd -

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

models_mindir_config=${basepath}/../${config_folder}/cloud_infer/models_mindir_cloud.cfg
models_mindir_reconstitution_config=${basepath}/../${config_folder}/models_mindir_reconstitution_cloud_process_only.cfg
# Prepare the config file list
arm64_cfg_file_list=("$models_mindir_config" "$models_mindir_reconstitution_config")

# Write benchmark result to temp file
run_benchmark_result_file=${basepath}/run_benchmark_result.txt
echo ' ' > ${run_benchmark_result_file}

run_arm64_log_file=${basepath}/run_arm64_cloud_log.txt
echo 'run arm64 cloud logs: ' > ${run_arm64_log_file}

backend=${backend:-"all"}
isFailed=0

if [[ $backend == "all" || $backend == "arm64_cloud_mindir" ]]; then
    # Run on arm64 cloud
    echo "start Run arm64 mindir cloud  $backend..."
    Run_arm64
    Run_arm64_status=$?
    # Check benchmark result and return value
    if [[ ${Run_arm64_status} != 0 ]];then
        echo "Run_arm64_mindir_cloud failed"
        cat ${run_arm64_log_file}
        isFailed=1
    fi
fi

echo "Run_arm64_mindir_cloud is ended"
Print_Benchmark_Result $run_benchmark_result_file

exit ${isFailed}
