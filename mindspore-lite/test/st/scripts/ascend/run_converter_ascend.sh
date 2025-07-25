#!/bin/bash

# Run converter for ascend x86 platform:
function Run_Converter() {
    cd ${x86_path} || exit 1
    tar -zxf mindspore-lite-${version}-linux-${arch}.tar.gz || exit 1
    cd ${x86_path}/mindspore-lite-${version}-linux-${arch}/ || exit 1
    cp tools/converter/converter/converter_lite ./ || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./tools/converter/lib/
    export LITE_HOME=$(pwd)
    echo "LITE_HOME:${LITE_HOME}"

    # Convert models:
    # $1:cfgFileList; $2:inModelPath; $3:outModelPath; $4:logFile; $5:resultFile; $6:faile_not_return;
    Convert "${ascend_cfg_file_list[*]}" $models_path $ms_models_path $run_converter_log_file $run_converter_result_file $ascend_fail_not_return $compile_type
}

# source ascend env
export ASCEND_PATH=/usr/local/Ascend
if [ -d "${ASCEND_PATH}/ascend-toolkit" ]; then
    source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
else
    source ${ASCEND_PATH}/latest/bin/setenv.bash
fi

backend=$1
device_id=$2
arch=$3
ascend_fail_not_return=$4
user_name=${USER}
echo "Current user is ${USER}"
benchmark_test=/home/${user_name}/benchmark_test/${device_id}
echo "Benchmark test path is ${benchmark_test}"

x86_path=${benchmark_test}
file_name=$(ls ${x86_path}/*-linux-${arch}.tar.gz)
IFS="-" read -r -a file_name_array <<< "$file_name"
version=${file_name_array[2]}

ms_models_path=${benchmark_test}/ms_models
rm -rf ${ms_models_path}
mkdir -p ${ms_models_path}

model_data_path=/home/workspace/mindspore_dataset/mslite
models_path=${model_data_path}/models/hiai

ascend_cfg_file_list=()
if [[ ${backend} =~ "lite" ]]; then
    compile_type="lite"
    models_ascend_config=${benchmark_test}/models_ascend_lite.cfg
    ascend_cfg_file_list=("$models_ascend_config")
elif [[ ${backend} =~ "cloud" ]]; then
    compile_type="cloud"
    models_ascend_config=${benchmark_test}/models_ascend_cloud.cfg
    models_ascend_with_config=${benchmark_test}/models_with_config_cloud_ascend.cfg
    ascend_cfg_file_list=("$models_ascend_config" "$models_ascend_with_config")
    if [[ ${backend} =~ "_ge" ]]; then
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib
        models_ascend_config=${benchmark_test}/models_ascend_ge_cloud.cfg
        models_ascend_on_the_fly_quant_config=${benchmark_test}/models_ascend_on_the_fly_quant_ge_cloud.cfg
        models_ascend_fake_model_on_the_fly_quant_config=${benchmark_test}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg
        models_ascend_fake_model_full_quant_config=${benchmark_test}/models_ascend_fake_model_full_quant_ge_cloud.cfg
        ascend_cfg_file_list=("$models_ascend_config" "$models_ascend_on_the_fly_quant_config" "$models_ascend_fake_model_on_the_fly_quant_config" "$models_ascend_fake_model_full_quant_config")
    fi
fi

# Write converter result to temp file
run_converter_log_file=${benchmark_test}/run_converter_log.txt
echo ' ' > ${run_converter_log_file}

run_converter_result_file=${benchmark_test}/run_converter_result.txt
echo ' ' > ${run_converter_result_file}

# Run converter
source ${benchmark_test}/base_functions.sh
echo "Start to run converter in ${backend}, device id ${device_id} ..."
Run_Converter
if [[ $? = 0 ]]; then
    echo "Run converter success"
    Print_Converter_Result $run_converter_result_file
else
    echo "Run converter failed"
    cat ${run_converter_log_file}
    Print_Converter_Result $run_converter_result_file
    exit 1
fi

# Run Benchmark
source ${benchmark_test}/run_benchmark_ascend.sh -v $version -b $backend -d $device_id -a ${arch} -c ${compile_type} -p ${ascend_fail_not_return}
Run_Benchmark_status=$?
exit ${Run_Benchmark_status}
