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

# Parallel execution constants
NUM_CARDS=${NUM_PARALLEL_CARDS:-4}
START_CARD=${START_CARD_ID:-0}

# Per-card benchmark runner — called in a subprocess.
# Reads models from sub_cfg_file and runs ./benchmark for each.
# $1: sub_cfg_file; $2: card_id; $3: pkg_dir; $4: ms_models_path; $5: model_data_path;
# $6: result_file; $7: log_file; $8: ascend_device; $9: compile_type
function Run_Benchmark_Card() {
    local sub_cfg_file=$1
    local card_id=$2
    local pkg_dir=$3
    local ms_models_path=$4
    local model_data_path=$5
    local result_file=$6
    local log_file=$7
    local ascend_device=$8
    local compile_type=${9:-"cloud"}
    local _fail=0

    if [[ ! -s "${sub_cfg_file}" ]]; then
        return 0
    fi

    export ASCEND_DEVICE_ID=${card_id}
    cd ${pkg_dir} || return 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib:./tools/converter/lib/

    # F5: Set GE backend policy when applicable
    if [[ ${backend} =~ "_ge" ]]; then
        export ASCEND_BACK_POLICY="ge"
    fi

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
        mode model_file input_files output_file data_path acc_limit enableFp16 \
        run_result elapsed_time ret extra_info use_parallel_predict infix \
        spec_shapes input_names sub_cfg_name model_type config_file
    data_path=${model_data_path}'/models/hiai/input_output/'
    sub_cfg_name=$(basename "${sub_cfg_file}")

    while read line; do
        line_info=${line}
        if [[ $line_info == \#* || $line_info == "" ]]; then
            continue
        fi

        model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
        spec_acc_limit=$(echo ${line_info} | awk -F ' ' '{print $2}')

        model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
        input_info=$(echo ${model_info} | awk -F ';' '{print $2}')
        input_shapes=$(echo ${model_info} | awk -F ';' '{print $3}')
        mode=$(echo ${model_info} | awk -F ';' '{print $4}')
        extra_info=$(echo ${model_info} | awk -F ';' '{print $5}')
        echo "Benchmarking ${model_name} on device ${card_id} ......" >> "${log_file}"
        input_num=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}')
        input_names=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}')

        # Determine infix and shapes from cfg file name
        spec_shapes=""
        infix=""
        config_file=""
        if [[ ${sub_cfg_name} =~ "wcfg" || ${extra_info} =~ "parallel_predict" ]]; then
            # with_config / parallel_predict: compute input shapes from name:shape pairs
            if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
                if [[ ${input_num} == "" ]]; then
                    input_num=1
                fi
                IFS="," read -r -a name_array <<< ${input_names}
                IFS=":" read -r -a shape_array <<< ${input_shapes}
                for i in $(seq 0 $((${input_num}-1))); do
                    spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
                done
            fi
            config_file="${pkg_dir}/${model_name}.config"
        elif [[ ${sub_cfg_name} =~ "on_the_fly" ]]; then
            infix="_on_the_fly_quant"
        elif [[ ${sub_cfg_name} =~ "fake_full" ]]; then
            infix="_full_quant"
        fi

        if [[ ${model_name##*.} == "caffemodel" ]]; then
            model_name=${model_name%.*}
        fi

        # F1: Detect parallel_predict
        use_parallel_predict="false"
        if [[ ${extra_info} =~ "parallel_predict" ]]; then
            use_parallel_predict="true"
        fi

        if [[ ${compile_type} == "cloud" ]]; then
            model_file=${ms_models_path}'/'${model_name}${infix}'.mindir'
        else
            model_file=${ms_models_path}'/'${model_name}${infix}'.ms'
        fi
        if [[ ${extra_info} =~ "parallel_predict" ]]; then
            export BENCHMARK_WEIGHT_PATH=${model_file}
        fi

        # F3: model_type-based input/output file paths
        model_type=${model_name##*.}
        input_files=""
        output_file=""
        if [[ ${model_type} == "mindir" || ${model_type} == "ms" ]]; then
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.ms.bin'
            else
                for i in $(seq 1 ${input_num}); do
                    input_files=${input_files}${data_path}'input/'${model_name}'.ms.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.ms.out'
        else
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.bin'
            else
                for i in $(seq 1 ${input_num}); do
                    input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.out'
        fi
        # F2: with_config/quant cfg overrides input files to .bin
        if [[ ${sub_cfg_name} =~ "wcfg" || ${sub_cfg_name} =~ "on_the_fly" || ${sub_cfg_name} =~ "fake_full" ]]; then
            input_files=""
            output_file=""
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.bin'
            else
                for i in $(seq 1 ${input_num}); do
                    input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.out'
        fi

        acc_limit="0.5"
        if [[ ${spec_acc_limit} != "" ]]; then
            acc_limit="${spec_acc_limit}"
        elif [[ ${mode} == "fp16" ]]; then
            acc_limit="5"
        fi

        enableFp16="false"
        if [[ ${mode} == "fp16" ]]; then
            enableFp16="true"
        fi

        Run_Benchmark_Model "${model_file}" "${input_files}" "${output_file}" \
            "${acc_limit}" "${enableFp16}" "${ascend_device}" \
            "${use_parallel_predict}" "${spec_shapes}" "${log_file}"
        ret=$?
        if [ ${ret} = 0 ]; then
            if [[ ${extra_info} =~ "parallel_predict" ]]; then
                echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} parallel_pass" >> ${result_file}
            else
                echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} pass" >> ${result_file}
            fi
        else
            if [[ ${extra_info} =~ "parallel_predict" ]]; then
                echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} parallel_failed" >> ${result_file}
            else
                echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} failed" >> ${result_file}
            fi
            _fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
    done < ${sub_cfg_file}
    return ${_fail}
}

# Per-card cloud benchmark runner — runs cloud mindir models from dataset path.
# $1: sub_cfg_file; $2: card_id; $3: pkg_dir; $4: models_path (dataset);
# $5: model_data_path; $6: result_file; $7: log_file; $8: ascend_device
function Run_Cloud_Benchmark_Card() {
    local sub_cfg_file=$1
    local card_id=$2
    local pkg_dir=$3
    local models_path=$4
    local model_data_path=$5
    local result_file=$6
    local log_file=$7
    local ascend_device=$8
    local _fail=0

    if [[ ! -s "${sub_cfg_file}" ]]; then
        return 0
    fi

    export ASCEND_DEVICE_ID=${card_id}
    cd ${pkg_dir} || return 1
    cp tools/benchmark/benchmark ./ || return 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./runtime/lib:./tools/converter/lib/:./runtime/third_party/glog:./runtime/third_party/libjpeg-turbo/lib:./runtime/third_party/dnnl

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
        mode model_file input_files output_file data_path acc_limit enableFp16 \
        run_result ret input_info input_names spec_shapes
    data_path=${model_data_path}'/models/hiai/input_output/'

    while read line; do
        line_info=${line}
        if [[ $line_info == \#* || $line_info == "" ]]; then
            continue
        fi

        model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
        spec_acc_limit=$(echo ${line_info} | awk -F ' ' '{print $2}')

        model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
        input_info=$(echo ${model_info} | awk -F ';' '{print $2}')
        input_shapes=$(echo ${model_info} | awk -F ';' '{print $3}')
        mode=$(echo ${model_info} | awk -F ';' '{print $4}')
        echo "Benchmarking ${model_name} on device ${card_id} ......" >> "${log_file}"
        input_num=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}')
        input_names=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}')

        spec_shapes=""
        if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
            if [[ ${input_num} == "" ]]; then
                input_num=1
            fi
            IFS="," read -r -a name_array <<< ${input_names}
            IFS=":" read -r -a shape_array <<< ${input_shapes}
            for i in $(seq 0 $((${input_num}-1))); do
                spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
            done
        fi

        if [[ ${model_name##*.} == "caffemodel" ]]; then
            model_name=${model_name%.*}
        fi

        model_file=${models_path}'/'${model_name}'.mindir'
        input_files=""
        output_file=""
        if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
            input_files=${data_path}'input/'${model_name}'.bin'
        else
            for i in $(seq 1 ${input_num}); do
                input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
            done
        fi
        output_file=${data_path}'output/'${model_name}'.out'

        # -1 means skip accuracy check
        if [[ ${spec_acc_limit} == "-1" ]]; then
            input_files=""
            output_file=""
        fi

        acc_limit="0.5"
        if [[ ${spec_acc_limit} != "" && ${spec_acc_limit} != "-1" ]]; then
            acc_limit="${spec_acc_limit}"
        elif [[ ${mode} == "fp16" ]]; then
            acc_limit="5"
        fi

        enableFp16="false"
        if [[ ${mode} == "fp16" ]]; then
            enableFp16="true"
        fi

        Run_Benchmark_Model "${model_file}" "${input_files}" "${output_file}" \
            "${acc_limit}" "${enableFp16}" "${ascend_device}" \
            "false" "${spec_shapes}" "${log_file}"
        ret=$?
        if [ ${ret} = 0 ]; then
            echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} pass" >> ${result_file}
        else
            echo "${backend}: ${model_name} ${MODEL_ELAPSED_TIME} failed" >> ${result_file}
            _fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
    done < ${sub_cfg_file}
    return ${_fail}
}

function PrePareLocal() {
  echo "Start to copy local file"
  rm -rf ${benchmark_test_path}
  mkdir -p ${benchmark_test_path}

  cp ./scripts/base_functions.sh ${benchmark_test_path} || exit 1
  cp ./scripts/run_benchmark_python.sh ${benchmark_test_path} || exit 1
  cp -r ./python ${benchmark_test_path} || exit 1
  cp -r ./cpp ${benchmark_test_path} || exit 1
  cp -r ./java ${benchmark_test_path} || exit 1
  cp ./scripts/ascend/*.sh ${benchmark_test_path} || exit 1
  cp ./scripts/cloud_infer/run_benchmark_cloud_ascend.sh ${benchmark_test_path} || exit 1
  if [[ ${backend} =~ "_cloud" ]]; then
      models_ascend_config=./../${config_folder}/models_ascend_cloud.cfg
      if [[ ${backend} =~ "_ge" ]]; then
          models_ascend_config=./../${config_folder}/models_ascend_ge_cloud.cfg
      fi
      cp ${models_ascend_config} ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_python_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_mindir_cloud_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_mindir_cloud_java_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/cloud_infer/models_with_config_cloud_ascend.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/ascend/*.config ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_on_the_fly_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp ./../${config_folder}/models_ascend_fake_model_full_quant_ge_cloud.cfg ${benchmark_test_path} || exit 1
      cp -r ./../${config_folder}/quant ${benchmark_test_path} || exit 1
  else
      cp ./../${config_folder}/models_ascend_lite.cfg ${benchmark_test_path} || exit 1
  fi
  if [[ ${backend} =~ "arm" ]]; then
      if [[ ${backend} =~ "_cloud" ]]; then
          md5sum ${release_path}/linux_aarch64/cloud_fusion/*-linux-${arch}.tar.gz
          cp ${release_path}/linux_aarch64/cloud_fusion/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
          md5sum ${benchmark_test_path}/*-linux-${arch}.tar.gz
          cp ${release_path}/linux_aarch64/cloud_fusion/*.whl ${benchmark_test_path} || exit 1
      else
          cp ${release_path}/linux_aarch64/ascend/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      fi
  else
      if [[ ${backend} =~ "_cloud" ]]; then
          cp ${release_path}/centos_x86/cloud_fusion/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      else
          cp ${release_path}/centos_x86/ascend/*-linux-${arch}.tar.gz ${benchmark_test_path} || exit 1
      fi
  fi
  echo "Copy file success"
}

function PrePareRemote() {
  echo "Start to copy remote file"
  ssh ${user_name}@${device_ip} "rm -rf ${benchmark_test_path}; mkdir -p ${benchmark_test_path}" || exit 1

  scp ./scripts/run_benchmark_python.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp ./scripts/base_functions.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./python ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./cpp ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp -r ./java ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  scp ./scripts/ascend/*.sh ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  if [[ ${backend} =~ "_cloud" ]]; then
      models_ascend_config=./../${config_folder}/models_ascend_cloud.cfg
      if [[ ${backend} =~ "_ge" ]]; then
          models_ascend_config=./../${config_folder}/models_ascend_ge_cloud.cfg
      fi
      scp ${models_ascend_config} ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/models_python_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_mindir_cloud_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_mindir_cloud_java_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/cloud_infer/models_with_config_cloud_ascend.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/ascend/*.config ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/models_ascend_on_the_fly_quant_ge_cloud.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp ./../${config_folder}/models_ascend_fake_model_full_quant_ge_cloud.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      scp -r ./../${config_folder}/quant ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  else
      scp ./../${config_folder}/models_ascend_lite.cfg ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
  fi
  if [[ ${backend} =~ "arm" ]]; then
      if [[ ${backend} =~ "_cloud" ]]; then
          scp ${release_path}/linux_aarch64/cloud_fusion/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
          scp ${release_path}/linux_aarch64/cloud_fusion/*.whl ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      else
          scp ${release_path}/linux_aarch64/ascend/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      fi
  else
      if [[ ${backend} =~ "_cloud" ]]; then
          scp ${release_path}/centos_x86/cloud_fusion/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      else
          scp ${release_path}/centos_x86/ascend/*-linux-${arch}.tar.gz ${user_name}@${device_ip}:${benchmark_test_path} || exit 1
      fi
  fi
  echo "Copy file success"
}

# Parallel converter — splits cfg files across cards, each card runs Convert() in isolated work dir.
# $1: num_cards; $2: start_card
function Run_Converter_Parallel() {
    local num_cards=$1
    local start_card=$2
    local pkg_dir=${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}
    local model_data_path=/home/workspace/mindspore_dataset/mslite
    local models_path=${model_data_path}/models/hiai
    local ms_models_path=${benchmark_test_path}/ms_models

    # Pre-setup (serial, once)
    cd ${benchmark_test_path} || return 1
    if [[ ! -d "${pkg_dir}" ]]; then
        tar -zxf *-linux-${arch}.tar.gz || return 1
    fi
    cd ${pkg_dir} || return 1
    cp tools/converter/converter/converter_lite ./ || return 1
    cd ${benchmark_test_path} || return 1
    rm -rf ${ms_models_path}
    mkdir -p ${ms_models_path}

    # Determine compile type
    local compile_type="cloud"
    if [[ ${backend} =~ "lite" ]]; then
        compile_type="lite"
    fi

    # Determine cfg files (mirrors run_converter_ascend.sh logic)
    local conv_cfg_files=()
    if [[ ${backend} =~ "lite" ]]; then
        conv_cfg_files=("${benchmark_test_path}/models_ascend_lite.cfg")
    elif [[ ${backend} =~ "_ge" ]]; then
        conv_cfg_files=(
            "${benchmark_test_path}/models_ascend_ge_cloud.cfg"
            "${benchmark_test_path}/models_ascend_on_the_fly_quant_ge_cloud.cfg"
            "${benchmark_test_path}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg"
            "${benchmark_test_path}/models_ascend_fake_model_full_quant_ge_cloud.cfg"
        )
    else
        conv_cfg_files=(
            "${benchmark_test_path}/models_ascend_cloud.cfg"
            "${benchmark_test_path}/models_with_config_cloud_ascend.cfg"
        )
    fi

    # Split each cfg across cards
    echo "Splitting converter configs into ${num_cards} parts ..."
    for cfg in "${conv_cfg_files[@]}"; do
        Split_Config "${cfg}" ${num_cards} "${cfg%.cfg}_conv_sub" || return 1
    done

    # Launch parallel converter
    local pids=()
    local conv_result_files=()
    local conv_log_files=()

    for ((card=0; card<num_cards; card++)); do
        local conv_result="${benchmark_test_path}/converter_result_card${card}.txt"
        local conv_log="${benchmark_test_path}/converter_log_card${card}.txt"
        conv_result_files+=("${conv_result}")
        conv_log_files+=("${conv_log}")
        true > "${conv_result}"
        true > "${conv_log}"

        # Collect sub-cfg list for this card
        local sub_cfgs=()
        for cfg in "${conv_cfg_files[@]}"; do
            local sub_cfg="${cfg%.cfg}_conv_sub_card${card}.cfg"
            if [[ -s "${sub_cfg}" ]]; then
                sub_cfgs+=("${sub_cfg}")
            fi
        done

        (
            local card_work_dir="${benchmark_test_path}/conv_work_${card}"
            mkdir -p ${card_work_dir}
            cd ${card_work_dir} || exit 1
            ln -sf ${pkg_dir}/converter_lite ./
            export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${pkg_dir}/tools/converter/lib/:${pkg_dir}/tools/converter/third_party/glog/lib
            Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Convert "${sub_cfgs[*]}" ${models_path} ${ms_models_path} "${conv_log}" "${conv_result}" "${ascend_fail_not_return}" "${compile_type}"
            exit $?
        ) &
        pids+=($!)
    done

    # Wait for all converter cards, track timeouts
    local conv_fail=0
    local conv_timeout_cards=()
    for ((i=0; i<${#pids[@]}; i++)); do
        wait ${pids[$i]}
        local ec=$?
        if [ ${ec} -eq 124 ]; then
            echo "Converter card $((start_card + i)) timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            conv_timeout_cards+=("${i}")
            conv_fail=1
        elif [ ${ec} -ne 0 ]; then
            echo "Converter card ${i} failed (exit ${ec})"
            conv_fail=1
        fi
    done

    # Merge converter logs and results
    local merged_conv_log=${benchmark_test_path}/run_converter_log.txt
    local merged_conv_result=${benchmark_test_path}/run_converter_result.txt
    true > "${merged_conv_log}"
    true > "${merged_conv_result}"
    for ((i=0; i<${#conv_log_files[@]}; i++)); do
        echo "--- Card ${i} converter log ---" >> "${merged_conv_log}"
        cat "${conv_log_files[$i]}" >> "${merged_conv_log}"
    done
    for ((i=0; i<${#conv_result_files[@]}; i++)); do
        cat "${conv_result_files[$i]}" >> "${merged_conv_result}"
    done
    # Append timeout notices for converter cards that timed out
    for tidx in "${conv_timeout_cards[@]}"; do
        echo "[converter] Card $((start_card + tidx)) timed out after ${WATCHDOG_TIMEOUT_SEC}s" >> "${merged_conv_log}"
    done

    # Cleanup per-card work dirs
    rm -rf ${benchmark_test_path}/conv_work_*

    return ${conv_fail}
}

# Run parallel benchmark across NUM_CARDS cards, then run post-steps (Python ST, C++ quick start).
function Run_Parallel_Benchmark() {
    local num_cards=$1
    local start_card=$2
    local pkg_dir=${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}
    local ms_models=${benchmark_test_path}/ms_models
    local model_data_path=/home/workspace/mindspore_dataset/mslite

    # Determine ascend device type (always Ascend for 310/310P)
    local ascend_device="Ascend"

    # Determine compile type for model file extension
    local compile_type="cloud"
    if [[ ${backend} =~ "lite" ]]; then
        compile_type="lite"
    fi

    # Select cfg files based on backend type (mirrors run_converter_ascend.sh logic)
    local main_cfg=""
    local wcfg="${benchmark_test_path}/models_with_config_cloud_ascend.cfg"
    local on_the_fly_cfg="${benchmark_test_path}/models_ascend_on_the_fly_quant_ge_cloud.cfg"
    local fake_on_the_fly_cfg="${benchmark_test_path}/models_ascend_fake_model_on_the_fly_quant_ge_cloud.cfg"
    local fake_full_cfg="${benchmark_test_path}/models_ascend_fake_model_full_quant_ge_cloud.cfg"
    if [[ ${backend} =~ "_ge" ]]; then
        main_cfg="${benchmark_test_path}/models_ascend_ge_cloud.cfg"
    elif [[ ${backend} =~ "cloud" ]]; then
        main_cfg="${benchmark_test_path}/models_ascend_cloud.cfg"
    else
        main_cfg="${benchmark_test_path}/models_ascend_lite.cfg"
    fi

    # Split benchmark configs round-robin across cards
    echo "Splitting benchmark configs into ${num_cards} parts ..."
    Split_Config "${main_cfg}" $num_cards "${benchmark_test_path}/ascend_cloud_sub" || return 1
    # wcfg only applies to non-GE cloud backends (converter skips it for _ge)
    if [[ ! ${backend} =~ "_ge" && -f "${wcfg}" ]]; then
        Split_Config "${wcfg}" $num_cards "${benchmark_test_path}/ascend_wcfg_sub" || return 1
    fi
    # Quant cfgs only apply to _ge backends (converter only converts them for _ge)
    if [[ ${backend} =~ "_ge" ]]; then
        if [[ -f "${on_the_fly_cfg}" ]]; then
            Split_Config "${on_the_fly_cfg}" $num_cards "${benchmark_test_path}/ascend_on_the_fly_sub" || return 1
        fi
        if [[ -f "${fake_on_the_fly_cfg}" ]]; then
            Split_Config "${fake_on_the_fly_cfg}" $num_cards "${benchmark_test_path}/ascend_fake_on_the_fly_sub" || return 1
        fi
        if [[ -f "${fake_full_cfg}" ]]; then
            Split_Config "${fake_full_cfg}" $num_cards "${benchmark_test_path}/ascend_fake_full_sub" || return 1
        fi
    fi

    # Copy benchmark binary once before parallel launch (avoids ETXTBSY across cards)
    cd ${pkg_dir} || return 1
    cp tools/benchmark/benchmark ./ || { echo "ERROR: Failed to copy benchmark binary to ${pkg_dir}"; return 1; }
    cd - > /dev/null

    local pids=()
    local result_files=()
    local log_files=()
    local fail=0

    for ((card=0; card<num_cards; card++)); do
        local card_id=$((start_card + card))
        local sub_cfg="${benchmark_test_path}/ascend_cloud_sub_card${card}.cfg"
        local sub_wcfg="${benchmark_test_path}/ascend_wcfg_sub_card${card}.cfg"
        local sub_on_the_fly="${benchmark_test_path}/ascend_on_the_fly_sub_card${card}.cfg"
        local sub_fake_on_the_fly="${benchmark_test_path}/ascend_fake_on_the_fly_sub_card${card}.cfg"
        local sub_fake_full="${benchmark_test_path}/ascend_fake_full_sub_card${card}.cfg"
        local result_file="${benchmark_test_path}/benchmark_result_card${card}.txt"
        local log_file="${benchmark_test_path}/benchmark_log_card${card}.txt"
        result_files+=("${result_file}")
        log_files+=("${log_file}")
        true > "${result_file}"
        true > "${log_file}"

        (
            local ret=0
            local this_ret=0
            # Main benchmark config
            echo "Card ${card_id}: Start main benchmark" >> "${log_file}"
            Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_Card "${sub_cfg}" "${card_id}" "${pkg_dir}" "${ms_models}" \
                "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}" "${compile_type}"
            this_ret=$?; [[ ${this_ret} -ne 0 ]] && ret=${this_ret}
            echo "Card ${card_id}: Main benchmark done (ret=${this_ret})" >> "${log_file}"
            [[ ${this_ret} -eq 124 ]] && exit ${ret}
            # With-config benchmark
            if [[ -f "${sub_wcfg}" && -s "${sub_wcfg}" ]]; then
                echo "Card ${card_id}: Start wcfg benchmark" >> "${log_file}"
                Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_Card "${sub_wcfg}" "${card_id}" "${pkg_dir}" "${ms_models}" \
                    "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}" "${compile_type}"
                this_ret=$?; [[ ${this_ret} -ne 0 ]] && ret=${this_ret}
                echo "Card ${card_id}: Wcfg benchmark done (ret=${this_ret})" >> "${log_file}"
                [[ ${this_ret} -eq 124 ]] && exit ${ret}
            fi
            # On-the-fly quant
            if [[ -f "${sub_on_the_fly}" && -s "${sub_on_the_fly}" ]]; then
                echo "Card ${card_id}: Start on_the_fly benchmark" >> "${log_file}"
                Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_Card "${sub_on_the_fly}" "${card_id}" "${pkg_dir}" "${ms_models}" \
                    "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}" "${compile_type}"
                this_ret=$?; [[ ${this_ret} -ne 0 ]] && ret=${this_ret}
                echo "Card ${card_id}: On_the_fly benchmark done (ret=${this_ret})" >> "${log_file}"
                [[ ${this_ret} -eq 124 ]] && exit ${ret}
            fi
            # Fake model on-the-fly quant
            if [[ -f "${sub_fake_on_the_fly}" && -s "${sub_fake_on_the_fly}" ]]; then
                echo "Card ${card_id}: Start fake_on_the_fly benchmark" >> "${log_file}"
                Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_Card "${sub_fake_on_the_fly}" "${card_id}" "${pkg_dir}" "${ms_models}" \
                    "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}" "${compile_type}"
                this_ret=$?; [[ ${this_ret} -ne 0 ]] && ret=${this_ret}
                echo "Card ${card_id}: Fake_on_the_fly benchmark done (ret=${this_ret})" >> "${log_file}"
                [[ ${this_ret} -eq 124 ]] && exit ${ret}
            fi
            # Fake model full quant
            if [[ -f "${sub_fake_full}" && -s "${sub_fake_full}" ]]; then
                echo "Card ${card_id}: Start fake_full benchmark" >> "${log_file}"
                Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_Card "${sub_fake_full}" "${card_id}" "${pkg_dir}" "${ms_models}" \
                    "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}" "${compile_type}"
                this_ret=$?; [[ ${this_ret} -ne 0 ]] && ret=${this_ret}
                echo "Card ${card_id}: Fake_full benchmark done (ret=${this_ret})" >> "${log_file}"
                [[ ${this_ret} -eq 124 ]] && exit ${ret}
            fi
            echo "Card ${card_id}: All done" >> "${log_file}"
            exit ${ret}
        ) &
        pids+=($!)
    done

    # Wait for all benchmark processes, track timeouts
    local timeout_cards=()
    for ((i=0; i<${#pids[@]}; i++)); do
        wait ${pids[$i]}
        ec=$?
        if [ ${ec} -eq 124 ]; then
            echo "Benchmark card $((start_card + i)) timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            if [[ -f "${log_files[$i]}" ]]; then
                echo "--- Card $((start_card + i)) log (timeout) ---"
                cat "${log_files[$i]}"
            fi
            timeout_cards+=("${i}")
            fail=1
        elif [ ${ec} -ne 0 ]; then
            local failed_card=$((start_card + i))
            echo "Benchmark card ${failed_card} failed (exit ${ec})"
            if [[ -f "${log_files[$i]}" ]]; then
                echo "--- Card ${failed_card} log ---"
                cat "${log_files[$i]}"
            fi
            fail=1
        fi
    done

    # Merge benchmark logs and results with device identification
    # Use separate file names to avoid overwrite by cloud fusion benchmark (H1)
    local merged_bench_log=${benchmark_test_path}/run_benchmark_parallel_log.txt
    local merged_bench_result=${benchmark_test_path}/run_benchmark_parallel_result.txt
    true > "${merged_bench_log}"
    true > "${merged_bench_result}"
    for ((i=0; i<${#log_files[@]}; i++)); do
        echo "--- Ascend:$((start_card + i)) benchmark log ---" >> "${merged_bench_log}"
        cat "${log_files[$i]}" >> "${merged_bench_log}"
    done
    for ((i=0; i<${#result_files[@]}; i++)); do
        cat "${result_files[$i]}" >> "${merged_bench_result}"
    done
    for tidx in "${timeout_cards[@]}"; do
        local tdev=$((start_card + tidx))
        echo "Ascend:${tdev} benchmark (timeout) TIMEOUT" >> "${merged_bench_result}"
        echo "[benchmark] Ascend:${tdev} timed out after ${WATCHDOG_TIMEOUT_SEC}s" >> "${merged_bench_log}"
    done

    return ${fail}
}

# Run post-benchmark steps (Java benchmark, cloud inference, Python ST, C++ quick start).
function Run_Post_Steps() {
    local pkg_dir=${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}
    local ms_models=${benchmark_test_path}/ms_models
    local model_data_path=/home/workspace/mindspore_dataset/mslite
    local models_path=${model_data_path}/models/hiai
    local ascend_device="Ascend"
    local _post_fail=0

    cd ${benchmark_test_path} || exit 1
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${pkg_dir}/runtime/lib:${pkg_dir}/tools/converter/lib/

    # Set benchmark_test for scripts sourced from here (run_benchmark_cloud_ascend.sh
    # and run_benchmark_python.sh use benchmark_test, not benchmark_test_path)
    benchmark_test=${benchmark_test_path}

    # Run cloud fusion inference: Java (serial) + C++ benchmark (parallel across cards)
    if [[ ${backend} =~ "cloud" && ! ${backend} =~ "ge" ]]; then
        echo "Run cloud fusion inference benchmark"

        # Phase 1: Java benchmark (2 models, ~1 min) — serial via skip_benchmark flag, with watchdog
        OPTIND=1
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} source ${benchmark_test_path}/run_benchmark_cloud_ascend.sh \
            -v ${version} -b ${backend} -d ${device_id} -a ${arch} -c "cloud" -s "skip_benchmark"
        local java_ret=$?
        if [ ${java_ret} -eq 124 ]; then
            echo "Cloud fusion Java benchmark timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [ ${java_ret} -ne 0 ]; then
            echo "Cloud fusion Java benchmark failed"
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi

        # Phase 2: C++ cloud benchmark (18 models) — parallel across cards
        local cloud_cfg="${benchmark_test_path}/models_mindir_cloud_ascend.cfg"
        if [[ -f "${cloud_cfg}" ]]; then
            echo "Splitting cloud C++ config into ${NUM_CARDS} parts ..."
            Split_Config "${cloud_cfg}" ${NUM_CARDS} "${benchmark_test_path}/cloud_sub" || return 1

            local pids=()
            local cloud_result_files=()
            local cloud_log_files=()
            local cloud_fail=0
            local cloud_timeout_cards=()

            for ((card=0; card<NUM_CARDS; card++)); do
                local card_id=$((START_CARD + card))
                local sub_cfg="${benchmark_test_path}/cloud_sub_card${card}.cfg"
                local result_file="${benchmark_test_path}/cloud_result_card${card}.txt"
                local log_file="${benchmark_test_path}/cloud_log_card${card}.txt"
                cloud_result_files+=("${result_file}")
                cloud_log_files+=("${log_file}")
                true > "${result_file}"
                true > "${log_file}"

                (
                    local this_ret=0
                    echo "Card ${card_id}: Start cloud benchmark" >> "${log_file}"
                    Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Cloud_Benchmark_Card "${sub_cfg}" "${card_id}" "${pkg_dir}" \
                        "${models_path}" "${model_data_path}" "${result_file}" "${log_file}" "${ascend_device}"
                    this_ret=$?
                    echo "Card ${card_id}: Cloud benchmark done (ret=${this_ret})" >> "${log_file}"
                    exit ${this_ret}
                ) &
                pids+=($!)
            done

            for ((i=0; i<${#pids[@]}; i++)); do
                wait ${pids[$i]}
                local ec=$?
                if [ ${ec} -eq 124 ]; then
                    echo "Cloud benchmark card $((START_CARD + i)) timed out after ${WATCHDOG_TIMEOUT_SEC}s"
                    if [[ -f "${cloud_log_files[$i]}" ]]; then
                        echo "--- Cloud card $((START_CARD + i)) log (timeout) ---"
                        cat "${cloud_log_files[$i]}"
                    fi
                    cloud_timeout_cards+=("${i}")
                    cloud_fail=1
                elif [ ${ec} -ne 0 ]; then
                    local cfail_card=$((START_CARD + i))
                    echo "Cloud benchmark card ${cfail_card} failed (exit ${ec})"
                    if [[ -f "${cloud_log_files[$i]}" ]]; then
                        echo "--- Cloud card ${cfail_card} log ---"
                        cat "${cloud_log_files[$i]}"
                    fi
                    cloud_fail=1
                fi
            done

            # Merge cloud results into run_benchmark_result.txt
            for ((i=0; i<${#cloud_result_files[@]}; i++)); do
                if [[ -f "${cloud_result_files[$i]}" ]]; then
                    cat "${cloud_result_files[$i]}" >> "${benchmark_test_path}/run_benchmark_result.txt"
                fi
            done
            # Merge cloud logs
            for ((i=0; i<${#cloud_log_files[@]}; i++)); do
                if [[ -f "${cloud_log_files[$i]}" ]]; then
                    echo "--- Ascend:$((START_CARD + i)) cloud benchmark log ---" >> ${benchmark_test_path}/run_benchmark_log.txt
                    cat "${cloud_log_files[$i]}" >> ${benchmark_test_path}/run_benchmark_log.txt
                fi
            done
            # Append TIMEOUT entries for timed-out cloud benchmark cards
            for tidx in "${cloud_timeout_cards[@]}"; do
                local tdev=$((START_CARD + tidx))
                echo "Ascend:${tdev} cloud_benchmark (timeout) TIMEOUT" >> "${benchmark_test_path}/run_benchmark_result.txt"
                echo "[cloud_benchmark] Ascend:${tdev} timed out after ${WATCHDOG_TIMEOUT_SEC}s" >> ${benchmark_test_path}/run_benchmark_log.txt
            done

            if [ ${cloud_fail} -ne 0 ]; then
                echo "Cloud fusion C++ benchmark failed"
                Print_Benchmark_Result ${benchmark_test_path}/run_benchmark_result.txt
                _post_fail=1
                [[ ${ascend_fail_not_return} == "ON" ]] || return 1
            fi
        fi
        Print_Benchmark_Result ${benchmark_test_path}/run_benchmark_result.txt
        echo "Cloud fusion inference benchmark success"
    fi

    # Run Python ST
    if [[ ${backend} =~ "cloud" && ! ${backend} =~ "ge" ]]; then
        local models_python_config=${benchmark_test_path}/models_python_ascend.cfg
        source ${benchmark_test_path}/run_benchmark_python.sh
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_python_ST ${benchmark_test_path} ${benchmark_test_path} ${ms_models} \
            ${model_data_path}'/models/hiai' "${models_python_config}" "Ascend"
        local py_ret=$?
        if [ ${py_ret} -eq 124 ]; then
            echo "Python ST (Ascend) timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            if [[ -f ${benchmark_test_path}/python/result_python_log.txt ]]; then
                echo "Python ST (Ascend) completed models:"
                cat ${benchmark_test_path}/python/result_python_log.txt
            fi
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [ ${py_ret} -ne 0 ]; then
            echo "Python ST (Ascend) failed"
            if [[ -f ${benchmark_test_path}/python/result_python_log.txt ]]; then
                echo "Python ST (Ascend) completed models:"
                cat ${benchmark_test_path}/python/result_python_log.txt
            fi
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_python_ST ${benchmark_test_path} ${benchmark_test_path} ${ms_models} \
            ${model_data_path}'/models/hiai' "${models_python_config}" "Ascend_Model_Group"
        py_ret=$?
        if [ ${py_ret} -eq 124 ]; then
            echo "Python ST (Ascend_Model_Group) timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            if [[ -f ${benchmark_test_path}/python/result_python_log.txt ]]; then
                echo "Python ST (Ascend_Model_Group) completed models:"
                cat ${benchmark_test_path}/python/result_python_log.txt
            fi
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [ ${py_ret} -ne 0 ]; then
            echo "Python ST (Ascend_Model_Group) failed"
            if [[ -f ${benchmark_test_path}/python/result_python_log.txt ]]; then
                echo "Python ST (Ascend_Model_Group) completed models:"
                cat ${benchmark_test_path}/python/result_python_log.txt
            fi
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
    fi

    # Run C++ device example
    if [[ ${backend} =~ "cloud" && ! ${backend} =~ "ge" ]]; then
        export LITE_HOME=${pkg_dir}
        export LITE_ST_MODEL=${model_data_path}/models/hiai/mindspore_uniir_mobilenetv2.mindir
        export LITE_ST_CPP_DIR=${benchmark_test_path}/cpp
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} bash ${benchmark_test_path}/run_device_mem_test.sh \
            > ${benchmark_test_path}/run_device_mem_test.log
        local cpp_ret=$?
        if [ ${cpp_ret} -eq 124 ]; then
            echo "Run device example timed out after ${WATCHDOG_TIMEOUT_SEC}s"
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [ ${cpp_ret} -ne 0 ]; then
            echo "Run device example failed"
            cat ${benchmark_test_path}/run_device_mem_test.log
            _post_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
        echo "Run device example success"
    fi

    return ${_post_fail}
}

while getopts "r:m:d:e:l:p:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${OPTARG}"
            ;;
        m)
            echo "models_path is ${OPTARG}"
            ;;
        d)
            device_ip=`echo ${OPTARG} | cut -d \: -f 1`
            device_id=`echo ${OPTARG} | cut -d \: -f 2`
            echo "device_ip is ${device_ip}, ascend_device_id is ${device_id}."
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${OPTARG}"
            ;;
        p)
            ascend_fail_not_return_cmdline=${OPTARG}
            echo "ascend_fail_not_return_cmdline is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

# ascend_fail_not_return priority: base_functions.sh global switch > -p parameter
if [[ ${ascend_fail_not_return} != "ON" ]]; then
    ascend_fail_not_return=${ascend_fail_not_return_cmdline:-OFF}
fi

if [[ ${backend} =~ "x86" ]]; then
  arch="x64"
elif [[ ${backend} =~ "arm" ]]; then
  arch="aarch64"
fi

config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

user_name=${USER}
echo "Current user name is ${user_name}"
basepath=$(pwd)/"${backend}_log_${device_id}"
rm -rf ${basepath}
mkdir -p ${basepath}
echo "Ascend base path is ${basepath}, device_ip: ${device_ip}, device_id: ${device_id}"
benchmark_test_path=/home/${user_name}/benchmark_test/${device_id}

ls /dev/davinci0
is_local=$?
if [ ${is_local} = 0 ]; then
  PrePareLocal
  if [ $? != 0 ]; then
    echo "Prepare local failed"
    exit 1
  fi
else
  PrePareRemote
  if [ $? != 0 ]; then
    echo "Prepare remote failed"
    exit 1
  fi
fi

# Parallel benchmark only supports local execution
if [ ${is_local} != 0 ]; then
    echo "ERROR: Parallel benchmark only supports local execution (requires /dev/davinci0)"
    exit 1
fi

# Write converter result to temp file
run_ascend_result_file=${basepath}'/run_'${backend}'_result.txt'
echo ' ' > ${run_ascend_result_file}

# Resolve version from release package tarball name
cd ${benchmark_test_path} || exit 1
release_file=$(ls *-linux-${arch}.tar.gz 2>/dev/null | head -1)
if [[ -z "${release_file}" ]]; then
    echo "ERROR: No *-linux-${arch}.tar.gz found in ${benchmark_test_path}"
    exit 1
fi
IFS="-" read -r -a file_name_array <<<"${release_file}"
version=${file_name_array[2]}
echo "Resolved version: ${version} from ${release_file}"
cd - > /dev/null

echo "Start to run in ${backend} with ${NUM_CARDS} cards (start_card=${START_CARD}) ..."

# Source Ascend environment (previously done inside run_converter_ascend.sh)
export ASCEND_PATH=/usr/local/Ascend
if [ -d "${ASCEND_PATH}/ascend-toolkit" ]; then
    source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
else
    source ${ASCEND_PATH}/latest/bin/setenv.bash
fi

# Step 1: Converter — parallel across cards
echo "----------------------------------------------------"
echo "Step 1: Converter (parallel, ${NUM_CARDS} cards)"
echo "----------------------------------------------------"
Run_Converter_Parallel ${NUM_CARDS} ${START_CARD}
Run_conv_status=$?
if [[ ${Run_conv_status} != 0 ]]; then
    echo "Converter failed" | tee -a ${run_ascend_result_file}
    Print_Converter_Result ${benchmark_test_path}/run_converter_result.txt
    if [[ ${ascend_fail_not_return} != "ON" ]]; then
        exit 1
    fi
    echo "Debug mode ON: continue to benchmark despite converter failures"
fi
echo "Converter success"
Print_Converter_Result ${benchmark_test_path}/run_converter_result.txt

# Step 2: Parallel benchmark across NUM_CARDS cards
echo "----------------------------------------------------"
echo "Step 2: Parallel benchmark (${NUM_CARDS} cards)"
echo "----------------------------------------------------"
Run_Parallel_Benchmark ${NUM_CARDS} ${START_CARD}
Run_bench_status=$?
if [[ -f ${benchmark_test_path}/run_benchmark_parallel_result.txt ]]; then
    Print_Benchmark_Result ${benchmark_test_path}/run_benchmark_parallel_result.txt
fi

# Step 3: Post steps (Java serial, cloud benchmark parallel, Python ST serial)
echo "----------------------------------------------------"
echo "Step 3: Post steps"
echo "----------------------------------------------------"
# Use the primary device (device_id from input) for post-steps
export ASCEND_DEVICE_ID=${device_id}
Run_Post_Steps
Run_post_status=$?

# Determine overall status
Run_ascend_status=0
if [[ ${Run_conv_status} != 0 || ${Run_bench_status} != 0 || ${Run_post_status} != 0 ]]; then
    Run_ascend_status=1
fi

if [[ ${Run_ascend_status} = 0 ]]; then
    run_result="run in ${backend} pass"; echo ${run_result} >> ${run_ascend_result_file};
else
    run_result="run in ${backend} failed"; echo ${run_result} >> ${run_ascend_result_file};
fi

# Debug mode: print failures summary
if [[ ${ascend_fail_not_return} == "ON" && ${Run_ascend_status} != 0 ]]; then
    echo ""
    echo "================================================"
    echo "  DEBUG MODE FAILURES SUMMARY"
    echo "================================================"
    for _f in "${benchmark_test_path}/run_converter_result.txt" \
              "${benchmark_test_path}/run_benchmark_parallel_result.txt" \
              "${benchmark_test_path}/run_benchmark_result.txt"; do
        if [[ -s "$_f" ]]; then
            echo "--- $(basename $_f) ---"
            grep "failed\|TIMEOUT" "$_f" || true
        fi
    done
    if [[ -f ${benchmark_test_path}/python/result_python_log.txt ]]; then
        echo "--- Python ST result_python_log.txt ---"
        grep "failed" ${benchmark_test_path}/python/result_python_log.txt
    fi
    echo "================================================"
fi

# Copy result files back to basepath
# Merge parallel benchmark results + cloud/post-step results into final files
run_converter_log_file=${basepath}'/run_'${backend}'_converter_log.txt'
run_converter_result_file=${basepath}'/run_'${backend}'_converter_result.txt'
run_benchmark_log_file=${basepath}'/run_'${backend}'_benchmark_log.txt'
run_benchmark_result_file=${basepath}'/run_'${backend}'_benchmark_result.txt'
if [ ${is_local} = 0 ]; then
  cp ${benchmark_test_path}/run_converter_log.txt ${run_converter_log_file} || exit 1
  cp ${benchmark_test_path}/run_converter_result.txt ${run_converter_result_file} || exit 1
  # Merge parallel + cloud benchmark logs/results
  true > "${run_benchmark_log_file}"
  true > "${run_benchmark_result_file}"
  if [[ -f ${benchmark_test_path}/run_benchmark_parallel_log.txt ]]; then
      cat ${benchmark_test_path}/run_benchmark_parallel_log.txt >> ${run_benchmark_log_file}
  fi
  if [[ -f ${benchmark_test_path}/run_benchmark_log.txt ]]; then
      cat ${benchmark_test_path}/run_benchmark_log.txt >> ${run_benchmark_log_file}
  fi
  if [[ -f ${benchmark_test_path}/run_benchmark_parallel_result.txt ]]; then
      cat ${benchmark_test_path}/run_benchmark_parallel_result.txt >> ${run_benchmark_result_file}
  fi
  if [[ -f ${benchmark_test_path}/run_benchmark_result.txt ]]; then
      cat ${benchmark_test_path}/run_benchmark_result.txt >> ${run_benchmark_result_file}
  fi
else
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_converter_log.txt ${run_converter_log_file} || exit 1
  scp ${user_name}@${device_ip}:${benchmark_test_path}/run_converter_result.txt ${run_converter_result_file} || exit 1
  # Remote: merge parallel + cloud results
  true > "${run_benchmark_log_file}"
  true > "${run_benchmark_result_file}"
  if ssh ${user_name}@${device_ip} "test -f ${benchmark_test_path}/run_benchmark_parallel_log.txt"; then
      scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_parallel_log.txt /tmp/parallel_log.txt || exit 1
      cat /tmp/parallel_log.txt >> ${run_benchmark_log_file}
  fi
  if ssh ${user_name}@${device_ip} "test -f ${benchmark_test_path}/run_benchmark_log.txt"; then
      scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_log.txt /tmp/cloud_log.txt || exit 1
      cat /tmp/cloud_log.txt >> ${run_benchmark_log_file}
  fi
  if ssh ${user_name}@${device_ip} "test -f ${benchmark_test_path}/run_benchmark_parallel_result.txt"; then
      scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_parallel_result.txt /tmp/parallel_result.txt || exit 1
      cat /tmp/parallel_result.txt >> ${run_benchmark_result_file}
  fi
  if ssh ${user_name}@${device_ip} "test -f ${benchmark_test_path}/run_benchmark_result.txt"; then
      scp ${user_name}@${device_ip}:${benchmark_test_path}/run_benchmark_result.txt /tmp/cloud_result.txt || exit 1
      cat /tmp/cloud_result.txt >> ${run_benchmark_result_file}
  fi
fi

echo "Run in ${backend} ended"
cat ${run_ascend_result_file}
exit ${Run_ascend_status}
