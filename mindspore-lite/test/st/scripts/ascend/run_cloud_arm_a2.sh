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

# Parallel execution constants (single source of truth)
NUM_CARDS=${NUM_PARALLEL_CARDS:-8}
START_CARD=${START_CARD_ID:-0}

# Run benchmark with specified cfg list and log/result files
# $1: cfg_file_list; $2: log_file; $3: result_file
function Run_Benchmark_With_Cfg() {
    local cfg_file_list=$1
    local bench_log_file=$2
    local bench_result_file=$3
    local _bench_fail=0
    echo "Start running benchmark models"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ || return 1
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/lib:./tools/converter/lib/
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/glog
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./runtime/third_party/dnnl
    # Copy benchmark binary if not already present (may be pre-copied in parallel mode)
    if [[ ! -f ./benchmark ]]; then
        cp tools/benchmark/benchmark ./ || return 1
    fi

    echo "benchmark model cfg list:"
    echo ${cfg_file_list}

    local line_info model_info spec_acc_limit model_name input_num input_shapes \
        mode model_file input_files output_file data_path acc_limit enableFp16 \
        run_result
    for cfg_file in "${cfg_file_list[@]}"; do
        while read line; do
            line_info=${line}
            if [[ $line_info == \#* || $line_info == "" ]]; then
                continue
            fi

            # model_info     accuracy_limit      run_mode
            model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
            accuracy_info=$(echo ${line_info} | awk -F ' ' '{print $2}')
            spec_acc_limit=$(echo ${accuracy_info} | awk -F ';' '{print $1}')

            # model_info detail
            model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
            input_info=$(echo ${model_info} | awk -F ';' '{print $2}')
            input_shapes=$(echo ${model_info} | awk -F ';' '{print $3}')
            mode=$(echo ${model_info} | awk -F ';' '{print $5}')
            input_num=$(echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $1}')
            input_names=`echo ${input_info} | sed 's/:/;/' | awk -F ';' '{print $2}'`
            spec_shapes=""
            if [[ ${input_shapes} != "" && ${input_names} != "" ]]; then
                if [[ ${input_num} == "" ]]; then
                    input_num=1
                fi
                IFS="," read -r -a name_array <<< ${input_names}
                IFS=":" read -r -a shape_array <<< ${input_shapes}
                for i in $(seq 0 $((${input_num}-1)))
                do
                    spec_shapes=${spec_shapes}${name_array[$i]}':'${shape_array[$i]}';'
                done
            fi
            if [[ ${model_name##*.} == "caffemodel" ]]; then
                model_name=${model_name%.*}
            fi

            # converter for distribution models
            use_parallel_predict="false"
            if [[ ${mode} =~ "parallel_predict" ]]; then
                use_parallel_predict="true"
            fi
            echo "Benchmarking ${model_name} on device ${ASCEND_DEVICE_ID} ......"
            model_file=${ms_models_path}'/'${model_name}'.mindir'
            if [[ ${mode} == "large_model" ]]; then
              model_file=${ms_models_path}'/'${model_name}'_graph.mindir'
            fi
            input_files=""
            output_file=""
            data_path=${models_path}'/input_output/'
            if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                input_files=${data_path}'input/'${model_name}'.bin'
            else
                for i in $(seq 1 $input_num); do
                    input_files=${input_files}${data_path}'input/'${model_name}'.bin_'$i','
                done
            fi
            output_file=${data_path}'output/'${model_name}'.out'

            # set accuracy limitation
            acc_limit="0.5"
            if [[ ${spec_acc_limit} != "" ]]; then
                acc_limit="${spec_acc_limit}"
            elif [[ ${mode} == "fp16" ]]; then
                acc_limit="5"
            fi
            # whether enable fp16
            enableFp16="false"
            if [[ ${mode} == "fp16" ]]; then
                enableFp16="true"
            fi
            echo "cfg_file: ${cfg_file}"
            benchmark_config_file=""
            if [[ ${cfg_file} =~ "ge_with_config" ]]; then
              input_files=""
              benchmark_config_file="${benchmark_test_path}/${model_name}.mindir.ge.config"
              if [[ ${input_num} == "" || ${input_num} == 1 ]]; then
                  input_files=${data_path}'input/'${model_name}'.mindir.bin'
              else
                  for i in $(seq 1 $input_num); do
                      input_files=${input_files}${data_path}'input/'${model_name}'.mindir.bin_'$i','
                  done
              fi
              output_file=${data_path}'output/'${model_name}'.mindir.out'
            fi
            echo './benchmark --enableParallelPredict='${use_parallel_predict}' --modelFile='${model_file}' --inputShape='${spec_shapes}' --inDataFile='${input_files}' --benchmarkDataFile='${output_file}' --enableFp16='${enableFp16}' --accuracyThreshold='${acc_limit}' --device='${benchmark_device}' --configFile='${benchmark_config_file}
            ./benchmark --enableParallelPredict=${use_parallel_predict} --modelFile=${model_file} --inputShape="${spec_shapes}" --inDataFile=${input_files} --benchmarkDataFile=${output_file} --enableFp16=${enableFp16} --accuracyThreshold=${acc_limit} --device=${benchmark_device} --configFile=${benchmark_config_file} >> "${bench_log_file}"
            if [ $? = 0 ]; then
                if [[ ${mode} =~ "parallel_predict" ]]; then
                    run_result="${benchmark_device}: ${model_name} parallel_pass"
                    echo ${run_result} >>${bench_result_file}
                else
                    run_result="${benchmark_device}: ${model_name} pass"
                    echo ${run_result} >>${bench_result_file}
                fi
            else
                if [[ ${mode} =~ "parallel_predict" ]]; then
                    run_result="${benchmark_device}: ${model_name} parallel_failed"
                    echo ${run_result} >>${bench_result_file}
                    _bench_fail=1
                    [[ ${ascend_fail_not_return} == "ON" ]] || return 1
                else
                    run_result="${benchmark_device}: ${model_name} failed"
                    echo ${run_result} >>${bench_result_file}
                    _bench_fail=1
                    [[ ${ascend_fail_not_return} == "ON" ]] || return 1
                fi
            fi

        done <${cfg_file}
    done
    return ${_bench_fail}
}

# Run GE benchmark with specified cfg list and log/result files
# $1: ge_cfg_file_list; $2: log_file; $3: result_file; $4: skip_model_cp (optional "true")
function Run_Benchmark_GE_With_Cfg() {
    local ge_cfg_list=$1
    local ge_log_file=$2
    local ge_result_file=$3
    local skip_model_cp=${4:-"false"}
    echo "Start running benchmark ge backend on device ${ASCEND_DEVICE_ID}"
    export ASCEND_BACK_POLICY="ge"
    cd ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ || return 1
    mkdir -p ${ms_models_path}
    # Copy GE models unless pre-copied by caller (parallel mode)
    if [[ ${skip_model_cp} != "true" ]]; then
        echo ${ge_cfg_list}
        for cfg_file in "${ge_cfg_list[@]}"; do
            while read line; do
                line_info=${line}
                if [[ ${line_info} == \#* || ${line_info} == "" ]]; then
                  echo ${line_info}
                  continue
                fi
                model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
                model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
                echo "${models_path}/${model_name}.mindir"
                cp "${models_path}/${model_name}.mindir" $ms_models_path/ || return 1
            done <${cfg_file}
        done
    fi
    # Empty config file is allowed, but warning message will be shown
    if [[ $(Exist_File_In_Path ${ms_models_path} ".mindir") != "true" ]]; then
        echo "No ms model found in ${ms_models_path}, ge config may be empty"
        return 0
    fi
    # add config file path for ge
    Run_Benchmark_With_Cfg "${ge_cfg_list}" "${ge_log_file}" "${ge_result_file}"
    return $?
}

# Run full pipeline (converter + benchmark ACL + benchmark GE) on a single card
# $1: card_idx; $2: card_id (physical device id)
# Requires: pkg_dir, ms_models_path to be set by caller
function Run_Single_Card() {
    local card_idx=$1
    local card_id=$2
    local _card_fail=0
    export ASCEND_DEVICE_ID=${card_id}
    local sub_acl="${benchmark_test_path}/models_with_large_model_acl_with_config_cloud_ascend_sub_card${card_idx}.cfg"
    local sub_ge="${benchmark_test_path}/models_with_large_model_ge_with_config_cloud_ascend_sub_card${card_idx}.cfg"
    local card_conv_log="${benchmark_test_path}/run_converter_log_card${card_idx}.txt"
    local card_conv_result="${benchmark_test_path}/run_converter_result_card${card_idx}.txt"
    local card_bench_log="${benchmark_test_path}/run_benchmark_log_card${card_idx}.txt"
    local card_bench_acl_result="${benchmark_test_path}/run_benchmark_acl_result_card${card_idx}.txt"
    local card_bench_ge_result="${benchmark_test_path}/run_benchmark_ge_result_card${card_idx}.txt"

    echo "run ${benchmark_device} benchmark logs on card ${card_id}: " > ${card_bench_log}
    true > ${card_conv_log}
    true > ${card_conv_result}
    true > ${card_bench_acl_result}
    true > ${card_bench_ge_result}

    # 1. Converter for models assigned to this card
    #    Use per-card working directory to isolate fifo_file/fail_file created by Convert()
    if [[ -s ${sub_acl} ]]; then
        echo "Card ${card_id}: Start converter" >> "${card_bench_log}"
        local card_work_dir="${benchmark_test_path}/card_work_${card_idx}"
        mkdir -p ${card_work_dir}
        cd ${card_work_dir} || return 1
        ln -sf ${pkg_dir}/converter_lite ./
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${pkg_dir}/tools/converter/lib/:${pkg_dir}/tools/converter/third_party/glog/lib
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Convert "${sub_acl}" $models_path $ms_models_path $card_conv_log $card_conv_result $ascend_fail_not_return
        local conv_status=$?
        if [[ ${conv_status} -eq 124 ]]; then
            echo "Card ${card_id}: Converter timed out" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [[ ${conv_status} != 0 ]]; then
            echo "Card ${card_id}: Converter failed" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
        echo "Card ${card_id}: Converter success" >> "${card_bench_log}"
    fi

    # 2. Benchmark ACL
    if [[ -s ${sub_acl} ]]; then
        echo "Card ${card_id}: Start benchmark ACL" >> "${card_bench_log}"
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_With_Cfg "${sub_acl}" "${card_bench_log}" "${card_bench_acl_result}"
        local acl_status=$?
        if [[ ${acl_status} -eq 124 ]]; then
            echo "Card ${card_id}: Benchmark ACL timed out" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [[ ${acl_status} != 0 ]]; then
            echo "Card ${card_id}: Benchmark ACL failed" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
        echo "Card ${card_id}: Benchmark ACL success" >> "${card_bench_log}"
    fi

    # 3. Benchmark GE
    if [[ -s ${sub_ge} ]]; then
        echo "Card ${card_id}: Start benchmark GE" >> "${card_bench_log}"
        Run_With_Watchdog ${WATCHDOG_TIMEOUT_SEC} Run_Benchmark_GE_With_Cfg "${sub_ge}" "${card_bench_log}" "${card_bench_ge_result}" "true"
        local ge_status=$?
        unset ASCEND_BACK_POLICY
        if [[ ${ge_status} -eq 124 ]]; then
            echo "Card ${card_id}: Benchmark GE timed out" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        elif [[ ${ge_status} != 0 ]]; then
            echo "Card ${card_id}: Benchmark GE failed" >> "${card_bench_log}"
            _card_fail=1
            [[ ${ascend_fail_not_return} == "ON" ]] || return 1
        fi
        echo "Card ${card_id}: Benchmark GE success" >> "${card_bench_log}"
    fi

    echo "Card ${card_id}: All done" >> "${card_bench_log}"
    return ${_card_fail}
}

# Run parallel converter + benchmark across multiple cards
# Globals: NUM_CARDS (default 8), START_CARD (default 0)
function Run_Parallel_All() {
    local num_cards=${NUM_CARDS}
    local start_card=${START_CARD}
    local pids=()
    pkg_dir="${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}"

    echo "Run_Parallel_All: splitting configs into ${num_cards} parts"
    # Split ACL cfg - keep "acl_with_config" in filename for Run_Benchmark_With_Cfg pattern matching
    Split_Config "${models_server_inference_cfg_file_list}" $num_cards "${benchmark_test_path}/models_with_large_model_acl_with_config_cloud_ascend_sub" || return 1
    # Split GE cfg - keep "ge_with_config" in filename for Run_Benchmark_With_Cfg pattern matching
    Split_Config "${models_ge_cfg_file_list}" $num_cards "${benchmark_test_path}/models_with_large_model_ge_with_config_cloud_ascend_sub" || return 1

    # Pre-copy binaries before fork to avoid concurrent cp conflict
    cd ${pkg_dir} || return 1
    cp tools/converter/converter/converter_lite ./ || return 1
    cp tools/benchmark/benchmark ./ || return 1

    # Pre-copy GE models to ms_models_path before fork to avoid concurrent cp conflict
    mkdir -p ${ms_models_path}
    for cfg_file in "${models_ge_cfg_file_list[@]}"; do
        while read line; do
            line_info=${line}
            if [[ ${line_info} == \#* || ${line_info} == "" ]]; then
                continue
            fi
            model_info=$(echo ${line_info} | awk -F ' ' '{print $1}')
            model_name=$(echo ${model_info} | awk -F ';' '{print $1}')
            if [[ -f "${models_path}/${model_name}.mindir" ]]; then
                cp "${models_path}/${model_name}.mindir" $ms_models_path/ || return 1
            else
                echo "WARNING: GE model ${models_path}/${model_name}.mindir not found, skipped"
            fi
        done <${cfg_file}
    done

    for ((i = 0; i < num_cards; i++)); do
        (
            Run_Single_Card $i $((start_card + i))
            exit $?
        ) &
        pids+=($!)
    done

    local fail=0
    local timeout_card_indices=()
    for ((i = 0; i < num_cards; i++)); do
        pid="${pids[$i]}"
        wait $pid
        ec=$?
        if [ ${ec} -eq 124 ]; then
            timeout_card_indices+=("${i}")
            fail=1
        elif [ ${ec} -ne 0 ]; then
            echo "Card $((start_card + i)) failed (exit ${ec})"
            fail=1
        fi
    done

    # Merge results with device identification into separate stage files
    local stage_conv_result="${benchmark_test_path}/stage_converter_result.txt"
    local stage_bench_acl_result="${benchmark_test_path}/stage_benchmark_acl_result.txt"
    local stage_bench_ge_result="${benchmark_test_path}/stage_benchmark_ge_result.txt"
    true > "${stage_conv_result}"
    true > "${stage_bench_acl_result}"
    true > "${stage_bench_ge_result}"
    for ((i = 0; i < num_cards; i++)); do
        local dev_id=$((start_card + i))
        Append_Result_With_Device "${benchmark_test_path}/run_converter_result_card${i}.txt" "${dev_id}" "converter" "${stage_conv_result}"
        Append_Result_With_Device "${benchmark_test_path}/run_benchmark_acl_result_card${i}.txt" "${dev_id}" "benchmark" "${stage_bench_acl_result}"
        Append_Result_With_Device "${benchmark_test_path}/run_benchmark_ge_result_card${i}.txt" "${dev_id}" "benchmark_ge" "${stage_bench_ge_result}"
    done
    # Append TIMEOUT rows for timed-out cards to stages that have no results
    for tidx in "${timeout_card_indices[@]}"; do
        local tdev=$((start_card + tidx))
        if [[ ! -s ${benchmark_test_path}/run_converter_result_card${tidx}.txt ]]; then
            printf "$RESULT_FMT" "Ascend:${tdev}" "converter" "(timed_out)" "TIMEOUT" >> "${stage_conv_result}"
        fi
        if [[ ! -s ${benchmark_test_path}/run_benchmark_acl_result_card${tidx}.txt ]]; then
            printf "$RESULT_FMT" "Ascend:${tdev}" "benchmark" "(timed_out)" "TIMEOUT" >> "${stage_bench_acl_result}"
        fi
        if [[ ! -s ${benchmark_test_path}/run_benchmark_ge_result_card${tidx}.txt ]]; then
            printf "$RESULT_FMT" "Ascend:${tdev}" "benchmark_ge" "(timed_out)" "TIMEOUT" >> "${stage_bench_ge_result}"
        fi
    done
    # Merge detailed logs
    cat ${benchmark_test_path}/run_benchmark_log_card*.txt > ${run_benchmark_log_file}
    cat ${benchmark_test_path}/run_converter_log_card*.txt > ${run_converter_log_file}
    # Append timeout notices to merged log so they are persisted
    for tidx in "${timeout_card_indices[@]}"; do
        echo "[conv/bench] Ascend:$((start_card + tidx)) timed out after ${WATCHDOG_TIMEOUT_SEC}s" >> "${run_benchmark_log_file}"
    done

    # Clean up per-card work dirs
    rm -rf ${benchmark_test_path}/card_work_*

    return $fail
}

# Run python benchmark in parallel across multiple cards
function Run_Python_Benchmark_Parallel() {
    local num_cards=${NUM_CARDS}
    local start_card=${START_CARD}
    local pids=()
    local card_ids=()
    local result_files=()

    echo "Run_Python_Benchmark_Parallel: splitting python config into ${num_cards} parts"
    Split_Config "${benchmark_test_path}/models_cloud_ascend_a2.cfg" $num_cards "${benchmark_test_path}/python_sub" || return 1

    for ((i = 0; i < num_cards; i++)); do
        local sub_python_cfg="${benchmark_test_path}/python_sub_card${i}.cfg"
        if [[ ! -s ${sub_python_cfg} ]]; then
            continue
        fi
        local card_id=$((start_card + i))
        local result_file="${benchmark_test_path}/pybench_card${i}.log"
        result_files+=("${result_file}")
        card_ids+=("${card_id}")
        (
            export ASCEND_DEVICE_ID=${card_id}
            timeout ${WATCHDOG_TIMEOUT_SEC} python3 run_python_benchmark.py ${models_path}/ ${ms_models_path} ${basepath}/../${config_folder}/ascend/ ${sub_python_cfg} ${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ ${card_id}
        ) > "${result_file}" 2>&1 &
        pids+=($!)
    done

    # Wait for all cards and collect exit codes
    local exit_codes=()
    for ((j=0; j<${#pids[@]}; j++)); do
        wait "${pids[$j]}"
        local ec=$?
        exit_codes+=("${ec}")
        echo "Python benchmark card ${card_ids[$j]} finished (exit ${ec})"
    done

    # Aggregate results and print table
    local pybench_result_file=${benchmark_test_path}/stage_python_benchmark_result.txt
    local pybench_log_file=${benchmark_test_path}/run_python_benchmark_log.txt
    true > "${pybench_result_file}"
    true > "${pybench_log_file}"
    local fail=0
    for ((i=0; i<${#result_files[@]}; i++)); do
        local rf="${result_files[$i]}"
        local cid="${card_ids[$i]}"
        local status="${exit_codes[$i]}"
        if [ "${status}" -eq 124 ]; then
            printf "$PY_BENCH_FMT" "Ascend:${cid}" "(timeout)" "-" "-" "-" "timeout" >> "${pybench_result_file}"
            echo "--- Python Benchmark Ascend:${cid} (timeout) ---" >> "${pybench_log_file}"
            cat "${rf}" >> "${pybench_log_file}"
            fail=1
        elif [ "${status}" -ne 0 ]; then
            # Parse per-model results (both succeeded and the failed one) from output
            local _grep_tmp
            _grep_tmp=$(mktemp)
            grep -E '\S+\s+\S+\s+\S+\s+\S+\s+(pass|failed)\s*$' "$rf" > "$_grep_tmp" 2>/dev/null
            while IFS= read -r line; do
                read -r model_name build_time predict_time accuracy result <<< "${line}"
                [[ "${build_time}" =~ ^[0-9] ]] && build_time=$(printf "%.2f" "${build_time}")
                [[ "${predict_time}" =~ ^[0-9] ]] && predict_time=$(printf "%.2f" "${predict_time}")
                local acc_num="${accuracy%\%}"; [[ "${acc_num}" =~ ^[0-9] ]] && accuracy=$(printf "%.4f" "${acc_num}")%
                printf "$PY_BENCH_FMT" "Ascend:${cid}" "${model_name}" "${build_time}" "${predict_time}" "${accuracy}" "${result}" >> "${pybench_result_file}"
            done < "$_grep_tmp"
            rm -f "$_grep_tmp"
            echo "--- Python Benchmark Ascend:${cid} (failed) ---" >> "${pybench_log_file}"
            cat "${rf}" >> "${pybench_log_file}"
            fail=1
        else
            # Parse per-model results from run_python_benchmark.py PrintResult output
            # Format: model_name  build_time  predict_time  accuracy  pass/failed  (fields are {:<30} padded, so allow trailing spaces)
            local card_had_results=0
            local _grep_tmp
            _grep_tmp=$(mktemp)
            grep -E '\S+\s+\S+\s+\S+\s+\S+\s+(pass|failed)\s*$' "$rf" > "$_grep_tmp" 2>/dev/null
            while IFS= read -r line; do
                read -r model_name build_time predict_time accuracy result <<< "${line}"
                [[ "${build_time}" =~ ^[0-9] ]] && build_time=$(printf "%.2f" "${build_time}")
                [[ "${predict_time}" =~ ^[0-9] ]] && predict_time=$(printf "%.2f" "${predict_time}")
                local acc_num="${accuracy%\%}"; [[ "${acc_num}" =~ ^[0-9] ]] && accuracy=$(printf "%.4f" "${acc_num}")%
                printf "$PY_BENCH_FMT" "Ascend:${cid}" "${model_name}" "${build_time}" "${predict_time}" "${accuracy}" "${result}" >> "${pybench_result_file}"
                card_had_results=1
            done < "$_grep_tmp"
            rm -f "$_grep_tmp"
            # Fallback: if no per-model lines parsed for this card, write a card-level summary
            if [[ ${card_had_results} -eq 0 ]]; then
                printf "$PY_BENCH_FMT" "Ascend:${cid}" "(no per-model results parsed)" "-" "-" "-" "pass" >> "${pybench_result_file}"
            fi
        fi
    done

    # Only print detailed logs when there are failures
    if [[ ${fail} -eq 1 && -s "${pybench_log_file}" ]]; then
        cat "${pybench_log_file}"
    fi
    Print_Python_Benchmark_Result "${pybench_result_file}" "PYTHON BENCHMARK"
    rm -f "${pybench_log_file}"
    rm -f "${result_files[@]}"
    return $fail
}

function ConfigAscend() {
    echo "Start to copy Ascend local file"
    benchmark_device=Ascend
    user_name=${USER}
    echo "Current user name is ${user_name}"
    benchmark_test_path=/home/${user_name}/benchmark_test/${device_id}
    echo "Ascend base path is ${benchmark_test_path}, device_id: ${device_id}"
    rm -rf ${benchmark_test_path}
    mkdir -p ${benchmark_test_path}
    models_path=/home/workspace/mindspore_dataset/mslite/models/hiai
    # mkdir -p ${benchmark_test_path}/large_models
    cp ${basepath}/../${config_folder}/ascend/*.config ${benchmark_test_path} || exit 1
    cp ${basepath}/../${config_folder}/models_with_large_model_python_with_config_cloud_ascend.cfg ${benchmark_test_path} || exit 1
    cp ${basepath}/../${config_folder}/models_with_large_model_acl_with_config_cloud_ascend.cfg ${benchmark_test_path} || exit 1
    cp ${basepath}/../${config_folder}/models_with_large_model_ge_with_config_cloud_ascend.cfg ${benchmark_test_path} || exit 1
    cp ${basepath}/../${config_folder}/models_cloud_ascend_a2.cfg ${benchmark_test_path} || exit 1
    # we do not convert ge models, because we will use benchmark to run mindir with ge backend
    models_server_inference_cfg_file_list=${benchmark_test_path}/models_with_large_model_acl_with_config_cloud_ascend.cfg
    models_ge_cfg_file_list=${benchmark_test_path}/models_with_large_model_ge_with_config_cloud_ascend.cfg
    if [[ ${arch} = "aarch64" ]]; then
        release_package_path=${release_path}/linux_aarch64/cloud_fusion/ || exit 1
    else
        release_package_path=${release_path}/centos_x86/cloud_fusion/ || exit 1
    fi
    echo "Copy file success"
    # source ascend env
    export ASCEND_PATH=/usr/local/Ascend
    if [ -d "${ASCEND_PATH}/ascend-toolkit" ]; then
        source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
    else
        source ${ASCEND_PATH}/latest/bin/setenv.bash
    fi
}

# Example:sh run_benchmark_graph_kernel.sh -r /home/temp_test -m /home/temp_test/models -e x86_gpu -d 192.168.1.1:0
# backend can be: x86_gpu,x86_cpu,arm64_cpu,arm64_android_cpu,x86_ascend,arm64_ascend
while getopts "r:m:d:e:l:" opt; do
    case ${opt} in
    r)
        release_path=${OPTARG}
        echo "release_path is ${OPTARG}"
        ;;
    m)
        models_path=${OPTARG}
        echo "models_path is ${OPTARG}"
        ;;
    d)
        device_ip=$(echo ${OPTARG} | cut -d \: -f 1)
        device_id=$(echo ${OPTARG} | cut -d \: -f 2)
        echo "device_ip is ${device_ip}, device_id is ${device_id}."
        ;;
    e)
        backend=${OPTARG}
        echo "backend is ${backend}"
        ;;
    l)
        level=${OPTARG}
        echo "level is ${OPTARG}"
        ;;
    ?)
        echo "unknown para"
        exit 1
        ;;
    esac
done

basepath=$(pwd)

# default working dir is benchmark_test_path
benchmark_test_path=${basepath}/benchmark_test

# clear working dir
rm -rf ${benchmark_test_path}
mkdir -p ${benchmark_test_path}

if [[ $backend =~ "arm" ]]; then
    arch="aarch64"
else
    arch="x64"
fi

# Set models config filepath
config_folder="config_level0"
if [[ ${level} == "level1" ]]; then
    config_folder="config_level1"
fi

# config file
ConfigAscend

# get release package path and version
ms_models_path=${benchmark_test_path}/ms_models
cd $release_package_path || exit 1
release_file=$(ls *-linux-*.tar.gz)
release_file_path="$release_package_path/$release_file"
IFS="-" read -r -a file_name_array <<<"$release_file"
version=${file_name_array[2]}

echo "installing mslite whl..."
python3 -m pip uninstall -y mindspore_lite || exit 1
python3 -m pip install *.whl
echo "install mslite success !"

echo "Running MSLite Large Model on ${backend}, release file path is $release_file_path, working dir is: $benchmark_test_path"
cd -
# uncompressing package file
echo "uncompressing package file..."
cd ${benchmark_test_path} || exit 1
tar -zxf $release_file_path  || exit 1

# Write converter result to temp file
run_converter_log_file=${benchmark_test_path}/run_converter_log.txt
true >${run_converter_log_file}

# Run converter + benchmark in parallel across multiple cards
echo "Running in parallel with ${NUM_CARDS} cards, start_card_id=${START_CARD}"

echo "Start parallel converter + benchmark ..."
run_benchmark_log_file=${benchmark_test_path}/run_benchmark_log.txt
echo "run ${benchmark_device} benchmark logs: " > ${run_benchmark_log_file}

# Track overall status for debug mode
overall_status=0

Run_Parallel_All
Run_parallel_status=$?
# Print detailed logs (always)
cat ${run_converter_log_file}
cat ${run_benchmark_log_file}
# Print per-stage result tables (always, regardless of success/failure)
Print_Stage_Result "${benchmark_test_path}/stage_converter_result.txt" "CONVERTER"
Print_Stage_Result "${benchmark_test_path}/stage_benchmark_acl_result.txt" "BENCHMARK ACL"
Print_Stage_Result "${benchmark_test_path}/stage_benchmark_ge_result.txt" "BENCHMARK GE"
if [[ ${Run_parallel_status} != 0 ]]; then
    echo "Run_Parallel_All failed"
    if [[ ${ascend_fail_not_return} != "ON" ]]; then
        exit ${Run_parallel_status}
    fi
    overall_status=1
    echo "Debug mode ON: continue to Python benchmark despite C++ failures"
fi
echo "Run_Parallel_All success"

# Empty config file is allowed, but warning message will be shown
if [[ $(Exist_File_In_Path ${ms_models_path} ".mindir") != "true" ]]; then
    echo "No mslite mindir model found in ${ms_models_path}, please check if config file is empty!"
    exit 0
fi

#---------------------------------------------------------
# run converter and predict by python api in parallel
echo "basepath: ${basepath}"
cd ${basepath}/python/benchmark/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/tools/converter/lib/:${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/tools/converter/third_party/glog/lib:${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/runtime/lib
Run_Python_Benchmark_Parallel
run_python_status=$?
if [[ ${run_python_status} != 0 ]]; then
    echo "run python benchmark failed"
    if [[ ${ascend_fail_not_return} != "ON" ]]; then
        exit ${run_python_status}
    fi
    overall_status=1
    echo "Debug mode ON: continue to pytest despite Python benchmark failures"
fi

# run python api test
echo "---------- Run MindSpore Lite API ----------"
cd ${basepath}/python/python_api/  || exit 1
cp -r ${ms_models_path}/sd1.5_unet.onnx* . || exit 1 # for Model Predict ST
cp -r ${ms_models_path}/single_matmul_model.onnx.mindir . || exit 1 # for Update weights ST
cp -r ${ms_models_path}/deepaudio.onnx* . || exit 1 # for ModelParallelRunner 'for-loop' Predict ST
cp -r ${ms_models_path}/resize.onnx.mindir . || exit 1 # for Model ST
cp -r ${basepath}/../${config_folder}/ascend/prof.json . || exit 1 # for test profiling
cp -r ${models_path}/single_matmul_model.onnx . || exit 1 # for dump graph ir ST
cp -r ${models_path}/ge_test_mul.mindir . || exit 1 # for GE ST
cp -r ${ms_models_path}/matmul_bf16.onnx.mindir . || exit 1 # for bf16 inference ST
#for code coverage in A2
MSLITE_COVERAGE_ARGS=""
if [[ "${MSLITE_ENABLE_COVERAGE}" == "on" || "${MSLITE_ENABLE_COVERAGE}" == "ON" ]]; then
    echo "MSLITE_ENABLE_COVERAGE: ${MSLITE_ENABLE_COVERAGE}, MSLITE_COVERAGE_FILE: ${MSLITE_COVERAGE_FILE}"
    MSLITE_COVERAGE_ARGS="-m coverage run --rcfile=${MSLITE_COVERAGE_FILE}"
fi
# File-level parallel distribution across NPU cards with xdist + --forked per-test isolation.
# Test files are split round-robin across cards to avoid concurrent converter
# output conflicts (tests in the same file share converter output directory).
# xdist -n 1 --dist=loadfile provides clean subprocess isolation per file group,
# within which pytest-forked provides per-test isolation.
PYTEST_ARGS="--mindir_dir=${models_path}/ --so_path=${benchmark_test_path}/mindspore-lite-${version}-linux-${arch}/ --config_dir=${basepath}/../${config_folder}/ascend/"
# Collect all test files
test_files_list=$(ls test_*.py *_test.py 2>/dev/null)
total_files=$(echo "${test_files_list}" | wc -w)
if [ "${total_files}" -eq 0 ]; then
    echo "ERROR: No test files found"
    exit 1
fi
echo "Distributing ${total_files} test files across ${NUM_CARDS} cards (xdist + --forked per-test isolation)"
# Launch one pytest process per card, each with xdist -n 1 + --forked
pids=()
result_files=()
card_ids_for_result=()
file_index=0
for test_file in ${test_files_list}; do
    card=$((file_index % NUM_CARDS))
    card_id=$((START_CARD + card))
    # Append file to card's file list
    eval "card_${card}_files=\"\${card_${card}_files} ${test_file}\""
    file_index=$((file_index + 1))
done
# Copy pre-existing models to per-card output directories to avoid concurrent
# converter_lite writes (from module-scoped pytest fixtures) racing on the same files.
for ((card=0; card<NUM_CARDS; card++)); do
    card_id=$((START_CARD + card))
    card_output_dir=${ms_models_path}/${card_id}
    mkdir -p ${card_output_dir}
    cp -r -l ${ms_models_path}/* ${card_output_dir}/ 2>/dev/null || true
done

for ((card=0; card<NUM_CARDS; card++)); do
    eval "files=\${card_${card}_files}"
    if [ -z "${files}" ]; then
        continue
    fi
    card_id=$((START_CARD + card))
    result_file=$(mktemp)
    result_files+=("${result_file}")
    card_ids_for_result+=("${card_id}")
    (
        timeout ${WATCHDOG_TIMEOUT_SEC} python3 ${MSLITE_COVERAGE_ARGS} -m pytest ${files} -n 1 --dist=loadfile --forked \
            --device_id ${card_id} --output_dir=${ms_models_path}/${card_id}/ ${PYTEST_ARGS} -q -rA
    ) > "${result_file}" 2>&1 &
    pids+=($!)
done
# Wait for all cards, tracking exit codes and timeouts
failed=0
pytest_timeout_cards=()
for ((i=0; i<${#pids[@]}; i++)); do
    pid="${pids[$i]}"
    cid="${card_ids_for_result[$i]}"
    wait "${pid}"
    ec=$?
    if [ ${ec} -eq 124 ]; then
        pytest_timeout_cards+=("${cid}")
        failed=1
    elif [ ${ec} -ne 0 ]; then
        failed=1
    fi
done
# Collect raw pytest output per card into merged log
pytest_log_file=${benchmark_test_path}/run_pytest_log.txt
true > "${pytest_log_file}"
echo "" >> "${pytest_log_file}"
for ((i=0; i<${#result_files[@]}; i++)); do
    rf="${result_files[$i]}"
    cid="${card_ids_for_result[$i]}"
    echo "--- Ascend:${cid} pytest output ---" >> "${pytest_log_file}"
    cat "${rf}" >> "${pytest_log_file}"
done
for cid in "${pytest_timeout_cards[@]}"; do
    echo "[pytest] Ascend:${cid} timed out after ${WATCHDOG_TIMEOUT_SEC}s, partial results above" >> "${pytest_log_file}"
done
# Aggregate pytest results into summary file for table printing
pytest_result_file=${benchmark_test_path}/stage_pytest_result.txt
true > "${pytest_result_file}"
total_passed=0
total_failed=0
for ((i=0; i<${#result_files[@]}; i++)); do
    rf="${result_files[$i]}"
    cid="${card_ids_for_result[$i]}"
    # Parse individual test results from short test summary
    _grep_tmp=$(mktemp)
    grep -E "^(PASSED|FAILED) " "$rf" > "$_grep_tmp" 2>/dev/null
    while IFS= read -r line; do
        status="${line%% *}"
        testcase="${line#* }"
        printf "$RESULT_FMT" "Ascend:${cid}" "pytest" "${testcase}" "${status}" >> "${pytest_result_file}"
        case "$status" in
            PASSED) total_passed=$((total_passed + 1)) ;;
            FAILED) total_failed=$((total_failed + 1)) ;;
        esac
    done < "$_grep_tmp"
    rm -f "$_grep_tmp"
    rm -f "${rf}"
done
# Print detailed pytest logs
cat "${pytest_log_file}"
# Print aggregated pytest result table
Print_Stage_Result "${pytest_result_file}" "PYTEST"
# Print summary statistics
echo "PYTEST SUMMARY: ${total_failed} failed, ${total_passed} passed"
rm -f "${pytest_log_file}"

if [ "${failed}" -ne 0 ]; then
    if [[ ${ascend_fail_not_return} != "ON" ]]; then
        exit 1
    fi
    overall_status=1
fi
echo "---------- Run MindSpore Lite API SUCCESS ----------"
#---------------------------------------------------------

if [[ ${ascend_fail_not_return} == "ON" && ${overall_status} -ne 0 ]]; then
    echo "Debug mode ON: completed with failures (overall_status=${overall_status})"
fi
echo "success"
exit ${overall_status}

