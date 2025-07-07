#!/bin/bash

CURRENT_PATH=$(pwd)
PROJECT_ROOT_HOME="${CURRENT_PATH}/../../.."
MINDSPORE_LITE_HOME=${PROJECT_ROOT_HOME}/mindspore-lite
MINDSPORE_HOME=${PROJECT_ROOT_HOME}/mindspore
echo "PROJECT_ROOT_HOME path is ${PROJECT_ROOT_HOME}"
cd "${PROJECT_ROOT_HOME}" || exit 1
CROPPER_OUTPUT_DIR=build/tools/cropper
mkdir -p ${CROPPER_OUTPUT_DIR}
MAPPING_OUTPUT_FILE_NAME_TMP=${CROPPER_OUTPUT_DIR}/cropper_mapping_tmp.cfg
MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP=${CROPPER_OUTPUT_DIR}/cropper_mapping_train_tmp.cfg
CPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_cpu.cfg
GPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_gpu.cfg
NPU_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_npu.cfg
CPU_TRAIN_MAPPING_OUTPUT_FILE=${CROPPER_OUTPUT_DIR}/cropper_mapping_cpu_train.cfg
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
[ -n "${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
[ -n "${CPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${CPU_MAPPING_OUTPUT_FILE}
[ -n "${GPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${GPU_MAPPING_OUTPUT_FILE}
[ -n "${NPU_MAPPING_OUTPUT_FILE}" ] && rm -f ${NPU_MAPPING_OUTPUT_FILE}
[ -n "${CPU_TRAIN_MAPPING_OUTPUT_FILE}" ] && rm -f ${CPU_TRAIN_MAPPING_OUTPUT_FILE}

ops_list=()
DEFINE_STR="-DENABLE_ANDROID -DENABLE_ARM -DENABLE_ARM64 -DENABLE_NEON -DNO_DLIB -DUSE_ANDROID_LOG -DANDROID -DENABLE_FP16 -DENABLE_MINDRT -DUSE_GLOG"
# get the flatbuffers path
if [ ${MSLIBS_CACHE_PATH} ]; then
  FLATBUFFERS_LIST=()
  while IFS='' read -r line; do FLATBUFFERS_LIST+=("$line"); done < <(ls -d ${MSLIBS_CACHE_PATH}/flatbuffers_*/include)
  FLATBUFFERS=${FLATBUFFERS_LIST[0]}
  echo "FLATBUFFERS path is ${FLATBUFFERS}"
else
  FLATBUFFERS=$(ls -d build/.mslib/flatbuffers_*/include)
  echo "FLATBUFFERS path is ${FLATBUFFERS}"
fi

if [ ${MSLIBS_CACHE_PATH} ]; then
  NLOHMANN_LIST=()
  while IFS='' read -r line; do NLOHMANN_LIST+=("$line"); done < <(ls -d ${MSLIBS_CACHE_PATH}/nlohmann_*/include)
  NLOHMANN=${NLOHMANN_LIST[0]}
  echo "NLOHMANN path is ${NLOHMANN}"
else
  NLOHMANN=$(ls -d build/.mslib/nlohmann_*/include)
  echo "NLOHMANN path is ${NLOHMANN}"
fi

# get the glog path
if [ ${MSLIBS_CACHE_PATH} ]; then
  GLOG_LIST=()
  while IFS='' read -r line; do GLOG_LIST+=("$line"); done < <(ls -d ${MSLIBS_CACHE_PATH}/glog_*/include)
  GLOG=${GLOG_LIST[0]}
  echo "GLOG path is ${GLOG}"
else
  GLOG=$(ls -d build/.mslib/glog_*/include)
  echo "GLOG path is ${GLOG}"
fi

HEADER_LOCATION="-I${MINDSPORE_HOME}
-I${MINDSPORE_HOME}/mindspore/core/include
-I${MINDSPORE_HOME}/mindspore/core/mindrt
-I${MINDSPORE_HOME}/mindspore/core/mindrt/include
-I${MINDSPORE_HOME}/mindspore/ccsrc
-I${MINDSPORE_LITE_HOME}
-I${MINDSPORE_LITE_HOME}/src
-I${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu
-I${PROJECT_ROOT_HOME}/third_party
-I${PROJECT_ROOT_HOME}/build
-I${PROJECT_ROOT_HOME}/third_party/securec/include
-I${FLATBUFFERS}
-I${NLOHMANN}
-I${GLOG}
-I${PROJECT_ROOT_HOME}/build/schema
-I${PROJECT_ROOT_HOME}/build/schema/inner
-I${PROJECT_ROOT_HOME}/build/src
-I${MINDSPORE_HOME}/mindspore/ops/kernel/cpu
-I${MINDSPORE_HOME}/mindspore/ccsrc/minddata/dataset"

REMOVE_LISTS_STR=""
getDeep() {
  map_files=$(gcc -MM ${2} ${DEFINE_STR} ${HEADER_LOCATION})
  if [[ ${map_files} == "" ]]; then
    echo "failed to get deep any file from file: ${2}, compile terminated unexpectedly"
  fi
  # first is *.o second is *.cc
  array_deep=()
  while IFS='' read -r line; do array_deep+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build|third_party|type_id.h|core/utils|glog' | egrep -v ${REMOVE_LISTS_STR})
  # shellcheck disable=SC2068
  for array_deep_file in ${array_deep[@]}; do
    # only add existing files
    if [[ -e ${array_deep_file%h*}cc ]]; then
      file_split=$(echo ${array_deep_file} | awk -F '/' '{print $NF}')
      if [[ "$4" != "train_source" ]] ; then
        echo "${1},${3},${file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
      fi
      echo "${1},${3},${file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
    fi
    if [[ -e ${array_deep_file%h*}c ]]; then
      file_split=$(echo ${array_deep_file} | awk -F '/' '{print $NF}')
      if [[ "$4" != "train_source" ]] ; then
        echo "${1},${3},${file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
      fi
      echo "${1},${3},${file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
    fi
  done
}

getNnaclKernelFile() {
  echo "start get operator mapping nnacl kernel $3"
  # shellcheck disable=SC2068
  for type in ${ops_list[@]}; do
    # get mapping
    ret=$(egrep -r -l "$1${type}, $4," $2)
    array=("${ret}")
    # shellcheck disable=SC2068
    for file in ${array[@]}; do
      # delete \n
      out_file=$(echo ${file} | awk -F '/' '{print $NF}')
      # concat schemaType + fileType + fileName append to files
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
      map_files=$(gcc -MM ${file} ${DEFINE_STR} ${HEADER_LOCATION})
      if [[ ${map_files} == "" ]]; then
        echo "failed to get operator mapping any file from file: ${file}, compile terminated unexpectedly"
      fi
      # first is *.o second is *.cc
      array_file=()
      while IFS='' read -r line; do array_file+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build|third_party|type_id.h|core/utils|glog' | egrep -v ${REMOVE_LISTS_STR})
      # shellcheck disable=SC2068
      for array_file in ${array_file[@]}; do
        # only add existing files
        if [[ -e ${array_file%h*}cc ]]; then
          getDeep ${type} ${array_file%h*}cc ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
        fi
        if [[ -e ${array_file%h*}c ]]; then
          getDeep ${type} ${array_file%h*}c ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
        fi
      done
    done
  done
}

getOpsFile() {
  echo "start get operator mapping file $3"
  # shellcheck disable=SC2068
  for type in ${ops_list[@]}; do
    # get mapping
    ret=$(egrep -r -l "$1${type}," $2)
    array=("${ret}")
    # shellcheck disable=SC2068
    for file in ${array[@]}; do
      # delete \n
      out_file=$(echo ${file} | awk -F '/' '{print $NF}')
      # concat schemaType + fileType + fileName append to files
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
      map_files=$(gcc -MM ${file} ${DEFINE_STR} ${HEADER_LOCATION})
      if [[ ${map_files} == "" ]]; then
        echo "failed to get operator mapping any file from file: ${file}, compile terminated unexpectedly"
      fi
      # first is *.o second is *.cc
      array_file=()
      while IFS='' read -r line; do array_file+=("$line"); done < <(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build|third_party|type_id.h|core/utils|glog' | egrep -v ${REMOVE_LISTS_STR})
      # shellcheck disable=SC2068
      for array_file in ${array_file[@]}; do
        # only add existing files
        if [[ -e ${array_file%h*}cc ]]; then
          getDeep ${type} ${array_file%h*}cc ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
        fi
        if [[ -e ${array_file%h*}c ]]; then
          getDeep ${type} ${array_file%h*}c ${3} &
          getDeep ${type} ${array_file} ${3} &
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
        fi
      done
    done
  done
}

getFilesFromArr() {
  local arr_files=${1}
  # echo " func parm 1 : ${arr_files[@]}"
  # echo " func parm 2 : $2"
  # shellcheck disable=SC2068
  for file in ${arr_files[*]}; do
    map_files=$(gcc -MM ${file} ${DEFINE_STR} ${HEADER_LOCATION})
    if [[ ${map_files} == "" ]]; then
      echo "failed to get any file from arr_file: ${file}, compile terminated unexpectedly"
    fi
    # first is *.o second is *.cc
    # shellcheck disable=SC2207
    array_runtime=($(echo ${map_files} | awk -F '\' '{for(i=3;i<=NF;i++){print $i}}' | egrep -v 'flatbuffers|build|third_party|type_id.h|glog' | egrep -v ${REMOVE_LISTS_STR}))
    # only add existing files
    for array_runtime_file in "${array_runtime[@]}"; do
      if [[ -e ${array_runtime_file%h*}cc && ! ${all_files[*]} =~ ${array_runtime_file%h*}cc ]]; then
        all_files=("${all_files[@]}" "${array_runtime_file%h*}cc")
        getDeep "CommonFile" ${array_runtime_file%h*}cc "common" $2
      fi
      if [[ -e ${array_runtime_file%h*}c && ! ${all_files[*]} =~ ${array_runtime_file%h*}c ]]; then
        all_files=("${all_files[@]}" "${array_runtime_file%h*}c")
        getDeep "CommonFile" ${array_runtime_file%h*}c "common" $2
      fi
    done
  done
}

getCommonFile() {
  echo "start get common files"
  include_h=()
  while IFS='' read -r line; do include_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/include/*.h)
  regist_include_h=()
  while IFS='' read -r line; do regist_include_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/include/registry/*kernel*.h)
  src_files_h=()
  while IFS='' read -r line; do src_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/*.h)
  common_files_h=()
  while IFS='' read -r line; do common_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/common/*.h)
  runtime_files_h=()
  while IFS='' read -r line; do runtime_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/*.h)
  while IFS='' read -r line; do runtime_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/control_flow/*.h)
  while IFS='' read -r line; do runtime_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/extendrt/mindir_loader/*.h)
  nnacl_files_h=(
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/cxx_utils.h
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/nnacl_kernel.h
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/nnacl_manager.h
  )
  mindrt_files_h=()
  while IFS='' read -r line; do mindrt_files_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/include/actor/*.h)
  while IFS='' read -r line; do mindrt_files_h+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/include/thread/*.h)
  others_files_h=(
    "${MINDSPORE_LITE_HOME}"/src/litert/infer_manager.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/infer_register.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/nnacl_utils.h
    "${MINDSPORE_LITE_HOME}"/src/common/ops/populate/populate_register.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/op_base.h
    "${MINDSPORE_HOME}"/mindspore/core/include/ir/dtype/type_id.h
    "${MINDSPORE_HOME}"/mindspore/core/include/utils/overload.h
    "${MINDSPORE_LITE_HOME}"/tools/common/option.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/intrinsics/ms_simd_instructions.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/intrinsics/ms_simd_instructions_fp16.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/infer.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/common_infer.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensor_c.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/errorcode.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/common_func.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensor_c_utils.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensorlist_c.h
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensorlist_c_utils.h
    "${MINDSPORE_HOME}"/mindspore/core/include/utils/log_adapter.h
    "${MINDSPORE_HOME}"/mindspore/core/include/ir/api_tensor_impl.h
    "${MINDSPORE_LITE_HOME}"/src/litert/cxx_api/tensor/tensor_impl.h
  )
  all_files_h=("${include_h[@]}" "${regist_include_h[@]}" "${src_files_h[@]}" "${common_files_h[@]}"
               "${runtime_files_h[@]}" "${nnacl_files_h[@]}" "${others_files_h[@]}" "${mindrt_files_h[@]}"
  )

  # concat regx
  REMOVE_LISTS_STR="${all_files_h[0]}"
  # shellcheck disable=SC2068
  for val in ${all_files_h[@]:1}; do
    REMOVE_LISTS_STR="$REMOVE_LISTS_STR|$val"
  done

  cxx_api_files=()
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/graph/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/model/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/tensor/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/*.cc)
  while IFS='' read -r line; do cxx_api_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/c_api/*.cc)
  mindrt_files=()
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/*.cc)
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/async/*.cc)
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/actor/*.cc)
  while IFS='' read -r line; do mindrt_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/core/mindrt/src/thread/*.cc)
  src_files=()
  while IFS='' read -r line; do src_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/*.cc)
  regist_files=()
  while IFS='' read -r line; do regist_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/registry/*.cc)
  common_files=()
  while IFS='' read -r line; do common_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/common/*.cc)
  runtime_files_cc=()
  while IFS='' read -r line; do runtime_files_cc+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/*.cc)
  while IFS='' read -r line; do runtime_files_cc+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/control_flow/*.cc)
  while IFS='' read -r line; do runtime_files_cc+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/extendrt/mindir_loader/*.cc)
  nnacl_files_cc=(
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/cxx_utils.cc
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/nnacl_kernel.cc
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/nnacl/nnacl_manager.cc
  )
  # sava all assembly files
  assembly_files=()
  while IFS='' read -r line; do assembly_files+=("$line"); done < <(ls ${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/assembly/*/*.S)
  others_files_c=(
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/nnacl_utils.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/errorcode.c
    "${MINDSPORE_LITE_HOME}"/src/litert/infer_manager.cc
    "${MINDSPORE_LITE_HOME}"/src/common/ops/populate/populate_register.cc
    "${MINDSPORE_LITE_HOME}"/src/common/ops/populate/custom_populate.cc
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/infer_register.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/shape_fusion_infer.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/infer/common_infer.c
    "${MINDSPORE_LITE_HOME}"/src/litert/kernel/cpu/fp32/shape_fusion_fp32.cc
    "${MINDSPORE_HOME}"/mindspore/core/utils/status.cc
    "${MINDSPORE_HOME}"/mindspore/core/utils/log_adapter.cc
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/kernel.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensor_c_utils.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/tensorlist_c_utils.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/base/format_transpose.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/base/cast_base.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/fp32/transpose_fp32.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/fp32/pack_fp32.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/fp16/pack_fp16.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/fp32/pack_fp32_opt.c
    "${MINDSPORE_HOME}"/mindspore/ops/kernel/cpu/nnacl/nnacl_common.c
  )
  all_files=("${src_files[@]}" "${regist_files[@]}" "${common_files[@]}" "${runtime_files_cc[@]}"
    "${others_files_c[@]}" "${assembly_files[@]}" "${nnacl_files_cc[@]}" "${mindrt_files[@]}"
    "${cxx_api_files[@]}"
  )
  getFilesFromArr "${all_files[*]}" &
  getFilesFromArr "${all_files_h[*]}" &
  wait
  # shellcheck disable=SC2068
  for file in ${all_files[@]}; do
    file=$(echo ${file} | awk -F '/' '{print $NF}')
    echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
    echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
  done
}

getTrainCommonFile() {
  # save train files
  train_files=()
  while IFS='' read -r line; do train_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/train/*.cc)
  while IFS='' read -r line; do train_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/callback/*.cc)
  while IFS='' read -r line; do train_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/metrics/*.cc)
  while IFS='' read -r line; do train_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/cxx_api/train/*.cc)
  others_train_files=(
    "${MINDSPORE_LITE_HOME}"/tools/common/storage.cc
  )
  all_files_train=("${train_files[@]}" "${others_train_files[@]}")
  train_files_h=()
  while IFS='' read -r line; do train_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/include/train/*.h)
  while IFS='' read -r line; do train_files_h+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/train/*.h)
  all_files_train_h=("${train_files_h[@]}"
  )
  getFilesFromArr "${all_files_train[*]}"  "train_source" &
  getFilesFromArr "${all_files_train_h[*]}"  "train_source" &
  wait
  sleep 0.5
  # shellcheck disable=SC2068
  for file in ${all_files_train[@]}; do
    file=$(echo ${file} | awk -F '/' '{print $NF}')
    echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
  done
}

# The x86 platform cannot search based on header files, so manually search for the first layer.
# opencl & ddk
getOpsFileWithNoDeepSearch() {
  echo "start get gpu/npu operator mapping file $3"
  # shellcheck disable=SC2068
  for type in ${ops_list[@]}; do
    # get mapping
    ret=$(egrep -r -l "$1${type}," $2)
    array=("${ret}")
    # shellcheck disable=SC2068
    for file in ${array[@]}; do
      # delete \n
      out_file=$(echo ${file} | awk -F '/' '{print $NF}')
      # concat schemaType + fileType + fileName append to files
      echo "${type},${3},${out_file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}

      local ret=""
      ret=$(egrep -r *.h\" ${file} | awk -F '\"' '{print $2}')
      local ret_h=""
      ret_h=$(egrep -r *.h\" ${file%cc*}h | awk -F '\"' '{print $2}')
      local depend_file=("${ret}" "${ret_h}")
      for array_file in ${depend_file[@]}; do
        # only add existing files
        if [[ -e ${MINDSPORE_LITE_HOME}/${array_file%h*}cc ]]; then
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}cc.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
        if [[ -e ${MINDSPORE_LITE_HOME}/${array_file%h*}c ]]; then
          array_file_split=$(echo ${array_file} | awk -F '/' '{print $NF}')
          echo "${type},${3},${array_file_split%h*}c.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
        fi
      done
    done
  done
}

# automatically generate operator list
generateOpsList() {
  echo "start generate operator list"
  ops_list=()
  while IFS='' read -r line; do ops_list+=("$line"); done < <(grep -Rn "^table" "${MINDSPORE_LITE_HOME}/schema/ops.fbs" | awk -F ' ' '{print $2}')
  ops_num=$((${#ops_list[@]}))
  echo "ops nums:${ops_num}"
}
echo "Start getting all file associations."
generateOpsList
getCommonFile
getTrainCommonFile
# get src/common/ops
getOpsFile "REG_POPULATE\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/common/ops/populate" "prototype" &
getOpsFile "REG_INFER\(.*?, PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/infer" "prototype" &
# support for cpu
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu" "kNumberTypeFloat32" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu" "kNumberTypeFloat16" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu" "kNumberTypeInt8" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu" "kNumberTypeInt32" &
getOpsFile "REG_KERNEL\(.*?, kNumberTypeBool, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu" "kNumberTypeInt32" &
#support for nnacl kernel
getNnaclKernelFile "REG_KERNEL_CREATOR\(PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/kernel" "kNumberTypeFloat32" "kNumberTypeFloat32" &
getNnaclKernelFile "REG_KERNEL_CREATOR\(PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/kernel" "kNumberTypeFloat16" "kNumberTypeFloat16" &
getNnaclKernelFile "REG_KERNEL_CREATOR\(PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/kernel" "kNumberTypeInt8" "kNumberTypeInt8" &
getNnaclKernelFile "REG_KERNEL_CREATOR\(PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/kernel" "kNumberTypeInt32" "kNumberTypeInt32" &
getNnaclKernelFile "REG_KERNEL_CREATOR\(PrimType_" "${MINDSPORE_HOME}/mindspore/ops/kernel/cpu/nnacl/kernel" "kNumberTypeInt32" "kNumberTypeBool" &
getNnaclKernelFile "NNACL_KERNEL\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu/nnacl" "kNumberTypeFloat32" "kNumberTypeFloat32" &
getNnaclKernelFile "NNACL_KERNEL\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu/nnacl" "kNumberTypeFloat16" "kNumberTypeFloat16" &
getNnaclKernelFile "NNACL_KERNEL\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu/nnacl" "kNumberTypeInt8" "kNumberTypeInt8" &
getNnaclKernelFile "NNACL_KERNEL\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu/nnacl" "kNumberTypeInt32" "kNumberTypeInt32" &
getNnaclKernelFile "NNACL_KERNEL\(PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/cpu/nnacl" "kNumberTypeInt32" "kNumberTypeBool" &

wait
sleep 0.5
# remove duplicate files
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | uniq >${CPU_MAPPING_OUTPUT_FILE}
chmod 444 ${CPU_MAPPING_OUTPUT_FILE}
# remove duplicate files
sort ${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP} | uniq >${CPU_TRAIN_MAPPING_OUTPUT_FILE}
chmod 444 ${CPU_TRAIN_MAPPING_OUTPUT_FILE}

# support for gpu
opencl_files=()
while IFS='' read -r line; do opencl_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/*.cc)
while IFS='' read -r line; do opencl_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/kernel/gpu/opencl/*.cc)
opencl_others_files=(
  "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel/fusion_eltwise.cc"
  "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel/to_format.cc"
  "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel/gl_to_cl.cc"
)
opencl_files=("${opencl_files[@]}" "${opencl_others_files[@]}")
# shellcheck disable=SC2068
for file in ${opencl_files[@]}; do
  file=$(echo ${file} | awk -F '/' '{print $NF}')
  echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
done

getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat32, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel" "kNumberTypeFloat32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeFloat16, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel" "kNumberTypeFloat16" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt8, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel" "kNumberTypeInt8" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeInt32, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel" "kNumberTypeInt32" &
getOpsFileWithNoDeepSearch "REG_KERNEL\(.*?, kNumberTypeBool, PrimitiveType_" "${MINDSPORE_LITE_HOME}/src/litert/kernel/opencl/kernel" "kNumberTypeInt32" &
sleep 0.5
wait
sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | uniq >${GPU_MAPPING_OUTPUT_FILE}
chmod 444 ${GPU_MAPPING_OUTPUT_FILE}

# support for npu
npu_files=()
while IFS='' read -r line; do npu_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/delegate/npu/*.cc)
while IFS='' read -r line; do npu_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/delegate/npu/op/*.cc)
while IFS='' read -r line; do npu_files+=("$line"); done < <(ls ${MINDSPORE_LITE_HOME}/src/litert/delegate/npu/pass/*.cc)
npu_others_files=("${MINDSPORE_LITE_HOME}/src/litert/delegate/delegate_utils.cc")
npu_files=("${npu_files[@]}" "${npu_others_files[@]}")

# shellcheck disable=SC2068
for file in ${npu_files[@]}; do
  file=$(echo ${file} | awk -F '/' '{print $NF}')
  echo "CommonFile,common,${file}.o" >>${MAPPING_OUTPUT_FILE_NAME_TMP}
done

sort ${MAPPING_OUTPUT_FILE_NAME_TMP} | uniq >${NPU_MAPPING_OUTPUT_FILE}
chmod 444 ${NPU_MAPPING_OUTPUT_FILE}

# modify file permissions to read-only
[ -n "${MAPPING_OUTPUT_FILE_NAME_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TMP}
[ -n "${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}" ] && rm -f ${MAPPING_OUTPUT_FILE_NAME_TRAIN_TMP}
echo "Complete all tasks."
