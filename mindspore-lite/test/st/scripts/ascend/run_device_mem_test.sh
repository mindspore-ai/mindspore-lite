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

echo "Begin run run_device_mem_test"
echo "cpp dir: ${LITE_ST_CPP_DIR}"
echo "model path: ${LITE_ST_MODEL}"
echo "lite home: ${LITE_HOME}"

export ASCEND_PATH=/usr/local/Ascend
if [ -d "${ASCEND_PATH}/ascend-toolkit" ]; then
    source ${ASCEND_PATH}/ascend-toolkit/set_env.sh
else
    source ${ASCEND_PATH}/latest/bin/setenv.bash
fi
export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH

cd ${LITE_ST_CPP_DIR}/device_example_cpp || exit 1

bash build.sh Ascend
if [ ! -f "./build/mindspore_quick_start_cpp" ];then
  echo "Failed to build device_example_cpp"
  exit 1
fi

build/mindspore_quick_start_cpp ${LITE_ST_MODEL}
Run_device_example_status=$?
if [[ ${Run_device_example_status} != 0 ]];then
  echo "Run device example failed"
else
  echo "Run device example success"
fi
exit ${Run_device_example_status}
