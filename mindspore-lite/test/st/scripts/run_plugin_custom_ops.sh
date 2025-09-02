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

function run_plugin_custom_ops() {
  while read line; do
    local plugin_custom_ops=${line}
    if [[ $plugin_custom_ops == \#* || $plugin_custom_ops == "" ]]; then
      continue
    fi
    echo "run ${plugin_custom_ops}"
    python ${basepath}/python/plugin_custom_ops/${plugin_custom_ops} Ascend
    local ret=$?
    if [ ${ret} -ne 0 ]; then
      echo "run ${plugin_custom_ops} failed."
      exit ${ret}
    fi
  done < ${ascend_custom_ops_config}
}

# Example:sh run_benchmark_nets.sh -r /home/temp_test -e plugin_custom_ops
while getopts "r:m:e:l:" opt; do
    case ${opt} in
        r)
            release_path=${OPTARG}
            echo "release_path is ${release_path}"
            ;;
        m)
            models_path=${OPTARG}
            echo "models_path is ${models_path}"
            ;;
        e)
            backend=${OPTARG}
            echo "backend is ${backend}"
            ;;
        l)
            level=${OPTARG}
            echo "level is ${level}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

ms_whl_path=`ls ${release_path}/mindspore-*.whl`
mslite_whl_path=`ls ${release_path}/mindspore_lite-*.whl`
basepath=$(pwd)

if [[ -f "${ms_whl_path}" ]]; then
  pip uninstall mindspore -y || exit 1
  pip install ${ms_whl_path}  || exit 1
  echo "install mindspore python whl success."
else
  echo "not find mindspore python whl.."
  exit 1
fi

if [[ -f "${mslite_whl_path}" ]]; then
  pip uninstall mindspore-lite -y || exit 1
  pip install ${mslite_whl_path}  || exit 1
  echo "install mindspore_lite python whl success."
else
  echo "not find mindspore_lite python whl.."
  exit 1
fi

ascend_custom_ops_config=${basepath}/config/ascend_custom_ops.cfg

echo "Run testcases of mindspore_lite plugin custom ops..."
echo "-----------------------------------------------------------------------------------------"
run_plugin_custom_ops
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_plugin_custom_ops failed."
  exit ${RET}
fi
echo "test_plugin_custom_ops success"
