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

# Example:sh run_benchmark_nets.sh -r /home/temp_test -m /home/temp_test/models -e import_ms_and_mslite
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

echo "Run testcases of import mindspore and mindspore_lite..."
echo "-----------------------------------------------------------------------------------------"
cp ${models_path}/mobilenetv2.mindir ${basepath}

pytest -vra ${basepath}/python/import_ms_and_mslite/test_api_import_ms_and_mslite.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_api_import_ms_and_mslite failed."
  exit ${RET}
fi
echo "test_api_import_ms_and_mslite success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_api_import_mslite_and_ms.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_api_import_mslite_and_ms failed."
  exit ${RET}
fi
echo "test_api_import_mslite_and_ms success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_only_import_ms_and_mslite.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_only_import_ms_and_mslite failed."
  exit ${RET}
fi
echo "test_only_import_ms_and_mslite success"

pytest -vra ${basepath}/python/import_ms_and_mslite/test_only_import_mslite_and_ms.py
RET=$?
if [ ${RET} -ne 0 ]; then
  echo "run test_only_import_mslite_and_ms failed."
  exit ${RET}
fi
rm -rf ${basepath}/mobilenetv2.mindir
echo "test_only_import_mslite_and_ms success"
