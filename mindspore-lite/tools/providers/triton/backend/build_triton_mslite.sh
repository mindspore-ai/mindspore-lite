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

function Run_Build() {
  # decompress release_pkg
  cd ${open_source_ms_path}/output/ || exit 1
  if [[ ${platform} = "arm64" ]]; then
    platform="aarch64"
  elif [[ ${platform} = "x86_64" ]]; then
    platform="x64"
  fi
  file_name=$(ls ./*linux-${platform}.tar.gz)
  IFS="-" read -r -a file_name_array <<< "$file_name"
  version=${file_name_array[2]}
  tar -xf mindspore-lite-${version}-linux-${platform}.tar.gz

  export MINDSPORE_LITE_PKG_ROOT_PATH=${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}
  # install rapidjson manually.
  mkdir -p ${open_source_ms_path}/mindspore-lite/tools/providers/triton/backend/third_party
  cd ${open_source_ms_path}/mindspore-lite/tools/providers/triton/backend/third_party
  if [ ! -d "RapidJSON" ]; then
    git clone https://gitee.com/Tencent/RapidJSON.git || exit 1
  fi

  # compile triton mslite backend
  cd ${open_source_ms_path}/mindspore-lite/tools/providers/triton/backend/
  rm -rf build; mkdir build; cd build;
  cmake -DCMAKE_INSTALL_PREFIX:PATH=../install -DTRITON_RAPID_JSON_PATH:PATH=./third_party/RapidJSON/include .. || exit 1
  make install -j ${thread_num} || exit 1
  cd -
  echo "build for triton backend success"

  # cp to release package folder
  mkdir -p ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}/tools/providers/triton/ || exit 1
  cp -r ${open_source_ms_path}/mindspore-lite/tools/providers/triton/backend/install/backends/mslite \
      ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}/tools/providers/triton/ || exit 1
  echo "cp triton backend so to release pkg success"

  cd ${open_source_ms_path}/output
  rm ./mindspore-lite-${version}-linux-${platform}.tar.gz
  tar -zcf ./mindspore-lite-${version}-linux-${platform}.tar.gz ./mindspore-lite-${version}-linux-${platform}/ || exit 1
  sha256sum ./mindspore-lite-${version}-linux-${platform}.tar.gz > ./mindspore-lite-${version}-linux-${platform}.tar.gz.sha256 || exit 1
  rm -rf ./mindspore-lite-${version}-linux-${platform}
  echo "package ${open_source_ms_path}/output/mindspore-lite-${version}-linux-${platform}.tar.gz updated."
  exit
}

# bashpath should be /home/jenkins/agent-working-dir/workspace/Compile_Lite_xxx/
basepath=$(pwd)
echo "basepath is ${basepath}"
#set -e
open_source_ms_path=${basepath}/mindspore-lite

# Example:sh build_triton_mslite.sh -I arm64 -jn
while getopts "I:j:" opt; do
    case ${opt} in
        I)
            platform=${OPTARG}
            echo "platform is ${OPTARG}"
            ;;
        j)
            thread_num=${OPTARG}
            echo "thread_num is ${OPTARG}"
            ;;
        ?)
        echo "unknown para"
        exit 1;;
    esac
done

echo "start building for triton mslite backend..."
Run_Build
Run_build_PID=$!
exit ${Run_build_PID}
