#!/bin/bash
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
BASEPATH=$(cd "$(dirname $0)"; pwd)
export CUDA_PATH=""
export BUILD_PATH="${BASEPATH}/build/"

source ./scripts/build/usage.sh
source ./scripts/build/default_options.sh
source ./scripts/build/option_proc_debug.sh
source ./scripts/build/option_proc_lite.sh
source ./scripts/build/process_options.sh

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

update_submodule()
{
  git submodule update --init --remote mindspore
#  cd "${BASEPATH}/mindspore"
#  TODO(compile so used in mindspore-lite)
}

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}

make_clean()
{
  echo "enable make clean"
  cd "${BUILD_PATH}/mindspore"
  cmake --build . --target clean
}
update_submodule
echo "---------------- MindSpore: build start ----------------"
init_default_options
process_options "$@"

export COMPILE_MINDDATA_LITE
export ENABLE_VERBOSE
export LITE_PLATFORM
export LITE_ENABLE_AAR
source mindspore-lite/build_lite.sh
echo "---------------- MindSpore: build end   ----------------"
