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
}

build_exit()
{
    echo "$@" >&2
    stty echo
    exit 1
}
echo "---------------- MindSpore-Lite: build start ----------------"
init_default_options
process_options "$@"
export ENABLE_VERBOSE
export LITE_PLATFORM
export LITE_ENABLE_AAR
if [[ "X${BUILD_OPTIONAL_TARGET:-off}" == "Xlite_boost" ]]; then
  LITE_BOOST_ARGS=(-j "${THREAD_NUM}")
  if [[ "${DEBUG_MODE}" == "on" ]]; then
    LITE_BOOST_ARGS+=(-d)
  else
    LITE_BOOST_ARGS+=(-r)
  fi
  [[ "${ENABLE_VERBOSE}" == "on" ]] && LITE_BOOST_ARGS+=(-v)
  [[ "${INC_BUILD}" == "on" ]] && LITE_BOOST_ARGS+=(-i)
  bash "${BASEPATH}/mindspore-lite/lite_boost/build.sh" "${LITE_BOOST_ARGS[@]}"
elif [[ "X${BUILD_OPTIONAL_TARGET:-off}" == "Xlite_llm" ]]; then
  LITE_LLM_ARGS=(-j "${THREAD_NUM}")
  if [[ "${DEBUG_MODE}" == "on" ]]; then
    LITE_LLM_ARGS+=(-d)
  else
    LITE_LLM_ARGS+=(-r)
  fi
  [[ "${ENABLE_VERBOSE}" == "on" ]] && LITE_LLM_ARGS+=(-v)
  [[ "${INC_BUILD}" == "on" ]] && LITE_LLM_ARGS+=(-i)
  # Build the unit tests when the main repo's testcases switch is on.  Both
  # host and cross builds compile them: host runs them via ctest, the
  # aarch64 cross build produces test binaries that can be pushed to a device
  # (BinRunner) for on-device verification.
  if [[ "${MSLITE_ENABLE_TESTCASES:-off}" == "on" || "${MSLITE_ENABLE_TESTCASES:-off}" == "ut" ]]; then
    LITE_LLM_ARGS+=(-t on)
  fi
  # OHOS cross-compile for the Kirin NPU when the NDK is provided.
  if [[ -n "${OHOS_NDK:-}" ]]; then
    LITE_LLM_ARGS+=(-b nnrt)
  fi
  bash "${BASEPATH}/mindspore-lite/lite_llm/build.sh" "${LITE_LLM_ARGS[@]}"
else
  if [ "${MSLITE_ENABLE_SKIP_SUBMODULE_UPDATE}" = "on" ]; then
    echo "Skipping update submodule"
  else
      update_submodule
  fi
  source ./scripts/build/build_lite.sh
fi
echo "---------------- MindSpore-Lite: build end   ----------------"
