#!/bin/bash
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

# shellcheck disable=SC2034

set -e

init_default_options()
{
  # Init default values of build options
  export THREAD_NUM=8
  export DEBUG_MODE="off"
  VERBOSE=""
  export MSLITE_ENABLE_COVERAGE="off"
  # export MSLITE_ENABLE_TESTCASES="off"
  export ENABLE_ASAN="off"
  export ENABLE_PROFILE="off"
  export INC_BUILD="off"
  export ENABLE_DUMP_IR="on"
  export COMPILE_MINDDATA_LITE="lite_cv"
  export ASCEND_VERSION="910"
  export COMPILE_LITE="off"
  export LITE_PLATFORM=""
  export LITE_ENABLE_AAR="off"
  export USE_GLOG="on"
  export ENABLE_ACL="off"
  export ENABLE_VERBOSE="off"
  export MSLITE_ENABLE_GITEE_MIRROR="off"
  export X86_64_SIMD="off"
  export ARM_SIMD="off"
  export USER_ENABLE_DUMP_IR=false
  export ENABLE_FAST_HASH_TABLE="on"
}
