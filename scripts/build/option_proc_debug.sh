#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

build_option_proc_v()
{
  export ENABLE_VERBOSE="on"
  export VERBOSE="VERBOSE=1"
}

build_option_proc_c()
{
  check_on_off $OPTARG c
  export MSLITE_ENABLE_COVERAGE="$OPTARG"
}

build_option_proc_t()
{
  if [[ "X$OPTARG" == "Xon" || "X$OPTARG" == "Xut" ]]; then
    export MSLITE_ENABLE_TESTCASES="on"
  elif [[ "X$OPTARG" == "Xoff" ]]; then
    export MSLITE_ENABLE_TESTCASES="off"
  else
    echo "Invalid value ${OPTARG} for option -t"
    usage
    exit 1
  fi
}

build_option_proc_g()
{
  check_on_off $OPTARG g
  export USE_GLOG="$OPTARG" 
}

build_option_proc_h()
{
  usage
  exit 0
}

build_option_proc_a()
{
  check_on_off $OPTARG a
  export ENABLE_ASAN="$OPTARG"
}

