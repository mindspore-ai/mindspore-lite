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

set -e

usage()
{
  echo "Usage:"
  echo "bash build.sh [-d] [-r] [-v] [-c on|off] [-t ut|st] [-g on|off] [-h] \\"
  echo "              [-a on|off] [-i] [-j[n]] [-n full|lite|off] \\"
  echo "              [-I arm64|arm32|x86_64] [-A on|off] [-S on|off] \\"
  echo "              [-W sse|neon|avx|avx512|off] [-F on|off] \\"
  echo ""
  echo "Options:"
  echo "    -d Debug mode"
  echo "    -r Release mode, default mode"
  echo "    -v Display build command"
  echo "    -c Enable code coverage, default off"
  echo "    -t Run testcases, default off"
  echo "    -g Use glog to output log, default on"
  echo "    -h Print usage"
  echo "    -a Enable ASAN, default off"
  echo "    -i Enable increment building, default off"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -n Compile minddata with mindspore lite, available: off, lite, full, lite_cv, full mode in lite train and lite_cv, wrapper mode in lite predict"
  echo "    -I Enable compiling mindspore lite for arm64, arm32 or x86_64, default disable mindspore lite compilation"
  echo "    -A Enable compiling mindspore lite aar package, option: on/off, default: off"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "    -W Enable SIMD instruction set, use [sse|neon|avx|avx512|off], default avx for cloud CPU backend"
  echo "    -F Use fast hash table in mindspore-lite compiler, default on"
}
