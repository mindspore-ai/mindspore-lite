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

# check and set options
process_options()
{
  # Process the options
  while getopts 'dfhirvA:I:S:W:a:c:g:j:n:t:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      d)
        DEBUG_MODE="on" ;;
      n)
        build_option_proc_n ;;
      r)
        export DEBUG_MODE="off" ;;
      v)
        build_option_proc_v ;;
      j)
        export THREAD_NUM=$OPTARG ;;
      c)
        build_option_proc_c ;;
      t)
        build_option_proc_t ;;
      g)
        build_option_proc_g ;;
      h)
        build_option_proc_h ;;
      a)
        build_option_proc_a ;;
      i)
        export INC_BUILD="on" ;;
      S)
        build_option_proc_upper_s ;;
      I)
        build_option_proc_upper_i ;;
      A)
        build_option_proc_upper_a ;;
      W)
        build_option_proc_upper_w ;;
      f)
        export FASTER_BUILD_FOR_PLUGINS="on" ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}
