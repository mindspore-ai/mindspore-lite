#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
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

# Single source of truth for tool versions used by install_tools.sh and
# githooks/pre-push. Sourced, not executed. To bump a tool, change only this
# file; install scripts, the pre-push gate, and the README version table all
# derive from here.
#
# Every var here is consumed by scripts that source this file; shellcheck
# analyzes each file in isolation and would flag every var as unused, so the
# whole file disables SC2034.

# shellcheck disable=SC2034
readonly CMAKELINT_VERSION="1.4.1"
readonly CODESPELL_VERSION="2.0.0"
readonly CPPLINT_VERSION="2.0.2"
readonly LIZARD_VERSION="1.17.19"
readonly PYLINT_VERSION="3.3.7"
readonly CLANG_FORMAT_VERSION="18.1.8"
# ARM64 Mac has a known incompatibility with 18.1.8; pin to 18.1.4 there.
readonly CLANG_FORMAT_VERSION_ARM64_MAC="18.1.4"
readonly SHELLCHECK_VERSION="0.7.1"
readonly CHEF_UTILS_VERSION="16.6.14"
