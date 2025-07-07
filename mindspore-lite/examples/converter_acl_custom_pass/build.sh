#!/bin/bash
# Copyright 2024 Huawei Technologies Co., Ltd
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

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
echo $BASEPATH
mkdir -p build
mkdir -p model

# ${LITE_HOME} is the absolute path of mindspore-lite-version-aarch64.
cp -r ${LITE_HOME}/tools/converter/lib ${BASEPATH}/
cp -r ${LITE_HOME}/tools/converter/include ${BASEPATH}/

cd ${BASEPATH}/build || exit
cmake ${BASEPATH}
make
