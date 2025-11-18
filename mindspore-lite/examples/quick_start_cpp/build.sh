#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
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

BASEPATH=$(cd "$(dirname $0)" || exit; pwd)
get_version() {
    VERSION_STR=$(cat ${BASEPATH}/../../../version.txt)
}
# 检查给定 URL 是否存在（返回 HTTP 状态码在 200~299 范围内）
# 参数: $1 - 待检查的 URL
# 返回值: 如果 URL 存在则返回 true，否则返回 false
is_url_exist() {
  [ "$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 5 "$1")" -ge 200 ] && [ "$(curl -o /dev/null -s -w "%{http_code}" --connect-timeout 5 "$1")" -lt 300 ]
}

get_version
MODEL_DOWNLOAD_URL="https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms"
MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/x86_64/${MINDSPORE_FILE}"

# 如果当前版本的 MindSpore Lite 包不存在，则使用最新的已发布版本（例如 2.7.1）
# 这是为了防止因为版本未发布而导致构建失败
if ! is_url_exist "$MINDSPORE_LITE_DOWNLOAD_URL"; then
    VERSION_STR=2.7.1 #latest release version
    MINDSPORE_FILE_NAME="mindspore-lite-${VERSION_STR}-linux-x64"
    MINDSPORE_FILE="${MINDSPORE_FILE_NAME}.tar.gz"
    MINDSPORE_LITE_DOWNLOAD_URL="https://ms-release.obs.cn-north-4.myhuaweicloud.com/${VERSION_STR}/MindSpore/lite/release/linux/x86_64/${MINDSPORE_FILE}"
    echo "Current version not released, use $VERSION_STR instead."
fi

mkdir -p build
mkdir -p lib
mkdir -p model
if [ ! -e ${BASEPATH}/model/mobilenetv2.ms ]; then
    echo "Downloading ${BASEPATH}/model/mobilenetv2.ms from ${MODEL_DOWNLOAD_URL}"
    wget -c -O ${BASEPATH}/model/mobilenetv2.ms --no-check-certificate ${MODEL_DOWNLOAD_URL}
fi
if [ ! -e ${BASEPATH}/build/${MINDSPORE_FILE} ]; then
    echo "Downloading ${BASEPATH}/build/${MINDSPORE_FILE} from ${MINDSPORE_LITE_DOWNLOAD_URL}"
    wget -c -O ${BASEPATH}/build/${MINDSPORE_FILE} --no-check-certificate ${MINDSPORE_LITE_DOWNLOAD_URL}
fi
tar xzvf ${BASEPATH}/build/${MINDSPORE_FILE} -C ${BASEPATH}/build/
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/runtime/lib/libmindspore-lite.so ${BASEPATH}/lib
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/runtime/third_party/glog/libmindspore_glog.so* ${BASEPATH}/lib/libmindspore_glog.so
cp -r ${BASEPATH}/build/${MINDSPORE_FILE_NAME}/runtime/include ${BASEPATH}/
cd ${BASEPATH}/build || exit
cmake ${BASEPATH}
make
