#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

function android_release_package()
{
    arch=$1
    device=$2
    pkg_name="mindspore-lite-${version}-android-${arch}"

    [ -n "${pkg_name}" ] && rm -rf ${pkg_name} || exit 1
    tar -xzf ${input_path}/android_${arch}/${device}/${pkg_name}.tar.gz || exit 1
    # Copy java runtime to Android package
    cp ${input_path}/aar/mindspore-lite-*.aar ${pkg_name} || exit 1

    mkdir -p ${output_path}/release/android/${device}/ || exit 1
    tar -czf ${output_path}/release/android/${device}/${pkg_name}.tar.gz ${pkg_name} || exit 1
    [ -n "${pkg_name}" ] && rm -rf ${pkg_name} || exit 1
    cd ${output_path}/release/android/${device}/ || exit 1
    sha256sum ${pkg_name}.tar.gz > ${pkg_name}.tar.gz.sha256 || exit 1
}

function linux_release_package()
{
    mkdir -p ${output_path}/release/linux/x86_64/ || exit 1
    mkdir -p ${output_path}/release/linux/aarch64/ || exit 1
    mkdir -p ${output_path}/release/linux/x86_64/cloud_fusion/ || exit 1
    mkdir -p ${output_path}/release/linux/aarch64/cloud_fusion/ || exit 1

    cp ${input_path}/centos_x86/avx/mindspore*.tar.gz* ${output_path}/release/linux/x86_64/ || exit 1
    cp ${input_path}/linux_aarch64/mindspore*.tar.gz* ${output_path}/release/linux/aarch64/ || exit 1
    cp -r ${input_path}/centos_x86/cloud_fusion/* ${output_path}/release/linux/x86_64/cloud_fusion/ || exit 1
    cp -r ${input_path}/linux_aarch64/cloud_fusion/* ${output_path}/release/linux/aarch64/cloud_fusion/ || exit 1
}

function windows_release_package()
{
    mkdir -p ${output_path}/release/windows/ || exit 1
    cp ${input_path}/windows_x64/avx/*.zip* ${output_path}/release/windows/ || exit 1
}

echo "============================== begin =============================="
echo "Usage: bash lite_release_package.sh input_path output_path"

input_path=$1
output_path=$2
version=$(ls ${input_path}/android_aarch64/npu/mindspore-lite-*-*.tar.gz | awk -F'/' '{print $NF}' | cut -d"-" -f3)

android_release_package aarch64 npu
# android_release_package aarch64 gpu

linux_release_package
windows_release_package

echo "Create release package success!"
echo "=============================== end ==============================="
