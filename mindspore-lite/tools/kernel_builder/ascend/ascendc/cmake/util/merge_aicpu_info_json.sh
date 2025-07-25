#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
# copy from https://gitee.com/ascend/samples/blob/master/operator/ascendc/tutorials/AddCustomSample/FrameworkLaunch/AddCustom/cmake/util/merge_aicpu_info_json.sh

project_path=$1
build_path=$2
vendor_name=customize
echo "$@"
if [[ ! -d "$project_path" ]]; then
    echo "[ERROR] No projcet path is provided"
    exit 1
fi

if [[ ! -d "$build_path" ]]; then
    echo "[ERROR] No build path is provided"
    exit 1
fi

if [[ ! -d "$ASCEND_OPP_PATH" ]]; then
    echo "[ERROR] No opp install path is provided"
    exit 1
fi
custom_exist_info_json=$ASCEND_OPP_PATH/vendors/$vendor_name/op_impl/cpu/config/cust_aicpu_kernel.json
custom_new_info_json=$build_path/makepkg/packages/vendors/$vendor_name/op_impl/cpu/config/cust_aicpu_kernel.json
temp_info_json=$build_path/makepkg/packages/vendors/$vendor_name/op_impl/cpu/config/temp_cust_aicpu_kernel.json

if [[ -f "$custom_exist_info_json" ]] && [[ -f "$custom_new_info_json" ]]; then
    cp -f $custom_exist_info_json $temp_info_json
    chmod +w $temp_info_json
    python3 ${project_path}/cmake/util/insert_op_info.py ${custom_new_info_json} ${temp_info_json}
    cp -f $temp_info_json $custom_new_info_json
    rm -f $temp_info_json
fi
