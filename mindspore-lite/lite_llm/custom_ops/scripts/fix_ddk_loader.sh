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
# ============================================================================
# 兼容封装：DDK loader 修复逻辑已合并进 install_ddk.sh（--fix-loader 模式），
# 本脚本仅转发，保留旧入口以便既有调用不受影响。
#
# 用法: scripts/fix_ddk_loader.sh [DDK根目录]    # 默认取 $DDK_PATH
# 等价于: install_ddk.sh --fix-loader=<DDK根目录>

DDK="${1:-$DDK_PATH}"
if [ -z "$DDK" ]; then
    echo "错误: 未指定 DDK 根目录（第一个参数或 DDK_PATH 环境变量）"
    echo "用法: fix_ddk_loader.sh [DDK根目录]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SCRIPT="$SCRIPT_DIR/../.claude/skills/setting-up-kirin-ai-ddk/install_ddk.sh"

exec bash "$INSTALL_SCRIPT" --fix-loader="$DDK"
