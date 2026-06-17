#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
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

###############################################################################
# lite_boost 组件测试入口脚本
#
# 功能说明:
#   本脚本用于 PR 门禁系统自动执行 lite_boost 组件的 Python 测试用例。
#   它将完成以下步骤:
#     1. 解析命令行参数（whl 包路径、用例级别、模型路径）
#     2. 查找并安装指定路径下的 lite_boost whl 包
#     3. 根据用例级别（level0 / level1），通过 pytest 执行对应的测试用例
#
# 使用示例:
#   # 仅运行 level0 级别用例（默认）
#   sh run.sh -r /path/to/lite_boost-1.0.0-py3-none-any.whl
#
#   # 运行 level0 + level1 级别用例
#   sh run.sh -r /path/to/lite_boost-1.0.0-py3-none-any.whl -l level1
#
#   # 指定模型路径（该参数暂不使用，保留处理）
#   sh run.sh -r /path/to/lite_boost-1.0.0-py3-none-any.whl -l level0 -m /path/models/hiai
#
# 参数说明:
#   -r : (必填) lite_boost 编译产出的 whl 包路径，支持传入目录或 whl 文件路径
#   -l : (可选) 测试用例级别，默认 level0。取值说明:
#        level0 - 仅执行 level0 级别用例（基础功能验证，PR 门禁触发）
#        level1 - 执行 level0 + level1 全部用例（全量验证，夜间门禁触发）
#   -m : (可选) 模型文件所在路径，当前版本 lite_boost 暂不依赖该参数，保留以备后续扩展
###############################################################################

#-----------------------------------------------------------------------------
# 第一步：解析命令行参数
# 使用 bash 内置的 getopts 解析 -r / -l / -m 三个选项
#-----------------------------------------------------------------------------

# 设置默认值
# level 默认为 level0，即仅执行基础用例
level="level0"
# models_path 默认为空字符串，lite_boost 当前不依赖此参数
models_path=""

# 如果没有任何参数传入，打印使用说明并退出
if [ $# -eq 0 ]; then
    echo "错误: 未提供任何参数。"
    echo "用法: sh run.sh -r <whl包路径> [-l <level0|level1>] [-m <模型路径>]"
    exit 1
fi

# 解析命令行选项
# getopts 格式说明: "r:l:m:" 中冒号表示该选项需要一个值
# 其中 r、l、m 都是带值选项，需要用户传入对应的参数值
while getopts "r:l:m:" opt; do
    case ${opt} in
        r)
            # whl 包路径: 可以是目录路径（脚本自动查找目录下的 whl 文件）
            #             也可以是具体的 whl 文件路径
            release_path="${OPTARG}"
            echo "release_path (whl包路径) 已设置为: ${release_path}"
            ;;
        l)
            # 用例级别: level0 或 level1
            # level0 - 仅运行 level0 标记的用例（PR 门禁）
            # level1 - 运行 level0+level1 标记的用例（包含 level0 子集）
            level="${OPTARG}"
            echo "level (用例级别) 已设置为: ${level}"
            ;;
        m)
            # 模型路径: 当前 lite_boost 暂未使用该参数，保留处理以待后续扩展
            models_path="${OPTARG}"
            echo "models_path (模型路径) 已设置为: ${models_path} (当前版本暂未使用)"
            ;;
        ?)
            # 未知参数或缺少参数值时的错误处理
            echo "错误: 未知参数或缺少参数值。"
            echo "支持的参数: -r <whl包路径> -l <level0|level1> -m <模型路径>"
            exit 1
            ;;
    esac
done

#-----------------------------------------------------------------------------
# 第二步：校验必填参数
#-----------------------------------------------------------------------------

# -r release_path 为必填参数，如果未提供则报错退出
if [ -z "${release_path}" ]; then
    echo "错误: 缺少必填参数 -r <whl包路径>。"
    echo "请通过 -r 指定 lite_boost whl 包的路径。"
    exit 1
fi

# 校验 -l level 参数的合法性，仅允许 level0 或 level1
if [ "${level}" != "level0" ] && [ "${level}" != "level1" ]; then
    echo "错误: 无效的用例级别 '${level}'，仅支持 level0 或 level1。"
    exit 1
fi

#-----------------------------------------------------------------------------
# 第三步：定位 whl 文件
# 如果 release_path 是一个目录，则在该目录下搜索 lite_boost-*.whl 文件
# 如果 release_path 直接指向一个 whl 文件，则直接使用
#-----------------------------------------------------------------------------

# 判断 release_path 是目录还是文件
if [ -d "${release_path}" ]; then
    # release_path 是目录: 在该目录下查找 lite_boost-*.whl 文件
    # ls 返回匹配到的文件列表，取第一个
    whl_file=$(ls "${release_path}"/lite_boost-*.whl 2>/dev/null | head -n 1)
    if [ -z "${whl_file}" ]; then
        echo "错误: 在目录 '${release_path}' 下未找到 lite_boost-*.whl 文件。"
        echo "请确认 whl 包已正确编译产出。"
        exit 1
    fi
elif [ -f "${release_path}" ]; then
    # release_path 直接是一个文件: 校验文件名是否以 lite_boost 开头、.whl 结尾
    whl_file="${release_path}"
    # 提取文件名（去掉路径前缀）
    whl_basename=$(basename "${whl_file}")
    # 简单校验: 文件名需要包含 lite_boost 且以 .whl 结尾
    case "${whl_basename}" in
        lite_boost-*.whl)
            # 文件名格式正确
            ;;
        *)
            echo "警告: '${whl_basename}' 不符合预期的 whl 命名格式 (lite_boost-*.whl)，将继续尝试安装。"
            ;;
    esac
else
    # release_path 既不是有效的目录也不是有效的文件
    echo "错误: '${release_path}' 不是有效的目录或文件路径。"
    echo "请通过 -r 指定正确的 lite_boost whl 包路径。"
    exit 1
fi

echo "找到 whl 文件: ${whl_file}"

#-----------------------------------------------------------------------------
# 第四步：安装 lite_boost whl 包
# 先卸载已有的 lite_boost 包，再安装新的 whl 包
#-----------------------------------------------------------------------------

echo "=================================================================="
echo "开始安装 lite_boost whl 包..."

# 先卸载已安装的 lite_boost（如果存在），避免版本冲突
# || true 确保即使卸载失败（如未安装）也不会中断脚本
# 使用 "python3 -m pip" 代替 "pip"，确保在当前 conda 环境的 Python 中执行，
# 避免调用到系统 Python 绑定的 pip
python3 -m pip uninstall lite_boost -y 2>/dev/null || true
echo "已清理旧的 lite_boost 包（如果存在）。"

# 安装新的 lite_boost whl 包
# || exit 1 确保安装失败时脚本立即退出并返回错误码
python3 -m pip install "${whl_file}" || exit 1
echo "lite_boost whl 包安装成功。"

#-----------------------------------------------------------------------------
# 第五步：确定测试根目录
# SCRIPT_DIR 为当前脚本所在的目录，即 lite_boost/test/
# pytest 将从此目录开始递归查找所有 Python 测试文件
#-----------------------------------------------------------------------------

# 获取脚本所在的绝对路径目录
# $(cd "$(dirname "$0")" && pwd) 是获取脚本目录的标准写法
#   1. dirname "$0" : 获取脚本的相对/绝对路径的目录部分
#   2. cd ... && pwd : 进入该目录并获取绝对路径
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "测试根目录: ${SCRIPT_DIR}"

#-----------------------------------------------------------------------------
# 第六步：根据用例级别构造 pytest 运行参数，并执行测试
# pytest 使用 -m 选项来按标记过滤用例:
#   level0 级别: 仅运行标记了 @pytest.mark.L0 的用例
#   level1 级别: 运行标记了 @pytest.mark.L0 或 @pytest.mark.L1 的用例（包含 level0 子集）
#-----------------------------------------------------------------------------

echo "=================================================================="
echo "开始执行 lite_boost 测试用例 (level=${level})..."
echo "------------------------------------------------------------------"

# 根据用例级别设置 pytest 的过滤表达式
#
# 默认硬件筛选: ascend_a2（Atlas 800I A2 / ascend910b）。与级别标记做"与"筛选，
# 即默认只跑该硬件上对应级别的用例（test_chunk_gated_delta_rule 标的是
# ascend_300iduo，默认不会被选中，需要时单独 `pytest -m ascend_300iduo` 触发）。
default_hardware="ascend_a2"

if [ "${level}" == "level0" ]; then
    # level0 级别: 仅执行标记为 L0 的用例（且为默认硬件 ascend_a2）
    pytest_mark_expr="L0 and ${default_hardware}"
    echo "当前运行模式: level0 - 仅执行基础功能验证用例 (硬件: ${default_hardware})。"
elif [ "${level}" == "level1" ]; then
    # level1 级别: 执行 L0 + L1 的全部用例（且为默认硬件 ascend_a2）
    # pytest -m 支持逻辑表达式，"(L0 or L1) and ascend_a2" 同时匹配级别与硬件
    pytest_mark_expr="(L0 or L1) and ${default_hardware}"
    echo "当前运行模式: level1 - 执行全部用例 (硬件: ${default_hardware})。"
fi

# 执行 pytest
# 注意: 使用 "python3 -m pytest" 而非直接调用 "pytest" 命令
# 这是因为 CI 环境中系统安装的 pytest 可能绑定到系统 Python（如 /usr/bin/python3），
# 而系统 Python 中没有安装 torch_npu、lite_boost 等必需依赖。
# "python3 -m pytest" 使用当前 PATH 中的 python3（即 conda 环境 Python），
# 确保 pytest 在正确的 Python 环境中运行，能够找到所有测试依赖。
#
# 参数说明:
#   -v    : verbose 模式，输出详细的用例执行信息
#   -ra   : 在测试总结中显示所有信息（passed/skipped/failed/error 等）
#   -s    : 不捕获 stdout/stderr，允许测试中的 print 输出直接显示在终端
#   -m    : 按 pytest marker 过滤，只运行符合级别要求的用例
#   --tb=short : 简化失败时的 traceback 输出，保持可读性
python3 -m pytest -v -ra -s -m "${pytest_mark_expr}" --tb=short "${SCRIPT_DIR}"

# 保存 pytest 的退出码
# 0 表示全部用例通过，非 0 表示有失败或错误
PYTEST_RET=$?

#-----------------------------------------------------------------------------
# 第七步：输出测试结果摘要
#-----------------------------------------------------------------------------

echo "=================================================================="
if [ ${PYTEST_RET} -eq 0 ]; then
    echo "lite_boost 测试全部通过 (level=${level})。"
else
    echo "lite_boost 测试存在失败用例，退出码: ${PYTEST_RET}。"
fi
echo "=================================================================="

# 将 pytest 的退出码作为脚本的退出码，以便 CI/CD 门禁系统判断测试是否通过
exit ${PYTEST_RET}
