# MindSpore Lite 云侧推理 - Claude 使用说明

本文件用于约束与指导在本仓库中云侧推理进行代码修改、构建、测试与交付的行为规范。请在开始任何改动前阅读并遵守。(只有云侧推理，不涉及端侧推理)

## 项目概览

MindSpore Lite 是以 C++ 为主的推理框架项目，包含 Python/C++/Java 绑定与多后端支持，主要目录包括：
  - build.sh 构建脚本
  - mindspore-lite/tools/** 工具目录
  - mindspore-lite/src/extendrt/** 云侧推理目录
  - mindspore-lite/python/** Python 接口目录
  - include/api/** C++ 接口文件目录

## 环境要求

- CMake 3.22.3
- GCC 7.3+（C++17）

如果缺少必要工具或版本不满足，应停止并提示用户完成环境准备，不要自行安装或替代。

## 构建（必须使用 build.sh）

仓库统一入口：

- Linux：`build.sh`

不要直接调用 CMake（除非用户明确要求）。

常用命令示例：

```bash
# x86_64 云侧推理（包含昇腾后端，和cpu后端）
export MSLITE_ENABLE_CLOUD_INFERENCE=on
export MSLITE_ENABLE_ACL=on
bash build.sh -I x86_64 -j8

# Arm 云侧推理（包含昇腾后端，和cpu后端）
export MSLITE_ENABLE_CLOUD_INFERENCE=on
export MSLITE_ENABLE_ACL=on
bash build.sh -I arm64 -j8
```
> 编译完成后在`output/`目录会生成对应的whl包和tar包，分别是python接口的安装包和二进制so库。

## 格式化与代码风格

### clang-format

仅使用仓库提供的 clang-format 配置与脚本：

```bash
# 自动格式化（会修改文件）
bash scripts/format_source_code.sh -l
```

### 代码风格与工程约束

- 日志必须使用 `MS_LOG` 系列宏；不要使用 `printf` / `std::cout` / `std::cerr`
- 返回码/错误码使用 `mindspore::StatusCode` 项目既有定义
- 堆内存使用智能指针，避免裸 `new`/`delete`
- 外部输入（模型文件、shape、用户参数）必须在边界处校验
- 内部代码可信任内部不变式，避免重复校验
- 命名习惯：变量/函数 snake_case；类/公开 API PascalCase
- 文件命名：`operator_datatype.cc`（例如 `conv2d_fp32.cc`）

## 变更原则

- 优先在已有文件中修改，避免无必要新增文件
- 不要引入仓库未使用的新依赖（新增第三方库需先确认仓库已有）
- 不要输出/提交密钥、token、私有路径等敏感信息
- 不要在未被用户明确要求时执行 commit

## 常用技能（Skill）

当任务匹配时优先使用对应 Skill：
- [onnx-model-conversion-and-deployment](skills/lite-cloud-side/onnx-model-conversion-and-deployment/SKILL.md)：ONNX模型转换与推理部署
- [open-source-model-migration](skills/lite-cloud-side/open-source-model-migration/SKILL.md)：开源模型迁移Ascend硬件推理部署
- [performance-optimization](skills/lite-cloud-side/performance-optimization/SKILL.md)：通过自定义算子优化推理性能
