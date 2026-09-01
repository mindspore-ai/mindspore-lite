# MindSpore Lite 云侧推理 - Claude 使用说明

本文件用于约束与指导在本仓库中云侧推理进行代码修改、构建、测试与交付的行为规范。请在开始任何改动前阅读并遵守。(只有云侧推理，不涉及端侧推理)

## 项目概览

MindSpore Lite 是以 C++ 为主的推理框架项目，包含 Python/C++/Java 绑定与多后端支持，主要目录包括：
  - build.sh 构建脚本
  - mindspore-lite/tools/** 工具目录
  - mindspore-lite/src/extendrt/** 云侧推理目录
  - mindspore-lite/python/** Python 接口目录
  - include/api/** C++ 接口文件目录

## 端到端流程（Pipeline）

云侧推理部署是一条完整链路。先定位当前任务处于哪个阶段，再调用对应 Skill（动手前先 `Read` 该 Skill 的 SKILL.md 及其 references/）：

```
模型迁移/导出 ONNX ─► ONNX→MindIR 转换 + 部署推理 ─► 性能优化 ─► 精度问题定位
```

| 阶段 | 做什么 | 对应 Skill |
|------|--------|-----------|
| ① 模型迁移/导出 | 开源模型按网络结构拆分导出 ONNX、ONNX Runtime 精度对齐、生成推理脚本与 README | [open-source-model-migration](.claude/skills/cloud/open-source-model-migration/SKILL.md) |
| ② 转换与部署推理 | ONNX→MindIR（固定 shape / 动态分档 / 纯动态 shape）+ Ascend 离线优化；MindIR 加载、Ascend 推理验证与部署注意事项 | [onnx-model-conversion-and-deployment](.claude/skills/cloud/onnx-model-conversion-and-deployment/SKILL.md) |
| ③ 性能优化 | 基线/profiling、融合算子改写、推理免拷贝、PTQ int8 量化、精度对齐与归档 | [performance-optimization](.claude/skills/cloud/performance-optimization/SKILL.md) |
| ④ 精度问题定位 | Torch→ONNX→MindIR 三阶段精度对齐、fp32/混合精度配置、Custom 融合算子对接排查；跨 CANN 版本 Profiling 对比定位可疑算子、配置算子 Dump、对比 Dump 数据确定精度差异根因 | [precision_troubleshooting](.claude/skills/cloud/precision_troubleshooting/SKILL.md) |

> 典型顺序：①→② 跑通后，再用 ③ 做性能优化；若发现精度问题则进入 ④ 定位（每步都需精度+性能验证，见各 Skill）。

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

当任务匹配时优先使用对应 Skill（详见上方「端到端流程」的阶段对照）：

- [onnx-model-conversion-and-deployment](.claude/skills/cloud/onnx-model-conversion-and-deployment/SKILL.md)：ONNX→MindIR 转换（固定 shape / 动态分档 / 纯动态 shape）与推理部署
- [open-source-model-migration](.claude/skills/cloud/open-source-model-migration/SKILL.md)：开源模型迁移到 MindSpore Lite 部署管线（按结构拆分导出 ONNX、精度对齐、生成推理脚本与 README）
- [performance-optimization](.claude/skills/cloud/performance-optimization/SKILL.md)：模型性能优化总攻略——基线/profiling、融合算子改写、推理免拷贝、PTQ int8 量化、精度对齐与归档（细化策略见 references/）
- [precision_troubleshooting](.claude/skills/cloud/precision_troubleshooting/SKILL.md)：精度保障总攻略——Torch→ONNX→MindIR 三阶段对齐、fp32/混合精度配置、Custom 融合算子对接排查（场景 A）；跨 CANN 版本 Profiling 对比定位可疑算子、配置算子 Dump、对比 Dump 数据确定精度差异根因（场景 B）。细化流程见 references/
  - **精度指标**：默认阈值 max_abs < 1e-3 / cosine > 0.999；图/视频/LLM 场景以端到端效果为准，数值略超阈值时交用户判定 MindIR vs Torch CPU 结果
  - **能力边界**：负责定位到根因算子；修复视情况——混合精度配置自助、单算子实现交 MindSpore Lite 开发者、CANN 算子交 CANN 侧责任人
  - **大模型标杆**：Torch CPU 太慢时用 Torch NPU——800I A2 直接用，300I Duo 视模型大小决定
  - **800I A2 额外手段**：
    - 饱和模式（`export MS_ASCEND_CHECK_OVERFLOW_MODE=SATURATION_MODE`）：force_fp32 仍不通过、怀疑溢出时用
    - bf16 混合精度（`allow_mix_precision_bf16`）：fp32 通过、fp16 不通过、且模型含 bf16 计算时用（默认转换即 fp16）

## 通用技能（跨侧共享）

- [clean-code-check](.claude/skills/common/clean-code-check/SKILL.md)：C++/Python/Shell/CMake 代码质量检查与 CI 门禁工具。云侧 C++ 改动同样适用，改完或 review 时调用。
