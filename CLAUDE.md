# MindSpore Lite - Claude 使用说明

本文件用于约束与指导在本仓库中进行代码修改、构建、测试与交付的行为规范。请在开始任何改动前阅读并遵守。

## 项目概览

MindSpore Lite推理框架主要包含两种推理形式，分别为：

- 云侧推理：用于服务侧设备的推理，主要适用于昇腾卡（Atlas 800I A2/A3、Atlas 300I Duo、Atlas 300I Pro等硬件），以及X86/Arm架构的CPU硬件，有关云侧推理的详细说明，请参考[云侧推理](.claude/cloud.md)章节。

- 端侧推理：主要用于端/边设备的推理，主要适用于麒麟NPU、Arm架构CPU等终端硬件，有关端侧推理的详细说明，请参考[端侧推理](.claude/device.md)章节。  

> 注意：MindSpore Lite的两种推理场景，用户在使用的时候，必须明确指定是云侧推理还是端侧推理，否则会导致推理失败。

此外，MindSpore Lite提供了完全独立的加速组件[Lite Boost](mindspore-lite/lite_boost/README.md)，用于提升基于PyTorch接口的云侧推理的推理性能，有关Lite Boost的详细说明，请参考[Lite Boost](mindspore-lite/lite_boost/CLAUDE.md)章节。

## 提交规范

写 commit 时遵守以下约定（commit-msg hook 强制执行）：

- **不要** `Co-Authored-By:` 行（任何 AI 工具/助手的署名都不要）
- **不要** 在 commit message 中贴源码片段或 diff 内容
- 描述「为什么改」而非「改了什么」——diff 已经说明了 what

Hook 配置：`git config core.hooksPath scripts/pre_commit/githooks`（同时启用 pre-push 和 commit-msg）。

## 云侧推理 Skill 速查（每个会话自动可见）

> 云侧推理（Ascend / X86 / Arm 服务侧）任务，先按下表定位阶段 → `Read` 对应 SKILL.md（及其 `references/`）再动手。

| 阶段 | 做什么 | 对应 Skill |
|------|--------|-----------|
| ① 模型迁移/导出 | 开源模型按网络结构拆分导出 ONNX、ONNX Runtime 精度对齐、生成推理脚本与 README | [open-source-model-migration](.claude/skills/cloud/open-source-model-migration/SKILL.md) |
| ② 转换与部署推理 | ONNX→MindIR（固定 shape / 动态分档 / 纯动态 shape）+ Ascend 离线优化；MindIR 加载、Ascend 推理验证与部署注意事项 | [onnx-model-conversion-and-deployment](.claude/skills/cloud/onnx-model-conversion-and-deployment/SKILL.md) |
| ③ 性能优化 | 基线/profiling、融合算子改写、推理免拷贝、PTQ int8 量化、精度对齐与归档 | [performance-optimization](.claude/skills/cloud/performance-optimization/SKILL.md) |
| ④ 多卡张量并行 | Megatron 权重分片导出、Custom(AllReduce) 通信算子、多进程 HCCL 推理（1p/2p/4p）、300I Duo vs 800I A2 差异 | [tensor-parallel-deployment](.claude/skills/cloud/tensor-parallel-deployment/SKILL.md) |

通用（跨侧共享）：[clean-code-check](.claude/skills/common/clean-code-check/SKILL.md) — C++/Python/Shell/CMake 代码质量与 CI 门禁，改完或 review 时调用。

> 完整云侧规范（构建/格式化/变更原则/流程详述）见 [.claude/cloud.md](.claude/cloud.md)，需要时再 `Read`。新增/变更云侧 skill 时，同步更新本表与 cloud.md。

## 端侧推理 Skill 速查（每个会话自动可见）

> 端侧推理（端/边设备：麒麟 NPU、Arm CPU、Android/iOS、MCU 等）任务，先按下表定位阶段 → `Read` 对应 SKILL.md 再动手。

| 阶段 | 做什么 | 对应 Skill |
|------|--------|-----------|
| ① 构建 | `build.sh` 配置、CMake 选项、交叉编译（ARM/iOS/MCU）、打包发布 | [lite-build](.claude/skills/device/lite-build/SKILL.md) |
| ② 模型转换 | 模型转 `.ms`、parser 开发、优化 pass、量化 | [lite-converter](.claude/skills/device/lite-converter/SKILL.md) |
| ③ 算子/内核开发 | 算子与 NNACL 内核、delegate（NPU/CoreML/Ascend）、自定义算子注册 | [lite-kernel-dev](.claude/skills/device/lite-kernel-dev/SKILL.md) |
| ④ 端侧推理 | LiteRT 推理、Android/iOS 集成、Micro(MCU) 代码生成、设备端训练、C/C++/Java/Python API | [lite-device-side-infer](.claude/skills/device/lite-device-side-infer/SKILL.md) |
| ⑤ 调试与测试 | gtest、benchmark、profiling 时延/精度、delegate 回退、内存泄漏定位 | [lite-debug-test](.claude/skills/device/lite-debug-test/SKILL.md) |

> 完整端侧规范见 [.claude/device.md](.claude/device.md)，需要时再 `Read`。通用跨侧 skill（clean-code-check）见上方云侧章节。
