# MindSpore Lite - Claude 使用说明

本文件用于约束与指导在本仓库中进行代码修改、构建、测试与交付的行为规范。请在开始任何改动前阅读并遵守。

## 项目概览

MindSpore Lite推理框架主要包含两种推理形式，分别为：

- 云侧推理：用于服务侧设备的推理，主要适用于昇腾卡（Atlas 800I A2/A3、Atlas 300I Duo、Atlas 300I Pro等硬件），以及X86/Arm架构的CPU硬件，有关云侧推理的详细说明，请参考[云侧推理](.claude/lite-cloud-side.md)章节。

- 端侧推理：主要用于端/边设备的推理，主要适用于麒麟NPU、Arm架构CPU等终端硬件，有关端侧推理的详细说明，请参考[端侧推理](.claude/lite-device-side.md)章节。  

> 注意：MindSpore Lite的两种推理场景，用户在使用的时候，必须明确指定是云侧推理还是端侧推理，否则会导致推理失败。

此外，MindSpore Lite提供了完全独立的加速组件[Lite Boost](mindspore-lite/lite_boost/README.md)，用于提升基于PyTorch接口的云侧推理的推理性能，有关Lite Boost的详细说明，请参考[Lite Boost](mindspore-lite/lite_boost/CLAUDE.md)章节。

## 提交规范

写 commit 时遵守以下约定（commit-msg hook 强制执行）：

- **不要** `Co-Authored-By:` 行（任何 AI 工具/助手的署名都不要）
- **不要** 在 commit message 中贴源码片段或 diff 内容
- 描述「为什么改」而非「改了什么」——diff 已经说明了 what

Hook 配置：`git config core.hooksPath scripts/pre_commit/githooks`（同时启用 pre-push 和 commit-msg）。
