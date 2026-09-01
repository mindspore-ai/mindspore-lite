---
name: precision_troubleshooting
description: MindSpore Lite（Ascend）云侧推理精度保障总攻略。覆盖 Torch→ONNX→MindIR 三阶段精度对齐、fp32/混合精度配置、Custom 融合算子对接排查、以及跨 CANN 版本精度差异的算子级定位。本文为总览与索引，细化流程见 references/。
---

# Precision（MindSpore Lite / Ascend 云侧推理）

## 目标

确保模型从 PyTorch 导出 ONNX、再经 MindSpore Lite 转换为 MindIR 后，在 Ascend 上推理的精度与原始 Torch CPU 一致；并在出现精度差异时，能定位到根因算子。

> **本 skill 的能力边界**：定位到具体算子导致的精度问题；修复则视情况——
> - 通过混合精度配置（fp32 黑名单）规避的精度问题，可在本 skill 流程内直接解决
> - 单算子经隔离验证能复现精度问题，且确认非模型导出/对接问题，则需交由 **MindSpore Lite 开发者**进一步排查单算子实现
> - 属于 CANN 算子实现的精度问题，需由 **CANN 侧相关责任人**修复
>
> 即：本 skill 负责"定位到根因算子"，修复主体可能是配置（自助）、MindSpore Lite 代码、或 CANN 算子（需对应责任人）。

> 本文档是**总览与索引**：给出精度保障的总体策略与两个切入场景（三阶段对齐 / 跨版本定位），各场景的完整流程落到 `references/` 下独立文档，便于持续补充。

## 何时调用

- 需要验证 Torch CPU、ONNX、MindIR 三者精度是否对齐
- 发现 ONNX 推理结果与 Torch CPU 不一致，需要定位精度问题
- 发现 MindIR 推理结果与 ONNX 不一致，需要定位精度问题
- 需要通过 fp32/混合精度配置解决 MindIR 精度问题
- 怀疑 Custom 融合算子对接引入精度问题，需要隔离验证
- 在不同 CANN 版本下、同一 MindSpore Lite 版本用 `convert` 转换出的不同 MindIR 模型，推理输出结果存在精度差异
- 需要从算子级别定位精度差异根因，而非仅判断整体输出是否一致

## 两个切入场景

精度问题按"差异出现的环节"分两类，各自走不同的定位流程：

| 场景 | 典型现象 | 核心方法 | 细节文档 |
|------|---------|---------|---------|
| **A. 三阶段对齐** | Torch→ONNX→MindIR 任意相邻两者输出对不上 | 逐层/逐算子对比、fp32/混合精度调优、ONNX 添加中间输出定位 | [references/three_stage_alignment.md](references/three_stage_alignment.md) |
| **B. 跨 CANN 版本定位** | 同一 MindSpore Lite 版本下，不同 CANN 版本用 `convert` 转换出的不同 MindIR 推理输出不一致 | Profiling 对比定位可疑算子、CANN Dump 对比算子输入输出 | [references/cann_dump_locating.md](references/cann_dump_locating.md) |

> ⚠️ **动手前必读（强制）**：上表是「场景 → 细化文档」的匹配表。`references/` **不会随 skill 自动加载**——确定属于哪个场景后，**必须先用 `Read` 工具读对应的 `references/` 文档**，再动手。各场景的命令模板、配置文件示例、判断阈值、归因逻辑只存在于细化文档中；只读本文件就动手，必然遗漏关键步骤。

## 通用原则（两个场景都必须遵守）

1. **统一测试输入**：构造一份固定的 numpy 输入数据（`test_input.npz`），所有对比阶段共用，避免输入差异污染精度判断。
2. **先确认基线再归因**：归因到某类问题（如 Custom 算子对接）前，必须先确认更基础的版本是 OK 的（如 fused MindIR 有问题，需先确认 non-fuse MindIR 精度 OK，才能归因到 Custom）。
3. **精度判断指标**：默认阈值统一为 max_abs < 1e-3 且 cosine > 0.999。
   > 该阈值适用于多数场景；但图片生成、视频生成、LLM 等场景下，数值差异不一定反映最终效果差异，**能保证端到端效果符合预期即可**。
   > **不确定时交用户判定**：若数值指标略超阈值但无法判断是否影响实际效果，应将 MindIR 与 Torch CPU 的端到端结果（生成的图片/视频/文本输出）提供给用户，由用户判断是否可接受，而非仅凭数值阈值一票否决。
4. **ONNX 基线用 non-fuse 版本**：带 Custom 融合算子的 ONNX 无法跑 ONNX Runtime，精度基线始终用未适配融合算子的 non-fuse ONNX。
5. **不凭记忆写格式转换代码**：涉及 CANN dump 的 FRACTAL_NZ 等 fractal 排布与 ND 格式转换时，优先用 `msaccucmp.py` 工具自带功能，勿手写 transpose/reshape。
6. **产物可复现**：保留每次尝试的导出/转换/推理/精度对比日志，记录对应模型路径，便于回溯。

## 场景 A：三阶段对齐（Torch → ONNX → MindIR）

模型迁移到 Ascend 推理的精度保障主流程，按三个阶段推进：

```
阶段1: Torch CPU vs ONNX    ──通过──→ 阶段2
                              └─不通过→ 逐层定位 Torch vs ONNX 差异

阶段2: ONNX vs MindIR       ──通过──→ 三者对齐，完成
        ├─ 前置: 确认 ONNX 基线为 non-fuse 版本
        ├─ non-fuse MindIR 不通过 → Step 2.4 常规定位（fp32/混合精度/中间输出）
        └─ fused MindIR 不通过（前提: non-fuse 已 OK）→ 归因 Custom 对接
```

阶段 2 的修复路径是渐进式的：默认配置 → force_fp32 → 混合精度（fp32 黑名单）→ ONNX 添加中间输出逐算子定位。大模型转换耗时长时可降层数加速定位。

完整流程见 [references/three_stage_alignment.md](references/three_stage_alignment.md)。

## 场景 B：跨 CANN 版本精度定位

同一 MindSpore Lite 版本下，不同 CANN 版本用 `convert` 转换出的不同 MindIR 推理结果不一致时的算子级定位流程：

```
Step 1: Profiling 对比（OK vs NOT_OK）→ 定位可疑算子（Block Num / Mix Block Num / Format 差异）
Step 2: 配置 CANN Dump（dump.json + dump_config.ini）
Step 3: 触发 Dump 推理 + msaccucmp.py 转 npy
Step 4: 对比同一算子的输入输出 → 输入一致输出不一致 = 根因算子
```

完整流程见 [references/cann_dump_locating.md](references/cann_dump_locating.md)。

## 与其他 skill 的关系

- **performance-optimization**：精度对齐是性能优化的前置约束（任何融合/替换/量化改写都要做功能一致性验证）。本 skill 的 profiling 采集方法直接引用 [open-source-model-migration/references/get_profiling_data.md](../open-source-model-migration/references/get_profiling_data.md)。
- **open-source-model-migration**：模型导出阶段的精度验证（Torch vs ONNX）属于本 skill 场景 A 的阶段 1。

## 执行检查清单（两个场景通用）

1. 统一测试输入已保存（`test_input.npz`），所有对比阶段共用
2. 已明确属于场景 A（三阶段对齐）还是场景 B（跨版本定位），并 Read 了对应 references 文档
3. 精度对比阈值采用默认值（max_abs < 1e-3 / cosine > 0.999）
4. 数值略超阈值且场景为图/视频生成、LLM 时，已向用户提供 MindIR vs Torch CPU 的端到端结果供判定，而非一票否决
5. 归因某类问题前，已确认更基础的版本是 OK 的
6. 涉及 Custom 算子时，已用 non-fuse ONNX/MindIR 作为基线
7. 涉及 CANN dump 时，已用 msaccucmp 工具处理格式转换，未手写 reshape
8. 最终精度满足阈值，或已定位到根因算子并明确修复责任归属（混合精度配置自助 / MindSpore Lite 开发者 / CANN 侧责任人）

## 参考文档索引

细化流程文档（位于 `references/`）：

- [three_stage_alignment.md](references/three_stage_alignment.md) — Torch→ONNX→MindIR 三阶段精度对齐完整流程（统一输入构造、逐层定位、fp32/混合精度配置、Custom 对接排查、ONNX 添加中间输出定位、大模型降层数加速）
- [cann_dump_locating.md](references/cann_dump_locating.md) — 跨 CANN 版本精度差异的算子级定位（Profiling 对比、CANN Dump 配置、msaccucmp 转换、逐算子输入输出对比判断根因）
