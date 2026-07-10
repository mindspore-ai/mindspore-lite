---
name: performance-optimization
description: MindSpore Lite（Ascend）模型性能优化总攻略。做基线/profiling、融合算子改写、推理免拷贝、PTQ int8 量化、精度对齐与归档时调用。本文为总览与索引，细化策略见 references/。
---

# Performance Optimization（MindSpore Lite / Ascend）

## 目标

为任意模型提供可复用的性能优化落地路径：在不破坏精度的前提下，把端到端推理（`Model execute time`、Host↔Device 拷贝、显存占用）从基线优化到目标值，并保证性能可复现。

> 本文档是**总览与索引**：给出性能优化的总体策略与「答题方向」（先量、再定位、再改一处、再验证），各优化手段的细节落到 `references/` 下独立文档，便于后续持续补充新手段。

## 何时调用

- 用户要求做模型性能优化（降时延 / 提吞吐 / 降显存 / 降拷贝）
- 需要做基线、benchmark、msprof profiling、精度对齐与归档
- 通过修改 ONNX 模型实现自定义融合算子适配（让图中出现可被 Custom Parser 接入的 `Custom` 节点）
- 优化推理链路的 Host↔Device 拷贝（多模型流水线 / 自回归 decode / KV cache）
- 对模型做 PTQ int8 量化导出，走低比特推理
- 转换失败或性能回归，需要定位与修复

## 通用原则（所有优化手段都必须遵守）

1. **先量后改**：没有基线与 profiling，不要开始优化——否则无法判断“改了是否真的变快”。
2. **一次一改**：每次只改一个点；只要“精度达标 + 性能收益”，就把该版本定为新基线，再尝试下一项。
3. **口径统一**：固定输入 shape、固定 `$Benchmark` 参数（warmUp/loopCount）、固定精度对齐方式；不同口径之间不比较。
4. **不破坏图语义**：任何融合/替换/量化都要做功能一致性验证（max abs / cosine / top-k 排序）。
5. **可用性优先于激进优化**：先“能转 + 能跑 + 精度达标”，再追求更深优化；激进手段叠加在不稳定基础上只会更难定位。
6. **产物可复现**：每次尝试保留导出/转换/性能/精度四份日志，并记录对应模型与 profiling 路径。
7. **融合不必然更快，必须实测**：融合算子（SwiGlu/GeGlu/Add+Norm/RmsNorm 等）看似省算子数，但常因引入额外的 `cat`/`slice`/`TransData` 反而变慢。**遇到融合激活/归一化类算子，默认都应做一次 fused vs unfused 的端到端 benchmark 对比**，不能凭直觉跳过。详见 [references/other_opt_methods.md](references/other_opt_methods.md) §4.6。

## 性能优化策略总览

MindSpore Lite（Ascend）的性能优化可以从两个角度切入，二者互补：

- **导出 / 图改写侧**：在 ONNX 导出或模型定义阶段改变图的形态（融合算子改写、量化算子注入、冗余算子合并）。收益在“图结构”层面。
- **运行 / Profiling 侧**：在不改图的前提下，通过基线 + msprof 定位热点算子，再调整精度模式、融合开关、算子选型，或优化推理调度（免拷贝）。收益在“后端实现 / 调度”层面。

| # | 优化手段 | 切入角度 | 适用场景 | 预期收益 | 细节文档 |
|---|---------|---------|---------|---------|---------|
| 1 | **融合算子 / Custom 改写** | 导出 / 图改写 | 后端（CANN）有更优融合算子（Attention/Norm/Add+Norm/激活/Rope 等），需让图出现 `Custom` 节点 | 减少算子数、走后端融合 kernel | [references/custom_operator_fusion.md](references/custom_operator_fusion.md) |
| 2 | **推理免拷贝** | 运行 / 调度 | 模型输出会被下一阶段继续使用：多模型流水线、自回归 decode、KV cache、多分支复用 | 降低 Host↔Device 拷贝与重复分配 | [references/zero_copy_inference.md](references/zero_copy_inference.md) |
| 3 | **PTQ Int8 量化导出** | 导出 / 图改写 | 访存密集型大线性层（LLM decode 等）需低比特推理减带宽/显存 | 减半带宽、降显存 | [references/torch_ptq_int8.md](references/torch_ptq_int8.md) |
| 4 | **Profiling 驱动的运行时调优** | 运行 / Profiling | 已有可跑模型，需定位热点算子并尝试精度模式 / 融合开关 / 算子替换（PFA、QKV 合并、MatMul 选型、混合精度、解融合等） | 针对热点算子的定向加速 | [references/other_opt_methods.md](references/other_opt_methods.md) |

> ⚠️ **动手前必读（强制）**：上表就是「任务 → 细化文档」的匹配表。`references/` **不会随 skill 自动加载**——确定要做哪类优化后，**必须先用 `Read` 工具读对应的 `references/` 文档，再动手改代码/模型**。Custom 属性规范、PTQ 校准流程、免拷贝 I/O 模板、profiling/精度对齐命令等关键细节**只存在于细化文档中**；只读本文件就动手，必然遗漏这些步骤、导致转换失败或返工。
>
> 新增优化手段时，在 `references/` 下新建一篇细化文档，并在上表追加一行即可，无需改动既有内容。

## 通用优化工作流（答题方向）

无论用哪种优化手段，都按下面这个闭环推进。具体的命令模板（`$Benchmark` / `msprof` / 精度对齐脚本）见 [references/other_opt_methods.md](references/other_opt_methods.md)。

```
固定口径 ─► 跑基线 ─► profiling 定位热点 ─► 提假设、改一处 ◄────────┐
                                              │                   │
                                              ▼                   │ 下一项
                          重新 benchmark（execute/H2D/D2H）─► 精度对齐 │
                                                  │               │
                                      ┌─ 达标+收益 → 归档(新基线) ─┘
                                      └─ 否 → 回退改动 ───────────┘
```

> 「固定口径」是一次性设置；每轮回路回到「提假设、改一处」，而非重新设口径。

1. **固定口径**：固定输入 shape、`$Benchmark` 参数、精度对齐基线（推荐 non-fuse ONNXRuntime CPU）。
2. **跑基线**：开 `GLOG_v=1`，从日志抓 `Model execute in ... us` 与 `AclrtMemcpy [H2D]/[D2H] ... us`。
3. **profiling 定位**：用 `msprof` 包裹 `$Benchmark`，导出 `op_statistic_*.csv` / `op_summary_*.csv`，找出热点算子与副作用（Transpose/TransData/Cast 是否暴涨）。
4. **提假设、改一处**：对照「策略总览」选定手段后，**必须先 `Read` 对应的 `references/` 细化文档**（严格按其中的属性规范 / 校准流程 / I/O 模板 / 命令执行），**每次只改一个点**。
5. **重新量测**：同口径 benchmark，记录 execute / H2D / D2H。
6. **精度对齐**：与基线比 max_abs_diff / mean_abs_diff / rmse / cosine，确认 top-k 排序一致。
7. **归档尝试表**：记录阶段 / 变更点 / execute(us) / H2D / D2H / 相对基线 / 精度 / 产物路径；收益且精度达标则定为新基线。

> profiling 只用于解释“为什么”，不能替代 benchmark 的口径。

## 交付清单（复用模板）

1. 改写/优化脚本（含算子映射、属性抽取、量化参数、免拷贝 I/O 等逻辑）
2. 导出后的 ONNX（及 external data）
3. 可复现转换命令与日志
4. 最终部署模型产物（如 `.mindir`）
5. 功能/性能对比报告（改写/优化前 vs 后）：execute / H2D / D2H / 精度指标

## 执行检查清单（每次必走）

1. 优化前后输出误差在阈值内（max abs / cosine / top-k）
2. 若涉及 Custom 节点：属性完整、`input_index` 合法（详见 [custom_operator_fusion.md](references/custom_operator_fusion.md)）
3. 转换成功且产物可加载
4. 推理功能可用，关键场景无回归
5. 性能收益明确且可重复（同口径 benchmark）

## 参考文档索引

细化策略文档（位于 `references/`）：

- [custom_operator_fusion.md](references/custom_operator_fusion.md) — 融合算子选型、Custom 改写标准流程、属性规范、控制流处理、转换排障
- [zero_copy_inference.md](references/zero_copy_inference.md) — 推理免拷贝核心做法与 Qwen3-VL 三阶段流水线示例
- [torch_ptq_int8.md](references/torch_ptq_int8.md) — Torch PTQ int8 量化导出六步法
- [other_opt_methods.md](references/other_opt_methods.md) — 基线 / msprof profiling / 精度对齐命令模板 + 运行时优化手段清单（PFA、QKV 合并、MatMul 选型、混合精度、解融合等）+ 转换失败定位
