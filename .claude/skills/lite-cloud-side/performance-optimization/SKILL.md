---
name: performance-optimization
description: 基于Ascend硬件后端的自定义算子适配对接的性能优化流程。用户涉及自定义融合算子选型、ONNX改写、CANN对齐、转换失败排障或性能回归验证时调用。
---

# Performance Optimization

## 目标

为任意模型提供可复用的性能优化落地路径：从“可融合算子识别”到“ONNX 自定义算子改写”、再到“转换与回归验证”，确保功能可用且性能可复现。

## 何时调用

- 用户要求做模型性能优化，尤其是融合算子优化
- 通过修改 ONNX 模型，实现自定义算子适配
- 优化模型结构，如合并算子、删除冗余算子等
- 需要让 ONNX 中出现 Custom 节点并可被 MindSpore Lite Custom Parser 接入

## 通用原则（必须）

1. 先以目标后端算子定义为准，不以训练框架导出形态为准。
2. `Custom` 节点必须显式带完整语义属性（至少 `type/input_names/output_names/output_num`）。
3. 有可选输入时必须新增 `input_index`，避免真实输入与槽位错位。
4. 性能优化不能破坏图语义：每次融合都要做功能一致性验证。
5. 转换可用性优先于激进融合，先“能转+能跑”再做更深优化。
6. 删除冗余算子，如合并多个 `Add`、`Mul` 等，减少节点数，保证图语义不变。

## 融合算子选型方法

1. 统计图中高频算子与热点子图（按节点数、耗时、内存访问）。
2. 对照目标后端（如 CANN）可用融合算子清单，做候选映射。
3. 按“收益/风险比”排序：
   - 高收益低风险：Attention、Norm、Add+Norm、激活融合
   - 高收益高风险：Rope 组合、多算子大融合
4. 当前可以融合的算子有：Attention、Norm、Add+Norm、激活融合、Rope 组合、多算子大融合

## Custom 改写标准流程

1. 确认 CANN 算子定义（规格对齐）

- 目录：`/path-of-cann/ascend-toolkit/latest/opp/built-in/op_proto/inc/`
- 目标：找到 `REG_OP(<OpName>)` 并确认：
  - 输入：`.INPUT(...)` / `.DYNAMIC_INPUT(...)` / `.OPTIONAL_INPUT(...)`
  - 输出：`.OUTPUT(...)`
  - 属性：`.REQUIRED_ATTR(...)` / `.ATTR(...)`
- dtype/shape/layout 限制（尤其 attention/rope 类算子）

2. 在模型导出脚本实现 PyTorch → ONNX Custom 的最小替换

- 统一用 `torch.autograd.Function`：
- 在 symbolic 内用：
  - `g.op("Custom", ...)`
  - 必填属性（按 [MindSpore Lite 解析逻辑](https://atomgit.com/mindspore/mindspore-lite/blob/master/mindspore-lite/tools/converter/parser/onnx/onnx_custom_parser.cc)）：
    - `type_s="<OpName>"`
    - `input_names_s=[...]`
    - `optional_input_names_s=[...]`（可为空，但会 warning）
    - `output_names_s=[...]`
    - `output_num_i=N`
    - `input_index_i=[...]`（建议提供）
  - 输出：
    - 单输出：`y = g.op(...); return y`
    - 多输出：`y0, y1 = g.op(..., outputs=2); return y0, y1`
  - 类型：
    - `y.setType(x.type())`，必要时对每个输出都 setType。
  - 形状（可选但强烈建议）：
    - `output_shapes_s="<rank,d0,d1,...>"`，动态维度用 `-1`。

3. 基于适配后的导出脚本，重新导出onnx模型
4. 导出的新onnx模型基于mindspore lite云侧转换工具验证模型转换是否成功

## 自定义算子属性规范

- 必填：
  - `type`
  - `input_names`
  - `output_names`
  - `output_num`
- 强烈建议：
  - `input_index`（有可选输入时必须）
  - `optional_input_names`（若解析器依赖）

## 控制流与动态图处理

- 若转换链路不支持控制流，优先在导出后做静态化改写（如 `If/Loop` 展开）。
- 将动态分支替换为确定路径时，需确认输入条件在部署场景固定成立。
- 对不能静态化的控制流，建议回退到不融合版本，保证可转换。

## 转换与验证策略（通用）

1. 转换前检查：
   - 目标算子是否全部改写完成
   - 关键属性是否完整且类型正确
   - 图中是否残留阻塞转换的控制流/非法常量
2. 转换执行：
   - 统一产物命名（如 `.mindir`）
3. 功能验证：
   - 随机输入冒烟测试
   - 与改写前输出做误差对比（max abs / cosine）
4. 性能验证：
   - 固定输入 shape 多轮统计（warmup + measure）
   - 记录时延、吞吐、内存峰值

## 常见故障排障手册

- `Cannot find input:  of node`
  - 原因：空输入与真实输入重排不一致
  - 处理：校验 `selected_inputs`、`input_names`、`input_index` 三者同序
- `unsupported onnx data type: 0`
  - 原因：不兼容常量（常见 STRING Constant）
  - 处理：删除未引用节点或改成合法 tensor 常量
- `Output tensor repeated`
  - 原因：错误修改 `node.output` 或拓扑冲突
  - 处理：回退输出改动，仅改属性与输入映射
- 控制流崩溃（`If/Loop`）
  - 原因：转换器不支持
  - 处理：静态化控制流；无法静态化则回退融合方案

## 交付清单（复用模板）

1. 改写脚本（含算子映射与属性抽取逻辑）
2. 导出后的 ONNX（及 external data）
3. 可复现转换命令与日志
4. 最终部署模型产物（如 `.mindir`）
5. 功能/性能对比报告（改写前 vs 改写后）

## 执行检查清单（每次必走）

1. 融合前后输出误差在阈值内
2. `Custom` 节点属性完整、`input_index` 合法
3. 转换成功且产物可加载
4. 推理功能可用，关键场景无回归
5. 性能收益明确且可重复

