---
name: open-source-model-migration
description: 把开源算法模型适配到 MindSpore Lite 部署管线：按网络结构拆分导出 ONNX、ONNX Runtime 推理验证、ONNX→MindIR 转换、MindSpore Lite 推理实现，并交付文档与常见问题。用户想把某个开源模型迁移到 MSLite 部署时调用。
---

# Adapt Open-Source Model to MindSpore Lite

本技能用于把开源算法模型适配到 MindSpore Lite 部署管线：按网络结构拆分导出 ONNX、ONNX Runtime 推理验证、ONNX→MindIR 转换、MindSpore Lite 推理实现，并交付文档与常见问题。

## 适用范围（何时调用）

当用户提出以下诉求之一时调用：

- "适配/移植某个开源模型到 MindSpore Lite"
- "按算法结构拆分导出 ONNX"
- "ONNX 转 MindIR，MindSpore Lite 部署"
- "做 ONNX vs MSLite 精度对齐验证"

## 用户必须提供的信息

| 输入 | 说明 | 示例 |
|------|------|------|
| `MODEL_NAME` | 模型标识，用于目录名和文件名 | `qwen3_vl_reranker_2b` |
| `MODEL_ID` | HuggingFace 模型 ID 或本地权重路径 | `Qwen/Qwen3-VL-Reranker-2B` |
| `MODEL_TYPE` | 模型类型，决定拆分策略 | `vl_reranker` / `llm` / `text_reranker` / `image_classification` |


## 交付物

在 `mindspore-lite/examples/base_models/{MODEL_NAME}/` 下生成：

```
{MODEL_NAME}/
├── export_{MODEL_NAME}_onnx.py    # ONNX 导出脚本
├── infer_{MODEL_NAME}_onnx.py     # ONNX Runtime 推理脚本
├── infer_{MODEL_NAME}_mslite.py   # MindSpore Lite 推理脚本
├── config.ini                     # (可选) converter_lite 配置，如 force_fp32
├── README.md                      # 部署教程文档
├── model_info/                    # (可选) ONNX 元信息（I/O、op 统计、opset、模型大小等）
├── tools/                         # (可选) 辅助工具（元信息收集/精度对齐评估/基准脚本等）
└── tests/                         # (可选) 基础测试（导出、加载、一次推理、shape 变更、异常分支）
```

**命名建议**：Python 脚本文件名不要包含 `.`（点）。如果模型名/版本号包含 `0.5b` 这类格式，建议在脚本文件名中使用 `0_5b`，否则 pylint 可能截断模块名并触发 `invalid-name`，同时也不利于作为模块被 import。

说明：
- `model_info/`、`tools/`、`tests/` 是否生成取决于用户需求与仓库/CI 约束，不应为“无需求”强制新增文件。

## 现有模型参考

仓库中已有多种模型适配实现，新模型适配时应先参考同类型模型的已有实现：

| 模型 | 类型 | 拆分方式 | 特点 |
|------|------|----------|------|
| `vit_base_patch16_224` | 图像分类 | 单模型导出 | 最简单，无拆分 |
| `jina_reranker_v3` | 文本 reranker | 单模型导出 | Listwise reranker：额外输入 doc/query token index；同一模型支持 listwise/pointwise 两种推理模式 |
| `qwen3_reranker_0.6b` | 文本 reranker | 单模型导出 | CausalLM + lm_head diff (yes-no) |
| `qwen2_0.5b` | 文本生成 | prefill + decode 拆分 | KV cache 拆分，自回归生成 |
| `qwen3_0.6b` | 文本生成 | prefill + decode 拆分 | 同上 |
| `qwen3_vl_2b_instruct` | VL 多模态生成 | Vision + LM 拆分 | Vision encoder + LLM |
| `qwen3_vl_embedding_2b` | VL 多模态 embedding | Vision + Embedding 拆分 | Vision encoder + embedding head |
| `qwen3_vl_reranker_2b` | VL 多模态 reranker | Vision + Score 拆分 | Vision encoder + score (yes-no diff + deepstack) |
| `qwen3_5_0.8b` | VL 多模态生成（混合注意力） | Vision + LM 拆分 | Vision encoder + LLM（18层线性注意力 + 6层全注意力），conv_state + recurrent_state + KV cache 三种状态 |

**关键结论**：不同模型的导出结构、推理逻辑差异很大，必须先分析模型架构再决定拆分策略和代码结构，不可套用固定模板。

---

## 阶段 1：模型结构分析与拆分策略

### 1.1 分析模型架构

1. 阅读 HuggingFace 模型文档和源码
2. 识别模型类及其子组件（vision encoder、language model、classification head 等）
3. 确定模型的输入输出格式（文本、图像、多模态）
4. 识别特殊机制：RoPE 类型、attention 实现、KV cache、deepstack、position_ids 维度等

### 1.2 确定拆分策略

| 模型类型 | 拆分方式 | 示例 |
|----------|----------|------|
| 图像分类/检测 | 单模型导出 | ViT |
| 文本 reranker | 单模型导出 | Jina-Reranker（listwise）、Qwen3-Reranker（yes-no diff） |
| 文本生成 (LLM) | Prefill + Decode 拆分（KV cache） | Qwen2-0.5B、Qwen3-0.6B |
| VL 多模态生成 | Vision + LM 拆分 | Qwen3-VL-2B-Instruct |
| VL 多模态 embedding | Vision + Embedding 拆分 | Qwen3-VL-Embedding-2B |
| VL 多模态 reranker | Vision + Score 拆分 | Qwen3-VL-Reranker-2B |

### 1.3 定义模块间接口

- 每个模块的输入/输出张量：name、dtype、shape、dynamic axis
- 固定/可变维度（batch、seq_len、H/W 等）
- 模块间数据流：vision 输出如何传入 score/LM 模块

---

## 阶段 2：ONNX 导出脚本

### 2.1 通用原则

- 每个模块写 `torch.nn.Module` wrapper，只暴露必要输入输出
- 统一输入名/输出名，便于后续推理脚本对齐
- 显式指定 `opset_version`（优先 14/17/18，结合转换器能力）
- `dynamic_axes` 覆盖 batch 与核心动态维度
- 推理模式：`eval()` + `torch.no_grad()`
- 如涉及自定义（Custom）算子接入，必须在导出阶段通过 torch 自定义算子（`torch.autograd.Function` + `symbolic()` 或 `torch.onnx.register_custom_op_symbolic`）直接导出 ONNX 中的 Custom 节点；禁止“先导出普通 ONNX，再通过 onnx.load/graph.node 遍历把节点替换成 Custom”这种后处理方案

### 2.2 导出器选择

- **优先使用 legacy 导出器** `torch.onnx.utils.export`，避免 `torch.export` 路径的 `GuardOnDataDependentSymNode` 问题
- 如遇 opset 兼容问题，可尝试不同 opset 版本（18 → 14 降级策略）

### 2.3 常见 wrapper 模式

**Vision Tower 固定 grid 导出**（Qwen3-VL 系列）：
- 缓存 position embeddings 到 wrapper 中，ONNX 只接收 `pixel_values`
- 推理时 `--image-size` 必须与导出 `--vision-image-size` 一致

**Reranker Score 导出**（yes-no diff 模式）：
- 从 `lm_head.weight` 取 `yes` 和 `no` token 权重差构建 1 维线性层
- 输出经 `sigmoid` 得到 0~1 分数

**LLM Prefill/Decode 拆分**：
- Prefill：完整输入，输出 logits + KV cache
- Decode：单 token 输入 + 历史 KV cache，输出 logits + 更新 KV cache

**Embedding 导出**：
- 在 LLM backbone 输出后接 embedding head（如 mean pooling + linear）
- 输出为固定维度向量，通常需 L2 normalize

### 2.4 代码质量要求

- 函数 NLOC <= 100，CCN <= 19（Lizard 扫描阈值）
- 超出时提取辅助函数（如 `_build_score_wrapper_and_dummies`）
- 模块、类、函数/方法均添加简洁 docstring（满足 pylint C0114/C0115/C0116，详见阶段 10.1）
- 行宽 <= 120 字符

### 2.5 生成 ONNX 元信息（可选）

当用户要求“模型信息统计/算子统计/导出可审计信息”时，为每个 ONNX 生成对应的元信息文件（建议 JSON），用于：
- 快速确认 I/O 规格（name/dtype/shape/dynamic axis）
- 统计算子分布与节点数（定位潜在不支持算子与性能热点）
- 记录模型大小与 opset（便于复现与排障）

建议字段：
- `inputs` / `outputs`：name、dtype、shape（含 dynamic 标注）
- `num_nodes`、`opset`
- `operators`：按 `op_type` 计数（TopK）
- `model_size_bytes`

实现建议：
- 使用 `onnx.load()` 读取 graph；使用 `graph.node` 统计算子
- shape 优先从 `value_info` 抽取；不足部分由导出阶段记录的 I/O spec 补全

---

## 阶段 3：ONNX Runtime 推理脚本

### 3.1 通用原则

- 完整 pipeline：加载 → 预处理 → 推理 → 后处理
- 支持动态 batch / 动态输入尺寸
- 可使用 `torch` 做预处理（ONNX 推理无此限制）
- 使用 `AutoProcessor` + `apply_chat_template` 或 `AutoTokenizer` 做分词

### 3.2 验证要求

- ONNX 推理结果应与原始 PyTorch 模型一致
- 这是后续 MSLite 精度对齐的基准（ground truth）

---

## 阶段 4：ONNX → MindIR 转换

### 4.1 转换命令

```bash
converter_lite --fmk=ONNX \
  --modelFile=xxx.onnx \
  --outputFile=xxx \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### 4.2 Ascend-oriented 固定 shape 约束（必须提前设计）

使用 `--optimize=ascend_oriented` 后，GE 往往会针对固定输入 shape 进行编译优化。推理侧若传入变长输入，可能出现：
- 推理直接失败（shape broadcast / memcpy size 等）
- 推理不报错但结果异常（最难排查）

建议做法：
- 在导出/转换阶段就确定一个固定 `max_length` 并在业务侧路由输入长度
- 推理脚本统一用 `padding="max_length"`，并要求 `--max-length` 与导出/转换一致

### 4.3 动态分档（Dynamic Dims Buckets，可选）

动态分档不是所有模型都需要做，取决于实际业务输入长度分布与性能目标：
- 若输入长度较稳定，优先固定 shape（单档），实现简单且更稳定
- 若输入长度跨度大且性能敏感，可使用少量 bucket（多档）降低 padding 浪费

当模型输入包含 `-1` 动态维度时，可通过 `ge.dynamicDims` 配置多档位形状，从而让同一个 MindIR 覆盖多个 batch/seq_len 组合（代价是编译/缓存多个 shape，增大转换与运行开销）。

**配置原则：**
- 先用 `input_shape` 声明哪些维度为 `-1`（动态）
- 再用 `ge.dynamicDims` 枚举每一档的真实取值
- `ge.dynamicDims` 每一条档位的数值个数必须等于 `input_shape` 中所有 `-1` 的总数，且按 `input_shape` 中 `-1` 的出现顺序依次填充

**示例：Listwise reranker（4 输入）同时支持 batch 与 seq_len 分档**

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;doc_token_indices:-1,64;query_token_index:-1,1"

# -1 出现顺序为：input_ids(b,s), attention_mask(b,s), doc_token_indices(b), query_token_index(b)
# 每个档位需要 6 个数，按顺序填：b,s,b,s,b,b
ge.dynamicDims="1,128,1,128,1,1;1,256,1,256,1,1;...;8,1280,8,1280,8,8"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

**提示：**
- `dynamic_dims / dynamic_batch_size / dynamic_image_size` 这三类配置不能同时存在，三选一。
- 对性能敏感场景推荐“少量 bucket”（例如 3~5 档），并在业务侧做路由；10×8=80 档会显著增加转换时间与运行时缓存压力。

### 4.4 FP16 溢出与 force_fp32

**关键经验**：部分模型（特别是 reranker/attention 密集型）在 FP16 精度下存在计算溢出，导致 MSLite 输出恒为常数（如 0.5）或全零。

**原因**：attention mask 使用 `torch.finfo(dtype).min`（FP16 下 -65504）作为极小值，FP16 矩阵运算中溢出导致中间结果异常。

**解决方案**：通过 `config.ini` 配置 `force_fp32`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

转换时指定 `--configFile=config.ini`。仅对需要 FP32 的子模型使用，Vision 模型通常无需此配置。

**注意**：默认使用动态分档或者纯动态作为样例，并且不建议使用fp32为样例，fp32性能会比较差；只有在精度不足的时候才使用fp32。

### 4.5 大模型 MindIR 文件结构

模型 >2GB 时，输出分为 `_graph.mindir` + `_variables/` 目录，推理时使用 `_graph.mindir` 路径加载。

---

## 阶段 5：MindSpore Lite 推理脚本

### 5.1 核心原则

1. **绝对禁止引入 torch 依赖** — MSLite 推理脚本中不允许出现 `import torch` 或任何 `torch.*` 操作。所有计算必须使用 numpy/PIL 完成。具体要求：
   - **不允许 `import torch`**：无论函数内部还是模块级别，均不允许导入 torch
   - **processor 返回的 torch tensor 必须立即转 numpy**：在 `_prepare_inputs` 中获取 processor 输出后，立即调用 `.numpy().astype(...)` 转换，后续所有操作均基于 numpy 数组
   - **position_ids 计算用 numpy 实现**：`torch.arange` → `np.arange`，`torch.zeros` → `np.zeros`，`torch.cat` → `np.concatenate`，`torch.stack` → `np.stack`，`torch.repeat_interleave` → `np.repeat`，`torch.unsqueeze` → `reshape`，`torch.masked_fill` → `np.where`，`torch.cumsum` → `np.cumsum` 等
   - **不允许 `.cpu().numpy()` / `.item()` 调用**：因为数据已经是 numpy 数组，直接用 numpy 操作即可
   - **唯一例外**：`AutoProcessor.apply_chat_template(..., return_tensors="pt")` 会返回 torch tensor，但必须在获取后立即转为 numpy，不允许在转换前做任何 torch 操作
2. **使用 `AutoTokenizer`（非 AutoProcessor）** — 避免 Fast ImageProcessor 要求 PyTorch tensor 的问题
3. **图像预处理使用 `AutoImageProcessor(..., use_fast=False)`** — 慢版 processor 支持 `return_tensors="np"`
4. **多模态图像 token 用字符串构建** — 如 `<|vision_start|><|image_pad|>...<|vision_end|>`，不依赖 processor 的图像处理
5. **不在代码中指定 `context.ascend.precision_mode`** — 精度模式由 `converter_lite` 转换时的 `config.ini`（如 `force_fp32`）控制，推理脚本中不应重复设置，避免与转换配置冲突
6. **`Model.predict` 返回值直接取用，无需类型校验** — `mslite.Model.predict()` 返回值一定是 `MSTensor` 列表，每个元素必然具备 `get_data_to_numpy()` 方法。直接调用 `outputs[0].get_data_to_numpy()` 取 numpy 数组即可，不要写 `hasattr(outputs[0], "get_data_to_numpy")` 之类的分支判断，也不要保留 `np.array(outputs[0])` 回退——MindSpore Lite 接口不会返回 numpy 类型，此类校验属于无效代码

### 5.2 输入 dtype 对齐

MindIR 模型对输入 dtype 有严格要求，必须显式转换。**注意**：不同模型/不同转换配置下期望 dtype 可能不同，需通过 `_describe_model_io()` 或检查转换日志确认。

常见 dtype 规则：

| 输入类别 | 常见期望 dtype | 说明 |
|----------|---------------|------|
| `input_ids` | `np.int32` 或 `np.int64` | 需检查模型实际要求，Ascend 上常见 int32 |
| `attention_mask` | `np.int32` 或 `np.int64` | 同上 |
| `position_ids` | `np.int32` 或 `np.int64` | 同上 |
| `pixel_values` | `np.float16` 或 `np.float32` | 取决于导出精度 |
| embedding 类输入 | `np.float16` | 如 `image_embeds`、`deepstack_embeds` 等 |

**最佳实践**：推理脚本中先打印模型输入信息确认 dtype，再显式 `.astype()` 对齐。

### 5.3 输入匹配机制

使用 `_build_mslite_inputs()` 按名称匹配模型输入，fallback 到 preferred_order：

```python
def _build_mslite_inputs(model, feed_dict, preferred_order=None):
    inputs = model.get_inputs()
    # 1. 尝试按名称匹配
    # 2. 名称不可用时按 preferred_order 顺序
    # 3. 都失败则抛出明确错误
```

### 5.4 position_ids 注意事项

不同模型的 position_ids 格式差异很大，需根据模型架构实现对应的 numpy 版本：

| 模型系列 | position_ids 格式 | 说明 |
|----------|-------------------|------|
| Qwen2/Qwen3 (纯文本) | `[1, batch, seq_len]` | 标准 1D 递增 |
| Qwen3-VL (多模态) | `[4, batch, seq_len]` | text_pos + 3D RoPE（含 mm_token_type_ids） |
| ViT | 无需 position_ids | 位置编码已内置 |

**关键点**：
- MSLite 推理脚本中需实现与 ONNX 脚本中 torch 版本逻辑一致的 numpy position_ids 计算
- 多模态模型中 `mm_token_type_ids` 通过 `(input_ids == image_token_id)` 计算，不从 processor 获取

### 5.5 性能计时

在推理脚本中添加关键步骤的耗时统计：

- Vision 推理时间
- 预处理时间
- Score/LM 推理时间
- 端到端总时间
- 序列长度与步数信息（例如 `src_len` / `feat_len` / `decode_steps` / `num_tokens`）
- 自回归场景需统计 `Total decode time / Avg decode step / steps`，并给出吞吐（tok/s）

说明：
- 性能数据建议以推理脚本的端到端打印为准（可覆盖后续“免拷贝”等整体优化收益）。
- `benchmark` 更适合单图算子级/子图级测量，通常无法体现端到端链路优化收益。

### 5.6 LLM 自回归生成（Prefill + Decode）

对于文本生成模型（如 Qwen2-0.5B、Qwen3-0.6B），MSLite 推理需实现 decode 循环：

1. **Prefill 阶段**：将完整 prompt 送入 prefill 模型，获取首个 token 的 logits 和初始 KV cache
2. **Decode 循环**：将上一步生成的 token + 历史 KV cache 送入 decode 模型，逐步生成后续 token
3. **KV cache 管理**：每次 decode 后更新 KV cache（past + new），作为下一步输入
4. **停止条件**：生成 EOS token 或达到 `max_new_tokens`

**关键注意**：
- KV cache 的 shape 和拼接方式必须与导出时的 `dynamic_axes` 定义一致
- Ascend 上 KV cache 的 dtype 通常为 float16

### 5.7 代码质量要求

- 函数 NLOC <= 100，CCN <= 19
- 从 `main()` 提取辅助函数（如 `_run_vision_inference`、`_prepare_score_inputs`）
- 模块、类、函数/方法均添加简洁 docstring（满足 pylint C0114/C0115/C0116，详见阶段 10.1）
- 行宽 <= 120 字符

---

## 阶段 6：验证与调试

### 6.1 精度对齐验证流程

1. 运行 ONNX 推理，记录输出（作为基准）
2. 运行 MSLite 推理，使用相同输入，比较输出
3. 误差阈值建议按计算精度分档：
   - FP32：`rtol=1e-4, atol=1e-4`
   - FP16 / 混合精度：`rtol=1e-3, atol=1e-3`
4. 若输出是文本/类别等离散结果，优先对齐最终业务输出（如文本一致、Top-1 一致），必要时再对齐中间张量（logits/embedding）。

批量精度对齐（可选，用户要求“精度评估/报告”时启用）：
- 样本规模：默认不少于 1000 样本（优先使用用户提供的真实数据集；否则提供固定 seed 的可复现实验数据生成器覆盖典型 shape 组合）
- 指标：
  - 输出余弦相似度（cosine similarity）
  - 最大绝对误差（max abs error）
  - 若存在明确业务指标（如 WER/EM/F1 等），同时报告业务指标差异（建议 < 0.1%）
- 报告：输出汇总统计（均值/分位数）与失败样本列表；如可抓取中间层，可定位首个差异节点辅助排障

### 6.2 常见问题排查

以下问题汇总自 `mindspore-lite/examples/base_models/` 下已有模型的实际适配经验：

#### 导出阶段

| # | 问题 | 现象 | 可能原因 | 解决方案 |
|---|------|------|----------|----------|
| 1 | torch.export 报错 | `GuardOnDataDependentSymNode` | 新导出器 `torch.onnx.export` 内部走 `torch.export`，不支持数据依赖控制流（如 `lengths.tolist()`） | 使用 legacy 导出器 `torch.onnx.utils.export` |
| 2 | Vision 导出返回类型异常 | `AttributeError: 'tuple' object has no attribute 'pooler_output'` | 部分导出/trace 路径下，即使传 `return_dict=True` 也可能返回 `tuple` | wrapper 中对 `tuple/list` 与 `ModelOutput` 两种返回做兼容 |
| 3 | transformers API 变化 | `AttentionInterface` 无 `get_interface` | `transformers>=4.57` 的 `ALL_ATTENTION_FUNCTIONS` 变为 `AttentionInterface` | 使用 `.get(key, fallback)` 替代 `.get_interface()` |
| 4 | legacy exporter 参数不支持 | `export() got an unexpected keyword argument 'use_external_data_format'` | legacy 导出器不支持该参数 | 移除该参数；如需外部权重用 `torch.onnx.export(..., external_data=True)` |
| 5 | 导出 OOM | `CUDA out of memory` | 模型过大或 GPU 显存不足 | 使用 `--device cpu` 导出；减小 `--vision-image-size` 或 `--dummy-seq-len` |
| 6 | transformers 版本过低 | `ImportError: cannot import name 'AutoModel'` | transformers 版本不满足模型要求 | `pip install --upgrade transformers` 或安装指定版本 |
| 7 | 模型下载失败 | `OSError: Can't load model from '...'` | 网络问题或 HF 访问受限 | 设置 `HF_ENDPOINT=https://hf-mirror.com`；或 `git clone` 到本地后用本地路径 |
| 8 | 混合注意力模型导出 | 线性注意力（GatedDeltaNet）层需要特殊处理 | Qwen3.5 等模型混合使用线性注意力和全注意力，线性注意力使用 conv_state + recurrent_state 而非 KV cache | 为线性注意力实现独立的 prefill（chunk）和 decode（recurrent）forward 函数，导出时同时管理 conv_state、recurrent_state 和 KV cache 三种状态 |
| 9 | conv1d 权重维度不匹配 | `Expected 4-dimensional input for 4-dimensional weight [6144, 1, 1, 4], but got 3-dimensional input` | causal_conv1d 的 Conv1d 权重形状为 `(out_channels, 1, kernel_size)`（groups=1 维度），但标准 F.conv1d 期望 `(out_channels, in_channels/groups, kernel_size)` | decode 阶段手动调用 `F.conv1d` 时，使用 `layer.conv1d.weight`（3D）而非 `layer.conv1d.weight.unsqueeze(1)`（4D），因为 groups=conv_dim 时 in_channels/groups=1 |
| 10 | onnxscript 模块缺失 | `ModuleNotFoundError: No module named 'onnxscript'` | 新版 PyTorch 的 `torch.onnx.export` 默认使用新导出器，依赖 onnxscript | 使用 legacy 导出器 `torch.onnx.utils.export` 替代 `torch.onnx.export` |
| 22 | Qwen3 JIT tracing 时 masking_utils 报错 | `IndexError: tuple index out of range` in `masking_utils.sdpa_mask` | Transformers v4.50+ 的 Qwen3 模型在 JIT tracing 时，`_preprocess_mask_arguments` 返回的 `q_length`/`kv_length` 变为 0 维 Tensor，导致向后兼容检查 `q_length.shape[0]` 失败 | 在导出前 monkey-patch `_preprocess_mask_arguments`，将 0 维 Tensor 转为 int；或在 wrapper 中显式传入 `cache_position=torch.arange(seq_len)` |
| 23 | Listwise reranker 的 boolean indexing 不可导出 | ONNX 导出时 boolean indexing（如 `hidden_states[input_ids == token_id]`）不支持动态形状 | boolean indexing 产生的输出形状依赖于数据，无法在 ONNX 中表达动态维度 | 使用 `torch.gather` 替代 boolean indexing：将特殊 token 位置作为显式输入传入，用 `torch.gather(hidden_states, 1, indices)` 提取对应位置的隐藏状态 |
| 24 | ONNX 模型含 FLOAT16 类型声明导致 MSLite 转换或 ORT 推理失败 | `do not support data_type: 10`（converter 报错）或 `Type Error: Type (tensor(float16)) of output arg does not match expected type (tensor(float))`（ORT 报错） | 模型以 `torch.float16` 加载导出时，ONNX 图的类型声明和初始值均为 FLOAT16，MindSpore Lite 的 Clip 等解析器不支持 FLOAT16；仅转换初始值 dtype 不够，图中节点的输入/输出类型声明也需同步修改 | **根本方案：** 以 `torch_dtype=torch.float32` 加载模型再导出，从源头确保 ONNX 全图为 float32；**补充方案：** 在优化步骤中将 FLOAT16 初始值转为 FLOAT32（`numpy_helper.to_array(init).astype(np.float32)`）作为防御性检查 |
| 25 | ONNX 优化后加载失败（外部数据文件） | `INVALID_PROTOBUF: Protobuf parsing failed` | 大模型 ONNX 使用外部数据文件（`.onnx.data`），优化后 `onnx.save` 未正确更新外部数据文件，导致 `.onnx` 与 `.onnx.data` 不匹配 | `onnx.load` 时加 `load_external_data=True`；保存时删除旧 `.onnx.data`，使用 `onnx.save_model(..., save_as_external_data=True, all_tensors_to_one_file=True, location=..., size_threshold=1024, convert_attribute=True)` |
| 26 | TTS Flow Encoder mask 被导出为常量 | 端到端合成音频“前面清晰、后面噪声”；打印 `flow_mask.mean()` 明显小于 1（例如 0.37） | wrapper 内使用 `torch.tensor([mel_len])` 等构造 mask，legacy ONNX exporter 可能把该长度 trace 为 dummy 常量，导致真实输入更长时尾部被 mask 掉 | 用动态 shape 直接构造全 1 mask（如 `attn_mask = h.new_ones((h.shape[0], 1, mel_len))`），重新导出 ONNX 并重新转换 MindIR |
| 27 | Flow Encoder ONNX 转 MindIR 失败（GE Resize 维数） | `The dim of input x is not 4`；节点如 `/encoder/up_layer/Resize` | CosyVoice `Upsample1D` 在 `(B, C, T)` 上做 `interpolate`，导出为 3D `Resize`；GE `ResizeNearestNeighborV2` 仅支持 4D | 导出侧等价改写为 `(B, C, T, 1)` + `scale_factor=(stride, 1)` + `squeeze(-1)`，或对 CosyVoice 源码中 `Upsample1D.forward` 做同样修改后重导出 ONNX |

#### 推理阶段

| # | 问题 | 现象 | 可能原因 | 解决方案 |
|---|------|------|----------|----------|
| 11 | MSLite 输出恒为常数或全零 | 如始终 0.5 或全零，ONNX 正常 | FP16 计算溢出：attention mask 的极小值在 FP16 下溢出 | 添加 `config.ini` 配置 `force_fp32`（注意不是 `enforce_fp32`，后者仅限制输入精度，中间计算仍可能 FP16） |
| 12 | 输入 dtype 不匹配 | `required 34, given 35` | MindIR 期望 int32，传入了 int64 | 显式 `.astype(np.int32)` |
| 13 | Fast ImageProcessor 报错 | `Only returning PyTorch tensors is currently supported` | Fast processor 不支持 numpy 输出 | 使用 `AutoImageProcessor(..., use_fast=False)` |
| 14 | position_ids 维度错误 | Shape 不匹配 | 维度或格式与导出不一致 | 检查是否为 4D `[4, batch, seq_len]`（Qwen3-VL 系列） |
| 15 | pixel_values 长度不匹配 | `pixel_values.shape[0]` 与 vision ONNX 固定长度不一致 | 推理 `--image-size` 与导出 `--vision-image-size` 不一致 | 确保两者一致；脚本应 pad+resize 到固定尺寸 |
| 16 | image_embeds 长度不匹配 | 推理时报 embeds 长度与 image_token 数不匹配 | processor 与模型版本不一致，或 grid_thw 计算错误 | 确保 processor 版本一致；检查 `image_grid_thw` |
| 17 | AutoProcessor 加载报 NoneType | `TypeError: expected str, bytes or os.PathLike object, not NoneType` | HF 模型 ID 的 chat template 解析异常 | 使用本地模型目录作为 `--processor` |
| 18 | 推理结果不正确 | ONNX/MSLite 输出与 PyTorch 不一致 | tokenizer 版本不匹配、KV cache 维度错误、精度问题 | 确认 tokenizer 一致；检查 KV cache；尝试 fp32 推理 |
| 19 | TTS 模型 HiFT 权重加载 size mismatch | `size mismatch for conv_pre.parametrizations.weight.original1` | yaml 中声明 `HiFTGenerator` 但实际权重可能是 `CausalHiFTGenerator` 格式（或反之），两者 `conv_pre` kernel size 和 ups 通道顺序不同 | 检查权重中 `conv_pre` 的 kernel size 推断 `conv_pre_look_right`；检查 `ups` 层的 in/out channels 确认是 `ConvTranspose1d`（HiFTGenerator）还是 `CausalConv1dUpsample`（CausalHiFTGenerator） |
| 20 | MSLite Pointwise/逐条推理结果与 ONNX 不一致 | Listwise 模式正常，Pointwise 部分结果严重偏移 | `--optimize=ascend_oriented` 下 GE 针对固定 shape 编译图，逐条推理时序列长度变化导致 GE 无法正确适配 | 将所有样本一起 tokenize + padding 至等长后逐条推理，保证输入 shape 一致 |
| 20 | 预导出 ONNX batch 固定导致推理失败 | `Got: 1 Expected: 2` | 部分模型仓库提供预导出的 ONNX 文件，但 batch 维度固定 | 自行导出 ONNX，设置 `dynamic_axes` 中 batch 为动态维度 |
| 21 | BFloat16 权重导出 ONNX dtype 不匹配 | `mat1 and mat2 must have the same dtype, but got Float and BFloat16` | 部分模型权重为 BFloat16，ONNX 导出需要 Float32 | 导出前调用 `.float()` 将模型转为 Float32 |
| 28 | MSLite 推理时报错：`Acl memcpy input X data to device failed, src input size: 0` | 日志显示 `src input size: 0, dst device buffer size: 0`，通常发生在有可选输入（如 prompt tokens/prompt_feat）的模型 | MSLite Ascend runtime (ACL) 无法处理 size=0 的 tensor，即使 ONNX 导出时用空 dummy 也无法解决。这是 ACL 层的限制，而非 ONNX 图的问题 | **推理侧 padding（推荐）：** 当可选输入为空时，填充最小非零值。例如：`speech_ids` 为空 `(1,0)` 时填充为 `(1,1)` 并使用 token 0；`prompt_feat` 为空 `(1,0,80)` 时填充为 `(1,1,80)` 并使用 zeros。**导出侧匹配：** 导出时使用相同的最小 dummy 值（如 `speech_len=1`, `feat_len=1`）以保持一致性。**注意：** 填充值的选择很重要——应使用对模型影响最小的值（如 token 0 而非 EOS，零帧而非随机值），并在后处理中扣除填充部分（如 mel 输出时移除 prompt_feat 对应的帧） |
| 29 | CosyVoice2 MSLite 推理音频内容错误且时长异常 | ONNX 推理音频正常（时长约 2.1s），MSLite 推理生成音频"瞎说"（能听出是汉语但内容错误）且时长异常（约 6.72s） | **根因 1**：当 `speech_len=0`（无 prompt 音频）时，MSLite Ascend runtime 无法处理空 tensor，导致 prefill 阶段 KV cache 维度错误，后续 decode 生成的 token 序列异常。**根因 2**：ONNX 导出时部分维度 `dim_value=0`（如空 speech 输入），MindIR 转换后这些零维可能被错误处理 | **推理侧**：当 `speech_len == 0` 时，padding 为 `[0]` token（`speech_ids = [[0]]`），推理后从输出中移除 padding（`logits[:, :-1, :]`, `past_kv[:, :, :, :-1, :]`）。**导出侧**：添加 `_sanitize_onnx_zero_dims()` 函数，将 `dim_value=0` 转为 `dim_param`（动态维度），避免零维固定值带来的转换问题。**验证方法**：添加 `--decode-mode greedy` 选项，使 ONNX/MSLite 均使用确定性采样，便于对比中间输出（logits/KV cache）定位精度差异 |

#### 转换阶段

| # | 问题 | 现象 | 可能原因 | 解决方案 |
|---|------|------|----------|----------|
| 19 | 大模型加载失败 | 找不到权重 | `_variables/` 目录缺失或与 `_graph.mindir` 不在同目录 | 确保两者在同一目录下 |
| 20 | converter_lite 算子不支持 | `Unsupported operator` | MindSpore Lite 当前版本不支持该算子 | 优先在导出侧替换/绕过不支持算子；向 MindSpore Lite 社区提 issue |
| 21 | 转换失败 | opset 不兼容 | MindSpore Lite 版本不支持高 opset | 尝试降低 opset 版本（如 18→14） |
| 22 | 转换失败但终端报错不全 | 终端只看到 `"[ERROR] FE(1087713,converter_lite):..."` 或 `"[ERROR] LITE(1087713,...,converter_lite):..."`，缺少更详细的堆栈/根因信息 | 终端日志被截断或 converter_lite 仅打印摘要错误 | 从错误行中提取进程号（如 `1087713`），查看 `~/ascend/log/debug/plog/plog-1087713` 获取更完整的 plog 以定位根因 |


### 6.3 诊断方法

当 ONNX 和 MSLite 输出不一致时：

1. 构造相同 dummy 输入，分别送入 ONNX 和 MindIR 模型
2. 逐层对比中间输出，定位首个差异节点
3. 检查该节点是否涉及 FP16 溢出或不支持算子
4. 尝试 `force_fp32` 或在导出侧替换问题算子

---

## 阶段 7：README 文档

README 书写规则与可直接落地的 9 板块模板见：

- references/README_WRITING_TEMPLATE_AND_RULES.md

---

## 阶段 8：更新仓库 README 模型支持列表

开发验证完成后，必须在仓库根目录的中英文 README 中更新"云侧推理模型支持列表"表格：

1. 打开 `README_CN.md` 和 `README.md`，找到"云侧推理模型支持列表"表格
2. 在表格中找到本次适配的模型名称，为其添加超链接和勾选标记（&#9989;）
3. 超链接格式：`[模型名](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/{MODEL_NAME}) &#9989;`
4. 如果模型名称不在表格中，需根据其类型（图像/视频生成、VLM、LLM、音频、其他）添加到对应列的合适位置
5. 中英文 README 必须同步修改

---

## 阶段 9：更新 Skill 文档经验库

开发验证完成后，必须将本次适配过程中遇到的新问题和重要经验更新到本 skill 文档的"常见问题排查"表格（阶段 6.2）中：

- 如果遇到了已有问题表中的问题但解决方案不同，补充替代方案
- 如果遇到了新问题（表中未覆盖的），按阶段分类新增一行，包含：问题、现象、可能原因、解决方案
- 如果发现了新的通用经验或最佳实践，更新到对应阶段的说明中

---

## 阶段 10：CI 告警检查（提交 PR 前执行）

当用户提出"即将提交 PR"或类似意图时，触发本阶段。本阶段确保代码和文档通过 CI 的静态检查。

### 10.1 Pylint 代码风格检查

对所有 `.py` 文件执行 pylint 检查，确保无以下常见告警：

| 告警代码 | 说明 | 修复方式 |
|----------|------|----------|
| C0114 | 缺少模块 docstring | 在文件首行 `import` 之前添加模块级 `"""..."""` |
| C0115 | 缺少类 docstring | 在 `class` 体首行添加 `"""docstring"""` |
| C0116 | 缺少函数/方法 docstring | 在函数/方法体首行添加 `"""docstring"""` |
| W0612 | 未使用的变量 | 移除变量；如需保留返回值/解包语义，用 `_` 承接未使用项（如 `y, _ = f(x)`），或改成只取所需字段（如 `q_len = input_ids.shape[1]`） |
| W0611 | 未使用的 import | 移除未使用的 import 语句 |
| W0613 | 未使用的参数 | 删除参数（若不影响接口）；或用 `del param` 显式标记未使用；或将参数名改为 `_param`（与 pylintrc 的 dummy variable 规则保持一致） |
| E0606 | 变量可能在赋值前使用 | 确保变量在所有分支都赋值；常见做法是循环/条件前初始化变量，或重构逻辑避免跨分支复用临时变量 |
| C0200 | 建议使用 enumerate | 将 `for i in range(len(xs))` 替换为 `for i, x in enumerate(xs)` |

执行命令：
```bash
cd mindspore-lite/examples/base_models/{MODEL_NAME}
pylint --rcfile=<repo_root>/.jenkins/rules/pylint/pylintrc *.py
```

**修复原则**：
- 不允许通过 `# pylint: disable=` 注释忽略告警，必须实质性修复
- 未使用变量直接移除；如需保留解包语义，直接取所需字段（如 `q_len = input_ids.shape[1]` 替代 `bsz, q_len = input_ids.shape`）
- 对“刻意未使用”的参数/变量，优先使用 `_` 或 `_name` 承接，或使用 `del` 显式丢弃，避免保留无意义变量名
- docstring 应简洁描述用途：**模块**（脚本做什么）、**类**（职责）、**函数/方法**（输入输出语义），无需冗长；`main()`、私有辅助方法（如 `_build_prompt`）、性能统计小方法（如 `add_*`）同样需要 docstring，否则仍会触发 C0116

### 10.2 MSLite 推理脚本 torch 依赖检查

MSLite 推理脚本（`infer_{MODEL_NAME}_mslite.py`）中**绝对不允许**出现 `import torch` 或 `torch.*` 操作。检查方法：

```bash
grep -n "import torch\|torch\." infer_{MODEL_NAME}_mslite.py
```

如果命中任何结果（注释除外），必须替换为 numpy 等价实现。常见替换：

| torch 操作 | numpy 等价 |
|------------|-----------|
| `torch.arange(n)` | `np.arange(n)` |
| `torch.zeros(shape)` | `np.zeros(shape)` |
| `torch.cat([...], dim=d)` | `np.concatenate([...], axis=d)` |
| `torch.stack([...], dim=d)` | `np.stack([...], axis=d)` |
| `torch.repeat_interleave(a, n)` | `np.repeat(a, n)` |
| `tensor.unsqueeze(d)` | `array.reshape(...)` |
| `tensor.masked_fill(mask, val)` | `np.where(mask, val, array)` |
| `tensor.cumsum(dim=d)` | `array.cumsum(axis=d)` |
| `tensor.bool()` | `array.astype(bool)` |
| `tensor.item()` | `int(array)` 或直接取标量 |
| `tensor.cpu().numpy()` | 不需要（已是 numpy） |

**唯一允许 torch 出现的场景**：`AutoProcessor.apply_chat_template(..., return_tensors="pt")` 的参数 `"pt"`，但返回值必须立即 `.numpy()` 转换。

### 10.3 Markdown 格式检查

对 `README.md` 执行 MarkdownLint 检查，确保无以下常见告警：

| 规则 | 说明 | 修复方式 |
|------|------|----------|
| MD001 | 标题层级必须逐级递增 | 不允许 `#` → `###`，必须 `#` → `##` → `###` |
| MD009 | 行尾不允许有多余空格 | 移除行尾空格 |
| MD022 | 标题前后需要空行 | 在标题前后添加空行 |
| MD023 | 标题必须从行首开始 | 移除标题前的缩进 |
| MD027 | 块引用后有多余空格 | 移除多余空格 |
| MD031 | 围栏代码块前后需要空行 | 在 ``` 前后各留一行空行 |
| MD032 | 列表前后需要空行 | 在列表前后添加空行 |

**特别关注 MD001**：`README.md` 开头的模型概述列表后，如果紧跟子标题，必须使用 `##`（二级标题），不能跳到 `###`（三级标题），因为文档的一级标题 `#` 已被文档标题占用。

### 10.4 代码复杂度检查

使用 Lizard 确保函数复杂度在阈值内：

```bash
lizard -C 20 -L 100 *.py
```

- CCN（圈复杂度）<= 19
- NLOC（有效代码行数）<= 100

超出阈值时，提取辅助函数降低复杂度。

### 10.5 推理验证

确保推理脚本可正常运行：

- ONNX 推理脚本至少验证一个文本/图像输入
- MSLite 推理脚本至少验证一个文本/图像输入
- 输出结果合理（非全零、非常数）

---

## 执行检查清单

- [ ] ONNX 导出成功，文件完整
- [ ] ONNX 能被 onnxruntime 正常加载并推理
- [ ] ONNX 推理结果与 PyTorch 模型一致
- [ ] 若模型声明支持动态维度：至少在 2 组不同 shape 下验证 ONNX 与 MSLite 推理可运行且输出一致
- [ ] ONNX → MindIR 转换成功
- [ ] MSLite 推理输出与 ONNX 在阈值内一致
- [ ] config.ini 已按需配置（如 force_fp32）
- [ ] Lizard 扫描通过：CCN <= 19，NLOC <= 100
- [ ] pylint 扫描无告警：C0114、C0115、C0116、W0613、R1716、W1309、R1735
- [ ] 行宽 <= 120 字符
- [ ] README 包含完整 9 个板块
- [ ] 常见问题已记录（含实际遇到的问题及解决方案）
- [ ] 仓库 README_CN.md 和 README.md 模型支持列表已添加超链接和勾选标记
- [ ] Skill 文档常见问题排查表已更新本次适配的新问题和经验

### 提交 PR 前额外检查（阶段 10）

- [ ] Pylint 检查无 C0114/C0115/C0116/W0612/W0611 告警
- [ ] MSLite 推理脚本无 `import torch` 或 `torch.*` 操作（grep 验证）
- [ ] MarkdownLint 检查无 MD001/MD009/MD022/MD023/MD031/MD032 告警
- [ ] Lizard 复杂度检查通过
- [ ] 推理脚本可正常运行
