# Jina Reranker V3.5 Listwise ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Jina Reranker V3.5 模型以原生 Listwise 架构导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

> 说明：本模型支持 **Listwise** 打分模式（query + N 篇 doc 在同一上下文窗口内交互排序）。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch | 2.8.0 |
| transformers | 5.4.0 |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 2.4.6 |
| mindspore-lite | 2.10.0 |
| CANN | 8.5 |

```bash
pip install transformers==5.4.0 torch==2.8.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==2.4.6
```

### 模型权重

从 [HuggingFace](https://huggingface.co/jinaai/jina-reranker-v3.5) 下载权重到本地目录（本教程以 `./weights/jina-reranker-v3.5` 为例）：

```bash
git lfs install
git clone https://huggingface.co/jinaai/jina-reranker-v3.5 ./weights/jina-reranker-v3.5
```

---

## 2. 模型导出 ONNX

### 导出命令（生产 / 融合 ONNX）

```bash
cd examples/base_models/jina_reranker_v3_5

python export_jina_reranker_v3_5_onnx.py \
  --model-id ./weights/jina-reranker-v3.5 \
  --output-dir ./onnx/fuse \
  --device cpu
```

说明：

- 默认导出融合 ONNX（包含 CANN Custom 融合算子），用于 MindSpore Lite/Ascend 侧的融合性能验证与部署。
- 如果要跑 ONNX Runtime 推理，必须加 `--disable-fusion-opt` 导出 non-fuse ONNX（见 2.3 节）。

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `jinaai/jina-reranker-v3.5` |
| `--output-dir` | 输出目录 | `./onnx` |
| `--device` | 导出设备 | `cpu` |
| `--disable-fusion-opt` | 禁用融合导出，生成 non-fuse ONNX（用于 ONNX Runtime 推理） | 默认关闭（未使能） |
| `--enable_bmm2mm_fusion` | 使能 BMM->MMv2 优化：Linear 走 2D MatMulV2 Custom（仅融合导出有效） | 默认关闭（未使能） |
| `--enable_rmsnorm_fusion` | 使能 RmsNorm/AddRmsNorm 融合 Custom 算子 | 默认关闭（未使能） |
| `--enable_qk_merge` | 使能 QK merge：合并 q_proj+k_proj 为一个 Linear（仅融合导出有效） | 默认关闭（未使能） |
| `--enable_swiglu_fusion` | 使能 SwiGlu 融合 Custom 算子（cat+silu*up） | 默认关闭（未使能） |
| `--enable_sliding_pfa` | 使能滑动注意力层的 PFA 路由（banded 滑窗 mask） | **默认开启（生产最优）** |
| `--disable_sliding_pfa` | 关闭滑动注意力层的 PFA 路由，回退到参考 SDPA 实现 | 默认关闭 |
| `--enable_sliding_sparse` | 额外在滑动 PFA 层设置 sparse_mode/pre_tokens（实验性，GE 不支持，勿用于生产） | 默认关闭（未使能） |
| `--pfa_layout` | PFA Custom op 的 q/k/v 张量布局：`BSH`(3D)/`BSND`(4D)/`BNSD`(4D)（BSND 与 BSH 性能等价，BNSD 因 N/S 转置劣化 ~5ms，见第 6 节布局消融） | `BSH` |

### 产出

```log
onnx/fuse/
├── jina_reranker_v3_5.onnx
├── jina_reranker_v3_5.onnx.data
└── (转换后) jina_reranker_v3_5_graph.mindir + jina_reranker_v3_5_variables/
```

### 模型架构说明

导出的 ONNX 模型包含以下核心组件：

- **Qwen3-0.6B Backbone**：28 层 Transformer，**混合注意力**（12 层 full_attention + 16 层 sliding_attention，`sliding_window=1024`，第一个 layer 为 sliding_attention），16 个 query heads、8 个 KV heads，head_dim=128
- **Think/NoThink 提示词**：输出侧附加 `think\n\n/think\n\n`（空 think 块）后缀，与官方 listwise 模板一致
- **MLP Projector**：1024→512→512 维度映射（带 ReLU 激活）
- **Cosine Similarity**：计算 query 与各 document 的相关性分数

**ONNX 模型输入：**

| 输入名 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `input_ids` | (batch, seq_len) | int64 | 完整 listwise prompt 的 token IDs |
| `attention_mask` | (batch, seq_len) | int64 | 注意力掩码 |
| `doc_token_indices` | (batch, num_docs) | int64 | `<\|embed_token\|>` 的位置索引 |
| `query_token_index` | (batch, 1) | int64 | `<\|rerank_token\|>` 的位置索引 |

**ONNX 模型输出：**

| 输出名 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `scores` | (batch, num_docs) | float | 每个 document 的相关性分数 |

### 导出 non-fuse ONNX（用于 ONNX Runtime）

```bash
python export_jina_reranker_v3_5_onnx.py \
  --model-id ./weights/jina-reranker-v3.5 \
  --output-dir ./onnx/non_fuse \
  --device cpu \
  --disable-fusion-opt
```

---

## 3. MindSpore Lite 转换

### 转换命令

```bash
# 动态分档：batch=1，两档 {1280, 4096}（加速转换；31 档转换需 60+ 分钟，两档约 4 分钟）
$Convert --fmk=ONNX \
  --modelFile=./onnx/fuse/jina_reranker_v3_5.onnx \
  --outputFile=./onnx/fuse/jina_reranker_v3_5 \
  --saveType=MINDIR \
  --optimize=ascend_oriented \
  --configFile=./config.ini
```

> 注意：必须从示例根目录执行上述命令（`op_fp32.json` 的相对路径依赖当前工作目录）。

### 参数说明

| 参数 | 说明 |
|---|---|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出 MindIR 路径（不带扩展名） |
| `--optimize` | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径 |

### 配置文件

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;doc_token_indices:1,64;query_token_index:1,1"
ge.dynamicDims="1280,1280;4096,4096"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./op_fp32.json"

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul
```

`op_fp32.json`（RmsNorm 链路上的关键算子锁定 FP32，保证混合精度下精度无损）：

```json
{
    "black-list":{
        "to-add":[
            "RealDiv",
            "SquareSumV1"
        ]
    }
}
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
onnx/fuse
├── jina_reranker_v3_5_graph.mindir      # MindIR 图定义
├── jina_reranker_v3_5_variables/data_0   # 权重数据
└── jina_reranker_v3_5.onnx               # 原始 ONNX 模型
```

---

## 4. MindSpore Lite 推理

说明：

- 动态分档 MindIR：推理侧必须实现“选档 + pad”，并保证传入的 shape 精确等于某个档位；每次推理前需执行 `model.resize(inputs, dims)`（本脚本已内置 resize）。

### Listwise 模式推理

```bash
python infer_jina_reranker_v3_5_mslite.py \
  --model-path onnx/fuse/jina_reranker_v3_5_graph.mindir \
  --tokenizer ./weights/jina-reranker-v3.5 \
  --max-length 1280 \
  --device ascend \
  --device-id 0
```

**执行日志（示例）：**

```log
Loading tokenizer from ./weights/jina-reranker-v3.5
Initializing MindSpore Lite context for ascend...
Loading model from onnx/fuse/jina_reranker_v3_5_graph.mindir...

Running inference in listwise mode with max_length=1280...

Reranking results (listwise mode):

[1] Score: 0.3751
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...
[2] Score: 0.2980
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....
[3] Score: 0.2977
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...
[4] Score: 0.1987
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....
[5] Score: -0.1759
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....
[6] Score: -0.1853
Document: Basketball is one of the most popular sports in the United States....

============================================================
Higher scores indicate better relevance to the query.
============================================================
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--tokenizer` | HuggingFace tokenizer 路径 | `jinaai/jina-reranker-v3.5` |
| `--max-length` | 最大序列长度 | `1280` |
| `--device` | 推理设备（ascend/cpu） | `cpu` |
| `--device-id` | Ascend 设备 ID | `0` |

> `--max-length` 必须命中 `ge.dynamicDims` 中的档位（`1280` 或 `4096`），否则 `model.resize` 会因 shape 不在允许列表而报错。

### 分块策略说明

当文档数量较多或总长度超过上下文窗口时，Listwise 模式自动执行分块处理：

1. 将文档按 token 长度分块，每块不超过 `max_length - 2 * query_length`，单块最多 64 个文档
2. 每块独立推理，获取该块内各文档的分数
3. 以各块最高分数的归一化值作为权重，用于跨块分数归一化
4. 合并所有块的分数，按降序排列返回结果

---

## 5. 性能数据（300I DUO）

> 本节数据来自对 8 个消融变体的逐一测量（两档动态 shape `{1280, 4096}`，`max_length=1280`，每个变体 3 次取中位数）。

### 端到端推理性能（生产配置，`max_length=1280`）

| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 3.36 |
| Model resize | 0.46 |
| Model predict | 95.16 |
| Postprocess | 0.14 |
| **总耗时** | **99.12** |

---

## 6. 常见问题

### Q1: ONNX 导出时报 `IndexError: tuple index out of range`

**现象：** 导出时 `masking_utils.sdpa_mask` 报错

**原因：** Transformers v4.50+ 的 Qwen3 模型在 JIT tracing 时，`_preprocess_mask_arguments` 返回的 `q_length` 变为 0 维 Tensor，导致向后兼容检查失败

**解决方案：** 导出脚本已内置 monkey-patch，自动将 0 维 Tensor 转为 int。如仍有问题，优先对齐本文档的 transformers/torch 版本。

### Q2: MSLite 推理结果恒为常数或全零

**现象：** ONNX 推理正常，但 MSLite 输出恒为常数（如 0.5）或全零

**原因：** FP16 精度下 attention mask 的极小值溢出

**解决方案：** 当前 `config.ini` 采用 `allow_mix_precision` + `op_fp32.json` mixlist 方案，把 `RealDiv`、`SquareSumV1` 等 RmsNorm 链路上的关键算子锁定 FP32。

### Q3: 动态分档不会自动 pad 命中档位

**现象：** 输入 `seq_len=1201`，期望自动 pad 到 1280 并命中 1280 档位

**原因：** `ge.dynamicDims` 只是“允许的 shape 列表”，不是自动 shape 变换器。是否 pad、如何 pad 属于业务语义，必须由推理侧实现。

**解决方案：** 业务侧或推理脚本实现“选档 + pad”，并保证传入的 shape 精确等于某个档位。动态分档 MindIR 每次推理前需 `model.resize(inputs, dims)`（本脚本已内置）。

### Q4: 转换/推理失败但终端报错不全

**现象：** 终端只看到类似 `"[ERROR] FE(1087713,converter_lite):..."` 或 `"[ERROR] LITE(1087713,...,converter_lite):..."`，缺少更详细的根因信息

**解决方案：** 从错误行中提取进程号（如 `1087713`），查看 `~/ascend/log/debug/plog/plog-1087713` 获取更完整的 plog 以定位根因。

### Q5: 输入 dtype 不匹配

**现象：** 报错 `required 34, given 35`

**原因：** MindIR 期望 int32，传入了 int64

**解决方案：** MSLite 推理脚本中已自动转换 dtype 为 int32。

### Q6: 滑动注意力层推理结果异常（排序错误）

**现象：** 使用 `is_causal=True` 的参考 SDPA 时，滑动注意力层忽略 `sliding_window`，导致文档排序错误

**原因：** 滑窗语义必须通过显式 banded mask（对角带宽 `sliding_window`）表达，`is_causal` 不携带窗口信息

**解决方案：** 导出脚本使用 `_build_sliding_window_additive_mask` 构造 banded 滑窗 mask；生产版本进一步将滑动层路由到 PFA（banded bool mask）。

### Q7: 文档数量超过 64 个

**现象：** 超过 MAX_DOCS 限制

**解决方案：** 模型自动执行分块处理，每块最多 64 个文档，通过加权融合得到最终排序。

### Q8: 转换时报 `ConvertGraphToOm failed` / NULL pointer

**现象：** 转换带 `sparse_mode` 属性的 PFA 时失败

**原因：** GE 不支持 PFA 上的 `sparse_mode/pre_tokens/next_tokens` 属性组合（即使语义上 pre_tokens=sliding_window 正确）

**解决方案：** 生产版本使用 banded bool mask 表达滑窗（不传 sparse 属性），详见第 6 节优化要点。

### Q9: 转换时 PromptFlashAttention Custom op 触发 `pretokens should larger than Qs`

**现象：** 转换阶段 GE 把 PromptFlashAttention 节点的 `sparse_mode/pre_tokens/next_tokens`（默认 0/0/0）当真值传递，PFA tiling 触发 `pretokens should larger than Qs` 断言

**原因：** 导出脚本在 `_CannPromptFlashAttention.symbolic` 中无条件把 `sparse_mode/pre_tokens/next_tokens` 写进 ONNX 节点属性。当这三个值均为 0 时，GE 仍按"启用 sparse 模式"解释，触发 tiling 断言

**解决方案：** 已在 `export_jina_reranker_v3_5_onnx.py` 的 symbolic 中修复——只有当稀疏属性非默认时（即 sparse_mode/pre_tokens/next_tokens 不全为 0）才写入节点，与 v3 基线导出行为一致

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Jina Reranker V3.5 官方文档](https://huggingface.co/jinaai/jina-reranker-v3.5)

---

## 8. 许可证

本教程遵循 Apache 2.0 许可证。模型权重遵循 CC BY-NC 4.0 许可证。
