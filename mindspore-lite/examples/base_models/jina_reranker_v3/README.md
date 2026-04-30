# Jina Reranker V3 Listwise ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Jina Reranker V3 模型以原生 Listwise 架构导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。同时支持 Listwise 和 Pointwise 两种打分模式。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0  |
| transformers   | 5.6.2  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 9.0    |
| mindspore-lite | 2.8.0  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/jina_reranker_v3_listwise

python export_jina_reranker_v3_listwise_onnx.py \
  --model-id /path/to/jina-reranker-v3 \
  --output-dir ./onnx \
  --max-length 8192 \
  --device cpu
```

### 参数说明

| 参数             | 说明                        | 默认值                    |
|----------------|---------------------------|------------------------|
| `--model-id`   | HuggingFace 模型路径或本地目录     | `jinaai/jina-reranker-v3` |
| `--output-dir` | 输出目录                      | `./onnx`               |
| `--max-length` | 最大序列长度                    | `8192`                 |
| `--device`     | 导出设备（cpu/cuda）            | `cpu`                  |

### 产出

```log
onnx/
└── jina_reranker_v3_listwise.onnx
```

### 模型架构说明

导出的 ONNX 模型包含以下核心组件：

- **Qwen3-0.6B Backbone**：28 层 Transformer，因果自注意力机制
- **MLP Projector**：1024→512→512 维度映射（带 ReLU 激活）
- **Cosine Similarity**：计算 query 与各 document 的相关性分数

**ONNX 模型输入：**

| 输入名                  | 形状                  | 类型     | 说明                          |
|----------------------|---------------------|--------|-----------------------------|
| `input_ids`          | (batch, seq_len)    | int64  | 完整 listwise prompt 的 token IDs |
| `attention_mask`     | (batch, seq_len)    | int64  | 注意力掩码                       |
| `doc_token_indices`  | (batch, num_docs)   | int64  | `<\|embed_token\|>` 的位置索引    |
| `query_token_index`  | (batch, 1)          | int64  | `<\|rerank_token\|>` 的位置索引   |

**ONNX 模型输出：**

| 输出名      | 形状                  | 类型    | 说明              |
|----------|---------------------|-------|-----------------|
| `scores` | (batch, num_docs)   | float | 每个 document 的相关性分数 |

### Listwise vs Pointwise 模式

两种模式使用同一个 ONNX 模型，区别在于输入格式：

| 模式          | 输入格式                       | 特点               |
|-------------|----------------------------|------------------|
| **Listwise**  | 1 个 query + N 个 doc 在同一上下文窗口 | 跨文档交互，精度更高       |
| **Pointwise** | 1 个 query + 1 个 doc 逐对推理     | 无跨文档交互，内存占用更低    |

---

## 3. ONNX 推理

### Listwise 模式推理

```bash
python infer_jina_reranker_v3_listwise_onnx.py \
  --model-path ./onnx/jina_reranker_v3_listwise.onnx \
  --tokenizer /path/to/jina-reranker-v3 \
  --max-length 8192 \
  --mode listwise \
  --device CPU
```

**执行日志：**

```log
Loading tokenizer from jina-reranker-v3
Loading ONNX model from ./onnx_listwise/jina_reranker_v3_listwise.onnx

Running inference in listwise mode...

Reranking results (listwise mode):

[1] Score: 0.2978
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.2246
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.1897
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.1619
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1606
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1701
Document: Basketball is one of the most popular sports in the United States....

Inference time: 3.232s
```

### Pointwise 模式推理

```bash
python infer_jina_reranker_v3_listwise_onnx.py \
  --model-path ./onnx/jina_reranker_v3_listwise.onnx \
  --tokenizer /path/to/jina-reranker-v3 \
  --max-length 8192 \
  --mode pointwise \
  --device CPU
```

**执行日志：**

```log
Loading tokenizer from jina-reranker-v3
Loading ONNX model from ./onnx_listwise/jina_reranker_v3_listwise.onnx

Running inference in pointwise mode...

Reranking results (pointwise mode):

[1] Score: 0.3454
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.3388
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.3152
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.2508
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1041
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1387
Document: Basketball is one of the most popular sports in the United States....

Inference time: 23.823s
```

### 参数说明

| 参数             | 说明                       | 默认值                              |
|----------------|--------------------------|----------------------------------|
| `--model-path` | ONNX 模型路径               | `./onnx/jina_reranker_v3_listwise.onnx` |
| `--tokenizer`  | HuggingFace tokenizer 路径 | `jinaai/jina-reranker-v3`        |
| `--max-length` | 最大序列长度                  | `8192`                           |
| `--mode`       | 打分模式：listwise / pointwise | `listwise`                       |
| `--device`     | 推理设备（CPU/CUDA）          | `CPU`                            |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

$Converter --fmk=ONNX \
  --modelFile=onnx/jina_reranker_v3_listwise.onnx \
  --outputFile=onnx/jina_reranker_v3_listwise \
  --optimize=ascend_oriented \
  --configFile=config.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |

### 配置文件

`config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

此配置用于避免 FP16 精度下 attention mask 溢出问题（详见常见问题 Q2）。

**注意：** `force_fp32` 与 `enforce_fp32` 是不同的配置项。`enforce_fp32` 仅限制算子输入为 FP32，但中间计算仍可能使用 FP16；`force_fp32` 强制所有计算使用 FP32，确保精度一致性。对于 attention mask 溢出问题，应使用 `force_fp32`。

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
onnx/
├── jina_reranker_v3_listwise_graph.mindir      # MindIR 图定义
├── jina_reranker_v3_listwise_variables/data_0   # 权重数据
└── jina_reranker_v3_listwise.onnx               # 原始 ONNX 模型
```

---

## 5. MindSpore Lite 推理

### Listwise 模式推理

```bash
python infer_jina_reranker_v3_listwise_mslite.py \
  --model-path ./onnx/jina_reranker_v3_listwise_graph.mindir \
  --tokenizer /path/to/jina-reranker-v3 \
  --max-length 8192 \
  --device ascend \
  --mode listwise
```

**执行日志：**

```log
Loading tokenizer from jina-reranker-v3
Initializing MindSpore Lite context for ascend...
Loading model from onnx_listwise/jina_reranker_v3_listwise_graph.mindir...
WARNING:root:Ascend custom operator path not found

Running inference in listwise mode...

Reranking results (listwise mode):

[1] Score: 0.2978
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.2246
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.1897
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.1619
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1606
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1701
Document: Basketball is one of the most popular sports in the United States....

Inference time: 0.856s
```

### Pointwise 模式推理

```bash
python infer_jina_reranker_v3_listwise_mslite.py \
  --model-path ./onnx/jina_reranker_v3_listwise_graph.mindir \
  --tokenizer /path/to/jina-reranker-v3 \
  --max-length 8192 \
  --device ascend \
  --mode pointwise
```

**执行日志：**

```log
Loading tokenizer from jina-reranker-v3
Initializing MindSpore Lite context for ascend...
Loading model from onnx_listwise/jina_reranker_v3_listwise_graph.mindir...
WARNING:root:Ascend custom operator path not found

Running inference in pointwise mode...

Reranking results (pointwise mode):

[1] Score: 0.3453
Document: Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells...

[2] Score: 0.3386
Document: 绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。...

[3] Score: 0.3150
Document: Studies show that drinking green tea regularly can improve brain function and boost metabolism....

[4] Score: 0.2507
Document: Le thé vert est riche en antioxydants et peut améliorer la function cérébrale....

[5] Score: -0.1041
Document: El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro....

[6] Score: -0.1386
Document: Basketball is one of the most popular sports in the United States....

Inference time: 0.740s
```

### 参数说明

| 参数             | 说明                                    | 默认值                     |
|----------------|---------------------------------------|-------------------------|
| `--model-path` | MindIR 模型路径（`*_graph.mindir`）         | 必填                      |
| `--tokenizer`  | HuggingFace tokenizer 路径              | `jinaai/jina-reranker-v3` |
| `--max-length` | 最大序列长度                                | `8192`                  |
| `--device`     | 推理设备（ascend/cpu）                      | `cpu`                   |
| `--device-id`  | Ascend 设备 ID                          | `0`                     |
| `--mode`       | 打分模式：listwise / pointwise             | `listwise`              |

### 分块策略说明

当文档数量较多或总长度超过上下文窗口时，Listwise 模式自动执行分块处理：

1. 将文档按 token 长度分块，每块不超过 `max_length - 2 * query_length`
2. 每块独立推理，获取该块内各文档的分数
3. 以各块最高分数的归一化值作为权重，用于跨块分数归一化
4. 合并所有块的分数，按降序排列返回结果

---

## 6. 性能数据

### 性能测试结果（Atlas 800I A2）

| 模式          | 6 文档推理时间 | 说明       |
|-------------|-----------|----------|
| Listwise    | 923ms  | 单次前向推理   |
| Pointwise   | 706ms  | 6 次前向推理  |

> 注：性能数据仅供参考，实际性能取决于硬件配置和文档长度。

---

## 7. 常见问题

### Q1: ONNX 导出时报 `IndexError: tuple index out of range`

**现象：** 导出时 `masking_utils.sdpa_mask` 报错

**原因：** Transformers v4.50+ 的 Qwen3 模型在 JIT tracing 时，`_preprocess_mask_arguments` 返回的 `q_length` 变为 0 维 Tensor，导致向后兼容检查失败

**解决方案：** 导出脚本已内置 monkey-patch，自动将 0 维 Tensor 转为 int。如仍有问题，请检查 Transformers 版本是否 >= 4.45

### Q2: MSLite 推理结果恒为常数或全零

**现象：** ONNX 推理正常，但 MSLite 输出恒为常数（如 0.5）或全零

**原因：** FP16 精度下 attention mask 的极小值溢出

**解决方案：** 转换时使用 `config.ini` 配置 `force_fp32`（注意不是 `enforce_fp32`，后者仅限制输入精度，中间计算仍可能 FP16）

### Q3: 输入 dtype 不匹配

**现象：** 报错 `required 34, given 35`

**原因：** MindIR 期望 int32，传入了 int64

**解决方案：** MSLite 推理脚本中已自动转换 dtype 为 int32

### Q4: Listwise 和 Pointwise 分数差异

**现象：** 两种模式的分数数值不同

**原因：** 这是正常行为。Listwise 模式中文档之间通过 causal attention 交互，分数反映了文档间的相对关系；Pointwise 模式各文档独立评分，无跨文档交互。推荐使用 Listwise 模式以获得更准确的排序

### Q5: 文档数量超过 64 个

**现象：** 超过 MAX_DOCS 限制

**解决方案：** 模型自动执行分块处理，每块最多 64 个文档，通过加权融合得到最终排序

### Q6: MindSpore Lite 转换或 ONNX Runtime 推理报 FLOAT16 类型错误

**现象：** converter_lite 报错 `do not support data_type: 10`，或 ONNX Runtime 报错 `Type Error: Type (tensor(float16)) of output arg does not match expected type (tensor(float))`

**原因：** 模型以 float16 加载导出时，ONNX 图的类型声明和初始值均为 FLOAT16，MindSpore Lite 的 Clip 等解析器不支持 FLOAT16；仅转换初始值 dtype 不够，图中节点的输入/输出类型声明也需同步修改

**解决方案：** 导出脚本已改为以 `torch_dtype=torch.float32` 加载模型，从源头确保 ONNX 全图为 float32。请确保使用最新版本的导出脚本重新导出 ONNX 模型

### Q7: ONNX 推理时报 `INVALID_PROTOBUF: Protobuf parsing failed`

**现象：** ONNX 模型导出成功，但 ONNX Runtime 加载时报错 `Protobuf parsing failed`

**原因：** 大模型 ONNX 使用外部数据文件（`.onnx.data`），优化后 `onnx.save` 未正确更新外部数据文件，导致 `.onnx` 与 `.onnx.data` 不匹配

**解决方案：** 导出脚本已修复此问题：`onnx.load` 时加 `load_external_data=True` 加载完整数据，保存时删除旧 `.onnx.data` 并使用 `onnx.save_model` 正确写入外部数据文件。请确保使用最新版本的导出脚本重新导出

### Q8: MSLite Pointwise 模式推理结果与 ONNX 不一致

**现象：** MSLite Listwise 模式结果与 ONNX 一致，但 Pointwise 模式部分文档分数严重偏移（如无关文档得分异常高）

**原因：** MSLite 使用 `--optimize=ascend_oriented` 转换时，GE 会针对固定输入 shape 编译优化图。Pointwise 模式下每个文档单独 tokenize 导致序列长度不同，GE 无法正确适配变长输入，部分推理结果错误

**解决方案：** 推理脚本已修改为将所有文档一起 tokenize 并 padding 至等长，保证每次推理的输入 shape 一致。与原始 Pointwise 版本（`jina_reranker_v3`）的处理方式保持一致

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Jina Reranker V3 官方文档](https://huggingface.co/jinaai/jina-reranker-v3)
- [Jina Reranker V3 技术报告](https://arxiv.org/abs/2509.25085)
- [Pointwise 版本实现](../jina_reranker_v3/)

---

## 9. 许可证

本教程遵循 Apache 2.0 许可证。模型权重遵循 CC BY-NC 4.0 许可证。
