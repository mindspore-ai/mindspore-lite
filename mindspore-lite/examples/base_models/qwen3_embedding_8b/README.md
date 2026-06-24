# Qwen3-Embedding-8B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-Embedding-8B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本。

## 概览

Qwen3-Embedding-8B 是通义千问团队推出的**文本嵌入模型**（Text Embedding），用于将文本映射为 4096 维稠密向量，支持语义搜索、文本检索、聚类等下游任务。

本实现采用 last-token pooling + L2 normalization：
- 取每个序列最后一个非 padding token 的 hidden state
- 对输出做 L2 归一化，得到单位向量
- 通过点积计算余弦相似度

> **注意：** 8B 模型权重较大（~15GB），在 Atlas 300I Duo 上需使用 `allow_fp32_to_fp16` 混合精度模式以保证推理精度和显存兼容性。

## 模型架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `intermediate_size` | 12288 |
| `vocab_size` | 151665 |
| `max_position_embeddings` | 32768 |
| `tie_word_embeddings` | true |

## 环境依赖

### 依赖版本

| 软件包 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.7.1 |
| transformers | 5.9.0 |
| onnx | 1.19+ |
| numpy | - |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen3_embedding_8b_onnx.py \
    --model-id ./Qwen/Qwen3-Embedding-8B \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `Qwen/Qwen3-Embedding-8B` |
| `--output-dir` | 导出输出目录 | `./onnx` |
| `--max-length` | 最大序列长度 | `8192` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |

### 导出产物

- `qwen3_embedding_8b.onnx`：模型图定义（~1.8MB）

## 模型 I/O 说明

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`

**输出：**

- `embeddings`：`float16`，形状 `(batch, hidden_size)`，即 `(batch, 4096)`

推理脚本对输出做 L2 归一化后，通过点积计算余弦相似度。

## 输入格式

Query 输入按以下模板拼接后再 tokenize：

```text
Instruct: {task_description}
Query: {query_text}
```

Document 输入直接使用原始文本，无需添加指令前缀。

## MindSpore Lite 模型转换

> **重要：** 8B 模型必须使用 `allow_fp32_to_fp16` 混合精度模式。使用 `force_fp16` 会导致精度严重丢失（相似度分数全部集中在 0.97-0.99），使用 `force_fp32` 会导致变量体积超限（42GB > 44GB 设备显存）。

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./onnx/qwen3_embedding_8b.onnx \
    --outputFile=./onnx/qwen3_embedding_8b \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件示例

#### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

## MindSpore Lite 推理

```bash
python infer_qwen3_embedding_8b_mslite.py \
    --model ./onnx/qwen3_embedding_8b_graph.mindir \
    --tokenizer ./Qwen/Qwen3-Embedding-8B \
    --max-length 512 \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | MindIR 模型路径 | 必填 |
| `--tokenizer` | Tokenizer 路径或 ID | `Qwen/Qwen3-Embedding-8B` |
| `--max-length` | 最大序列长度 | `512` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device ID | `0` |

## 精度对齐

MindSpore Lite 推理结果与原始 PyTorch 模型对比（`allow_fp32_to_fp16` 混合精度）：

| 相似度对 | PyTorch 原模型 | MindSpore Lite | 差异 |
|---|---|---|---|
| Q1-D1 (China→Beijing) | 0.7421 | 0.7417 | 0.0004 |
| Q1-D3 (China→Eiffel) | 0.2048 | 0.2047 | 0.0001 |
| Q2-D2 (Gravity→Gravity) | 0.6006 | 0.6011 | 0.0005 |
| Q2-D4 (Gravity→Python) | 0.1471 | 0.1472 | 0.0001 |

排序完全一致，精度误差 < 0.001。

## 性能数据

### 推理结果示例

```
Encoding queries...
Batch 1: 1 texts, 234.553ms
Batch 2: 1 texts, 118.113ms
Query embeddings shape: (2, 4096)

Encoding documents...
Batch 1: 1 texts, 113.114ms
Batch 2: 1 texts, 115.266ms
Batch 3: 1 texts, 115.542ms
Batch 4: 1 texts, 115.507ms
Document embeddings shape: (4, 4096)

Similarity scores:

Query: What is the capital of China?
  [0.7417] The capital of China is Beijing.
  [0.2047] The Eiffel Tower is located in Paris, France.
  [0.1223] Python is a popular programming language.
  [0.0906] Gravity is a fundamental force of nature that attracts two bodies towards each other.

Query: Explain gravity
  [0.6011] Gravity is a fundamental force of nature that attracts two bodies towards each other.
  [0.1472] Python is a popular programming language.
  [0.0843] The Eiffel Tower is located in Paris, France.
  [0.0786] The capital of China is Beijing.
```

### 推理性能（Atlas 300I Duo）

| 指标 | 值 |
|---|---|
| 单条 Query 编码延迟（首条） | ~235ms |
| 单条 Query 编码延迟（稳态） | ~118ms |
| 单条 Document 编码延迟（稳态） | ~115ms |
| 嵌入维度 | 4096 |

### 模型大小

| 组件 | 大小 |
|---|---|
| 原始权重（safetensors） | ~15GB |
| ONNX 图 | ~1.8MB |
| MindIR Variables（混合精度） | ~29GB |

## 目录结构

```
qwen3_embedding_8b/
├── export_qwen3_embedding_8b_onnx.py   # ONNX 导出脚本
├── infer_qwen3_embedding_8b_mslite.py  # MindSpore Lite 推理脚本
├── README.md                             # 本说明
├── configs/
│   └── config.ini                        # MindSpore Lite 转换配置
├── Qwen/
│   └── Qwen3-Embedding-8B/              # 模型权重目录
└── onnx/
    ├── qwen3_embedding_8b.onnx          # ONNX 模型图
    ├── qwen3_embedding_8b_graph.mindir  # MindIR 模型
    └── qwen3_embedding_8b_variables/    # MindIR 权重
```

## 常见问题

### 转换时出现 Warning

MindSpore Lite 模型转换过程中出现的 Warning 日志可以忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 即表示转换成功。

### 推理加载模型失败（aclmdlLoadFromMem failed）

8B 模型变量约 29GB，需确保 Ascend 设备有足够显存（≥44GB）。若使用 `force_fp32` 模式，变量将膨胀至 ~42GB，可能导致加载失败。建议使用 `allow_fp32_to_fp16` 混合精度模式。

### force_fp16 模式精度异常

使用 `force_fp16` 会强制所有计算为 fp16，导致 8B 模型的 36 层累积误差过大，相似度分数全部集中在 0.97-0.99 范围内，无法区分。必须使用 `allow_fp32_to_fp16` 混合精度模式。

### 导出时内存不足

8B 模型导出需要约 32GB 系统内存。如果内存不足，可以减小 `--max-length` 或关闭其他程序。

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-Embedding-8B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Embedding-8B)
- [Qwen3-Embedding-8B HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3 模型的许可证要求。
