# Qwen3-Embedding-4B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-Embedding-4B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本。

## 概览

Qwen3-Embedding-4B 是通义千问团队推出的**文本嵌入模型**（Text Embedding），用于将文本映射为 2560 维稠密向量，支持语义搜索、文本检索、聚类等下游任务。

本实现采用 last-token pooling + L2 normalization：

- 取每个序列最后一个非 padding token 的 hidden state
- 对输出做 L2 归一化，得到单位向量
- 通过点积计算余弦相似度

## 模型架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `intermediate_size` | 9728 |
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
python export_qwen3_embedding_4b_onnx.py \
    --model-id ./Qwen/Qwen3-Embedding-4B \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `Qwen/Qwen3-Embedding-4B` |
| `--output-dir` | 导出输出目录 | `./onnx` |
| `--max-length` | 最大序列长度 | `8192` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |

### 导出产物

- `qwen3_embedding_4b.onnx`：模型图定义（~1.9MB）

## 模型 I/O 说明

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`

**输出：**

- `embeddings`：`float16`，形状 `(batch, hidden_size)`，即 `(batch, 2560)`

推理脚本对输出做 L2 归一化后，通过点积计算余弦相似度。

## 输入格式

Query 输入按以下模板拼接后再 tokenize：

```text
Instruct: {task_description}
Query: {query_text}
```

Document 输入直接使用原始文本，无需添加指令前缀。

## MindSpore Lite 模型转换

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./onnx/qwen3_embedding_4b.onnx \
    --outputFile=./onnx/qwen3_embedding_4b \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件示例

#### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = force_fp32
```

## MindSpore Lite 推理

```bash
python infer_qwen3_embedding_4b_mslite.py \
    --model ./onnx/qwen3_embedding_4b_graph.mindir \
    --tokenizer ./Qwen/Qwen3-Embedding-4B \
    --max-length 512 \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | MindIR 模型路径 | 必填 |
| `--tokenizer` | Tokenizer 路径或 ID | `Qwen/Qwen3-Embedding-4B` |
| `--max-length` | 最大序列长度 | `512` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device ID | `0` |

## 性能数据

### 推理结果示例

```text
Encoding queries...
Batch 1: 1 texts, 157.454ms
Batch 2: 1 texts, 92.790ms
Query embeddings shape: (2, 2560)

Encoding documents...
Batch 1: 1 texts, 91.995ms
Batch 2: 1 texts, 91.896ms
Batch 3: 1 texts, 91.634ms
Batch 4: 1 texts, 74.568ms
Document embeddings shape: (4, 2560)

Similarity scores:

Query: What is the capital of China?
  [0.7573] The capital of China is Beijing.
  [0.2424] The Eiffel Tower is located in Paris, France.
  [0.1619] Python is a popular programming language.
  [0.0982] Gravity is a fundamental force of nature that attracts two bodies towards each other.

Query: Explain gravity
  [0.6035] Gravity is a fundamental force of nature that attracts two bodies towards each other.
  [0.1349] Python is a popular programming language.
  [0.1064] The Eiffel Tower is located in Paris, France.
  [0.0266] The capital of China is Beijing.
```

### 推理性能（Atlas 300I Duo）

| 指标 | 值 |
|---|---|
| 单条 Query 编码延迟（首条） | ~157ms |
| 单条 Query 编码延迟（稳态） | ~93ms |
| 单条 Document 编码延迟 | ~92ms |
| 嵌入维度 | 2560 |

## 常见问题

### 转换时出现 Warning

MindSpore Lite 模型转换过程中出现的 Warning 日志可以忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 即表示转换成功。

### 导出时内存不足

4B 模型导出需要约 16GB 内存。如果内存不足，可以减小 `--max-length` 或关闭其他程序。

### 转换时 ONNX 解析失败

对于大于 2GB 的 ONNX 文件，需要使用外部数据格式。导出脚本会自动处理外部数据的合并。

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-Embedding-4B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Embedding-4B)
- [Qwen3-Embedding-4B HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3 模型的许可证要求。
