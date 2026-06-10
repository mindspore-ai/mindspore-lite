# Qwen3-Reranker-4B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-Reranker-4B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本。

## 概览

Qwen3-Reranker-4B 是通义千问团队推出的**重排序模型**（Cross-Encoder Reranker），用于对检索结果进行精排。输入 Query 与 Document，输出相关性分数。

本实现采用二分类式打分：

- 取最后一个 token 位置的 logits
- 取 `yes/no` 两个 token 的 logits 做 softmax
- `P(yes)` 作为相关性分数

## 模型架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `intermediate_size` | 9728 |
| `vocab_size` | 151669 |
| `max_position_embeddings` | 40960 |

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
python export_qwen3_reranker_4b_onnx.py \
    --model-id ./Qwen/Qwen3-Reranker-4B \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `Qwen/Qwen3-Reranker-4B` |
| `--output-dir` | 导出输出目录 | `./onnx` |
| `--max-length` | 最大序列长度 | `8192` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |

### 导出产物

- `qwen3_reranker_4b.onnx`：模型图定义（~1.9MB）
- 外部权重文件：由 `torch.onnx.export` 自动生成的独立权重文件（~7.5GB）

## 模型 I/O 说明

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`

**输出：**

- `logits`：`float16`，形状 `(batch, seq_len, vocab_size)`

推理脚本取最后一个 token 位置的 logits，提取 `yes`（token 9693）和 `no`（token 2152）的 logits，通过 softmax 计算 `P(yes)` 作为相关性分数。

## 输入格式

输入按以下模板拼接后再 tokenize：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
```

## MindSpore Lite 模型转换

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./onnx/qwen3_reranker_4b.onnx \
    --outputFile=./onnx/qwen3_reranker_4b \
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
python infer_qwen3_reranker_4b_mslite.py \
    --model ./onnx/qwen3_reranker_4b_graph.mindir \
    --tokenizer ./Qwen/Qwen3-Reranker-4B \
    --max-length 8192 \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | MindIR 模型路径 | 必填 |
| `--tokenizer` | Tokenizer 路径或 ID | `Qwen/Qwen3-Reranker-4B` |
| `--max-length` | 最大序列长度 | `8192` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device ID | `0` |

## 性能数据

### 推理结果示例

```text
Running reranking inference...
Pair 1: seq_len=100, latency=192.885ms
Pair 2: seq_len=100, latency=131.988ms

Reranking scores:

[1] Score: 0.9771
Query: What is the capital of China?
Document: The capital of China is Beijing.

[2] Score: 0.9873
Query: Explain gravity
Document: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.
```

### 推理性能（Atlas 300I Duo）

| 指标 | 值 |
|---|---|
| 单条 Rerank 延迟（首条） | ~193ms |
| 单条 Rerank 延迟（稳态） | ~132ms |
| 序列长度 | 100 tokens |

### 模型大小

| 组件 | 大小 |
|---|---|
| ONNX 图 | ~1.9MB |
| ONNX 外部数据 | ~7.5GB |
| MindIR Variables | ~25GB |

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-Reranker-4B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Reranker-4B)
- [Qwen3-Reranker-4B HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3 模型的许可证要求。
