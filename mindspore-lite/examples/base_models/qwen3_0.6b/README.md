# Qwen3-0.6B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-0.6B 纯文本模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3-0.6B 是 Qwen 系列中体积最小的对话模型，适合作为轻量级对话 / 抬头组件部署在 Atlas 300I Duo 等 NPU 上。模型在导出时被拆分为两个 ONNX：

1. **LLM Prefill**（`qwen3_llm_prefill.onnx`）：一次性处理完整 prompt，输出首 token logits 与初始 KV cache
2. **LLM Decode**（`qwen3_llm_decode.onnx`）：基于 KV cache 做自回归增量生成

## 模型架构

Qwen3-0.6B 是一个 28 层的 decoder-only Transformer：

| 项目                | 值     |
|-------------------|-------|
| 层数                | 28    |
| num_attention_heads | 16    |
| num_key_value_heads | 8（GQA）|
| head_dim          | 128   |
| hidden_size       | 1024  |
| vocab_size        | 151936 |

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0 |
| transformers   | 4.51.3 |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 8.5    |
| mindspore-lite | 2.9.0  |

```bash
pip install transformers==4.51.3 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

### 权重准备

从 HuggingFace 下载 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 权重，解压后放到本地目录（本文以 `./Qwen3-0.6B` 为例）。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu
```

### 参数说明

| 参数            | 说明                  | 默认值                  |
|---------------|-----------------------|------------------------|
| `--model-id`  | HuggingFace 模型路径或本地目录 | `Qwen/Qwen3-0.6B`     |
| `--output-dir`| 输出目录                | `./qwen3_onnx`         |
| `--device`    | 导出设备（cpu / cuda）    | `cpu`                  |

### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill.onnx   # Prefill（一次性处理 prompt）
└── qwen3_llm_decode.onnx    # Decode（自回归增量生成）
```

### ONNX 模型输入输出 Shape

**LLM Prefill** — `qwen3_llm_prefill.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, seq_len)`     | int64   | 输入 token IDs     |
| 输入 | `attention_mask` | `(batch, seq_len)`     | int64   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, seq_len)`     | int64   | 位置 ID            |
| 输出 | `logits`         | `(batch, seq_len, 151936)` | float32 | 下一个 token 预测 logits |
| 输出 | `past_kv`        | `(56, batch, 8, seq_len, 128)` | float32 | 初始 KV cache（56 = 28 × 2） |

**LLM Decode** — `qwen3_llm_decode.onnx`

| 方向  | 名称               | Shape                          | Dtype   | 说明                     |
|-----|------------------|--------------------------------|---------|------------------------|
| 输入 | `input_ids`      | `(batch, 1)`                   | int64   | 单步 token               |
| 输入 | `attention_mask` | `(batch, total_seq_len)`       | int64   | 累积注意力掩码              |
| 输入 | `position_ids`   | `(batch, 1)`                   | int64   | 单步位置 ID              |
| 输入 | `past_key_values` | `(56, batch, 8, past_seq_len, 128)` | float32 | 上一步 KV cache          |
| 输出 | `logits`         | `(batch, 1, 151936)`           | float32 | 单步 logits             |
| 输出 | `past_kv`        | `(56, batch, 8, past_seq_len+1, 128)` | float32 | 更新后的 KV cache        |

---

## 3. ONNX 转 MindIR

### 转换命令

```bash
Convert=mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# Prefill 转换（动态 shape，无需分档）
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./qwen3_0.6b_prefill.ini

# Decode 转换（KV cache 13 档分档）
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_decode.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./qwen3_0.6b_decode.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出格式（MINDIR）                |
| `--configFile` | 配置文件路径（**所有模型都必须指定**）    |

### 配置文件

`qwen3_0.6b_prefill.ini`（Prefill 用）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

`qwen3_0.6b_decode.ini`（Decode 用，13 档）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_values:56,1,8,-1,128"
ge.dynamicDims="17,16;33,32;65,64;97,96;129,128;193,192;257,256;385,384;513,512;769,768;1025,1024;1537,1536;2049,2048"
```

- `past_key_values:56,1,8,-1,128`：56 = 28 层 × 2（K+V），1 = batch，8 = num_kv_heads，-1 = 动态 KV 长度，128 = head_dim
- `ge.dynamicDims`：13 个分档对应 `past_kv_len ∈ {16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048}`，覆盖 16 到 2048 的 KV cache 长度

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen3_onnx/
├── qwen3_llm_prefill_graph.mindir           # Prefill 主图
├── qwen3_llm_prefill_variables/             # Prefill 权重
│   └── data_0
├── qwen3_llm_decode_graph.mindir            # Decode 主图（含 13 档）
└── qwen3_llm_decode_variables/              # Decode 权重
    └── data_0
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_0.6b_mindir.py \
  --prefill-model ./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --decode-model ./qwen3_onnx/qwen3_llm_decode_graph.mindir \
  --tokenizer ./Qwen3-0.6B \
  --prompt "你好，请介绍一下你自己。" \
  --max-new-tokens 128 \
  --decode-buckets "16,32,64,96,128,192,256,384,512,768,1024,1536,2048" \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|---------------------------|
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                       |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                       |
| `--tokenizer`      | HuggingFace tokenizer 路径              | `Qwen/Qwen3-0.6B-Instruct` |
| `--prompt`         | 输入文本                                  | `"你好，请介绍一下你自己。"`   |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                     |
| `--max-length`     | 模型最大上下文长度                            | `2048`                    |
| `--decode-buckets` | Decode 模型的 KV cache 分档列表（逗号分隔） | `16,32,64,...,2048`（13 档） |
| `--no-chat-template` | 关闭 chat template，进入 raw completion 模式 | 关闭                       |
| `--device`         | 推理设备（ascend/cpu）                    | `ascend`                  |
| `--device-id`      | Ascend 设备 ID                          | `0`                       |
| `--force-no-pre-alloc` | 关闭输出 buffer 预分配（部分精度模式下需要）     | 关闭                       |

> `--decode-buckets` 必须与转换时 `ge.dynamicDims` 中的 `past_kv_len` 列表**完全一致**，否则会触发 `aclmdlSetInputDynamicDims failed`。脚本会自动把 `past_kv` 与 `attention_mask` 补零到下一档边界，并在每步把模型追加在分档尾部的 K/V 切回真实长度。

### 推理示例输出

```text
Initializing MindSpore Lite context for ascend...
Loading prefill model from ./qwen3_onnx/qwen3_llm_prefill_graph.mindir...
Loading decode model from ./qwen3_onnx/qwen3_llm_decode_graph.mindir...
Loading tokenizer from ./Qwen3-0.6B...
[zero-copy] pre-allocated outputs: enabled (probe OK)

============================================================
Input Prompt: 你好，请介绍一下你自己。
============================================================

Running LLM prefill...
Prefill time: 123.90 ms
Running LLM decode...
Total decode time: 1497.49 ms, avg decode step: 11.79 ms, steps: 127
Total time: 1621.39 ms, throughput: 78.94 tok/s

============================================================
Generated Response: <think>
好的，用户问我的介绍。我需要先确认用户的需求是什么。可能他们想了解我的功能，或者想进行互动。我应该保持友好和专业的态度，同时提供有用的信息。

首先，我应该简要介绍我的功能，比如处理各种请求、提供帮助等。然后，可以
============================================================
```

---

## 5. 性能数据

### 测试环境

| 项目   | 配置                     |
|------|------------------------|
| 硬件   | Atlas 300I Duo（Ascend NPU） |
| 模型   | Qwen3-0.6B（28 层，8 KV head，head_dim=128） |
| 精度   | force_fp32              |
| Decode 分档 | 13 档（16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048） |
| 推理脚本 | `infer_qwen3_0.6b_mindir.py`（zero-copy + 预分配输出 buffer） |

### 各阶段推理输入 Shape 与性能

**LLM Prefill**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids` |
| input_ids Shape  | `(1, seq_len)`                |
| 输出 logits Shape | `(1, seq_len, 151936)`        |
| 输出 past_kv Shape | `(56, 1, 8, seq_len, 128)`    |
| 推理耗时        | **123.90 ms**                    |

**LLM Decode（单步）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids`, `past_key_values` |
| input_ids Shape  | `(1, 1)`                        |
| attention_mask Shape | `(1, past_kv_len+1)`         |
| past_key_values Shape | `(56, 1, 8, past_kv_len, 128)` |
| 输出 logits Shape | `(1, 1, 151936)`               |
| 单步平均耗时      | **11.79 ms**                     |

### 端到端推理性能（128 tokens 生成）

| 场景                          | Prefill (ms) | Avg decode (ms) | Total (ms) | 吞吐 (tok/s) |
|-----------------------------|--------------|-----------------|------------|--------------|
| 中文对话：你好，请介绍一下你自己。           | 123.90       | 11.79           | 1621.39    | **78.94**    |
| 英文 QA：What is the capital of France? | 140.56       | 11.90           | 1652.16    | **77.47**    |
| 代码生成：Write a short Python function that reverses a string. | 195.91       | 19.32           | 2649.09    | **48.32**    |

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-0.6B 官方文档](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 7. 许可证

本教程遵循 Qwen3-0.6B 模型的许可证。
