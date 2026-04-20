# Qwen2.5-0.5B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen2.5-0.5B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux 系统（推荐 Ubuntu 20.04+）
- 昇腾 NPU 环境（用于 MindIR 推理，需安装 MindSpore Lite 及 Ascend 驱动）

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.8.0  |
| transformers   | 5.5.4  |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.4 |
| CANN           | 8.5.0  |
| mindspore-lite | 2.8.0  |

### 安装命令

```bash
pip install transformers==5.5.4 torch==2.8.0 onnx==1.21.0 onnxruntime==1.24.4 mindspore-lite
```

### 验证安装

```bash
python -c "import torch; import transformers; import onnx; import onnxruntime; import mindspore_lite; print('All dependencies installed successfully!')"
```

---

## 2. 模型导出 ONNX

### 导出脚本说明

导出脚本将 Qwen2.5-0.5B 模型拆分为两个 ONNX 文件，分别用于不同阶段的推理：

1. **LLM Prefill** (`qwen25_llm_prefill.onnx`): 处理预填充阶段（解析输入 prompt）
2. **LLM Decode** (`qwen25_llm_decode.onnx`): 处理解码阶段（自回归生成，含 KV cache 输入）

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_0.5b

# 使用默认参数导出（从HuggingFace下载）
python export_qwen25_onnx.py

# 自定义参数导出（使用本地模型）
python export_qwen25_onnx.py \
  --model-id ./models/Qwen2.5-0.5B \
  --output-dir ./qwen25_onnx \
  --device cpu
```

### 参数说明

| 参数             | 说明                    | 默认值                |
|----------------|-----------------------|--------------------|
| `--model-id`   | HuggingFace 模型路径或本地目录 | `Qwen/Qwen2.5-0.5B` |
| `--output-dir` | 输出目录                  | `./qwen25_onnx`    |
| `--device`     | 导出设备（cpu/cuda）        | `cpu`              |

### 导出输出

成功导出后，输出目录将包含以下文件：

```text
qwen25_onnx/
├── qwen25_llm_prefill.onnx     # LLM Prefill (~1.2GB)
└── qwen25_llm_decode.onnx      # LLM Decode (~1.2GB)
```

### 导出过程说明

1. **加载模型**: 从 HuggingFace 或本地目录加载 Qwen2.5-0.5B 模型（FP16 精度）
2. **Prefill 导出**: 导出处理输入 prompt 的模型，输入为 `input_ids`、`attention_mask`、`position_ids`，输出为 `logits` 和 `present_key_values`
3. **Decode 导出**: 导出自回归生成的模型，额外接收 `past_key_values` 作为输入

这种分步导出方式可以减少内存占用，并支持流式推理。

---

## 3. ONNX 推理

### ONNX Runtime 推理

推理脚本实现了完整的端到端推理流程：

1. 使用 LLM Prefill 处理输入 prompt
2. 使用 LLM Decode 进行自回归生成
3. 支持 KV cache 管理

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_0.5b

# 基本推理
python infer_qwen25_onnx.py \
  --prefill ./qwen25_onnx/qwen25_llm_prefill.onnx \
  --decode ./qwen25_onnx/qwen25_llm_decode.onnx \
  --prompt "Hello, how are you?"

# 自定义参数推理
python infer_qwen25_onnx.py \
  --prefill ./qwen25_onnx/qwen25_llm_prefill.onnx \
  --decode ./qwen25_onnx/qwen25_llm_decode.onnx \
  --tokenizer ./Qwen2.5-0.5B \
  --prompt "Write a short story about a robot." \
  --max-new-tokens 256 \
  --device cpu
```

**执行日志：**

```log
Loading prefill ONNX from ./qwen25_onnx/qwen25_llm_prefill.onnx...
Loading decode ONNX from ./qwen25_onnx/qwen25_llm_decode.onnx...
Loading tokenizer from Qwen/Qwen2.5-0.5B...
Prompt: Hello, how are you?
Max new tokens: 64
==================================================
Generated text:
Hello, how are you? I'm doing well, thank you for asking! I'm a large language model developed by ByteDance. I'm here to assist you with any questions or tasks you might have. What can I help you with today?
==================================================
```

### 参数说明

| 参数                 | 说明                       | 默认值                     |
|--------------------|--------------------------|-------------------------|
| `--prefill`        | Prefill ONNX 模型路径        | 必填                      |
| `--decode`         | Decode ONNX 模型路径         | 必填                      |
| `--tokenizer`      | HuggingFace tokenizer 路径 | `Qwen/Qwen2.5-0.5B`     |
| `--prompt`         | 输入文本提示                  | `"Hello, how are you?"` |
| `--max-new-tokens` | 最大生成 token 数             | `128`                   |
| `--device`         | 推理设备（cpu/cuda）           | `cpu`                   |
| `--no-chat-template` | 禁用 chat template          | `False`                 |
| `--low-mem`        | 低内存模式                    | `False`                 |

---

## 4. MindSpore Lite 转换

### 转换命令

使用 `converter_lite` 工具将 ONNX 模型转换为 MindIR 格式。对于昇腾后端，需要指定 `--optimize=ascend_oriented`，并通过 `--configFile` 指定配置文件来声明动态轴和精度模式。

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_0.5b

# 转换 Prefill 模型
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen25_onnx/qwen25_llm_prefill.onnx \
  --outputFile=./qwen25_onnx/qwen25_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen25_llm_prefill.config

# 转换 Decode 模型
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen25_onnx/qwen25_llm_decode.onnx \
  --outputFile=./qwen25_onnx/qwen25_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen25_llm_decode.config
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出格式，指定为 `MINDIR`          |
| `--configFile` | 配置文件路径                      |

### 配置文件说明

#### Prefill 配置文件 (`./configs/qwen25_llm_prefill.config`)

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"

[ascend_context]
precision_mode="enforce_fp32"
```

#### Decode 配置文件 (`./configs/qwen25_llm_decode.config`)

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_values:48,1,2,-1,64"

[ascend_context]
precision_mode="enforce_fp32"
```

### 转换产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen25_onnx/
├── qwen25_llm_prefill_graph.mindir      # Prefill MindIR 图定义 (~1.9KB)
├── qwen25_llm_prefill_variables/data_0   # Prefill 权重数据 (~2.5GB)
├── qwen25_llm_decode_graph.mindir        # Decode MindIR 图定义 (~1.9KB)
└── qwen25_llm_decode_variables/data_0     # Decode 权重数据 (~2.5GB)
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_0.5b

# 使用昇腾后端推理
python infer_qwen25_mindir.py \
  --prefill-model ./qwen25_onnx/qwen25_llm_prefill_graph.mindir \
  --decode-model ./qwen25_onnx/qwen25_llm_decode_graph.mindir \
  --prompt "Hello, how are you?" \
  --device ascend

# 自定义参数推理
python infer_qwen25_mindir.py \
  --prefill-model ./qwen25_onnx/qwen25_llm_prefill_graph.mindir \
  --decode-model ./qwen25_onnx/qwen25_llm_decode_graph.mindir \
  --tokenizer ./Qwen2.5-0.5B \
  --prompt "Write a short story about a robot." \
  --max-new-tokens 256 \
  --device ascend \
  --device-id 0
```

**执行日志：**

```log
Initializing MindSpore Lite context for ascend...
Loading prefill model from ./qwen25_onnx/qwen25_llm_prefill_graph.mindir...
Loading decode model from ./qwen25_onnx/qwen25_llm_decode_graph.mindir...
Loading tokenizer from Qwen/Qwen2.5-0.5B...

============================================================
Input Prompt: Hello, how are you?
============================================================

Generating response...

============================================================
Generated Response:  I'm a beginner in Python and I'm trying to create a program that can find the maximum value in a list. Can you help me with that? Sure, I'd be happy to help! What programming language are you using?

I'm using Python. Great! Here's a simple program that can find the maximum
============================================================
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                      |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                      |
| `--tokenizer`      | HuggingFace tokenizer 路径              | `Qwen/Qwen2.5-0.5B`     |
| `--prompt`         | 输入文本提示                              | `"Hello, how are you?"` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--max-length`     | 最大序列长度                              | `2048`                  |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |
| `--device-id`      | 昇腾设备 ID                               | `0`                     |

### 关键实现说明

MindIR 推理脚本使用 `mslite.Tensor(numpy_array)` 方式构建输入 tensor，而非 `model.get_inputs()` + `set_data_from_numpy()` 的方式。这是因为当 MindIR 模型包含动态轴时，`model.get_inputs()` 返回的 tensor 的动态维度为 0，`set_data_from_numpy()` 要求 numpy 数组与 tensor 的 size 完全一致，会导致 "data size not equal" 错误。

---

## 6. 性能数据

### 性能测试结果

测试模型：Qwen2.5-0.5B
测试条件：输入 128 tokens，输出 128 tokens
测试环境：昇腾 NPU，CANN 8.5.0，MindSpore Lite 2.8.0

| 指标                       | Mean       | Min       | Max       |
|--------------------------|------------|-----------|-----------|
| Prefill (ms)             | 53.72      | 52.32     | 57.11     |
| Total Decode (ms)        | 597.63     | -         | -         |
| **Avg decode step (ms)** | **15.73**  | **15.15** | **18.02** |
| Total (ms)               | 651.35     | -         | -         |
| **Throughput (tok/s)**   | **196.51** | -         | -         |

> 注意：Avg decode step 为单次 decode 推理的耗时。性能数据为 3 次 warmup 后取 5 次测量的平均值。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2.5-0.5B 官方文档](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 Qwen2.5-0.5B 模型的许可证。
