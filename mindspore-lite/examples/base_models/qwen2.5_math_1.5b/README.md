# Qwen2.5-Math-1.5B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen2.5-Math-1.5B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本。

## 概览

Qwen2.5-Math-1.5B 是通义千问团队推出的**数学专用语言模型**，基于 Qwen2 架构，专为数学推理和理解任务设计。支持数学计算、公式推导、数学问题求解等任务。

本实现采用 Prefill + Decode 分离的推理架构：

- **Prefill 阶段**：处理完整输入序列，初始化 KV Cache
- **Decode 阶段**：逐 token 自回归生成，复用 KV Cache

## 模型架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen2ForCausalLM |
| `hidden_size` | 1536 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 2 (GQA) |
| `intermediate_size` | 8960 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 4096 |
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
python export_qwen2_5_math_1_5b_onnx.py \
    --model-id ./Qwen2.5-Math-1.5B \
    --output-dir ./qwen2_5_math_1_5b_onnx \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `./Qwen2.5-Math-1.5B` |
| `--output-dir` | 导出输出目录 | `./qwen2_5_math_1_5b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |

### 导出产物

```text
qwen2_5_math_1_5b_onnx/
├── prefill/
│   └── qwen2_5_math_1_5b_llm_prefill.onnx   # Prefill 模型 (~752KB)
└── decode/
    └── qwen2_5_math_1_5b_llm_decode.onnx     # Decode 模型 (~424KB)
```

## MindSpore Lite 模型转换

```bash
# 转换 Prefill
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen2_5_math_1_5b_onnx/prefill/qwen2_5_math_1_5b_llm_prefill.onnx \
    --outputFile=./qwen2_5_math_1_5b_onnx/prefill/qwen2_5_math_1_5b_llm_prefill \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Decode
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen2_5_math_1_5b_onnx/decode/qwen2_5_math_1_5b_llm_decode.onnx \
    --outputFile=./qwen2_5_math_1_5b_onnx/decode/qwen2_5_math_1_5b_llm_decode \
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
python infer_qwen2_5_math_1_5b_mslite.py \
    --prefill-model ./qwen2_5_math_1_5b_onnx/prefill/qwen2_5_math_1_5b_llm_prefill_graph.mindir \
    --decode-model ./qwen2_5_math_1_5b_onnx/decode/qwen2_5_math_1_5b_llm_decode_graph.mindir \
    --tokenizer ./Qwen/Qwen2.5-Math-1.5B \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 模型路径 | 必填 |
| `--decode-model` | Decode MindIR 模型路径 | 必填 |
| `--tokenizer` | Tokenizer 路径或 ID | `Qwen/Qwen2.5-Math-1.5B` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device ID | `0` |

## 性能数据

### 推理结果示例

```text
Input: 你好，请介绍一下你自己。

[Input Info]
  actual_input_len:  33
  padded_input_len:  40 (gear)

[Prefill] 115.80 ms
[Decode]  127 steps, avg 41.70 ms/step

Output: 嗨，我是你的assistant。很高兴能帮助你。...
```

### 推理性能（Atlas 300I Duo）

| 指标 | 值 |
|---|---|
| Prefill 延迟 | 115.80 ms |
| Decode 平均延迟（稳态） | ~37 ms/step |
| Decode 首步延迟 | ~717 ms |
| 吞吐量 | 23.65 tok/s |
| KV Cache 长度 | 512 |

### 模型大小

| 组件 | 大小 |
|---|---|
| 原始权重（safetensors） | ~2.9GB |
| ONNX Prefill 图 | ~752KB |
| ONNX Decode 图 | ~424KB |

## 常见问题

### 转换时出现 Warning

MindSpore Lite 模型转换过程中出现的 Warning 日志可以忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 即表示转换成功。

### 推理时 Decode 首步延迟较高

Decode 首步需要初始化 KV Cache 和编译计算图，延迟较高（~700ms），后续步骤延迟降至 ~37ms，属于正常现象。

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen2.5-Math-1.5B ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-Math-1.5B)
- [Qwen2.5-Math-1.5B HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)

## 许可证

本工具遵循 Qwen2.5 模型的许可证要求。
