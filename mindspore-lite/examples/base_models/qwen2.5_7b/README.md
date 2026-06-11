# Qwen2.5-7B-Instruct ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen2.5-7B-Instruct` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并完成端到端推理与精度对齐验证。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux 系统（推荐 Ubuntu 22.04）
- 昇腾环境（用于 MindIR 推理，需安装 MindSpore Lite 与 Ascend 驱动）

### 依赖版本（建议）

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.8.0 |
| transformers   | 4.51.0 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.0 |
| mindspore-lite | 2.8.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.8.0 transformers==4.51.0 onnx==1.21.0 onnxruntime==1.24.0 mindspore-lite==2.8.0
```

### 验证安装

```bash
python -c "import torch, transformers, onnx, onnxruntime, mindspore_lite; print('All dependencies installed successfully!')"
```

---

## 2. 模型架构参数

| 参数 | 值 |
|------|------|
| hidden_size | 3584 |
| num_attention_heads | 28 |
| num_hidden_layers | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| intermediate_size | 18944 |
| max_position_embeddings | 32768 |
| vocab_size | 152064 |
| rms_norm_eps | 1e-06 |
| torch_dtype | bfloat16 |
| 架构 | Qwen2ForCausalLM |

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本会将 Qwen2.5-7B 拆分为两个 ONNX 子图：

1. **LLM Prefill** (`qwen2_5_7b_llm_prefill.onnx`)：处理输入 prompt，输出 `logits`、`present_key_cache`、`present_value_cache`
2. **LLM Decode** (`qwen2_5_7b_llm_decode.onnx`)：单 token 递归生成，输入 `past_key_cache`、`past_value_cache`，输出更新后的 cache

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_7b

# 使用本地权重目录导出 FP32 onnx 模型（用于降低数值误差）
python export_qwen2_5_7b_onnx.py \
  --model-id ./Qwen2.5-7B-Instruct \
  --output-dir ./qwen2_5_7b_onnx \
  --device cpu \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen2.5-7B-Instruct` |
| `--output-dir` | 导出输出目录 | `./qwen2_5_7b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dummy-seq-len` | 导出用 dummy 序列长度 | `8` |
| `--kv-cache-len` | KV cache 固定长度（prefill 输出与 decode 输入） | `512` |
| `--dtype` | 导出精度（fp16/fp32/bf16） | `fp16` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |

### 导出输出

导出目录采用分目录结构，避免 external data 命名冲突：

```text
qwen2_5_7b_onnx/
├── prefill/
│   ├── qwen2_5_7b_llm_prefill.onnx
│   └── onnx__* / model.* (external data)
└── decode/
    ├── qwen2_5_7b_llm_decode.onnx
    └── onnx__* / model.* (external data)
```

---

## 4. MindSpore Lite 转换

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_7b

# Prefill（Ascend 优化，动态形状）
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_5_7b_onnx/prefill/qwen2_5_7b_llm_prefill.onnx \
  --outputFile=./qwen2_5_7b_onnx/prefill/qwen2_5_7b_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_5_7b_llm_prefill.config
```

### Decode 转换命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_7b

# Decode（Ascend 优化）
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_5_7b_onnx/decode/qwen2_5_7b_llm_decode.onnx \
  --outputFile=./qwen2_5_7b_onnx/decode/qwen2_5_7b_llm_decode \
  --inputShape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,4,512,128;past_value_cache:28,1,4,512,128" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_5_7b_llm_decode.config
```

### config 文件示例

#### `./configs/qwen2_5_7b_llm_prefill.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
ge.dynamicDims="64,64,64;128,128,128;256,256,256"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

#### `./configs/qwen2_5_7b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,4,512,128;past_value_cache:28,1,4,512,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

> 注：
> - Qwen2.5-7B 有 28 层（`num_hidden_layers=28`），4 个 KV 头（`num_key_value_heads=4`），head_dim=128。
> - Decode 的 KV cache shape 为 `28,1,4,512,128`。

---

## 5. MindSpore Lite 推理

### 推理架构

由于 7B 模型超过 protobuf 限制，推理采用混合方案：

- **Prefill**：PyTorch 原生模型（CPU）— 处理输入 prompt，提取 KV cache
- **Decode**：MindSpore Lite MindIR（Ascend）— 硬件加速的逐 token 生成

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_7b

python infer_qwen2_5_7b_mslite.py \
  --model-id ./Qwen2.5-7B-Instruct \
  --decode-model ./qwen2_5_7b_onnx/decode/qwen2_5_7b_llm_decode_graph.mindir \
  --prompt "你好，请用一句话介绍一下你自己" \
  --max-new-tokens 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | PyTorch 模型路径（用于 prefill） | `./Qwen2.5-7B-Instruct` |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | tokenizer 路径 | 同 model-id |
| `--prompt` | 输入提示词 | `"你好，请用一句话介绍一下你自己"` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--max-length` | 最大序列长度 | `4096` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |
| `--torch-dtype` | PyTorch 模型精度 | `float16` |

---

## 6. 性能数据

### 性能测试结果

```text
============================================================
Input Prompt: 你好，请用一句话介绍一下你自己
============================================================
Generated Response: [Prefill done: 35 input tokens, 11.260s]
你好，我叫Qwen，是来自阿里云的大规模语言模型，可以帮你回答问题、创作文字等。
============================================================

--- Performance ---
  Input tokens:     35
  Output tokens:    26
  Prefill (PyTorch CPU): 11260.29 ms
  Total Decode (Ascend): 2632.73 ms
  Avg decode step:  105.31 ms
  Total time:       13893.02 ms
  Decode throughput: 9.5 tok/s
```

| 指标                       | 300I Duo time |
|--------------------------|------------|
| Prefill (PyTorch CPU, ms) | 11260.29   |
| Total Decode (Ascend, ms) | 2632.73   |
| **Avg decode step (ms)** | **105.31** |
| Total (ms)               | 13893.02   |
| **Decode Throughput (tok/s)** | **9.5** |

> 注意：
> - Avg decode step 为单次 decode 推理的耗时。
> - 性能数据为 3 次 warmup 后取 5 次测量的平均值。

---

## 7. 常见问题

### 1) Prefill 转换报 protobuf 超过 2GB 限制

Qwen2.5-7B 的 Prefill ONNX 模型约 15GB，超过 GE protobuf 的 2GB 硬限制。这是 7B+ 参数模型的已知限制。解决方案是使用 PyTorch 原生模型在 CPU 上执行 prefill，Decode 使用 MindIR 在 Ascend 上执行。

### 2) `apply_chat_template` 返回类型不一致

不同 tokenizer 版本可能返回 `BatchEncoding`、`dict` 或 `ndarray`，推理脚本已兼容多种返回格式。

### 3) ONNX Runtime 报 external data 越界

通常由多个 ONNX 导出到同一目录导致 external data 文件重名冲突。请使用本教程的 `prefill/`、`decode/` 分目录导出方式。

### 4) 输出"被截断"

可能由以下原因导致：

- `--max-new-tokens` 达到上限
- 提前生成 `eos_token`
- KV cache 长度达到上限（512 tokens）

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2.5-7B-Instruct 模型页](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen2.5-7B-Instruct 模型及相关依赖的许可证要求。
