# Qwen2.5-3B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen2.5-3B` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并完成端到端推理与精度对齐验证。

---

## 1. 模型信息

| 项目 | 值 |
|------|------|
| 模型名称 | Qwen2.5-3B-Instruct |
| 模型类型 | Qwen2ForCausalLM |
| hidden_size | 2048 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 (GQA) |
| num_hidden_layers | 36 |
| intermediate_size | 11008 |
| head_dim | 128 |
| vocab_size | 151936 |
| tie_word_embeddings | true |
| 参数量 | ~3B |

---

## 2. 环境准备

### 系统要求

- Python 3.11
- Linux 系统（推荐 Ubuntu 22.04）
- 昇腾环境（用于 MindIR 推理，需安装 MindSpore Lite 与 Ascend 驱动）

### 依赖版本（建议）

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.7.0+ |
| transformers   | 4.51.0+ |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.0 |
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch transformers onnx onnxruntime mindspore-lite
```

### 验证安装

```bash
python -c "import torch, transformers, onnx, onnxruntime, mindspore_lite; print('All dependencies installed successfully!')"
```

---

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本会将 Qwen2.5-3B 拆分为两个 ONNX 子图：

1. **LLM Prefill** (`qwen2_5_3b_llm_prefill.onnx`)：处理输入 prompt，输出 `logits`、`present_key_cache`、`present_value_cache`
2. **LLM Decode** (`qwen2_5_3b_llm_decode.onnx`)：单 token 递归生成，输入 `past_key_cache`、`past_value_cache`，输出更新后的 cache

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_3b

# 使用本地权重目录导出
python export_qwen2_5_3b_onnx.py \
  --model-id ./Qwen2.5-3B-Instruct \
  --output-dir ./qwen2_5_3b_onnx \
  --device cpu \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen2.5-3B-Instruct` |
| `--output-dir` | 导出输出目录 | `./qwen2_5_3b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dummy-seq-len` | 导出用 dummy 序列长度 | `8` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |

说明：

- Prefill 模型输入为动态长度（seq_len 动态），用于处理 prompt。
- Decode 模型为固定 shape（单 token + 固定 KV cache length = 512），用于逐 token 生成。
- KV cache shape 为 `[num_layers, batch, num_kv_heads, kv_cache_len, head_dim]` = `[36, 1, 2, 512, 128]`。

### 导出输出

导出目录采用分目录结构，避免 external data 命名冲突：

```text
qwen2_5_3b_onnx/
├── prefill/
│   ├── qwen2_5_3b_llm_prefill.onnx
│   └── onnx__* / model.* (external data)
└── decode/
    ├── qwen2_5_3b_llm_decode.onnx
    └── onnx__* / model.* (external data)
```

---

## 4. MindSpore Lite 转换

### 转换命令

使用 `converter_lite` 将 ONNX 转换为 MindIR。建议为 prefill/decode 分别提供 config 文件声明动态 shape。

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_3b

# Prefill
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_5_3b_onnx/prefill/qwen2_5_3b_llm_prefill.onnx \
  --outputFile=./qwen2_5_3b_onnx/prefill/qwen2_5_3b_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_5_3b_llm_prefill.config

# Decode
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_5_3b_onnx/decode/qwen2_5_3b_llm_decode.onnx \
  --outputFile=./qwen2_5_3b_onnx/decode/qwen2_5_3b_llm_decode \
  --inputShape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:36,1,2,512,128;past_value_cache:36,1,2,512,128" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_5_3b_llm_decode.config
```

### config 文件示例

#### `./configs/qwen2_5_3b_llm_prefill.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
ge.dynamicDims="128,128,128;256,256,256"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

#### `./configs/qwen2_5_3b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:36,1,2,512,128;past_value_cache:36,1,2,512,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

> 注：
> - prefill 模型在转换为 MindIR 时会转换为动态分档（dynamicDims），需要在 config 中配置 `ge.dynamicDims`。
> - `ge.dynamicDims` 的每个分号分隔项，对应一次"动态分档"的实际值；数值个数需与 `input_shape` 中 `-1` 的数量一致。
> - decode 的 cache 已拆分为 `past_key_cache`/`past_value_cache` 两个输入。
> - Qwen2.5-3B 的 KV cache shape 为 `[36, 1, 2, 512, 128]`（num_layers=36, num_kv_heads=2, kv_cache_len=512, head_dim=128）。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2.5_3b

python infer_qwen2_5_3b_mslite.py \
  --prefill-model ./qwen2_5_3b_onnx/prefill/qwen2_5_3b_llm_prefill_graph.mindir \
  --decode-model ./qwen2_5_3b_onnx/decode/qwen2_5_3b_llm_decode_graph.mindir \
  --tokenizer ./Qwen2.5-3B-Instruct \
  --prompt "你好，介绍一下你自己" \
  --max-new-tokens 512 \
  --device ascend \
  --device-id 0
```

### 推理输出示例

```text
============================================================
Input Prompt: 你好，介绍一下你自己
============================================================
Generated Response: 你好！我是Qwen，是由阿里云开发的语言模型。我被训练成能够理解、解释和生成人类语言，提供信息、回答问题、创作文字，比如写故事、写公文、写邮件、写剧本等等。我还可以帮助进行创意生成、问题解答、文本创作、对话模拟等任务。我能够处理各种主题，包括科学、技术、艺术、历史、文化等。同时，我也能够学习和适应新的信息，以便更好地服务于用户。我致力于提供准确、有用和有帮助的信息，以满足用户的需求。
============================================================

[Performance]
  Prefill:        150.34 ms
  Total Decode:   5803.82 ms
  Avg Decode:     49.61 ms/step
  Total Time:     5954.16 ms
  Tokens:         118
  Throughput:     19.82 tok/s
============================================================
```

### 参数说明

脚本为适配 prefill 的动态分档，会在推理前对输入序列做 padding，使其落入 `ge.dynamicDims` 配置的挡位，推理结束后再按真实长度截断输出。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | tokenizer 路径 | `./Qwen2.5-3B-Instruct` |
| `--prompt` | 输入提示词 | `"你好，介绍一下你自己"` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--max-length` | 最大序列长度 | `4096` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |

---

## 6. 性能数据

### 性能测试结果

测试模型：Qwen2.5-3B-Instruct
测试条件：输入 ~30 tokens，输出 128 tokens
测试环境：CANN 8.5.0，MindSpore Lite 2.9.0，Ascend 300I Duo

| 指标                       | 300I Duo time     |
|--------------------------|------------|
| Prefill (ms)             | 150.34      |
| Total Decode (ms)        | 5803.82     |
| **Avg decode step (ms)** | **49.61** |
| Total (ms)               | 5954.16     |
| **Throughput (tok/s)**   | **19.82** |

> 注意：Avg decode step 为单次 decode 推理的耗时。性能数据为 3 次 warmup 后取 5 次测量的平均值。

---

## 7. 常见问题

### 1) `apply_chat_template` 返回类型不一致

不同 tokenizer 版本可能返回 `dict`、`BatchEncoding` 或 `ndarray`，推理脚本已兼容多种返回格式。

### 2) ONNX Runtime 报 external data 越界

通常由多个 ONNX 导出到同一目录导致 external data 文件重名冲突。请使用本教程的 `prefill/`、`decode/` 分目录导出方式。

### 3) 输出"被截断"

可能由以下原因导致：

- `--max-new-tokens` 达到上限
- 提前生成 `eos_token`
- 输入长度超过 KV cache 长度（512）

### 4) 精度不达标

建议逐项尝试：

- 使用 `--dtype fp32` 导出
- 使用 `--torch-dtype fp32` 做对齐评估
- 检查 tokenizer 与权重目录是否一致
- 固定相同 prompt 与 decode step 后再对比

### 5) aclmdlSetInputDynamicDims failed

Prefill 模型推理时输入序列长度未落入 `ge.dynamicDims` 配置的挡位。需确保 config 文件中的 `ge.dynamicDims` 覆盖推理脚本可能 padding 到的所有挡位（当前为 128 和 256）。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2.5-3B 模型页](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen2.5-3B 模型及相关依赖的许可证要求。
