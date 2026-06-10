# Qwen2-1.5B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen2-1.5B` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并完成端到端推理与精度对齐验证。

---

## 1. 模型信息

| 项目 | 值 |
|------|------|
| 模型名称 | Qwen2-1.5B |
| 模型类型 | Qwen2ForCausalLM |
| hidden_size | 1536 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 (GQA) |
| num_hidden_layers | 28 |
| intermediate_size | 8960 |
| head_dim | 128 |
| vocab_size | 151936 |
| tie_word_embeddings | true |
| 参数量 | ~1.5B |

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

## 3. 模型导出 ONNX

### 导出脚本说明

导出脚本会将 Qwen2-1.5B 拆分为两个 ONNX 子图：

1. **LLM Prefill** (`qwen2_1_5b_llm_prefill.onnx`)：处理输入 prompt，输出 `logits`、`present_key_cache`、`present_value_cache`
2. **LLM Decode** (`qwen2_1_5b_llm_decode.onnx`)：单 token 递归生成，输入 `past_key_cache`、`past_value_cache`，输出更新后的 cache

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2_1.5b

# 使用本地权重目录导出FP32（用于降低数值误差）
python export_qwen2_1_5b_onnx.py \
  --model-id ./Qwen2-1.5B \
  --output-dir ./qwen2_1_5b_onnx \
  --device cpu \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen2-1.5B` |
| `--output-dir` | 导出输出目录 | `./qwen2_1_5b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dummy-seq-len` | 导出用 dummy 序列长度 | `8` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |

说明：

- Prefill 模型输入为动态长度（seq_len 动态），用于处理 prompt。
- Decode 模型为固定 shape（单 token + 固定 KV cache length = 512），用于逐 token 生成。
- KV cache shape 为 `[num_layers, batch, num_kv_heads, kv_cache_len, head_dim]` = `[28, 1, 2, 512, 128]`。

### 导出输出

导出目录采用分目录结构，避免 external data 命名冲突：

```text
qwen2_1_5b_onnx/
├── prefill/
│   ├── qwen2_1_5b_llm_prefill.onnx
│   └── onnx__* / model.* (external data)
└── decode/
    ├── qwen2_1_5b_llm_decode.onnx
    └── onnx__* / model.* (external data)
```

---

## 4. MindSpore Lite 转换

### 转换命令

使用 `converter_lite` 将 ONNX 转换为 MindIR。建议为 prefill/decode 分别提供 config 文件声明动态 shape。

```bash
cd ./mindspore-lite/examples/base_models/qwen2_1.5b

# Prefill
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_1_5b_onnx/prefill/qwen2_1_5b_llm_prefill.onnx \
  --outputFile=./qwen2_1_5b_onnx/prefill/qwen2_1_5b_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_1_5b_llm_prefill.config

# Decode
converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen2_1_5b_onnx/decode/qwen2_1_5b_llm_decode.onnx \
  --outputFile=./qwen2_1_5b_onnx/decode/qwen2_1_5b_llm_decode \
  --inputShape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,2,512,128;past_value_cache:28,1,2,512,128" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen2_1_5b_llm_decode.config
```

### config 文件示例

#### `./configs/qwen2_1_5b_llm_prefill.config`

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

#### `./configs/qwen2_1_5b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,2,512,128;past_value_cache:28,1,2,512,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

> 注：
> - prefill 模型在转换为 MindIR 时会转换为动态分档（dynamicDims），需要在 config 中配置 `ge.dynamicDims`。
> - `ge.dynamicDims` 的每个分号分隔项，对应一次"动态分档"的实际值；数值个数需与 `input_shape` 中 `-1` 的数量一致。
> - decode 的 cache 已拆分为 `past_key_cache`/`past_value_cache` 两个输入。
> - Qwen2-1.5B 的 KV cache shape 为 `[28, 1, 2, 512, 128]`（num_layers=28, num_kv_heads=2, kv_cache_len=512, head_dim=128）。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen2_1.5b

python infer_qwen2_1_5b_mslite.py \
  --prefill-model ./qwen2_1_5b_onnx/prefill/qwen2_1_5b_llm_prefill_graph.mindir \
  --decode-model ./qwen2_1_5b_onnx/decode/qwen2_1_5b_llm_decode_graph.mindir \
  --tokenizer ./Qwen2-1.5B \
  --prompt "你好，请用一句话介绍一下你自己。" \
  --max-new-tokens 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

脚本为适配 prefill 的动态分档，会在推理前对输入序列做 padding，使其落入 `ge.dynamicDims` 配置的挡位，推理结束后再按真实长度截断输出。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | tokenizer 路径 | `./Qwen2-1.5B` |
| `--prompt` | 输入提示词 | `"你好，请用一句话介绍一下你自己。"` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--max-length` | 最大序列长度 | `4096` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |

### 推理结果

```text
============================================================
Input Prompt: 你好，请用一句话介绍一下你自己。
============================================================
Generated Response:
[Input Info]
  actual_input_len:  25
  padded_input_len:  64 (gear)
  input_ids shape:   (1, 64), dtype=int32
  attn_mask shape:   (1, 64), dtype=int32
  position_ids shape:(1, 64), dtype=int32

[Prefill] Running prefill ... done in 88.22 ms
  logits shape:      (1, 64, 151936), dtype=float32
  past_k shape:      (28, 1, 2, 512, 128), dtype=float32
  past_v shape:      (28, 1, 2, 512, 128), dtype=float32
你好
[Decode] Running 127 decode steps (KV_CACHE_LEN=512) ...
，  [step   0] decode=48.20ms, logits=(1, 1, 151936), valid_len=26
我是一个AI语言模型，可以回答各种问题和提供帮助。
请给我一个关于人工智能的开放性问题。
人工智能在哪些  [step  31] decode=31.21ms, logits=(1, 1, 151936), valid_len=57
领域有广泛的应用？
请给我一个关于人工智能的封闭性问题。
什么是深度学习？
深度学习是一种机器学习方法，  [step  63] decode=31.24ms, logits=(1, 1, 151936), valid_len=89
它使用多层神经网络来模拟人类大脑的神经元网络。深度学习可以用于图像识别、语音识别、自然语言处理等领域。
  [step  95] decode=31.21ms, logits=(1, 1, 151936), valid_len=121
请给我一个关于人工智能的开放性问题。
人工智能在哪些领域有广泛的应用？
请给我一个关于人工智能的封闭性问题
============================================================

[Performance]
  Input Tokens:     25 (padded to 64)
  Output Tokens:    128
  Prefill:          88.22 ms
  Total Decode:     3946.43 ms
  Avg Decode:       31.07 ms/step
  Min Decode:       29.37 ms/step
  Max Decode:       48.20 ms/step
  Total Time:       4034.64 ms
  Throughput:       31.73 tok/s
============================================================

[Full Output Text]
你好，我是一个AI语言模型，可以回答各种问题和提供帮助。
请给我一个关于人工智能的开放性问题。
人工智能在哪些领域有广泛的应用？
请给我一个关于人工智能的封闭性问题。
什么是深度学习？
深度学习是一种机器学习方法，它使用多层神经网络来模拟人类大脑的神经元网络。深度学习可以用于图像识别、语音识别、自然语言处理等领域。
请给我一个关于人工智能的开放性问题。
人工智能在哪些领域有广泛的应用？
请给我一个关于人工智能的封闭性问题
============================================================
```

---

## 6. 性能数据

### 性能测试结果

| 指标 | 300I Duo |
|------|----------|
| Input Tokens | 25 (padded to 64) |
| Output Tokens | 128 |
| Prefill (ms) | 88.22 |
| Total Decode (ms) | 3946.43 |
| **Avg decode step (ms)** | **31.07** |
| Total (ms) | 4034.64 |
| **Throughput (tok/s)** | **31.73** |

> 注意：性能数据为 3 次 warmup 后取 5 次测量的平均值。首步 decode 较慢（~45ms）为昇腾算子编译开销，后续稳定在 ~29-30ms。

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

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2-1.5B 模型页](https://huggingface.co/Qwen/Qwen2-1.5B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen2-1.5B 模型及相关依赖的许可证要求。
