# Qwen3-8B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen3-8B` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并完成端到端推理与精度对齐验证。

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
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0，8.3.x 可用于 ONNX 导出和 general 模式转换 |

### 安装命令

```bash
pip install torch==2.8.0 transformers==4.51.0 onnx==1.21.0 onnxruntime==1.24.0
```

### 验证安装

```bash
python -c "import torch, transformers, onnx, onnxruntime, mindspore_lite; print('All dependencies installed successfully!')"
```

---

## 2. 模型导出 ONNX

### 导出脚本说明

导出脚本会将 Qwen3-8B 拆分为两个 ONNX 子图：

1. **LLM Prefill** (`qwen3_8b_llm_prefill.onnx`)：处理输入 prompt，输出 `logits`、`present_key_cache`、`present_value_cache`
2. **LLM Decode** (`qwen3_8b_llm_decode.onnx`)：单 token 递归生成，输入 `past_key_cache`、`past_value_cache`，输出更新后的 cache

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b

# 推荐 fp16 导出（权重约 16GB；fp32 会增大一倍并触发转换期 protobuf 限制，见 FAQ 6）
python export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_onnx \
  --device cpu \
  --dtype fp16
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3-8B` |
| `--output-dir` | 导出输出目录 | `./qwen3_8b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dummy-seq-len` | 导出用 dummy 序列长度 | `8` |
| `--dtype` | 导出精度（fp16/fp32/bf16） | `fp16` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |

说明：

- Prefill 模型输入为动态长度（seq_len 动态），用于处理 prompt。
- Decode 模型为固定 shape（单 token + 固定 KV cache length），用于逐 token 生成。

### 导出输出

导出目录采用分目录结构，避免 external data 命名冲突：

```text
qwen3_8b_onnx/
├── prefill/
│   ├── qwen3_8b_llm_prefill.onnx
│   └── onnx__* / model.* (external data)
└── decode/
    ├── qwen3_8b_llm_decode.onnx
    └── onnx__* / model.* (external data)
```

### 模型架构参数

| 参数 | 值 |
|------|------|
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_hidden_layers | 36 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 12288 |
| max_position_embeddings | 40960 |
| vocab_size | 151936 |

---

## 3. ONNX 模型验证

### 模型结构说明

导出的 ONNX 模型使用了 MindSpore Lite 自定义算子（RmsNorm、RotaryMul、SwiGlu、IncreFlashAttention、Scatter），**不支持直接使用 ONNX Runtime 推理**。需通过 `converter_lite` 转换为 MindIR 格式后在 MindSpore Lite 上运行。

### 模型中包含的自定义算子

| 算子 | Prefill 数量 | Decode 数量 | 说明 |
|------|-------------|-------------|------|
| RmsNorm | 145 | 73 | RMS 归一化 |
| RotaryMul | 72 | 72 | 旋转位置编码 |
| SwiGlu | 36 | 36 | 激活函数 |
| PromptFlashAttention | 36 | - | 全量注意力（prefill 阶段） |
| IncreFlashAttention | - | 36 | 增量注意力（decode 阶段） |
| Scatter | - | 72 | KV cache 更新（decode 阶段） |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b

# Prefill
./converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill.onnx \
  --outputFile=./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_8b_llm_prefill.config

# Decode
./converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_8b_onnx/decode/qwen3_8b_llm_decode.onnx \
  --outputFile=./qwen3_8b_onnx/decode/qwen3_8b_llm_decode \
  --inputShape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:36,1,8,512,128;past_value_cache:36,1,8,512,128" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_8b_llm_decode.config
```

### config 文件示例

#### `./configs/qwen3_8b_llm_prefill.config`

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

#### `./configs/qwen3_8b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:36,1,8,512,128;past_value_cache:36,1,8,512,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b

python infer_qwen3_8b_mslite.py \
  --prefill-model ./qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_graph.mindir \
  --decode-model ./qwen3_8b_onnx/decode/qwen3_8b_llm_decode_graph.mindir \
  --tokenizer ./Qwen3-8B \
  --prompt "你好，请用一句话介绍一下你自己。" \
  --max-new-tokens 512 \
  --device ascend \
  --device-id 0
```

### 参数说明

脚本为适配 prefill 的动态分档，会在推理前对输入序列做 padding，使其落入 `ge.dynamicDims` 配置的挡位，推理结束后再按真实长度截断输出。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | tokenizer 路径 | `./Qwen3-8B` |
| `--prompt` | 输入提示词 | `"你好，请介绍一下你自己。"` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--max-length` | 最大序列长度 | `4096` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |

---

## 6. 性能数据

### 性能测试结果

| 指标                       | 300I Duo time |
|--------------------------|------------|
| Prefill (ms)             | 186.58     |
| Total Decode (ms)        | 23492.61    |
| **Avg decode step (ms)** | **123.00** |
| Total (ms)               | 23679.18   |
| Tokens Generated (token) | 192.00     |
| **Throughput (tok/s)**   | **8.11**   |

### 输出效果

```text
============================================================
Input Prompt: 你好，请用一句话介绍一下你自己。
============================================================
Generated Response: <think>
好的，用户让我用一句话介绍自己。首先，我需要确定用户的需求是什么。他们可能是在测试我的功能，或者想快速了解我的用途。作为通义千问，我应该突出我的核心能力，比如多语言支持、知识库、对话理解等。同时，要保持简洁，避免信息过载。用户可能希望这句话既专业又易于理解，所以需要平衡技术术语和通俗表达。另外，要确保涵盖主要特点，比如回答问题、创作内容、逻辑推理等。还要注意句子的流畅性和自然性，避免生硬。最后，检查是否符合用户要求的一句话限制，确保没有冗余信息。现在把这些点整合成一句简洁的话。
</think>

我是通义千问，一个由通义实验室开发的超大规模语言模型，能够进行多轮对话、创作文字、逻辑推理和编程，并支持超过100种语言的交流。

--- Performance ---
  Prefill:           186.58 ms
  Total Decode:      23492.61 ms
  Avg Decode Step:   123.00 ms
  Total:             23679.18 ms
  Tokens Generated:  192
  Throughput:        8.11 tok/s
============================================================
```

---

## 7. 常见问题

### 1) `apply_chat_template` 返回类型不一致

不同 tokenizer 版本可能返回 `dict` 或 `ndarray`，推理脚本已兼容多种返回格式。

> transformers 5.x 中 `apply_chat_template(..., return_tensors="np")` 返回 `BatchEncoding`（**非** `dict` 子类），脚本已用 `hasattr(enc, "input_ids")` 兼容；4.x 返回 `ndarray` 也已兼容。

### 2) ONNX Runtime 报 external data 越界

通常由多个 ONNX 导出到同一目录导致 external data 文件重名冲突。请使用本教程的 `prefill/`、`decode/` 分目录导出方式。

### 3) 输出"被截断"

可能由以下原因导致：

- `--max-new-tokens` 达到上限
- 提前生成 `eos_token`

### 4) 精度不达标

建议逐项尝试：

- 使用 `--dtype fp32` 导出
- 使用 `--torch-dtype fp32` 做对齐评估
- 检查 tokenizer 与权重目录是否一致
- 固定相同 prompt 与 decode step 后再对比

### 5) 内存不足

Qwen3-8B 模型较大，导出和推理过程需要充足内存：

- 导出时建议使用 CPU 设备，确保系统内存 ≥ 32GB
- 推理时建议使用昇腾 NPU，充分利用硬件加速

### 6) 转换时 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`

Qwen3-8B 权重约 16GB（fp16），`ascend_oriented` 转换过程中会打印这条信息。该日志**不影响最终转换结果与使用**——转换器在支持大模型权重外部化的工具链上会正常产出 `*_graph.mindir` + `*_variables/data_0`（本目录随附的 `qwen3_8b_onnx/decode` 即此类产物，可在昇腾正常 build/predict）。

提示：

- 优先用 `--dtype fp16` 导出（权重相对 fp32 减半），并确保 prefill/decode 使用**相同** dtype；
- 若个别旧版转换器构建在该信息后未写出产物，请使用配套的可外部化权重的 mslite 构建版本进行转换。

### 7) 推理时 `kernel_name only support CustomAscend`

若 MindIR 是用 `--optimize=general` 转换的，图中会保留标准算子（如 `/embed_tokens/Gather`），昇腾 delegate build 时报 `Only support CustomAscend, but got Gather`。

解决：使用 `--optimize=ascend_oriented` 重新转换即可（参见 FAQ 6）。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-8B 模型页](https://huggingface.co/Qwen/Qwen3-8B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen3-8B 模型及相关依赖的许可证要求。
