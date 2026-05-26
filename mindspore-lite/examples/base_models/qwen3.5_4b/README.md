# Qwen3.5-4B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3.5-4B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3.5-4B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits、conv_state、recurrent_state 与 KV cache
3. **LLM Decode**（`qwen3_5_llm_decode.onnx`）：基于 conv_state + recurrent_state + KV cache 做自回归增量生成

## 模型架构

Qwen3.5-4B 采用混合注意力架构：

- **线性注意力层**（GatedDeltaNet）：使用 conv_state + recurrent_state 进行状态传递，无需 KV cache
- **全注意力层**（Full Attention）：使用标准 KV cache 进行状态传递

| 参数 | 值 |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 32 |
| num_attention_heads | 16 |
| num_key_value_heads | 4 |
| head_dim | 256 |
| vocab_size | 248320 |
| linear_attention_layers | 24 |
| full_attention_layers | 8 |
| image_token_id | 248056 |
| patch_size | 16 |

各层类型由 `config.json` 中的 `layer_types` 字段定义，每 4 层中有 3 个 linear_attention 和 1 个 full_attention。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
|--------|------|
| Python | 3.11 |
| torch | 2.10.0 |
| transformers | 5.6.2 |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 1.26.4 |
| CANN | 8.5.0 |
| mindspore-lite | 2.9.0 |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

### 模型权重

将 Qwen3.5-4B 模型权重下载到当前目录下的 `Qwen3.5-4B/` 文件夹。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/qwen3.5_4b

python export_qwen3_5_4b_onnx.py \
  --model-id ./Qwen3.5-4B \
  --output-dir ./qwen3_5_4b_onnx \
  --device cpu \
  --vision-image-size 128
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3.5-4B` |
| `--output-dir` | 输出目录 | `./qwen3_5_4b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--vision-image-size` | Vision 模型输入图像尺寸（正方形边长） | `128` |
| `--dummy-seq-len` | 导出时 dummy 序列长度 | `8` |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |

### 导出产出与模型 Shape

```log
qwen3_5_4b_onnx/
├── qwen3_5_vision.onnx          # Vision Tower 模型 (~637MB)
├── qwen3_5_llm_prefill.onnx     # Prefill 模型 (~22MB, 外部权重 ~23GB)
└── qwen3_5_llm_decode.onnx      # Decode 模型 (~1.1MB, 外部权重 ~20GB)
```

#### Vision Tower Shape

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | pixel_values | `[64, 1536]` | float16 | 64=8x8 patches, 1536=3x2x16x16 |
| Output | image_embeds | `[16, 2560]` | float16 | 16 个 image token, hidden_size=2560 |

#### LLM Prefill Shape（动态轴）

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[batch, seq_len]` | int64 | 输入 token IDs |
| Input | attention_mask | `[batch, seq_len]` | int64 | 注意力掩码 |
| Input | position_ids | `[4, batch, seq_len]` | int64 | 4D mRoPE 位置编码 |
| Input | image_embeds | `[num_image_tokens, 2560]` | float16 | 图像特征 |
| Output | logits | `[batch, seq_len, 248320]` | float16 | 预测 logits |
| Output | present_conv_states | `[24, batch, 8192, 3]` | float16 | 卷积状态（24 层线性注意力） |
| Output | present_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 循环状态（24 层线性注意力） |
| Output | present_kv_cache | `[16, batch, 4, seq_len, 256]` | float16 | KV cache（8 层全注意力 x 2） |

#### LLM Decode Shape（动态轴）

| 输入/输出 | 名称 | Shape | 数据类型 | 说明 |
|-----------|------|-------|----------|------|
| Input | input_ids | `[batch, step]` | int64 | 单步 token ID (step=1) |
| Input | attention_mask | `[batch, total_seq_len]` | int64 | 累积注意力掩码 |
| Input | position_ids | `[4, batch, step]` | int64 | 4D mRoPE 位置编码 |
| Input | past_conv_states | `[24, batch, 8192, 3]` | float16 | 上一步卷积状态 |
| Input | past_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 上一步循环状态 |
| Input | past_kv_cache | `[16, batch, 4, past_seq_len, 256]` | float16 | 上一步 KV cache |
| Output | logits | `[batch, step, 248320]` | float16 | 预测 logits |
| Output | present_conv_states | `[24, batch, 8192, 3]` | float16 | 更新后卷积状态 |
| Output | present_recurrent_states | `[24, batch, 32, 128, 128]` | float32 | 更新后循环状态 |
| Output | present_kv_cache | `[16, batch, 4, total_seq_len, 256]` | float16 | 更新后 KV cache |

---

## 3. ONNX 转 MindSpore Lite MindIR

### 转换命令

使用 `converter_lite` 工具将 ONNX 模型转换为 MindIR 格式：

```bash
# 设置 converter_lite 路径
Convert=/path/to/mindspore-lite/tools/converter/converter/converter_lite

# Vision 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_vision \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_4b_onnx/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_4b_mindir/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR
```

### 转换产出

模型超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
qwen3_5_4b_mindir/
├── qwen3_5_vision.mindir                          # Vision MindIR (~683MB)
├── qwen3_5_llm_prefill_graph.mindir               # Prefill 图定义 (~2.8KB)
├── qwen3_5_llm_prefill_variables/data_0           # Prefill 权重 (~23GB)
├── qwen3_5_llm_decode_graph.mindir                # Decode 图定义 (~2.9KB)
└── qwen3_5_llm_decode_variables/data_0            # Decode 权重 (~21GB)
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_5_4b_mslite.py \
  --vision-model qwen3_5_4b_mindir/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_4b_mindir/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_4b_mindir/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-4B \
  --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --image-size 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vision-model` | Vision MindIR 模型路径 | 必填 |
| `--prefill-model` | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--decode-model` | Decode MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--processor` | HuggingFace processor 路径 | `./Qwen3.5-4B` |
| `--image` | 输入图像路径或 URL | 必填 |
| `--prompt` | 输入文本 | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--image-size` | 图像尺寸（必须与导出 `--vision-image-size` 一致） | `128` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |

---

## 5. 性能数据

### 测试环境

- **硬件**: Atlas 300I Duo
- **CANN**: 8.5.0
- **MindSpore Lite**: 2.9.0
- **测试图片**: Qwen3.5 官方 demo 图 (`demo.jpeg`)
- **Prompt**: "Describe this image."
- **image_size**: 128 (pixel_values: [64, 1536])
- **max_new_tokens**: 128

### 推理性能（Ascend NPU）

| 阶段 | 输入 Shape | 输出 Shape | 推理时间 | 说明 |
|------|-----------|-----------|---------|------|
| Vision Tower | `pixel_values: [64, 1536]` | `image_embeds: [16, 2560]` | **10.17 ms** | 图像编码 |
| LLM Prefill | `input_ids: [1, 33]`, `attention_mask: [1, 33]`, `position_ids: [4, 1, 33]`, `image_embeds: [16, 2560]` | `logits: [1, 33, 248320]` + states | **5137.89 ms** | 首次前向计算 |
| LLM Decode (per step) | `input_ids: [1, 1]`, `attention_mask: [1, seq]`, `position_ids: [4, 1, 1]` + states | `logits: [1, 1, 248320]` + states | **138.26 ms/step** | 自回归生成 |
| LLM Decode (127 steps) | - | - | **17559.05 ms** | 总 decode 时间 |
| **Total** | - | - | **22707.10 ms** | 端到端 |

### 性能指标

| 指标 | 值 |
|------|-----|
| 吞吐量 | **5.64 tok/s** |
| Vision 推理 | 10.17 ms |
| Prefill 推理 | 48543.19 ms (seq_len=33) |
| Decode 单步推理 | 128.22 ms (avg) |
| 总生成 token 数 | 128 |

> 注意：首次运行时 Ascend GE 图编译耗时较长（约 40 分钟），后续运行加载已编译的 OM 模型会更快。上述性能数据为推理阶段耗时，不包含模型加载时间。

### Prefill 输入 Shape 详细说明

以 `image_size=128`、prompt="Describe this image." 为例：

- `input_ids: [1, 33]` - 33 个 token (system + image placeholder + user prompt + generation prefix)，包含 16 个 image token（由 Vision Tower 产生 16 个 patch）
- `attention_mask: [1, 33]` - 全 1
- `position_ids: [4, 1, 33]` - 4D mRoPE 位置编码 (text_pos, temporal, height, width)
- `image_embeds: [16, 2560]` - Vision Tower 输出

### Decode 输入 Shape 详细说明

每步 decode 的输入：

- `input_ids: [1, 1]` - 单个 token
- `attention_mask: [1, seq]` - 随步数递增 (34, 35, 36, ...)
- `position_ids: [4, 1, 1]` - 单步位置编码
- `past_conv_states: [24, 1, 8192, 3]` - 24 层线性注意力的卷积状态
- `past_recurrent_states: [24, 1, 32, 128, 128]` - 24 层线性注意力的循环状态
- `past_kv_cache: [16, 1, 4, seq, 256]` - 8 层全注意力的 KV cache (16=8x2)

---

## 6. 推理结果示例

使用 Qwen3.5 官方 demo 图片和 prompt "Describe this image." 的推理结果：

```text

==================================================
Input Prompt: Describe this image.
Generated Response: The user wants a description of the image.

1.  **Identify the main subjects:** There are two figures in the water. One is a person, and the other is a dog.
2.  **Describe the person:**
    *   Gender: Female (long blonde hair).
    *   Clothing: Wearing a plaid shirt (blue and white/grey pattern) and dark pants (possibly jeans or leggings).
    *   Action: Sitting in the shallow water, facing the dog.
3.  **Describe the dog:**
    *   Breed: Looks like a Golden Retriever
==================================================

```

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-4B 官方文档](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 Qwen3.5-4B 模型的许可证。
