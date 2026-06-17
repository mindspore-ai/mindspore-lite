# Qwen3.5-2B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen3.5-2B` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3.5-2B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），
   输出 logits、conv_state、recurrent_state 与 KV cache
3. **LLM Decode**（`qwen3_5_llm_decode.onnx`）：基于 conv_state + recurrent_state + KV cache 做自回归增量生成

## 模型架构

Qwen3.5-2B 的 24 层 decoder 中：

- **18 层线性注意力**（GatedDeltaNet）：使用 conv_state + recurrent_state 进行状态传递，无需 KV cache
- **6 层全注意力**（Full Attention）：使用标准 KV cache 进行状态传递

| 参数 | 值 |
|---|---|
| hidden_size | 2048 |
| num_hidden_layers | 24 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| intermediate_size | 6144 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 16 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |

---

## 1. 环境准备

### 依赖版本（建议）

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0  |
| transformers   | 5.6.2  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 8.3.RC1 |
| mindspore-lite | 2.9.0  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/qwen3.5_2b

# 导出 FP32（用于降低数值误差）
python export_qwen3_5_2b_onnx.py \
  --model-id ./Qwen3.5-2B \
  --output-dir ./qwen3_5_2b_onnx_fp32 \
  --device cpu \
  --dtype fp32
```

### 参数说明

| 参数                    | 说明                       | 默认值                |
|-----------------------|--------------------------|--------------------|
| `--model-id`          | HuggingFace 模型路径或本地目录    | `./Qwen/Qwen3.5-2B`   |
| `--output-dir`        | 输出目录                     | `./qwen3_5_2b_onnx` |
| `--device`            | 导出设备（cpu/cuda）           | `cpu`              |
| `--vision-image-size` | Vision 模型输入图像尺寸（正方形边长）   | `128`              |
| `--dummy-seq-len`     | 导出用 dummy 序列长度           | `8`                |
| `--dtype`             | 导出精度（fp16/fp32）          | `fp16`             |
| `--num-layers`        | 导出层数（默认全部24层，设4用于快速验证） | `全部24层`         |

### 产出

```text
qwen3_5_2b_onnx/
├── qwen3_5_vision.onnx          # Vision Tower 模型
├── qwen3_5_llm_prefill.onnx     # Prefill 模型
└── qwen3_5_llm_decode.onnx      # Decode 模型
```

---

## 3. MindSpore Lite 转换

### 转换命令

```bash
# Vision 转换
./converter_lite --fmk=ONNX \
  --modelFile=./qwen3_5_2b_onnx_fp32/qwen3_5_vision.onnx \
  --outputFile=./qwen3_5_2b_onnx_fp32/qwen3_5_vision \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_5_2b_vision.config

# Prefill 转换
./converter_lite --fmk=ONNX \
  --modelFile=./qwen3_5_2b_onnx_fp32/qwen3_5_llm_prefill.onnx \
  --outputFile=./qwen3_5_2b_onnx_fp32/qwen3_5_llm_prefill_fp16 \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_5_2b_llm_prefill.config

# Decode 转换
./converter_lite --fmk=ONNX \
  --modelFile=./qwen3_5_2b_onnx_fp32/qwen3_5_llm_decode.onnx \
  --outputFile=./qwen3_5_2b_onnx_fp32/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_5_2b_llm_decode.config
```

> **注意**：Prefill 模型因含有 GatedDeltaNet 的 chunk parallel computation（`exp()` + `cumsum()` + `tril()`），在 FP16 ONNX 导出时会出现数值溢出导致输出全为 NaN。因此 Prefill 必须使用 **FP32 ONNX** 导出，但转换时 `allow_fp32_to_fp16` 精度模式已验证可正常工作且推理速度更快。

### config 文件说明

#### `./configs/qwen3_5_2b_vision.config`

```ini
[acl_init_options]
ge.exec.precision_mode=allow_fp32_to_fp16
```

#### `./configs/qwen3_5_2b_llm_prefill.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:4,1,-1;image_embeds:-1,2048"

[acl_init_options]
ge.exec.precision_mode=allow_fp32_to_fp16
```

#### `./configs/qwen3_5_2b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,-1;position_ids:4,1,1;past_conv_states:18,1,6144,3;past_recurrent_states:18,1,16,128,128;past_kv_cache:12,1,2,-1,256"

[acl_init_options]
ge.exec.precision_mode=allow_fp32_to_fp16
```

> 注：`past_conv_states` 和 `past_recurrent_states` 的维度基于模型架构自动计算：
> - 18 个线性注意力层，conv_dim = 128*16*2 + 128*16 = 6144，conv_kernel_size-1 = 3
> - recurrent_states: 18层 × 16 value_heads × 128 key_dim × 128 value_dim
> - KV cache: 12（6层 × 2）× 2 kv_heads × seq_len × 256 head_dim

### 产出

```text
qwen3_5_2b_onnx_fp32/
├── qwen3_5_vision.mindir                              # Vision MindIR (~647MB)
├── qwen3_5_llm_prefill_fp16_graph.mindir              # Prefill MindIR 图定义 (~2.9KB)
├── qwen3_5_llm_prefill_fp16_variables/data_0           # Prefill 权重数据 (~14GB)
├── qwen3_5_llm_decode_graph.mindir                     # Decode MindIR 图定义 (~3.1KB)
└── qwen3_5_llm_decode_variables/data_0                 # Decode 权重数据 (~5.9GB)
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_5_2b_mslite.py \
  --vision-model ./qwen3_5_2b_onnx_fp32/qwen3_5_vision.mindir \
  --prefill-model ./qwen3_5_2b_onnx_fp32/qwen3_5_llm_prefill_fp16_graph.mindir \
  --decode-model ./qwen3_5_2b_onnx_fp32/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-2B \
  --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --image-size 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--vision-model`  | Vision MindIR 模型路径                    | 必填                      |
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                      |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                      |
| `--processor`      | HuggingFace processor 路径              | `./Qwen/Qwen3.5-2B`     |
| `--image`          | 输入图像路径或 URL                          | `./demo.jpeg`            |
| `--prompt`         | 输入文本                                  | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--image-size`     | 图像尺寸（必须与导出 `--vision-image-size` 一致） | `128`                   |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |
| `--device-id`      | Ascend 设备 ID                          | `0`                     |

### 外部资源说明

- README 示例中使用 Qwen 官方 demo 图片：`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`。
- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

---

## 5. 性能评估

### 测试环境

- **硬件**: Atlas 300I Duo
- **CANN**: 8.3.RC1
- **MindSpore Lite**: 2.9.0
- **测试图片**: Qwen3.5 官方 demo 图 (`demo.jpeg`)
- **Prompt**: "Describe this image."
- **image_size**: 128 (pixel_values: [64, 1536])
- **max_new_tokens**: 128
- **精度模式**: Vision/Decode/Prefill = `allow_fp32_to_fp16`

### 推理性能（Ascend NPU）

| 阶段 | 输入 Shape | 输出 Shape | 推理时间 | 说明 |
|------|-----------|-----------|---------|------|
| Vision Tower | `pixel_values: [64, 1536]` | `image_embeds: [16, 2048]` | **11.01 ms** | 图像编码 |
| LLM Prefill | `input_ids: [1, 48]`, `attention_mask: [1, 48]`, `position_ids: [4, 1, 48]`, `image_embeds: [16, 2048]` | `logits: [1, 48, 248320]` + states | **3497.80 ms** | 首次前向计算（34 token 对齐填充至 48） |
| LLM Decode (per step) | `input_ids: [1, 1]`, `attention_mask: [1, seq]`, `position_ids: [4, 1, 1]` + states | `logits: [1, 1, 248320]` + states | **71.62 ms/step** | 自回归生成 |
| LLM Decode (127 steps) | - | - | **9095.73 ms** | 总 decode 时间 |
| **Total** | - | - | **12604.54 ms** | 端到端 |

### 性能指标

| 指标 | 值 |
|------|-----|
| **吞吐量** | **10.16 tok/s** |
| Vision 推理 | 11.01 ms |
| Prefill 推理 | 3497.80 ms (seq_len=34, padded to 48) |
| Decode 单步推理 | 71.62 ms (avg) |
| 总生成 token 数 | 128 |

### Prefill 输入 Shape 详细说明

以 `image_size=128`、prompt="Describe this image." 为例：

- `input_ids: [1, 34]` - 34 个 token，包含 16 个 image token（由 Vision Tower 产生 16 个 patch）
- `attention_mask: [1, 34]` - 全 1
- `position_ids: [4, 1, 34]` - 4D mRoPE 位置编码 (text_pos, temporal, height, width)
- `image_embeds: [16, 2048]` - Vision Tower 输出

### Decode 输入 Shape 详细说明

每步 decode 的输入：

- `input_ids: [1, 1]` - 单个 token
- `attention_mask: [1, seq]` - 随步数递增 (35, 36, 37, ...)
- `position_ids: [4, 1, 1]` - 单步位置编码
- `past_conv_states: [18, 1, 6144, 3]` - 18 层线性注意力的卷积状态
- `past_recurrent_states: [18, 1, 16, 128, 128]` - 18 层线性注意力的循环状态
- `past_kv_cache: [12, 1, 2, seq, 256]` - 6 层全注意力的 KV cache (12=6x2)

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-2B 模型页 (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-2B 模型页 (ModelScope)](https://modelscope.cn/models/Qwen/Qwen3.5-2B)
- [Transformers 文档](https://huggingface.co/docs/transformers)

---

## 7. 许可证

本教程遵循 Qwen3.5-2B 模型的许可证。
