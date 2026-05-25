# Qwen3.5-0.8B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3.5-0.8B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3.5-0.8B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits、conv_state、recurrent_state 与 KV cache
3. **LLM Decode**（`qwen3_5_llm_decode.onnx`）：基于 conv_state + recurrent_state + KV cache 做自回归增量生成

## 模型架构

Qwen3.5-0.8B 的 24 层 decoder 中：

- **18 层线性注意力**（GatedDeltaNet）：使用 conv_state + recurrent_state 进行状态传递，无需 KV cache
- **6 层全注意力**（Full Attention）：使用标准 KV cache 进行状态传递

这种混合架构在保持模型能力的同时降低了推理复杂度。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0 |
| transformers   | 5.6.2  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 9.0    |
| mindspore-lite | 2.8.0  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_qwen3_5_0.8b_onnx.py \
  --model-id ./Qwen3.5-0.8B \
  --output-dir ./qwen3_5_0.8b_onnx \
  --device cpu \
  --vision-image-size 128
```

### 参数说明

| 参数                    | 说明                     | 默认值                  |
|-----------------------|------------------------|--------------------|
| `--model-id`          | HuggingFace 模型路径或本地目录  | `./Qwen3.5-0.8B`   |
| `--output-dir`        | 输出目录                   | `./qwen3_5_0.8b_onnx` |
| `--device`            | 导出设备（cpu/cuda）         | `cpu`              |
| `--vision-image-size` | Vision 模型输入图像尺寸（正方形边长） | `128`              |
| `--dummy-seq-len`     | LLM 导出时 dummy 序列长度     | `8`                |

### 产出

```text
qwen3_5_0.8b_onnx/
├── qwen3_5_vision.onnx          # Vision Tower 模型 (~192 MB)
├── qwen3_5_llm_prefill.onnx     # Prefill 模型 (~1.89 GB)
└── qwen3_5_llm_decode.onnx      # Decode 模型 (~1.88 GB)
```

### ONNX 模型输入输出 Shape

**Vision Tower** — `qwen3_5_vision.onnx`

| 方向   | 名称             | Shape         | Dtype   | 说明                   |
|------|----------------|---------------|---------|----------------------|
| 输入  | `pixel_values` | `(64, 1536)`  | float16 | 64 patches, 每patch 1536维 |
| 输出  | `image_embeds` | `(16, 1024)`  | float16 | 16个图像token, 1024维 hidden |

> patch 数量 = (128/14)^2 = 64，1536 = 3 channels × 2 temporal × 14 × 14 patch_size

**LLM Prefill** — `qwen3_5_llm_prefill.onnx`

| 方向   | 名称                        | Shape                         | Dtype   | 说明                      |
|------|---------------------------|-------------------------------|---------|-------------------------|
| 输入  | `input_ids`               | `(batch, seq_len)`            | int64   | token IDs（含图像占位符）      |
| 输入  | `attention_mask`           | `(batch, seq_len)`            | int64   | 注意力掩码                  |
| 输入  | `position_ids`             | `(4, batch, seq_len)`         | int64   | MRoPE位置ID（text+3D视觉）   |
| 输入  | `image_embeds`             | `(num_img_tokens, 1024)`      | float16 | 视觉编码输出                 |
| 输出  | `logits`                  | `(batch, seq_len, 248320)`    | float16 | 下一个token预测logits        |
| 输出  | `present_conv_states`       | `(18, batch, 6144, 3)`        | float16 | 18层线性注意力的conv状态       |
| 输出  | `present_recurrent_states`  | `(18, batch, 16, 128, 128)`   | float32 | 18层线性注意力的recurrent状态  |
| 输出  | `present_kv_cache`          | `(12, batch, 2, seq_len, 256)` | float16 | 6层全注意力的KV cache（12=6×2） |

> conv_dim=6144 = (key_dim×2 + value_dim) = (128×16×2 + 128×16) = 6144; conv_kernel_size=4, state保留3个历史值

**LLM Decode** — `qwen3_5_llm_decode.onnx`

| 方向   | 名称                        | Shape                              | Dtype   | 说明                       |
|------|---------------------------|------------------------------------|---------|---------------------------|
| 输入  | `input_ids`               | `(batch, 1)`                       | int64   | 单步token                   |
| 输入  | `attention_mask`           | `(batch, total_seq_len)`            | int64   | 累积注意力掩码                |
| 输入  | `position_ids`             | `(4, batch, 1)`                    | int64   | 单步MRoPE位置               |
| 输入  | `past_conv_states`          | `(18, batch, 6144, 3)`             | float16 | 上一步conv状态               |
| 输入  | `past_recurrent_states`     | `(18, batch, 16, 128, 128)`        | float32 | 上一步recurrent状态          |
| 输入  | `past_kv_cache`             | `(12, batch, 2, past_seq_len, 256)` | float16 | 上一步KV cache              |
| 输出  | `logits`                  | `(batch, 1, 248320)`               | float16 | 单步logits                 |
| 输出  | `present_conv_states`        | `(18, batch, 6144, 3)`             | float16 | 更新后的conv状态             |
| 输出  | `present_recurrent_states`   | `(18, batch, 16, 128, 128)`        | float32 | 更新后的recurrent状态        |
| 输出  | `present_kv_cache`           | `(12, batch, 2, total_seq_len, 256)` | float16 | 更新后的KV cache           |

---

## 3. ONNX 转 MindIR

### 转换命令

> **重要**：所有三个模型转换时都必须指定 `--configFile=config.ini`（`force_fp32`），以确保 Ascend 推理精度与 PyTorch 一致。不使用 `config.ini` 会导致 Vision Tower 在 Ascend FP16 下产生严重精度损失。

```bash
Convert=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Vision 转换（必须使用 config.ini）
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_vision \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR
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

`config.ini`（所有模型转换时都需要）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen3_5_0.8b_onnx/
├── qwen3_5_vision.mindir                          # Vision MindIR (~202 MB)
├── qwen3_5_llm_prefill_graph.mindir               # Prefill MindIR 图定义 (~2.7 KB)
├── qwen3_5_llm_prefill_variables/data_0            # Prefill 权重数据 (~5.4 GB)
├── qwen3_5_llm_decode_graph.mindir                 # Decode MindIR 图定义 (~2.9 KB)
└── qwen3_5_llm_decode_variables/data_0             # Decode 权重数据 (~4.1 GB)
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_5_0.8b_mslite.py \
  --vision-model qwen3_5_0.8b_onnx/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_0.8b_onnx/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_0.8b_onnx/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-0.8B \
  --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
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
| `--processor`      | HuggingFace processor 路径              | `./Qwen3.5-0.8B`        |
| `--image`          | 输入图像路径或 URL                          | Qwen VL demo 图片        |
| `--prompt`         | 输入文本                                  | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--image-size`     | 图像尺寸（必须与导出 `--vision-image-size` 一致） | `128`                   |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |
| `--device-id`      | Ascend 设备 ID                          | `0`                     |

### 推理示例输出

使用 Qwen VL 官方提供的 demo.jpeg（https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg）：

```text
Initializing MindSpore Lite context for Ascend...
Loading vision model from qwen3_5_0.8b_onnx/qwen3_5_vision.mindir...
Loading prefill model from qwen3_5_0.8b_onnx/qwen3_5_llm_prefill_graph.mindir...
Loading decode model from qwen3_5_0.8b_onnx/qwen3_5_llm_decode_graph.mindir...
Loading processor from ./Qwen3.5-0.8B...
Running vision tower...
Vision time: 5.38 ms
Running LLM prefill...
Prefill time: 3431.77 ms
Running LLM decode...
Total decode time: 8695.44 ms, avg decode step: 68.47 ms, steps: 127
Total time: 12132.59 ms, throughput: 10.55 tok/s

==================================================
Input Prompt: Describe this image.
Generated Response: This is a serene, emotionally evocative photograph capturing a tender moment
between two women in the ocean at sunset.

**Setting & Atmosphere:**
- The scene takes place on the ocean, with the horizon glowing in warm hues of orange, pink,
  and soft yellow — characteristic of a sunset or sunrise.
- The water is calm, reflecting the sky and the silhouettes of the women.
- The overall mood is peaceful, intimate, and nostalgic.

**Subjects:**
- Two women are seated side by side in the shallow water, facing each other.
- The woman on the left has long, light
==================================================
```

---

## 5. 性能数据

### 测试环境

| 项目   | 配置                     |
|------|------------------------|
| 硬件   | Atlas 300I Duo（Ascend NPU） |
| 模型   | Qwen3.5-0.8B           |
| 图片   | Qwen VL 官方 [demo.jpeg](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg) |
| 图像尺寸 | 128 × 128（vision_image_size=128） |
| 精度   | force_fp32（config.ini）  |

### 各模型推理输入 Shape 与性能

**Vision Tower**

| 项目           | 值                  |
|--------------|--------------------|
| 输入名称        | `pixel_values`     |
| 输入 Shape    | `(64, 1536)`       |
| 输入 Dtype    | float16            |
| 输出 Shape    | `(16, 1024)`       |
| 推理耗时        | **5.38 ms**        |

> 输入 64 个 patch（128/14 × 128/14 ≈ 9×9=81，实际 grid 为 8×8=64），输出 16 个图像 token。

**LLM Prefill**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids`, `image_embeds` |
| input_ids Shape  | `(1, 34)`                     |
| attention_mask Shape | `(1, 34)`                  |
| position_ids Shape | `(4, 1, 34)`                  |
| image_embeds Shape | `(16, 1024)`                  |
| 输出 logits Shape | `(1, 34, 248320)`             |
| 推理耗时        | **3431.77 ms**                   |

> seq_len=34 包含：系统 prompt token + 16 个图像 token + 用户文本 token + 生成 prompt token

**LLM Decode（单步）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids`, `past_conv_states`, `past_recurrent_states`, `past_kv_cache` |
| input_ids Shape  | `(1, 1)`                        |
| attention_mask Shape | `(1, 34+step)`               |
| position_ids Shape | `(4, 1, 1)`                    |
| past_kv_cache Shape | `(12, 1, 2, 34+step, 256)`   |
| 输出 logits Shape | `(1, 1, 248320)`               |
| 单步平均耗时      | **68.47 ms**                     |

### 端到端推理性能（128 tokens 生成）

| 指标                       | 耗时 (ms)  |
|--------------------------|----------|
| Vision Tower             | 5.38     |
| LLM Prefill              | 3431.77  |
| LLM Decode（127 steps）   | 8695.44  |
| **总耗时**                  | **12132.59** |
| **Avg decode step**       | **68.47** |
| **吞吐量**                  | **10.55 tok/s** |
| **生成 token 数**           | **128**  |

> 注意：首次推理 Prefill 时间较长为正常现象，后续推理会趋于稳定。Decode 阶段随着 KV cache 增长，单步耗时可能略有增加。

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-0.8B 官方文档](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 7. 许可证

本教程遵循 Qwen3.5-0.8B 模型的许可证。
