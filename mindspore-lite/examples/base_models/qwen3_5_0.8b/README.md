# Qwen3.5-0.8B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3.5-0.8B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

## 目录内容

| 文件 / 目录 | 说明 |
|---|---|
| `export_qwen3_5_0_8b_onnx.py` | 一键导出 Vision Tower / LLM Prefill / LLM Decode 的 ONNX 模型 |
| `infer_qwen3_5_0_8b_mslite.py` | MindSpore Lite 端到端推理（vision + prefill + decode） |
| `configs/` | `converter_lite` 转换配置文件（vision / prefill / decoder 各一份，**转换时必须全部指定**） |
| `README.md` | 本教程 |

> **注意**：本工程导出/推理的 ONNX 包含 Ascend Custom 算子（如 `ChunkGatedDeltaRule`、`RecurrentGatedDeltaRule`、`IncreFlashAttention`、`VisionFlashAttention`、`PromptFlashAttention` 等），**无法使用 CPU/CUDA 上的 ONNX Runtime 直接推理**，仅用于通过 `converter_lite` 转换为 MindIR 后在 Ascend 部署。

Qwen3.5-0.8B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 next_token_id、conv_state、recurrent_state 与 KV cache
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
python export_qwen3_5_0_8b_onnx.py \
  --model-id ./Qwen3.5-0.8B \
  --output-dir ./qwen3_5_onnx_static \
  --device cpu \
  --vision-image-size 1024
```

> 注：导出时始终启用以下自定义算子：ChunkGatedDeltaRule（prefill 线性注意力）、IncreFlashAttention（decode 全注意力）、VisionFlashAttention（vision 注意力）、PromptFlashAttention（prefill 全注意力）；RecurrentGatedDeltaRule（decode 线性注意力）默认**关闭**，需通过 `--use-rgdr-custom` 启用。

### 参数说明

| 参数                          | 说明                                     | 默认值                   |
|-----------------------------|----------------------------------------|-----------------------|
| `--model-id`                | HuggingFace 模型路径或本地目录                    | `./Qwen3.5-0.8B`      |
| `--output-dir`              | 输出目录                                   | `./qwen3_5_onnx_static` |
| `--device`                  | 导出设备（cpu/cuda）                          | `cpu`                 |
| `--vision-image-size`       | Vision 模型输入图像尺寸（正方形边长）                   | `1024`                |
| `--dummy-seq-len`           | LLM 导出时 dummy 序列长度                       | `8`                   |
| `--max-seq-len`             | 最大序列长度（固定 shape KV cache padding）         | `2048`                |
| `--use-rgdr-custom`         | 启用 RecurrentGatedDeltaRule 自定义算子          | `False`               |

### 产出

```text
qwen3_5_onnx_static/
├── qwen3_5_vision.onnx              # Vision Tower 模型
├── prefill/
│   └── qwen3_5_llm_prefill.onnx     # Prefill 模型
└── decoder/
    └── qwen3_5_llm_decode.onnx      # Decode 模型（固定 shape，padded to max_seq_len）
```

### ONNX 模型输入输出 Shape

**Vision Tower** — `qwen3_5_vision.onnx`

| 方向   | 名称             | Shape                        | Dtype   | 说明                              |
|------|----------------|------------------------------|---------|---------------------------------|
| 输入  | `pixel_values` | `(num_patches, 1536)`        | float32 | num_patches=(image_size/16)^2, 每patch 1536维 |
| 输出  | `image_embeds` | `(num_img_tokens, 1024)`     | float32 | num_img_tokens=(grid/2)^2, 1024维 hidden |

> patch 数量 = (vision_image_size/16)^2，1536 = 3 channels × 2 temporal × 16 × 16 patch_size。默认为 1024 图像尺寸 → 4096 patches，spatial_merge_size=2 压缩后输出 1024 个图像 token。

**LLM Prefill** — `prefill/qwen3_5_llm_prefill.onnx`

| 方向   | 名称                        | Shape                         | Dtype   | 说明                      |
|------|---------------------------|-------------------------------|---------|-------------------------|
| 输入  | `input_ids`               | `(batch, seq_len)`            | int64   | token IDs（含图像占位符）      |
| 输入  | `attention_mask`           | `(batch, seq_len)`            | int64   | 注意力掩码                  |
| 输入  | `position_ids`             | `(4, batch, seq_len)`         | int64   | MRoPE位置ID（text+3D视觉）   |
| 输入  | `image_embeds`             | `(num_img_tokens, 1024)`      | float32 | 视觉编码输出                 |
| 输出  | `next_token_id`           | `(batch, 1)`                  | int32   | 贪婪解码的 next token ID    |
| 输出  | `present_conv_states`       | `(18, batch, 6144, 3)`        | float32 | 18层线性注意力的conv状态       |
| 输出  | `present_recurrent_states`  | `(18, batch, 16, 128, 128)`   | float32 | 18层线性注意力的recurrent状态  |
| 输出  | `present_kv_cache`          | `(12, batch, 2, max_seq_len, 256)` | float32 | 6层全注意力的固定 shape KV cache（12=6×2） |

> conv_dim=6144 = (key_dim×2 + value_dim) = (128×16×2 + 128×16) = 6144; conv_kernel_size=4, state保留3个历史值; max_seq_len 由 `--max-seq-len` 指定，默认 2048

**LLM Decode** — `decoder/qwen3_5_llm_decode.onnx`（固定 shape）

| 方向   | 名称                        | Shape                              | Dtype   | 说明                       |
|------|---------------------------|------------------------------------|---------|---------------------------|
| 输入  | `input_ids`               | `(batch, 1)`                       | int64   | 单步token                   |
| 输入  | `attention_mask`           | `(batch, max_seq_len)`             | int64   | 固定长度注意力掩码（前序≥1，后续=0）    |
| 输入  | `position_ids`             | `(4, batch, 1)`                    | int64   | 单步MRoPE位置               |
| 输入  | `past_conv_states`          | `(18, batch, 6144, 3)`             | float32 | 上一步conv状态               |
| 输入  | `past_recurrent_states`     | `(18, batch, 16, 128, 128)`        | float32 | 上一步recurrent状态          |
| 输入  | `past_kv_cache`             | `(12, batch, 2, max_seq_len, 256)`  | float32 | 固定 shape KV cache（由 max_seq_len 填充） |
| 输出  | `logits`                  | `(batch, 1, 248320)`               | float32 | 单步logits                 |
| 输出  | `present_conv_states`        | `(18, batch, 6144, 3)`             | float32 | 更新后的conv状态             |
| 输出  | `present_recurrent_states`   | `(18, batch, 16, 128, 128)`        | float32 | 更新后的recurrent状态        |
| 输出  | `present_kv_cache`           | `(12, batch, 2, max_seq_len, 256)`  | float32 | 更新后的 KV cache（固定 shape） |

---

## 3. ONNX 转 MindIR

### 转换命令

> **重要**：所有三个模型转换时都必须指定对应的 `--configFile`，以确保 Ascend 推理精度与 PyTorch 一致。不使用配置文件会导致 Vision Tower 在 Ascend FP16 下产生严重精度损失。

```bash
Convert=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Vision 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_onnx_static/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_onnx_static/qwen3_5_vision \
  --optimize=ascend_oriented \
  --configFile=configs/config_vision.ini \
  --saveType=MINDIR

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_onnx_static/prefill/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_onnx_static/qwen3_5_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=configs/config_prefill.ini \
  --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_5_onnx_static/decoder/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_onnx_static/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --configFile=configs/config_decoder.ini \
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
| `--configFile` | 各模型对应的配置文件（**所有模型都必须指定**） |

### 配置文件

各模型使用独立的配置文件（位于 `configs/` 目录下）：

**`configs/config_vision.ini`**

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All

[acl_build_options]
input_format="ND"
input_shape="pixel_values:-1,1536;grid_h:-1;grid_w:-1"
ge.dynamicDims="256,16,16;4096,64,64"
```

**`configs/config_prefill.ini`**

```ini
[acl_init_options]
ge.exec.precision_mode = force_fp32

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul

[acl_build_options]
input_format='ND'
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:4,1,-1;image_embeds:-1,1024"
ge.dynamicDims="82,82,82,64;1874,1874,1874,1024"
```

**`configs/config_decoder.ini`**

```ini
[acl_init_options]
ge.exec.precision_mode = force_fp32

[ascend_context]
plugin_custom_ops=BatchMatmulToMatmul
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen3_5_onnx_static/
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
python infer_qwen3_5_0_8b_mslite.py \
  --vision-model qwen3_5_onnx_static/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_onnx_static/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_onnx_static/qwen3_5_llm_decode_graph.mindir \
  --processor ./Qwen3.5-0.8B \
  --image "https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg" \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --device ascend \
  --device-id 0
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--vision-model`   | Vision MindIR 模型路径                    | 必填                      |
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                      |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                      |
| `--processor`      | HuggingFace processor 路径              | `./Qwen3.5-0.8B`        |
| `--image`          | 输入图像路径或 URL                          | `https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg`          |
| `--prompt`         | 输入文本                                  | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--image-size`     | 图像尺寸 | `256`                   |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |
| `--device-id`      | Ascend 设备 ID                          | `0`                     |

### 外部资源说明

- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

### 推理示例输出

使用示例图片 `https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg`：

```text
Loading vision model from ./qwen3_5_onnx_static/qwen3_5_vision.mindir...
Loading prefill model from ./qwen3_5_onnx_static/qwen3_5_llm_prefill_graph.mindir...
Loading decode model from ./qwen3_5_onnx_static/qwen3_5_llm_decode_graph.mindir...
Loading processor from ./Qwen3.5-0.8B...
Running vision tower...
Vision time: 13.56 ms
Running LLM prefill...
Prefill time: 799.09 ms
Running LLM decode (zero-copy)...
Total decode time: 6281.00 ms, avg decode step: 49.46 ms, steps: 127
Total time: 7093.65 ms, throughput: 18.04 tok/s

==================================================
Input Prompt: Describe this image.
Generated Response: This is a banana on a wooden table.
==================================================
```

---

## 5. 性能数据

### 测试环境

| 项目   | Atlas 300I Duo          | Atlas 800I A2          |
|------|------------------------|------------------------|
| 硬件   | Atlas 300I Duo（Ascend NPU） | Atlas 800I A2（Ascend NPU） |
| 模型   | Qwen3.5-0.8B           | Qwen3.5-0.8B           |
| 图片   | https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg             | https://hbr.org/resources/images/article_assets/2018/03/mar18_9_824179306.jpg             |
| 图像尺寸 | 256 × 256              | 256 × 256              |
| 精度   | force_fp32（config.ini）  | force_fp32（config.ini）  |

### 各模型推理输入 Shape 与性能

**Vision Tower**

| 项目           | Atlas 300I Duo | Atlas 800I A2 |
|--------------|----------------|----------------|
| 输入 Shape    | `(64, 1536)`   | `(64, 1536)`   |
| 输入 Dtype    | float32        | float32        |
| 输出 Shape    | `(16, 1024)`   | `(16, 1024)`   |
| 推理耗时        | **13.56 ms**   | **5 ms**     |

> 输入 64 个 patch（128/16 × 128/16 = 8×8=64），1536 = 3ch × 2temporal × 16patch × 16patch。输出 16 个图像 token（spatial_merge_size=2 压缩后 4×4=16）。

**LLM Prefill**

| 项目           | Atlas 300I Duo | Atlas 800I A2 |
|--------------|----------------|----------------|
| input_ids Shape  | `(1, 82)`     | `(1, 82)`     |
| attention_mask Shape | `(1, 82)`  | `(1, 82)`  |
| position_ids Shape | `(4, 1, 82)`  | `(4, 1, 82)`  |
| image_embeds Shape | `(64, 1024)`  | `(64, 1024)`  |
| 输出 next_token_id Shape | `(1, 1)`  | `(1, 1)`  |
| 推理耗时        | **799.09 ms**  | **14.0 ms**    |

> seq_len=82 包含：系统 prompt token + 64 个图像 token + 用户文本 token + 生成 prompt token

**LLM Decode（单步）**

| 项目           | Atlas 300I Duo | Atlas 800I A2 |
|--------------|----------------|----------------|
| input_ids Shape  | `(1, 1)`      | `(1, 1)`      |
| attention_mask Shape | `(1, 82+step)` | `(1, 82+step)` |
| position_ids Shape | `(4, 1, 1)`   | `(4, 1, 1)`   |
| past_kv_cache Shape | `(12, 1, 2, 82+step, 256)` | `(12, 1, 2, 82+step, 256)` |
| 输出 logits Shape | `(1, 1, 248320)` | `(1, 1, 248320)` |
| 单步平均耗时      | **49.46 ms**   | **5.4 ms**     |

### 端到端推理性能对比

| 指标                       | Atlas 300I Duo | Atlas 800I A2 |
|--------------------------|----------------|----------------|
| Vision Tower             | 13.56 ms       | 5.4 ms         |
| LLM Prefill              | 799.09 ms      | 14.0 ms        |
| LLM Decode（127 steps）   | 6281.00 ms     | 719 ms              |
| **总耗时**                  | **7093.65 ms** | **738 ms**              |
| **Avg decode step**       | **49.46 ms**   | **5.4 ms**     |
| **吞吐量**                  | **18.04 tok/s** | **173.27 tok/s**              |
| **生成 token 数**           | **127**        | **127**              |

> 注意：Atlas 800I A2 相比 Atlas 300I Duo 在 Prefill 和 Decode 阶段均有显著性能提升。

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-0.8B 官方文档](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 7. 许可证

本教程遵循 Qwen3.5-0.8B 模型的许可证。
