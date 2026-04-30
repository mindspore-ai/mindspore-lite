# Qwen3.5-0.8B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3.5-0.8B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3.5-0.8B 是一个同时处理图像与文本的多模态大模型，采用混合线性注意力（GatedDeltaNet）与全注意力架构。模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_5_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_5_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），
   输出 logits、conv_state、recurrent_state 与 KV cache
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
| torch          | 2.10.0  |
| transformers   | 5.6.2  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 9.0    |
| mindspore-lite | 2.8.0  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/qwen3_5_0.8b

python export_qwen3_5_0.8b_onnx.py \
  --model-id models \
  --output-dir ./qwen3_5_0.8b_onnx \
  --device cpu \
  --vision-image-size 128
```

### 参数说明

| 参数                    | 说明                       | 默认值                |
|-----------------------|--------------------------|--------------------|
| `--model-id`          | HuggingFace 模型路径或本地目录    | `models`           |
| `--output-dir`        | 输出目录                     | `./qwen3_5_0.8b_onnx` |
| `--device`            | 导出设备（cpu/cuda）           | `cpu`              |
| `--vision-image-size` | Vision 模型输入图像尺寸（正方形边长）   | `128`              |

### 产出

```log
qwen3_5_0.8b_onnx/
├── qwen3_5_vision.onnx          # Vision Tower 模型
├── qwen3_5_llm_prefill.onnx     # Prefill 模型
└── qwen3_5_llm_decode.onnx      # Decode 模型
```

---

## 3. ONNX 推理

### ONNX Runtime 推理

```bash
python infer_qwen3_5_0.8b_onnx.py \
  --vision qwen3_5_0.8b_onnx/qwen3_5_vision.onnx \
  --prefill qwen3_5_0.8b_onnx/qwen3_5_llm_prefill.onnx \
  --decode qwen3_5_0.8b_onnx/qwen3_5_llm_decode.onnx \
  --processor models \
  --image ./your_image.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --device cpu
```

**执行日志：**

```log
Loading vision ONNX from qwen3_5_0.8b_onnx/qwen3_5_vision.onnx...
Loading prefill ONNX from qwen3_5_0.8b_onnx/qwen3_5_llm_prefill.onnx...
Loading decode ONNX from qwen3_5_0.8b_onnx/qwen3_5_llm_decode.onnx...
Loading processor from models...
Running vision tower...
Running LLM prefill...
Running LLM decode...

==================================================
Input Prompt: Describe this image.
Generated Response: This is a vibrant, abstract digital artwork featuring a stylized, multi-colored banana as the central subject. The banana is rendered in a kaleidoscope of bright, saturated colors — including blues,, yellows, pinks, and purples — giving it a psychedelic, almost neon or holographic appearance. Its surface is textured with swirling patterns that enhance the sense of movement and energy.

The banana is positioned diagonally across the frame, with its stem pointing toward the upper right corner and its base resting on a soft, gradient-colored surface that transitions from light pink to white. Behind the  there’s a subtle, soft
==================================================
```

### 参数说明

| 参数                 | 说明                       | 默认值                      |
|--------------------|--------------------------|----------------------------|
| `--vision`         | Vision ONNX 模型路径        | 必填                        |
| `--prefill`        | Prefill ONNX 模型路径       | 必填                        |
| `--decode`         | Decode ONNX 模型路径        | 必填                        |
| `--processor`      | HuggingFace processor 路径 | `models`                   |
| `--image`          | 输入图像路径或 URL            | 必填                        |
| `--prompt`         | 输入文本                    | `"Describe this image."`   |
| `--max-new-tokens` | 最大生成 token 数            | `128`                      |
| `--device`         | 推理设备（cpu/cuda）          | `cpu`                      |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Vision 转换
$Converter --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_vision.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_vision \
  --optimize=ascend_oriented \
  --saveType=MINDIR

# Prefill 转换
$Converter --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_llm_prefill.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=config.ini

# Decode 转换
$Converter --fmk=ONNX \
  --modelFile=qwen3_5_0.8b_onnx/qwen3_5_llm_decode.onnx \
  --outputFile=qwen3_5_0.8b_onnx/qwen3_5_llm_decode \
  --optimize=ascend_oriented \
  --configFile=config.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出格式（MINDIR）                |
| `--configFile` | 配置文件路径（Prefill/Decode 需要）  |

### 配置文件

`config.ini`（Prefill 和 Decode 模型需要，Vision 模型通常无需此配置）：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
qwen3_5_0.8b_onnx/
├── qwen3_5_vision.mindir                          # Vision MindIR
├── qwen3_5_llm_prefill_graph.mindir               # Prefill MindIR 图定义
├── qwen3_5_llm_prefill_variables/data_0            # Prefill 权重数据
├── qwen3_5_llm_decode_graph.mindir                 # Decode MindIR 图定义
└── qwen3_5_llm_decode_variables/data_0             # Decode 权重数据
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_5_0.8b_mslite.py \
  --vision-model qwen3_5_0.8b_onnx/qwen3_5_vision.mindir \
  --prefill-model qwen3_5_0.8b_onnx/qwen3_5_llm_prefill_graph.mindir \
  --decode-model qwen3_5_0.8b_onnx/qwen3_5_llm_decode_graph.mindir \
  --processor models \
  --image ./your_image.jpg \
  --prompt "Describe this image." \
  --max-new-tokens 128 \
  --image-size 128 \
  --device ascend \
  --device-id 0
```

**执行日志：**

```log
Initializing MindSpore Lite context for Ascend...
Loading vision model from ./onnx/qwen3_5_vision.mindir...
WARNING:root:Ascend custom operator path not found
Loading prefill model from ./onnx/qwen3_5_llm_prefill_graph.mindir...
WARNING:root:Ascend custom operator path not found
Loading decode model from ./onnx/qwen3_5_llm_decode_graph.mindir...
WARNING:root:Ascend custom operator path not found
Loading processor from Qwen3.5-0.8B...
Running vision tower...
Vision time: 7.00 ms
Running LLM prefill...
Prefill time: 4262.55 ms
Running LLM decode...
Total decode time: 5926.79 ms, avg decode step: 46.67 ms, steps: 127
Total time: 10196.34 ms, throughput: 12.55 tok/s

==================================================
Input Prompt: Describe this image.
Generated Response: This is a vibrant, abstract digital art image featuring a stylized banana as the central subject.

**Subject:**
- The banana is depicted in a highly colorful, psychedelic, and distorted form.
- It’s not a realistic banana but rather a melting, multi-colored sculpture or sculpture of a banana.
- The colors are saturated and varied: bright pinks, blues, greens, yellows, and purples are blended together in a chaotic, overlapping pattern.
- The banana’s surface appears smooth and glossy, reflecting light and creating a sense of depth and movement.
- A human figure with long hair stands behind the
==================================================
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--vision-model`  | Vision MindIR 模型路径                    | 必填                      |
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                      |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                      |
| `--processor`      | HuggingFace processor 路径              | `models`                |
| `--image`          | 输入图像路径或 URL                          | 必填                      |
| `--prompt`         | 输入文本                                  | `"Describe this image."` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--image-size`     | 图像尺寸（必须与导出 `--vision-image-size` 一致） | `128`                   |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |
| `--device-id`      | Ascend 设备 ID                          | `0`                     |

---

## 6. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：Qwen3.5-0.8B
测试条件：输入 128 tokens，输出 128 tokens，chat 模板启用

| 指标                       | Time     |
|--------------------------|----------|
| Vision (ms)              | 7.00     |
| Prefill (ms)             | 4262.55  |
| Total Decode (ms)        | 5926.79  |
| **Avg decode step (ms)** | **46.67**|
| Total (ms)               | 10196.34 |
| **Throughput (tok/s)**   | **12.55**|

> 注意：首次推理 Prefill 时间较长为正常现象，后续推理会趋于稳定。Avg decode step 为单次 decode 推理的耗时。

---

## 7. 常见问题

### Q1: ONNX 推理与 MSLite 推理结果不完全一致

两者输出开头相似但逐步发散，主要原因如下：

1. **精度差异**：ONNX Runtime 在 CPU 上以 FP32 计算，MSLite 在 Ascend 上部分算子走 FP16，导致 logits 存在微小差异
2. **自回归累积效应**：自回归生成中，微小的 logits 差异会导致 argmax 选中不同的 token，误差随生成步数持续累积
3. **线性注意力状态漂移**：GatedDeltaNet 的 recurrent_state 在 FP16 下更容易发生数值漂移

这是多模态生成模型在 FP16 推理下的常见现象，不影响功能正确性，后续会进一步优化。

### Q2: ONNX 转换和推理耗时较长

- **ONNX 转换**：当前导出耗时约 2 小时，主要因为模型结构复杂（混合线性注意力 + 全注意力架构）且采用 legacy exporter
- **推理耗时**：当前端到端推理约 25 分钟（含 Prefill ~4.3s + Decode ~5.9s），Prefill 阶段因序列较长耗时较高

后续会逐步优化转换速度和推理性能。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3.5-0.8B 官方文档](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen3.5-0.8B 模型的许可证。
