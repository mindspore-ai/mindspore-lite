# Qwen3-ASR-1.7B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-ASR-1.7B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

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
| qwen-asr       | 0.0.6  |

```bash
pip install transformers==5.6.2 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite qwen-asr==0.0.6
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_asr_1.7b

python export_qwen3_asr_1.7b_onnx.py \
  --model-path ./Qwen3-ASR-1.7B \
  --output-dir ./onnx \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | HuggingFace 模型路径或本地目录 | `./Qwen3-ASR-1.7B` |
| `--output-dir` | 输出目录 | `./onnx` |
| `--opset` | ONNX opset 版本 | `17` |

### 产出

```log
onnx/
├── qwen3_asr_audio_encoder_fp32.onnx
├── qwen3_asr_text_decoder_fp32.onnx
└── *.onnx_data / *.data (external data)
```

---

## 3. ONNX 推理

### ONNX Runtime 推理

```bash
python infer_qwen3_asr_1.7b_onnx.py \
  --model-path ./Qwen3-ASR-1.7B \
  --onnx-dir ./onnx \
  --audio asr_zh.wav \
  --max-new-tokens 256 \
  --language Chinese
```

### 执行日志

```log
Chinese
甚至出现交易几乎停滞的情况。

English
Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well. But he did very well when he started writing for other people.
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | HuggingFace tokenizer / feature extractor 路径 | `./Qwen3-ASR-1.7B` |
| `--onnx-dir` | ONNX 模型目录（包含 audio encoder / text decoder） | `./onnx` |
| `--audio` | 音频文件路径（支持常见音频格式） | 必填 |
| `--context` | 上下文提示词（可选，用于提升领域词识别） | `""` |
| `--language` | 指定语言（可选，不指定则自动识别） | `None` |
| `--max-chunk-sec` | 长音频分段长度（秒） | `30.0` |
| `--max-new-tokens` | 每段最大生成 token 数 | `256` |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash
Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Audio Encoder 转换
$Converter --fmk=ONNX \
  --modelFile=onnx/qwen3_asr_audio_encoder_fp32.onnx \
  --outputFile=onnx/qwen3_asr_audio_encoder_fp32.onnx \
  --optimize=ascend_oriented \
  --configFile=utils/config.ini

# Text Decoder 转换
$Converter --fmk=ONNX \
  --modelFile=onnx/qwen3_asr_text_decoder_fp32.onnx \
  --outputFile=onnx/qwen3_asr_text_decoder_fp32.onnx \
  --optimize=ascend_oriented \
  --configFile=utils/config.ini
```

### 参数说明

| 参数 | 说明 |
|---|---|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出 MindIR 路径（不带扩展名） |
| `--optimize` | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径 |

### 配置文件

`utils/config.ini`:

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
onnx/
├── qwen3_asr_audio_encoder_fp32.onnx.mindir
├── qwen3_asr_text_decoder_fp32.onnx_graph.mindir
└── qwen3_asr_text_decoder_fp32.onnx_variables/
    └── data_0
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen3_asr_1.7b_mslite.py \
  --model-path ./Qwen3-ASR-1.7B \
  --mindir-dir ./onnx \
  --audio asr_zh.wav \
  --max-new-tokens 256 \
  --device-id 0 \
  --language Chinese
```

### 执行日志

```log
Chinese
甚至出现交易几乎停滞的情况。
Perf: AudioEncoder(ms) mean=24.49, min=24.49, max=24.49; DecodeStep(ms) mean=452.10, min=371.07, max=990.81; Throughput(tok/s)=2.21; TokenLength=390

English
Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well. But he did very well when he started writing for other people.
Perf: AudioEncoder(ms) mean=26.22, min=26.22, max=26.22; DecodeStep(ms) mean=387.90, min=370.34, max=995.17; Throughput(tok/s)=2.58; TokenLength=390
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | HuggingFace tokenizer / feature extractor 路径 | `./Qwen3-ASR-1.7B` |
| `--mindir-dir` | MindIR 模型目录 | `./mindir` |
| `--audio` | 音频文件路径（支持常见音频格式） | 必填 |
| `--context` | 上下文提示词（可选，用于提升领域词识别） | `""` |
| `--language` | 指定语言（可选，不指定则自动识别） | `None` |
| `--max-chunk-sec` | 长音频分段长度（秒） | `30.0` |
| `--max-new-tokens` | 每段最大生成 token 数 | `64` |
| `--device-id` | Ascend 设备 ID | `1` |
| `--config-path` | MindSpore Lite 配置文件路径 | `""` |
| `--precision-mode` | 可选精度模式（如 `force_fp32`） | `None` |

---

## 6. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：Qwen3-ASR-1.7B

本用例音频 token 长度为 390。计算方式为读取音频编码器输出的序列长度 `audio_features.shape[1]`；默认 30 秒音频会被特征提取为 3000 帧，按 `chunk_size = n_window * 2 = 200` 切成 15 个 chunk，每个 chunk 经过 CNN 后得到 26 个音频 token，因此 `15 * 26 = 390`。

| 指标 | Chinese音频 | English音频 |
|---|---:|---:|
| Audio Token Length | 390 | 390 |
| Audio Encoder (ms) | 24.49 | 26.22 |
| Text Decoder / step (ms) | 452.10 | 387.90 |
| Throughput (tok/s) | 2.21 | 2.58 |

> 说明：运行 `infer_qwen3_asr_1.7b_mslite.py` 后，会在末尾额外打印一行 `Perf:`，包含表格所需的 Mean、吞吐与 TokenLength 数据，可直接填入上述表格。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [Qwen3-ASR-1.7B 模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

---

## 8. 许可证

本教程遵循 Qwen3-ASR-1.7B 模型的许可证。
