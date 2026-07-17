# Qwen3-ASR-1.7B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-ASR-1.7B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0 |
| transformers   | 4.57.6 |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 9.0    |
| mindspore-lite | 2.8.0  |
| qwen-asr       | 0.0.6  |

```bash
pip install transformers==4.57.6 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite qwen-asr==0.0.6
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_asr_1.7b

python export_qwen3_asr_1.7b_onnx.py \
  --model-path ./Qwen3-ASR-1.7B \
  --output-dir ./onnx \
  --kv-cache-len 1024 \
  --dtype fp32
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | HuggingFace 模型路径或本地目录 | `./Qwen3-ASR-1.7B` |
| `--output-dir` | 输出目录 | `./onnx` |
| `--opset` | ONNX opset 版本 | `18` |
| `--kv-cache-len` | KV cache 最大长度（音频 token + 模板 + 生成余量） | `1024` |
| `--dtype` | 导出精度。`fp32` 必选：FP16 权重在 Ascend matmul 上会溢出 | `fp32` |

### 产出

```log
onnx/
├── audio_encoder/
│   └── qwen3_asr_audio_encoder_fp32.onnx
├── prefill/
│   └── qwen3_asr_text_prefill_fp32.onnx   (+ external data)
└── decode/
    └── qwen3_asr_text_decode_fp32.onnx    (+ external data)
```

---

## 3. MindSpore Lite 转换

### 转换命令

```bash
Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Audio Encoder 转换（force_fp32）
$Converter --fmk=ONNX \
  --modelFile=onnx/audio_encoder/qwen3_asr_audio_encoder_fp32.onnx \
  --outputFile=onnx/audio_encoder/qwen3_asr_audio_encoder_fp32_graph \
  --optimize=ascend_oriented \
  --configFile=configs/qwen3_asr_audio_encoder.config

# Text Prefill 转换（动态分档 + lm_head 入图，force_fp32）
$Converter --fmk=ONNX \
  --modelFile=onnx/prefill/qwen3_asr_text_prefill_fp32.onnx \
  --outputFile=onnx/prefill/qwen3_asr_text_prefill_fp32_graph \
  --optimize=ascend_oriented \
  --configFile=configs/qwen3_asr_text_prefill.config

# Text Decode 转换（固定 shape + KV cache + lm_head 入图，force_fp32）
$Converter --fmk=ONNX \
  --modelFile=onnx/decode/qwen3_asr_text_decode_fp32.onnx \
  --outputFile=onnx/decode/qwen3_asr_text_decode_fp32_graph \
  --optimize=ascend_oriented \
  --configFile=configs/qwen3_asr_text_decode.config
```

> 转换日志会打印 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`——这是 CANN 内部告警（prefill 图体积约 8GB），**不影响最终结果**。只要结尾出现 `CONVERT RESULT SUCCESS:0` 即成功。

### 配置文件

`configs/qwen3_asr_audio_encoder.config`（音频编码器，固定 shape）:

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

`configs/qwen3_asr_text_prefill.config`（Prefill，动态分档；`audio_features` 静态 390）:

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;audio_features:1,390,2048;attention_mask:1,-1;position_ids:3,1,-1"
ge.dynamicDims="512,512,512;640,640,640;768,768,768"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

`configs/qwen3_asr_text_decode.config`（Decode，固定 shape + KV cache）:

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,1024;position_ids:3,1,1;past_key_cache:28,1,8,1024,128;past_value_cache:28,1,8,1024,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

> `past_key_cache` 第一维 `28` = 文本 decoder 层数（全部层在单模型内）。

### 产出

```log
onnx/
├── audio_encoder/qwen3_asr_audio_encoder_fp32_graph.mindir
├── prefill/
│   └── qwen3_asr_text_prefill_fp32_graph.mindir
└── decode/
    └── qwen3_asr_text_decode_fp32_graph.mindir
```

---

## 4. MindSpore Lite 推理

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

English
Hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well. But he did very well when he started writing for other people.
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-path` | HuggingFace tokenizer / feature extractor 路径 | `./Qwen3-ASR-1.7B` |
| `--mindir-dir` | MindIR 模型目录（包含 audio_encoder/ prefill/ decode/ 子目录） | `./onnx` |
| `--audio` | 音频文件路径（支持常见音频格式） | 必填 |
| `--context` | 上下文提示词（可选，用于提升领域词识别） | `""` |
| `--language` | 指定语言（可选，不指定则自动识别） | `None` |
| `--max-chunk-sec` | 长音频分段长度（秒） | `30.0` |
| `--max-new-tokens` | 每段最大生成 token 数 | `256` |
| `--device-id` | Ascend 设备 ID | `0` |

---

## 5. 性能数据

### 端到端推理性能

测试模型：Qwen3-ASR-1.7B。音频 token 长度 390（30 秒音频 → 特征提取 3000 帧 → 按 `chunk_size = n_window * 2 = 200` 切 15 chunk → 每 chunk CNN 后 26 token → `15 × 26 = 390`）。

#### Atlas 800I A2

| 指标 | Chinese (ms) | English (ms) |
|---|---:|---:|
| FeatureExt (CPU) | 56.98 | 51.33 |
| AudioEncoder | 24.99 | 23.41 |
| Prefill | 31.80 | 30.70 |
| Decode (7 / 44 steps) | 45.50 | 270.16 |
| Host (argmax + D2H + detokenize) | 240.27 | 276.24 |
| **总耗时** | **399.54** | **651.84** |
| **Avg decode step** | **6.50** | **6.14** |
| **吞吐量** | **17.52 tok/s** | **67.50 tok/s** |
| **生成 token 数** | **7** | **44** |

#### Atlas 300I Duo

| 指标 | Chinese (ms) | English (ms) |
|---|---:|---:|
| FeatureExt (CPU) | 56.98 | 67.16 |
| AudioEncoder | 41.92 | 43.23 |
| Prefill | 143.99 | 145.31 |
| Decode (7 / 47 steps) | 232.26 | 1443.84 |
| Host (argmax + D2H + detokenize) | 653.95 | 654.86 |
| **总耗时** | **1129.10** | **2354.40** |
| **Avg decode step** | **33.18** | **30.72** |
| **吞吐量** | **6.20 tok/s** | **19.96 tok/s** |
| **生成 token 数** | **7** | **47** |

> **两套硬件独立对比**：Atlas 800I A2 与 Atlas 300I Duo 算力差距大（FP32 算力约 5 倍），跨硬件比较耗时无意义。

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Transformers 文档](https://huggingface.com/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [Qwen3-ASR-1.7B 模型](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)

---

## 7. 许可证

本教程遵循 Qwen3-ASR-1.7B 模型的许可证。
