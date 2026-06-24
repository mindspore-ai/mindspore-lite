# Qwen2.5-Omni-3B ONNX 导出与推理

本目录提供 Qwen2.5-Omni-3B Thinker Text LLM 导出为 ONNX 以及端到端推理的完整脚本。

## 概览

Qwen2.5-Omni-3B 是一个多模态全能模型，支持文本、图像、音频等多种输入模态。本目录提取其 Thinker 组件中的文本 LLM 骨干网络，导出为 ONNX 格式用于文本生成任务。

- **ONNX 导出**：将 Thinker 文本 LLM 导出为 ONNX 模型
- **MindSpore Lite 集成**：将 ONNX 转换为 `.mindir` 以便在 Ascend 上部署
- **文本推理**：基于 MindIR 模型完成文本生成

> **注意**：本目录仅包含 Thinker 的文本 LLM 部分，不包含 Vision/Audio 编码器以及 Talker/Token2Wav 组件。

## 环境依赖

### 依赖版本

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11+ |
| torch          | 2.7+ |
| transformers   | 5.0+ |
| onnx           | 1.21+ |
| numpy          | 2.x |
| mindspore-lite | 2.8+ |

### 模型下载

请从 ModelScope 下载模型权重到当前目录：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-Omni-3B --local_dir ./Qwen2.5-Omni-3B
```

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen2_5_omni_3b_onnx.py \
    --model-id ./Qwen2.5-Omni-3B \
    --output-dir ./qwen2_5_omni_3b_onnx \
    --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 HuggingFace ID | `Qwen/Qwen2.5-Omni-3B` |
| `--output-dir` | 导出输出目录 | `./qwen2_5_omni_3b_onnx` |
| `--device` | 导出设备 | `cpu` |

导出产物：
- `qwen2_5_omni_3b_text.onnx`：Thinker 文本 LLM 模型

## MindSpore Lite 集成

如需在 Ascend 上部署，可将 ONNX 转换为 `.mindir`：

```bash
# 转换 Text LLM 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen2_5_omni_3b_onnx/qwen2_5_omni_3b_text.onnx \
    --outputFile=./qwen2_5_omni_3b_onnx/qwen2_5_omni_3b_text \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件示例

#### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

随后用 MindSpore Lite 推理：

```bash
python infer_qwen2_5_omni_3b_mslite.py \
    --prefill-model ./qwen2_5_omni_3b_onnx/qwen2_5_omni_3b_text_graph.mindir \
    --decode-model ./qwen2_5_omni_3b_onnx/qwen2_5_omni_3b_text_graph.mindir \
    --tokenizer ./Qwen2.5-Omni-3B \
    --prompt "你好，请介绍一下你自己。" \
    --max-new-tokens 128 \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | Tokenizer 路径 | `./Qwen2.5-Omni-3B` |
| `--prompt` | 输入文本 | `你好，请介绍一下你自己。` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--device` | MindSpore Lite 设备 | `ascend` |
| `--device-id` | Ascend device id | `0` |

## 模型 I/O 说明

### Text LLM 模型

**输入：**
- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`

**输出：**
- `logits`：`float16`，形状 `(batch, seq_len, vocab_size)`

## 目录结构

```Shell
qwen2.5_omni_3b/
├── export_qwen2_5_omni_3b_onnx.py          # ONNX 导出脚本
├── infer_qwen2_5_omni_3b_mslite.py         # MindSpore Lite 推理脚本
├── README.md                                # 本说明
├── configs/
│   └── config.ini                           # 转换配置文件
└── qwen2_5_omni_3b_onnx/                   # 导出模型目录
    ├── qwen2_5_omni_3b_text.onnx            # Thinker 文本 LLM ONNX 模型
    ├── *_graph.mindir                       # MindIR 图结构文件
    └── *_variables/                         # MindIR 权重目录
```

## 关键点

### 模型架构

Qwen2.5-Omni-3B 采用 Thinker-Talker 架构：
- **Thinker**：负责理解输入并生成文本表示
- **Talker**：负责将文本转换为语音输出

本目录仅导出 Thinker 的文本 LLM 部分，用于文本生成任务。

### 动态形状

模型通过 ONNX dynamic axes 支持动态 batch 与序列长度。推理时通过 Prefill Gears 机制选择最优的序列长度对齐。

## 常见问题

### 导出时内存不足（OOM）

- 使用 `--device cpu` 在 CPU 上导出
- 关闭其它占用内存的程序

### 推理结果不理想

- 确保使用正确的 tokenizer
- 本目录仅包含文本 LLM，不包含多模态能力

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni)
- [Qwen2.5-Omni HuggingFace](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)

## 许可证

本工具遵循 Qwen2.5-Omni 模型的许可证要求，详见 [Qwen2.5-Omni license](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)。
