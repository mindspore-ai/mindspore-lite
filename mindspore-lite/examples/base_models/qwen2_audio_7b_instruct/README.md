# Qwen2-Audio-7B-Instruct ONNX 导出与推理

本目录提供 Qwen2-Audio-7B-Instruct 文本 LLM 导出为 ONNX 以及基于 MindSpore Lite 端到端推理的完整脚本。

## 概览

Qwen2-Audio-7B-Instruct 是 Qwen2-Audio-7B 的指令微调版本，基于 Whisper-large-v3 音频编码器和 Qwen2-7B 文本生成模型，支持语音对话和音频分析等任务。本目录提取其文本 LLM 骨干网络，导出为 ONNX 格式用于文本生成任务。

- **ONNX 导出**：将 Qwen2-7B 文本 LLM 导出为 ONNX 模型
- **MindSpore Lite 集成**：将 ONNX 转换为 `.mindir` 以便在 Ascend 上部署
- **文本推理**：基于 MindIR 模型完成贪婪解码文本生成

> **注意**：本目录仅包含文本 LLM 部分，不包含音频编码器（Whisper）和多模态投影层。Instruct 版本经过指令微调，文本生成质量优于 base 版本。

## 模型架构

| 组件 | 实现 | 说明 |
|------|------|------|
| `audio_tower` | Qwen2AudioEncoder (Whisper-large-v3) | 音频编码器（本目录不包含） |
| `multi_modal_projector` | Linear(1280→3584) | 多模态投影层（本目录不包含） |
| `language_model` | Qwen2ForCausalLM (Qwen2-7B) | 文本生成 LLM |

导出的 ONNX 模型仅包含 `language_model.model`（Qwen2-7B backbone）和 `language_model.lm_head`。

## 环境依赖

### 依赖版本

| 软件包 | 版本 |
|--------|------|
| Python | 3.11+ |
| torch | 2.7+ |
| transformers | 5.0+ |
| onnx | 1.21+ |
| numpy | 2.x |
| mindspore-lite | 2.8+ |

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen2_audio_7b_instruct_onnx.py \
    --model-id ./Qwen2-Audio-7B-Instruct \
    --output-dir ./qwen2_audio_7b_instruct_onnx \
    --device cpu
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | 模型路径或 ModelScope ID | `Qwen/Qwen2-Audio-7B-Instruct` |
| `--output-dir` | 导出输出目录 | `./qwen2_audio_7b_instruct_onnx` |
| `--device` | 导出设备 | `cpu` |

#### 导出产物

- `qwen2_audio_7b_instruct_text.onnx`：文本 LLM ONNX 模型（含外部数据文件）

### 2. 转换为 MINDIR

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen2_audio_7b_instruct_onnx/qwen2_audio_7b_instruct_text.onnx \
    --outputFile=./qwen2_audio_7b_instruct_onnx/qwen2_audio_7b_instruct_text \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

#### config 文件示例

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

### 3. MindIR 推理

```bash
python infer_qwen2_audio_7b_instruct_mslite.py \
    --model ./qwen2_audio_7b_instruct_onnx/qwen2_audio_7b_instruct_text_graph.mindir \
    --tokenizer ./Qwen2-Audio-7B-Instruct \
    --prompt "你好，请介绍一下你自己。" \
    --max-new-tokens 64 \
    --device ascend \
    --device-id 0
```

#### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | MindIR 图文件路径 | 必填 |
| `--tokenizer` | Tokenizer 路径 | `./Qwen2-Audio-7B-Instruct` |
| `--prompt` | 输入文本 | `你好，请介绍一下你自己。` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--max-input-length` | 最大输入序列长度 | `256` |
| `--device` | 推理设备 | `ascend` |
| `--device-id` | Ascend device id | `0` |

## 模型 I/O 说明

### ONNX / MindIR 模型

**输入：**

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `input_ids` | int32 | `(batch, seq_len)` | 输入 token IDs |
| `attention_mask` | int32 | `(batch, seq_len)` | 注意力掩码 |

**输出：**

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `logits` | float16 | `(batch, seq_len, 156032)` | 词表 logits |

## 性能数据

以下数据基于 Atlas 300I Duo (Ascend 310P3) 测试，输入 24 tokens，输出 55 tokens：

| 指标 | 数值 |
|------|------|
| Prefill 延迟 | 139.04 ms |
| 平均 Decode 延迟 | 132.71 ms/step |
| 最小 Decode 延迟 | 115.92 ms/step |
| 最大 Decode 延迟 | 234.12 ms/step |
| 总推理时间 | 7305.28 ms |
| 吞吐量 | 7.53 tok/s |

> **说明**：性能数据采用贪婪解码（token-by-token re-encode）模式测得，解码阶段每步需重新编码完整序列，因此 decode 延迟随序列长度线性增长。

## 常见问题

### 导出时内存不足（OOM）

- 使用 `--device cpu` 在 CPU 上导出
- 关闭其它占用内存的程序
- 建议系统内存 ≥ 64GB

### 转换时 WARNING 日志

- 转换过程中出现的 WARNING 日志（如 `ConstantOfShape` 相关）不影响最终模型功能，可安全忽略

### 推理结果与 base 版本的差异

- Instruct 版本经过指令微调，回复更加自然流畅
- 本目录仅包含文本 LLM，不包含音频理解能力

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/qwen2-audio)
- [Qwen2-Audio-Instruct HuggingFace](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- [Qwen2-Audio-Instruct ModelScope](https://modelscope.cn/models/qwen/Qwen2-Audio-7B-Instruct)
- [Qwen2-Audio 技术报告](https://arxiv.org/abs/2407.10759)

## 许可证

本工具遵循 Qwen2-Audio 模型的许可证要求，详见 [Qwen2-Audio license](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)。
