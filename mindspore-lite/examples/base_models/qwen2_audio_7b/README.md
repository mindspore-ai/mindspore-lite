# Qwen2-Audio-7B ONNX 导出与推理

本目录提供 Qwen2-Audio-7B 文本 LLM 导出为 ONNX 以及基于 MindSpore Lite 端到端推理的完整脚本。

## 概览

Qwen2-Audio-7B 是一个音频-语言多模态大模型，基于 Whisper-large-v3 音频编码器和 Qwen2-7B 文本生成模型。本目录提取其文本 LLM 骨干网络，导出为 ONNX 格式用于文本生成任务。

- **ONNX 导出**：将 Qwen2-7B 文本 LLM 导出为 ONNX 模型
- **MindSpore Lite 集成**：将 ONNX 转换为 `.mindir` 以便在 Ascend 上部署
- **文本推理**：基于 MindIR 模型完成贪婪解码文本生成

> **注意**：本目录仅包含文本 LLM 部分，不包含音频编码器（Whisper）和多模态投影层。

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

### 硬件要求

- Ascend Atlas 300I Duo / Atlas 800I A2 / Atlas 300I Pro
- 磁盘空间：≥ 50GB（含模型权重、ONNX 和 MINDIR 文件）

### 模型下载

请从 ModelScope 下载模型权重到当前目录：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2-Audio-7B --local_dir ./Qwen2-Audio-7B
```

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen2_audio_7b_onnx.py \
    --model-id ./Qwen2-Audio-7B \
    --output-dir ./qwen2_audio_7b_onnx \
    --device cpu
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | 模型路径或 ModelScope ID | `Qwen/Qwen2-Audio-7B` |
| `--output-dir` | 导出输出目录 | `./qwen2_audio_7b_onnx` |
| `--device` | 导出设备 | `cpu` |

#### 导出产物

- `qwen2_audio_7b_text.onnx`：文本 LLM ONNX 模型（含外部数据文件）

### 2. 转换为 MINDIR

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen2_audio_7b_onnx/qwen2_audio_7b_text.onnx \
    --outputFile=./qwen2_audio_7b_onnx/qwen2_audio_7b_text \
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
python infer_qwen2_audio_7b_mslite.py \
    --model ./qwen2_audio_7b_onnx/qwen2_audio_7b_text_graph.mindir \
    --tokenizer ./Qwen2-Audio-7B \
    --prompt "你好，请介绍一下你自己。" \
    --max-new-tokens 64 \
    --device ascend \
    --device-id 0
```

#### 推理参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | MindIR 图文件路径 | 必填 |
| `--tokenizer` | Tokenizer 路径 | `./Qwen2-Audio-7B` |
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

以下数据基于 Atlas 300I Duo (Ascend 310P3) 测试，输入 24 tokens，输出 64 tokens：

| 指标 | 数值 |
|------|------|
| Prefill 延迟 | 155.84 ms |
| 平均 Decode 延迟 | 129.08 ms/step |
| 最小 Decode 延迟 | 115.46 ms/step |
| 最大 Decode 延迟 | 144.68 ms/step |
| 总推理时间 | 8287.69 ms |
| 吞吐量 | 7.72 tok/s |

> **说明**：性能数据采用贪婪解码（token-by-token re-encode）模式测得，解码阶段每步需重新编码完整序列，因此 decode 延迟随序列长度线性增长。

## 精度验证

MINDIR 推理结果与 PyTorch 原始模型推理结果完全一致（贪婪解码），Top-5 token IDs 和生成文本完全匹配。

| 验证项 | 结果 |
|--------|------|
| Top-5 Token IDs | ✅ 完全一致 |
| 生成文本 | ✅ 完全一致 |
| 余弦相似度 | 0.999969 |
| 最大绝对误差 | 0.351562 |

## 目录结构

```
qwen2_audio_7b/
├── export_qwen2_audio_7b_onnx.py          # ONNX 导出脚本
├── infer_qwen2_audio_7b_mslite.py         # MindSpore Lite 推理脚本
├── README.md                               # 本说明文档
├── configs/
│   └── config.ini                          # 转换配置文件
├── Qwen2-Audio-7B/                        # 模型权重目录
└── qwen2_audio_7b_onnx/                   # 导出模型目录
    ├── qwen2_audio_7b_text.onnx            # ONNX 模型文件
    ├── qwen2_audio_7b_text_graph.mindir    # MindIR 图结构文件
    └── qwen2_audio_7b_text_variables/      # MindIR 权重目录
```

## 常见问题

### 导出时内存不足（OOM）

- 使用 `--device cpu` 在 CPU 上导出
- 关闭其它占用内存的程序
- 建议系统内存 ≥ 64GB

### 转换时 WARNING 日志

- 转换过程中出现的 WARNING 日志（如 `ConstantOfShape` 相关）不影响最终模型功能，可安全忽略

### 推理结果不理想

- 确保使用正确的 tokenizer（需与模型权重配套）
- 本目录仅包含文本 LLM，不包含音频理解能力
- Qwen2-Audio-7B 为 base 模型，未经过指令微调，生成质量可能不如 Instruct 版本

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen2-Audio GitHub](https://github.com/QwenLM/qwen2-audio)
- [Qwen2-Audio HuggingFace](https://huggingface.co/Qwen/Qwen2-Audio-7B)
- [Qwen2-Audio ModelScope](https://modelscope.cn/models/qwen/Qwen2-Audio-7B)
- [Qwen2-Audio 技术报告](https://arxiv.org/abs/2407.10759)

## 许可证

本工具遵循 Qwen2-Audio 模型的许可证要求，详见 [Qwen2-Audio license](https://huggingface.co/Qwen/Qwen2-Audio-7B)。
