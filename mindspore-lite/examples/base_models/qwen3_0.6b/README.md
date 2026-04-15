# Qwen3-0.6B ONNX 导出与推理完整教程

本教程详细介绍如何将 Qwen3-0.6B 纯文本模型导出为 ONNX 格式，并使用 ONNX Runtime 进行推理，最后转换为 MindSpore Lite 格式。

## 目录

1. [环境准备](#环境准备)
2. [依赖安装](#依赖安装)
3. [模型导出](#模型导出)
4. [ONNX 推理](#onnx-推理)
5. [MindSpore Lite 转换](#mindspore-lite-转换)
6. [常见问题](#常见问题)

## 环境准备

### 系统要求

- Python 3.11

- Linux 系统（推荐 Ubuntu 20.04+）

## 依赖安装

### 检查现有依赖

```bash
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"
```

### 安装缺失的依赖

```bash
pip install onnx onnxruntime
```

### 验证安装

```bash
python -c "import torch; import transformers; import onnx; import onnxruntime; print('All dependencies installed successfully!')"
```

## 模型导出

### 导出脚本说明

导出脚本将 Qwen3-0.6B 模型拆分为两个 ONNX 文件：

1. **LLM Prefill** (`qwen3_llm_prefill.onnx`): 处理预填充阶段（处理输入 prompt）

2. **LLM Decode** (`qwen3_llm_decode.onnx`): 处理解码阶段（自回归生成）

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_0.6b

python export_qwen3_onnx.py

python export_qwen3_onnx.py \
  --model-id Qwen/Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu
```

### 参数说明

- `--model-id`: HuggingFace 模型 ID（默认：Qwen/Qwen3-0.6B）

- `--output-dir`: 输出目录（默认：./qwen3_onnx）

- `--device`: 导出设备（cpu 或 cuda，默认：cpu）

### 导出输出

成功导出后，输出目录将包含以下文件：

```bash
qwen3_onnx/
├── qwen3_llm_prefill.onnx     # LLM Prefill
└── qwen3_llm_decode.onnx      # LLM Decode
```

### 导出过程说明

导出过程：

1. **加载模型**: 从 HuggingFace 加载 Qwen3-0.6B 模型

2. **Prefill 导出**: 导出处理输入 prompt 的模型

3. **Decode 导出**: 导出自回归生成的模型

这种分步导出方式可以减少内存占用，并支持流式推理。

## ONNX 推理

### 推理脚本说明

推理脚本实现了完整的端到端推理流程：

1. 使用 LLM Prefill 处理输入 prompt

2. 使用 LLM Decode 进行自回归生成

3. 支持 KV cache 管理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_0.6b

python infer_qwen3_onnx.py \
  --prefill ./qwen3_onnx/qwen3_llm_prefill.onnx \
  --decode ./qwen3_onnx/qwen3_llm_decode.onnx \
  --prompt "Hello, how are you?"

python infer_qwen3_onnx.py \
  --prefill ./qwen3_onnx/qwen3_llm_prefill.onnx \
  --decode ./qwen3_onnx/qwen3_llm_decode.onnx \
  --tokenizer ./Qwen3-0.6B \
  --prompt "Write a short story about a robot." \
  --max-new-tokens 256 \
  --device cpu
```

### 参数说明

- `--prefill`: Prefill ONNX 模型路径
- `--decode`: Decode ONNX 模型路径
- `--tokenizer`: HuggingFace tokenizer 路径（默认：Qwen/Qwen3-0.6B）
- `--prompt`: 输入文本提示（默认："Hello, how are you?"）
- `--max-new-tokens`: 最大生成 token 数（默认：128）
- `--device`: 推理设备（cpu 或 cuda，默认：cpu）

### 推理示例

```bash
python infer_qwen3_onnx.py \
  --prefill ./qwen3_onnx/qwen3_llm_prefill.onnx \
  --decode ./qwen3_onnx/qwen3_llm_decode.onnx \
  --prompt "What is the capital of France?"

python infer_qwen3_onnx.py \
  --prefill ./qwen3_onnx/qwen3_llm_prefill.onnx \
  --decode ./qwen3_onnx/qwen3_llm_decode.onnx \
  --prompt "Write a Python function to calculate factorial:" \
  --max-new-tokens 200

python infer_qwen3_onnx.py \
  --prefill ./qwen3_onnx/qwen3_llm_prefill.onnx \
  --decode ./qwen3_onnx/qwen3_llm_decode.onnx \
  --prompt "Summarize the following text in one sentence: [your text here]" \
  --max-new-tokens 100
```

## MindSpore Lite 转换

### 转换 ONNX 模型

```bash
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR

./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_decode.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### 转换参数说明

- `--fmk`: 输入模型格式（ONNX）
- `--modelFile`: 输入 ONNX 模型路径
- `--outputFile`: 输出 MINDIR 模型路径（不带扩展名）
- `--optimize`: 优化模式（ascend_oriented），昇腾硬件，必须指定ascend_oriented
- `--saveType`: 保存类型（MINDIR）

### 转换输出

成功转换后，输出目录将包含：

```bash
qwen3_onnx/
├── qwen3_llm_prefill.onnx
├── qwen3_llm_prefill.mindir
├── qwen3_llm_decode.onnx
└── qwen3_llm_decode.mindir
```

## 常见问题

### 1. transformers 版本过低

**错误信息**:

```bash
ImportError: cannot import name 'AutoModelForCausalLM' from 'transformers'
```

**解决方案**:

```bash
python -c "import transformers; print(transformers.__version__)"

pip install --upgrade transformers

pip install transformers==4.40.0
```

### 2. 内存不足

**错误信息**:

```bash
RuntimeError: CUDA out of memory
```

**解决方案**:

```bash
python export_qwen3_onnx.py --device cpu
```

### 3. 模型下载失败

**错误信息**:

```bash
OSError: Can't load model from 'Qwen/Qwen3-0.6B'
```

**解决方案**:

```bash
export HF_ENDPOINT=https://hf-mirror.com

git clone https://huggingface.co/Qwen/Qwen3-0.6B
python export_qwen3_onnx.py --model-id ./Qwen3-0.6B
```

### 4. ONNX 导出失败

**错误信息**:

```bash
RuntimeError: Failed to export LLM prefill
```

**解决方案**:

```bash
pip install --upgrade transformers

python -c "import torch; print(torch.__version__)"
```

### 5. 推理结果不正确

**可能原因**:

1. tokenizer 版本不匹配
2. KV cache 维度不正确
3. 模型精度问题，如果精度存在问题，建议使用fp32进行推理

**解决方案**:

```bash
python infer_qwen3_onnx.py \
  --tokenizer Qwen/Qwen3-0.6B \
  ...
```

### 6. MindSpore Lite 转换失败

**错误信息**:

```bash
Error: Unsupported operator
```

**解决方案**:

可以在MindSpore Lite官方社区提交相关issue进行反馈，等待官方修复

## 性能优化建议

### 1. 导出优化

- 使用 FP16 精度减少模型大小（默认已使用）
- 使用 GPU 加速导出过程（如果有 CUDA）
- 调整 dummy_seq 参数以匹配实际使用场景

### 2. 推理优化

- 使用 ONNX Runtime GPU 加速
- 批量处理多个 prompt
- 调整 max-new-tokens 参数

### 3. 部署优化

- 使用 MindSpore Lite 进行量化
- 针对目标设备优化模型
- 使用异步推理提高吞吐量

## 文件结构

```bash
qwen3_0.6B/
├── export_qwen3_onnx.py          # ONNX 导出脚本
├── infer_qwen3_onnx.py           # ONNX 推理脚本
├── README.md                     # 本教程文档
└── qwen3_onnx/                   # 导出输出目录
    ├── qwen3_llm_prefill.onnx    # Prefill 模型
    ├── qwen3_llm_decode.onnx     # Decode 模型
    ├── qwen3_llm_prefill.mindir  # Prefill MINDIR（转换后）
    └── qwen3_llm_decode.mindir   # Decode MINDIR（转换后）
```

## 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

- [Qwen3-0.6B 官方文档](https://huggingface.co/Qwen/Qwen3-0.6B)

- [Transformers 文档](https://huggingface.co/docs/transformers)

- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本教程遵循 Qwen3-0.6B 模型的许可证。
