# Jina Reranker V3 ONNX 导出与推理教程

本教程介绍如何将 Jina Reranker V3 模型导出为 ONNX 格式，并使用 MindSpore Lite 进行推理。

## 目录

- [环境准备](#环境准备)
- [模型导出](#模型导出)
- [模型转换](#模型转换)
- [推理测试](#推理测试)
- [常见问题](#常见问题)

## 环境准备

### 1. Python 环境

确保使用Python 3.11或者更高版本的Python环境：

```bash
# 检查 Python 版本
python --version
```

### 2. 安装依赖

```bash
pip install torch transformers onnx onnxruntime numpy
```

### 3. 验证安装

```bash
# 检查 transformers 版本
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 检查 torch 版本
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 模型导出

### 1. 导出统一模型

导出完整的 Jina Reranker V3 模型为单个 ONNX 文件：

```bash
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3 \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | Hugging Face 模型 ID 或本地路径 | jinaai/jina-reranker-v3 |
| `--output-dir` | ONNX 模型输出目录 | ./onnx |
| `--max-length` | 最大序列长度 | 8192 |
| `--device` | 导出设备 | cpu |

### 2. 导出输出

成功导出后，输出目录结构如下：

```bash
onnx/
└── jina_reranker_v3.onnx              # 统一模型
```

## 模型转换

使用 MindSpore Lite 将 ONNX 模型转换为 MindSpore Lite 格式：

### 1. 转换统一模型

```bash
# 进入 MindSpore Lite 根目录
cd /home/xsfd/WorkSpace/mindspore-lite

# 转换 ONNX 模型
./mindspore-lite/tools/converter/converter \
    --fmk=ONNX \
    --modelFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3.onnx \
    --outputFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3 \
    --optimize=ascend_oriented
```

## 推理测试

### 1. ONNX Runtime 推理

使用 ONNX Runtime 直接测试导出的模型：

#### 统一模型推理

```bash
python infer_jina_reranker_v3_onnx.py \
    --model-path ./onnx/jina_reranker_v3.onnx \
    --tokenizer jinaai/jina-reranker-v3 \
    --max-length 8192 \
    --device CPU
```

### 2. MindSpore Lite 推理

使用 `infer_jina_reranker_v3_mslite.py` 进行推理：

运行 MindSpore Lite 推理：

```bash
python infer_jina_reranker_v3_mslite.py \
    --model-path ./onnx/jina_reranker_v3.mindir \
    --tokenizer jinaai/jina-reranker-v3 \
    --device ascend
```

## 常见问题

### Q1: 转换时 MindSpore Lite 报错

**解决方案：**

1. 检查 ONNX 模型是否正确导出

2. 确认 MindSpore Lite 版本支持 ONNX opset 17

3. 查看转换日志中的具体错误信息

### Q2: 推理结果与原始模型不一致

**解决方案：**

1. 确认使用相同的分词器和预处理方式

2. 检查输入数据的格式是否正确

3. 验证模型转换时的精度设置

### Q4: 如何使用其他 Jina Reranker V3 变体

**解决方案：** 修改 `--model-id` 参数：

```bash
# 使用 small 版本
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3-small \
    --output-dir ./onnx

# 使用 large 版本
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3-large \
    --output-dir ./onnx
```

## 性能优化建议

1. **调整序列长度**：根据实际应用场景调整 `--max-length`，避免不必要的计算

2. **选择合适设备**：在支持 Ascend 的环境中可使用 `--device ascend` 获取更高性能

## 参考资料

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

- [Jina Reranker V3 官方文档](https://huggingface.co/jinaai/jina-reranker-v3)

## 许可证

本教程遵循 Apache 2.0 许可证。
