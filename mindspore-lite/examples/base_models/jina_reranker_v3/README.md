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
    --model-id jinaai/jina-reranker-v3-base \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-id` | Hugging Face 模型 ID 或本地路径 | jinaai/jina-reranker-v3-base |
| `--output-dir` | ONNX 模型输出目录 | ./onnx |
| `--max-length` | 最大序列长度 | 8192 |
| `--device` | 导出设备 | cpu |

### 2. 导出拆分模型

为了获得更好的推理性能，可以将模型拆分为编码器和分类头两部分：

```bash
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3-base \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu \
    --split
```

使用 `--split` 参数后，将生成两个 ONNX 文件(可以不拆分)：

- `jina_reranker_v3_encoder.onnx`：编码器部分（包含嵌入层和 Transformer 层）

- `jina_reranker_v3_head.onnx`：分类头部分

### 3. 导出输出

成功导出后，输出目录结构如下：

```bash
onnx/
├── jina_reranker_v3.onnx              # 统一模型
├── jina_reranker_v3_encoder.onnx      # 编码器（拆分模式）
└── jina_reranker_v3_head.onnx         # 分类头（拆分模式）
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

### 2. 转换拆分模型

```bash
# 转换编码器
./mindspore-lite/tools/converter/converter \
    --fmk=ONNX \
    --modelFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3_encoder.onnx \
    --outputFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3_encoder \
    --optimize=ascend_oriented

# 转换分类头
./mindspore-lite/tools/converter/converter \
    --fmk=ONNX \
    --modelFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3_head.onnx \
    --outputFile=./mindspore-lite/examples/base_models/jina_reranker_v3/onnx/jina_reranker_v3_head \
    --optimize=ascend_oriented
```

## 推理测试

### 1. ONNX Runtime 推理

使用 ONNX Runtime 直接测试导出的模型：

#### 统一模型推理

```bash
python infer_jina_reranker_v3_onnx.py \
    --model-path ./onnx/jina_reranker_v3.onnx \
    --tokenizer jinaai/jina-reranker-v3-base \
    --max-length 8192 \
    --device CPU
```

#### 拆分模型推理

```bash
python infer_jina_reranker_v3_onnx.py \
    --model-path ./onnx/jina_reranker_v3_encoder.onnx \
    --head-path ./onnx/jina_reranker_v3_head.onnx \
    --tokenizer jinaai/jina-reranker-v3-base \
    --max-length 8192 \
    --device CPU
```

### 2. MindSpore Lite 推理

创建 MindSpore Lite 推理脚本 `infer_jina_reranker_v3_mslite.py`：

```python
#!/usr/bin/env python3
import argparse
import numpy as np
import mindspore_lite as mslite
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description='Inference with Jina Reranker V3 using MindSpore Lite')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .ms model')
    parser.add_argument('--tokenizer', type=str, default='jinaai/jina-reranker-v3-base', help='Tokenizer model ID')
    parser.add_argument('--max-length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/ascend)')
    parser.add_argument('--device-id', type=int, default=0, help='Device ID for ascend')
    args = parser.parse_args()

    # 初始化上下文
    context = mslite.Context()
    context.target = [args.device]
    if args.device == "ascend":
        context.ascend.device_id = args.device_id

    # 加载模型
    model = mslite.Model()
    model.build_from_file(args.model_path, mslite.ModelType.MINDIR, context)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试数据
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]

    # 准备输入
    pairs = [f"Query: {q} Document: {d}" for q, d in zip(queries, documents)]
    inputs = tokenizer(pairs, padding=True, truncation=True, max_length=args.max_length, return_tensors="np")

    # 推理
    scores = []
    for i in range(len(queries)):
        mslite_inputs = [
            mslite.Tensor(inputs['input_ids'][i:i+1].astype(np.int32)),
            mslite.Tensor(inputs['attention_mask'][i:i+1].astype(np.int32)),
        ]
        outputs = model.predict(mslite_inputs)
        logits = outputs[0].get_data_to_numpy()
        score = logits[0, 1]
        scores.append(score)

    # 输出结果
    print("\nReranking scores:")
    for i, (query, doc, score) in enumerate(zip(queries, documents, scores)):
        print(f"\n[{i+1}] Score: {score:.4f}")
        print(f"Query: {query}")
        print(f"Document: {doc}")

if __name__ == '__main__':
    main()
```

运行 MindSpore Lite 推理：

```bash
python infer_jina_reranker_v3_mslite.py \
    --model-path ./onnx/jina_reranker_v3.mindir \
    --tokenizer jinaai/jina-reranker-v3-base \
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

### Q3: 拆分模型推理速度更慢

**解决方案：**
拆分模型的主要优势在于：

- 更好的内存管理

- 支持更灵活的部署策略

- 便于模型优化和量化

如果推理速度是主要关注点，建议使用统一模型。

### Q4: 如何使用其他 Jina Reranker V3 变体

**解决方案：** 修改 `--model-id` 参数：

```bash
# 使用 small 版本
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3-small \
    --output-dir ./onnx \
    --split

# 使用 large 版本
python export_jina_reranker_v3_onnx.py \
    --model-id jinaai/jina-reranker-v3-large \
    --output-dir ./onnx \
    --split
```

## 性能优化建议

1. **使用拆分模型**：对于大规模部署，拆分模型可以提供更好的资源利用率

2. **调整序列长度**：根据实际应用场景调整 `--max-length`，避免不必要的计算

3. **使用基于Ascend的高性能融合算子**：在支持Ascend的环境中使用Ascend进行推理

## 参考资料

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

- [Jina Reranker V3 官方文档](https://huggingface.co/jinaai/jina-reranker-v3-base)

## 许可证

本教程遵循 Apache 2.0 许可证。
