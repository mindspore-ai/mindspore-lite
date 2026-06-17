# Qwen3-VL-Embedding-2B ONNX/MindIR 导出与推理完整教程

本教程详细介绍如何将 Qwen3-VL-Embedding-2B 多模态嵌入模型导出为 ONNX 格式，并使用 ONNX Runtime 进行推理，最后转换为 MindIR 并基于 MindSpore Lite Python API 推理。

## 目录

1. [环境准备](#环境准备)
2. [依赖安装](#依赖安装)
3. [模型导出](#模型导出)
4. [ONNX 推理](#onnx-推理)
5. [MindSpore Lite 转换](#mindspore-lite-转换)
6. [MindIR 推理](#mindir-推理)
7. [常见问题](#常见问题)

## 环境准备

### 系统要求

- Linux 系统（推荐 Ubuntu 20.04+）
- Python 3.9+（推荐 3.10 / 3.11）

## 依赖安装

本示例涉及三段流程（ONNX 导出 / ONNX 推理 / MindIR 推理），依赖按“必选/可选”列出。

### 必选依赖（导出 + ONNX 推理）

- mindspore_lite
- torch（用于导出）
- transformers（用于加载模型与 processor/tokenizer）
- accelerate（transformers `device_map` 相关依赖，建议安装）
- numpy
- onnx
- onnxruntime（CPU 推理）

### 可选依赖

- onnxruntime-gpu（如需 CUDA 推理）
- scikit-learn（如需在 ONNX 推理脚本中计算相似度）
- pillow（如需多模态输入图片）

```bash
# 导出 + ONNX 推理（CPU）
pip install torch transformers accelerate numpy onnx onnxruntime

# 相似度计算（可选）
pip install scikit-learn

# CUDA 推理（可选，二选一安装）
# pip install onnxruntime-gpu

# MindIR 推理（可选：安装 MindSpore Lite Python wheel）
# pip install /path/to/mindspore_lite-*.whl

# 多模态输入图片（可选）
pip install pillow
```

### 环境校验

```bash
# Python 版本
python -V

# 核心依赖版本
python -c "import numpy as np; print('numpy:', np.__version__)"
python -c "import torch; print('torch:', torch.__version__, 'cuda_available:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"
python -c "import onnx; print('onnx:', onnx.__version__)"
python -c "import onnxruntime as ort; print('onnxruntime:', ort.__version__, 'providers:', ort.get_available_providers())"

# 可选依赖校验（按需执行）
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
python -c "import PIL; print('pillow:', PIL.__version__)"
python -c "import mindspore_lite as mslite; print('mindspore_lite:', mslite.__version__)"
```

## 模型导出

### 导出脚本说明

导出脚本将 Qwen3-VL-Embedding-2B 模型导出为单个 ONNX 文件：

- **Embedding Model** (`qwen3_vl_embedding_2b.onnx`): 完整的嵌入模型，支持文本和图像输入

### 导出命令

```bash
cd examples/base_models/qwen3_vl_embedding_2b

# 使用默认参数导出
python export_qwen3_vl_embedding_onnx.py

# 自定义参数导出
python export_qwen3_vl_embedding_onnx.py \
  --model-id Qwen/Qwen3-VL-Embedding-2B \
  --output-dir ./qwen3_vl_embedding_onnx \
  --device cpu
```

### 参数说明

- `--model-id`: HuggingFace 模型 ID（默认：Qwen/Qwen3-VL-Embedding-2B）
- `--output-dir`: 输出目录（默认：./qwen3_vl_embedding_onnx）
- `--device`: 导出设备（cpu 或 cuda，默认：cpu）

### 导出输出

成功导出后，输出目录将包含以下文件：

```bash
qwen3_vl_embedding_onnx/
└── qwen3_vl_embedding_2b.onnx     # Embedding 模型
```

### 导出过程说明

导出过程：

1. **加载模型**: 从 HuggingFace 加载 Qwen3-VL-Embedding-2B 模型（FP16 精度）
2. **创建包装器**: 创建 PyTorch 包装器以适配 ONNX 导出
3. **导出模型**: 使用 torch.onnx.export 导出模型

### 导出注意事项

- 使用 FP16 精度导出以减少模型大小
- 使用 `torch._dynamo.disable()` 避免动态编译问题
- 使用 opset_version=18 确保兼容性
- 支持文本和图像输入（pixel_values 和 image_grid_thw）

## ONNX 推理

### 推理脚本说明

推理脚本实现了完整的嵌入推理流程：

1. 使用 processor 处理文本
2. 使用 ONNX 模型计算嵌入向量
3. 可选：计算嵌入向量之间的相似度

### 推理命令

```bash
cd examples/base_models/qwen3_vl_embedding_2b

# 基本推理
python infer_qwen3_vl_embedding_onnx.py \
  --model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --texts "Hello world" "Hi there" "Good morning"

# 自定义参数推理
python infer_qwen3_vl_embedding_onnx.py \
  --model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --tokenizer ./Qwen3-VL-Embedding-2B \
  --texts "What is machine learning?" "Deep learning uses neural networks." "Python is a programming language." \
  --device cpu

# 计算相似度矩阵
python infer_qwen3_vl_embedding_onnx.py \
  --model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --texts "The cat is on the mat" "A dog is running" "The feline is resting" \
  --compute-similarity
```

### 参数说明

- `--model`: ONNX 模型路径
- `--tokenizer`: HuggingFace tokenizer 路径（默认：./Qwen3-VL-Embedding-2B）
- `--texts`: 文本列表（空格分隔）
- `--device`: 推理设备（cpu 或 cuda，默认：cpu）
- `--compute-similarity`: 计算嵌入向量之间的相似度矩阵

### 推理示例

```bash
# 文本嵌入
python infer_qwen3_vl_embedding_onnx.py \
  --model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --texts "Machine learning is a subset of AI" "Deep learning uses neural networks" "Python is a programming language"

# 语义相似度
python infer_qwen3_vl_embedding_onnx.py \
  --model ./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --texts "The cat is sleeping" "A feline is resting" "The car is driving" \
  --compute-similarity
```

## MindSpore Lite 转换

### 转换 ONNX 模型

```bash
# 转换 Embedding 模型
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./mindspore-lite/examples/base_models/qwen3_vl_embedding_2b/qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.onnx \
  --outputFile=./mindspore-lite/examples/base_models/qwen3_vl_embedding_2b/qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### 转换参数说明

- `--fmk`: 输入模型格式（ONNX）
- `--modelFile`: 输入 ONNX 模型路径
- `--outputFile`: 输出 MINDIR 模型路径（不带扩展名）
- `--optimize`: 优化选项（ascend_oriented）
- `--saveType`: 保存类型（MINDIR）

### 转换输出

成功转换后，输出目录将包含：

```bash
qwen3_vl_embedding_onnx/
├── qwen3_vl_embedding_2b.onnx
└── qwen3_vl_embedding_2b.mindir
```

## MindIR 推理

### 依赖安装（MindSpore Lite Python）

需要安装 mindspore_lite Python 包（建议使用编译产物 wheel 安装）：

```bash
# 安装 MindSpore Lite Python wheel（示例，按实际 wheel 路径修改）
pip install /path/to/mindspore_lite-*.whl
```

### 最小推理代码（示例）

下面示例展示如何用 MindSpore Lite Python API 加载 `.mindir` 并执行一次推理（输入名与 dtype 以实际转换出来的模型为准）：

```python
import numpy as np
import mindspore_lite as mslite

context = mslite.Context()
context.target = ["ascend"]

model = mslite.Model()
model.build_from_file(
    "./qwen3_vl_embedding_onnx/qwen3_vl_embedding_2b.mindir",
    mslite.ModelType.MINDIR,
    context,
)

inputs = model.get_inputs()
for t in inputs:
    print("input:", t.name, t.shape, t.data_type)

# 仅示例：请按模型真实输入名/shape/dtype 准备数据
feed = {}
for t in inputs:
    if t.name == "input_ids":
        feed[t.name] = np.ones((1, 8), dtype=np.int64)
    elif t.name == "attention_mask":
        feed[t.name] = np.ones((1, 8), dtype=np.int64)
    elif t.name == "pixel_values":
        feed[t.name] = np.zeros((1, 3, 224, 224), dtype=np.float16)
    elif t.name in ("image_grid_thw", "grid_thw"):
        feed[t.name] = np.array([[1, 16, 16]], dtype=np.int64)

mslite_inputs = [mslite.Tensor(feed[t.name]) for t in inputs]
outputs = model.predict(mslite_inputs)
out0 = outputs[0].get_data_to_numpy()
print("output[0] shape:", out0.shape)
```

## 常见问题

### 1. transformers 版本过低

**错误信息**:

```bash
ImportError: cannot import name 'AutoModel' from 'transformers'
```

**解决方案**:

```bash
# 检查当前版本
python -c "import transformers; print(transformers.__version__)"

# 如果版本过低，升级 transformers（在当前环境中）
pip install --upgrade transformers

# 或者安装特定版本
pip install transformers==4.40.0
```

### 2. 模型下载失败

**错误信息**:

```bash
OSError: Can't load model from 'Qwen/Qwen3-VL-Embedding-2B'
```

**解决方案**:

```bash
export HF_ENDPOINT=https://hf-mirror.com

git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
python export_qwen3_vl_embedding_onnx.py --model-id ./Qwen3-VL-Embedding-2B
```

> **外部资源说明**：`HF_ENDPOINT=https://hf-mirror.com` 和 `git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B` 仅作为模型下载失败时的手动下载示例；导出脚本本身未硬编码权重下载 URL，生产或离线环境可直接传入本地权重目录。

### 3. ONNX 导出失败

**错误信息**:

```bash
RuntimeError: Failed to export embedding model
```

**解决方案**:

```bash
pip install --upgrade transformers

python -c "import torch; print(torch.__version__)"

# 尝试使用不同的 opset 版本（修改脚本中的 opset_version 参数）
```

## 性能优化建议

- 使用 FP16 精度减少模型大小（默认已使用）
- 调整 dummy_seq 参数以匹配实际使用场景
- 批量处理多个文本
- 针对目标设备优化模型

## 文件结构

```bash
qwen3_vl_embedding_2b/
├── export_qwen3_vl_embedding_onnx.py          # ONNX 导出脚本
├── infer_qwen3_vl_embedding_onnx.py           # ONNX 推理脚本
├── README.md                                  # 本教程文档
└── qwen3_vl_embedding_onnx/                    # 导出输出目录
    ├── qwen3_vl_embedding_2b.onnx             # Embedding 模型
    └── qwen3_vl_embedding_2b.mindir            # Embedding MINDIR（转换后）
```

## 参考资源

- [Qwen3-VL-Embedding-2B 官方文档](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)
- [MindSpore Lite 文档](https://www.mindspore.cn/lite)

## 许可证

本教程遵循 Qwen3-VL-Embedding-2B 模型的许可证。
