# ViT-Base-Patch16-224 ONNX 导出与推理教程

本教程介绍如何将 `google/vit-base-patch16-224` 模型导出为 **单个统一 ONNX 模型**，并分别使用 ONNX Runtime 与 MindSpore Lite（MindIR）进行推理验证。

## 目录

- [环境准备](#环境准备)
- [模型导出](#模型导出)
- [模型转换](#模型转换)
- [推理测试](#推理测试)
- [常见问题](#常见问题)
- [参考资料](#参考资料)
- [许可证](#许可证)

## 环境准备

### 1. Python 环境

建议使用 Python 3.11 或更高版本；建议使用 PyTorch 2.1 或更高版本。

```bash
python --version
```

### 2. 安装依赖

ONNX 导出与 ONNX Runtime 推理依赖：

```bash
pip install -U torch transformers onnx onnxruntime pillow numpy
```

如需运行 `infer_vit_base_patch16_224_mslite.py`，还需确保当前 Python 环境可正常导入 `mindspore_lite`。

### 3. 验证安装

```bash
python -c "import torch; import transformers; import onnx; import onnxruntime; import PIL; import numpy; print('deps ok')"
```

如需验证 MindSpore Lite Python API：

```bash
python -c "import mindspore_lite; print('mindspore_lite ok')"
```

## 模型导出

导出脚本：`export_vit_base_patch16_224.py`

### 1. 导出统一模型

```bash
cd ./mindspore-lite/examples/base_models/vit_base_patch16_224

python export_vit_base_patch16_224.py \
  --model-id google/vit-base-patch16-224 \
  --output-dir ./vit_onnx \
  --device cpu
```

导出产物：

- `vit_onnx/vit_unified.onnx`：统一模型（输入 `pixel_values`，输出 `logits`）

### 导出参数说明

- `--model-id`：HuggingFace 模型 ID 或本地模型目录，默认 `google/vit-base-patch16-224`
- `--output-dir`：ONNX 输出目录，默认 `./vit_onnx`
- `--device`：导出设备，支持 `cpu` 和 `cuda`
- `--opset`：ONNX opset 版本，默认 `15`
- `--dynamic-image-size`：导出动态高宽输入

### 2. 动态分辨率（可选）

默认导出使用固定分辨率 \(224 \times 224\)。若希望导出时支持动态 H/W，可打开：

```bash
python export_vit_base_patch16_224.py \
  --output-dir ./vit_onnx \
  --dynamic-image-size
```

说明：

- `--dynamic-image-size` 会启用 `interpolate_pos_encoding=True`，并将 `pixel_values` 的 `height/width` 设为动态轴。
- 动态形状可能降低部分转换链路的兼容性；如转换失败，建议先使用默认固定尺寸导出。

## 模型转换

将 ONNX 转换为 MindSpore Lite 可加载的 MindIR（`.mindir`），用于 `infer_vit_base_patch16_224_mslite.py`。下面示例沿用 `base_models` 目录中的转换写法；实际使用时请根据目标后端和本地 Lite 包能力调整转换参数。

### 1. 转换统一模型

```bash
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./vit_onnx/vit_unified.onnx \
  --outputFile=./vit_onnx/vit_unified \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

转换后得到：

- `vit_onnx/vit_unified.mindir`

## 推理测试

推理脚本支持单张图片或逗号分隔的 batch 输入（例如 `--image a.jpg,b.jpg`），并输出 TopK 分类结果与简单性能统计。

### 1. ONNX Runtime 推理

统一模型推理：

```bash
python infer_vit_base_patch16_224_onnx.py \
  --model ./vit_onnx/vit_unified.onnx \
  --model-id google/vit-base-patch16-224 \
  --image ./your_image.jpg \
  --device cpu \
```

可选参数：

- `--model-id`：标签映射与预处理对应的模型 ID，默认 `google/vit-base-patch16-224`
- `--warmup`：预热轮数（默认 5）
- `--runs`：统计轮数（默认 20）
- `--device`：`cpu` 或 `cuda`

### 2. MindSpore Lite 推理（MindIR）

统一 MindIR 推理：

```bash
python infer_vit_base_patch16_224_mslite.py \
  --model ./vit_onnx/vit_unified.mindir \
  --image ./your_image.jpg \
  --device cpu \
```

在 Ascend 上推理（示例）：

```bash
python infer_vit_base_patch16_224_mslite.py \
  --model ./vit_onnx/vit_unified.mindir \
  --image ./your_image.jpg \
  --device ascend \
  --device-id 0
```

可选参数：

- `--model-id`：标签映射与预处理对应的模型 ID，默认 `google/vit-base-patch16-224`
- `--warmup`：预热轮数（默认 5）
- `--runs`：统计轮数（默认 20）
- `--device`：`cpu` 或 `ascend`
- `--device-id`：Ascend 设备 ID，仅 `--device ascend` 时生效

## 常见问题

### Q1: 导出时报 transformers 不兼容或找不到模块

**现象：**

- `Error: transformers not found or version is incompatible.`

**解决方案：**

```bash
pip install -U transformers
python -c "import transformers; print(transformers.__version__)"
```

### Q2: ONNXRuntime 未安装

**现象：**

- `onnxruntime not installed. Please install: pip install onnxruntime`

**解决方案：**

```bash
pip install -U onnxruntime
```

### Q3: MindSpore Lite Python 包不可用

**现象：**

- `mindspore_lite not installed.`

**解决方案：**

- 确认当前环境中已安装并可导入 `mindspore_lite`
- 确认所使用的 Python 版本与 MindSpore Lite Python 包兼容
- 若仅验证 ONNX 链路，可先使用 `infer_vit_base_patch16_224_onnx.py`

### Q4: MindSpore Lite 转换失败或算子不支持

**建议：**

- 优先使用默认固定分辨率导出（不要开启 `--dynamic-image-size`）
- 检查转换日志定位具体不支持的算子
- 如为已知兼容性问题，可在 MindSpore Lite 社区反馈 issue

## 目录结构

```bash
vit_base_patch16_224/
├── export_vit_base_patch16_224.py            # ONNX 导出脚本（统一模型）
├── infer_vit_base_patch16_224_onnx.py        # ONNXRuntime 推理脚本
├── infer_vit_base_patch16_224_mslite.py      # MindSpore Lite（MindIR）推理脚本
├── README.md                                 # 本教程文档
└── vit_onnx/                                  # 导出与转换产物目录（示例）
    ├── vit_unified.onnx
    └── vit_unified.mindir
```

## 参考资料

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [ViT 模型（HuggingFace）](https://huggingface.co/google/vit-base-patch16-224)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本教程遵循 `google/vit-base-patch16-224` 模型的许可证要求，详见其 HuggingFace 页面说明。
