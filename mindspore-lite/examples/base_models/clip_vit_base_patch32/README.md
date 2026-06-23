# CLIP-ViT-Base-Patch32 ONNX 导出与推理教程

本教程介绍如何将 `openai/clip-vit-base-patch32` 的 **视觉编码器（CLIPVisionModelWithProjection）** 导出为单个统一 ONNX 模型，并分别使用 ONNX Runtime 与 MindSpore Lite（MindIR）进行推理与精度对齐。

与完整的 `CLIPModel`（图像+文本双塔）不同，本教程只导出视觉塔，输出可直接用于图像检索、图文相似度计算，配合离线计算的文本嵌入即可完成零样本分类（zero-shot classification）。

## 目录

- [模型说明](#模型说明)
- [环境准备](#环境准备)
- [模型导出](#模型导出)
- [模型转换](#模型转换)
- [推理测试](#推理测试)
- [精度对齐](#精度对齐)
- [常见问题](#常见问题)
- [目录结构](#目录结构)
- [参考资料与许可证](#参考资料与许可证)

## 模型说明

`openai/clip-vit-base-patch32` 的视觉塔结构（`CLIPVisionConfig` 默认值）：

| 项目 | 取值 |
|---|---|
| 输入分辨率 | 224 × 224 |
| patch 大小 | 32 × 32 |
| 序列长度 | 49 patch + 1 CLS = 50 |
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| projection_dim | 512 |
| 归一化均值 | [0.48145466, 0.4578275, 0.40821073] |
| 归一化方差 | [0.26862954, 0.26130258, 0.27577711] |

本教程导出的统一模型 I/O：

- 输入：`pixel_values`，形状 `[batch, 3, 224, 224]`，dtype `float32`
- 输出 1：`image_embeds`，形状 `[batch, 512]`（经过 `visual_projection` 投影后的 CLIP 图像嵌入，用于检索/零样本分类）
- 输出 2：`last_hidden_state`，形状 `[batch, 50, 768]`（视觉 Transformer 最后一层隐状态，含 CLS token）

> 说明：文本编码器**不**纳入导出 ONNX。零样本分类所需的文本嵌入可在 CPU 侧用 HuggingFace `transformers` 一次性离线计算（见推理脚本的 `--zero-shot` 选项），随后与部署侧的图像嵌入做余弦相似度即可。这样部署模型保持单 ONNX、自包含。

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

如需运行 `infer_clip_vit_base_patch32_mslite.py`，还需确保当前 Python 环境可正常导入 `mindspore_lite`。

### 3. 验证安装

```bash
python -c "import torch; import transformers; import onnx; import onnxruntime; import PIL; import numpy; print('deps ok')"
```

如需验证 MindSpore Lite Python API：

```bash
python -c "import mindspore_lite; print('mindspore_lite ok')"
```

## 模型导出

导出脚本：`export_clip_vit_base_patch32_onnx.py`

### 1. 导出统一视觉模型

```bash
cd ./mindspore-lite/examples/base_models/clip_vit_base_patch32

python export_clip_vit_base_patch32_onnx.py \
  --model-id openai/clip-vit-base-patch32 \
  --output-dir ./clip_onnx \
  --device cpu
```

导出产物：

- `clip_onnx/clip_vision.onnx`：统一视觉模型（输入 `pixel_values`，输出 `image_embeds` 与 `last_hidden_state`）

### 导出参数说明

- `--model-id`：HuggingFace 模型 ID 或本地模型目录，默认 `openai/clip-vit-base-patch32`
- `--output-dir`：ONNX 输出目录，默认 `./clip_onnx`
- `--device`：导出设备，支持 `cpu` 和 `cuda`
- `--opset`：ONNX opset 版本，默认 `17`
- `--dynamic-image-size`：导出动态高宽输入（需启用 `interpolate_pos_encoding=True`，可能降低转换兼容性）

### 2. 动态分辨率（可选）

默认导出使用固定分辨率 (224 × 224)。若希望导出时支持动态 H/W，可打开：

```bash
python export_clip_vit_base_patch32_onnx.py \
  --output-dir ./clip_onnx \
  --dynamic-image-size
```

说明：

- `--dynamic-image-size` 会启用 `interpolate_pos_encoding=True`，并将 `pixel_values` 的 `height/width` 设为动态轴。
- 动态形状可能降低部分转换链路的兼容性；如转换失败，建议先使用默认固定尺寸导出。

## 模型转换

将 ONNX 转换为 MindSpore Lite 可加载的 MindIR（`.mindir`），用于 `infer_clip_vit_base_patch32_mslite.py`。下面示例沿用 `base_models` 目录中的转换写法；实际使用时请根据目标后端和本地 Lite 包能力调整转换参数。

### 1. 转换统一模型

```bash
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./clip_onnx/clip_vision.onnx \
  --outputFile=./clip_onnx/clip_vision \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

转换后得到：

- `clip_onnx/clip_vision.mindir`

### 2. 配置文件 `config.ini`

视觉塔为静态形状输入，配置如下：

```ini
[acl_build_options]
input_format="NCHW"
input_shape="pixel_values:1,3,224,224"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

- `input_shape` 固定 `pixel_values:1,3,224,224`，对应 batch=1 的 CLIP 视觉输入。
- `force_fp16` + `plugin_custom_ops=All` 为本目录推荐的视觉模型默认精度/算子策略。

### 转换参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出类型，`MINDIR`               |
| `--configFile` | 配置文件路径                      |

## 推理测试

推理脚本支持单张图片或逗号分隔的 batch 输入（例如 `--image a.jpg,b.jpg`），并输出嵌入形状、L2 范数、（可选）零样本分类结果与简单性能统计。

预处理默认走 `AutoImageProcessor(use_fast=False)`（与 CLIP 官方一致：resize→center crop 224→OpenAI 均值方差归一化）；可通过 `--no-processor` 切换到纯 numpy 实现，便于在无 `transformers` 的部署环境运行。

### 1. MindSpore Lite 推理（MindIR）

在 Ascend 上推理（示例）：

```bash
python infer_clip_vit_base_patch32_mslite.py \
  --model ./clip_onnx/clip_vision.mindir \
  --image ./your_image.jpg \
  --device ascend \
  --device-id 0
```

可选参数：

- `--model-id`：预处理与文本编码器对应的模型 ID，默认 `openai/clip-vit-base-patch32`
- `--no-processor`：使用纯 numpy 预处理
- `--zero-shot`：逗号分隔的候选标签，启用零样本分类
- `--warmup`：预热轮数（默认 5）
- `--runs`：统计轮数（默认 20）
- `--device`：`cpu` 或 `ascend`
- `--device-id`：Ascend 设备 ID，仅 `--device ascend` 时生效

### 2. 性能参考

测试环境：Ascend Atlas 300I Duo（Ascend NPU），CANN 8.5.0，MindSpore Lite 2.10.0，输入 224×224、batch=1，warmup 5 + 20 次计时。

| 后端 | 设备 | batch | latency_ms_mean  | 备注 |
|---|---|---|---|---|
| MindSpore Lite | Ascend(300I Duo) | 1 | **3.349** | force_fp16，Ascend后端推理 |

实测运行日志（`infer_clip_vit_base_patch32_mslite.py --device ascend`）：

```log
Output:
  image_embeds      shape=(1, 512) dtype=float32
  last_hidden_state shape=(1, 50, 768) dtype=float32
  embed_norm[0]     =11.180791
Perf:
  batch_size: 1
  warmup: 5 runs: 20
  latency_ms_mean: 3.349
  latency_ms_p50:  3.350
  latency_ms_p90:  3.376
  latency_ms_p99:  3.381
  mem_before: {'vmrss_kb': 1861888, 'vmhwm_kb': 2245848}
  mem_after:  {'vmrss_kb': 1861888, 'vmhwm_kb': 2245848}
```

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
- 若仅验证 ONNX 链路，可先使用 `infer_clip_vit_base_patch32_onnx.py`

### Q4: MindSpore Lite 转换失败或算子不支持

**建议：**

- 优先使用默认固定分辨率导出（不要开启 `--dynamic-image-size`）
- 检查 `config.ini` 中 `input_shape` 是否与 ONNX 输入一致（`pixel_values:1,3,224,224`）
- 检查转换日志定位具体不支持的算子
- 如为已知兼容性问题，可在 MindSpore Lite 社区反馈 issue

### Q5: 零样本分类（`--zero-shot`）报缺模块

**原因：** 零样本需要文本编码器，本脚本通过 `transformers`（`CLIPModel`）在 CPU 上即时计算文本嵌入。

**解决方案：**

```bash
pip install -U torch transformers
```

若部署环境不便安装 `transformers`，可预先在 CPU 侧离线计算并保存文本嵌入（`np.save`），推理时直接加载做余弦相似度。

## 目录结构

```bash
clip_vit_base_patch32/
├── export_clip_vit_base_patch32_onnx.py    # ONNX 导出脚本（视觉塔统一模型）
├── infer_clip_vit_base_patch32_mslite.py   # MindSpore Lite（MindIR）推理脚本
├── config.ini                              # converter_lite 转换配置
├── README.md                               # 本教程文档
└── clip_onnx/                              # 导出与转换产物目录（示例）
    ├── clip_vision.onnx
    └── clip_vision.mindir
```

## 参考资料与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [CLIP 模型（HuggingFace）](https://huggingface.co/openai/clip-vit-base-patch32)
- [Transformers CLIP 文档](https://huggingface.co/docs/transformers/model_doc/clip)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

本教程遵循 `openai/clip-vit-base-patch32` 模型的许可证要求，详见其 HuggingFace 页面说明。
