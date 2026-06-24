# Grounding-DINO-Base ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 IDEA-Research/grounding-dino-base 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Grounding-DINO-Base 是一个开放词汇（open-vocabulary）目标检测模型：输入图像和一段文字描述（如 `"a cat, a remote control"`），输出每个匹配对象的边界框 + 置信度 + 文字标签。模型被整合为单个 ONNX 文件，由三部分组成：

1. **视觉分支**：Swin Transformer backbone + 多尺度特征金字塔（FPN），输出 4 个尺度的视觉特征
2. **文本分支**：BERT 编码器，对输入文字标签进行编码
3. **跨模态 Transformer**：6 层 encoder + 6 层 decoder，通过多尺度可变形注意力（MSDA）实现视觉与文本特征的对齐

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本      |
|----------------|---------|
| Python         | 3.11    |
| torch          | 2.10.0  |
| transformers   | 4.57.0  |
| onnx           | 1.19.1  |
| onnxruntime    | 1.24.2  |
| numpy          | 2.4.4   |
| pillow         | 12.2.0  |
| CANN           | 8.5.0   |
| mindspore-lite | 2.9.0   |

```bash
pip install torch==2.10.0 transformers==4.57.0 onnx==1.19.1 onnxruntime==1.24.2 \
    numpy==2.4.4 pillow==12.2.0
```

### 获取模型权重

```bash
git clone https://www.modelscope.cn/AI-ModelScope/grounding-dino-base.git \
    ./grounding-dino-base
```

`MODEL_DIR` 需包含以下关键文件：

- `model.safetensors`：模型权重
- `config.json`：模型结构配置（含 `num_queries=900`、`d_model=256` 等）
- `preprocessor_config.json`：图像预处理参数（`image_mean`、`image_std`、`size`）
- `tokenizer.json` / `tokenizer_config.json` / `vocab.txt`：BERT 词表（含特殊 token `[CLS]=101, [SEP]=102, .=1012, ?=1029`）

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_grounding_dino_base_onnx.py \
  --model-dir ./grounding-dino-base \
  --output-dir ./outputs \
  --name grounding_dino_base.onnx \
  --opset 17
```

### 参数说明

| 参数            | 说明                                                                       | 默认值                                                  |
|---------------|--------------------------------------------------------------------------|------------------------------------------------------|
| `--model-dir` | HuggingFace 权重目录                                                        | `./grounding-dino-base`  |
| `--output-dir`| 输出目录                                                                    | `./outputs`                                          |
| `--name`      | 输出 ONNX 文件名                                                             | `grounding_dino_base.onnx`                           |
| `--opset`     | ONNX opset 版本                                                            | `17`                                                 |

### 产出

```text
outputs/
└── grounding_dino_base.onnx          # 单文件，~925 MB，opset=17，固定 shape
```

### 模型架构

Grounding-DINO-Base 的整体结构：

- **视觉分支**：Swin Transformer backbone + FPN，对 `1×3×800×1333` 输入生成 4 个多尺度特征图（共 22223 个视觉 token）
- **文本分支**：BERT 编码器，对最长 256 token 的文本做自注意力编码；为支持多 phrase 检测，采用块对角 `text_self_attention_masks`（phrase 之间互不可见）与块内 `text_position_ids`（每块从 0 重新计数）
- **跨模态 Transformer**：
  - 6 层 encoder，每层用 MSDA 跨 4 个特征层做可变形采样（encoder 侧 numQueries=22223）
  - 6 层 decoder，每层包含 self-attention + cross-attention MSDA（decoder 侧 numQueries=900）
- **检测头**：输出 900 个 query 的 `logits`（2562 类，含 no-object 类）与 `pred_boxes`（cx, cy, w, h，归一化坐标）

MSDA 在导出时通过热补丁（monkey-patch）将上游 C++ CUDA 算子替换为 `grid_sample` 路径（纯 PyTorch 实现），导出的 ONNX 图中共 48 个 GridSample 节点（12 层 × 4 个特征层），无需导出后再修改 ONNX 图。

### ONNX 模型输入输出 Shape

**Grounding-DINO-Base** — `grounding_dino_base.onnx`

| 方向   | 名称                          | Shape                | Dtype  | 说明                                              |
|------|-----------------------------|----------------------|--------|-------------------------------------------------|
| 输入  | `pixel_values`             | `(1, 3, 800, 1333)`  | float32 | 图像 RGB，按 ImageNet mean/std 归一化，zero-pad 到 800×1333 |
| 输入  | `pixel_mask`               | `(1, 800, 1333)`     | int32   | 图像有效区域标记（pad 区域为 0）                            |
| 输入  | `input_ids`                | `(1, 256)`           | int32   | 文本 token IDs（含 `[CLS]=101, [SEP]=102, .=1012, ?=1029`） |
| 输入  | `token_type_ids`           | `(1, 256)`           | int32   | 段 ID                                            |
| 输入  | `attention_mask`           | `(1, 256)`           | int32   | 文本注意力掩码                                        |
| 输入  | `text_self_attention_masks`| `(1, 256, 256)`      | bool    | 文本自注意力掩码（块对角，phrase 之间互不可见）                    |
| 输入  | `text_position_ids`        | `(1, 256)`           | int32   | 文本位置 ID（块内 0..k，每块重新计数）                        |
| 输出  | `logits`                   | `(1, 900, 2562)`     | float32 | 900 query × 2562 类（含 no-object 类）              |
| 输出  | `pred_boxes`               | `(1, 900, 4)`        | float32 | (cx, cy, w, h)，归一化到 [0, 1]                      |

> 2562 = 2561（config.num_labels）+ 1（no-object）；900 = num_queries；MSDA 每 query 在 4 个 level 各采 4 个点（num_levels=4, num_points=4）。

### 导出关键点

- **MSDA 通过热补丁替换为 `grid_sample`**：transformers 内置 `MultiScaleDeformableAttention.forward` 调用 C++ CUDA 算子，导出 ONNX 时不可追踪；本仓库在导出前通过 monkey-patch 将其替换为纯 PyTorch（`grid_sample`）实现，导出的 ONNX 图仅包含标准算子，ONNX Runtime 与 MindSpore Lite 都能直接执行，无需导出后再修改 ONNX 图。
- **文本 mask 与 position_ids 必须外部预计算**：`generate_masks_with_special_tokens_and_transfer_map` 混用 `torch.eye`（导出为不支持的 `EyeLike`）与数据依赖循环，wrapper 显式接收这两个张量作为输入。

---

## 3. ONNX 转 MindIR

### 转换命令

> **重要**：转换时必须指定 `--configFile=config.ini`（含 `force_fp32` 与固定 shape）。默认 `preferred_fp16` 在 attention 密集计算下出现 FP16 溢出，导致部分 query 的 `logits` 全 `-inf`、检测置信度严重偏低。

```bash
Convert=converter_lite

$Convert --fmk=ONNX \
  --modelFile=./outputs/grounding_dino_base.onnx \
  --outputFile=./outputs/grounding_dino_base \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出格式（MINDIR）                |
| `--configFile` | 配置文件路径（**必须指定**）    |

### 配置文件

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="pixel_values:1,3,800,1333;pixel_mask:1,800,1333;input_ids:1,256;token_type_ids:1,256;attention_mask:1,256;text_self_attention_masks:1,256,256;text_position_ids:1,256"

[ascend_context]
plugin_custom_ops=All

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

> `plugin_custom_ops=All` 启用 GridSampler 等 CANN 算子融合；固定 shape 是 `ascend_oriented` 编译 GE 图的前提，与 §2 中的输入输出 Shape 一一对应。

### 产出

```text
outputs/
└── grounding_dino_base.mindir         # ~685 MB（单文件，未超过 2GB 阈值）
```

执行日志：

```text
CONVERT RESULT SUCCESS:0
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_grounding_dino_base_mslite.py \
  --model ./outputs/grounding_dino_base.mindir \
  --model-dir ./grounding-dino-base \
  --image ./sample.jpg \
  --text "a cat, a remote control" \
  --threshold 0.25 \
  --text-threshold 0.25 \
  --device-id 0 \
  --warmup 3 \
  --runs 10
```

### 参数说明

| 参数                  | 说明                              | 默认值                                                  |
|---------------------|---------------------------------|------------------------------------------------------|
| `--model`           | MindIR 模型路径                     | `./outputs/grounding_dino_base.mindir`               |
| `--model-dir`       | HuggingFace 权重目录（用于 tokenizer） | `./grounding-dino-base`  |
| `--image`           | 输入图像路径                          | 必填                                                   |
| `--text`            | 逗号分隔的候选文字标签                     | `a cat`                                              |
| `--threshold`       | 检测框置信度阈值                        | `0.25`                                               |
| `--text-threshold`  | 每个 query 对应 token 的二值化阈值        | `0.25`                                               |
| `--device-id`       | Ascend 设备 ID                    | `0`                                                  |
| `--warmup`          | 预热轮数                            | `2`                                                  |
| `--runs`            | 计时轮数                            | `5`                                                  |

### 推理示例输出

使用 COCO val2017 示例图（[000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)，2 只猫 + 3 个遥控器）：

```text
[batch 0] detected 6 objects
  label='a remote control' score=0.3621 box=[39.09, 70.94, 175.85, 116.31]
  label='a cat' score=0.3757 box=[344.67, 22.33, 636.72, 376.42]
  label='a remote control' score=0.2980 box=[332.45, 74.37, 370.63, 187.32]
  label='a remote control' score=0.3365 box=[29.94, 56.88, 184.01, 128.78]
  label='a cat' score=0.3643 box=[10.26, 51.48, 317.04, 469.04]
  label='remote' score=0.2515 box=[319.5, 65.69, 383.2, 198.43]
Perf:
  warmup: 2 runs: 5
  input_build_ms_mean:  102.791  (image resize/normalize/pad + text tokenization)
  inference_ms_mean:    2779.789  (model forward on Ascend)
  postprocess_ms_mean:    5.123  (sigmoid + thresholding + phrase extraction)
  e2e_ms_mean:          2887.671
```

> 说明：所有输入固定为 batch=1、文本长度=256、图像 canvas=800×1333；推理脚本内部完成按比例缩放与 zero-pad，输出 boxes 会按原图 (height, width) 反缩放，直接对应原图坐标。MindSpore Lite 推理脚本为纯 numpy/PIL 实现（无 `import torch`）。性能计时分为三段：**Input Build**（图像预处理 + 文本 tokenization）、**Inference**（Ascend NPU 前向推理）、**Postprocess**（sigmoid + 阈值过滤 + phrase 提取）。

---

## 5. 性能数据

### 测试环境

| 项目   | 配置                                                    |
|------|-------------------------------------------------------|
| 硬件   | Atlas 300I Duo（Ascend NPU）                            |
| 模型   | grounding-dino-base                                   |
| 图片   | COCO val2017 [000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg) |
| 图像尺寸 | 800 × 1333（COCO 风格短长边）                                |
| 文本   | `"a cat, a remote control"`（9 token + zero-pad 到 256） |
| 数据类型   | force_fp32（config.ini）                                |

### 模型推理输入 Shape 与性能

**Grounding-DINO-Base**

| 项目                              | 值                                                                                                |
|---------------------------------|--------------------------------------------------------------------------------------------------|
| 输入名称                            | `pixel_values`, `pixel_mask`, `input_ids`, `token_type_ids`, `attention_mask`, `text_self_attention_masks`, `text_position_ids` |
| pixel_values Shape              | `(1, 3, 800, 1333)`                                                                              |
| pixel_mask Shape                | `(1, 800, 1333)`                                                                                 |
| input_ids Shape                 | `(1, 256)`                                                                                       |
| text_self_attention_masks Shape | `(1, 256, 256)`                                                                                  |
| 输出 logits Shape                 | `(1, 900, 2562)`                                                                                 |
| 输出 pred_boxes Shape             | `(1, 900, 4)`                                                                                    |
| Input Build 耗时                 | 102.79 ms                                                                                        |
| Inference 耗时                    | 2779.79 ms                                                                                       |
| Postprocess 耗时                  | 5.12 ms                                                                                          |

### 端到端推理性能

| 指标              | 耗时 (ms)    | 说明                                                   |
|-----------------|------------|------------------------------------------------------|
| Input Build     | 102.79     | 图像 resize/normalize/pad + 文本 tokenization            |
| Inference       | 2779.79    | 模型前向推理（Ascend NPU）                                  |
| Postprocess     | 5.12       | sigmoid + 阈值过滤 + phrase 提取                           |
| **端到端 (mean)**  | **2887.67**|                                                      |

> 说明：MSLite 检测到主目标（2 只猫 + 3 个遥控器），box 坐标与置信度与原始 PyTorch 模型对齐。MSLite 多保留了一个置信度 0.25 的边界检测；阈值降到 0.24 时检出集合与原始模型一致。

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [GroundingDINO 官方仓库](https://github.com/IDEA-Research/GroundingDINO)
- [HuggingFace grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 7. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游模型与代码许可证以其仓库为准（Apache License 2.0）。
