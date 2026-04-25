# Qwen3-VL-Reranker-2B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-VL-Reranker-2B 模型导出为 ONNX 格式（拆分为 Vision + Score 两段），转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0  |
| transformers   | 4.57.0  |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| CANN           | 9.0  |
| mindspore-lite | 2.8.0  |

```bash
pip install torch==2.10.0 numpy pillow onnx==1.19.1 onnxruntime==1.24.2 transformers==4.57.0 mindspore-lite==2.8.0
```

如需 GPU 推理（可选）：

```bash
pip install -U onnxruntime-gpu
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/base_models/qwen3_vl_reranker_2b

python export_qwen3_vl_reranker_2b_onnx.py \
  --model-id Qwen/Qwen3-VL-Reranker-2B \
  --output-dir ./onnx \
  --device cpu \
  --dtype fp16 \
  --vision-image-size 128 \
  --dummy-seq-len 128
```

### 参数说明

| 参数                   | 说明                                                         | 默认值                       |
|----------------------|------------------------------------------------------------|----------------------------|
| `--model-id`         | HuggingFace 模型路径或本地目录                                    | `Qwen/Qwen3-VL-Reranker-2B` |
| `--output-dir`       | 输出目录                                                       | `./onnx`                   |
| `--device`           | 导出设备（cpu/cuda）                                             | `cpu`                      |
| `--dtype`            | 导出精度（fp16/fp32），CPU 环境若 fp16 算子不支持可改成 fp32               | `fp16`                     |
| `--vision-image-size`| Vision 模型导出固定的图像边长（像素），**推理时 `--image-size` 必须一致**     | `128`                      |
| `--dummy-seq-len`    | 仅影响导出 tracing 的 dummy shape，不限制实际推理的 seq_len（score 模型声明了 dynamic axes） | `128` |

### 产出

```text
onnx/
├── qwen3_vl_reranker_vision.onnx
└── qwen3_vl_reranker_score.onnx
```

---

## 3. ONNX 推理

### 文本-文本 rerank

```bash
python infer_qwen3_vl_reranker_2b_onnx.py \
  --vision ./onnx/qwen3_vl_reranker_vision.onnx \
  --score  ./onnx/qwen3_vl_reranker_score.onnx \
  --processor Qwen/Qwen3-VL-Reranker-2B \
  --query "A woman playing with her dog on a beach at sunset." \
  --doc   "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset."
```

**执行日志：**

```log
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 15
============================================================
score: 0.862305
============================================================
```

### 文本-图像 / 图文 rerank

```bash
python infer_qwen3_vl_reranker_2b_onnx.py \
  --vision ./onnx/qwen3_vl_reranker_vision.onnx \
  --score  ./onnx/qwen3_vl_reranker_score.onnx \
  --processor Qwen/Qwen3-VL-Reranker-2B \
  --query "A woman playing with her dog on a beach at sunset." \
  --doc   "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset." \
  --doc-image ./demo.jpeg \
  --image-size 128
```

**执行日志：**

```log
onnxruntime cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 15
============================================================
score: 0.856934
============================================================
```

输出为 `score`（0~1），值越大表示越相关。

### 参数说明

| 参数              | 说明                                    | 默认值                       |
|-----------------|---------------------------------------|----------------------------|
| `--vision`      | Vision ONNX 模型路径                      | 必填                         |
| `--score`       | Score ONNX 模型路径                       | 必填                         |
| `--processor`   | HuggingFace processor 路径或本地目录          | `Qwen/Qwen3-VL-Reranker-2B` |
| `--query`       | 查询文本                                  | 必填                         |
| `--doc`         | 文档文本                                  | 必填                         |
| `--doc-image`   | 文档图片路径（可选）                             | 无                          |
| `--image-size`  | 图片尺寸，必须与导出 `--vision-image-size` 一致    | `128`                      |

---

## 4. MindSpore Lite 转换

### 转换命令

使用 MindSpore Lite `converter_lite` 将两个 ONNX 分别转换为 `.mindir`。

```bash
Converter=./output/bin/converter_lite

# Vision 转换
$Converter --fmk=ONNX \
  --modelFile=./onnx/qwen3_vl_reranker_vision.onnx \
  --outputFile=./onnx/qwen3_vl_reranker_vision \
  --optimize=ascend_oriented \
  --saveType=MINDIR

# Score 转换
$Converter --fmk=ONNX \
  --modelFile=./onnx/qwen3_vl_reranker_score.onnx \
  --outputFile=./onnx/qwen3_vl_reranker_score \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR
```

### 参数说明

| 参数             | 说明                                       |
|----------------|------------------------------------------|
| `--fmk`        | 输入模型格式（ONNX）                              |
| `--modelFile`  | 输入 ONNX 模型路径                              |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）                       |
| `--optimize`   | 优化模式，推荐 `ascend_oriented`                  |
| `--configFile` | 配置文件路径（Score 模型必须指定，配置 enforce_fp32）      |
| `--saveType`   | 保存类型（MINDIR）                              |

### 配置文件

Score 模型转换时**必须**使用配置文件 `config.ini`，设置 `enforce_fp32` 避免 FP16 计算溢出：

```ini
[acl_init_options]
ge.exec.precision_mode=enforce_fp32
```

> **重要**：Qwen3-VL-Reranker-2B 的 Score 模型在 FP16 精度下存在计算溢出问题，会导致推理输出恒为 0.5。必须通过 `config.ini` 设置 `enforce_fp32`，让 Ascend GE 在计算时使用 FP32 精度。详见常见问题第 10 条。

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
onnx/
├── qwen3_vl_reranker_vision.mindir
├── qwen3_vl_reranker_score_graph.mindir      # Score MindIR 图定义
└── qwen3_vl_reranker_score_variables/         # Score 权重数据
    └── data_0
```

---

## 5. MindSpore Lite 推理

### 文本-文本 rerank

```bash
python infer_qwen3_vl_reranker_2b_mslite.py \
  --vision-model ./onnx/qwen3_vl_reranker_vision.mindir \
  --score-model  ./onnx/qwen3_vl_reranker_score_graph.mindir \
  --processor Qwen/Qwen3-VL-Reranker-2B \
  --query "A woman playing with her dog on a beach at sunset." \
  --doc   "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset." \
  --device-id 0
```

**执行日志：**

```log
Loading vision model from ./onnx/qwen3_vl_reranker_vision.mindir...
Loading score model from ./onnx/qwen3_vl_reranker_score_graph.mindir...
Loading processor from Qwen/Qwen3-VL-Reranker-2B...
============================================================
score: 0.856934
============================================================

--- Performance ---
  Preprocessing:     55.66 ms
  Score inference:   964.83 ms
  Total:             1020.50 ms
```

### 文本-图像 / 图文 rerank

```bash
python infer_qwen3_vl_reranker_2b_mslite.py \
  --vision-model ./onnx/qwen3_vl_reranker_vision.mindir \
  --score-model  ./onnx/qwen3_vl_reranker_score_graph.mindir \
  --processor Qwen/Qwen3-VL-Reranker-2B \
  --query "A woman playing with her dog on a beach at sunset." \
  --doc   "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset." \
  --doc-image ./demo.jpeg \
  --image-size 128 \
  --device-id 0
```

**执行日志：**

```log
Loading vision model from ./onnx/qwen3_vl_reranker_vision.mindir...
Loading score model from ./onnx/qwen3_vl_reranker_score_graph.mindir...
Loading processor from Qwen/Qwen3-VL-Reranker-2B...
============================================================
score: 0.858398
============================================================

--- Performance ---
  Vision inference:  8.94 ms
  Preprocessing:     161.39 ms
  Score inference:   620.92 ms
  Total:             782.31 ms
```

### 参数说明

| 参数                | 说明                                        | 默认值                       |
|-------------------|-------------------------------------------|----------------------------|
| `--vision-model`  | Vision MindIR 模型路径                         | 必填                         |
| `--score-model`   | Score MindIR 模型路径（`*_graph.mindir`）       | 必填                         |
| `--processor`     | HuggingFace processor 路径或本地目录              | `Qwen/Qwen3-VL-Reranker-2B` |
| `--query`         | 查询文本                                      | 必填                         |
| `--doc`           | 文档文本                                      | 必填                         |
| `--doc-image`     | 文档图片路径（可选）                                 | 无                          |
| `--image-size`    | 图片尺寸，必须与导出 `--vision-image-size` 一致        | `128`                      |
| `--device-id`     | Ascend 设备 ID                               | `0`                        |

---

## 6. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：Qwen3-VL-Reranker-2B
测试条件：文本-文本 rerank，输入 ~95 tokens

| 指标                       | Time       |
|--------------------------|------------|
| Score inference (ms)     | 964.83     |
| Preprocessing (ms)       | 55.66      |
| **Total (ms)**           | **1020.50**|

测试条件：文本-图像 rerank，输入 ~113 tokens，图片 128x128

| 指标                       | Time       |
|--------------------------|------------|
| Vision inference (ms)    | 8.94       |
| Score inference (ms)     | 620.92     |
| Preprocessing (ms)       | 161.39     |
| **Total (ms)**           | **782.31** |

> 注意：以上为单次运行数据，Min/Max 待多次运行后补充。

---

## 7. 常见问题

### 1) MindSpore Lite 推理提示输入 dtype 不匹配（34 vs 35）

**现象**：`CheckInputTensors] ... required 34, given 35`。

**原因**：MindIR 模型输入期望 `int32(34)`，但脚本传入了 `int64(35)`。

**解决**：将 `input_ids / attention_mask / position_ids` 按模型输入 dtype 转成 `np.int32`。本目录的 `infer_qwen3_vl_reranker_2b_mslite.py` 已显式将上述输入转为 `np.int32`。

### 2) MindSpore Lite 图片预处理报 "Only returning PyTorch tensors is currently supported."

**现象**：`ValueError: Only returning PyTorch tensors is currently supported.`

**原因**：部分环境会默认加载 `Qwen2VLImageProcessorFast`，其 `preprocess()` 仅支持返回 PyTorch Tensor。

**解决**：使用 `AutoImageProcessor.from_pretrained(..., use_fast=False)` 强制使用慢版 image processor（支持 `return_tensors="np"`）。本目录推理脚本已内置该处理。

### 3) converter_lite 输出大模型 MindIR 缺少权重目录

**现象**：`converter_lite` 生成了 `xxx_graph.mindir`，但运行时找不到或缺少同级 `xxx_variables/`，导致构建/推理失败。

**解决**：

- 确保 `xxx_graph.mindir` 与 `xxx_variables/` 目录在**同一目录**下
- 推理时使用 `--score-model xxx_graph.mindir`

### 4) MindSpore Lite 推理 Score 模型输出恒为 0.5

**现象**：ONNX Runtime 推理结果正常（随输入变化），但 MindSpore Lite 推理 Score 模型输出恒为 `0.500000`，无论输入如何变化。

**原因**：Qwen3-VL-Reranker-2B 的 Score 模型在 FP16 精度下存在**计算溢出**问题。模型内部的 attention mask 使用 `torch.finfo(dtype).min`（FP16 下为 -65504）作为极小值，在 FP16 精度的矩阵运算中容易溢出，导致中间计算结果异常，最终 `score_linear` 输出恒为 0，经 `sigmoid(0) = 0.5` 后输出恒为 0.5。

**解决**：在 ONNX 转 MindIR 时，通过 `config.ini` 配置 `enforce_fp32`，让 Ascend GE 在计算时使用 FP32 精度：

```ini
[acl_init_options]
ge.exec.precision_mode=enforce_fp32
```

转换命令：

```bash
./output/bin/converter_lite \
  --fmk=ONNX \
  --modelFile=./onnx/qwen3_vl_reranker_score.onnx \
  --outputFile=./onnx/qwen3_vl_reranker_score \
  --optimize=ascend_oriented \
  --configFile=config.ini \
  --saveType=MINDIR
```

> **注意**：仅 Score 模型需要此配置，Vision 模型无需指定 `--configFile`。

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-VL-Reranker-2B 官方文档](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen3-VL-Reranker-2B 模型的许可证。
