# UPerNet ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 UPerNet（Unified Perceptual Parsing）导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

UPerNet 是一种统一感知解析网络，能够在单次前向推理中同时完成场景分类、物体分割、部件分割和材质分割四项任务。模型基于 ResNet-50 编码器和 FPN+PPM 解码器，输入为 BGR 图像（减均值），输出为四个任务的分割 logits。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11.14 |
| torch | 2.3.1 |
| onnx | 1.22.0 |
| onnxruntime | 1.27.0 |
| numpy | 1.26.4 |
| opencv-python | 4.13.0 |
| CANN | 8.5.1 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.3.1 onnx==1.22.0 onnxruntime==1.27.0 numpy==1.26.4 opencv-python==4.13.0
```

### 获取模型权重与源码

```bash
# 模型权重目录结构
# weights_dir/
#   upernet/
#     encoder_epoch_40.pth   # ResNet-50 编码器权重
#     decoder_epoch_40.pth   # UPerNet 解码器权重

# 上游源码（参考）
# https://github.com/CSAILVision/unifiedparsing
```

说明：

- `weights_dir` 为权重目录，包含 `encoder_epoch_40.pth` 和 `decoder_epoch_40.pth` 两个文件。
- 本目录下的 `upernet_model.py` 为独立的模型定义脚本，已移除对 `broden_dataset`、`PrRoIPool2D`、`SynchronizedBatchNorm2d` 的依赖，可独立加载权重并导出。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
python export_upernet_onnx.py \
  --weights-dir /path/to/upernet/weight \
  --output upernet.onnx \
  --input-size 576 \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--weights-dir` | 权重目录（含 encoder/decoder .pth 文件） | 必填 |
| `--output` | 输出 ONNX 文件路径 | `upernet.onnx` |
| `--input-size` | 模型输入尺寸（正方形边长） | `576` |
| `--opset` | ONNX opset 版本 | `17` |

### 产出文件

```text
./upernet.onnx    # 283 MB, opset 17
```

### 导出注意事项

- **输入尺寸选择**：输入尺寸必须为 32 的倍数（编码器总步幅为 32），且特征图尺寸（`input_size / 32`）必须能被 PPM 池化尺度 (1, 2, 3, 6) 整除。576 / 32 = 18，18 可被 1/2/3/6 整除，满足要求。512 不可用（512/32=16，16 不能被 3 整除，ONNX 不支持导出非整除的 `adaptive_avg_pool2d`）。
- **算子替换**：
  - `PrRoIPool2D`（自定义 CUDA 算子）→ `nn.AdaptiveAvgPool2d`（ONNX 标准算子）
  - `SynchronizedBatchNorm2d` → `nn.BatchNorm2d`（推理模式下等价）
- **输出为原始 logits**：不包含 softmax/插值/部件拆分，后处理在推理脚本中完成。

### 执行日志

```log
Torch forward OK:
  scene: torch.Size([1, 365, 1, 1])
  object: torch.Size([1, 336, 144, 144])
  part: torch.Size([1, 427, 144, 144])
  material: torch.Size([1, 26, 144, 144])

ONNX exported to: upernet.onnx
File size: 282.89 MB
Opset: 17
Input: img [1, 3, 576, 576]
Outputs: ['scene_logits', 'object_logits', 'part_logits', 'material_logits']
```

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_upernet_onnx.py \
  --model upernet.onnx \
  --image test_image.jpg \
  --input-size 576 \
  --output-dir ./onnx_output
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | ONNX 模型文件路径 | `upernet.onnx` |
| `--image` | 输入图像路径 | 必填 |
| `--input-size` | 模型输入尺寸 | `576` |
| `--output-dir` | 输出可视化目录 | `./onnx_output` |

### 执行日志

```log
Input: img
Outputs: ['scene_logits', 'object_logits', 'part_logits', 'material_logits']

=== UPerNet ONNX Inference Results ===
Scene top-5: [331 292 150 330  11]
Object pred unique: [ 1  2  3  4 29 31]
Part pred unique: [  2  26  43  54  66  77 125 172 279]
Material pred unique: [ 4  5  9 13 15 18 22 23 25]

=== Timing ===
Preprocess:  23.21 ms
Inference:   3512.25 ms
Postprocess: 412.87 ms
Total:       3948.33 ms
Saved object prediction to ./onnx_output/object_pred.png
Saved part prediction to ./onnx_output/part_pred.png
Saved material prediction to ./onnx_output/material_pred.png
```

### 精度对齐验证（PyTorch vs ONNX）

```bash
python verify_precision.py \
  --weights-dir /path/to/upernet/weight \
  --onnx upernet.onnx \
  --input-size 576 \
  --num-tests 3
```

```log
--- Test 1/3 ---
  scene     : cos=1.000000  max_diff=9.536743e-06  mean_diff=2.127802e-06  [PASS]
  object    : cos=1.000000  max_diff=8.678436e-05  mean_diff=3.695235e-06  [PASS]
  part      : cos=1.000000  max_diff=3.838539e-05  mean_diff=1.717462e-06  [PASS]
  material  : cos=1.000000  max_diff=4.601479e-05  mean_diff=2.888046e-06  [PASS]

Overall: ALL PASSED
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
converter_lite --fmk=ONNX \
  --modelFile=./upernet.onnx \
  --outputFile=./upernet_mindir \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX 文件 |
| `--outputFile` | 输出 MindIR 文件前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR 格式 |

### 产出说明

```text
./upernet_mindir.mindir    # 151 MB
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

转换过程中的 WARNING（如 `Cannot find input of node: /decoder/Resize`、`GlobalAveragePool has no attr kernel_size`）可忽略，不影响最终推理结果。

### 固定 shape 约束

使用 `--optimize=ascend_oriented` 后，GE 针对固定输入 shape `[1, 3, 576, 576]` 进行编译优化。推理侧必须保证输入尺寸与导出/转换时一致（576x576），否则推理会失败。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_upernet_mslite.py \
  --model upernet_mindir.mindir \
  --image test_image.jpg \
  --input-size 576 \
  --output-dir ./mslite_output
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | MindIR 模型文件路径 | `upernet_mindir.mindir` |
| `--image` | 输入图像路径 | 必填 |
| `--input-size` | 模型输入尺寸 | `576` |
| `--output-dir` | 输出可视化目录 | `./mslite_output` |

### 执行日志

```log
Loading model: upernet_mindir.mindir
Input:  img [1, 3, 576, 576] DataType.FLOAT32
Output: scene_logits [1, 365, 1, 1] DataType.FLOAT32
Output: object_logits [1, 336, 144, 144] DataType.FLOAT32
Output: part_logits [1, 427, 144, 144] DataType.FLOAT32
Output: material_logits [1, 26, 144, 144] DataType.FLOAT32
Warmup done.

=== UPerNet MSLite Inference Results ===
Scene top-5: [331 292 150 330  11]
Object pred unique: [ 1  2  3  4 29 31]
Part pred unique: [  2  26  43  54  66  77 125 172 279]
Material pred unique: [ 4  5  9 13 15 18 22 23 25]

=== Timing ===
Preprocess:  25.53 ms
Inference:   65.98 ms
Postprocess: 454.03 ms
Total:       545.54 ms
Saved object prediction to ./mslite_output/object_pred.png
Saved part prediction to ./mslite_output/part_pred.png
Saved material prediction to ./mslite_output/material_pred.png
```

### 精度对齐验证（PyTorch vs MSLite）

```bash
python verify_mslite_precision.py \
  --weights-dir /path/to/upernet/weight \
  --mindir upernet_mindir.mindir \
  --input-size 576 \
  --num-tests 3
```

```log
--- Test 1/3 ---
  scene     : cos=0.999997  max_diff=2.551842e-02  mean_diff=4.698472e-03  [PASS]
  object    : cos=1.000014  max_diff=6.918216e-02  mean_diff=6.939344e-03  [PASS]
  part      : cos=1.000011  max_diff=2.667618e-02  mean_diff=2.445108e-03  [PASS]
  material  : cos=0.999993  max_diff=5.692434e-02  mean_diff=6.032469e-03  [PASS]

Overall: ALL PASSED
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo，CANN 8.5.1，MindSpore Lite 2.9.0

性能数据以推理脚本端到端打印为准（10 次推理取平均，排除首次 warmup）。

| 指标 | PyTorch (CPU) | ONNX Runtime (CPU) | MSLite (Ascend) |
| --- | ---: | ---: | ---: |
| Preprocess | ~25 ms | ~23 ms | ~25 ms |
| Inference | ~4076 ms | ~3512 ms | **~72 ms** |
| Postprocess | ~413 ms | ~413 ms | ~454 ms |
| **总耗时** | ~4514 ms | ~3948 ms | **~551 ms** |

| 模型文件 | 大小 |
| --- | ---: |
| ONNX (opset 17) | 283 MB |
| MindIR | 151 MB |

说明：

- MSLite Ascend 推理相比 PyTorch CPU 提速约 **57 倍**。
- 后处理耗时主要来自 numpy softmax（336 通道 × 144×144），可通过 C++ 后处理或 GPU 加速进一步优化。
- 模型输出分辨率为 144×144（输入 576 / 步幅 4），如需原始分辨率需在推理脚本中添加 `cv2.resize` 插值。

---

## 7. 常见问题

1. 现象：ONNX 导出报错 `Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size`
   - 原因：PPM 池化尺度 (1, 2, 3, 6) 在特征图尺寸非整除时（如 512 输入 → 特征图 16，16 不能被 3 整除），PyTorch 的 ONNX 导出器不支持非整除的 `adaptive_avg_pool2d`
   - 解决方案：使用输入尺寸 576（特征图 18，可被 1/2/3/6 整除），或其他满足整除条件的尺寸（如 384 → 特征图 12）

2. 现象：converter_lite 转换时出现大量 WARNING（`Cannot find input of node: /decoder/Resize`）
   - 原因：ONNX 图中的 Resize（bilinear 插值）节点在解析时部分输入信息未找到，属于解析器警告
   - 解决方案：可忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 即可

3. 现象：权重加载时出现 `unexpected keys`（`_tmp_running_mean`、`_tmp_running_var`、`_running_iter`）
   - 原因：原始模型使用 `SynchronizedBatchNorm2d`，包含额外的统计量 buffer；替换为 `nn.BatchNorm2d` 后这些 key 不再需要
   - 解决方案：使用 `strict=False` 加载权重，忽略 SyncBN 额外 key

4. 现象：MSLite 推理结果与 PyTorch 存在小幅数值差异（max_diff ~0.07）
   - 原因：Ascend 上使用 FP16 混合精度计算，与 PyTorch FP32 存在精度差异
   - 解决方案：cos 相似度 > 0.9999，业务输出（argmax 预测）完全一致，精度满足要求。如需更高精度可通过 `config.ini` 配置 `force_fp32`

---

## 8. 参考资源

- 上游模型仓库：<https://github.com/unifiedperceptualparsing/unifiedparsing>
- 论文：Unified Perceptual Parsing for Scene Understanding (ECCV 2018) <https://arxiv.org/abs/1807.10221>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- ONNX Runtime 文档：<https://onnxruntime.ai>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游模型与代码许可证以其仓库为准（MIT License）。
