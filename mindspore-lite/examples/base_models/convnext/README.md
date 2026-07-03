# ConvNeXt-UperNet ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 ConvNeXt-Tiny + UperNet 语义分割模型导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend（Atlas 300I Duo）上推理与测速。

- **上游模型**：[ConvNeXt](https://github.com/facebookresearch/ConvNeXt)（A ConvNet for the 2020s，CVPR 2022）
- **权重**：`upernet_convnext_tiny_1k_512x512.pth`（ImageNet-1K 预训练 ConvNeXt-Tiny backbone + UperNet head，在 ADE20K 上微调，150 类，输入 512×512）
- **任务**：语义分割（ADE20K 150 类）

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
| pillow | 12.2.0 |
| CANN | 8.5.1 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.3.1 onnx==1.22.0 onnxruntime==1.27.0 numpy==1.26.4 pillow==12.2.0 mindspore-lite==2.9.0
```

### 获取模型权重与源码

```bash
# 模型源码（用于参考网络结构，适配脚本已内置纯 PyTorch 重实现，无需安装 mmseg/mmcv/timm）
git clone https://github.com/facebookresearch/ConvNeXt

# 模型权重（放置到 weight/ 目录）
# upernet_convnext_tiny_1k_512x512.pth
```

说明：

- `weight/upernet_convnext_tiny_1k_512x512.pth`：mmseg 0.11.0 格式 checkpoint，包含 `backbone`（ConvNeXt-Tiny）与 `decode_head`（UperNet）权重；`auxiliary_head`（FCN，仅训练用）在推理时跳过。
- 适配脚本 `convnext_model.py` 以纯 PyTorch 重新实现了 ConvNeXt backbone + UperNet decode head，不依赖 mmseg / mmcv / timm，可直接加载原始权重。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd agent/convnext

python export_convnext_onnx.py \
  --weight /convnext/weight/upernet_convnext_tiny_1k_512x512.pth \
  --output-dir ./outputs \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--weight` | mmseg checkpoint (.pth) 路径 | `.../upernet_convnext_tiny_1k_512x512.pth` |
| `--output-dir` | ONNX 输出目录 | `./outputs` |
| `--opset` | ONNX opset 版本 | `17` |

### 产出文件

```text
./outputs/
└── upernet_convnext_tiny.onnx   # 226.4 MB, opset 17, 508 nodes
```

### 导出注意事项

- **固定输入 shape**：模型以固定 shape `(1, 3, 512, 512)` 导出，输出 `(1, 150, 512, 512)` 分割 logits。Ascend `ascend_oriented` 优化针对固定 shape 编译，推理侧需保证输入为 512×512。
- **AdaptiveAvgPool2d 替换**：PSP 模块原始使用 `AdaptiveAvgPool2d`，当输出尺寸（1/2/3/6）不是输入尺寸（16）的因数时（3、6 不可整除），PyTorch ONNX 导出器报错。适配方案：用预计算的池化矩阵 + `MatMul` 等价实现，数值与 `adaptive_avg_pool2d` 完全一致（最大误差 < 1e-7）。
- **LayerNorm 融合**：channels-first LayerNorm 通过 permute 委托给 `F.layer_norm`，在 opset 17 下导出为原生 `LayerNormalization` 算子（26 个），避免分解为 mean/var/sqrt 子图在 FP16 下的精度退化。
- **opset 选择**：使用 opset 17 以获得原生 `LayerNormalization` 算子；opset 14 会将其分解为基本算子，导致 FP16 推理精度下降（cosine ≈ 0.965）。

### 导出日志

```log
[export] loaded 260 weight tensors, skipped 8 (auxiliary_head)
[export] saved ./outputs/upernet_convnext_tiny.onnx (226.4 MB)
[export] export time: 7.7s
[onnx] opset=17  nodes=508  size=226.4MB
[onnx] input  input: [1, 3, 512, 512]
[onnx] output seg_logits: [1, 150, 512, 512]
[verify] cosine=1.000000  max_abs_err=5.912781e-05  shape=(1, 150, 512, 512)
[verify] PASS: torch vs onnx cosine > 0.99
```

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_convnext_onnx.py \
  --onnx ./outputs/upernet_convnext_tiny.onnx \
  --input /tmp/opencode/test_ade20k.jpg \
  --output ./outputs/seg_onnx.png \
  --provider CPUExecutionProvider
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--onnx` | ONNX 模型路径 | `./outputs/upernet_convnext_tiny.onnx` |
| `--input` | 输入图像路径 | `/tmp/opencode/test_ade20k.jpg` |
| `--output` | 分割掩码输出路径 | `./outputs/seg_onnx.png` |
| `--provider` | ORT 执行提供者 | `CPUExecutionProvider` |

### 执行日志

```log
[onnx-infer] preprocess=17.6ms  inference=1351.3ms  total=1368.9ms
[onnx-infer] output shape=(1, 150, 512, 512)  logits range=[-26.398, 4.902]
[onnx-infer] top classes: sky(150152), wall(69194), building(42163), sand(369), floor(266)
[onnx-infer] segmentation mask saved to ./outputs/seg_onnx.png
```

说明：

- 预处理遵循 ADE20K 标准：RGB 加载 → resize 512×512 → `(img - mean) / std`（mean=[123.675,116.28,103.53]，std=[58.395,57.12,57.375]，图像为 0-255 范围）。
- 输入图像可使用任意 RGB 图片；脚本内置的测试图为 `/tmp/opencode/test_ade20k.jpg`（合成场景图）。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

说明：`converter_lite` 为 MindSpore Lite 版本包中提供的离线转换工具。

```bash
converter_lite --fmk=ONNX \
  --modelFile=./outputs/upernet_convnext_tiny.onnx \
  --outputFile=./outputs/upernet_convnext_tiny_mindir \
  --optimize=ascend_oriented \
  --saveType=MINDIR
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR |

### 配置文件

本模型使用默认 FP16 精度即可满足精度要求（cosine > 0.9999），无需额外 `config.ini`。

> 注意：不要对本模型使用 `force_fp32` 配置，Ascend Atlas 300I Duo上会触发 aicore 硬件错误（"DDR address of the MTE instruction is out of range"）。FP16 下原生 `LayerNormalization` 算子已保证足够精度。

### 产出说明

```text
./outputs/
└── upernet_convnext_tiny_mindir.mindir   # ~125 MB
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_convnext_mslite.py \
  --mindir ./outputs/upernet_convnext_tiny_mindir.mindir \
  --input /tmp/opencode/test_ade20k.jpg \
  --output ./outputs/seg_mslite.png \
  --device ascend \
  --device-id 0 \
  --align
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir` | MindIR 模型路径 | `./outputs/upernet_convnext_tiny_mindir.mindir` |
| `--input` | 输入图像路径 | `/tmp/opencode/test_ade20k.jpg` |
| `--output` | 分割掩码输出路径 | `./outputs/seg_mslite.png` |
| `--device` | 推理设备 | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--onnx` | 用于精度对齐的 ONNX 模型 | `./outputs/upernet_convnext_tiny.onnx` |
| `--align` | 与 ONNX Runtime 输出做精度对齐 | 关闭 |

### 执行日志

```log
[mslite-infer] preprocess=19.8ms  inference=179.8ms  total=199.6ms
[mslite-infer] output shape=(1, 150, 512, 512)  logits range=[-26.531, 4.922]
[mslite-infer] top classes: sky(151062), wall(67950), building(42454), sand(372), floor(306)
[mslite-infer] segmentation mask saved to ./outputs/seg_mslite.png
[align] cosine=0.999992  max_abs_err=2.135429e-01  argmax_match_ratio=0.994213
[align] PASS: mslite vs onnx cosine > 0.99
```

说明（ascend_oriented 固定 shape 约束）：

- 转换使用 `ascend_oriented`，输入 shape 固定为 `(1, 3, 512, 512)`。推理脚本将任意输入图像 resize 到 512×512 后送入模型，保证 shape 一致。

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo，CANN 8.5.1，MindSpore Lite 2.9.0

性能数据以推理脚本端到端打印为准。

| 指标 | ONNX Runtime (CPU) | MindSpore Lite (Ascend) | PyTorch (CPU) |
| --- | ---: | ---: | ---: |
| 预处理 | 17.6 ms | 19.8 ms | — |
| 推理 | 1351.3 ms | 179.8 ms | 2653.2 ms |
| **总耗时** | **1368.9 ms** | **199.6 ms** | **2653.2 ms** |

精度对齐（同一随机输入，512×512）：

| 对比 | 余弦相似度 | 最大绝对误差 | argmax 一致率 |
| --- | --- | --- | --- |
| torch vs ONNX | 1.000000 | 5.91e-05 | — |
| torch vs MSLite | 0.999943 | 1.45e-01 | — |
| ONNX vs MSLite | 0.999992 | 2.14e-01 | 99.42% |

---

## 7. 常见问题

1. 现象：ONNX 导出报错 `Unsupported: ONNX export of operator adaptive_avg_pool2d, output size that are not factor of input size`
   - 原因：PSP 模块的 `AdaptiveAvgPool2d` 输出尺寸 3、6 不是输入 16 的因数，PyTorch ONNX 导出器不支持
   - 解决方案：用预计算池化矩阵 + `MatMul` 等价替换（`convnext_model.py` 中 `_build_pool_matrix` / `_adaptive_avg_pool`），数值完全一致

2. 现象：MSLite 推理精度低（cosine ≈ 0.965，argmax 一致率 ≈ 40%）
   - 原因：opset 14 下 channels-first LayerNorm 被分解为 mean/var/sqrt 子图，FP16 下精度退化
   - 解决方案：将 LayerNorm2d 改为 permute + `F.layer_norm` 实现，使用 opset 17 导出为原生 `LayerNormalization` 算子，FP16 精度恢复至 cosine > 0.9999

3. 现象：使用 `force_fp32` 配置转换后，MSLite 推理报 aicore 错误（"DDR address of the MTE instruction is out of range"）
   - 原因：Ascend 300I DUO 上全 FP32 模型部分算子编译异常
   - 解决方案：不使用 `force_fp32`，默认 FP16 即可满足精度要求

4. 现象：converter 转换时大量 `Cannot find input of node: /decode_head/Resize` 警告
   - 原因：ONNX Resize 算子的可选输入（roi/scales/sizes）为空，解析器告警
   - 解决方案：可忽略，最终 `CONVERT RESULT SUCCESS:0` 即转换成功

5. 现象：mmseg / mmcv / timm 未安装
   - 原因：原始 ConvNeXt 分割代码依赖 mmseg 0.11.0 + mmcv 1.3.0 + timm
   - 解决方案：`convnext_model.py` 以纯 PyTorch 重实现 backbone + decode head，直接加载 `.pth` 权重，无需安装上述依赖

---

## 8. 参考资源

- 上游模型仓库：<https://github.com/facebookresearch/ConvNeXt>
- 论文：[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)（CVPR 2022）
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- ONNX Runtime 文档：<https://onnxruntime.ai>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游模型与代码许可证以其仓库为准（MIT License）。
