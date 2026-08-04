# SAM2 (sam2.1-hiera-base-plus) ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Meta SAM2（`sam2.1-hiera-base-plus`，图像分割）导出为两个 ONNX 模型（`sam2_encoder` 与 `sam2_decoder`），使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend（Atlas 300I Duo）上推理与测速。

SAM2 原始仓库同时支持图像与视频分割。视频路径依赖 `memory_attention`（含复数 RoPE 算子）与多帧记忆库，无法直接导出为 ONNX。本教程按图像预测路径（`SAM2ImagePredictor`）将模型拆分为：

- **sam2_encoder**：Hiera trunk + FPN neck + 高分辨率投影（`conv_s0`/`conv_s1`）+ `no_mem_embed`
- **sam2_decoder**：SAM prompt encoder + mask decoder（TwoWayTransformer）

两个子图各自小且算子标准，便于 Ascend GE 编译。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.13.0 |
| torchvision | 0.28.0 |
| onnx | 1.22.0 |
| onnxruntime | 1.28.0 |
| numpy | 2.4.6 |
| pillow | 12.3.0 |
| hydra-core | 1.3.4 |
| iopath | 0.1.10 |
| CANN | 8.5.1 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.13.0 torchvision==0.28.0 onnx==1.22.0 onnxruntime==1.28.0 \
            numpy pillow hydra-core iopath mindspore-lite==2.9.0
```

### 获取模型权重与源码

- 模型源码目录`/path/to/sam2`（上游 <https://github.com/facebookresearch/sam2>），已 `pip install -e .` 安装为 `sam2` 包。
- 模型权重目录`/path/to/weight`，包含：
  - `sam2.1_hiera_base_plus.pt`（约 308 MB 权重）

---

## 2. 模型导出 ONNX

### 导出命令

需要在 SAM2 源码目录下执行（脚本依赖 `sam2` 包与 hydra 配置 `configs/sam2.1/sam2.1_hiera_b+.yaml`）：

```bash
python export_sam2_onnx.py \
  --ckpt /path/to/weight/sam2.1_hiera_base_plus.pt \
  --config /path/to/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --output-dir /path/to/onnx
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--ckpt` | SAM2 权重路径 |
| `--config` | hydra 配置名 |
| `--output-dir` | ONNX 输出目录 |

### 模块说明

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| `sam2_encoder` | `image` [1,3,1024,1024] float32 | `image_embed` [1,256,64,64]、`high_res_s0` [1,32,256,256]、`high_res_s1` [1,64,128,128] |
| `sam2_decoder` | 上述三个特征图 + `point_coords` [1,1,2] float32 + `point_labels` [1,1] int32 | `low_res_masks` [1,3,256,256]、`iou_predictions` [1,3] |

### 产出文件

```text
/path/to/onnx/
├── sam2_encoder.onnx   # 约 293 MB
└── sam2_decoder.onnx   # 约 16 MB
```

### 导出注意事项（实际踩坑点）

- **SDPA 算子替换**：PyTorch 的 `F.scaled_dot_product_attention` 在 ONNX 导出时会生成 `If` 控制流节点（在高效/回退注意力路径间选择），MindSpore Lite Ascend 转换器无法处理该 `If`（报 `i: 3 out of range`）。导出脚本在 `export_sam2_onnx.py:_patch_sdpa()` 中将 Hiera 的 `MultiScaleAttention` 与 SAM 的 `Attention` 前向替换为等价的 `MatMul + Softmax + MatMul`，数值等价且生成无控制流的干净图。
- **位置编码 `.tile()` 替换**：`Hiera._get_pos_embed` 使用 `window_embed.tile([shape-derived dims])`，其 ONNX 导出同样生成 `If` 守卫节点。脚本将 `_get_pos_embed` 改为对固定 1024 输入（patch grid 256×256）预计算并缓存位置编码，作为常量进入图，消除 `If`。
- 导出使用 `torch.onnx.utils.export`（legacy 导出器），固定 shape（无 `dynamic_axes`），配合 Ascend `ascend_oriented` 编译。

执行日志（节选）：

```log
=== Exporting sam2_encoder ===
  Output shapes: [(1, 256, 64, 64), (1, 32, 256, 256), (1, 64, 128, 128)]
  Done (292.4 MB)
=== Exporting sam2_decoder ===
  Output shapes: [(1, 3, 256, 256), (1, 3)]
  Done (15.6 MB)
=== All ONNX exports complete ===
```

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_sam2_onnx.py \
  --encoder /path/to/onnx/sam2_encoder.onnx \
  --decoder /path/to/onnx/sam2_decoder.onnx \
  --image /path/to/sam2/notebooks/images/truck.jpg \
  --point 500 375 \
  --ckpt /path/to/weight/sam2.1_hiera_base_plus.pt \
  --config /path/to/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --output ./output/mask_onnx.png
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--encoder` | encoder ONNX 路径 | 必填 |
| `--decoder` | decoder ONNX 路径 | 必填 |
| `--image` | 输入图片路径 | 必填 |
| `--point` | 前景点像素坐标 `X Y` | `500 375` |
| `--ckpt` | PyTorch 权重（精度对齐用，不传则跳过） | `None` |
| `--config` | hydra 配置 | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `--output` | 掩码叠加图保存路径 | `None` |
| `--runs` | 性能测试轮数 | `10` |

### 执行日志

```log
=== SAM2 ONNX Inference ===
  image: .../truck.jpg (orig 1800x1200)
  point: (500.0, 375.0)
  low_res_masks: (1, 3, 256, 256), iou_predictions: (1, 3)
  ious: [0.0471, 0.8977, 0.526]
  best_mask_iou: 0.8977, foreground_pixels: 22496
=== Precision alignment (ONNX vs PyTorch) ===
  low_res_masks cos_sim: 0.999803
  iou_predictions cos_sim: 0.999717
  [PASS] cosine similarity > 0.99
=== Performance (3 runs) ===
  encoder ms: mean=2968.7, p50=2985.1
  decoder ms: mean=129.1, p50=58.8
  total ms:   mean=3097.9, p50=3024.3
```

说明：ONNX 推理在 CPU 上运行（`CPUExecutionProvider`），用于精度对齐基准。掩码日志中 `best_mask_iou` 为 3 个候选掩码中 IoU 最高者，`foreground_pixels` 为上采样到原图后的前景像素数。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

使用 MindSpore Lite Python `Converter` API（`convert_sam2_to_mindir.py`）：

```bash
python convert_sam2_to_mindir.py \
  --onnx-dir /path/to/onnx \
  --output-dir /path/to/mindir
```

等价于对每个模型执行：

```python
from mindspore_lite import Converter, FmkType, ModelType
c = Converter()
c.optimize = "ascend_oriented"
c.save_type = ModelType.MINDIR
c.convert(FmkType.ONNX, "onnx/sam2_encoder.onnx", "mindir/sam2_encoder")
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--onnx-dir` | ONNX 输入目录 |
| `--output-dir` | MindIR 输出目录 |

### 产出文件

```text
/path/to/mindir/
├── sam2_encoder.mindir   # 约 229 MB
└── sam2_decoder.mindir   # 约 14 MB
```

执行日志：

```log
=== Converting SAM2 ONNX -> MindIR (ascend_oriented, fp32_weights) ===
Converting sam2_encoder.onnx ...
CONVERT RESULT SUCCESS:0
  sam2_encoder.mindir: 228.7 MB (weight_fp16=False)
Converting sam2_decoder.onnx ...
CONVERT RESULT SUCCESS:0
  sam2_decoder.mindir: 13.4 MB (weight_fp16=False)
=== Conversion complete ===
```

说明：

- 转换使用 `ascend_oriented`，输入 shape 固定（encoder `[1,3,1024,1024]`；decoder 五个输入均为固定 shape），推理侧必须保证相同 shape。图片统一 resize 到 1024×1024，点坐标映射到 1024 帧。
- 转换过程中打印的 WARNING 日志（如 `LayerNorm has no attr stride`、`Cannot find input of node` 等）可忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 且 `.mindir` 文件生成即表示转换成功。
- 无需 `config.ini` / `force_fp32`：decoder 默认 fp16 精度即满足 cos 相似度 1.0（见第 6 节）。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash

python infer_sam2_mslite.py \
  --encoder /path/to/mindir/sam2_encoder.mindir \
  --decoder /path/to/mindir/sam2_decoder.mindir \
  --image /path/to/sam2/notebooks/images/truck.jpg \
  --point 500 375 \
  --encoder-onnx /path/to/onnx/sam2_encoder.onnx \
  --decoder-onnx /path/to/onnx/sam2_decoder.onnx \
  --output ./output/mask_mslite.png \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--encoder` | encoder MindIR 路径 | 必填 |
| `--decoder` | decoder MindIR 路径 | 必填 |
| `--image` | 输入图片路径 | 必填 |
| `--point` | 前景点像素坐标 `X Y` | `500 375` |
| `--device` | 推理设备（cpu/ascend） | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--encoder-onnx` | ONNX 参考（精度对齐用） | `None` |
| `--decoder-onnx` | ONNX 参考（精度对齐用） | `None` |
| `--output` | 掩码叠加图保存路径 | `None` |

### 执行日志

```log
=== Building MindSpore Lite models ===
  encoder inputs:
    image dtype=DataType.FLOAT32 shape=[1, 3, 1024, 1024]
  decoder inputs:
    image_embed dtype=DataType.FLOAT32 shape=[1, 256, 64, 64]
    high_res_s0 dtype=DataType.FLOAT32 shape=[1, 32, 256, 256]
    high_res_s1 dtype=DataType.FLOAT32 shape=[1, 64, 128, 128]
    point_coords dtype=DataType.FLOAT32 shape=[1, 1, 2]
    point_labels dtype=DataType.INT32 shape=[1, 1]
  zero_copy=True

=== SAM2 MindSpore Lite Inference ===
  image: .../truck.jpg (orig 1800x1200)
  point: (500.0, 375.0)
  ious: [0.0472, 0.8965, 0.5264]
  best_mask_iou: 0.8965, foreground_pixels: 22483
  latency_ms: encoder=294.7, decoder=7.9, total=302.5

=== Precision alignment (MindIR vs ONNX) ===
  low_res_masks cos_sim: 1.000000
  iou_predictions cos_sim: 1.000000
  max_abs_error (masks): 0.130003
  [PASS] cosine similarity > 0.99

=== Performance (10 runs, device=ascend) ===
  encoder ms: mean=270.4, p50=270.4
  decoder ms: mean=6.5, p50=6.2
  total ms:   mean=276.9, p50=276.6
```

说明：

- 推理脚本 `infer_sam2_mslite.py` **不依赖 torch**，预处理（resize/归一化）与后处理（掩码上采样/阈值化）均用 numpy + PIL 实现。
- 输入 dtype 由脚本按模型声明自动对齐（`point_labels` 必须为 int32，否则报 `input data type not match`）。
- 掩码后处理：从 3 个候选掩码中选 IoU 最高者，clip 到 [-32,32]，双线性上采样到原图尺寸，阈值 0 二值化。
- 零拷贝优化：encoder 输出的 3 个特征图（~20MB）不再回拷 Host，而是作为 device MSTensor 直接传入 decoder 的 `predict()`，消除一次 D2H + H2D 往返。

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo，CANN 8.5.1，MindSpore Lite 2.9.0，输入 1024×1024，单点提示。

| 指标 | ONNX Runtime (CPU) ms | MindSpore Lite (Ascend) ms |
| --- | ---: | ---: |
| Encoder | 2968.7 | 270.4 |
| Decoder | 129.1 | 6.5 |
| **总耗时（mean）** | **3097.9** | **276.9** |
| 总耗时（p50） | 3024.3 | 276.6 |

Ascend 相对 CPU ONNX Runtime 端到端加速约 **11.2×**。

### 性能优化

推理链路通过零拷贝 + 预分配 device tensor 优化：

1. **零拷贝（encoder→decoder device tensor 直传）**：encoder 输出的 3 个特征图（~20MB）不再 `get_data_to_numpy()` 回拷 Host，而是作为 device MSTensor 直接传入 decoder 的 `predict()`，消除一次 D2H + H2D 往返。
2. **预分配 device tensor**：image / point_coords / point_labels 创建为 `mslite.Tensor(device="ascend:<id>")` 并在循环中复用，消除每次推理的输入 H2D 拷贝。

### 精度对齐

| 对比项 | cos 相似度 | 结论 |
| --- | ---: | --- |
| ONNX vs PyTorch（low_res_masks） | 0.999803 | PASS (>0.99) |
| ONNX vs PyTorch（iou_predictions） | 0.999717 | PASS (>0.99) |
| MindIR vs ONNX（low_res_masks） | 1.000000 | PASS (>0.99) |
| MindIR vs ONNX（iou_predictions） | 1.000000 | PASS (>0.99) |

### 精度策略

Ascend GE 编译（`ascend_oriented`）自动将 fp32 权重转 fp16 计算，所有算子实际以 FLOAT16 运行。无需额外配置 `force_fp32`，默认 fp16 精度即可满足 cos 相似度 1.0。

---

## 7. 常见问题

1. **转换报 `Convert model failed ... NULL pointer returned`，plog 显示 `AclBuildInit failed`**
   - 原因：CANN TBE 编译器依赖 `decorator`/`attrs`/`cloudpickle`/`psutil`/`scipy`/`tornado` 等 Python 包缺失。
   - 解决方案：`pip install decorator attrs cloudpickle psutil scipy tornado`。

2. **转换报 `i: 3 out of range: 3, cnode: ... ValueNode<If>`**
   - 原因：ONNX 图含 `If` 控制流节点，来自 `F.scaled_dot_product_attention` 的路径选择与 `Hiera._get_pos_embed` 的 `.tile()` 守卫。
   - 解决方案：导出脚本 `_patch_sdpa()` 将 SDPA 替换为手动 MatMul+Softmax+MatMul，并将位置编码预计算为常量；导出后确认 ONNX `If` 节点数为 0。

3. **推理报 `input data type not match, required 34, given 43`**
   - 原因：decoder 的 `point_labels` 期望 int32（dtype=34），传入了 float32（43）。
   - 解决方案：推理脚本 `_to_tensor` 按 `mslite.DataType` 枚举映射 dtype，确保 `point_labels` 转为 int32。

4. **导出耗时较长（数分钟）**
   - 原因：encoder 模型大（约 4500 节点），legacy ONNX 导出 + 常量折叠耗时。
   - 解决方案：正常现象，单次导出约 3–5 分钟，耐心等待至 `All ONNX exports complete`。

5. **想用多点击或框提示**
   - 原因：当前 decoder 固定 `point_coords` shape 为 `[1,1,2]`（单前景点）。
   - 解决方案：修改 `export_sam2_onnx.py` 中 `export_decoder` 的 dummy `point_coords`/`point_labels` shape 后重新导出；prompt encoder 内部会自动补一个 padding 点。

6. **转换过程中出现大量 WARNING 日志**
   - 原因：MindSpore Lite 转换器在解析 ONNX 图时对部分算子属性（如 LayerNorm 的 stride）和常量节点（如 Constant_7/8/9）发出警告。
   - 解决方案：这些 WARNING 可忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 且 `.mindir` 文件正常生成即表示转换成功。

---

## 8. 参考资源

- 上游模型仓库：<https://github.com/facebookresearch/sam2>
- SAM2 论文：<https://arxiv.org/abs/2408.00714>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- ONNX Runtime 文档：<https://onnxruntime.ai/>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游 SAM2 模型与代码遵循 Apache 2.0 许可证（以其仓库为准）。
