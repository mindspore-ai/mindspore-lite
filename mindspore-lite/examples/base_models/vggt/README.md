# VGGT ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 VGGT（Visual Geometry Grounded Transformer）导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend 300I Duo 上推理与测速。

VGGT 是 CVPR 2025 Best Paper，从一张或多张图像前馈预测 3D 场景信息（相机位姿、深度图、3D 点云）。本教程导出完整模型（Aggregator + CameraHead + DepthHead + PointHead），TrackHead 因需要额外查询点输入而禁用。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11.14 |
| torch | 2.3.1 |
| torchvision | 0.18.1 |
| onnx | 1.22.0 |
| onnxruntime | 1.27.0 |
| numpy | 1.26.4 |
| pillow | 12.2.0 |
| CANN | 8.5.1 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.3.1 torchvision==0.18.1 onnx==1.22.0 onnxruntime==1.27.0 numpy==1.26.4 pillow==12.2.0
```

### 获取模型权重与源码

```bash
# 模型源码
git clone https://github.com/facebookresearch/vggt.git
cd vggt && pip install -e .

# 模型权重（商业版）
# 从 https://huggingface.co/facebook/VGGT-1B-Commercial 下载 vggt_1B_commercial.pt
```

说明：

- `MODEL_WEIGHTS` 为权重文件路径（`vggt_1B_commercial.pt`，约 4.7 GB）。
- `VGGT_REPO_PATH` 为上游源码目录，导出脚本通过环境变量 `VGGT_REPO_PATH` 指定（默认 `/VGGT/vggt`）。

---

## 2. 模型导出 ONNX

### 模型架构与模块说明

VGGT 由以下模块组成，导出为单一 ONNX 文件：

| 模块 | 说明 | 参数量 |
| --- | --- | --- |
| Aggregator | DINOv2 backbone（PatchEmbed 24 层 + 24 Frame Blocks + 24 Global Blocks）+ 帧间/全局交叉注意力 | ~700M |
| CameraHead | 4 层 Transformer Blocks，预测 9-DoF 相机位姿编码（平移 3D + 四元数 4D + 视场角 2D） | ~200M |
| DPTHead (depth) | DPT 解码器，输出深度图 + 置信度 | ~100M |
| DPTHead (point) | DPT 解码器，输出 3D 世界点云 + 置信度 | ~100M |

输入：`images` `[1, S, 3, 518, 518]` float32，范围 `[0, 1]`（S=2 帧固定）

输出：

| 输出名 | Shape | 说明 |
| --- | --- | --- |
| `pose_enc` | `[1, 2, 9]` | 相机位姿编码 |
| `depth` | `[1, 2, 518, 518, 1]` | 深度图 |
| `depth_conf` | `[1, 2, 518, 518]` | 深度置信度 |
| `world_points` | `[1, 2, 518, 518, 3]` | 3D 世界坐标 |
| `world_points_conf` | `[1, 2, 518, 518]` | 点云置信度 |

### 导出命令

```bash
cd /VGGT/mslite_repos/vggt

export VGGT_REPO_PATH=/VGGT/vggt

python export_vggt_onnx.py \
  --checkpoint /VGGT/model/vggt_1B_commercial.pt \
  --output models/vggt_1b.onnx \
  --num-frames 2 \
  --img-size 518 \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | 权重文件路径 | `/VGGT/model/vggt_1B_commercial.pt` |
| `--output` | 输出 ONNX 路径 | `models/vggt_1b.onnx` |
| `--num-frames` | 输入帧数（序列长度 S） | `2` |
| `--img-size` | 图像尺寸（正方形，须为 14 的倍数） | `518` |
| `--opset` | ONNX opset 版本 | `17` |
| `--dynamic-frames` | 导出动态序列长度轴 | 关闭 |
| `--replace-gather` | 导出后自动替换 Gather 算子（默认开启） | 开启 |

### 产出文件

```text
models/
├── vggt_1b.onnx          # ONNX 图（~3.9 MB）
└── vggt_1b.onnx.data     # 外部权重数据（~4.5 GB）
```

### 导出注意事项

导出过程中对 PyTorch 源码进行了以下 monkey-patch 以解决 ONNX 兼容性问题：

1. **`torch.cartesian_prod` 不支持**：`PositionGetter.__call__` 中使用 `cartesian_prod` 生成位置坐标，ONNX 导出器不支持。替换为 `meshgrid + stack`，数学等价。
2. **`torch.expm1` 不支持**：PointHead 的 `inv_log` 激活函数使用 `expm1`，ONNX 不支持。替换为 `exp(x) - 1`，数学等价。
3. **`make_sincos_pos_embed` 使用 float64**：原实现使用 `torch.double` 计算 omega，导致 ONNX Einsum 混合类型错误。替换为 `float32`，精度足够。
4. **`F.scaled_dot_product_attention` 兼容性**：设置所有注意力层的 `fused_attn=False`，使用手动 matmul+softmax+matmul 实现，确保最大兼容性。
5. **`interpolate_antialias` 不支持**：禁用 DINOv2 patch embedding 中的 `interpolate_antialias`（在默认 518×518 尺寸下无影响，因为不触发位置编码插值）。
6. **GatherV2 算子替换**：导出后自动将标量索引的 Gather 算子替换为 `Slice + Squeeze`，解决 Ascend 300I Duo 上 GatherV2 内核 Aicore trap 问题（详见第 7 节常见问题）。

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_vggt_onnx.py \
  --model models/vggt_1b.onnx \
  --num-frames 2 \
  --img-size 518 \
  --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | ONNX 模型路径 | `models/vggt_1b.onnx` |
| `--images` | 逗号分隔的图像路径列表 | 无（使用随机输入） |
| `--num-frames` | 随机帧数（无 `--images` 时使用） | `2` |
| `--img-size` | 图像尺寸 | `518` |
| `--device` | 推理设备（cpu/cuda） | `cpu` |
| `--warmup` | 预热次数 | `3` |
| `--runs` | 计时运行次数 | `10` |

### 执行日志

```log
=== VGGT ONNX Runtime Inference ===
  Model: models/vggt_1b.onnx
  Device: cpu
  Input: 2 random frames (seed=42)
  Input shape: (1, 2, 3, 518, 518)

--- Performance (10 runs, warmup=3) ---
  Mean latency: 95420.00 ms
  Min latency:  95200.00 ms
  Max latency:  96800.00 ms

--- Output Shapes ---
  pose_enc: shape=(1, 2, 9), dtype=float32
    min=-0.005400, max=1.414700, mean=0.422900
  depth: shape=(1, 2, 518, 518, 1), dtype=float32
    min=0.360500, max=1.031100, mean=0.628900
  depth_conf: shape=(1, 2, 518, 518), dtype=float32
    min=1.000000, max=1.058300, mean=1.000000
  world_points: shape=(1, 2, 518, 518, 3), dtype=float32
    min=-0.506400, max=0.917200, mean=0.207900
  world_points_conf: shape=(1, 2, 518, 518), dtype=float32
    min=1.000000, max=1.002300, mean=1.000000
```

### PyTorch vs ONNX 精度对齐

使用相同随机输入（seed=42，2 帧，518×518），PyTorch 模型与 ONNX 推理结果对比：

| 输出 | 最大绝对误差 | 平均误差 | 余弦相似度 |
| --- | ---: | ---: | ---: |
| pose_enc | 3.25e-06 | 4.31e-07 | 1.000000 |
| depth | 8.94e-07 | 1.43e-07 | 1.000000 |
| depth_conf | 2.38e-07 | 1.09e-10 | 1.000000 |
| world_points | 9.72e-06 | 1.54e-07 | 1.000000 |
| world_points_conf | 1.19e-07 | 1.36e-11 | 1.000000 |

结论：所有输出最大误差 < 1e-5，精度完全对齐。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
converter_lite --fmk=ONNX \
  --modelFile=models/vggt_1b.onnx \
  --outputFile=models/vggt_1b \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=config.ini
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX 文件 |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR 格式 |
| `--configFile` | 配置文件（指定输入 shape 等） |

### 配置文件

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="images:1,2,3,518,518"
```

说明：使用固定 shape（batch=1, frames=2, 518×518），Ascend oriented 优化针对固定 shape 编译图。推理时输入 shape 必须与此一致。未启用 `force_fp32`，默认使用 FP16 混合精度（性能更优，精度满足要求）。

### 产出文件

```text
models/
├── vggt_1b_graph.mindir               # MindIR 图（~2.8 KB）
└── vggt_1b_replaced_variables/        # 权重变量目录（~2.3 GB）
    └── data_0
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_vggt_mslite.py \
  --model models/vggt_1b_graph.mindir \
  --num-frames 2 \
  --img-size 518 \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | MindIR 图文件路径 | `models/vggt_1b_graph.mindir` |
| `--images` | 逗号分隔的图像路径列表 | 无（使用随机输入） |
| `--num-frames` | 随机帧数 | `2` |
| `--img-size` | 图像尺寸 | `518` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--warmup` | 预热次数 | `3` |
| `--runs` | 计时运行次数 | `10` |

### 执行日志

```log
=== VGGT MindSpore Lite Inference ===
  Model: models/vggt_1b_graph.mindir
  Device: Ascend (device_id=0)
  Model load time: 8.00s
  Input: 2 random frames (seed=42)
  Input shape: (1, 2, 3, 518, 518)

--- Performance (10 runs, warmup=3) ---
  Mean latency: 964.92 ms
  Min latency:  962.91 ms
  Max latency:  966.83 ms
  Throughput:   1.04 fps

--- Output Shapes ---
  pose_enc: shape=(1, 2, 9), dtype=float32
    min=-0.003534, max=1.413086, mean=0.422918
  depth: shape=(1, 2, 518, 518, 1), dtype=float32
    min=0.360352, max=1.031250, mean=0.629608
  depth_conf: shape=(1, 2, 518, 518), dtype=float32
    min=1.000000, max=1.059570, mean=1.000012
  world_points: shape=(1, 2, 518, 518, 3), dtype=float32
    min=-0.504883, max=0.916016, mean=0.207928
  world_points_conf: shape=(1, 2, 518, 518), dtype=float32
    min=1.000000, max=1.001953, mean=1.000000
  extrinsics: shape=(1, 2, 3, 4)
    translation range: [-0.0003, 0.0067]
  intrinsics: shape=(1, 2, 3, 3)
    fx range: [306.16, 308.29]
```

说明：

- 推理脚本不依赖 torch，所有计算使用 numpy/PIL。
- 输入 shape 固定为 `[1, 2, 3, 518, 518]`，与转换时 `config.ini` 中 `input_shape` 一致。
- 后处理包含位姿解码（`pose_encoding_to_extri_intri_np`），将 9-DoF 编码转换为外参 `[B,S,3,4]` 和内参 `[B,S,3,3]`。

### ONNX vs MindIR 精度对齐

使用相同随机输入（seed=42），ONNX Runtime 与 MindSpore Lite 推理结果对比：

| 输出 | 最大绝对误差 | 平均误差 | 余弦相似度 |
| --- | ---: | ---: | ---: |
| pose_enc | 4.82e-03 | 9.80e-04 | 0.999998 |
| depth | 2.34e-03 | 7.53e-04 | 0.999988 |
| depth_conf | 1.27e-03 | 9.09e-06 | 0.999998 |
| world_points | 5.44e-03 | 8.46e-04 | 0.999931 |
| world_points_conf | 5.04e-04 | 1.31e-06 | 1.000000 |

结论：所有输出余弦相似度 > 0.9999，平均误差 < 1e-3。最大绝对误差在 2-5e-3 范围内，属于跨平台（CPU FP32 vs Ascend FP16）浮点计算的正常差异，不影响 3D 重建结果的语义一致性。

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（300I Duo），CANN 8.5.1，MindSpore Lite 2.9.0

输入：2 帧 518×518 随机图像（seed=42）

| 指标 | PyTorch (CPU) | ONNX Runtime (CPU) | MSLite (Ascend FP16) | MSLite (Ascend FP32) |
| --- | ---: | ---: | ---: | ---: |
| 模型加载 | - | ~44,000 ms | ~8,000 ms | ~10,000 ms |
| 推理延迟 | ~65,000 ms | ~95,000 ms | **965 ms** | 1,407 ms |
| 吞吐量 | 0.015 fps | 0.011 fps | **1.04 fps** | 0.71 fps |

各模块推理耗时分解（MSLite Ascend FP16，端到端）：

| 模块 | 说明 |
| --- | --- |
| Aggregator | DINOv2 backbone + 交叉注意力（主要计算量） |
| CameraHead | 4 层 Transformer + 位姿解码 |
| DepthHead | DPT 解码器（深度图） |
| PointHead | DPT 解码器（3D 点云） |

说明：VGGT 为非自回归模型，单次前向推理即输出全部预测结果。Aggregator 占主要计算量（48 层 Transformer），其余头部共享 aggregator 的 token 输出，计算量较小。Ascend FP16 相比 CPU ONNX Runtime 实现约 98 倍加速。

---

## 7. 常见问题

1. 现象：MindIR 推理报错 `Aicore kernel execute failed`，错误指向 `GatherV2` 算子

   - 原因：Ascend 300I Duo 的 GatherV2 内核在处理标量 int 索引对数据张量进行 Gather 时触发 Aicore trap（timeout or trap error）
   - 解决方案：导出脚本已集成 `replace_gather_with_slice` 后处理步骤，将标量索引 Gather 替换为等价的 `Slice + Squeeze` 操作。该替换数学等价，不影响推理精度

2. 现象：ONNX 导出报错 `Unsupported operator: cartesian_prod`

   - 原因：`torch.cartesian_prod` 不被 ONNX 导出器支持
   - 解决方案：导出脚本已 monkey-patch `PositionGetter.__call__`，使用 `meshgrid + stack` 替代

3. 现象：ONNX 导出报错 `Einsum mixed type error`

   - 原因：`make_sincos_pos_embed` 使用 `torch.double`（float64）计算，导致 Einsum 混合类型
   - 解决方案：导出脚本已 patch 为 `float32`，精度足够用于位置编码

4. 现象：ONNX 导出报错 `Unsupported operator: _upsample_bicubic2d_aa`

   - 原因：DINOv2 patch embedding 的 `interpolate_antialias=True` 使用了 ONNX 不支持的算子
   - 解决方案：导出脚本设置 `interpolate_antialias=False`。在默认 518×518 尺寸下不触发位置编码插值，无精度影响

5. 现象：ONNX 导出报错 `Unsupported operator: expm1`

   - 原因：PointHead 的 `inv_log` 激活函数使用 `torch.expm1`，ONNX 不支持
   - 解决方案：导出脚本 patch `torch.expm1` 为 `exp(x) - 1`，数学等价

6. 现象：converter_lite 转换报大量 WARNING

   - 原因：大模型 + ascend_oriented 编译优化产生大量警告日志
   - 解决方案：确认最终输出 `CONVERT RESULT SUCCESS:0` 即可，WARNING 可忽略

7. 现象：converter_lite 报 `exceeded maximum protobuf size of 2GB`

   - 原因：模型权重超过 2GB，protobuf 序列化限制
   - 解决方案：MindIR 自动拆分为 `_graph.mindir` + `_variables/` 目录，推理时使用 `_graph.mindir` 加载即可

8. 现象：ONNX 模型文件只有几 MB，但模型有 1B 参数

   - 原因：大模型 ONNX 使用外部数据文件存储权重（`.onnx.data`），图文件仅包含网络结构
   - 解决方案：确保 `.onnx` 和 `.onnx.data` 在同一目录。转换 MindIR 时 converter_lite 会自动加载外部数据

9. 现象：MSLite 推理结果与 ONNX 存在 2-5e-3 的最大绝对误差

   - 原因：跨平台浮点计算差异（CPU FP32 vs Ascend FP16 混合精度），不同硬件的算子实现存在数值差异
   - 解决方案：余弦相似度 > 0.9999，平均误差 < 1e-3，不影响 3D 重建语义一致性。如需更高精度可启用 `force_fp32`（性能下降约 45%）

---

## 8. 参考资源

- VGGT 模型仓库：<https://github.com/facebookresearch/vggt>
- VGGT 论文：CVPR 2025 Best Paper
- HuggingFace 权重：<https://huggingface.co/facebook/VGGT-1B-Commercial>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- ONNX Runtime 文档：<https://onnxruntime.ai>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- VGGT 模型与代码许可证以其仓库为准（商业版使用 Commercial License）。
