# VGGT-Omega ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 VGGT-Omega（1B）导出为 ONNX（单模型，固定 shape），使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend（Atlas 300I Duo）上推理与测速。

VGGT-Omega 是一个前馈式（非自回归）视觉模型，输入一段图像序列，一次性输出每帧的相机位姿（9 维编码：平移 3 + 四元数 4 + 视场角 2）与稠密深度图及深度置信度。因其非自回归特性，无需按 prefill/decode 拆分，导出为单一 ONNX 即可。

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
| Pillow | 12.2.0 |
| CANN | 8.5.1 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.3.1 onnx==1.22.0 onnxruntime==1.27.0 numpy==1.26.4 Pillow==12.2.0 mindspore-lite
```

### 获取模型权重与源码

```bash
# 模型源码（推理与导出需 import 上游代码）
git clone https://github.com/facebookresearch/vggt-omega.git

# 模型权重（需先申请访问权限）
# https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt
```

说明：

- `MODEL_DIR` / `--checkpoint` 为权重文件 `vggt_omega_1b_512.pt`（约 4.4 GB，1.14 B 参数，fp32）。
- `--upstream-dir` / `vggt-omega/` 为上游源码目录（导出与 ONNX 精度对齐脚本需 import `vggt_omega` 包）。本目录默认在脚本同级目录下查找 `vggt-omega/`。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mslite_repos/vggt_omega

python export_vggt_omega_onnx.py \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --output-dir ./outputs \
  --num-frames 2 \
  --img-h 512 \
  --img-w 512 \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | 权重文件路径 | `/VGGT-omega/model/vggt_omega_1b_512.pt` |
| `--output-dir` | ONNX 输出目录 | `./outputs` |
| `--num-frames` | 输入图像帧数（固定） | `2` |
| `--img-h` | 输入图像高度（须为 16 的倍数） | `512` |
| `--img-w` | 输入图像宽度（须为 16 的倍数） | `512` |
| `--opset` | ONNX opset 版本 | `17` |

### 产出文件

```text
./outputs/
├── vggt_omega.onnx          # ONNX 图（约 2.0 MB）
└── vggt_omega.onnx.data     # 外部权重数据（约 4.4 GB，fp32）
```

### 导出注意事项（实际踩坑点）

VGGT-Omega 的 PyTorch 实现有几处在 ONNX 追踪时会引入 `If` 控制流节点或类型不匹配，`converter_lite` 与 ONNX Runtime 均无法处理，导出脚本已做等价改写：

1. **绕过 CUDA autocast**：上游 `forward` 用 `torch.autocast(device_type="cuda")` 包裹，在无 CUDA 的环境会直接抛错。导出包装器直接调用 aggregator / camera_head / dense_head 三个子模块，以 fp32 运行（也是 MindSpore Lite 推荐的导出精度）。
2. **预计算 RoPE 位置编码**：`RopePositionEmbedding.forward` 内的 `torch.arange` 在追踪时会生成 `Greater`/`Less`/`If` 范围检查。由于导出为固定 shape，RoPE sin/cos 为常量，脚本在追踪前预计算并缓存，追踪时直接返回常量，消除全部 `If`。
3. **简化 `custom_interpolate`**：稠密头中的 `if tuple(x.shape[-2:]) == tuple(size)` 形状比较在追踪时变为 `Equal`/`If`。改写为无条件 `F.interpolate`（尺寸相同时为恒等 Resize），数学等价。
4. **位置编码 float32 化**：`make_sincos_pos_embed` 用 `torch.double` 的 `omega` 与 float32 坐标做 `Einsum`，产生混合 dtype（ORT 报 `Einsum bound to different types`）。改为全程 float32（结果本就 `.float()` 转回）。
5. **单张量 attention 路径**：`SelfAttentionBlock.forward` 即使输入单张量也走 `_forward_list`（含 concat/split 开销）。eval 模式下改走 `_forward`，数学等价且显著减小图规模。
6. **外部数据合并**：`torch.onnx` 默认为每个大算子单独输出外部数据文件（数百个）。脚本加载后用 `onnx.save_model(all_tensors_to_one_file=True)` 合并为单一 `vggt_omega.onnx.data`，便于归档与拷贝。

最终 ONNX 含 8769 个节点、0 个 `If` 节点，opset 17，输入 `images: [1,2,3,512,512]`，输出 `pose_enc:[1,2,9]`、`depth:[1,2,512,512,1]`、`depth_conf:[1,2,512,512]`。

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_vggt_omega_onnx.py \
  --onnx-dir ./outputs \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --num-frames 2 \
  --img-h 512 \
  --img-w 512 \
  --compare-torch
```

不带 `--compare-torch` 时仅运行 ONNX 推理；带该参数会额外加载原始 PyTorch 模型（直接调用子模块，fp32，无 autocast）并在相同输入上对比余弦相似度。

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--onnx-dir` | ONNX 模型目录（含 `vggt_omega.onnx` 与 `.onnx.data`） | `./outputs` |
| `--checkpoint` | 权重文件（仅 `--compare-torch` 需要） | 见脚本默认值 |
| `--upstream-dir` | 上游源码目录（仅 `--compare-torch` 需要） | 脚本同级 `vggt-omega/` |
| `--input` | 输入图片路径/目录/glob（默认用 `samples/`） | `samples/*.jpg` |
| `--num-frames` | 帧数（须与导出一致） | `2` |
| `--img-h` / `--img-w` | 输入尺寸（须与导出一致） | `512` / `512` |
| `--provider` | ORT provider | `CPUExecutionProvider` |
| `--compare-torch` | 是否与 PyTorch 对齐 | 关闭 |

### 执行日志

```log
[onnx] images: ['samples/frame_0.jpg', 'samples/frame_1.jpg']
[onnx] preprocess=66.4ms inference=25934.0ms
[onnx] pose_enc (1, 2, 9) depth (1, 2, 512, 512, 1) depth_conf (1, 2, 512, 512)
[onnx] extrinsics[0,0]=
[[ 9.9999982e-01  7.4546449e-05  5.5377034e-04  6.0040504e-05]
 [-7.4508280e-05  1.0000000e+00 -6.8945526e-05  9.5635653e-05]
 [-5.5377546e-04  6.8904257e-05  9.9999982e-01 -6.3739717e-06]]
[onnx] intrinsics[0,0]=
[[646.99774   0.      256.     ]
 [  0.      653.7639 256.     ]
 [  0.        0.        1.     ]]
[onnx] depth mean=2.9797 min=0.3318 max=23.1864
[onnx] depth_conf mean=2.3508
[onnx] torch vs onnx cosine similarity:
  pose_enc: 1.000000 max_abs=4.768372e-07
  depth: 1.000000 max_abs=7.314682e-04
  depth_conf: 1.000000 max_abs=1.487732e-04
```

说明：

- 输入图像统一 resize 到固定 `512x512`（BICUBIC），保持 `[0,1]` 范围，ResNet 均值/方差归一化已在模型内部完成。
- `pose_enc` 经 `encoding_to_camera` 解码为外参（`extrinsics`，3x4）与内参（`intrinsics`，3x3）。
- ONNX Runtime 仅在 CPU 上运行（本机无 GPU）；512x512 单次推理约 26 s。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

说明：`converter_lite` 为 MindSpore Lite 版本包中提供的离线转换工具。

```bash
converter_lite --fmk=ONNX \
  --modelFile=./outputs/vggt_omega.onnx \
  --outputFile=./outputs/vggt_omega_mindir \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR |
| `--configFile` | 配置文件（本模型需 `force_fp32`） |

### 配置文件

`config.ini`：

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

本模型稠密头使用 `torch.exp` 解码深度，fp16 下指数运算会放大误差，导致深度均值偏差约 2 倍、位姿余弦相似度降至 0.97（< 0.99）。因此必须使用 `force_fp32`。这是 SKILL 中“精度不足时才使用 fp32”的典型场景。

### 产出说明

```text
./outputs/
├── vggt_omega_mindir_graph.mindir       # MindIR 图（约 2 KB）
└── vggt_omega_mindir_variables/
    └── data_0                           # 外部权重（约 2.2 GB）
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

转换过程中可能出现 `InferShapeByNNACL for op: /dense_head/Pow_* failed` 等 warning，属于常量折叠阶段的形状推断告警，不影响最终 MindIR 生成与推理，可忽略。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_vggt_omega_mslite.py \
  --mindir-dir ./outputs \
  --num-frames 2 \
  --img-h 512 \
  --img-w 512 \
  --device ascend \
  --device-id 0 \
  --compare-onnx
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含 `vggt_omega_mindir_graph.mindir` 与 `_variables/`） | `./outputs` |
| `--input` | 输入图片路径/目录/glob（默认 `samples/`） | `samples/*.jpg` |
| `--num-frames` | 帧数（须与导出一致） | `2` |
| `--img-h` / `--img-w` | 输入尺寸（须与导出一致） | `512` / `512` |
| `--device` | 推理设备 | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--compare-onnx` | 是否与 ONNX Runtime 对齐 | 关闭 |
| `--onnx-dir` | ONNX 目录（仅 `--compare-onnx` 需要） | `./outputs` |

### 执行日志（含性能数据）

```log
[mslite] images: ['samples/frame_0.jpg', 'samples/frame_1.jpg']
[mslite] preprocess=62.3ms inference=780.2ms total=842.5ms
[mslite] pose_enc (1, 2, 9) depth (1, 2, 512, 512, 1) depth_conf (1, 2, 512, 512)
[mslite] extrinsics[0,0]=
[[ 9.9999982e-01  1.2811788e-05  5.7018356e-04  5.1297244e-05]
 [-1.2784689e-05  1.0000000e+00 -4.7530684e-05  9.2333954e-05]
 [-5.7018414e-04  4.7523386e-05  9.9999982e-01  5.3798565e-05]]
[mslite] intrinsics[0,0]=
[[644.43774   0.      256.     ]
 [  0.      652.0447 256.     ]
 [  0.        0.        1.     ]]
[mslite] depth mean=2.9691 min=0.3323 max=22.8809
[mslite] depth_conf mean=2.3552
[mslite] mslite vs onnx cosine similarity:
  pose_enc: 0.999997 max_abs=4.117489e-03
  depth: 0.999974 max_abs=5.661821e-01
  depth_conf: 0.999994 max_abs=6.494999e-02
```

说明（ascend_oriented 固定 shape 约束）：

- 转换使用 `ascend_oriented`，GE 针对固定输入 shape 编译图。推理侧必须保证输入 shape 与导出/转换完全一致：`--num-frames`、`--img-h`、`--img-w` 三个参数在导出、转换、推理三步必须保持相同（默认 2 / 512 / 512）。预处理将任意尺寸图片 resize 到该固定 shape。
- 如需支持其它分辨率或帧数，须用对应 shape 重新导出 ONNX 并重新转换 MindIR，再在业务侧按 shape 路由到对应模型。
- 推理脚本全程使用 numpy/PIL 完成预处理与后处理，**不依赖 torch**。

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo，CANN 8.5.1，MindSpore Lite 2.9.0

输入：2 帧 512x512 RGB 图像（1.14 B 参数模型，单次前馈推理，非自回归）。

| 指标 | 耗时 (ms) |
| --- | ---: |
| 预处理（2 帧加载 + resize） | 71 |
| ONNX Runtime 推理（CPU，16 线程） | 25934 |
| **MindSpore Lite 推理（Ascend，force_fp32）** | **781** |
| **MindSpore Lite 端到端** | **852** |

一次性开销参考：

| 阶段 | 耗时 |
| --- | ---: |
| ONNX 导出 | 282 s |
| ONNX → MindIR 转换（force_fp32） | 216 s |

精度对齐（余弦相似度，阈值 > 0.99）：

| 对比项 | pose_enc | depth | depth_conf |
| --- | ---: | ---: | ---: |
| PyTorch vs ONNX | 1.000000 | 1.000000 | 1.000000 |
| MindSpore Lite vs ONNX | 0.999997 | 0.999974 | 0.999994 |

说明：

- Ascend 推理相比 CPU ONNX Runtime 提速约 33 倍（781 ms vs 25934 ms）。
- 性能数据以推理脚本端到端打印为准。

---

## 7. 常见问题

1. 现象：导出的 ONNX 含 `If` 节点，`converter_lite` 转换报 `i: 3 out of range` / `ValueNode<If>`。
   - 原因：`RopePositionEmbedding` 的 `torch.arange` 与 `custom_interpolate` 的形状比较在追踪时生成 ONNX `If` 控制流。
   - 解决方案：导出脚本预计算 RoPE 常量、简化 `custom_interpolate`，消除全部 `If`（导出日志会校验 `If nodes=0`）。

2. 现象：ONNX Runtime 加载报 `Einsum bound to different types (tensor(float) and tensor(double))`。
   - 原因：`make_sincos_pos_embed` 用 float64 的 `omega` 与 float32 坐标做 `Einsum`。
   - 解决方案：导出脚本将该函数改写为全程 float32。

3. 现象：MindIR 推理深度均值约为 ONNX 的 2 倍、位姿余弦相似度 0.97（< 0.99）。
   - 原因：稠密头 `torch.exp` 在 fp16 下放大误差。
   - 解决方案：转换时通过 `config.ini` 指定 `ge.exec.precision_mode=force_fp32`。

4. 现象：原始 PyTorch 模型 `forward` 报 `Torch not compiled with CUDA enabled`。
   - 原因：上游用 `torch.cuda.is_bf16_supported()` 与 `torch.autocast(device_type="cuda")`，无 CUDA 环境下直接抛错。
   - 解决方案：导出包装器与精度对齐脚本直接调用 aggregator / camera_head / dense_head 子模块，绕过 autocast，以 fp32 运行。

5. 现象：转换日志出现 `InferShapeByNNACL for op: /dense_head/Pow_* failed` warning。
   - 原因：常量折叠阶段对部分 `Pow` 节点的形状推断告警。
   - 解决方案：可忽略，只要最终打印 `CONVERT RESULT SUCCESS:0` 且 MindIR 文件生成即可。

6. 现象：清理 outputs 目录后 ONNX Runtime 加载报 `External data path does not exist`。
   - 原因：`vggt_omega.onnx`（2 MB 图）依赖同目录外部权重文件。
   - 解决方案：保留 `vggt_omega.onnx` 与 `vggt_omega.onnx.data` 一起；导出脚本已将权重合并为单一 `.onnx.data` 文件。

---

## 8. 参考资源

- 上游模型仓库：<https://github.com/facebookresearch/vggt-omega>
- 模型权重：<https://huggingface.co/facebook/VGGT-Omega>
- MindSpore Lite 文档：<https://www.mindspore.cn/lite>
- ONNXRuntime 文档：<https://onnxruntime.ai/>

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游模型与代码许可证以其仓库（LICENSE）为准。
