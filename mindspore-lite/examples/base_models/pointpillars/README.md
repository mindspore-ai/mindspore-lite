# PointPillars ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 PointPillars（LiDAR 点云 3D 检测）的检测头导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。PointPillars 把点云体素化为柱体（pillar），scatter 成伪图像后用 2D 检测网络。

> **导出范围**：本 scaffold 导出 **backbone+neck+head** 路径（输入为伪图像）。体素化（voxelization）+ scatter 步骤含难导出算子，留阶段 2 扩展。

---

## 1. 环境准备

| 软件包            | 版本                          |
|----------------|-----------------------------|
| Python         | 3.8（导出 mmdet3d）/ 3.11（推理） |
| torch          | 1.10.0+cpu（导出）           |
| mmcv-full      | 1.7.0                       |
| mmdet          | 2.28.2                      |
| mmdetection3d  | 1.0.0rc4                    |
| onnxruntime    | 1.24.0（CPU 精度/性能基准）  |
| CANN           | 8.5.0                       |
| mindspore-lite | 转换器 2.9.0 / 运行时 2.10.0 |

> torch 1.10 的 ONNX 导出器**仅支持 opset ≤ 14**（导出步骤已固定为 `--opset 14`）。aarch64 上 `mmcv-full` 需源码编译（`MMCV_WITH_OPS=1 pip install -e .`）。

```bash
pip install torch==1.10.0+cpu  # 或对应 aarch64 wheel
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2
pip install onnxruntime==1.24.0
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/pointpillars/`。

---

## 2. 模型说明

```log
点云 → voxelization+scatter（阶段2，本scaffold跳过）→ 伪图像 (1,64,256,256) → 2D backbone → neck → Anchor3DHead
                                                                                                  ↓
                                                            cls_scores + bbox_preds + dir_cls（anchor-based）
```

| 类型 | 名称          | Shape               | 说明                  |
| ---- | ------------- | ------------------- | --------------------- |
| 输入 | pseudo_img    | \[1, 64, 256, 256]  | pillar 散射后的伪图像 |
| 输出 | cls_scores    | \[1, A, h, w]       | anchor 级类别分数     |
| 输出 | bbox_preds    | \[1, A*code, h, w]  | anchor 级 3D 框回归   |
| 输出 | dir_cls       | \[1, A*2, h, w]     | 方向分类              |

> A 为每位置 anchor 数。3D 框解码（含 anchor 编码）留后处理。

---

## 3. ONNX 导出

```bash
cd examples/base_models/pointpillars
python export_pointpillars_onnx.py \
  --config upstream/mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py \
  --checkpoint upstream/pp_nus.pth \
  --output pointpillars_onnx/pointpillars.onnx --opset 14
```

`hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d` 为 nuScenes SecFPN 配置（10 类、code_size=9），与 `upstream/pp_nus.pth` 权重匹配。opset 必须 ≤14（torch 1.10 限制）。

产出：`pointpillars_onnx/pointpillars.onnx`（约 19.7 MB，opset 14，pytorch 1.10 导出）

---

## 4. ONNX 推理

```bash
python infer_pointpillars_onnx.py --model ./pointpillars_onnx/pointpillars.onnx
```

执行日志（warmup=5, runs=20, CPU）：

```log
Using random pseudo-image, shape=(1, 64, 256, 256), seed=1024
  cls_scores: (1, 140, 128, 128)
  bbox_preds: (1, 126, 128, 128)
  dir_cls: (1, 28, 128, 128)
latency_ms_mean: 123.866  (p99: 226.167)
```

---

## 5. MindSpore Lite 转换

```bash
source /home/yf/CANN/cann-8.5.0/set_env.sh          # 必须，提供 atc/lib
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./pointpillars_onnx/pointpillars.onnx \
  --outputFile=./pointpillars_onnx/pointpillars_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_pointpillars_mslite.py \
  --model ./pointpillars_onnx/pointpillars_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（warmup=5, runs=50）：

```log
latency_ms_mean: 20.604
latency_ms_p99:  32.158
VmRSS: 1373848 KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.5.0，MindSpore Lite 运行时 2.10.0 / 转换器 2.9.0，onnxruntime 1.24.0
测试条件：伪图像 `(1,64,256,256)`，固定随机种子 1024，warmup=5；ORT runs=20，MSLite runs=50（单进程单卡，选最空闲卡）

| 后端           | 设备  | 延迟 mean (ms) | 延迟 p99 (ms) | 备注               |
| -------------- | ----- | -------------- | ------------- | ------------------ |
| ONNX Runtime   | CPU   | 123.87         | 226.17        | 精度基准(fp32,CPU) |
| MindSpore Lite | 310P3 | 20.60          | 32.16         | force_fp16, Ascend |

**精度对齐**（seed=1024；ORT fp32 CPU vs MSLite fp16 Ascend，dense-head 输出）：cls_scores cos=1.000000 / mean_abs 2.53e-3；bbox_preds cos=0.999999 / mean_abs 4.80e-4；dir_cls cos=0.999998 / mean_abs 1.11e-3。进程内存 VmRSS≈1.31GB（1373848 KB）。Ascend fp16 较 CPU fp32 端到端**约 6× 加速**（123.87→20.60 ms）。

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **导出范围说明**：PointPillars **不含 spconv**——`middle_encoder` 为 `PointPillarsScatter`（柱体特征 scatter 成稠密伪图像，纯 dense 算子）。本导出已覆盖 `pseudo-image → backbone(SECOND) → neck(SECONDFPN) → Anchor3DHead` 整网检测路径，并完成 ONNX→MindIR 转换与 Ascend 推理/精度对齐（cos≈1.0）。仅体素化（点云→柱体特征）用 numba，不在 ONNX 图内，需端到端时作为 numpy/numba 预处理前置即可。

2. **现象**：fp16 检测精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / PointPillars](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
