# CenterPoint (voxel) ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 CenterPoint（voxel 版，LiDAR 点云 3D 检测）的检测头导出为 ONNX，转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理测速。CenterPoint 用 spconv 体素主干 + CenterHead（anchor-free，center-based）。

> **导出范围**：scaffold 导出 **dense 2D 路径**（backbone+neck+CenterHead，输入为 BEV 特征图）。spconv + voxelization 是阶段 2 阻塞点（同 SECOND）。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full / mmdet / mmdetection3d | 1.7.0 / 2.28.2 / 1.0.0rc4 |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 onnx==1.14.0 onnxruntime==1.16.0
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/centerpoint/`。

---

## 2. 模型说明

```log
点云 → voxelization+spconv（阶段2阻塞）→ dense BEV (1,256,180,180) → 2D backbone → neck → CenterHead
                                                                                          ↓
                                                              reg/height/dim/rot/vel/heatmap（center-based）
```

| 类型 | 名称          | Shape               | 说明                |
| ---- | ------------- | ------------------- | ------------------- |
| 输入 | bev_feat      | \[1, 256, 180, 180] | spconv 后的稠密 BEV |
| 输出 | reg/height/dim/rot/vel/heatmap | 各 \[1, C, h, w] | CenterHead 分支 |

---

## 3. ONNX 导出

```bash
cd examples/base_models/centerpoint_voxel
python export_centerpoint_voxel_onnx.py \
  --config /path/to/mmdetection3d/configs/centerpoint/voxelnet_...nus.py \
  --checkpoint /path/to/centerpoint_...pth \
  --output centerpoint_voxel_onnx/centerpoint_voxel.onnx --opset 17
```

产出：`centerpoint_voxel_onnx/centerpoint_voxel.onnx`

---

## 4. ONNX 推理

```bash
python infer_centerpoint_voxel_onnx.py --model ./centerpoint_voxel_onnx/centerpoint_voxel.onnx
```

执行日志（warmup=5, runs=20, CPU）：

```log
Using random dense BEV, shape=(1, 256, 180, 180), seed=1024
  reg: (1, 2, 180, 180)   height: (1, 1, 180, 180)   dim: (1, 3, 180, 180)
  rot: (1, 2, 180, 180)   vel: (1, 2, 180, 180)      heatmap: (1, 1, 180, 180)
latency_ms_mean: 706.072  (p99: 807.933)
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./centerpoint_voxel_onnx/centerpoint_voxel.onnx \
  --outputFile=./centerpoint_voxel_onnx/centerpoint_voxel_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_centerpoint_voxel_mslite.py \
  --model ./centerpoint_voxel_onnx/centerpoint_voxel_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（warmup=5, runs=50, Ascend）：

```log
latency_ms_mean: 21.633
latency_ms_p99:  22.639
VmRSS: 1490400 KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：dense BEV `(1,256,180,180)`，固定随机种子 1024，warmup=5；MSLite runs=50

| 后端           | 设备  | 延迟 mean (ms) | 延迟 p99 (ms) | 备注               |
| -------------- | ----- | -------------- | ------------- | ------------------ |
| ONNX Runtime   | CPU   | 706.07         | 807.93        | 精度基准(fp32,CPU) |
| MindSpore Lite | 310P3 | 21.63          | 22.64         | force_fp16, Ascend |

**精度对齐**（seed=1024；ORT fp32 CPU vs MSLite fp16 Ascend，CenterHead 6 分支）：reg/height/dim/heatmap cos=1.000000；rot cos=0.999994；vel cos=0.999993（worst cos=0.999993，mean_abs ~1e-3）。进程内存 VmRSS≈1.49GB。

> **导出范围说明**：CenterPoint 的 `middle_encoder`（SparseEncoder）用 **spconv 稀疏卷积**（Ascend 无原生支持，记录为阻塞）。本导出覆盖 `dense BEV → backbone(SECOND) → neck(SECONDFPN) → CenterHead` 的 2D 检测路径并完成转换/推理/对齐；体素化+spconv 需作为 numba/AscendC 预处理前置。

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：需要端到端（点云→检测）
   - **原因**：spconv 稀疏卷积 Ascend 无原生支持
   - **解决方案**：阶段 2 用 AscendC spconv 或 dense 近似；暂无则记录暂搁

2. **现象**：fp16 检测精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / CenterPoint](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
