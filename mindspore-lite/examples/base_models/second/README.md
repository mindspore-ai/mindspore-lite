# SECOND ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 SECOND（LiDAR 点云 3D 检测）的检测头导出为 ONNX，转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理测速。SECOND 用 spconv 稀疏卷积在体素上做 3D 特征，再 height-collapse 到 dense BEV 做 2D 检测。

> **导出范围**：scaffold 导出 **dense 2D 路径**（backbone+neck+AnchorHead，输入为 BEV 特征图）。spconv 稀疏卷积 + voxelization 是阶段 2 阻塞点，按 plan「记录暂搁」。

---

## 1. 环境准备

导出（依赖 mmdet3d 旧栈）与推理分属两个 conda 环境。

**导出环境**（mmdet3d，Python 3.8）

| 软件包            | 版本        |
|----------------|-----------|
| torch          | 1.10.0    |
| mmcv-full      | 1.7.0     |
| mmdet          | 2.28.2    |
| mmdetection3d  | 1.0.0rc4  |

**推理环境**（py3.11）

| 软件包            | 版本      |
|----------------|---------|
| Python         | 3.11    |
| onnx           | 1.19.1  |
| onnxruntime    | 1.24.2  |
| mindspore-lite | 2.10.0  |
| CANN           | 8.5.0   |

```bash
# 导出环境（aarch64 需源码编译 mmcv-full，torch 锁定 1.10.0）
pip install torch==1.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install 'setuptools<70' mmcv-full==1.7.0 --no-build-isolation mmdet==2.28.2 "git+https://github.com/open-mmlab/mmdetection3d.git@v1.0.0rc4"
# 推理环境
pip install onnx==1.19.1 onnxruntime==1.24.2 mindspore-lite
```

权重与 config 来源：[mmdetection3d](https://github.com/open-mmlab/mmdetection3d) 的 `configs/second/`。

---

## 2. 模型说明

```log
点云 → voxelization+spconv（阶段2阻塞，本scaffold跳过）→ dense BEV (1,256,180,180) → 2D backbone → neck → AnchorHead
                                                                                                        ↓
                                                                      cls_scores + bbox_preds + dir_cls
```

| 类型 | 名称          | Shape               | 说明                |
| ---- | ------------- | ------------------- | ------------------- |
| 输入 | bev_feat      | \[1, 256, 180, 180] | spconv 后的稠密 BEV |
| 输出 | cls_scores    | \[1, A, h, w]       | anchor 级类别分数   |
| 输出 | bbox_preds    | \[1, A*code, h, w]  | anchor 级 3D 框回归 |
| 输出 | dir_cls       | \[1, A*2, h, w]     | 方向分类            |

---

## 3. ONNX 导出

```bash
cd examples/base_models/second
# 须在 mmdet3d 环境（torch 1.10.0）中运行；torch 1.10 最高稳定 opset 为 14
python export_second_onnx.py \
  --config <mmdetection3d>/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py \
  --checkpoint ./upstream/second_kitti.pth \
  --output second_onnx/second.onnx --opset 14
```

产出：`second_onnx/second.onnx`（dense BEV `[1,256,180,180]` → cls `[1,18,180,180]` / bbox `[1,42,180,180]` / dir `[1,12,180,180]`）

---

## 4. ONNX 推理

```bash
python infer_second_onnx.py --model ./second_onnx/second.onnx
```

执行日志（warmup=5, runs=20, CPU fp32）：

```log
Using random dense BEV, shape=(1, 256, 180, 180), seed=1024
  cls_scores: (1, 18, 180, 180)
  bbox_preds: (1, 42, 180, 180)
  dir_cls: (1, 12, 180, 180)

latency_ms_mean: 523.179, p99: 586.332
VmRSS: 496436 KB
```

> 完整日志见 `second_onnx/onnx_infer_cpu.log`。

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./second_onnx/second.onnx \
  --outputFile=./second_onnx/second_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_second_mslite.py \
  --model ./second_onnx/second_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（warmup=20, runs=100, Ascend fp16，日志已保存 `second_onnx/mslite_infer_ascend.log`）：

```log
Using random dense BEV, shape=(1, 256, 180, 180), seed=1024
  cls_scores: (1, 18, 180, 180)
  bbox_preds: (1, 42, 180, 180)
  dir_cls: (1, 12, 180, 180)

latency_ms_mean: 19.703, p99: 27.857
VmRSS: 1352400 KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.5.0，MindSpore Lite 2.10.0（converter_lite 取自 2.9.0 工具包）
测试条件：dense BEV `(1,256,180,180)`，固定随机种子 1024；ORT warmup=5/runs=20，MSLite warmup=20/runs=100

| 后端           | 设备  | 延迟 mean (ms) | 延迟 p99 (ms) | 备注                  |
| -------------- | ----- | -------------- | ------------- | --------------------- |
| ONNX Runtime   | CPU   | 523.18         | 586.33        | 精度基准（fp32,CPU）  |
| MindSpore Lite | 310P3 | 19.70          | 27.86         | force_fp16, Ascend，约 27× 加速 |

**精度对齐**（seed=1024；ORT fp32 CPU vs MSLite fp16 Ascend，dense-head 输出）：
- `cls_scores`：cos=**1.000000**，mean_abs=2.24e-3，max_abs=9.67e-3
- `bbox_preds`：cos=**1.000000**，mean_abs=2.16e-4，max_abs=7.18e-3
- `dir_cls`：cos=**1.000000**，mean_abs=6.10e-4，max_abs=3.45e-3

进程内存 VmRSS≈1.32GB（远低于 44GB 显存的 80% 阈值）。完整推理执行日志已保存：`second_onnx/onnx_infer_cpu.log`（ORT）、`second_onnx/mslite_infer_ascend.log`（MSLite）。

> **导出范围说明**：SECOND 的 `middle_encoder`（SparseEncoder）使用 **spconv 稀疏卷积**（Ascend 暂无原生支持，记录为阻塞）。本导出覆盖 `dense BEV → backbone(SECOND) → neck(SECONDFPN) → Anchor3DHead` 的 2D 检测路径并完成转换/推理/对齐；体素化+spconv 需作为 numba/AscendC 预处理前置（阶段 2 记录暂搁）。

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：需要端到端（点云→检测）
   - **原因**：spconv 稀疏卷积 Ascend 无原生支持
   - **解决方案**：阶段 2 用 AscendC 实现 spconv，或 dense 卷积近似（精度损失）；暂无则记录暂搁

2. **现象**：fp16 检测精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [mmdetection3d / SECOND](https://github.com/open-mmlab/mmdetection3d)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 mmdetection3d 上游为准（Apache-2.0）。
