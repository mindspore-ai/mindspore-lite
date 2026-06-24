# UniAD ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 UniAD（首个真正的端到端多任务自动驾驶框架）导出为 ONNX，并转换为 MindSpore Lite MindIR 在 Ascend NPU 上推理部署。UniAD 联合执行 BEVFormer 编码 + 检测 + 跟踪 + 运动预测 + 占用预测 + 规划。

> **阶段 2 核心风险点（最高难度）**：含 BEVFormer 的 Temporal Deformable Attention + 跟踪/占用多模块胶水逻辑。阶段 2 大概率在 deformable 算子处阻塞，且工程量极大；按 plan「记录暂搁」。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| mmcv-full / mmdet / mmdetection3d / mmdet (UniAD fork) | 1.x |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
# 从 OpenDriveLab/UniAD 安装其 mmdet fork 与依赖
pip install onnx==1.14.0 onnxruntime==1.16.0
```

权重来源：[OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)。

---

## 2. 模型说明

```log
多视图图像 (1,6,3,320,800) → BEVFormer(deformable) → 多任务联合头(检测/跟踪/运动/占用/规划)
                                                              ↓
                                          planning_traj (1,6,3) + 各任务输出
```

| 类型 | 名称           | Shape                | 说明               |
| ---- | -------------- | -------------------- | ------------------ |
| 输入 | imgs           | \[1, 6, 3, 320, 800] | 6 路相机图像       |
| 输出 | planning_traj  | \[1, 6, 3]           | 自车未来轨迹 (x,y,θ)|
| 输出 | det_cls        | \[1, Q, num_cls]     | 检测 query 类别    |

> 完整 UniAD 还输出跟踪/运动/占用；scaffold 仅暴露 planning + det 作为代表。

---

## 3. ONNX 导出

```bash
cd examples/base_models/uniad
python export_uniad_onnx.py \
  --model-module model --model-class UniAD \
  --checkpoint /path/to/uniad.pth \
  --output uniad_onnx/uniad.onnx --opset 17
```

产出：`uniad_onnx/uniad.onnx`（含 Custom deformable 节点）

---

## 4. ONNX 推理

> Custom 算子 ORT 无法执行，`infer_uniad_onnx.py` 仅用于结构检查；推理走 MindIR。

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./uniad_onnx/uniad.onnx \
  --outputFile=./uniad_onnx/uniad_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp32`。

> 阶段 2 在 deformable + 多模块处大概率失败 → 记录暂搁。

---

## 6. MindSpore Lite 推理

```bash
python infer_uniad_mslite.py \
  --model ./uniad_onnx/uniad_ascend.mindir \
  --device ascend --device-id 0
```

执行日志（占位）：

```log
latency_ms_mean: TBD
VmRSS: TBD KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1,6,3,320,800)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 备注                       |
| -------------- | ----- | --------------------- | -------------------------- |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | 需 deformable + 多模块     |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。UniAD 体量大，阶段 2 尤需严格监控；验证前 `npu-smi info` + `free -h`，单进程单卡、fp32 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：转换报 deformable / 多模块算子不支持
   - **原因**：BEVFormer deformable + track/occ 模块
   - **解决方案**：阶段 2 用 AscendC deformable + 逐模块拆分；暂无则记录暂搁（优先级最低）

2. **现象**：显存超限
   - **解决方案**：拆分为子图（BEV encoder / 各 task head）分批推理

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 UniAD 上游为准。
