# LaneGCN ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 LaneGCN（车道线图卷积轨迹预测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。LaneGCN 用图卷积在车道线节点上聚合特征，结合 agent 历史预测未来轨迹。

> **阶段 2 风险点**：图卷积若用稀疏 gather/scatter 实现，可能需替换为 dense matmul 等价表达（见 README FAQ）。

---

## 1. 环境准备

| 软件包            | 版本        |
|----------------|-----------|
| Python         | 3.10      |
| torch          | 2.0.0     |
| onnx / onnxruntime | 1.14.0 / 1.16.0 |
| CANN           | 8.3.RC1   |
| mindspore-lite | 2.10.0    |

```bash
pip install torch==2.0.0 onnx==1.14.0 onnxruntime==1.16.0 --index-url https://download.pytorch.org/whl/cpu
```

权重来源：[Turoad/LaneGCN-pytorch](https://github.com/Turoad/LaneGCN-pytorch)（提供 `model.LaneGCN`）。

---

## 2. 模型说明

```log
actor_hist (1,20,6) ─┐
                     ├─→ LaneGCN(图卷积) → trajectory (1,30,2)
lane_nodes (1,500,4) ┘
```

| 类型 | 名称         | Shape           | 说明                |
| ---- | ------------ | --------------- | ------------------- |
| 输入 | actor_hist   | \[1, 20, 6]     | 目标 agent 历史     |
| 输入 | lane_nodes   | \[1, 500, 4]    | 车道线节点特征      |
| 输出 | trajectory   | \[1, 30, 2]     | 未来 30 步 (x, y)   |

---

## 3. ONNX 导出

```bash
cd examples/base_models/lanegcn
python export_lanegcn_onnx.py \
  --model-module model --model-class LaneGCN \
  --checkpoint /path/to/lanegcn.pt \
  --output lanegcn_onnx/lanegcn.onnx --opset 17
```

产出：`lanegcn_onnx/lanegcn.onnx`

---

## 4. ONNX 推理

```bash
python infer_lanegcn_onnx.py --model ./lanegcn_onnx/lanegcn.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input: actor_hist=(1, 20, 6), lane_nodes=(1, 500, 4), seed=1024
  trajectory: (1, 30, 2)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./lanegcn_onnx/lanegcn.onnx \
  --outputFile=./lanegcn_onnx/lanegcn_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_lanegcn_mslite.py \
  --model ./lanegcn_onnx/lanegcn_ascend.mindir \
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
测试条件：`actor_hist=(1,20,6)`、`lane_nodes=(1,500,4)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：转换报 gather/scatter 算子不支持
   - **原因**：LaneGCN 图卷积用稀疏聚合
   - **解决方案**：阶段 2 将图卷积改写为 dense matmul（用全邻接矩阵）重导出

2. **现象**：fp16 轨迹精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Turoad/LaneGCN-pytorch](https://github.com/Turoad/LaneGCN-pytorch)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 LaneGCN 上游为准（Apache-2.0）。
