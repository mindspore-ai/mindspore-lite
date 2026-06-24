# LaneATT ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 LaneATT（anchor-based 车道线检测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。LaneATT 为纯卷积（ResNet 主干 + 注意力 head）结构，算子友好。

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
# 从 lucastabelini/LaneAtt 源码安装（提供 models.lanenet.LaneNet）
```

权重来源：[lucastabelini/LaneAtt](https://github.com/lucastabelini/LaneAtt)。

---

## 2. 模型说明

```log
输入图像 (1,3,288,800) → ResNet backbone → LaneATT head(anchor + attention) → 车道存在性 + 偏移
```

| 类型 | 名称             | 说明                       |
| ---- | ---------------- | -------------------------- |
| 输入 | img              | \[1, 3, 288, 800] 前视图像 |
| 输出 | lane_existence   | 车道存在性（按 anchor）    |
| 输出 | lane_offsets     | 车道横向偏移回归           |

> 导出为 head 原始输出；NMS/拟合留后处理（阶段 2 numpy 实现，用于精度对齐）。

---

## 3. ONNX 导出

```bash
cd examples/base_models/laneatt
python export_laneatt_onnx.py \
  --model-module models.lanenet --model-class LaneNet \
  --checkpoint /path/to/laneatt.pth \
  --output laneatt_onnx/laneatt.onnx --img-h 288 --img-w 800 --opset 17
```

产出：`laneatt_onnx/laneatt.onnx`

---

## 4. ONNX 推理

```bash
python infer_laneatt_onnx.py --model ./laneatt_onnx/laneatt.onnx --img-h 288 --img-w 800
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input, shape=(1, 3, 288, 800), seed=1024
  lane_existence: (1, N_anchors)
  lane_offsets: (1, N_anchors, ...)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./laneatt_onnx/laneatt.onnx \
  --outputFile=./laneatt_onnx/laneatt_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_laneatt_mslite.py \
  --model ./laneatt_onnx/laneatt_ascend.mindir \
  --device ascend --device-id 0 --img-h 288 --img-w 800
```

执行日志（占位）：

```log
latency_ms_mean: TBD
VmRSS: TBD KB
```

---

## 7. 性能数据与资源约束

测试环境：Atlas 300I Duo（310P3），CANN 8.3.RC1，MindSpore Lite 2.10.0
测试条件：输入 `(1, 3, 288, 800)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：导出报模型类/字段不符
   - **原因**：LaneAtt 上游版本接口差异
   - **解决方案**：阶段 2 按实际 `LaneNet` 接口调整 `LaneATTWrapper`

2. **现象**：fp16 精度不足
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [lucastabelini/LaneAtt](https://github.com/lucastabelini/LaneAtt)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 LaneAtt 上游为准（MIT）。
