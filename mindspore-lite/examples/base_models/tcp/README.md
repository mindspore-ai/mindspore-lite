# TCP ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 TCP（Trajectory-guided Control Prediction，端到端驾驶）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。TCP 以 ResNet 感知前视图像 + 速度 + 导航命令，直接输出控制信号与轨迹，结构为 CNN+MLP，是最轻量的端到端基线。

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

权重来源：[OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP)（提供 `model.TCP`）。

---

## 2. 模型说明

```log
front_img (1,3,256,512) ─┐
speed (1,1)              ─┤→ ResNet + 双分支(trajectory / control) → control (1,3) + trajectory (1,10,2)
command one-hot (1,4)    ─┘                                          (steer, throttle, brake)
```

| 类型 | 名称        | Shape               | 说明                       |
| ---- | ----------- | ------------------- | -------------------------- |
| 输入 | front_img   | \[1, 3, 256, 512]   | 前视相机图像               |
| 输入 | speed       | \[1, 1]             | 当前车速                   |
| 输入 | command     | \[1, 4] one-hot     | 导航命令(左/右/直行/未知)   |
| 输出 | control     | \[1, 3]             | steer / throttle / brake   |
| 输出 | trajectory  | \[1, 10, 2]         | 未来 10 个路径点 (x, y)    |

---

## 3. ONNX 导出

```bash
cd examples/base_models/tcp
python export_tcp_onnx.py \
  --model-module model --model-class TCP \
  --checkpoint /path/to/tcp.ckpt \
  --output tcp_onnx/tcp.onnx --opset 17
```

产出：`tcp_onnx/tcp.onnx`

---

## 4. ONNX 推理

```bash
python infer_tcp_onnx.py --model ./tcp_onnx/tcp.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input: front_img=(1, 3, 256, 512), speed=(1, 1), command=(1, 4), seed=1024
  control: (1, 3)
  trajectory: (1, 10, 2)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./tcp_onnx/tcp.onnx \
  --outputFile=./tcp_onnx/tcp_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_tcp_mslite.py \
  --model ./tcp_onnx/tcp_ascend.mindir \
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
测试条件：`front_img=(1,3,256,512)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：command 输入维度不符
   - **原因**：TCP 不同版本 command 为 one-hot \[1,4] 或标量 \[1]
   - **解决方案**：阶段 2 按实际 forward 调整 `--command` shape

2. **现象**：fp16 控制信号异常
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [OpenDriveLab/TCP](https://github.com/OpenDriveLab/TCP)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 TCP 上游为准。
