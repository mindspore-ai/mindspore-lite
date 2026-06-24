# VectorNet ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 VectorNet（轨迹预测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。VectorNet 将 agent 历史与地图折线向量化后用层级图网络预测目标 agent 的未来轨迹，主要算子为 MLP/GEMM/attention，算子友好。

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

权重来源：Argoverse forecasting / VectorNet 上游实现（提供 `model.VectorNet`）。

---

## 2. 模型说明

```log
agent_hist (1,20,4) ─┐
                     ├─→ polyline encoder + global graph → trajectory (1,30,2)
map_polyline (1,100,9)┘
```

| 类型 | 名称           | Shape               | 说明                |
| ---- | -------------- | ------------------- | ------------------- |
| 输入 | agent_hist     | \[1, 20, 4]         | 目标 agent 历史轨迹 |
| 输入 | map_polyline   | \[1, 100, 9]        | 地图车道折线特征    |
| 输出 | trajectory     | \[1, 30, 2]         | 未来 30 步 (x, y)   |

> 上述 shape 为导出默认（可经 `--obs-len/--pred-len/--map-poly-num` 调整）。多模态版本可输出 `[1, K, 30, 2]`（K 个模态）。

---

## 3. ONNX 导出

```bash
cd examples/base_models/vectornet
python export_vectornet_onnx.py \
  --model-module model --model-class VectorNet \
  --checkpoint /path/to/vectornet.pth \
  --output vectornet_onnx/vectornet.onnx --opset 17
```

产出：`vectornet_onnx/vectornet.onnx`

---

## 4. ONNX 推理

```bash
python infer_vectornet_onnx.py --model ./vectornet_onnx/vectornet.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input: agent_hist=(1, 20, 4), map_polyline=(1, 100, 9), seed=1024
  trajectory: (1, 30, 2)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./vectornet_onnx/vectornet.onnx \
  --outputFile=./vectornet_onnx/vectornet_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_vectornet_mslite.py \
  --model ./vectornet_onnx/vectornet_ascend.mindir \
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
测试条件：`agent_hist=(1,20,4)`、`map_polyline=(1,100,9)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：模型构造参数不符
   - **原因**：上游 VectorNet 构造签名不同
   - **解决方案**：阶段 2 按实际签名调整 `export_vectornet_onnx.py` 的 `ModelCls(...)` 调用

2. **现象**：fp16 轨迹精度下降
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Argoverse Forecasting / VectorNet](https://github.com/jagjeet-singh/argoverse-forecasting)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以上游为准。
