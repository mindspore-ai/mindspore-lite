# MTR++ ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 MTR++（Motion Transformer + ego-actor 场景交互）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。MTR++ 在 MTR 基础上引入 agent 间场景级交互建模，IO 与 MTR 完全一致，算子以标准 Transformer 为主。

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

权重来源：[sshaoshuai/MTR](https://github.com/sshaoshuai/MTR)（MTR++ checkpoint，提供 `model.MotionTransformer`）。

---

## 2. 模型说明

```log
actor_history (1,64,11,9) ─┐
                           ├─→ MTR++(interaction) → trajectory (1,6,80,2) + scores (1,6)
map_polylines (1,768,9)   ─┘
```

| 类型 | 名称            | Shape             | 说明                 |
| ---- | --------------- | ----------------- | -------------------- |
| 输入 | actor_history   | \[1, 64, 11, 9]   | 64 agent × 11 步    |
| 输入 | map_polylines   | \[1, 768, 9]      | 地图折线特征         |
| 输出 | trajectory      | \[1, 6, 80, 2]    | 6 模态 × 80 步 (x,y) |
| 输出 | scores          | \[1, 6]           | 各模态概率           |

> 与 MTR 区别：MTR++ 用 ego-actor 交互模块联合解码所有 agent，输出含交互上下文。

---

## 3. ONNX 导出

```bash
cd examples/base_models/mtr_plus
python export_mtr_plus_onnx.py \
  --model-module model --model-class MotionTransformer \
  --checkpoint /path/to/mtr_plus.pt \
  --output mtr_plus_onnx/mtr_plus.onnx --opset 17
```

产出：`mtr_plus_onnx/mtr_plus.onnx`

---

## 4. ONNX 推理

```bash
python infer_mtr_plus_onnx.py --model ./mtr_plus_onnx/mtr_plus.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input: actor_history=(1, 64, 11, 9), map_polylines=(1, 768, 9), seed=1024
  trajectory: (1, 6, 80, 2)
  scores: (1, 6)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./mtr_plus_onnx/mtr_plus.onnx \
  --outputFile=./mtr_plus_onnx/mtr_plus_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_mtr_plus_mslite.py \
  --model ./mtr_plus_onnx/mtr_plus_ascend.mindir \
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
测试条件：`actor_history=(1,64,11,9)`、`map=(1,768,9)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：交互模块 forward 额外参数
   - **原因**：MTR++ ego-actor 交互需额外输入（如 valid mask）
   - **解决方案**：阶段 2 按实际 forward 调整 wrapper 输入

2. **现象**：fp16 轨迹发散
   - **解决方案**：`config.ini` 改 `force_fp32`

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [sshaoshuai/MTR (MTR++)](https://github.com/sshaoshuai/MTR)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 MTR 上游为准（Apache-2.0）。
