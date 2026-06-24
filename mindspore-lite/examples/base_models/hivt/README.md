# HiVT ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 HiVT（层级向量 Transformer 轨迹预测）导出为 ONNX，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上推理测速。HiVT 用 local attention 编码 agent/地图向量，输出多模态轨迹，算子以 Transformer + MLP 为主，算子友好。

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

权重来源：[ZikangZhou/HiVT](https://github.com/ZikangZhou/HiVT)（提供 `model.HiVT`）。

---

## 2. 模型说明

```log
actor_history (1,20,20,7) ─┐
                           ├─→ HiVT(local attention) → trajectory (1,6,30,2)
map_lanes (1,200,7)        ┘     (6 个模态)
```

| 类型 | 名称           | Shape               | 说明                 |
| ---- | -------------- | ------------------- | -------------------- |
| 输入 | actor_history  | \[1, 20, 20, 7]     | 20 个 agent × 20 步 |
| 输入 | map_lanes      | \[1, 200, 7]        | 地图车道向量         |
| 输出 | trajectory     | \[1, 6, 30, 2]      | 6 模态 × 30 步 (x,y) |

> shape 可经 `--num-agents/--map-num/--pred-len/--num-modes` 调整。

---

## 3. ONNX 导出

```bash
cd examples/base_models/hivt
python export_hivt_onnx.py \
  --model-module model --model-class HiVT \
  --checkpoint /path/to/hivt.ckpt \
  --output hivt_onnx/hivt.onnx --opset 17
```

产出：`hivt_onnx/hivt.onnx`

---

## 4. ONNX 推理

```bash
python infer_hivt_onnx.py --model ./hivt_onnx/hivt.onnx
```

执行日志（占位，阶段 2 实测后替换）：

```log
Using random input: actor_history=(1, 20, 20, 7), map_lanes=(1, 200, 7), seed=1024
  trajectory: (1, 6, 30, 2)
latency_ms_mean: TBD
```

---

## 5. MindSpore Lite 转换

```bash
Converter=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$Converter --fmk=ONNX --modelFile=./hivt_onnx/hivt.onnx \
  --outputFile=./hivt_onnx/hivt_ascend \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：`ge.exec.precision_mode=force_fp16`。

```log
CONVERT RESULT SUCCESS:0
```

---

## 6. MindSpore Lite 推理

```bash
python infer_hivt_mslite.py \
  --model ./hivt_onnx/hivt_ascend.mindir \
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
测试条件：`actor_history=(1,20,20,7)`、`map_lanes=(1,200,7)`，固定随机种子 1024，warmup=5 / runs=20

| 后端           | 设备  | 延迟 mean (ms)        | 延迟 p99 (ms) | 备注       |
| -------------- | ----- | --------------------- | ------------- | ---------- |
| ONNX Runtime   | CPU   | TBD                   | TBD           | 精度基准   |
| MindSpore Lite | 310P3 | TBD（阶段2实测刷新）  | TBD           | force_fp16 |

> **资源约束**：内存/显存占用不得超过总量 80%（310P3 显存阈值 ~35.4GB；系统 RAM 同理）。阶段 2 验证前执行 `npu-smi info` + `free -h`，单进程单卡、选最空闲卡、fp16 优先；脚本已内置 VmRSS/VmHWM 监控。

---

## 8. 常见问题

1. **现象**：state_dict key 前缀不符
   - **原因**：HiVT checkpoint 可能含 `state_dict` 包裹或 module 前缀
   - **解决方案**：阶段 2 按 `torch.load` 实际结构调整解包

2. **现象**：fp16 轨迹发散
   - **解决方案**：`config.ini` 改 `force_fp32`

3. **阶段 2 阻塞（已记录暂搁）**：HiVT 依赖 `torch_geometric`（`MessagePassing`/`propagate`/`scatter`/`subgraph`）并以图结构 `TemporalData` 为输入（含 `num_nodes`/`edge_index`/`padding_mask`/`rotate_angles`/`positions` 等），当前环境未装 torch_geometric，且 MessagePassing 的稀疏图聚合无法直接 `torch.onnx.export`，需将稀疏注意力改写为 dense 等价实现（属模型移植，非纯验证）。权重 ckpt 已从仓库 `upstream/HiVT/checkpoints/` 成功获取（HiVT-64 8.1MB），待移植后继续。

---

## 9. 参考资源与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [ZikangZhou/HiVT](https://github.com/ZikangZhou/HiVT)

脚本遵循 MindSpore Lite 仓库许可证；模型权重许可证以 HiVT 上游为准。
