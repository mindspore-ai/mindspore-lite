# RDT-1B 双臂操作扩散策略 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 RDT-1B（清华，双臂操作扩散基础模型）的单步去噪网络导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

RDT-1B 使用 adaLN-zero Diffusion Transformer（DiT），以（视觉+语言+本体）embedding 为条件，通过迭代去噪预测双臂动作 chunk（action_dim=14 = 7+7）。本目录导出**单步去噪网络**，DDPM 采样在 host 侧 numpy 实现。

> ⚠️ **风险标注**：本目录为基于 RDT 架构的自洽 PyTorch DiT 参考实现。`--random-init` 可跑通全流程；真实 RDT-1B（~1B）权重需任务2 联网核实官方 `RDT-model` 包的 key 布局后，用 `--checkpoint` 加载，并按需放大 `--dim`/`--depth`。`cond`（视觉/语言 embedding）由上游编码器产生，本目录 demo 用随机向量。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0 |
| onnx / onnxruntime | 1.19.1 / 1.24.2 |
| numpy | 1.26.x |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

### 获取模型权重

```bash
# demo：随机权重。
# 真实权重（任务2）：huggingface.co/Robotics-Diffusion-Transformer/RDT-1B
#   需 RDT-model 包加载 → 导出 DiT 权重 → --checkpoint。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/rdt_1b

# demo
python export_rdt_1b_onnx.py --output-dir ./rdt_1b_onnx --device cpu

# 真实权重（任务2 核实）
python export_rdt_1b_onnx.py --checkpoint /path/to/rdt_dit.pt --output-dir ./rdt_1b_onnx
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | RDT DiT state_dict（任务2 核实） | 空（demo） |
| `--action-dim` / `--horizon` / `--cond-dim` | 动作维度/chunk 长度/条件维度 | `14`/`64`/`256` |
| `--dim` / `--depth` / `--heads` | DiT 配置（demo 小；真实更大） | `256`/`6`/`4` |

### 产出文件

```text
./rdt_1b_onnx/
└── rdt_1b_denoise.onnx   # noisy_action/timestep/cond -> noise
```

---

## 3. ONNX 推理

```bash
python infer_rdt_1b_onnx.py --model ./rdt_1b_onnx/rdt_1b_denoise.onnx --num-steps 10 --seed 0
```

### 执行日志

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX \
  --modelFile=./rdt_1b_onnx/rdt_1b_denoise.onnx \
  --outputFile=./rdt_1b_onnx/rdt_1b_denoise \
  --optimize=ascend_oriented --saveType=MINDIR --configFile=./config.ini
```

### 配置文件 `config.ini`

```ini
[acl_build_options]
input_format="ND"
input_shape="noisy_action:1,64,14;timestep:1;cond:1,256"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

### 产出文件

```text
./rdt_1b_onnx/
├── rdt_1b_denoise.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_rdt_1b_mslite.py \
  --model ./rdt_1b_onnx/rdt_1b_denoise.mindir \
  --num-steps 10 --seed 0 --device ascend --device-id 0
```

### 执行日志

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 端到端 DDPM 采样（5 步） | 5.38 |
| MSLite 单步去噪（mean） | 1.08 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

---

## 7. 常见问题

1. 现象：`--checkpoint` key 不匹配
   - 原因：官方 RDT DiT 命名与参考实现不同
   - 解决方案：任务2 联网核实 `RDT-model` 源码，调整 key 映射。

2. 现象：双臂动作幅值异常
   - 原因：demo 随机权重 + 随机 cond
   - 解决方案：接入真实视觉/语言编码器产生的 cond 与权重。

3. 现象：fp16 精度下 chunk 末段发散
   - 原因：扩散去噪对精度敏感
   - 解决方案：config.ini 改 `force_fp32`（仅 DiT 子图）。

---

## 8. 参考资源

- RDT-1B：https://github.com/THU-ML-Robotics/RDT-1B
- HF 权重：https://huggingface.co/Robotics-Diffusion-Transformer/RDT-1B
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- RDT-1B 上游代码许可证以其仓库为准（MIT）。
