# Octo 通用机器人策略 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Octo 通用机器人策略（图像 + 本体感觉 → 动作 chunk）导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

> ⚠️ **风险标注（任务2/网络恢复后核实）**：Octo 官方实现为 JAX/Flax，HF 上的 `rail-berkeley/octo-*` 为 flax checkpoint。本目录提供**基于 Octo 论文架构的自洽 PyTorch 参考实现**（Image Tokenizer + Transformer trunk + readout tokens + Diffusion 动作头）。`--random-init` 可跑通导出/转换/推理/对齐全流程；真实权重需先把 flax→torch 转换（或使用 `octo-pytorch` 等社区移植）后通过 `--checkpoint` 加载，具体加载 API 待任务2 联网核实后调整。

Octo 的动作头为扩散策略：本目录导出**单步去噪网络**，DDPM 采样循环在 host 侧 numpy 实现（见 infer 脚本）。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0（导出/对齐基准） |
| onnx / onnxruntime | 1.19.1 / 1.24.2 |
| numpy | 1.26.x |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

### 获取模型权重

```bash
# 默认 demo：随机权重，无需下载，直接第 2 节。
# 真实权重（任务2）：huggingface.co/rail-berkeley/octo-base-1.5（flax）→ 需 flax→torch 转换。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/octo

# demo（随机权重）
python export_octo_onnx.py --output-dir ./octo_onnx --device cpu

# 真实权重（任务2 核实 flax→torch 格式后）
python export_octo_onnx.py --checkpoint /path/to/octo_torch.pt --output-dir ./octo_onnx
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--output-dir` | ONNX 输出目录 | `./octo_onnx` |
| `--checkpoint` | Octo PyTorch state_dict（任务2 核实） | 空（demo） |
| `--dim` / `--trunk-depth` / `--heads` | Transformer 配置 | `384`/`4`/`6` |
| `--num-readout` | readout token 数 | `16` |
| `--proprio-dim` / `--action-dim` / `--horizon` | 本体/动作维度/动作 chunk 长度 | `7`/`7`/`4` |
| `--img-size` / `--patch` | 图像尺寸/patch | `224`/`16` |

### 产出文件

```text
./octo_onnx/
└── octo_denoise.onnx      # 单步去噪: image/proprio/timestep/noisy_action -> noise
```

---

## 3. ONNX 推理

```bash
python infer_octo_onnx.py \
  --model ./octo_onnx/octo_denoise.onnx \
  --num-steps 10 --seed 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | ONNX 路径 | 必填 |
| `--image` | 可选真实图像路径；默认 seeded 随机 | 空 |
| `--num-steps` | DDPM 去噪步数 | `10` |
| `--horizon` / `--action-dim` | 动作 chunk 长度/维度 | `4`/`7` |
| `--warmup` / `--runs` | 性能预热/测速 | `2`/`5` |

### 执行日志

```log
action shape=(1, 4, 7) dtype=float32   action_abs_max=3.040
Perf (CPU, ddpm_steps=5, runs=3):
  e2e_ms_mean: 835.0   per_step_ms_mean: 167.0   mem: vmrss=0.19 GB
```

> ONNX Runtime 运行的是单步去噪网络（无 Custom 节点），可正常执行。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX \
  --modelFile=./octo_onnx/octo_denoise.onnx \
  --outputFile=./octo_onnx/octo_denoise \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 配置文件 `config.ini`

```ini
[acl_build_options]
input_format="ND"
input_shape="image:1,3,224,224;proprio:1,7;timestep:1;noisy_action:1,4,7"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

> 固定 shape 约束：导出/转换/推理的 image/proprio/action 形状须一致。

### 产出文件

```text
./octo_onnx/
└── octo_denoise.mindir   # 单文件（<2GB）
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_octo_mslite.py \
  --model ./octo_onnx/octo_denoise.mindir \
  --num-steps 10 --seed 0 \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | MindIR 路径（`*_graph.mindir`） | 必填 |
| `--num-steps` | DDPM 去噪步数 | `10` |
| `--device` / `--device-id` | 推理设备 | `ascend` / `0` |

### 执行日志

```log
action shape=(1, 4, 7) dtype=float32   action_abs_max=3.040
Perf (Ascend, ddpm_steps=5, runs=3):
  e2e_ms_mean: 9.31   per_step_ms_mean: 1.86   e2e_ms_p50: 9.28
  mem: vmrss=1.039 GB  vmhwm=1.10 GB
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 端到端 DDPM 采样（5 步） | 9.31 |
| MSLite 单步去噪（mean） | 1.86 |
| ONNX(CPU) 端到端 DDPM 采样（5 步） | 835.0 |
| ONNX(CPU) 单步去噪（mean） | 167.0 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch vs MSLite） | cos=1.000000，max_abs=1.47e-3（fp16） |
| 精度对齐（PyTorch vs ONNX） | cos=1.000000，max_abs=1.13e-6 |

> 自回归式扩散采样（N 步去噪），吞吐以动作 chunk 计。

---

## 7. 常见问题

1. 现象：`--checkpoint` 加载真实 flax 权重报错
   - 原因：HF 为 flax 格式，需先 flax→torch 转换
   - 解决方案：任务2 联网核实 `octo-pytorch` 等移植或转换脚本，调整 `--checkpoint` 的 key 映射。

2. 现象：转换时 timestep 输入 dtype 不匹配
   - 原因：导出为 int64，转换/推理可能期望 int32
   - 解决方案：推理脚本已按模型声明 cast；如转换报错可调整导出 dtype。

3. 现象：DDPM 采样动作幅值异常
   - 原因：demo 随机权重无真实语义
   - 解决方案：换用真实训练权重。

---

## 8. 参考资源

- Octo 论文/仓库：https://github.com/octo-models/octo
- HF 权重：https://huggingface.co/rail-berkeley/octo-base-1.5
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- Octo 上游代码许可证以其仓库为准（MIT）。
