# Legged Gym 运动控制策略（rsl_rl MLP Actor）ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Legged Gym / rsl_rl 训练得到的运动控制策略（MLP Actor，观测向量 → tanh 有界动作）导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

Legged Gym 使用 `rsl_rl` 训练 MLP Actor-Critic；部署时仅需确定性的 Actor 主干。本目录脚本**自包含**，无需 import legged_gym / rsl_rl 源码：

- 默认 `--random-init`：用固定种子随机权重构建 demo 策略，可跑通导出/转换/推理/对齐全流程（精度对齐为 ONNX vs MSLite，验证转换正确性）。
- `--checkpoint <actor_critic.pt>`：加载真实 rsl_rl 训练权重，得到真实运动控制行为。

默认网络（ANYmal 类四足）：obs_dim=235，action_dim=18，hidden_dims=(512,256,128)，activation=elu，输出 tanh。可通过 CLI 覆盖。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0（导出/对齐基准用） |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 1.26.x |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

### 获取模型权重

```bash
# 方式一：demo 模式（无需权重，默认随机初始化），直接跳到第 2 节。

# 方式二：使用 Legged Gym / rsl_rl 训练好的 actor_critic.pt
#   来自 legged_gym 训练产物 logs/<exp>/<run>/model_*.pt
#   导出时通过 --checkpoint 指定。
```

> `MODEL_DIR`/checkpoint：训练得到的 `actor_critic.pt`（含 `model_state_dict`，其中 `actor.*` 为 Actor 主干权重）。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/legged_gym_policy

# demo 模式（随机权重，跑通流程）
python export_legged_gym_policy_onnx.py \
  --output-dir ./legged_gym_policy_onnx \
  --device cpu

# 真实训练权重
python export_legged_gym_policy_onnx.py \
  --checkpoint /path/to/actor_critic.pt \
  --obs-dim 235 --action-dim 18 --hidden-dims 512,256,128 \
  --output-dir ./legged_gym_policy_onnx --device cpu
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--output-dir` | ONNX 输出目录 | `./legged_gym_policy_onnx` |
| `--checkpoint` | rsl_rl ActorCritic `.pt`，省略则用 demo 随机权重 | 空 |
| `--obs-dim` | 观测维度 | `235` |
| `--action-dim` | 动作维度 | `18` |
| `--hidden-dims` | 隐藏层维度（逗号分隔） | `512,256,128` |
| `--activation` | 激活函数 | `elu` |
| `--no-output-tanh` | 关闭输出 tanh（无界动作） | 关 |
| `--opset` | ONNX opset 版本 | `17` |

### 产出文件

```text
./legged_gym_policy_onnx/
└── legged_gym_policy.onnx      # 输入 observation[1,235] -> 输出 action[1,18]
```

---

## 3. ONNX 推理

### 推理命令

```bash
python infer_legged_gym_policy_onnx.py \
  --model ./legged_gym_policy_onnx/legged_gym_policy.onnx \
  --obs-dim 235 --seed 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | ONNX 路径 | 必填 |
| `--obs-npy` | 观测 `.npy`（`[obs_dim]` 或 `[B,obs_dim]`） | 空 |
| `--obs` | 逗号分隔观测向量 | 空 |
| `--obs-dim` | 观测维度 | `235` |
| `--seed` | 随机观测种子（无 `--obs`/`--obs-npy` 时生效） | `0` |
| `--warmup` / `--runs` | 性能预热/测速次数 | `10` / `50` |

### 执行日志

```log
action shape=(1, 18) dtype=float32
action_abs_max=0.256836
Perf (CPU, runs=5):
  latency_ms_mean: 0.123   p50: 0.121   p90: 0.127
  mem: vmrss=0.062 GB
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

`converter_lite` 为 MindSpore Lite 版本包中的离线转换工具。

```bash
converter_lite --fmk=ONNX \
  --modelFile=./legged_gym_policy_onnx/legged_gym_policy.onnx \
  --outputFile=./legged_gym_policy_onnx/legged_gym_policy \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 配置文件 `config.ini`

```ini
[acl_build_options]
input_format="ND"
input_shape="observation:1,235"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

> 固定 shape 约束：`ascend_oriented` 按 `input_shape` 编译；推理侧观测须 pad/截断到 `obs_dim=235`（或按实际 `--obs-dim` 同步修改 `config.ini` 与导出参数）。

### 产出文件

```text
./legged_gym_policy_onnx/
└── legged_gym_policy.mindir   # 单文件（模型 <2GB，无 _variables/）
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_legged_gym_policy_mslite.py \
  --model ./legged_gym_policy_onnx/legged_gym_policy.mindir \
  --obs-dim 235 --seed 0 \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | MindIR 路径（`*_graph.mindir`） | 必填 |
| `--obs-npy` / `--obs` | 真实观测 | 空 |
| `--device` | 推理设备 | `ascend` |
| `--device-id` | Ascend 设备 ID | `0` |
| `--warmup` / `--runs` | 性能预热/测速次数 | `10` / `50` |

### 执行日志

```log
action[0][:6]=[ 0.2568  0.1104  0.0812  0.1007  0.0911 -0.0162]
action_abs_max=0.256836
Perf (Ascend, runs=20):
  latency_ms_mean: 0.356   p50: 0.352   p90: 0.375
  mem: vmrss=1.028 GB  vmhwm=1.029 GB
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 单步前向（mean） | 0.356 |
| MSLite 单步前向（p50） | 0.352 |
| MSLite 单步前向（p90） | 0.375 |
| ONNX(CPU) 单步前向（mean） | 0.123 |
| 进程 RSS | 1.03 |
| 精度对齐（PyTorch vs MSLite） | cos=1.000000，max_abs=1.66e-4（fp16） |
| 精度对齐（PyTorch vs ONNX） | cos=1.000000，max_abs=1.12e-7 |

> 非自回归模型（单次前向给出动作），无 decode 循环。性能以 `infer_*_mslite.py` 端到端打印为准。

---

## 7. 常见问题

1. 现象：`--checkpoint` 导出时报 `No actor weights matched`
   - 原因：rsl_rl 版本的 Actor 命名与脚本假设不符
   - 解决方案：检查 checkpoint 的 `state_dict` key（`torch.load(...).get('model_state_dict')`），调整导出脚本的 key 前缀；或先用 demo 模式跑通流程。

2. 现象：转换或推理时观测维度不匹配
   - 原因：`config.ini` 的 `input_shape` 与实际 `--obs-dim` 不一致
   - 解决方案：保持导出 `--obs-dim`、`config.ini` `input_shape`、推理 `--obs-dim` 三者一致。

3. 现象：demo 模式下 action 数值"随机"
   - 原因：随机权重无真实语义，仅用于流程验证
   - 解决方案：换用 `--checkpoint` 加载真实训练策略。

---

## 8. 参考资源

- Legged Gym：https://github.com/leggedrobotics/legged_gym
- rsl_rl：https://github.com/leggedrobotics/rsl_rl
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- Legged Gym / rsl_rl 上游代码许可证以其仓库为准（BSD-3-Clause）。
