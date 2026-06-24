# Diffusion Policy 动作扩散策略 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Diffusion Policy（Chi et al.，基于动作扩散的视觉运动策略）的单步去噪网络导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

Diffusion Policy 用 FiLM 条件的 1D U-Net（`ConditionalUnet1D`）通过迭代去噪预测动作 chunk。本目录导出**单步去噪网络**，DDPM 采样循环在 host 侧 numpy 实现。默认配置对应 PushT 低维示例（action_dim=2，obs_dim=2，horizon=16）。

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
# demo：随机权重，无需下载。
# 真实权重：diffusion_policy 训练产物 logs/<exp>/<run>/checkpoints/latest.pth
#   （含 ema_unet / model.state_dict()），导出时 --checkpoint 指定。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/diffusion_policy

# demo
python export_diffusion_policy_onnx.py --output-dir ./diffusion_policy_onnx --device cpu

# 真实权重
python export_diffusion_policy_onnx.py \
  --checkpoint /path/to/latest.pth \
  --action-dim 2 --obs-dim 2 --horizon 16 \
  --output-dir ./diffusion_policy_onnx
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--output-dir` | ONNX 输出目录 | `./diffusion_policy_onnx` |
| `--checkpoint` | diffusion_policy `.pth`（ema_unet/model） | 空（demo） |
| `--action-dim` / `--obs-dim` / `--horizon` | 动作/观测维度/动作 chunk 长度 | `2`/`2`/`16` |
| `--emb-dim` / `--down-dims` | UNet 配置 | `256`/`256,512,1024` |

### 产出文件

```text
./diffusion_policy_onnx/
└── diffusion_policy.onnx   # noisy_action/timestep/obs -> noise
```

---

## 3. ONNX 推理

```bash
python infer_diffusion_policy_onnx.py \
  --model ./diffusion_policy_onnx/diffusion_policy.onnx \
  --num-steps 10 --seed 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | ONNX 路径 | 必填 |
| `--num-steps` | DDPM 去噪步数 | `10` |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `2`/`16` |
| `--warmup` / `--runs` | 性能预热/测速 | `2`/`5` |

### 执行日志

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX \
  --modelFile=./diffusion_policy_onnx/diffusion_policy.onnx \
  --outputFile=./diffusion_policy_onnx/diffusion_policy \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./config.ini
```

### 配置文件 `config.ini`

```ini
[acl_build_options]
input_format="ND"
input_shape="noisy_action:1,2,16;timestep:1;obs:1,2"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

> 固定 shape 约束：导出/转换/推理的 `action_dim`/`horizon`/`obs_dim` 须一致。

### 产出文件

```text
./diffusion_policy_onnx/
├── diffusion_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_diffusion_policy_mslite.py \
  --model ./diffusion_policy_onnx/diffusion_policy.mindir \
  --num-steps 10 --seed 0 \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model` | MindIR 路径（`*.mindir`） | 必填 |
| `--num-steps` | DDPM 去噪步数 | `10` |
| `--device` / `--device-id` | 推理设备 | `ascend` / `0` |

### 执行日志

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 端到端 DDPM 采样（5 步） | 82.9 |
| MSLite 单步去噪（mean） | 16.6 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999996 |

---

## 7. 常见问题

1. 现象：`--checkpoint` 加载时 key 不匹配
   - 原因：diffusion_policy 的 state_dict 含 `ema_unet.`/`model.` 前缀，且含 critic 等多余权重
   - 解决方案：脚本已剥离 `ema_unet.` 前缀并 `strict=False`；任务2 按训练配置核对 action_dim/obs_dim/horizon。

2. 现象：GroupNorm 报 groups>channels
   - 原因：action_dim/中间通道数小于 groups
   - 解决方案：FiLMBlock 已用 `min(groups, channels)`；自定义通道时确保为 groups 倍数。

3. 现象：horizon 必须能被 4 整除
   - 原因：3 层 encoder 含 2 次 stride-2 downsample
   - 解决方案：选用 horizon=16/32/64 等。

---

## 8. 参考资源

- Diffusion Policy：https://github.com/real-stanford/diffusion_policy
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- Diffusion Policy 上游代码许可证以其仓库为准（MIT）。
