# HPT 异构预训练 Transformer 策略 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 HPT（Heterogeneous Pre-trained Transformer）策略导出为 ONNX，使用 ONNX Runtime 验证推理结果，并将 ONNX 转换为 MindSpore Lite MindIR 后在 Ascend Atlas 300I Duo 上推理与测速。

HPT 把策略分解为：每本体的 **stem**（观测 → 共享 latent token）+ 共享 **trunk**（Transformer）+ 每本体的 **head**（latent → 动作 chunk）。本目录导出**单本体的 stem+trunk+head 单前向**：observation → action chunk。

> ⚠️ **风险标注**：本目录为基于 HPT 架构的自洽 PyTorch 参考实现（单本体）。`--random-init` 可跑通全流程；真实 HPT 权重需任务2 联网核实官方 `liruiw/HPT` 的异构 stem/head key 布局后加载。跨本体部署需替换 stem/head（trunk 共享）。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# demo：随机权重。
# 真实权重（任务2）：huggingface.co/liruiw/hpt-base 等官方仓库。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/hpt
python export_hpt_onnx.py --output-dir ./hpt_onnx --device cpu
# 真实权重：python export_hpt_onnx.py --checkpoint /path/to/hpt.pt --output-dir ./hpt_onnx
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | HPT state_dict（任务2 核实） | 空（demo） |
| `--obs-dim` / `--action-dim` / `--horizon` | 观测/动作维度/chunk 长度 | `30`/`14`/`16` |
| `--dim` / `--depth` / `--heads` | trunk 配置 | `256`/`4`/`4` |

```text
./hpt_onnx/
└── hpt_policy.onnx   # observation -> action[B,horizon,action_dim]
```

---

## 3. ONNX 推理

```bash
python infer_hpt_onnx.py --model ./hpt_onnx/hpt_policy.onnx --obs-dim 30 --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./hpt_onnx/hpt_policy.onnx \
  --outputFile=./hpt_onnx/hpt_policy --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="observation:1,30"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./hpt_onnx/
├── hpt_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_hpt_mslite.py --model ./hpt_onnx/hpt_policy.mindir \
  --obs-dim 30 --seed 0 --device ascend --device-id 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 单步前向（mean） | 0.44 |
| MSLite 单步前向（p50） | 0.43 |
| 进程 RSS | 1.03 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999997 |

> 非自回归（单次前给动作 chunk），无 decode 循环。

---

## 7. 常见问题

1. 跨本体部署：trunk 共享，stem/head 按本体替换 → 需为每个本体单独导出 stem/head 子图（任务2）。
2. `--checkpoint` key 不匹配：任务2 联网核实 `liruiw/HPT` 源码调整映射。
3. obs_dim 不匹配：保持导出/转换/推理一致。

---

## 8. 参考资源

- HPT：https://github.com/liruiw/HPT
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- HPT 上游代码许可证以其仓库为准（MIT）。
