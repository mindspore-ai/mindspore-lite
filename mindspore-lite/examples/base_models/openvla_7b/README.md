# OpenVLA-7B 视觉-语言-动作策略 ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 OpenVLA（Prismatic VLM + 动作 token）导出为 ONNX，并在 Ascend Atlas 300I Duo 上转换为 MindIR 推理。

> ⚠️ **风险标注（重要）**：真实 OpenVLA-7B = SigLIP+DINOv2 视觉塔 + Llama-2-7B LLM + 自回归动作 token（7B VLM，需 prefill/decode 拆分，参照 `internvl3_5_1b`）。本目录提供**视觉条件回归骨架**（image + 语言 task token → 动作 chunk，单前向）用于跑通迁移管线；`--random-init` 可端到端验证导出/转换/推理/对齐。任务2（网络恢复后）应改用 `prismatic`/transformers 官方加载真实权重 + prefill/decode 子图拆分，替换本骨架。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：huggingface.co/openvla/openvla-7b（prismatic 格式）。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/openvla_7b
python export_openvla_7b_onnx.py --output-dir ./openvla_7b_onnx --device cpu
# 真实权重（任务2）：python export_openvla_7b_onnx.py --checkpoint /path/to/openvla.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | OpenVLA state_dict（任务2 用 prismatic） | 空（demo） |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |
| `--task-len` / `--vocab-size` | 语言 token 长度/词表 | `16`/`32000` |
| `--dim` / `--depth` | 骨架 transformer 配置（demo 小） | `384`/`4` |

```text
./openvla_7b_onnx/
└── openvla_7b_policy.onnx   # image+task_tokens -> action[B,horizon,action_dim]
```

---

## 3. ONNX 推理

```bash
python infer_openvla_7b_onnx.py --model ./openvla_7b_onnx/openvla_7b_policy.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./openvla_7b_onnx/openvla_7b_policy.onnx \
  --outputFile=./openvla_7b_onnx/openvla_7b_policy --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="image:1,3,224,224;task_tokens:1,16"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./openvla_7b_onnx/
├── openvla_7b_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_openvla_7b_mslite.py --model ./openvla_7b_onnx/openvla_7b_policy.mindir \
  --seed 0 --device ascend --device-id 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 单步前向（mean） | 1.48 |
| MSLite 单步前向（p50） | 1.48 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

---

## 7. 常见问题

1. 真实 OpenVLA 自回归 action token：任务2 用 prismatic 加载 + vision/LLM prefill/decode 拆分（参照 internvl3_5_1b）。
2. fp16 精度：7B 模型在 44GB 上用 `allow_fp32_to_fp16`；先 DEFAULT(fp16) 转换确认输出。
3. task_tokens dtype：导出 int64，推理按模型声明 cast。

---

## 8. 参考资源

- OpenVLA：https://github.com/openvla/openvla
- HF 权重：https://huggingface.co/openvla/openvla-7b
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- OpenVLA 上游代码许可证以其仓库为准（MIT）。
