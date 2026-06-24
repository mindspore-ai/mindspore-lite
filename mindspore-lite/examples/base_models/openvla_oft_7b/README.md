# OpenVLA-OFT-7B 流式动作策略 ONNX 导出与 MindSpore Lite 推理部署教程

OpenVLA-OFT / OFT+（Stanford）是 OpenVLA 的优化微调，采用**流式 action token（action-flow 头）**提升精度并支持更长动作 chunk。本目录骨架与 `openvla_7b` 同构（image + 语言 task token → 动作 chunk，单前向），用于跑通迁移管线。

> ⚠️ **风险标注**：真实 OFT = Prismatic VLM（Llama-2-7B）+ action-flow 头，需 prefill/decode 拆分（参照 `internvl3_5_1b`）。本目录为视觉条件回归骨架，`--random-init` 可端到端验证管线；任务2（网络恢复后）改用官方 prismatic/transformers 加载真实权重。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：HF openvla/openvla-oft / openvla-oft+。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/openvla_oft_7b
python export_openvla_oft_7b_onnx.py --output-dir ./openvla_oft_7b_onnx --device cpu
# 真实权重（任务2）：python export_openvla_oft_7b_onnx.py --checkpoint /path/to/oft.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | OFT state_dict（任务2） | 空（demo） |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |
| `--task-len` / `--vocab-size` | 语言 token 长度/词表 | `16`/`32000` |

```text
./openvla_oft_7b_onnx/
└── openvla_oft_7b_policy.onnx   # image+task_tokens -> action[B,horizon,action_dim]
```

---

## 3. ONNX 推理

```bash
python infer_openvla_oft_7b_onnx.py --model ./openvla_oft_7b_onnx/openvla_oft_7b_policy.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./openvla_oft_7b_onnx/openvla_oft_7b_policy.onnx \
  --outputFile=./openvla_oft_7b_onnx/openvla_oft_7b_policy --optimize=ascend_oriented \
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
./openvla_oft_7b_onnx/
├── openvla_oft_7b_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_openvla_oft_7b_mslite.py --model ./openvla_oft_7b_onnx/openvla_oft_7b_policy.mindir \
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
| MSLite 单步前向（mean） | 1.49 |
| MSLite 单步前向（p50） | 1.49 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

---

## 7. 常见问题

1. 真实 OFT 流式 action token：任务2 用 prismatic 加载 + prefill/decode 拆分（参照 internvl3_5_1b）。
2. 7B 在 44GB：用 `allow_fp32_to_fp16`；先 DEFAULT(fp16) 转换确认。
3. 与 OpenVLA 差异：OFT 头为 action-flow（更长 chunk、更高精度）。

---

## 8. 参考资源

- OpenVLA-OFT：https://github.com/openvla/openvla（OFT 变体）
- HF 权重：https://huggingface.co/openvla/openvla-oft
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- OpenVLA-OFT 上游代码许可证以其仓库为准（MIT）。
