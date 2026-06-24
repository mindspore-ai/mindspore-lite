# CogACT 视觉-语言-动作策略 ONNX 导出与 MindSpore Lite 推理部署教程

CogACT（Microsoft）基于 Prismatic VLM，用"以动作为中心"的动作头（FAST tokenization + 可选扩散/混合头）替代 OpenVLA 的动作头，提升操作精度。本目录骨架与 `openvla_7b` 同构（image + 语言 task token → 动作 chunk，单前向），用于跑通迁移管线。

> ⚠️ **风险标注**：真实 CogACT（~8B）= Prismatic VLM + 动作中心头，需 prefill/decode 拆分（参照 `internvl3_5_1b`）。本目录为视觉条件回归骨架，`--random-init` 可端到端验证管线；任务2（网络恢复后）改用 `microsoft/cogact` 官方包加载真实权重。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：github.com/microsoft/cogact。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/cogact_8b
python export_cogact_8b_onnx.py --output-dir ./cogact_8b_onnx --device cpu
# 真实权重（任务2）：python export_cogact_8b_onnx.py --checkpoint /path/to/cogact.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | CogACT state_dict（任务2） | 空（demo） |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |
| `--task-len` / `--vocab-size` | 语言 token 长度/词表 | `16`/`32000` |

```text
./cogact_8b_onnx/
└── cogact_8b_policy.onnx   # image+task_tokens -> action[B,horizon,action_dim]
```

---

## 3. ONNX 推理

```bash
python infer_cogact_8b_onnx.py --model ./cogact_8b_onnx/cogact_8b_policy.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./cogact_8b_onnx/cogact_8b_policy.onnx \
  --outputFile=./cogact_8b_onnx/cogact_8b_policy --optimize=ascend_oriented \
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
./cogact_8b_onnx/
├── cogact_8b_policy.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_cogact_8b_mslite.py --model ./cogact_8b_onnx/cogact_8b_policy.mindir \
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
| 精度对齐（PyTorch↔MSLite） | cos=1.000000 |

---

## 7. 常见问题

1. 真实 CogACT 动作中心头（FAST/扩散）：任务2 用官方包加载 + prefill/decode。
2. 8B 在 44GB：`allow_fp32_to_fp16`；先 DEFAULT(fp16) 转换确认。
3. 与 OpenVLA 差异：动作头更强（FAST tokenization + 可选扩散），精度更高。

---

## 8. 参考资源

- CogACT：https://github.com/microsoft/cogact
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- CogACT 上游代码许可证以其仓库为准（MIT）。
