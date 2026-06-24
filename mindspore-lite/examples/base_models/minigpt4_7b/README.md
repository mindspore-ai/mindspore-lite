# MiniGPT-4 视觉语言模型 ONNX 导出与 MindSpore Lite 推理部署教程

MiniGPT-4（Vision-CAIR）通过 Q-Former + 线性投影器对齐视觉编码器与 LLM（Vicuna），是 BLIP-2 谱系。本目录骨架为 vision encoder + Q-Former（learnable query + cross-attn）+ projector + 小 LLM → logits（单次前向）。

> ⚠️ **风险标注**：真实 MiniGPT-4（7B/13B）= ViT + Q-Former + Vicuna，自回归生成。本目录为单前向 logits 骨架，`--random-init` 可验证管线；任务2 改用 MiniGPT-4 官方包加载 + prefill/decode（复用 `blip2_opt_2_7b` 的 Q-Former 经验）。与已适配 VLM 重复，低优先级。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：核实 Vision-CAIR/MiniGPT-4 发布后从 ModelScope 下载。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/minigpt4_7b
python export_minigpt4_7b_onnx.py --output-dir ./minigpt4_7b_onnx --device cpu
# 真实权重（任务2）：python export_minigpt4_7b_onnx.py --checkpoint /path/to/minigpt4.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | MiniGPT-4 state_dict（任务2） | 空（demo） |
| `--num-query` | Q-Former query 数 | `32` |
| `--seq-len` / `--vocab-size` | 文本长度/词表 | `32`/`32000` |

```text
./minigpt4_7b_onnx/
└── minigpt4_7b.onnx   # image+input_ids -> logits
```

---

## 3. ONNX 推理

```bash
python infer_minigpt4_7b_onnx.py --model ./minigpt4_7b_onnx/minigpt4_7b.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./minigpt4_7b_onnx/minigpt4_7b.onnx \
  --outputFile=./minigpt4_7b_onnx/minigpt4_7b --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="image:1,3,224,224;input_ids:1,32"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./minigpt4_7b_onnx/
├── minigpt4_7b.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_minigpt4_7b_mslite.py --model ./minigpt4_7b_onnx/minigpt4_7b.mindir \
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
| MSLite 单步前向（mean） | 6.53 |
| MSLite 单步前向（p50） | 6.52 |
| 进程 RSS | 1.07 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

---

## 7. 常见问题

1. Q-Former 复用：可参照 `blip2_opt_2_7b` 的 Q-Former 转换经验。
2. 真实自回归：任务2 用官方包 + prefill/decode。
3. 与已适配 VLM 重复：低优先级。

---

## 8. 参考资源

- MiniGPT-4：https://github.com/Vision-CAIR/MiniGPT-4
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- MiniGPT-4 上游代码许可证以其仓库为准（BSD-3-Clause）。
