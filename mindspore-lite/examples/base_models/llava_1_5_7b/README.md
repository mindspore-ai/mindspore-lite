# LLaVA-1.5 视觉语言模型 ONNX 导出与 MindSpore Lite 推理部署教程

LLaVA-1.5（CLIP-ViT 视觉编码器 + MLP 投影器 + Vicuna LLM）是经典开源 VLM。本目录骨架为 vision encoder + projector + 小 LLM → logits（单次前向）。

> ⚠️ **风险标注**：真实 LLaVA-1.5-7B = CLIP-ViT-L + projector + Vicuna-7B，自回归生成（prefill/decode）。本目录为单前向 logits 骨架，`--random-init` 可验证管线；任务2（网络恢复后）改用 `transformers.LlavaForConditionalGeneration` 加载真实权重 + prefill/decode 拆分（参照 `internvl3_5_1b`）。仓库已适配 InternVL/Qwen-VL/BLIP，LLaVA 与之重复，低优先级。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：modelscope.cn/llava-hf/llava-1.5-7b-hf。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/llava_1_5_7b
python export_llava_1_5_7b_onnx.py --output-dir ./llava_1_5_7b_onnx --device cpu
# 真实权重（任务2）：python export_llava_1_5_7b_onnx.py --checkpoint /path/to/llava.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | LLaVA state_dict（任务2 用 transformers） | 空（demo） |
| `--seq-len` / `--vocab-size` | 文本长度/词表 | `32`/`32000` |
| `--dim` / `--depth` | 骨架 LLM 配置（demo 小） | `384`/`4` |

```text
./llava_1_5_7b_onnx/
└── llava_1_5_7b.onnx   # image+input_ids -> logits
```

---

## 3. ONNX 推理

```bash
python infer_llava_1_5_7b_onnx.py --model ./llava_1_5_7b_onnx/llava_1_5_7b.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./llava_1_5_7b_onnx/llava_1_5_7b.onnx \
  --outputFile=./llava_1_5_7b_onnx/llava_1_5_7b --optimize=ascend_oriented \
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
./llava_1_5_7b_onnx/
├── llava_1_5_7b.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_llava_1_5_7b_mslite.py --model ./llava_1_5_7b_onnx/llava_1_5_7b.mindir \
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
| MSLite 单步前向（mean） | 8.06 |
| MSLite 单步前向（p50） | 8.07 |
| 进程 RSS | 1.15 |
| 精度对齐（PyTorch↔MSLite） | cos=1.000000 |

---

## 7. 常见问题

1. 真实 LLaVA 自回归：任务2 用 transformers + prefill/decode（参照 internvl3_5_1b）。
2. 7B 在 44GB：`allow_fp32_to_fp16`；先 DEFAULT(fp16) 转换。
3. 与已适配 VLM 重复：低优先级，复用 InternVL/Qwen-VL 栈。

---

## 8. 参考资源

- LLaVA：https://github.com/haotian-liu/LLaVA
- ModelScope 权重：https://modelscope.cn/llava-hf/llava-1.5-7b-hf
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- LLaVA 上游代码许可证以其仓库为准（Apache-2.0）。
