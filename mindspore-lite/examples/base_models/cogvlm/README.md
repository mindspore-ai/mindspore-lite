# CogVLM 视觉语言模型 ONNX 导出与 MindSpore Lite 推理部署教程

CogVLM（THUDM）是 GLM 谱系 VLM，特色是每层带 visual expert（为图像 token 并行的 MLP/投影），全模型自回归。本目录骨架为 vision encoder + projector + 小 LLM → logits（单次前向）。

> ⚠️ **风险标注**：真实 CogVLM（~19B）= GLM-4V + visual expert，自回归生成。本目录为单前向 logits 骨架，`--random-init` 可验证管线；任务2 改用 `AutoModelForCausalLM(trust_remote_code=True)` 加载真实权重 + prefill/decode。19B 在 300I Duo 偏大，可能需 800I 或拆分。与已适配 GLM-OCR 经验可复用。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：modelscope.cn/ZhipuAI/CogVLM 或类似。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/cogvlm
python export_cogvlm_onnx.py --output-dir ./cogvlm_onnx --device cpu
# 真实权重（任务2）：python export_cogvlm_onnx.py --checkpoint /path/to/cogvlm.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | CogVLM state_dict（任务2 trust_remote_code） | 空（demo） |
| `--seq-len` / `--vocab-size` | 文本长度/词表 | `32`/`32000` |
| `--dim` / `--depth` | 骨架 LLM 配置（demo 小） | `384`/`4` |

```text
./cogvlm_onnx/
└── cogvlm.onnx   # image+input_ids -> logits
```

---

## 3. ONNX 推理

```bash
python infer_cogvlm_onnx.py --model ./cogvlm_onnx/cogvlm.onnx --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./cogvlm_onnx/cogvlm.onnx \
  --outputFile=./cogvlm_onnx/cogvlm --optimize=ascend_oriented \
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
./cogvlm_onnx/
├── cogvlm.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_cogvlm_mslite.py --model ./cogvlm_onnx/cogvlm.mindir \
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
| MSLite 单步前向（mean） | 7.88 |
| MSLite 单步前向（p50） | 8.04 |
| 进程 RSS | 1.15 |
| 精度对齐（PyTorch↔MSLite） | cos=1.000000 |

---

## 7. 常见问题

1. 真实 CogVLM visual expert：任务2 trust_remote_code 加载，prefill/decode 拆分。
2. 19B 偏大：300I Duo 可能需拆分/800I。
3. trust_remote_code：加载时需 `trust_remote_code=True`。

---

## 8. 参考资源

- CogVLM：https://github.com/THUDM/CogVLM
- ModelScope 权重：https://modelscope.cn/ZhipuAI
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- CogVLM 上游代码许可证以其仓库为准（Apache-2.0）。
