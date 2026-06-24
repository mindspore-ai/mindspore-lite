# InternVL3.5-1B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [OpenGVLab/InternVL3_5-1B](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-1B) 视觉语言模型按结构拆分导出为 ONNX，转换为 MindSpore Lite MindIR，并在 **Ascend Atlas 300I Duo** 上完成图文推理与精度对齐。

InternVL3.5 = InternViT 视觉编码器 + pixel-shuffle + MLP 投影（`extract_feature`）+ 自回归 LLM。InternVL3.5 的 LLM 就是原版 `Qwen3ForCausalLM`（`model.language_model`），因此 LLM 导出直接复用已验证的 `qwen3_8b` 模板（手动驱动 Qwen3 各层：融合 QKV 线性 → QK-norm → rotary → 带**显式因果+填充 mask** 的注意力 → SwiGLU MLP）。固定 shape 部署拆分为三子模型：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `internvl_vision`（InternViT+mlp1） | pixel_values[1,3,448,448] | image_embeds[1,256,1024] |
| `internvl_llm_prefill` | inputs_embeds[1,320,1024] + attention_mask[1,320] + position_ids[1,320] | logits + present KV[28,1,8,1024,128] |
| `internvl_llm_decode` | inputs_embeds[1,1,1024] + attention_mask[1,1024] + position_ids[1,1] + past KV | logits + updated KV |

多模态融合在推理侧完成：embed 输入 token，把 `<IMG_CONTEXT>`（id=151671）位置替换为视觉 embed。1B 实测配置：hidden=1024，28 层，16 注意力头，8 KV 头，head_dim=128，vocab=151936。**同一套脚本适用于 2B/4B/8B 及 Flash 变体**，仅模型 id 与固定 shape（从 checkpoint 读取）不同。

> **关键经验**：LLM 注意力导出为**标准 ONNX 算子**（MatMul+Softmax+显式 mask），**不要**用 CANN `PromptFlashAttention` Custom 算子。在本仓库的 CANN/转换器版本下，`PromptFlashAttention(sparse_mode=0)` + 显式 attention mask **不会应用 mask**（退化为全双向注意力 → 输出乱码）；标准算子则转换+运行正确（与 HF 的 prefill 首 token logits 余弦相似度 = 1.0）。详见第 2 节「导出注意事项」。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch（导出/对齐用） | 2.12.1 |
| transformers（导出/对齐用） | 4.51.3（须 `trust_remote_code`；主环境 5.9.0 与 InternVL 的 `trust_remote_code` 不兼容） |
| onnx / onnxruntime | 1.19.1 / 1.24.2 |
| mindspore-lite（推理） | 2.10.0 |
| CANN | 8.5.0 |

> 推荐用独立 conda 环境（如 `mslite_export`，torch 2.12 + transformers 4.51.3，无 torch_npu）做导出与 HF 对齐；推理在带 `mindspore-lite` 的环境（如 `py3.11`）执行。

```bash
# 导出/对齐环境
conda create -n mslite_export python=3.11 -y && conda activate mslite_export
pip install torch==2.12.1 transformers==4.51.3 onnx==1.19.1 onnxruntime==1.24.2 modelscope
# 推理环境（已有 mindspore-lite 2.10.0）
pip install transformers mindspore-lite  # py3.11
```

### 获取模型权重

```bash
python -c "from modelscope import snapshot_download as s; print(s('OpenGVLab/InternVL3_5-1B', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/OpenGVLab/InternVL3_5-1B ./InternVL3_5-1B
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/internvl3_5_1b
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python export_internvl3_5_1b_onnx.py \
  --model-id ./InternVL3_5-1B --output-dir ./internvl3_5_1b_onnx \
  --image-size 448 --num-img-tokens 256 --max-text-len 64 --dtype float32
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-id` | 本地权重目录或 ModelScope id | `./InternVL3_5-1B` |
| `--output-dir` | ONNX / embed 输出目录 | `./internvl3_5_1b_onnx` |
| `--parts` | 导出哪些子模型 | `vision,prefill,decode,embeds` |
| `--image-size` | 视觉输入边长 | `448` |
| `--num-img-tokens` | 图像 token 数（pixel-shuffle 后） | `256` |
| `--max-text-len` | prefill 文本最大长度（seq=图像+文本=320） | `64` |
| `--dtype` | 导出精度（`float32`/`float16`） | `float32` |

### 产出文件

```
internvl3_5_1b_onnx/
├── internvl_vision.onnx            # InternViT+mlp1（含权重，~1.2GB）
├── internvl_llm_prefill.onnx       # LLM prefill（权重外部化）
├── internvl_llm_decode.onnx        # LLM decode（权重外部化）
├── *.data                          # LLM 外部权重文件
└── embed_weights.npy               # LLM token embedding [151936,1024]，供推理侧多模态融合
```

### 导出注意事项（实际踩坑点）

1. **必须用 `trust_remote_code=True` 加载**，且 transformers 版本须兼容 InternVL3.5（4.51.x 验证通过；5.9.0 报 `all_tied_weights_keys`）。
2. **注意力用标准 ONNX 算子，不用 CANN `PromptFlashAttention` Custom 算子**：本环境 `sparse_mode=0` + 显式 mask 不应用 mask → 全双向注意力 → 输出乱码；标准 MatMul/Softmax + 显式因果 mask 转换运行正确（cos=1.0 vs HF）。
3. **legacy exporter + opset 18 + `do_constant_folding=False`**，`dynamo=False`，避免 `torch.export` 控制流问题。
4. LLM 子模型接收 `inputs_embeds`（多模态已融合），embedding 矩阵单独导出为 `embed_weights.npy` 供 torch-free 推理做 token→embed 查表 + `<IMG_CONTEXT>` 位替换。
5. KV cache 导出为固定 `KV_CACHE_LEN=1024`，prefill 输出经 pad 到 1024，decode 用 scatter 写入指定位置。

```log
[export] vision encoder ...
[export] saved internvl3_5_1b_onnx/internvl_vision.onnx
[export] llm prefill ...
[export] prefill: hidden=1024 num_layers=28 num_kv_heads=8 head_dim=128 seq=320 kv_len=1024
[export] llm decode ...
[export] decode: num_layers=28 num_kv_heads=8 head_dim=128 kv_len=1024
[export] llm embed weights ...
[export] embed_weights (151936, 1024) -> internvl3_5_1b_onnx/embed_weights.npy
```

---

## 3. ONNX 推理（正确性验证）

ONNX 图层面的正确性通过 **PyTorch eager 包装器 vs HF** 验证：`_InternVLPrefill` wrapper（与导出同一套算子）在 eager 下跑同一份融合 embeds，prefill 首 token logits 与 HF `language_model` 余弦相似度 = **1.00001**，argmax 一致（都为 785 "The"）。该 eager 等价验证即覆盖 ONNX 导出的算子正确性，后续以 MSLite 端到端 + HF 对齐作为最终验证（第 5、7 节），故不单独提供 ORT 推理脚本。

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

```bash
source /home/yf/env.sh
CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$CONV --fmk=ONNX --modelFile=./internvl3_5_1b_onnx/internvl_vision.onnx      --outputFile=./internvl3_5_1b_onnx/internvl_vision      --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_vision.config
$CONV --fmk=ONNX --modelFile=./internvl3_5_1b_onnx/internvl_llm_prefill.onnx --outputFile=./internvl3_5_1b_onnx/internvl_llm_prefill --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_llm_prefill.config
$CONV --fmk=ONNX --modelFile=./internvl3_5_1b_onnx/internvl_llm_decode.onnx  --outputFile=./internvl3_5_1b_onnx/internvl_llm_decode  --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_llm_decode.config
```

### 配置文件（`configs/`）

vision 用 `force_fp16`；prefill/decode 用 `force_fp32`（本模型精度优先；更大模型可改 `force_fp16` 省显存）。三者均设 `input_format="ND"`、固定 `input_shape`、`plugin_custom_ops=All`。例如 prefill：

```ini
[acl_build_options]
input_format="ND"
input_shape="inputs_embeds:1,320,1024;attention_mask:1,320;position_ids:1,320"
[acl_init_options]
ge.exec.precision_mode=force_fp32
[ascend_context]
plugin_custom_ops=All
```

### 产出说明

`internvl_vision.mindir`（~628MB，含权重）；prefill/decode 因 >2GB 输出 `_graph.mindir` + `_variables/data_0`（外部权重，推理加载 `_graph.mindir`）。转换日志 `CONVERT RESULT SUCCESS:0`。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python infer_internvl3_5_1b_mslite.py \
  --mindir-dir ./internvl3_5_1b_onnx --model-dir ./InternVL3_5-1B \
  --image ./test.jpg --prompt "Describe this image in detail." --max-new-tokens 128
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--mindir-dir` | MindIR 目录（含三个子模型 + embed_weights.npy） | 必填 |
| `--model-dir` | 权重目录（取 tokenizer） | 必填 |
| `--image` | 输入图像 | 必填 |
| `--prompt` | 提问文本 | `Describe this image in detail.` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--vision-device` / `--llm-device` | 视觉/LLM 所在 Ascend 设备 id | `1` / `0` |

### 执行日志（真实输出，test.jpg = 蓝底+黄圆+绿矩形+红三角）

```log
[output] The image features a simple, minimalistic design with a light blue background.
There are two primary shapes:
1. Yellow Circle: Positioned at the top right.
2. Green Rectangle: Located at the bottom left.
3. Red Triangle: Positioned at the bottom right.
The overall composition is clean and uses basic geometric shapes with a limited color palette.

--- Performance ---
  Vision encode:   88.42 ms
  LLM prefill:     433.92 ms (seq=320)
  LLM decode:      19442.33 ms (94 steps)
  End-to-end:      19964.67 ms
  Avg decode step: 206.83 ms
  Throughput:      4.83 tok/s (decode)
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0，fp32。

| 指标 | 耗时 |
| --- | ---: |
| Vision 编码（InternViT, 448×448） | 88 ms |
| LLM prefill（seq=320） | 434 ms |
| LLM decode（单步平均 / 94 步） | 207 ms / 步 |
| **端到端（128 tok 上限）** | **~20.0 s** |
| **decode 吞吐** | **4.83 tok/s** |

> 注意力采用标准 ONNX 算子（非 CANN flash op），decode 单步偏慢；如需更高吞吐，可在支持 mask 的 CANN 版本上切回 `PromptFlashAttention` Custom 算子。

---

## 7. 精度对齐

```bash
# 导出环境（transformers 4.51.x）：生成 HF 参考
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python align_internvl3_5_1b.py \
  --model-dir ./InternVL3_5-1B --image ./test.jpg --dump-dir ./align_dump --max-new-tokens 32
# 推理环境（mindspore-lite）：MSLite prefill 首 token 余弦对齐
python align_internvl3_5_1b.py --mindir-dir ./internvl3_5_1b_onnx --dump-dir ./align_dump --check-mslite
```

| 指标 | 结果 |
| --- | --- |
| prefill 首 token logits 余弦（MSLite vs HF） | **1.00001** |
| 首 token argmax | MSLite=785 "The" / HF=785 "The"（一致） |
| 生成文本 | MSLite 与 HF 均以 "The image features a simple, minimalistic design with a light blue background…" 开头，正确描述黄圆/绿矩形/红三角 |

---

## 8. 常见问题

1. **`PromptFlashAttention` 输出乱码** —— 本 CANN/转换器版本下 `sparse_mode=0` + 显式 mask 不应用 mask（全双向）。本教程改用标准 ONNX 注意力算子（MatMul/Softmax + 显式因果 mask）。
2. **`Not support dynamic input`（prefill/decode predict 报错）** —— 注意力 mask 的 batch 维缺失：必须传 `[1,seq]` 而非 `[seq]`；同时 prefill/decode 含 Custom 风格算子时 shape 推断失败会触发 Resize，输入 shape 需与导出完全一致。
3. **InternVL 加载报错** —— 须 `trust_remote_code=True`；transformers 5.9.0 与 InternVL3.5 不兼容，用 4.51.x。
4. **`<IMG_CONTEXT>` 替换** —— 图像 token id=151671；推理侧把该 id 位置的 embed 替换为 vision 输出的 256 个视觉 embed。
5. **多模态融合在推理侧** —— LLM 子模型接收 `inputs_embeds`；token→embed 查表用导出的 `embed_weights.npy`（torch-free）。

---

## 9. 参考资源与许可证

- 上游：<https://github.com/OpenGVLab/InternVL>、ModelScope `OpenGVLab/InternVL3_5-1B`
- MindSpore Lite：<https://www.mindspore.cn/lite>
- 脚本遵循 MindSpore Lite 仓库许可证；上游模型/代码许可证以其仓库为准。
