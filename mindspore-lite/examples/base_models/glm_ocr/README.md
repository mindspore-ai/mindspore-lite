# GLM-OCR ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 `zai-org/GLM-OCR`（0.9B 多模态 OCR 模型）导出为 ONNX，转换为 MindSpore Lite MindIR，并在昇腾（Atlas 300I Duo）上完成端到端推理与精度对齐。

GLM-OCR 基于 GLM4v 编码器-解码器架构（CogViT 视觉编码器 + GLM 文本解码器），使用多模态 RoPE（mRoPE）。模型被拆分为三个子图导出：视觉编码器、LLM Prefill、LLM Decode。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux（昇腾环境，用于 MindIR 推理，需安装 MindSpore Lite 与 Ascend 驱动）

### 依赖版本（建议）

| 软件包            | 版本   |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.9.0（CPU 即可，仅用于导出） |
| transformers   | 5.9.0（原生支持 `glm_ocr`） |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| mindspore-lite | 2.9.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.9.0 transformers==5.9.0 onnx==1.19.1 onnxruntime==1.24.2
```

### 验证安装

```bash
python -c "import torch, transformers, onnx, onnxruntime, mindspore_lite; print('All dependencies installed successfully!')"
```

> 说明：GLM-OCR 在 transformers ≥ 5.0 中为原生模型（`glm_ocr`），无需 `trust_remote_code`。

---

## 2. 模型导出 ONNX

### 导出脚本说明

导出脚本将 GLM-OCR 拆分为三个 ONNX 子图：

1. **Vision 编码器**（`glm_ocr_vision.onnx`）：`pixel_values`（展平 patch）→ 图像嵌入（1024×1536）
2. **LLM Prefill**（`glm_ocr_llm_prefill.onnx`）：完整 prompt → `logits` + KV cache（PromptFlashAttention）
3. **LLM Decode**（`glm_ocr_llm_decode.onnx`）：单 token + 历史 KV cache → `logits` + 更新 KV cache（IncreFlashAttention + Scatter）

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/glm_ocr

python export_glm_ocr_onnx.py \
  --model-id ./GLM-OCR \
  --output-dir ./glm_ocr_onnx \
  --device cpu \
  --vision-image-size 896 \
  --kv-cache-len 2048 \
  --dtype fp32
```

> **推荐 fp32 导出**：GLM-OCR 的 LLM 在 fp16 下 decode 阶段 KV cache 会逐 token 累积精度漂移，导致若干步后输出偏移。使用 `--dtype fp32` 导出 + 转换时 `force_fp32`（见第 5 节）可彻底消除漂移，输出与 HuggingFace 完全一致。如需 fp16（更快但精度有损），把 `--dtype` 改为 `fp16` 并去掉转换配置里的 `force_fp32`。

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 本地权重目录或模型 ID | `./GLM-OCR` |
| `--output-dir` | 导出输出目录 | `./glm_ocr_onnx` |
| `--device` | 导出设备（cpu） | `cpu` |
| `--vision-image-size` | 固定图像尺寸（必须能被 patch_size=14 整除） | `896` |
| `--kv-cache-len` | Decode 阶段固定 KV cache 长度 | `2048` |
| `--dtype` | LLM 导出精度（fp16/fp32） | `fp16` |
| `--no-custom-op` | 不导出融合 CANN 算子为 Custom 节点 | `False` |

说明：

- 视觉模型采用**固定图像网格**：`vision_image_size=896`，patch_size=14 → 64×64 patch → 经 spatial_merge(2) 下采样后得到 1024 个图像 token。
- 推理时 `--image-size` 必须与导出的 `--vision-image-size` 一致。
- Prefill 的 seq 维度为动态，Decode 为固定 shape。

### 模型架构参数

| 参数 | 值 |
|------|------|
| text hidden_size | 1536 |
| text num_attention_heads | 16 |
| text num_hidden_layers | 16 |
| text num_key_value_heads | 8（GQA） |
| head_dim | 128 |
| text intermediate_size | 4608 |
| vocab_size | 59392 |
| vision hidden_size | 1024 |
| vision depth | 24 |
| vision image_size / patch_size | 336 / 14 |
| spatial_merge_size | 2 |
| mrope_section | [16, 24, 24] |

---

## 3. MindSpore Lite 转换

### 转换命令

```bash
cd ./mindspore-lite/examples/base_models/glm_ocr

CONV=./converter_lite   # 由 env.sh 设置，或使用绝对路径

# Vision
$CONV --fmk=ONNX \
  --modelFile=./glm_ocr_onnx/glm_ocr_vision.onnx \
  --outputFile=./glm_ocr_onnx/glm_ocr_vision \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/glm_ocr_vision.config

# Prefill
$CONV --fmk=ONNX \
  --modelFile=./glm_ocr_onnx/glm_ocr_llm_prefill.onnx \
  --outputFile=./glm_ocr_onnx/glm_ocr_llm_prefill \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/glm_ocr_llm_prefill.config

# Decode
$CONV --fmk=ONNX \
  --modelFile=./glm_ocr_onnx/glm_ocr_llm_decode.onnx \
  --outputFile=./glm_ocr_onnx/glm_ocr_llm_decode \
  --optimize=ascend_oriented --saveType=MINDIR \
  --configFile=./configs/glm_ocr_llm_decode.config
```

### config 文件说明

- `configs/glm_ocr_vision.config`：固定 4096 patch 输入，`force_fp16`。
- `configs/glm_ocr_llm_prefill.config`：`input_shape` 声明 4 个动态维度，`ge.dynamicDims` 配置 seq 档位（对应 vision_image_size=896 的 1024 个图像 token），`force_fp32`。
- `configs/glm_ocr_llm_decode.config`：固定 shape 单步 decode，`force_fp32`。

---

## 6. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/glm_ocr

python infer_glm_ocr_mslite.py \
  --vision-model ./glm_ocr_onnx/glm_ocr_vision_graph.mindir \
  --prefill-model ./glm_ocr_onnx/glm_ocr_llm_prefill_graph.mindir \
  --decode-model ./glm_ocr_onnx/glm_ocr_llm_decode_graph.mindir \
  --processor ./GLM-OCR \
  --image ./your_image.png \
  --prompt "Text Recognition:" \
  --max-new-tokens 512 \
  --image-size 896 \
  --kv-cache-len 2048 \
  --device ascend --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--vision-model` | Vision MindIR 路径 | 必填 |
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--processor` | tokenizer / processor 目录 | `./GLM-OCR` |
| `--image` | 图像 URL 或本地路径 | 必填 |
| `--prompt` | 文本提示词 | `Text Recognition:` |
| `--max-new-tokens` | 最大生成 token 数 | `256` |
| `--image-size` | 固定图像尺寸（须与导出一致） | `896` |
| `--kv-cache-len` | KV cache 长度（须与导出一致） | `2048` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |

GLM-OCR 支持两类提示词：文档解析（`Text Recognition:` / `Formula Recognition:` / `Table Recognition:`）与信息抽取（JSON schema）。

---

## 7. 性能数据

> Atlas 300I Duo 实测，`vision_image_size=896`，单图，fp32 LLM（`force_fp32`）。数值取自推理脚本端到端打印。

| 指标 | 300I Duo 耗时 |
|---|---|
| Vision 推理 (ms) | 1825 |
| Prefill (ms) | 1128 |
| Total Decode (ms) | 2239（48 token） |
| Avg Decode Step (ms) | 47.6 |
| Total (ms) | 5672 |
| Throughput (tok/s) | 8.46 |

### 输出效果

输入图像为含三行文字的测试图（"GLM-OCR MindSpore Lite" / "Hello OCR 2026" / "1234567890"）：

```text
Prompt: Text Recognition:
Response:
GLM-OCR MindSpore Lite

Hello OCR 2026

1234567890
```

---

## 9. 已知限制与常见问题

### 0) 必须用 fp32 导出 + force_fp32 转换（精度对齐的关键）

- **现象**：若用 fp16 导出（`--dtype fp16`）+ 默认 fp16 转换，prefill 正确（首 token "GL"），但 decode 阶段 KV cache 在 fp16 下逐 token 累积漂移，若干步后输出偏移（"GLMlooks..." 而非 "GLM-OCR"）。
- **根因**：fp16 权重 + fp16 KV cache 在自回归 decode 中精度累积。
- **解决方案（已验证，输出与 HF 完全一致）**：
  1. 导出用 `--dtype fp32`（fp32 权重）；
  2. LLM 转换 config 加 `ge.exec.precision_mode=force_fp32`（prefill/decode 都是）；
  3. 推理脚本自动检测并匹配 `image_embeds` / `past_key_values` / decode 输出 buffer 的 dtype（fp32）。
- **注意**：仅当 ONNX 是 fp32 权重时 force_fp32 才正确；fp16 权重 + force_fp32 会得到空串/错乱。Vision 始终用 `force_fp32`（避免 ViT 注意力 fp16 溢出）。

### 1) `apply_chat_template` 返回类型不一致

### 1) 转换时报 FLOAT16 不支持

部分算子初始值为 FLOAT16。导出时 Vision 使用 fp32 加载，LLM 使用 `--dtype fp16`；若仍报错，可改用 `--dtype fp32` 导出 LLM。

### 2) `image_embeds length mismatch`

推理 `--image-size` 与导出 `--vision-image-size` 不一致，导致图像 token 数与 MindIR 期望不符。确保两者一致。

### 3) `prompt_len >= kv_cache_len`

输入过长。增大导出/推理的 `--kv-cache-len`，或减小 `--image-size`。

### 4) 转换时 `ge.proto.ModelDef exceeded maximum protobuf size`

LLM 权重较大（fp16 ~1.8GB），`ascend_oriented` 转换会打印此信息，**不影响**最终产物（产出 `*_graph.mindir` + `*_variables/`）。

### 5) `Only support CustomAscend, but got ...`

MindIR 用 `--optimize=general` 转换会保留标准算子。请使用 `--optimize=ascend_oriented` 重新转换。

### 6) MTP（Multi-Token Prediction）层

GLM-OCR 文本配置含 `num_nextn_predict_layers=1`（MTP），仅用于训练，推理时不使用；加载时该层权重被忽略。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [GLM-OCR 模型页（ModelScope）](https://www.modelscope.cn/models/zai-org/GLM-OCR)
- [GLM-OCR GitHub](https://github.com/zai-org/GLM-OCR)
- [Transformers 文档](https://huggingface.co/docs/transformers)

---

## 11. 许可证

GLM-OCR 模型遵循 MIT License。本教程遵循相应依赖的许可证要求。
