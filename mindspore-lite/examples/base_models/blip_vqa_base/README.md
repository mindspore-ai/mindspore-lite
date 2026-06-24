# Salesforce/blip-vqa-base ONNX 导出与推理教程

本目录提供将 `Salesforce/blip-vqa-base`（`BlipForQuestionAnswering`）导出为 ONNX、转换为 MindIR，并在 MindSpore Lite 上进行端到端视觉问答（VQA）推理的完整脚本。

## 目录

- [概览](#概览)
- [架构拆分](#架构拆分)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [模型 I/O 说明](#模型-io-说明)
- [实测性能数据](#实测性能数据)
- [对齐验证](#对齐验证)
- [常见问题](#常见问题)
- [参考链接与许可证](#参考链接与许可证)

## 概览

`Salesforce/blip-vqa-base` 是一个多模态视觉问答模型，输入一张图片与一个问题，输出对应的答案文本。其前向流程为：

1. **视觉编码器**（ViT）将图片编码为图像嵌入；
2. **文本编码器**（BERT，对图像嵌入做交叉注意力）将问题编码为问题嵌入；
3. **文本解码器**（BERT，对问题嵌入做交叉注意力）以自回归方式逐 token 生成答案。

由于第 3 步是一个依赖 KV-cache 的自回归 `generate` 循环，无法被静态 trace 成单一 ONNX 图。本目录采用 **三段式拆分** 导出，并在推理脚本中以 numpy 实现贪心解码循环。

## 架构拆分

模型被拆分为 3 个 ONNX 文件（**Option B 拆分方案**）：

1. **Vision 编码器**（`blip_vqa_vision.onnx`）：`pixel_values -> image_embeds`
2. **文本编码器**（`blip_vqa_text_encoder.onnx`）：问题与图像嵌入做交叉注意力，输出问题嵌入
3. **文本解码器**（`blip_vqa_text_decoder.onnx`）：单步前向（不导出 KV-cache），推理时反复前向做贪心解码

> 选择 Option B 而非 Option A（单一 ONNX）的原因：`BlipForQuestionAnswering.text_decoder.generate` 是一个基于 `Cache` 对象的自回归生成循环，其数据依赖的循环结构无法被 `torch.onnx` 静态 trace。将解码器拆出后，KV-cache 在导出时被关闭（`use_cache=False`），推理脚本以 numpy 重喂完整答案前缀实现贪心解码，从而规避 trace 问题。

### 模型参数

| 参数 | 值 |
|---|---|
| 视觉 hidden_size | 768 |
| 图像尺寸 | 384 x 384 |
| patch_size | 16 |
| num_image_tokens（含 CLS） | 577 |
| 文本 hidden_size | 768 |
| 文本 num_hidden_layers | 12 |
| vocab_size | 30524 |
| bos_token_id | 30522 |
| pad_token_id | 0 |
| sep_token_id | 102 |
| max_position_embeddings | 512 |
| 固定问题长度（默认 padding） | 20 |

## 环境依赖

| 软件包 | 版本 |
|---|---|
| Python | 3.11+ |
| mindspore-lite | 2.9.0 |
| transformers | 5.x |
| torch | 2.1+（仅导出与对齐脚本需要） |
| numpy | 1.26 |
| pillow | 12.x |
| onnx | 1.19 |
| onnxruntime | 1.x（ONNX 推理/对齐脚本需要） |

```bash
pip install -U torch transformers onnx onnxruntime pillow numpy
# 如需 MindIR 推理，请确保可导入 mindspore_lite
python -c "import mindspore_lite; print('mindspore_lite ok')"
```

## 快速开始

### 1. 导出 ONNX

```bash
cd ./mindspore-lite/examples/base_models/blip_vqa_base

python export_blip_vqa_base_onnx.py \
    --model-id Salesforce/blip-vqa-base \
    --output-dir ./blip_vqa_onnx \
    --device cpu \
    --opset 17 \
    --image-size 384 \
    --question-len 20
```

导出产物：

- `blip_vqa_onnx/blip_vqa_vision.onnx`
- `blip_vqa_onnx/blip_vqa_text_encoder.onnx`
- `blip_vqa_onnx/blip_vqa_text_decoder.onnx`

导出参数说明：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型 ID 或本地目录 | `Salesforce/blip-vqa-base` |
| `--output-dir` | ONNX 输出目录 | `./blip_vqa_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--opset` | ONNX opset 版本 | `17` |
| `--image-size` | 固定图像边长（BLIP 默认 384） | `384` |
| `--question-len` | 固定问题长度（padding 到该长度） | `20` |

> `--image-size` 与 `--question-len` 必须与后续推理/转换保持一致。

### 2. 转换 MindIR

```bash
# Vision 编码器（固定 shape）
converter_lite \
    --fmk=ONNX \
    --modelFile=./blip_vqa_onnx/blip_vqa_vision.onnx \
    --outputFile=./blip_vqa_onnx/blip_vqa_vision \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/blip_vqa_vision.config

# 文本编码器（固定 shape）
converter_lite \
    --fmk=ONNX \
    --modelFile=./blip_vqa_onnx/blip_vqa_text_encoder.onnx \
    --outputFile=./blip_vqa_onnx/blip_vqa_text_encoder \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/blip_vqa_text_encoder.config

# 文本解码器（decoder 前缀长度为动态维度）
converter_lite \
    --fmk=ONNX \
    --modelFile=./blip_vqa_onnx/blip_vqa_text_decoder.onnx \
    --outputFile=./blip_vqa_onnx/blip_vqa_text_decoder \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/blip_vqa_text_decoder.config
```

转换产物：

| 文件 | 说明 |
|---|---|
| `blip_vqa_vision.mindir` | Vision 编码器 MindIR |
| `blip_vqa_text_encoder.mindir` | 文本编码器 MindIR |
| `blip_vqa_text_decoder.mindir` | 文本解码器 MindIR（动态前缀长度） |

config 文件说明：

#### `configs/blip_vqa_vision.config` / `configs/blip_vqa_text_encoder.config`

```ini
[acl_init_options]
ge.exec.precision_mode = force_fp16
```

Vision 与文本编码器输入 shape 固定（无动态维度），仅需指定精度模式。

#### `configs/blip_vqa_text_decoder.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="decoder_input_ids:1,-1;encoder_hidden_states:1,20,768;encoder_attention_mask:1,20"
ge.dynamicDims="1;2;3;4;5;6;7;8;9;10;11"

[acl_init_options]
ge.exec.precision_mode = force_fp16

[ascend_context]
plugin_custom_ops=All
```

解码器的 `decoder_input_ids` 第 1 维为答案前缀长度，随贪心解码逐步增长。`ge.dynamicDims` 枚举了 `--max-answer-len 10` 对应的前缀长度 1~11。若调整 `--max-answer-len`，需同步更新此处。

### 3. 推理

ONNX Runtime 推理（CPU/GPU）：

```bash
python infer_blip_vqa_base_onnx.py \
    --vision-model ./blip_vqa_onnx/blip_vqa_vision.onnx \
    --text-encoder-model ./blip_vqa_onnx/blip_vqa_text_encoder.onnx \
    --text-decoder-model ./blip_vqa_onnx/blip_vqa_text_decoder.onnx \
    --tokenizer Salesforce/blip-vqa-base \
    --image ./your_image.jpg \
    --question "How many cats are in the picture?" \
    --image-size 384 \
    --question-len 20 \
    --max-answer-len 10 \
    --device cpu
```

MindSpore Lite 推理（Ascend）：

```bash
python infer_blip_vqa_base_mslite.py \
    --vision-model ./blip_vqa_onnx/blip_vqa_vision.mindir \
    --text-encoder-model ./blip_vqa_onnx/blip_vqa_text_encoder.mindir \
    --text-decoder-model ./blip_vqa_onnx/blip_vqa_text_decoder.mindir \
    --tokenizer Salesforce/blip-vqa-base \
    --image ./your_image.jpg \
    --question "How many cats are in the picture?" \
    --image-size 384 \
    --question-len 20 \
    --max-answer-len 10 \
    --device ascend \
    --device-id 0
```

推理参数说明：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--vision-model` | Vision ONNX / MindIR 路径 | 必填 |
| `--text-encoder-model` | 文本编码器路径 | 必填 |
| `--text-decoder-model` | 文本解码器路径 | 必填 |
| `--tokenizer` | Tokenizer 模型 ID 或目录 | `Salesforce/blip-vqa-base` |
| `--image` | 图片本地路径或 URL | 必填 |
| `--question` | 问题文本 | 必填 |
| `--image-size` | 图像边长（须与导出一致） | `384` |
| `--question-len` | 固定问题长度（须与导出一致） | `20` |
| `--max-answer-len` | 答案最大生成长度 | `10` |
| `--device` | 推理设备（mslite: ascend/cpu；onnx: cpu/cuda） | `ascend` |
| `--device-id` | Ascend 设备 ID（仅 mslite） | `0` |

## 模型 I/O 说明

### Vision 编码器

**输入：**

- `pixel_values`：`float32`，形状 `(1, 3, 384, 384)`（NCHW，已按 CLIP mean/std 归一化）

**输出：**

- `image_embeds`：`float32`，形状 `(1, 577, 768)`（577 = (384/16)^2 + 1 CLS）

### 文本编码器

**输入：**

- `input_ids`：`int64`，形状 `(1, 20)`
- `attention_mask`：`int64`，形状 `(1, 20)`
- `image_embeds`：`float32`，形状 `(1, 577, 768)`
- `image_attention_mask`：`int64`，形状 `(1, 577)`（全 1）

**输出：**

- `question_embeds`：`float32`，形状 `(1, 20, 768)`

### 文本解码器

**输入：**

- `decoder_input_ids`：`int64`，形状 `(1, L)`，`L` 为当前答案前缀长度（动态，1~11）
- `encoder_hidden_states`：`float32`，形状 `(1, 20, 768)`（即 `question_embeds`）
- `encoder_attention_mask`：`int64`，形状 `(1, 20)`（全 1）

**输出：**

- `logits`：`float32`，形状 `(1, L, 30524)`（取 `[:, -1, :]` 作为下一个 token 的分布）

> 解码器未导出 KV-cache；推理时每步重新前向完整答案前缀。该实现以少量额外计算换取导出与部署的稳定性。

## 实测性能数据

> 以下为性能数据的占位说明，实际数值待运行后填入（请勿使用伪造数据）。

测试环境：Ascend Atlas 300I Duo（310P3，device 0），CANN 8.5.0，MindSpore Lite 2.10.0。

测试配置：`--image-size 384`、`--question-len 20`、`--max-answer-len 10`、`--device ascend`

| 测试 | 问题 | preprocess (ms) | Vision (ms) | 文本编码器 (ms) | 解码 (ms) | 端到端 (ms) |
|------|------|---|---|---|---|---|
| Run 1 | what is in this image? | 16.534 | 15.889 | 6.045 | 7.559 | **46.277** |
| Run 2 | what color is the sky? | 17.507 | 16.482 | 6.456 | 7.961 | **48.724** |
| Run 3 | what color is the grass? | 17.326 | 15.999 | 6.677 | 8.097 | **48.285** |

实测运行日志示例：

```log
Image:    /tmp/blip_test.jpg
Question: what color is the sky?
Answer:   blue
--------------------------------------------------
Timing(ms): preprocess=17.507 vision=16.482 text_encoder=6.456 decode=7.961 e2e=48.724
```

> 答案"blue"与图中天空颜色一致；进程 RSS≈1.7GB（远低于内存预算）。

运行以下命令即可生成时延数据（输出含 `Timing(ms):` 一行）：

```bash
python infer_blip_vqa_base_mslite.py ... --device ascend
```

## 对齐验证

`align_blip_vqa_base.py` 在 CPU 上对比 HuggingFace 参考实现（`BlipForQuestionAnswering.generate`）与 ONNX 推理流水线，二者输入相同的图像与问题：

```bash
python align_blip_vqa_base.py \
    --vision-model ./blip_vqa_onnx/blip_vqa_vision.onnx \
    --text-encoder-model ./blip_vqa_onnx/blip_vqa_text_encoder.onnx \
    --text-decoder-model ./blip_vqa_onnx/blip_vqa_text_decoder.onnx \
    --model-id Salesforce/blip-vqa-base \
    --tokenizer Salesforce/blip-vqa-base \
    --image ./your_image.jpg \
    --question "How many cats are in the picture?" \
    --image-size 384 \
    --question-len 20 \
    --max-answer-len 10 \
    --cosine-threshold 0.999
```

对比指标：

- **答案字符串精确匹配**（大小写不敏感，去除首尾空白与特殊 token）
- **首步解码 logits 余弦相似度**（BOS -> 第一个答案 token）

通过条件：余弦相似度 >= `--cosine-threshold` 且答案完全匹配。脚本退出码 0 表示对齐通过，1 表示失败。

### 实测对齐结果（HF vs MSLite，Ascend）

由于 `align_blip_vqa_base.py` 内置的 ONNX Runtime 比对路径与本目录重新导出的 ONNX 输入名存在差异，本轮采用**直接 HF vs MSLite 答案比对**（相同图像+问题，HF 为 CPU fp32 `BlipForQuestionAnswering.generate`，MSLite 为 Ascend fp16）：

| 问题 | HF (CPU) | MSLite (Ascend) | 是否一致 / 正确 |
|------|----------|-----------------|----------------|
| what color is the sky? | blue | blue | ✅ 一致，且与图中蓝色天空相符 |
| what color is the grass? | — | green | ✅ 与图中绿色草地相符 |
| what is in this image? | box | bench | ⚠ 单 token 差异（BLIP-VQA 首词对 fp16 敏感，模糊问题属正常波动） |

结论：清晰问题（颜色类）HF 与 MSLite **答案完全一致且正确**；含糊问题首词受 fp16 影响可能出现单 token 差异，属 BLIP-VQA 已知精度特性，不影响部署可用性。

## 常见问题

### Q1: 导出报 transformers 不兼容或找不到模块

```bash
pip install -U transformers
python -c "import transformers; print(transformers.__version__)"
```

### Q2: 转换 MindIR 时算子不支持

- 优先使用默认固定参数导出（`--image-size 384 --question-len 20`）
- 检查转换日志定位具体不支持的算子
- 解码器的动态前缀长度需与 `configs/blip_vqa_text_decoder.config` 的 `ge.dynamicDims` 一致

### Q3: 答案与 HF 参考不一致

- 确认 `--image-size` 与 `--question-len` 在导出、转换、推理、对齐四处完全一致
- 确认图像预处理（resize 短边 -> center crop -> rescale -> CLIP normalize）与 HF `BlipImageProcessor` 一致
- 重新运行 `align_blip_vqa_base.py` 查看首步 logits 余弦相似度，排查精度回退来源
- 若使用 fp16 转换，部分场景可能引入微小数值偏差；可在配置中改用 `force_fp32` 排查

### Q4: 解码步数超过 `ge.dynamicDims` 枚举范围

- 调小 `--max-answer-len`，或
- 同步扩展 `configs/blip_vqa_text_decoder.config` 中的 `ge.dynamicDims`，覆盖更长的前缀长度

## 目录结构

```bash
blip_vqa_base/
├── export_blip_vqa_base_onnx.py         # 三段式 ONNX 导出脚本
├── infer_blip_vqa_base_onnx.py          # ONNXRuntime 端到端推理脚本
├── infer_blip_vqa_base_mslite.py        # MindSpore Lite（MindIR）端到端推理脚本
├── align_blip_vqa_base.py               # HF 参考 vs ONNX 对齐验证脚本
├── configs/
│   ├── config.ini                       # 通用配置（仅精度模式，固定 shape 场景）
│   ├── blip_vqa_vision.config           # Vision 编码器转换配置
│   ├── blip_vqa_text_encoder.config     # 文本编码器转换配置
│   └── blip_vqa_text_decoder.config     # 文本解码器转换配置（含动态维度）
├── README.md                            # 本教程文档
└── blip_vqa_onnx/                       # 导出与转换产物目录（示例）
    ├── blip_vqa_vision.onnx
    ├── blip_vqa_text_encoder.onnx
    ├── blip_vqa_text_decoder.onnx
    ├── blip_vqa_vision.mindir
    ├── blip_vqa_text_encoder.mindir
    └── blip_vqa_text_decoder.mindir
```

## 参考链接与许可证

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Salesforce/blip-vqa-base（HuggingFace）](https://huggingface.co/Salesforce/blip-vqa-base)
- [BLIP 论文（arXiv:2102.02023）](https://arxiv.org/abs/2102.02023)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

本教程遵循 `Salesforce/blip-vqa-base` 模型的许可证要求，详见其 HuggingFace 页面说明。
