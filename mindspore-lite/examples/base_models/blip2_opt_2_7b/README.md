# Salesforce/blip2-opt-2.7b ONNX 导出与推理教程

## 目录
- [概览](#概览)
- [架构拆分](#架构拆分)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [模型 I/O 说明](#模型-io-说明)
- [实测性能数据](#实测性能数据)
- [对齐验证](#对齐验证)
- [常见问题](#常见问题)
- [目录结构](#目录结构)
- [参考链接](#参考链接)

## 概览

本教程演示如何将 **Salesforce/blip2-opt-2.7b**（视觉-语言多模态模型）适配到
MindSpore Lite 云侧推理框架。模型由三部分组成：

1. **EVA-ViT-G 视觉编码器**：224×224 图像 → `image_embeds[1,257,1408]`。
2. **Q-Former**：将 257 个图像 token 通过 32 个可学习查询 token 压缩为
   `query_embeds[1,32,768]`，再经 `language_projection` 投影到 OPT 隐藏维度
   `language_model_inputs[1,32,2560]`。
3. **OPT-2.7B 语言模型**：自回归解码（prefill + decode + KV cache）。

由于 BLIP-2 的 `generate` 内部依赖 `DynamicCache`，无法被 `torch.onnx` 静态追踪，
因此我们采用 **四阶段固定形状拆分**：vision / qformer / opt-prefill / opt-decode。
拆分方案参照 `qwen2.5_vl_3b_instruct` 的 prefill+decode KV cache 模式。

模型规模（Salesforce/blip2-opt-2.7b 文本配置）：

| 参数 | 值 |
|---|---|
| vision hidden_size | 1408 |
| vision num_hidden_layers | 39 |
| vision num_attention_heads | 16 (head_dim=88) |
| vision image_size / patch_size | 224 / 14 (256 patches + 1 CLS = 257) |
| qformer hidden_size | 768 |
| qformer num_hidden_layers | 12 |
| qformer num_query_tokens | 32 |
| qformer cross_attention_frequency | 2 |
| OPT hidden_size | 2560 |
| OPT num_hidden_layers | 32 |
| OPT num_attention_heads | 32 (head_dim=80) |
| OPT ffn_dim | 10240 |
| vocab_size | 50272 |

## 架构拆分

固定形状拆分（导出时确定，运行时不可变）：

| 阶段 | 模块 | 输入 | 输出 |
|---|---|---|---|
| 1. Vision | EVA-ViT-G | `pixel_values[1,3,224,224]` fp32 | `image_embeds[1,257,1408]` fp32 |
| 2. Q-Former | Q-Former + Linear | `image_embeds[1,257,1408]` fp32 | `query_embeds[1,32,768]`, `language_model_inputs[1,32,2560]` |
| 3. Prefill | OPT prefill (无 past) | `inputs_embeds[1, 32+q_len, 2560]`, `attention_mask[1, 32+q_len]`, `position_ids[1, 32+q_len]` | `logits[1, 32+q_len, 50272]`, `present_key_values[64, 1, 32, 32+q_len, 80]` |
| 4. Decode | OPT decode (单步 + past) | `inputs_embeds[1,1,2560]`, `attention_mask[1, max_total_len]`, `position_ids[1,1]`, `past_key_values[64, 1, 32, max_total_len, 80]`, `cache_pos[1]` | `logits[1,1,50272]`, `present_key_values[64, 1, 32, max_total_len, 80]` |

默认 `q_len = 32`（问题 token 数），`max_total_len = 256`，所以 prefill 序列长度为 64。

**KV cache 布局**：堆叠在 dim 0，层间交替 key/value：
`[2*num_layers=64, batch=1, num_heads=32, seq, head_dim=80]`。
decode 每步仅在 `cache_pos` 列写入新 KV，其余列保持不变。

**关键技术点**：
- `do_constant_folding=False`，opset 17，float32 导出（参照 blip_vqa_base 约定）。
- 转换 MindIR 时使用 `--optimize=ascend_oriented --saveType=MINDIR`。
- config 中 `ge.exec.precision_mode = force_fp16`，prefill/decode 开启
  `plugin_custom_ops=All` 以融合 OPT 的注意力算子。
- 推理核心路径 **不依赖 torch**（仅 numpy + mslite + PIL）。
- OPT 的 `embed_tokens` 需要在推理前一次性导出为 `.npy`（见下文）。

## 环境依赖

| 软件包 | 版本 |
|---|---|
| Python | 3.11 |
| mindspore-lite | 2.9.0 |
| transformers | 5.x（仅用于导出/对齐/分词） |
| torch | 2.1+（仅用于导出/对齐/导出 embeddings） |
| numpy | 1.26 |
| pillow | 12.x |
| onnx | 1.19 |

```bash
pip install mindspore-lite==2.9.0 transformers torch numpy pillow onnx
```

昇腾环境还需 CANN 工具链及 `atc` 转换器在 `PATH` 中。

## 快速开始

### 1. 导出 ONNX

```bash
python export_blip2_opt_2_7b_onnx.py \
    --model-id Salesforce/blip2-opt-2.7b \
    --output-dir ./blip2_opt_2_7b_onnx \
    --image-size 224 \
    --question-len 32 \
    --max-total-len 256
```

产物：

```
blip2_opt_2_7b_onnx/
├── blip2_vision.onnx
├── blip2_qformer.onnx
├── blip2_opt_prefill.onnx
└── blip2_opt_decode.onnx
```

导出参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-id` | Salesforce/blip2-opt-2.7b | HF 模型 id 或本地路径 |
| `--output-dir` | ./blip2_opt_2_7b_onnx | ONNX 输出目录 |
| `--device` | cpu | 导出设备（cpu/cuda） |
| `--opset` | 17 | ONNX opset |
| `--image-size` | 224 | 视觉输入尺寸 |
| `--question-len` | 32 | 固定问题长度（pad 后） |
| `--max-total-len` | 256 | KV cache 固定总长度 |

### 2. 转换 MindIR

使用 `atc` 或 `converter` 将每个 ONNX 转为 MindIR，对应 config 在 `configs/` 下：

```bash
# Vision
converter --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
    --modelFile=blip2_opt_2_7b_onnx/blip2_vision.onnx \
    --outputFile=mindir/blip2_vision \
    --configFile=configs/blip2_vision.config

# Q-Former
converter --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
    --modelFile=blip2_opt_2_7b_onnx/blip2_qformer.onnx \
    --outputFile=mindir/blip2_qformer \
    --configFile=configs/blip2_qformer.config

# OPT prefill
converter --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
    --modelFile=blip2_opt_2_7b_onnx/blip2_opt_prefill.onnx \
    --outputFile=mindir/blip2_opt_prefill \
    --configFile=configs/blip2_opt_prefill.config

# OPT decode
converter --fmk=ONNX --optimize=ascend_oriented --saveType=MINDIR \
    --modelFile=blip2_opt_2_7b_onnx/blip2_opt_decode.onnx \
    --outputFile=mindir/blip2_opt_decode \
    --configFile=configs/blip2_opt_decode.config
```

#### config 说明

- **`configs/blip2_vision.config`** / **`configs/blip2_qformer.config`**：
  仅设置 `ge.exec.precision_mode = force_fp16`（静态形状）。
- **`configs/blip2_opt_prefill.config`**：声明 prefill 的 3 个动态输入维度，并
  通过 `ge.dynamicDims` 枚举支持的序列长度（默认 `32;48;64`，对应
  `32+q_len` 取 `q_len=0/16/32`）。如改了 `--question-len`，需要同步修改本档。
  开启 `plugin_custom_ops=All`。
- **`configs/blip2_opt_decode.config`**：完全静态形状。
  `past_key_values: 64,1,32,256,80`（`64=2*32 层`，`32 头`，`256=max_total_len`，
  `80=head_dim`）。如改 `--max-total-len`，需要同步修改本档与导出。

### 3. 导出 OPT embeddings（一次性）

OPT 的 `embed_tokens` 查找不包含在 prefill/decode ONNX 中（它们直接接收
`inputs_embeds`）。运行推理前，需要把 `embed_tokens.weight` 导出为 numpy：

```bash
python infer_blip2_opt_2_7b_mslite.py --dump-embeddings \
    --model-id Salesforce/blip2-opt-2.7b \
    --opt-embeddings opt_embed_tokens.npy
```

产物 `opt_embed_tokens.npy` 形状 `[50272, 2560]`，约 488 MB（fp32）。

### 4. 推理

```bash
python infer_blip2_opt_2_7b_mslite.py \
    --vision-model mindir/blip2_vision.mindir \
    --qformer-model mindir/blip2_qformer.mindir \
    --prefill-model mindir/blip2_opt_prefill.mindir \
    --decode-model mindir/blip2_opt_decode.mindir \
    --tokenizer Salesforce/blip2-opt-2.7b \
    --opt-embeddings opt_embed_tokens.npy \
    --image path/to/image.jpg \
    --question "What is in this image?" \
    --max-new-tokens 32 \
    --device ascend --device-id 0
```

推理参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--vision-model` | (必填) | vision MINDIR 路径 |
| `--qformer-model` | (必填) | qformer MINDIR 路径 |
| `--prefill-model` | (必填) | OPT prefill MINDIR 路径 |
| `--decode-model` | (必填) | OPT decode MINDIR 路径 |
| `--tokenizer` | Salesforce/blip2-opt-2.7b | 分词器 |
| `--opt-embeddings` | opt_embed_tokens.npy | OPT embed_tokens .npy 路径 |
| `--image` | (必填) | 图像路径或 URL |
| `--question` | (必填) | 问题文本 |
| `--image-size` | 224 | 视觉输入尺寸 |
| `--question-len` | 32 | 固定问题长度（须与导出一致） |
| `--max-total-len` | 256 | KV cache 总长度（须与导出一致） |
| `--max-new-tokens` | 32 | 最大生成 token 数 |
| `--device` | ascend | ascend / cpu |
| `--device-id` | 0 | 设备 id |

## 模型 I/O 说明

### Vision 模型 (`blip2_vision.onnx`)
- 输入：`pixel_values` fp32 `[1,3,224,224]`
- 输出：`image_embeds` fp32 `[1,257,1408]`

### Q-Former 模型 (`blip2_qformer.onnx`)
- 输入：`image_embeds` fp32 `[1,257,1408]`
- 输出：`query_embeds` fp32 `[1,32,768]`，`language_model_inputs` fp32 `[1,32,2560]`

### OPT Prefill 模型 (`blip2_opt_prefill.onnx`)
- 输入：
  - `inputs_embeds` fp32 `[1, 32+q_len, 2560]`（32 个 Q-Former 投影 + 问题 token 嵌入）
  - `attention_mask` int64 `[1, 32+q_len]`
  - `position_ids` int64 `[1, 32+q_len]`
- 输出：
  - `logits` fp16 `[1, 32+q_len, 50272]`
  - `present_key_values` fp16 `[64, 1, 32, 32+q_len, 80]`

### OPT Decode 模型 (`blip2_opt_decode.onnx`)
- 输入：
  - `inputs_embeds` fp32 `[1,1,2560]`
  - `attention_mask` int64 `[1, max_total_len]`
  - `position_ids` int64 `[1,1]`
  - `past_key_values` fp16 `[64, 1, 32, max_total_len, 80]`
  - `cache_pos` int64 `[1]`（当前写入位置）
- 输出：
  - `logits` fp16 `[1,1,50272]`
  - `present_key_values` fp16 `[64, 1, 32, max_total_len, 80]`

## 实测性能数据

> **待运行填入**：以下表格中的数值需要在实际昇腾硬件（如 Atlas 300I Duo /
> Atlas 800I A2）上运行后填入。请勿使用伪造数据。

### 推理时延（image_size=224, question_len=32, max_total_len=256, device=ascend）

| 运行 | preprocess(ms) | vision(ms) | qformer(ms) | prefill(ms) | decode_total(ms) | decode_steps | decode_avg(ms) | e2e(ms) | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 |
| 2 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 |
| 3 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 | 待运行填入 |

## 对齐验证

使用 `align_blip2_opt_2_7b.py` 将 MSLite 流水线与 HF 官方 `generate` 输出对齐：

```bash
python align_blip2_opt_2_7b.py \
    --vision-model mindir/blip2_vision.mindir \
    --qformer-model mindir/blip2_qformer.mindir \
    --prefill-model mindir/blip2_opt_prefill.mindir \
    --decode-model mindir/blip2_opt_decode.mindir \
    --opt-embeddings opt_embed_tokens.npy \
    --model-id Salesforce/blip2-opt-2.7b \
    --image path/to/image.jpg \
    --question "What is in this image?" \
    --max-new-tokens 32 \
    --cosine-threshold 0.999
```

通过条件：
1. 答案字符串精确匹配（忽略大小写、首尾空白）。
2. prefill 最后一位 logits 余弦相似度 ≥ `--cosine-threshold`（默认 0.999）。

参考与 MSLite 的图像预处理一致（resize 224×224 bicubic → 1/255 → CLIP 均值/方差
归一化）；脚本会打印 `pixel_values max|d|` 用于检查预处理一致性。

## 常见问题

### Q1：导出时报 OOM（内存不足）
BLIP-2 OPT-2.7B 全量 fp32 约需 12 GB+。导出脚本将 vision/qformer 与 OPT 分两步
加载（先加载并删除 `language_model`，再重载并删除 `vision_model`/`qformer`）。
如仍 OOM，尝试：缩小 `--max-total-len`；在 CPU 上导出（`--device cpu`）；
或使用 `low_cpu_mem_usage=True` 的模型加载（脚本已默认开启）。

### Q2：MindIR 转换时出现 unsupported op
检查 config 是否开启 `plugin_custom_ops=All`（prefill/decode 必须）。
若 vision/qformer 出现不支持的算子，可尝试调整 `ge.exec.precision_mode`
或升级 `atc` / CANN 版本。

### Q3：MSLite 答案与 HF 不一致
- 检查 `pixel_values max|d|`（对齐脚本输出），> 1e-3 通常表示预处理不一致。
- 确认 `--question-len`、`--max-total-len` 与导出时一致。
- 确认 `opt_embed_tokens.npy` 来自同一个 checkpoint。
- 若 cosine 接近但仍 < 0.999，可尝试 `force_fp32` 替换 `force_fp16` 排查精度问题。

### Q4：解码超过 max_total_len 被截断
- 增大 `--max-total-len`（同步修改导出命令与 `blip2_opt_decode.config`）。
- 或减少 `--max-new-tokens`。

### Q5：为什么需要单独导出 `opt_embed_tokens.npy`？
OPT 的 `embed_tokens` 查找未包含在 prefill/decode ONNX 中——这两个模块直接接收
`inputs_embeds`。为了保持推理核心路径不依赖 torch，我们一次性把权重导出为 numpy
数组，运行时通过纯 numpy 索引完成 token → embed 的查找。

## 目录结构

```
blip2_opt_2_7b/
├── README.md
├── export_blip2_opt_2_7b_onnx.py   # 四阶段 ONNX 导出
├── infer_blip2_opt_2_7b_mslite.py  # 纯 numpy + mslite 推理
├── align_blip2_opt_2_7b.py         # HF vs MSLite 对齐验证
└── configs/
    ├── blip2_vision.config
    ├── blip2_qformer.config
    ├── blip2_opt_prefill.config
    └── blip2_opt_decode.config
```

## 参考链接

- 模型：https://huggingface.co/Salesforce/blip2-opt-2.7b
- 论文：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image
  Encoders and Large Language Models (https://arxiv.org/abs/2301.12597)
- MindSpore Lite 文档：https://www.mindspore.cn/lite
- 参考实现：
  - `examples/base_models/qwen2.5_vl_3b_instruct/`（prefill+decode KV cache 拆分）
  - `examples/base_models/blip_vqa_base/`（vision+text 拆分，numpy 贪心解码）

许可证：本教程代码遵循 Apache License 2.0。模型权重遵循其原始许可证
（Salesforce/blip2-opt-2.7b 使用 OPT 的 license，详见 HF 模型页）。
