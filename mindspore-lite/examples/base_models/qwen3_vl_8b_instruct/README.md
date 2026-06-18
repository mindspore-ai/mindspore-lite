# Qwen3-VL-8B-Instruct ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-VL-8B-Instruct 导出为 ONNX 以及基于 MindSpore Lite 在昇腾 Ascend 上端到端推理的完整脚本。

## 概览

Qwen3-VL-8B-Instruct 是通义千问团队推出的**指令微调**多模态大模型（Vision-Language Model），具备图像理解、视觉问答、图文描述等多种能力。该模型在视觉编码后直接生成回答，适用于对话式图像理解场景。

本目录提供：

- **ONNX 导出**：将模型拆分为 Vision、LLM Prefill、LLM Decode 三个部分导出
- **MindSpore Lite 集成**：将 ONNX 转换为 `.mindir` 以便在昇腾 Ascend 上部署
- **端到端推理**：基于 MindSpore Lite 的完整推理流程

## 模型架构参数

### 文本模型

| 参数 | 值 |
|---|---|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `intermediate_size` | 12288 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 262144 |

### 视觉编码器

| 参数 | 值 |
|---|---|
| `depth` | 27 |
| `hidden_size` | 1152 |
| `patch_size` | 16 |
| `spatial_merge_size` | 2 |
| `deepstack_visual_indexes` | [8, 16, 24] |

## 架构拆分

模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_vl_vision.onnx`）：对图像进行编码，输出视觉特征及 DeepStack 特征

2. **LLM Prefill**（`qwen3_vl_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits 与 KV cache

3. **LLM Decode**（`qwen3_vl_llm_decode.onnx`）：基于 KV cache 做自回归增量生成

这种拆分避免每步生成都重复计算历史 token 的注意力，从而提升推理效率。

## 环境依赖

### 依赖版本

| 软件包 | 版本 |
|---|---|
| Python | 3.11 |
| torch | 2.7.1 |
| transformers | 5.9.0 |
| onnx | - |
| numpy | - |
| pillow | - |
| mindspore-lite | 2.9.0 |
| CANN | 8.5.0 |

## 快速开始

### 1. 导出 ONNX

导出全部三段模型：

```bash
python export_qwen3_vl_8b_instruct_onnx.py \
    --model-id ./Qwen3-VL-8B-Instruct \
    --output-dir ./qwen3_vl_8b_onnx \
    --device cpu \
    --vision-image-size 128 \
    --kv-cache-len 512
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `./Qwen3-VL-8B-Instruct` |
| `--output-dir` | 导出输出目录 | `./qwen3_vl_8b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--vision-image-size` | Vision 导出图像尺寸（必须能被 patch_size=16 整除） | `128` |
| `--kv-cache-len` | KV cache 固定长度（decode 导出固定 shape） | `512` |
| `--no-custom-op` | 禁用导出 Custom 融合算子节点 | `False` |
| `--no-fuse-weights` | 禁用权重融合（QKV/GateUp）以避免大模型外部数据问题 | `False` |

### 导出产物

每段子图独立保存到各自的子目录，避免 external data 文件重名冲突：

```text
qwen3_vl_8b_onnx/
├── vision/
│   └── qwen3_vl_vision.onnx
├── prefill/
│   └── qwen3_vl_llm_prefill.onnx
└── decode/
    └── qwen3_vl_llm_decode.onnx
```

- `vision/qwen3_vl_vision.onnx`：Vision Tower
- `prefill/qwen3_vl_llm_prefill.onnx` + 外部权重文件：LLM Prefill
- `decode/qwen3_vl_llm_decode.onnx` + 外部权重文件：LLM Decode

## 模型 I/O 说明

### Vision 模型

**输入：**

- `pixel_values`：`float16`，形状 `(seq_len, 1536)`
  其中 `seq_len = grid_h * grid_w`，`grid_h = ceil(image_h / patch_size)`，`grid_w = ceil(image_w / patch_size)`。
  注意：当前导出脚本会把 `grid_thw` 固化在模型内部，因此 Vision ONNX 仅有一个输入。

**输出：**

- `image_embeds`：`float32`，形状 `(num_image_tokens, 4096)`
- `deepstack_embeds`：`float32`，形状 `(3, num_image_tokens, 4096)`

### LLM Prefill 模型

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`
- `position_ids`：`int64`，形状 `(4, batch, seq_len)`
- `image_embeds`：`float16`，形状 `(num_image_tokens, hidden_size)`
- `deepstack_embeds`：`float16`，形状 `(num_deepstack, num_image_tokens, hidden_size)`

**输出：**

- `logits`：`float16`，形状 `(batch, seq_len, vocab_size)`
- `present_key_values`：`float16`，形状 `(2*num_layers, batch, num_kv_heads, seq_len, head_dim)`
  以 Qwen3-VL-8B-Instruct 为例：`2*num_layers=72`、`num_kv_heads=8`、`head_dim=128`。

### LLM Decode 模型

**输入：**

- `input_ids`：`int64`，形状 `(1, 1)`（单 token，固定 batch=1）
- `attention_mask`：`int64`，形状 `(1, 512)`（固定 max_seq_len=512）
- `position_ids`：`int64`，形状 `(4, 1, 1)`
- `past_key_values`：`float16`，形状 `(72, 1, 8, 512, 128)`（固定 cache）
- `cache_pos`：`int64`，形状 `(1,)`（当前有效 cache 长度/位置）

**输出：**

- `logits`：`float16`，形状 `(1, 1, vocab_size)`
- `present_key_values`：`float16`，形状 `(72, 1, 8, 512, 128)`

## MindSpore Lite 模型转换

将 ONNX 转换为 `.mindir`：

```bash
# 转换 Vision 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_8b_onnx/vision/qwen3_vl_vision.onnx \
    --outputFile=./qwen3_vl_8b_onnx/vision/qwen3_vl_vision \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Prefill 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_8b_onnx/prefill/qwen3_vl_llm_prefill.onnx \
    --outputFile=./qwen3_vl_8b_onnx/prefill/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Decode 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_8b_onnx/decode/qwen3_vl_llm_decode.onnx \
    --outputFile=./qwen3_vl_8b_onnx/decode/qwen3_vl_llm_decode \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件示例

#### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

## MindSpore Lite 推理

### 单卡推理

> **300I Duo 用户**：三段子模型权重合计约 35GB（vision ~1.1GB + prefill ~17.6GB + decode ~16.5GB），
> 加上激活与 KV cache 后单卡 44GB 难以容纳，请直接使用下方的**多卡推理**（vision/prefill/decode 分别部署到不同卡）。
> 单卡推理仅适用于显存充裕的卡型（如 800I A2）。

```bash
python infer_qwen3_vl_8b_instruct_mslite.py \
    --vision-model ./qwen3_vl_8b_onnx/vision/qwen3_vl_vision_graph.mindir \
    --prefill-model ./qwen3_vl_8b_onnx/prefill/qwen3_vl_llm_prefill_graph.mindir \
    --decode-model ./qwen3_vl_8b_onnx/decode/qwen3_vl_llm_decode_graph.mindir \
    --processor ./Qwen3-VL-8B-Instruct \
    --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --max-new-tokens 512 \
    --image-size 128 \
    --device ascend \
    --device-id 0
```

### 多卡推理

当单个昇腾卡的显存不足以容纳全部三个子模型时，可将不同子模型部署到不同卡上，通过 `--vision-device-id`、`--prefill-device-id`、`--decode-device-id` 指定各子模型所在设备。

#### 300I Duo（单卡 ~44GB）显存约束

三段子模型权重（fp16 外部数据）实测大小：

| 子模型 | 权重大小 |
|---|---|
| Vision | ~1.1GB |
| Prefill | ~17.6GB |
| Decode | ~16.5GB |

`allow_fp32_to_fp16` 下的实测部署结论：

- **单卡**放不下三段：合计 ~35GB 权重叠加激活/KV，超出 44GB。
- **Prefill 与 Decode 不能共用一张卡**：两者权重合计 ~34GB，第二段加载时 `aclmdlLoadFromMem` 报 OOM（`ret=500002`）。
- **300I Duo 推荐使用 3 张卡**：Vision / Prefill / Decode 各占一张（如下例，已验证可跑通）。

#### 推荐：三段子模型分别在卡 0、1、2（已验证）

```bash
python infer_qwen3_vl_8b_instruct_mslite.py \
    --vision-model ./qwen3_vl_8b_onnx/vision/qwen3_vl_vision_graph.mindir \
    --prefill-model ./qwen3_vl_8b_onnx/prefill/qwen3_vl_llm_prefill_graph.mindir \
    --decode-model ./qwen3_vl_8b_onnx/decode/qwen3_vl_llm_decode_graph.mindir \
    --processor ./Qwen3-VL-8B-Instruct \
    --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --max-new-tokens 512 \
    --image-size 128 \
    --device ascend \
    --device-id 0 \
    --vision-device-id 0 \
    --prefill-device-id 1 \
    --decode-device-id 2
```

> **注意**：
> - 未指定 `--vision-device-id`、`--prefill-device-id`、`--decode-device-id` 时，所有子模型默认使用 `--device-id` 指定的同一张卡。
> - 若 Prefill 在昇腾卡上显存仍吃紧，可加 `--prefill-device cpu` 把 Prefill 卸载到 CPU 上运行（牺牲速度换显存）。

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--vision-model` | Vision MindIR 路径 | 必填 |
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--processor` | Processor 模型路径或目录 | `./Qwen3-VL-8B-Instruct` |
| `--image` | 图片路径或 URL | Qwen demo 图片 |
| `--prompt` | 文本 prompt | `Describe this image.` |
| `--max-new-tokens` | 最大生成 token 数 | `512` |
| `--image-size` | Processor 图像尺寸 | `128` |
| `--no-pad-to-square` | 禁用推理前将图片 pad 成正方形 | `False` |
| `--device` | MindSpore Lite 设备（ascend/cpu） | `ascend` |
| `--device-id` | 全局 Ascend device id | `0` |
| `--vision-device-id` | Vision 模型所在设备 ID（不指定时与 `--device-id` 一致） | 同 `--device-id` |
| `--prefill-device-id` | Prefill 模型所在设备 ID（不指定时与 `--device-id` 一致） | 同 `--device-id` |
| `--decode-device-id` | Decode 模型所在设备 ID（不指定时与 `--device-id` 一致） | 同 `--device-id` |
| `--prefill-device` | Prefill 模型设备类型，如 `cpu` 可将 Prefill 运行在 CPU 上 | 同 `--device` |

## 性能数据

> **测量口径说明**：下表为**单次运行**（single run）实测，**未做多轮平均、也未隔离 warm-up**，仅作量级参考；实际数值会随图像尺寸、prompt 长度、采样参数波动。
>
> 测试条件：300I Duo（3 卡部署：Vision/Prefill/Decode 分别在卡 0/1/2，`allow_fp32_to_fp16`，image-size=128，Qwen demo 图，`--max-new-tokens 64`）。

| 指标 | 数值 | 来源 |
|---|---|---|
| Vision 编码 | ~18 ms | demo 图单次调用 |
| Prefill（首 token） | ~248 ms | 单次调用 |
| Decode（单步） | ~95 ms | 63 步平均 |
| Decode 吞吐 | ~10.6 tokens/s | 由 `decode_steps / decode_total` 推算（非直接测量） |
| 端到端（生成 64 个新 token） | ~6.9 s | 含 vision + prefill + decode |

> 说明：image-size=128 时 prompt 较短；增大图像尺寸或 prompt 长度会显著增加 Prefill 与 Decode 耗时。Decode 已开启 Ascend I/O buffer 复用（推理脚本 `_ensure_decode_io`），避免每步重复分配/拷贝 KV cache，从而降低单步延迟。

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-8B-Instruct ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3-VL-8B-Instruct HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3-VL 模型的许可证要求，详见 [Qwen3-VL license](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。
