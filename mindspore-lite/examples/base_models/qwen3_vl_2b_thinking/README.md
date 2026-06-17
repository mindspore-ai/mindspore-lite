# Qwen3-VL-2B-Thinking ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-VL-2B-Thinking 导出为 ONNX 以及基于 MindSpore Lite 在昇腾 Ascend 上端到端推理的完整脚本。

## 概览

Qwen3-VL-2B-Thinking 是一个具备**思维链推理能力**的多模态大模型（Vision-Language Model），在生成回答前会先进行逐步推理（Chain-of-Thought），从而提升复杂视觉理解任务的准确度。

本目录提供：

- **ONNX 导出**：将模型拆分为 Vision、LLM Prefill、LLM Decode 三个部分导出
- **MindSpore Lite 集成**：将 ONNX 转换为 `.mindir` 以便在昇腾 Ascend 上部署
- **端到端推理**：基于 MindSpore Lite 的完整推理流程

## 模型架构参数

### 文本模型

| 参数 | 值 |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 (GQA) |
| `head_dim` | 128 |
| `intermediate_size` | 6144 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 262144 |

### 视觉编码器

| 参数 | 值 |
|---|---|
| `depth` | 24 |
| `hidden_size` | 1024 |
| `patch_size` | 16 |
| `spatial_merge_size` | 2 |
| `deepstack_visual_indexes` | [5, 11, 17] |

### Thinking 特性

Qwen3-VL-2B-Thinking 与 Instruct 版本具有相同的模型架构，但增加了思维链推理模式：

- 模型在回答前自动生成推理过程
- 推荐推理参数：`temperature=0.6`、`top_p=0.95`
- 可通过在 prompt 中添加 `/no_think` 禁用思考模式

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
python export_qwen3_vl_2b_thinking_onnx.py \
    --model-id ./Qwen/Qwen3-VL-2B-Thinking \
    --output-dir ./qwen3_vl_2b_thinking_onnx \
    --device cpu \
    --vision-image-size 128 \
    --kv-cache-len 512
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope ID | `./Qwen/Qwen3-VL-2B-Thinking` |
| `--output-dir` | 导出输出目录 | `./qwen3_vl_2b_thinking_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--vision-image-size` | Vision 导出图像尺寸（必须能被 patch_size=16 整除） | `128` |
| `--kv-cache-len` | KV cache 固定长度（decode 导出固定 shape） | `512` |
| `--no-custom-op` | 禁用导出 Custom 融合算子节点 | `False` |

### 导出产物

- `qwen3_vl_vision.onnx`：Vision Tower
- `qwen3_vl_llm_prefill.onnx` + 外部权重文件：LLM Prefill
- `qwen3_vl_llm_decode.onnx` + 外部权重文件：LLM Decode

## 模型 I/O 说明

### Vision 模型

**输入：**

- `pixel_values`：`float16`，形状 `(seq_len, 1536)`
  其中 `seq_len = grid_h * grid_w`，`grid_h = ceil(image_h / patch_size)`，`grid_w = ceil(image_w / patch_size)`。
  注意：当前导出脚本会把 `grid_thw` 固化在模型内部，因此 Vision ONNX 仅有一个输入。

**输出：**

- `image_embeds`：`float32`，形状 `(num_image_tokens, 2048)`
- `deepstack_embeds`：`float32`，形状 `(3, num_image_tokens, 2048)`

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
  以 Qwen3-VL-2B-Thinking 为例：`2*num_layers=56`、`num_kv_heads=8`、`head_dim=128`。

### LLM Decode 模型

**输入：**

- `input_ids`：`int64`，形状 `(1, 1)`（单 token，固定 batch=1）
- `attention_mask`：`int64`，形状 `(1, 512)`（固定 max_seq_len=512）
- `position_ids`：`int64`，形状 `(4, 1, 1)`
- `past_key_values`：`float16`，形状 `(56, 1, 8, 512, 128)`（固定 cache）
- `cache_pos`：`int64`，形状 `(1,)`（当前有效 cache 长度/位置）

**输出：**

- `logits`：`float16`，形状 `(1, 1, vocab_size)`
- `present_key_values`：`float16`，形状 `(56, 1, 8, 512, 128)`

## MindSpore Lite 模型转换

将 ONNX 转换为 `.mindir`：

```bash
# 转换 Vision 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_vision.onnx \
    --outputFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_vision \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Prefill 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_prefill.onnx \
    --outputFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Decode 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_decode.onnx \
    --outputFile=./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_decode \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件示例

#### `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = force_fp32
```

## MindSpore Lite 推理

```bash
python infer_qwen3_vl_2b_thinking_mslite.py \
    --vision-model ./qwen3_vl_2b_thinking_onnx/qwen3_vl_vision.mindir \
    --prefill-model ./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_prefill_graph.mindir \
    --decode-model ./qwen3_vl_2b_thinking_onnx/qwen3_vl_llm_decode_graph.mindir \
    --processor ./Qwen/Qwen3-VL-2B-Thinking \
    --image "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \
    --prompt "Describe this image." \
    --max-new-tokens 512 \
    --image-size 128 \
    --device ascend \
    --device-id 0
```

### 推理参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--vision-model` | Vision MindIR 路径 | 必填 |
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--processor` | Processor 模型路径或目录 | `./Qwen/Qwen3-VL-2B-Thinking` |
| `--image` | 图片路径或 URL | Qwen demo 图片 |
| `--prompt` | 文本 prompt | `Describe this image.` |
| `--max-new-tokens` | 最大生成 token 数 | `512` |
| `--image-size` | Processor 图像尺寸 | `128` |
| `--no-pad-to-square` | 禁用推理前将图片 pad 成正方形 | `False` |
| `--device` | MindSpore Lite 设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device id | `0` |

### 外部资源说明

- README 示例中使用 Qwen 官方 demo 图片：`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`。
- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

## 性能数据

### 性能测试结果

测试模型：Qwen3-VL-2B-Thinking
测试条件：输入图片 + 文本 prompt，输出 128 token
测试环境：昇腾 Atlas 300I Duo，CANN 8.5.0，MindSpore Lite 2.9.0

| 指标 | 耗时 (ms) |
|---|---|
| Vision | 13.079 |
| Prefill | 176.734 |
| Total Decode | 3805.330 |
| Decode 步数 | 127 |
| **Avg decode step** | **29.963** |
| Total | 3994.183 |
| E2E | 4794.628 |
| **吞吐量 (tok/s)** | **26.55** |

### 推理结果示例

```text
Input Prompt: Describe this image.
Response: Got it, let's describe this image. First, the scene is a beach at sunset or
sunrise, with the ocean in the background. There are two people: one is a person sitting
on the sand, and the other is a dog...
```

## 关键点

### Prefill / Decode 拆分

将 LLM 拆成 prefill 与 decode 有助于优化推理：

- **Prefill**：一次性处理完整 prompt（含图像 token），支持动态序列长度
- **Decode**：利用固定长度 KV cache 做增量生成，采用 Scatter 算子更新 cache
- **收益**：避免每步生成都重新计算历史 token 的注意力

### Thinking 模式

- 模型在回答前自动进行思维链推理，生成中间推理步骤
- 推理内容有助于提高复杂视觉问答任务的准确度
- 可通过 prompt 中的 `/no_think` 关键字禁用思考模式

### DeepStack 特征注入

Qwen3-VL 的视觉编码器在层 5、11、17 提取中间特征，通过 PatchMerger 注入到文本解码器的前 3 层，从而增强视觉理解能力。

### 内存管理

导出脚本包含必要的内存回收逻辑以降低导出峰值内存：

- 导出阶段之间清理 CUDA cache
- 删除不再使用的子模块以释放内存
- Vision 和 LLM 分两步加载，避免同时占用全部内存

### 自定义算子

导出脚本使用以下 CANN 自定义算子以提升推理性能：

| 算子 | 说明 |
|---|---|
| `RmsNorm` | RMS 归一化融合算子 |
| `SwiGlu` | SwiGLU 激活函数融合算子 |
| `RotaryMul` | 旋转位置编码融合算子 |
| `PromptFlashAttention` | Prefill 阶段 Flash Attention |
| `IncreFlashAttention` | Decode 阶段增量 Flash Attention |
| `Scatter` | KV Cache 更新算子 |

## 常见问题

### 导出时内存不足（OOM）

导出过程中出现 OOM 时可尝试：

- 使用 `--device cpu` 在 CPU 上导出（更慢但通常更省显存）
- 减小 `--vision-image-size`（默认 128，必须能被 16 整除）
- 减小 `--kv-cache-len`（默认 512）
- 关闭其它占用内存的程序

### image_embeds 长度不匹配

若推理时报 `image_embeds` 长度不匹配：

- 确保 processor 与模型版本一致
- 确认 `--vision-image-size` 与导出配置一致
- 检查 `image_grid_thw` 是否与输入图像对应

### 转换时出现 Warning

MindSpore Lite 模型转换过程中出现的 Warning 日志可以忽略，只要最终输出 `CONVERT RESULT SUCCESS:0` 即表示转换成功。

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL-2B-Thinking ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Thinking)
- [Qwen3-VL-2B-Thinking HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3-VL 模型的许可证要求，详见 [Qwen3-VL license](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking)。
