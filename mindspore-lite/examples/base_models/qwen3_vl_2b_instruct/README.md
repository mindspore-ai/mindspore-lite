# Qwen3-VL-2B-Instruct ONNX 导出与推理

本目录提供 Qwen3-VL-2B-Instruct 导出为 ONNX 以及端到端推理的完整脚本。实现上将模型拆分为 3 个组件，便于部署与加速。

## 概览

Qwen3-VL-2B-Instruct 是一个同时处理图像与文本的多模态大模型。本目录提供：

- **ONNX 导出**：将模型拆分为 Vision、LLM Prefill、LLM Decode 三个部分导出

- **ONNX 推理**：使用三段 ONNX 模型完成端到端推理

- **MindSpore Lite 集成**：可选，将 ONNX 转换为 `.mindir` 以便在 Ascend 上部署

## 架构拆分

模型被拆分为 3 个 ONNX 文件：

1. **Vision Tower**（`qwen3_vl_vision.onnx`）：对图像进行编码，输出视觉特征

2. **LLM Prefill**（`qwen3_vl_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits 与 KV cache

3. **LLM Decode**（`qwen3_vl_llm_decode.onnx`）：基于 KV cache 做自回归增量生成

这种拆分避免每步生成都重复计算历史 token 的注意力，从而提升推理效率。

## 环境依赖

### 依赖版本

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.12.0 |
| torch          | 2.12.0+cu130 |
| transformers   | 5.4.0 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.26.0 |
| onnxscript     | 0.7.0 |
| numpy          | 2.4.4 |
| pillow         | 12.2.0 |
| mindspore-lite | 2.8.0 |

### 模型访问

请确保你能从 HuggingFace 访问 `Qwen/Qwen3-VL-2B-Instruct`。

## 快速开始

### 1. 导出 ONNX

导出全部三段模型：

```bash
python export_qwen3_vl_2b_instruct_onnx.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --output-dir ./qwen3_vl_2b_instruct_onnx \
    --device cpu \
    --vision-image-size 128
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3-VL-2B-Instruct` |
| `--output-dir` | 导出输出目录 | `./qwen3_vl_2b_instruct_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--vision-image-size` | Vision 导出图像尺寸（正方形输入时为边长） | `128` |
| `--kv-cache-len` | KV cache 固定长度（decode 导出固定 shape） | `512` |
| `--no-custom-op` | 禁用导出 Custom 融合算子节点 | `False` |

导出产物：

- `qwen3_vl_vision.onnx`：Vision Tower

- `qwen3_vl_llm_prefill.onnx` + `.data`：LLM Prefill

- `qwen3_vl_llm_decode.onnx` + `.data`：LLM Decode

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
  以 Qwen3-VL-2B-Instruct 为例：`2*num_layers=56`、`num_kv_heads=8`、`head_dim=128`。

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

## MindSpore Lite 集成（可选）

如需在 Ascend 上部署，可将 ONNX 转换为 `.mindir`：

```bash
# 转换 Vision 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_vision.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_vision \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Prefill 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini

# 转换 Decode 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode \
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

随后用 MindSpore Lite 推理：

```bash
python infer_qwen3_vl_2b_instruct_mslite.py \
    --vision-model ./qwen3_vl_2b_instruct_onnx/qwen3_vl_vision.mindir \
    --prefill-model ./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill.mindir \
    --decode-model ./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode.mindir \
    --image ./your_image.jpg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
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
| `--processor` | Processor 模型路径或目录 | `./Qwen3-VL-2B-Instruct` |
| `--image` | 图片路径或 URL | `./demo.jpeg` |
| `--prompt` | 文本 prompt | `Describe this image.` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--image-size` | Processor 图像尺寸 | `128` |
| `--no-pad-to-square` | 禁用推理前将图片 pad 成正方形 | `False` |
| `--device` | MindSpore Lite 设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device id | `0` |

### 外部资源说明

- README 示例中的 `--image` 使用 Qwen 官方 demo 图片：`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`。
- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径，例如 `--image ./your_image.jpg`。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

### 性能数据

#### 性能测试结果

测试模型：Qwen3-VL-2B-Instruct  
测试条件：输出128 token
测试环境：昇腾Atlas 300I Duo，CANN 8.5.0，MindSpore Lite 2.8.0

| 指标                       | time |
|--------------------------|------------|
| Vision (ms)              | 12.895      |
| Prefill (ms)             | 121.855      |
| Total Decode (ms)        | 3863.855     |
| **Avg decode step (ms)** | **30.42** |
| Total (ms)               | 3998.605     |
| E2E (ms)                 | 4354.784     |
| **Throughput (tok/s)**   | **33.13** |

#### 推理结果

Input Prompt: Describe this image.  
Generated Response: This is a photograph capturing a tender moment between a woman and a dog in the ocean at sunset.

- **Setting**: The scene is set on a beach, with the ocean and a calm horizon visible in the background. The sun is low on the horizon, casting a warm, golden light across the water and the subjects.
- **Subjects**: A woman with long, light brown hair is sitting on the sand, holding a small, dark-colored dog. The dog is wearing a light-colored, possibly beige or tan, harness. They are both facing each other, and the woman is holding a small, dark object, possibly a piece of

## 目录结构

```Shell
qwen3_vl_2b_instruct/
├── export_qwen3_vl_2b_instruct_onnx.py          # ONNX 导出脚本（3 段模型）
├── infer_qwen3_vl_2b_instruct_mslite.py         # MindSpore Lite 推理脚本
├── README.md                        # 本说明
└── qwen3_vl_2b_instruct_onnx/       # 导出模型目录
    ├── qwen3_vl_vision.onnx
    ├── qwen3_vl_llm_prefill.onnx + .data
    └── qwen3_vl_llm_decode.onnx + .data
```

## 关键点

### Prefill / Decode 拆分

将 LLM 拆成 prefill 与 decode 有助于优化推理：

- **Prefill**：一次性处理完整 prompt（含图像 token）

- **Decode**：利用 KV cache 增量生成

- **收益**：避免每步生成都重新计算历史 token 的注意力

### 内存管理

导出脚本包含必要的内存回收逻辑以降低导出峰值内存：

- 导出阶段之间清理 CUDA cache

- 删除不再使用的子模块以释放内存

- 可在 8GB 内存机器上以较小配置完成导出（视环境而定）

### 动态形状

模型通过 ONNX dynamic axes 支持动态 batch 与序列长度。

## 常见问题

### 导出时内存不足（OOM）

导出过程中出现 OOM 时可尝试：

- 使用 `--device cpu` 在 CPU 上导出（更慢但通常更省显存）

- 减小 `--vision-image-size`（默认 128）

- 关闭其它占用内存的程序

### image_embeds 长度不匹配

若推理时报 `image_embeds` 长度不匹配：

- 确保 processor 与模型版本一致

- 确认 `--vision-image-size` 与导出配置一致

- 检查 `image_grid_thw` 是否与输入图像对应

## 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)

- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)

- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)

- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3-VL 模型的许可证要求，详见 [Qwen3-VL license](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)。
