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

### Python 环境

```bash
pip install -U "transformers>=4.50" torch onnx onnxscript pillow numpy
pip install -U onnxruntime
```

如需 GPU 加速：

```bash
pip install -U onnxruntime-gpu
```

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

导出产物：

- `qwen3_vl_vision.onnx`：Vision Tower

- `qwen3_vl_llm_prefill.onnx` + `.data`：LLM Prefill

- `qwen3_vl_llm_decode.onnx` + `.data`：LLM Decode

### 2. 运行 ONNX 推理

端到端推理示例：

```bash
python infer_qwen3_vl_2b_instruct_onnx.py \
    --vision qwen3_vl_2b_instruct_onnx/qwen3_vl_vision.onnx \
    --prefill qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill.onnx \
    --decode qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode.onnx \
    --processor Qwen/Qwen3-VL-2B-Instruct \
    --image ./your_image.jpg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --device cpu
```

## 模型 I/O 说明

### Vision 模型

**输入：**

- `pixel_values`：`float16`，形状 `(seq_len, 1536)`

- `grid_thw`：`int64`，形状 `(1, 3)`，分别表示 `t/h/w` 网格维度（可选；部分导出变体会把它固化进模型）

**输出：**

- `image_embeds`：`float16`，形状 `(num_image_tokens, hidden_size)`

- `deepstack_embeds`：`float16`，形状 `(num_deepstack, num_image_tokens, hidden_size)`

### LLM Prefill 模型

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`

- `attention_mask`：`int64`，形状 `(batch, seq_len)`

- `position_ids`：`int64`，形状 `(4, batch, seq_len)`

- `image_embeds`：`float16`，形状 `(num_image_tokens, hidden_size)`

- `deepstack_embeds`：`float16`，形状 `(num_deepstack, num_image_tokens, hidden_size)`

**输出：**

- `logits`：`float16/float32`，形状 `(batch, seq_len, vocab_size)`

- `present_key_values`：`float16`，形状 `(2*num_layers, batch, num_kv_heads, seq_len, head_dim)`

### LLM Decode 模型

**输入：**

- `input_ids`：`int64`，形状 `(batch, 1)`（单 token）

- `attention_mask`：`int64`，形状 `(batch, total_seq_len)`

- `position_ids`：`int64`，形状 `(4, batch, 1)`

- `past_key_values`：`float16`，形状 `(2*num_layers, batch, num_kv_heads, past_seq_len, head_dim)`

**输出：**

- `logits`：`float16/float32`，形状 `(batch, 1, vocab_size)`

- `present_key_values`：`float16`，形状 `(2*num_layers, batch, num_kv_heads, total_seq_len, head_dim)`

## MindSpore Lite 集成（可选）

如需在 Ascend 上部署，可将 ONNX 转换为 `.mindir`：

```bash
# 转换 Vision 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_vision.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_vision \
    --optimize=ascend_oriented \
    --saveType=MINDIR

# 转换 Prefill 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented \
    --saveType=MINDIR

# 转换 Decode 模型
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode.onnx \
    --outputFile=./qwen3_vl_2b_instruct_onnx/qwen3_vl_llm_decode \
    --optimize=ascend_oriented \
    --saveType=MINDIR
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

## 目录结构

```Shell
qwen3_vl_2b_instruct/
├── export_qwen3_vl_2b_instruct_onnx.py          # ONNX 导出脚本（3 段模型）
├── infer_qwen3_vl_2b_instruct_onnx.py           # ONNX 端到端推理脚本
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
