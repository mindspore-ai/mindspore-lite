# Qwen3-VL-4B-Instruct ONNX 导出与推理

本目录提供 Qwen3-VL-4B-Instruct 导出为 ONNX、端到端推理、以及 MindSpore Lite Ascend 部署的完整脚本。模型被拆分为 Vision、LLM Prefill、LLM Decode 三个组件。

本demo中 Vision 模块为固定shape 128x128的输入图片， prefill和decode模块为动态shape。

## 架构拆分

1. **Vision Tower**（`qwen3_vl_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`qwen3_vl_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits 与 KV cache
3. **LLM Decode**（`qwen3_vl_llm_decode.onnx`）：基于 KV cache 做自回归增量生成

## 环境依赖

### Python 环境

```bash
pip install -U "transformers>=4.50" torch onnx onnxscript pillow numpy
pip install -U onnxruntime
```

### MindSpore Lite

- Python wheel: `mindspore_lite >= 2.9.0`
- Converter: `converter_lite`（与 wheel 版本一致）
- Ascend 环境：CANN + NPU 驱动，`npu-smi info` 可正常显示设备

## 快速开始

### 1. 导出 ONNX

```bash
python export_qwen3_vl_4b_instruct_onnx.py \
    --model-id ./Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./qwen3_vl_4b_instruct_onnx \
    --device cpu \
    --vision-image-size 128
```

导出产物在 `qwen3_vl_4b_instruct_onnx/` 下：

- `qwen3_vl_vision.onnx`
- `qwen3_vl_llm_prefill.onnx`
- `qwen3_vl_llm_decode.onnx`

### 2. ONNX 推理（CPU）

```bash
python infer_qwen3_vl_4b_instruct_onnx.py \
    --vision qwen3_vl_4b_instruct_onnx/qwen3_vl_vision.onnx \
    --prefill qwen3_vl_4b_instruct_onnx/qwen3_vl_llm_prefill.onnx \
    --decode qwen3_vl_4b_instruct_onnx/qwen3_vl_llm_decode.onnx \
    --processor ./Qwen/Qwen3-VL-4B-Instruct \
    --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --device cpu
```

### 3. ONNX → MindIR 转换（Ascend）

使用 MindSpore Lite 2.9.0 converter，必须加 `--optimize=ascend_oriented`：

```bash
export LD_LIBRARY_PATH=/data/chenyh/miniconda3/envs/ms-py311/lib:$LD_LIBRARY_PATH
CONVERTER=/data/chenyh/ms/2.9.0/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
ONNX_DIR=./qwen3_vl_4b_instruct_onnx

# 转换 Vision（权重内嵌，产物为 .mindir 单文件）
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/qwen3_vl_vision.onnx \
    --outputFile=$ONNX_DIR/qwen3_vl_vision \
    --optimize=ascend_oriented --saveType=MINDIR --configFile=config.ini

# 转换 Prefill（大模型权重外置，产物为 _graph.mindir + _variables/）
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/qwen3_vl_llm_prefill.onnx \
    --outputFile=$ONNX_DIR/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented --saveType=MINDIR  --configFile=config.ini

# 转换 Decode
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/qwen3_vl_llm_decode.onnx \
    --outputFile=$ONNX_DIR/qwen3_vl_llm_decode \
    --optimize=ascend_oriented --saveType=MINDIR --configFile=config.ini
```

config.ini 内容如下：

```text
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

转换产物：

- `qwen3_vl_vision.mindir`
- `qwen3_vl_llm_prefill_graph.mindir` + `qwen3_vl_llm_prefill_variables/`
- `qwen3_vl_llm_decode_graph.mindir` + `qwen3_vl_llm_decode_variables/`

### 4. MindSpore Lite Ascend 推理

```bash
python infer_qwen3_vl_4b_instruct_mslite.py \
    --vision-model ./qwen3_vl_4b_instruct_onnx/qwen3_vl_vision.mindir \
    --prefill-model ./qwen3_vl_4b_instruct_onnx/qwen3_vl_llm_prefill_graph.mindir \
    --decode-model ./qwen3_vl_4b_instruct_onnx/qwen3_vl_llm_decode_graph.mindir \
    --processor ./Qwen/Qwen3-VL-4B-Instruct \
    --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --image-size 128 \
    --device ascend \
    --device-id 0
```

## 5. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：Qwen3-VL-4b-instruct
测试条件：图片-文本 输入图片  https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
image size 128; max-new-tokens 128

输入图片会被resize成128x128, 输出128token

| 指标             | Time      |
|----------------|-----------|
| vision (ms)    | 10        |
| prefill (ms)   | 85        |
| decode (ms)    | 84        |
| **Total (ms)** | **10970** |

## 模型 I/O 说明

### Vision 模型

| | 名称 | dtype | 形状 |
|------|------|-------|------|
| 输入 | `pixel_values` | float32 | `(seq_len, 1536)` |
| 输出 | `image_embeds` | float32 | `(num_image_tokens, hidden_size)` |
| 输出 | `deepstack_embeds` | float32 | `(num_deepstack, num_image_tokens, hidden_size)` |

### LLM Prefill 模型

| | 名称 | dtype | 形状 |
|------|------|-------|------|
| 输入 | `input_ids` | int64 | `(batch, seq_len)` |
| 输入 | `attention_mask` | int64 | `(batch, seq_len)` |
| 输入 | `position_ids` | int64 | `(4, batch, seq_len)` |
| 输入 | `image_embeds` | float32 | `(num_image_tokens, hidden_size)` |
| 输入 | `deepstack_embeds` | float32 | `(num_deepstack, num_image_tokens, hidden_size)` |
| 输出 | `logits` | float32 | `(batch, seq_len, vocab_size)` |
| 输出 | `present_key_values` | float32 | `(2*num_layers, batch, num_kv_heads, seq_len, head_dim)` |

### LLM Decode 模型

| | 名称 | dtype | 形状 |
|------|------|-------|------|
| 输入 | `input_ids` | int64 | `(batch, 1)` |
| 输入 | `attention_mask` | int64 | `(batch, total_seq_len)` |
| 输入 | `position_ids` | int64 | `(4, batch, 1)` |
| 输入 | `past_key_values` | float32 | `(2*num_layers, batch, num_kv_heads, past_seq_len, head_dim)` |
| 输出 | `logits` | float32 | `(batch, 1, vocab_size)` |
| 输出 | `present_key_values` | float32 | `(2*num_layers, batch, num_kv_heads, total_seq_len, head_dim)` |

> **注意**：Ascend 推理时，MSLite 自动将 int64 输入转换为 int32，float32 保持不变。

## 模型规格

| 参数 | 4B |
|------|-----|
| hidden_size | 2560 |
| num_hidden_layers | 36 |
| num_attention_heads | 32 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| intermediate_size | 9728 |
| vocab_size | 151936 |
| vision_depth | 24 |
| vision_hidden_size | 1024 |

## 目录结构

```text
qwen3_vl_4b_instruct/
├── export_qwen3_vl_4b_instruct_onnx.py    # ONNX 导出脚本（3 段模型）
├── infer_qwen3_vl_4b_instruct_onnx.py     # ONNX Runtime CPU 推理脚本
├── infer_qwen3_vl_4b_instruct_mslite.py   # MindSpore Lite Ascend 推理脚本
├── README.md
└── qwen3_vl_4b_instruct_onnx/             # 导出模型目录
    ├── qwen3_vl_vision.onnx / .mindir
    ├── qwen3_vl_llm_prefill.onnx / _graph.mindir + _variables/
    └── qwen3_vl_llm_decode.onnx / _graph.mindir + _variables/
```

## 关键点

### Prefill / Decode 拆分

- **Prefill**：一次性处理完整 prompt（含图像 token）
- **Decode**：利用 KV cache 增量生成，避免每步重复计算历史 token

### Ascend 推理注意事项

- **int64 → int32**：MSLite 转换时自动将 ONNX 的 int64 输入转为 int32，推理脚本需传入 int32 数据
- **precision_mode**：2.9.0 必须用 `enforce_fp16`（不支持 `preferred_fp16`）
- **converter 产物命名**：`--outputFile=xxx` 产生 `xxx_graph.mindir`，部署时需用完整的 `_graph.mindir` 文件名

## 常见问题

### 导出时内存不足（OOM）

- 使用 `--device cpu` 导出
- 减小 `--vision-image-size`（默认 128）
- 修改 config.json 中 `num_hidden_layers` 和 `vision_config.depth` 为较小值做快速验证

### MindIR 转换失败

- 确保 converter 和 Python wheel 版本一致（2.9.0）
- 某些模型可能需要先 `onnxsim` 简化再转换
- 查看 converter 日志中的 `ERROR` 定位失败节点

### MSLite 推理报错

| 错误 | 解决 |
|------|------|
| `Input data type is wrong` | int64 改 int32 |
| `precision_mode` ValueError | `preferred_fp16` 改 `enforce_fp16` |
| `build_from_file failed` | 检查 mindir 文件路径和 `_variables/` 目录是否在同级目录 |

### image_embeds 长度不匹配

- 确保 processor 与模型版本一致
- 确认 `--image-size` 与导出时的 `--vision-image-size` 一致

## 参考链接

- [MindSpore Lite](https://www.mindspore.cn/lite/)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

## 许可证

本工具遵循 Qwen3-VL 模型的许可证要求，详见 [Qwen3-VL license](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。
