# Qwen3-VL-4B-Instruct ONNX 导出与推理

本目录提供 Qwen3-VL-4B-Instruct 导出为 ONNX、端到端推理、以及 MindSpore Lite Ascend 部署的完整脚本。模型被拆分为 Vision、LLM Prefill、LLM Decode 三个组件。

本demo中 Vision 模块为固定shape 128x128的输入图片， prefill和decode模块为动态shape。

## 架构拆分

1. **Vision Tower**（`vision/qwen3_vl_vision.onnx`）：对图像进行编码，输出视觉特征
2. **LLM Prefill**（`prefill/qwen3_vl_llm_prefill.onnx`）：一次性处理完整 prompt（文本 + 图像 token），输出 logits 与 KV cache
3. **LLM Decode**（`decode/qwen3_vl_llm_decode.onnx`）：基于固定长度 KV cache 做自回归增量生成，通过 scatter 更新

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

### 1. 导出 ONNX（opset 17）

```bash
python export_qwen3_vl_4b_instruct_onnx.py \
    --model-id ../Qwen/Qwen3-VL-4B-Instruct \
    --output-dir ./qwen3_vl_4b_instruct_onnx \
    --device cpu \
    --vision-image-size 128
```

导出产物：

```text
qwen3_vl_4b_instruct_onnx/
├── vision/qwen3_vl_vision.onnx
├── prefill/qwen3_vl_llm_prefill.onnx
└── decode/qwen3_vl_llm_decode.onnx
```

### 2. ONNX 推理（CPU）

```bash
python infer_qwen3_vl_4b_instruct_onnx.py \
    --vision qwen3_vl_4b_instruct_onnx/vision/qwen3_vl_vision.onnx \
    --prefill qwen3_vl_4b_instruct_onnx/prefill/qwen3_vl_llm_prefill.onnx \
    --decode qwen3_vl_4b_instruct_onnx/decode/qwen3_vl_llm_decode.onnx \
    --processor ../Qwen/Qwen3-VL-4B-Instruct \
    --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --device cpu
```

### 3. ONNX → MindIR 转换（Ascend）

使用 MindSpore Lite 2.9.0 converter，加 `--optimize=ascend_oriented`：

```bash
export LD_LIBRARY_PATH=/data/chenyh/miniconda3/envs/ms-py311/lib:$LD_LIBRARY_PATH
CONVERTER=/data/chenyh/ms/2.9.0/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
ONNX_DIR=./qwen3_vl_4b_instruct_onnx

# Vision（权重内嵌，产物为 .mindir 单文件）
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/vision/qwen3_vl_vision.onnx \
    --outputFile=$ONNX_DIR/vision/qwen3_vl_vision \
    --optimize=ascend_oriented --configFile=./configs/config.ini --saveType=MINDIR

# Prefill（大模型权重外置，产物为 _graph.mindir + _variables/）
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/prefill/qwen3_vl_llm_prefill.onnx \
    --outputFile=$ONNX_DIR/prefill/qwen3_vl_llm_prefill \
    --optimize=ascend_oriented --configFile=./configs/config.ini --saveType=MINDIR

# Decode
$CONVERTER --fmk=ONNX --modelFile=$ONNX_DIR/decode/qwen3_vl_llm_decode.onnx \
    --outputFile=$ONNX_DIR/decode/qwen3_vl_llm_decode \
    --optimize=ascend_oriented --configFile=./configs/config.ini --saveType=MINDIR
```

> **config.ini 内容**（`configs/config.ini`）：
> ```ini
> [acl_init_options]
> ge.exec.precision_mode = force_fp32
> ```

转换产物：

- `vision/qwen3_vl_vision.mindir`
- `prefill/qwen3_vl_llm_prefill_graph.mindir` + `prefill/qwen3_vl_llm_prefill_variables/`
- `decode/qwen3_vl_llm_decode_graph.mindir` + `decode/qwen3_vl_llm_decode_variables/`

### 4. MindSpore Lite Ascend 推理

```bash
python infer_qwen3_vl_4b_instruct_mslite.py \
    --vision-model ./qwen3_vl_4b_instruct_onnx/vision/qwen3_vl_vision.mindir \
    --prefill-model ./qwen3_vl_4b_instruct_onnx/prefill/qwen3_vl_llm_prefill_graph.mindir \
    --decode-model ./qwen3_vl_4b_instruct_onnx/decode/qwen3_vl_llm_decode_graph.mindir \
    --processor ../Qwen/Qwen3-VL-4B-Instruct \
    --image https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --image-size 128 \
    --device ascend \
    --device-id 0
```

### 外部资源说明

- README 示例中使用 Qwen 官方 demo 图片：`https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg`。
- 该 URL 仅用于示例推理和性能测试，运行时会从网络读取图片；离线环境或网络受限环境请显式传入本地图片路径。
- 权重和 processor 路径通过 `--model-id` / `--processor` 参数传入，推理/导出代码未硬编码权重或图片下载 URL。

## 5. 性能数据

### 性能测试结果（Atlas 800I A2）

测试模型：Qwen3-VL-4b-instruct
测试条件：图片-文本 输入图片  https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg
image size 128; max-new-tokens 128

输入图片会被resize成128x128, 输出128token

| 指标             | Time        |
|----------------|-------------|
| vision (ms)    | 8           |
| prefill (ms)   | 69.64       |
| decode (ms)    | 14.392      |
| **Total (ms)** | **2139.95** |

## 模型 I/O 说明

### Vision 模型（固定 shape，128×128 图像）

| | 名称 | dtype | 形状 |
|------|------|-------|------|
| 输入 | `pixel_values` | float16 | `(64, 1536)` |
| 输出 | `image_embeds` | float16 | `(16, 2560)` |
| 输出 | `deepstack_embeds` | float16 | `(3, 16, 2560)` |

### LLM Prefill 模型（动态 batch/seq_len）

| | 名称 | dtype | 形状（以 batch=1, seq_len=30 为例） |
|------|------|-------|------|
| 输入 | `input_ids` | int64 | `(batch, seq_len)` |
| 输入 | `attention_mask` | int64 | `(batch, seq_len)` |
| 输入 | `position_ids` | int64 | `(4, batch, seq_len)` |
| 输入 | `image_embeds` | float16 | `(num_image_tokens, 2560)` |
| 输入 | `deepstack_embeds` | float16 | `(3, num_image_tokens, 2560)` |
| 输出 | `logits` | float16 | `(batch, seq_len, 151936)` |
| 输出 | `present_key_values` | float16 | `(72, batch, 8, seq_len, 128)` |

### LLM Decode 模型（固定 KV cache 512）

| | 名称 | dtype | 形状 |
|------|------|-------|------|
| 输入 | `input_ids` | int64 | `(1, 1)` |
| 输入 | `attention_mask` | int64 | `(1, 512)` |
| 输入 | `position_ids` | int64 | `(4, 1, 1)` |
| 输入 | `past_key_values` | float16 | `(72, 1, 8, 512, 128)` |
| 输入 | `cache_pos` | int64 | `(1,)` |
| 输出 | `logits` | float16 | `(1, 1, 151936)` |
| 输出 | `present_key_values` | float16 | `(72, 1, 8, 512, 128)` |

> **注意**：Ascend 推理时，MSLite 自动将 int64 输入转换为 int32，float16 保持不变。

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
├── export_qwen3_vl_4b_instruct_onnx.py    # ONNX 导出脚本（3 段模型 + Custom 算子）
├── infer_qwen3_vl_4b_instruct_onnx.py     # ONNX Runtime CPU 推理脚本
├── infer_qwen3_vl_4b_instruct_mslite.py   # MindSpore Lite Ascend 推理脚本
├── configs/
│   └── config.ini                         # converter 配置文件（force_fp32）
├── README.md
└── qwen3_vl_4b_instruct_onnx/
    ├── vision/
    │   └── qwen3_vl_vision.onnx / .mindir
    ├── prefill/
    │   └── qwen3_vl_llm_prefill.onnx / _graph.mindir + _variables/
    └── decode/
        └── qwen3_vl_llm_decode.onnx / _graph.mindir + _variables/
```

## 关键点

### Custom 融合算子

导出脚本默认启用 CANN 融合算子（`_USE_CUSTOM_OP = True`），包括：

- **RMSNorm**：替换 LayerNorm
- **RotaryMul**：RoPE 旋转位置编码
- **SwiGlu**：SwiGLU 激活函数
- **PromptFlashAttention**：Prefill 阶段 FlashAttention
- **IncreFlashAttention**：Decode 阶段增量 FlashAttention
- **Scatter**：KV cache 定点更新

可通过 `--no-custom-op` 禁用，回退到标准 ONNX 算子。

### Decode 固定 KV Cache

Decode 模型使用固定长度 KV cache（默认 512）。Prefill 输出被 pad 到 512 长度，decode 阶段通过 scatter 方式在指定位置更新 KV cache，避免 `Concat` 导致的动态 shape 变化。

### Ascend 推理注意事项

- **int64 → int32**：推理脚本传入 int32 数据
- **converter 产物命名**：`--outputFile=xxx` 产生 `xxx_graph.mindir` + `xxx_variables/`
- **config.ini**：2B 模型可用 `force_fp32`，4B 模型（36 层）内存过大建议去掉 config

## 常见问题

### 导出时内存不足（OOM）

- 使用 `--device cpu` 导出
- 减小 `--vision-image-size`（默认 128）
- 修改 config.json 中 `num_hidden_layers` 和 `vision_config.depth` 为较小值做快速验证

### MindIR 转换失败

- 确保 converter 和 Python wheel 版本一致（2.9.0）
- vision, prefill, decode 导出的onnx路径需要分开。不然外置权重文件可能重名，导致转换失败
- 查看 converter 日志中的 `ERROR` 定位失败节点

### MSLite 推理报错

| 错误 | 解决 |
|------|------|
| `Input data type is wrong` | int64 改 int32 |
| `build_from_file failed` | 检查 mindir 路径和 `_variables/` 目录 |

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
