# Qwen3-1.7B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 `Qwen3-1.7B` 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并完成端到端推理与精度对齐验证。

---

## 1. 环境准备

### 系统要求

- Python 3.11
- Linux 系统（推荐 Ubuntu 22.04）
- 昇腾环境（用于 MindIR 推理，需安装 MindSpore Lite 与 Ascend 驱动）

### 依赖版本（建议）

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.8.0 |
| transformers   | 4.51.0 |
| onnx           | 1.21.0 |
| onnxruntime    | 1.24.0 |
| mindspore-lite | 2.8.0 |
| CANN           | 8.5.0 |

### 安装命令

```bash
pip install torch==2.8.0 transformers==4.51.0 onnx==1.21.0 onnxruntime==1.24.0 mindspore-lite==2.8.0
```

### 验证安装

```bash
python -c "import torch, transformers, onnx, onnxruntime, mindspore_lite; print('All dependencies installed successfully!')"
```

---

## 2. 模型导出 ONNX

### 导出脚本说明

导出脚本会将 Qwen3-1.7B 拆分为两个 ONNX 子图，并支持 PTQ（Post-Training Quantization）静态 int8 量化与 SmoothQuant 算法：

1. **LLM Prefill** (`qwen3_1_7b_llm_prefill.onnx`)：处理输入 prompt，输出 `logits`、`present_key_cache`、`present_value_cache`
2. **LLM Decode** (`qwen3_1_7b_llm_decode_ptq_int8.onnx`)：单 token 递归生成，输入 `past_key_cache`、`past_value_cache`，输出更新后的 cache。默认启用 INT8 量化（Weight Only + Activation Quantization + SmoothQuant），以精度换性能。可通过 `--disable-torch-ptq-int8` 关闭量化，输出文件名为 `qwen3_1_7b_llm_decode.onnx`。

### 导出命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_1.7b

# 默认导出（PTQ INT8 量化 + SmoothQuant + FP32 权重）
python export_qwen3_1_7b_onnx.py \
  --model-id ./Qwen3-1.7B \
  --output-dir ./qwen3_1_7b_onnx \
  --device cpu

# 关闭量化，导出纯 FP32 模型
python export_qwen3_1_7b_onnx.py \
  --model-id ./Qwen3-1.7B \
  --output-dir ./qwen3_1_7b_onnx \
  --device cpu \
  --disable-torch-ptq-int8

# 使用自定义校准数据导出
python export_qwen3_1_7b_onnx.py \
  --model-id ./Qwen3-1.7B \
  --output-dir ./qwen3_1_7b_onnx \
  --device cpu \
  --torch-ptq-calib-jsonl ./calib.jsonl \
  --torch-ptq-max-samples 32 \
  --smooth-alpha 0.65

# 导出 FP32（关闭量化，降低数值误差）
python export_qwen3_1_7b_onnx.py \
  --model-id ./Qwen3-1.7B \
  --output-dir ./qwen3_1_7b_onnx \
  --device cpu \
  --dtype fp32 \
  --disable-torch-ptq-int8
```

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3-1.7B` |
| `--output-dir` | 导出输出目录 | `./qwen3_1_7b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dummy-seq-len` | 导出用 dummy 序列长度 | `8` |
| `--kv-cache-len` | KV cache 固定长度（prefill 输出与 decode 输入） | `512` |
| `--dtype` | 导出精度（fp16/bf16/fp32） | `fp32` |
| `--use-dynamo` | 启用新 ONNX dynamo 导出路径 | `False` |
| `--disable-torch-ptq-int8` | 关闭 PTQ int8 量化（默认启用量化，加上此标志则关闭） | `False`（不关闭） |
| `--torch-ptq-calib-jsonl` | 校准数据 JSONL 文件路径 | `calib.jsonl` |
| `--torch-ptq-max-samples` | 最大校准样本数 | `32` |
| `--torch-ptq-max-decode-steps` | 每样本最大 decode 步数 | `32` |
| `--smooth-alpha` | SmoothQuant alpha 系数（0.0-1.0，越小激活值越平滑） | `0.65` |
| `--weight-clip-ratio` | 量化前裁剪权重离群值比例（如 0.01 = 裁剪 top 1%） | `0.0` |

说明：

- Prefill 模型输入为动态长度（seq_len 动态），用于处理 prompt。
- Decode 模型为固定 shape（单 token + 固定 KV cache length），用于逐 token 生成。
- **PTQ Int8 量化（默认启用）**：通过校准数据统计激活值范围，对权重和激活值做对称 int8 量化，并可选使用 SmoothQuant 算法平滑激活值，降低量化精度损失。加上 `--disable-torch-ptq-int8` 标志可**关闭**量化。
- 校准数据可由 `infer_qwen3_1_7b_mslite.py --dump-calib` 导出为 JSONL 格式。

### 导出输出

导出目录采用分目录结构，避免 external data 命名冲突：

```text
qwen3_1_7b_onnx/
├── prefill/
│   ├── qwen3_1_7b_llm_prefill.onnx
│   └── onnx__* / model.* (external data)
└── decode/
    ├── qwen3_1_7b_llm_decode_ptq_int8.onnx   （量化启用时）
    ├── qwen3_1_7b_llm_decode.onnx              （关闭量化时，--disable-torch-ptq-int8）
    └── onnx__* / model.* (external data)
```

## 4. MindSpore Lite 转换

### 转换命令

使用 `converter_lite` 将 ONNX 转换为 MindIR。建议为 prefill/decode 分别提供 config 文件声明动态 shape。

```bash
cd ./mindspore-lite/examples/base_models/qwen3_1.7b

# Prefill
./converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_1_7b_onnx/prefill/qwen3_1_7b_llm_prefill.onnx \
  --outputFile=./qwen3_1_7b_onnx/prefill/qwen3_1_7b_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_1_7b_llm_prefill.config

# Decode（量化启用时，使用 ptq_int8 版本）
export KEEP_ORIGIN_DTYPE=1
./converter_lite \
  --fmk=ONNX \
  --modelFile=./qwen3_1_7b_onnx/decode/qwen3_1_7b_llm_decode_ptq_int8.onnx \
  --outputFile=./qwen3_1_7b_onnx/decode/qwen3_1_7b_llm_decode_ptq_int8 \
  --inputShape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,8,512,128;past_value_cache:28,1,8,512,128" \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_1_7b_llm_decode.config
```

### config 文件示例

#### `./configs/qwen3_1_7b_llm_prefill.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1"
ge.dynamicDims="10,10,10;20,20,20"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

#### `./configs/qwen3_1_7b_llm_decode.config`

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,512;position_ids:1,1;past_key_cache:28,1,8,512,128;past_value_cache:28,1,8,512,128"

[acl_init_options]
ge.exec.precision_mode=force_fp32

[ascend_context]
plugin_custom_ops=All
```

> 注：
> - prefill 模型在转换为 MindIR 时会转换为动态分档（dynamicDims），需要在 config 中配置 `ge.dynamicDims`。
> - `ge.dynamicDims` 的每个分号分隔项，对应一次“动态分档”的实际值；数值个数需与 `input_shape` 中 `-1` 的数量一致。
> - decode 的 cache 已拆分为 `past_key_cache`/`past_value_cache` 两个输入。

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
cd ./mindspore-lite/examples/base_models/qwen3_1.7b

python infer_qwen3_1_7b_mslite.py \
  --prefill-model ./qwen3_1_7b_onnx/prefill/qwen3_1_7b_llm_prefill_graph.mindir \
  --decode-model ./qwen3_1_7b_onnx/decode/qwen3_1_7b_llm_decode_ptq_int8_graph.mindir \
  --tokenizer ./Qwen3-1.7B \
  --prompt "你好，请用一句话介绍 MindSpore Lite。" \
  --max-new-tokens 512 \
  --device ascend \
  --device-id 0
```

### 参数说明

脚本为适配 prefill 的动态分档，会在推理前对输入序列做 padding，使其落入 `ge.dynamicDims` 配置的挡位，推理结束后再按真实长度截断输出。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--prefill-model` | Prefill MindIR 路径 | 必填 |
| `--decode-model` | Decode MindIR 路径 | 必填 |
| `--tokenizer` | tokenizer 路径 | `./Qwen3-1.7B` |
| `--prompt` | 输入提示词 | `"你好，请介绍一下你自己。"` |
| `--max-new-tokens` | 最大生成 token 数 | `512` |
| `--max-length` | 最大序列长度 | `4096` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | 昇腾设备 ID | `0` |

---

## 6. 性能数据

### 性能测试结果

测试模型：Qwen3-1.7B
测试条件：默认 PTQ INT8 量化 Decode + FP16 Prefill，输入约 128 tokens，输出约 128 tokens
测试环境：CANN 8.5.0，MindSpore Lite 2.8.0

| 指标                       | **PTQ INT8 (300I Duo)** | **非量化 FP32 (300I Duo)** | **PTQ INT8 (800I A2)** | **非量化 FP32 (800I A2)** |
|--------------------------|----------------------|------------------------|----------------------|------------------------|
| Prefill (ms)             | 38.99                | 38.76                  | 14.36                | 14.28                  |
| Total Decode (ms)        | **2250.83**          | **3504.32**            | 766.30               | 1065.59                |
| **Avg decode step (ms)** | **17.72**            | **27.59**              | **6.03**             | **8.37**               |
| Total (ms)               | **2289.82**          | **3543.07**            | 780.66                  | **1079.87**            |
| **Throughput (tok/s)**   | **56.42**            | **36.24**              | **165.73**           | **119.18**             |

> 注意：Avg decode step 为单次 decode 推理的耗时。Prefill 使用非量化 FP32 模型。

---

## 7. 常见问题

### 1) `apply_chat_template` 返回类型不一致

不同 tokenizer 版本可能返回 `dict` 或 `ndarray`，推理脚本已兼容两种返回格式。

### 2) ONNX Runtime 报 external data 越界

通常由多个 ONNX 导出到同一目录导致 external data 文件重名冲突。请使用本教程的 `prefill/`、`decode/` 分目录导出方式。

### 3) 输出“被截断”

可能由以下原因导致：

- `--max-new-tokens` 达到上限
- 提前生成 `eos_token`

### 4) 精度不达标

建议逐项尝试：

- 使用 `--dtype fp32` 导出
- 使用 `--torch-dtype fp32` 做对齐评估
- 检查 tokenizer 与权重目录是否一致
- 固定相同 prompt 与 decode step 后再对比

---

## 8. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-1.7B 模型页](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 9. 许可证

本教程遵循 Qwen3-1.7B 模型及相关依赖的许可证要求。
