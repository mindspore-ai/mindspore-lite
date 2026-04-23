# Qwen2-0.5B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen2-0.5B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.8.0  |
| transformers   | 5.5.4  |
| onnx           | 1.21.0 |z
| onnxruntime    | 1.24.4 |
| CANN           | 8.5.0  |
| mindspore-lite | 2.8.0  |

```bash
pip install transformers==5.5.4 torch==2.8.0 onnx==1.21.0 onnxruntime==1.24.4 mindspore-lite
```

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd examples/qwen2_0.5b

python export_qwen2_onnx.py \
  --model-id models \
  --output-dir ./qwen2_onnx \
  --device cpu
```

### 参数说明

| 参数             | 说明                    | 默认值            |
|----------------|-----------------------|----------------|
| `--model-id`   | HuggingFace 模型路径或本地目录 | `models`       |
| `--output-dir` | 输出目录                  | `./qwen2_onnx` |
| `--device`     | 导出设备（cpu/cuda）        | `cpu`          |

### 产出

```log
qwen2_onnx/
├── qwen2_llm_prefill.onnx     # Prefill 模型 (~952MB)
└── qwen2_llm_decode.onnx     # Decode 模型 (~952MB)
```

---

## 3. ONNX 推理

### PyTorch 推理验证

```bash
python infer_qwen2_torch.py \
  --model-id models \
  --prompt "Hello, how are you?" \
  --max-new-tokens 64
```

**执行日志：**

```log
Loading PyTorch model from models...
Loading tokenizer from models...
Prompt: Hello, how are you?
Max new tokens: 64
==================================================
Generated text:
Based on the context of the text, it seems that the user is asking about the user's health and well-being.
==================================================
```

### ONNX Runtime 推理

```bash
python infer_qwen2_onnx.py \
  --prefill qwen2_onnx/qwen2_llm_prefill.onnx \
  --decode qwen2_onnx/qwen2_llm_decode.onnx \
  --tokenizer models \
  --prompt "Hello, how are you?" \
  --max-new-tokens 64
```

**执行日志：**

```log
Loading prefill ONNX from qwen2_onnx/qwen2_llm_prefill.onnx...
Loading decode ONNX from qwen2_onnx/qwen2_llm_decode.onnx...
Loading tokenizer from models...
Prompt: Hello, how are you?
Max new tokens: 64
==================================================
Generated text:
Based on the context of the text, it seems that the user is asking about the user's health and well-being.
==================================================
```

### 参数说明

| 参数                 | 说明                       | 默认值                     |
|--------------------|--------------------------|-------------------------|
| `--prefill`        | Prefill ONNX 模型路径        | 必填                      |
| `--decode`         | Decode ONNX 模型路径         | 必填                      |
| `--tokenizer`      | HuggingFace tokenizer 路径 | `models`                |
| `--prompt`         | 输入文本                     | `"Hello, how are you?"` |
| `--max-new-tokens` | 最大生成 token 数             | `128`                   |
| `--device`         | 推理设备（cpu/cuda）           | `cpu`                   |

---

## 4. MindSpore Lite 转换

### 转换命令

```bash

Converter=mindspore-lite-2.8.0-linux-aarch64/tools/converter/converter/converter_lite

# Prefill 转换
$Converter --fmk=ONNX \
  --modelFile=qwen2_onnx/qwen2_llm_prefill.onnx \
  --outputFile=qwen2_onnx/qwen2_llm_prefill \
  --optimize=ascend_oriented \
  --configFile=utils/config.ini

# Decode 转换
$Converter --fmk=ONNX \
  --modelFile=qwen2_onnx/qwen2_llm_decode.onnx \
  --outputFile=qwen2_onnx/qwen2_llm_decode \
  --optimize=ascend_oriented \
  --configFile=utils/config.ini
```

### 参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--configFile` | 配置文件路径                      |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |

### 配置文件

`utils/config.ini`:

```ini
[acl_init_options]
ge.exec.precision_mode=force_fp32
```

### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```log
qwen2_onnx/
├── qwen2_llm_prefill_graph.mindir      # Prefill MindIR 图定义 (~1.9KB)
├── qwen2_llm_prefill_variables/data_0   # Prefill 权重数据 (~2.2GB)
├── qwen2_llm_decode_graph.mindir        # Decode MindIR 图定义 (~1.9KB)
└── qwen2_llm_decode_variables/data_0     # Decode 权重数据 (~2.2GB)
```

---

## 5. MindSpore Lite 推理

### 推理命令

```bash
python infer_qwen2_mindir.py \
  --prefill-model qwen2_onnx/qwen2_llm_prefill_graph.mindir \
  --decode-model qwen2_onnx/qwen2_llm_decode_graph.mindir \
  --tokenizer models \
  --prompt "Hello, how are you?" \
  --max-new-tokens 64 \
  --device ascend
```

**执行日志：**

```log
Loading prefill model from qwen2_onnx/qwen2_llm_prefill_graph.mindir...
Loading decode model from qwen2_onnx/qwen2_llm_decode_graph.mindir...
Loading tokenizer from models...
============================================================
Input Prompt: Hello, how are you?
============================================================
============================================================
Generated Response: Based on the context of the text, it seems that the user is asking about the user's health and well-being.
============================================================
```

### 参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|-------------------------|
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                      |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填                      |
| `--tokenizer`      | HuggingFace tokenizer 路径              | `models`                |
| `--prompt`         | 输入文本                                  | `"Hello, how are you?"` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                   |
| `--device`         | 推理设备（ascend/cpu）                      | `ascend`                |

---

## 6. 性能数据

### 性能测试结果（300IDUO）

测试模型：Qwen2-0.5B
测试条件：输入 128 tokens，输出 128 tokens，chat 模板启用

| 指标                       | Mean       | Min       | Max       |
|--------------------------|------------|-----------|-----------|
| Prefill (ms)             | 39.76      | 32.08     | 63.25     |
| Total Decode (ms)        | 628.65     | 578.41    | 800.81    |
| **Avg decode step (ms)** | **26.19**  | **24.10** | **33.37** |
| Total (ms)               | 668.41     | -         | -         |
| **Throughput (tok/s)**   | **191.50** | -         | -         |

> 注意：首次推理 Prefill 时间较长（63ms）为正常现象，后续推理稳定在 ~32ms。Avg decode step 为单次 decode 推理的耗时。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen2-0.5B 官方文档](https://huggingface.co/Qwen/Qwen2-0.5B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程遵循 Qwen2-0.5B 模型的许可证。