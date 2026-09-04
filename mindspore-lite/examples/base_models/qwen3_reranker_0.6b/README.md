# Qwen3-Reranker-0.6B ONNX 导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 Qwen3-Reranker-0.6B 导出为包含 Ascend 融合算子的 ONNX 模型，转换为 MindSpore Lite MindIR，并在 Ascend NPU 上完成推理、精度验证和性能测试。

## 目录内容

| 文件 / 目录 | 说明 |
|---|---|
| `export_qwen3_reranker_onnx.py` | ONNX 导出脚本，包含 PromptFlashAttention、RotaryMul、`slice_last` 和 `lm_head` 切片逻辑 |
| `infer_qwen3_reranker_mslite.py` | MindSpore Lite Ascend 推理脚本，支持动态分档、left padding 和计时 |
| `configs/qwen3_reranker.ini` | `converter_lite` 的 Ascend 转换配置和动态分档配置 |
| `configs/op_fp32.json` | RmsNorm 相关算子的 FP32 精度配置 |
| `README_final.md` | 本部署教程 |

> **注意**：最终 ONNX 包含 `PromptFlashAttention` 和 `RotaryMul` 等 Ascend Custom 算子，不能直接使用通用 ONNX Runtime 加载，必须先转换为 MindIR，再在 Ascend 上运行。

Qwen3-Reranker-0.6B 是一个 cross-encoder reranker。输入 Query 与 Document，模型输出 `yes/no` 二分类 logits，并通过 softmax 得到 `P(yes)` 作为相关性分数。

## 模型架构

模型为 28 层 Qwen3 decoder，使用 GQA（Grouped Query Attention）：

| 项目 | 值 |
|---|---:|
| 层数 | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `vocab_size` | 151669 |
| `tie_word_embeddings` | true |

模型导出配置包含以下优化：

- **PromptFlashAttention**：融合 QK、Softmax 和 V，避免显式构造完整的 `[seq_len, seq_len]` attention 矩阵。
- **RotaryMul**：融合 RoPE 旋转和 cos/sin 乘法。
- **`slice_last`**：只保留最后一个 token 的 hidden state，输出从 `[batch, seq_len, vocab]` 缩减为 `[batch, 1, vocab]`。
- **`lm_head` 切片**：只保留 `yes/no` 两行权重，最终输出为 `[batch, 1, 2]`。
- **RmsNorm 不做 Custom 融合**：保留原始计算，并通过 FP32 mixlist 保证精度稳定性。

---

## 1. 环境准备

### 1.1 软件版本

以下版本为本次本地验证实际使用的版本：

| 软件包 | 版本 |
|---|---|
| Python | 3.11.15 |
| torch | 2.8.0+cpu |
| transformers | 5.4.0 |
| onnx | 1.19.1 |
| onnxruntime | 1.24.2 |
| numpy | 2.4.6 |
| CANN | 9.0.1 |
| mindspore-lite | 2.11.0 |

加载 CANN 环境：

```bash
source <CANN_INSTALL_PATH>/set_env.sh
```

其中 `<CANN_INSTALL_PATH>` 需要替换为用户本机路径，不要将机器相关绝对路径提交到仓库。

### 1.2 Python 依赖

```bash
pip install -U \
  torch==2.8.0 \
  transformers==5.4.0 \
  onnx==1.19.1 \
  onnxruntime==1.24.2 \
  numpy==2.4.6
```

MindSpore Lite 需要根据目标架构安装对应发行包。本次验证使用 `mindspore-lite 2.11.0`。

### 1.3 模型权重

可以从 Hugging Face 下载模型：

```bash
pip install -U modelscope
modelscope download \
  --model Qwen/Qwen3-Reranker-0.6B \
  --local_dir ./Qwen3-Reranker-0.6B
```

也可以将已经下载好的权重放在其他目录，后续通过 `--model-id` 参数传入。本文命令统一假设权重目录为：

```text
./Qwen3-Reranker-0.6B
```

---

## 2. 模型导出 ONNX

### 2.1 导出命令

在本目录执行：

```bash
python3 export_qwen3_reranker_onnx.py \
  --model-id ./Qwen3-Reranker-0.6B \
  --output-dir ./onnx \
  --max-length 8192 \
  --device cpu \
  --output-name qwen3_reranker_0.6b.onnx
```

### 2.2 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | Hugging Face 模型 ID 或本地权重目录 | `Qwen/Qwen3-Reranker-0.6B` |
| `--output-dir` | ONNX 输出目录 | `./onnx` |
| `--max-length` | 导出支持的最大序列长度 | `8192` |
| `--device` | 导出设备，通常使用 `cpu` | `cpu` |
| `--output-name` | ONNX 文件名 | 脚本默认值 |
| `--no-slice-last` | 关闭最后 token 裁剪 | 默认关闭 |
| `--no-slice-lm-head` | 保留完整词表 lm_head | 默认关闭 |

最终配置不要添加 `--no-slice-last` 或 `--no-slice-lm-head`。

### 2.3 导出产物

```text
onnx/
├── qwen3_reranker_0.6b.onnx
├── qwen3_reranker_0.6b.onnx.data
└── ...                                  # 可能包含外部权重文件
```

大于 2GB 的模型权重会以 ONNX external data 形式保存，`.onnx` 和 `.onnx.data` 必须同时保留。

最终融合模型的关键图特征：

- 28 个 `PromptFlashAttention` Custom 节点。
- 56 个 `RotaryMul` Custom 节点。
- 不包含 attention `Softmax` 节点。
- 不包含导出器产生的 `IsNaN` 后处理节点。
- 默认输出 shape 为 `[batch, 1, 2]`，两列顺序为 `[yes, no]`。

---

## 3. ONNX 转 MindIR

最终融合 ONNX 含有 Ascend Custom 算子，不能直接使用通用 ONNX Runtime 推理。应先转换为 MindIR，再使用 MindSpore Lite Ascend 后端运行。

### 3.1 转换命令

将 MindSpore Lite 工具包中的 `converter_lite` 设置为环境变量。下面的路径是相对写法，请替换为本机 MindSpore Lite 安装目录：

```bash
Convert=<MINDSPORE_LITE_ROOT>/tools/converter/converter/converter_lite
```

执行转换：

```bash
$Convert \
  --fmk=ONNX \
  --modelFile=./onnx/qwen3_reranker_0.6b.onnx \
  --outputFile=./onnx/qwen3_reranker_0.6b \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_reranker.ini
```

成功日志：

```text
CONVERT RESULT SUCCESS:0
```

### 3.2 `qwen3_reranker.ini` 配置

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,-1;attention_mask:1,-1"
ge.dynamicDims="128,128;256,256;512,512;768,768;1024,1024;1280,1280;1536,1536;2048,2048;3072,3072;4096,4096;8192,8192"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./configs/op_fp32.json"

[ascend_context]
plugin_custom_ops=All
```

配置说明：

- `input_ids` 和 `attention_mask` 的 batch 固定为 1，序列长度动态。
- 每个 `ge.dynamicDims` 档位包含两个值，分别对应两个输入的动态序列维度。
- `plugin_custom_ops=All` 用于加载和转换 Ascend Custom 算子。
- `allow_mix_precision` 允许非关键路径使用混合精度。
- `op_fp32.json` 将 RmsNorm 链路关键算子保留为 FP32。
- 转换时必须从包含 `configs/` 的项目目录执行，保证 `op_fp32.json` 相对路径可解析。

### 3.3 转换产物

```text
onnx/
├── qwen3_reranker_0.6b_graph.mindir
└── qwen3_reranker_0.6b_variables/
    └── data_0
```

超过 2GB 的权重会放在 `*_variables/data_0` 中，运行时必须保留 `*_graph.mindir` 与对应的 variables 目录。

---

## 4. MindSpore Lite 推理

### 4.1 推理命令

```bash
python3 infer_qwen3_reranker_mslite.py \
  --model ./onnx/qwen3_reranker_0.6b_graph.mindir \
  --tokenizer ./Qwen3-Reranker-0.6B \
  --max-length 8192 \
  --device ascend \
  --device-id 0 \
  --warmup 1
```

### 4.2 推理逻辑

输入使用 left padding：

1. 将 Query、Document 和指令按照 reranker 对话模板拼接。
2. tokenize 后选择不小于真实长度的最小动态档位。
3. 将 `input_ids` 和 `attention_mask` left pad 到该档位。
4. 调用 `model.resize` 设置两个输入的完整 shape。
5. 在 Ascend 上执行 `predict`。
6. 读取两列 `[yes, no]` logits，softmax 后得到 `P(yes)`。

当前模型输入输出：

| 方向 | 名称 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| 输入 | `input_ids` | `(1, seq_len)` | int32（MSLite） | token IDs |
| 输入 | `attention_mask` | `(1, seq_len)` | int32（MSLite） | left padding mask |
| 输出 | `logits` | `(1, 1, 2)` | float32 | `[yes, no]` logits |

### 4.3 Benchmark 单档位测试

MindSpore Lite Benchmark 可以直接验证某个动态档位：

```bash
Benchmark=<MINDSPORE_LITE_ROOT>/tools/benchmark/benchmark

$Benchmark \
  --modelFile=./onnx/qwen3_reranker_0.6b_graph.mindir \
  --device=Ascend \
  --inputShape='input_ids:1,1024;attention_mask:1,1024'
```

例如验证 1280 档位：

```bash
$Benchmark \
  --modelFile=./onnx/qwen3_reranker_0.6b_graph.mindir \
  --device=Ascend \
  --inputShape='input_ids:1,1280;attention_mask:1,1280'
```

注意：当前模型只有 `input_ids` 和 `attention_mask` 两个输入，不需要传入 `position_ids` 或 `past_key_values`。

---

## 5. 推理示例输出与性能数据

### 5.1 推理示例输出

```text
Initializing MindSpore Lite context for ascend...
Loading model from ./onnx/qwen3_reranker_0.6b_graph.mindir...
Loading tokenizer from ./Qwen3-Reranker-0.6B...
Token IDs - yes: 9693, no: 2152

[1] Score: 0.6187
Query: What is the capital of China?
Document: The capital of China is Beijing.

[2] Score: 0.5742
Query: Explain gravity
Document: Gravity is a force that attracts two bodies towards each other...

=== 端到端推理性能（本次运行） ===
| 指标 | 耗时 (ms) |
|---|---:|
| Tokenize + pad | 0.48 |
| Bucket 选择 | 0.00 |
| Model predict (单条平均 13.66 ms × 2) | 27.32 |
| Postprocess | 0.13 |
| **总耗时** | **28.53** |
```

### 5.2 性能数据(300 IDUO)

本次六档验证测得的单次 MindSpore Lite `predict` 平均耗时如下：

| bucket | 平均耗时 |
|---:|---:|
| 128 | 13.66 ms |
| 256 | 20.24 ms |
| 512 | 36.96 ms |
| 768 | 60.10 ms |
| 1024 | 90.46 ms |
| 1280 | 127.63 ms |

> 不同测试脚本、warmup 次数、执行轮数和设备状态会造成小幅波动。提交 PR 时应优先引用同一命令、同一设备、相同 warmup 和 rounds 的结果。

---

## 6. 常见问题

### Q1：为什么最终融合 ONNX 不能用 ONNX Runtime 加载？

因为图中包含 Ascend Custom 算子。请先使用 `converter_lite --optimize=ascend_oriented` 转换为 MindIR，再使用 MindSpore Lite Ascend 后端推理。

### Q2：为什么转换时要指定 `--configFile`？

配置文件声明动态档位、Custom 算子加载方式和 RmsNorm FP32 mixlist。缺少配置文件可能导致动态 shape 不匹配、Custom 算子无法识别或精度变化。

### Q3：`ge.dynamicDims` 为什么每档需要两个数？

`input_shape` 中有两个动态维度：`input_ids` 的序列长度和 `attention_mask` 的序列长度。因此每个档位必须同时填写两个值，例如 `1024,1024`。

### Q4：为什么 MindSpore Lite 推理需要 `resize`？

动态档位只是转换时允许的 shape 集合，不会自动完成 padding。推理侧必须先将输入补齐到目标 bucket，再使用 `model.resize` 设置完整输入 shape。

### Q5：为什么模型文件有 `*_graph.mindir` 和 `*_variables/`？

大模型权重超过 protobuf 单文件限制时，MindSpore Lite 会将图结构与权重数据分开保存。两者必须同时保留，加载时传入 `*_graph.mindir`。

---

## 7. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [MindSpore Lite Ascend 文档](https://www.mindspore.cn/lite/docs/zh-CN/master/use/ascend_info.html)
- [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 8. 许可证

本教程和示例代码遵循 Apache License 2.0。Qwen3-Reranker-0.6B 模型及权重遵循其官方仓库发布的模型许可证，使用前请阅读并遵守相应许可条款。
