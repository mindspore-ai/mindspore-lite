# Qwen3-Reranker-8B ONNX 导出与 MindSpore Lite 推理

本目录提供 Qwen3-Reranker-8B 的 ONNX 导出脚本及基于 MindSpore Lite 的推理脚本，已在 **Atlas 300I Duo** 上完成端到端流程打通、精度对齐与性能验证。

---

## 1. 概览

Qwen3-Reranker-8B 是通义千问团队推出的**重排序模型**（Cross-Encoder Reranker），用于对检索结果进行精排。输入 Query 与 Document，输出相关性分数。

本实现采用与官方一致的二分类式打分：

- 取最后一个 token 位置的 logits
- 取 `yes` / `no` 两个 token 的 logits 做 softmax
- `P(yes)` 作为相关性分数

### 模型架构参数（取自 `config.json`）

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8（GQA） |
| `head_dim` | 128 |
| `intermediate_size` | 12288 |
| `vocab_size` | 151669 |
| `max_position_embeddings` | 40960 |
| 原始权重精度 | bfloat16 |

---

## 2. 环境依赖

| 软件包 | 验证版本 |
|---|---|
| Python | 3.11 |
| torch | 2.9.0（CPU 即可，仅用于导出） |
| transformers | 5.9.0 |
| onnx | 1.19.1 |
| numpy | - |
| mindspore-lite | ≥ 2.9.0（converter_lite 来自 2.9.0 工具包；Python 运行时 2.10.0 验证通过） |
| CANN | 8.5.0 |

```bash
pip install torch transformers onnx numpy
```

> 导出（export）仅需 CPU + torch；推理（infer）在昇腾 NPU 上执行。

---

## 3. 导出 ONNX

```bash
cd ./mindspore-lite/examples/base_models/qwen3_reranker_8b

python export_qwen3_reranker_8b_onnx.py \
    --model-id ./Qwen/Qwen3-Reranker-8B \
    --output-dir ./onnx \
    --max-length 8192 \
    --device cpu
```

### 导出参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 模型路径或 ModelScope/HF ID | `Qwen/Qwen3-Reranker-8B` |
| `--output-dir` | 导出输出目录 | `./onnx` |
| `--max-length` | 最大序列长度（控制 dummy 输入） | `8192` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |

### 导出说明

- 导出脚本将 `Qwen3ForCausalLM` 拆为 `model.model`（主干）+ `lm_head`，输出完整 `logits`，动态 `batch` 与 `sequence` 轴（opset 17）。
- 导出时强制使用 **legacy (TorchScript) exporter**（`dynamo=False`）：torch ≥ 2.9 默认改用 dynamo exporter（opset 18），既无法降级到 opset 17，产出的图也无法被 `converter_lite` 的 `ascend_oriented` 优化正确处理。脚本已对 torch < 2.7（无 `dynamo` 参数）做兼容。
- 导出时会自动清理 ONNX 图中的 `IsNaN/Where` 节点，并将外部权重合并为单一 `.data` 文件（图编辑阶段以 `load_external_data=False` 加载，避免把 ~16GB 权重重新写回 `.onnx` 造成 >2GB 的非法 protobuf）。
- 8B 模型导出峰值约占 32GB 内存，请确保内存充足（或减小 `--max-length`）。

### 导出产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen3_reranker_8b.onnx` | ~1.9MB | 模型图定义 |
| `qwen3_reranker_8b.data` | ~16GB | 外部权重（fp16） |

### 模型 I/O

**输入：**

- `input_ids`：`int64`，形状 `(batch, seq_len)`
- `attention_mask`：`int64`，形状 `(batch, seq_len)`

**输出：**

- `logits`：`float`，形状 `(batch, seq_len, 151669)`

推理脚本取最后一个 token 位置的 logits，提取 `yes`（token 9693）和 `no`（token 2152）的 logits，通过 softmax 计算 `P(yes)` 作为相关性分数。

---

## 4. 输入格式

输入按以下模板拼接后再 tokenize（与官方 Qwen3-Reranker 示例一致）：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
```

其中 `instruction` 默认为 `Given a web search query, retrieve relevant passages that answer the query`。

---

## 5. MindSpore Lite 模型转换

```bash
converter_lite \
    --fmk=ONNX \
    --modelFile=./onnx/qwen3_reranker_8b.onnx \
    --outputFile=./onnx/qwen3_reranker_8b_fp16 \
    --optimize=ascend_oriented \
    --saveType=MINDIR \
    --configFile=./configs/config.ini
```

### config 文件 `./configs/config.ini`

```ini
[acl_init_options]
ge.exec.precision_mode = allow_fp32_to_fp16
```

> ⚠️ **精度模式选择（关键）**：8B 模型原始权重为 bf16。
> - 使用 `allow_fp32_to_fp16`（本目录默认）：权重以 fp16 存储，转换产物 `*_variables/data_0` 约 **31GB**，可放入 300I Duo 单卡 44GB 显存，**推荐**。
> - 若改用 `force_fp32`：权重被上提到 fp32，产物约 **45GB**，**超出 300I Duo 单卡 44GB 显存**，推理时 `build_from_file` 会以 `aclmdlLoadFromMem failed, ret=500002`（显存 OOM）失败。`force_fp32` 仅适用于显存 ≥ 48GB 的更大卡型（如 Ascend 910 系列 48GB 卡）。

### 转换产物

| 文件 | 大小 | 说明 |
|---|---|---|
| `qwen3_reranker_8b_fp16_graph.mindir` | ~0.9KB | 图定义（引用外部变量） |
| `qwen3_reranker_8b_fp16_variables/data_0` | ~31GB | 编译后的 ACL om 权重（fp16） |

---

## 6. MindSpore Lite 推理

```bash
python infer_qwen3_reranker_8b_mslite.py \
    --model ./onnx/qwen3_reranker_8b_fp16_graph.mindir \
    --tokenizer ./Qwen/Qwen3-Reranker-8B \
    --max-length 8192 \
    --device ascend \
    --device-id 0
```

### 推理参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model` | MindIR 模型路径（`*_graph.mindir`） | 必填 |
| `--tokenizer` | Tokenizer 路径或 ID | `Qwen/Qwen3-Reranker-8B` |
| `--max-length` | 最大序列长度 | `8192` |
| `--device` | 推理设备（ascend/cpu） | `ascend` |
| `--device-id` | Ascend device ID | `0` |

### 推理结果示例

```text
Token IDs - yes: 9693, no: 2152

Running reranking inference...
Pair 1: seq_len=100, latency=215.281ms
Pair 2: seq_len=100, latency=163.619ms

Reranking scores:

[1] Score: 0.6123
Query: What is the capital of China?
Document: The capital of China is Beijing.

[2] Score: 0.6885
Query: Explain gravity
Document: Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.
```

---

## 7. 性能数据（Atlas 300I Duo，单卡）

| 指标 | 值 |
|---|---|
| 首条延迟（含 ACL kernel 编译预热） | ~215ms |
| 稳态延迟（seq_len=100） | ~164ms |
| 稳态延迟（seq_len=81，10 次平均） | **152.05ms**（min 151.20 / max 156.50 / std 1.52） |
| 输入 logits 输出 | `(1, seq_len, 151669)` |

> 8B 模型规模较大，延迟随 `seq_len` 线性增长；上表为单条 Rerank（batch=1）的端到端 `predict` 耗时。首条 predict 会触发 ACL kernel 编译，故明显高于稳态。

### 模型体积

| 组件 | 大小 |
|---|---|
| ONNX 图 | ~1.9MB |
| ONNX 外部数据（fp16） | ~16GB |
| MindIR Variables（fp16，本流程） | ~31GB |
| MindIR Variables（force_fp32，对比） | ~45GB（300I Duo 不可用） |

---

## 8. 常见问题

### 1) 导出报错 `axes_input_to_attribute ... No initializer or constant input`（或 `.onnx` 高达 ~16GB 无法解析）

这是 **torch ≥ 2.9** 默认改用 dynamo ONNX exporter 导致的：dynamo 导出 opset 18 后无法降级到 opset 17，会抛出该 `RuntimeError`；若侥幸完成，产出的 `.onnx` 会把 ~16GB 权重内联进 protobuf（>2GB 非法），转换/加载都会失败。

本目录导出脚本已强制 `dynamo=False`（legacy TorchScript exporter，直接产出 opset 17 图），可避免该问题。若你自行修改脚本，请勿去掉 `dynamo=False`。

### 2) `build_from_file failed! ... aclmdlLoadFromMem failed, ret=500002`

`ret=500002` 为 ACL 显存不足（OOM）。8B 模型 `force_fp32` 产物约 45GB，超过 300I Duo 单卡 44GB。请改用本目录默认的 `allow_fp32_to_fp16` 配置（产物约 31GB，可正常运行），见第 5 节。

### 3) 转换时出现 `ge.proto.ModelDef exceeded maximum protobuf size of 2GB`

8B 权重约 16GB（fp16），`ascend_oriented` 转换过程中会打印此信息。该日志**不影响最终转换结果**——只要结尾出现 `CONVERT RESULT SUCCESS:0` 即成功，工具链会正常写出 `*_graph.mindir` + `*_variables/data_0`。

### 4) 转换时出现 `ConstantOfShape infershape failed` Warning

常量折叠阶段对动态 shape 节点的告警，可忽略，不影响最终产物。

### 5) 导出时内存不足

8B 模型导出峰值约占 32GB 内存。若内存不足，可减小 `--max-length` 或关闭其他程序；导出本身只需 CPU。

### 6) 精度对齐建议

- 确认 tokenizer 与权重目录一致（`./Qwen/Qwen3-Reranker-8B`）。
- mslite fp16 与 PyTorch **fp16** 对齐最紧；与 bf16 原始分数存在 ~0.02 量级表示差异，属正常。
- 若对极长序列精度有更高要求且卡型显存 ≥ 48GB，可改用 `force_fp32` 转换。

---

## 9. 参考链接

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-Reranker-8B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Reranker-8B)
- [Qwen3-Reranker-8B HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-8B)

## 10. 许可证

本工具遵循 Qwen3 模型的许可证要求。
