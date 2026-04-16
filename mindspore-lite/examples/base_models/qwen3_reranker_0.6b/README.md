# Qwen3-Reranker-0.6B 导出与推理

本目录提供 Qwen3-Reranker-0.6B 的 ONNX 导出脚本、ONNX Runtime 推理脚本，以及可选的 MindSpore Lite（MindIR）推理脚本。

## 1. 模型简介

Qwen3-Reranker-0.6B 是 cross-encoder reranker：输入 Query 与 Document，输出一个相关性分数，用于排序检索结果。

本实现采用二分类式打分：

- 取最后一个 token 位置的 logits
- 取 `yes/no` 两个 token 的 logits 做 softmax
- `P(yes)` 作为相关性分数

## 2. 环境准备

### 2.1 Python 依赖

```bash
pip install -U \
  transformers==4.57.6 \
  torch==2.11.0 \
  onnx==1.16.0 \
  numpy==1.26.4 \
  onnxruntime==1.24.4
```

如需 MindSpore Lite 推理：

```bash
pip install -U mindspore-lite==2.7.0
```

## 3. 目录结构

```text
qwen3_reranker_0.6b/
├── export_qwen3_reranker_onnx.py
├── infer_qwen3_reranker_onnx.py
├── infer_qwen3_reranker_mslite.py
└── README.md
```

## 4. 导出 ONNX

```bash
python export_qwen3_reranker_onnx.py \
  --model-id Qwen/Qwen3-Reranker-0.6B \
  --output-dir ./onnx \
  --max-length 8192 \
  --device cpu
```

导出产物：

- `./onnx/qwen3_reranker_0.6b.onnx`

说明：

- 脚本会在导出后对 ONNX 做一次图后处理，移除 `IsNaN` 相关节点（见导出脚本 `_remove_isnan_nodes`）。

## 5. ONNX Runtime 推理

```bash
python infer_qwen3_reranker_onnx.py \
  --model-path ./onnx/qwen3_reranker_0.6b.onnx \
  --tokenizer Qwen/Qwen3-Reranker-0.6B \
  --max-length 8192 \
  --device CPU
```

参数说明：

- `--device`: 仅支持 `CPU` 或 `CUDA`（需 onnxruntime-gpu）

## 6. ONNX 转 MindIR

在 MindSpore Lite 的 `output` 目录下执行（示例）：

```bash
./converter_lite \
  --fmk=ONNX \
  --modelFile=./onnx/qwen3_reranker_0.6b.onnx \
  --outputFile=./onnx/qwen3_reranker_0.6b \
  --optimize=ascend_oriented
```

产物示例：

- `./onnx/qwen3_reranker_0.6b.mindir`

## 7. MindSpore Lite 推理

```bash
python infer_qwen3_reranker_mslite.py \
  --model ./onnx/qwen3_reranker_0.6b.mindir \
  --tokenizer Qwen/Qwen3-Reranker-0.6B \
  --max-length 8192 \
  --device ascend \
  --device-id 0
```

说明：

- `infer_qwen3_reranker_mslite.py` 默认 `--device ascend`。
- mslite 推理脚本当前将 `input_ids/attention_mask` 传为 `int32`（见脚本实现）。

## 8. 输入格式

输入会按以下模板拼接后再 tokenize：

```text
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
```

其中 `instruction` 默认值为：
`Given a web search query, retrieve relevant passages that answer the query`

## 9. 模型 I/O（推理侧）

ONNX / MindIR 输入：

- `input_ids`: shape `(batch, seq_len)`，整型
- `attention_mask`: shape `(batch, seq_len)`，整型

输出：

- `logits`: shape `(batch, seq_len, vocab_size)`，浮点型

推理脚本会逐条样本执行（每次 batch=1），以适配不同后端的输入限制。

## 10. 参考链接

- https://www.mindspore.cn/lite/docs/zh-CN/master/use/ascend_info.html
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
- https://onnxruntime.ai/docs/

