# GLiNER large-v2.5 MindSpore Lite 推理部署教程

本教程介绍如何将 [GLiNER large-v2.5](https://github.com/urchade/GLiNER) 导出为 ONNX 后转换为 MindSpore Lite MindIR，在 Atlas 300I Duo 上推理与测速。

GLiNER 是一种可指定任意标签的命名实体识别（NER）模型。本教程基于 GLiNER v0.2.27 的 `UniEncoderSpanGLiNER` 架构（DeBERTa-v3-large 主干 + 双向 LSTM + SpanMarkerV0 span 表示）。

---

## 1. 环境准备

### 依赖版本

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.10.0+cpu |
| onnx | 1.19.1 |
| numpy | 2.4.4 |
| transformers | 4.57.0 |
| gliner | 0.2.27 |
| CANN | 8.5 |
| mindspore-lite | 2.9.0 |

```bash
pip install torch==2.10.0 onnx==1.19.1 transformers==4.57.0 gliner==0.2.27
```

### 获取模型权重与源码

```bash
# 模型源码
git clone https://github.com/urchade/GLiNER.git models/model_code/GLiNER
pip install -e models/model_code/GLiNER

# 模型权重（HuggingFace 下载）
mkdir -p models/model_weight
cd models/model_weight
git clone https://huggingface.co/gliner-community/gliner_large-v2.5
```

说明：

- `MODEL_DIR`=`models/model_weight/gliner_large-v2.5`，包含 `pytorch_model.bin`、`gliner_config.json`、`tokenizer.json`、`spm.model` 等。
- 上游源码用于 import `gliner` 包；本目录脚本会在导出前对 `gliner` 内部函数做 monkey patch，把动态 shape 的 Python 控制流改造为可被 Ascend 接受的图。

---

## 2. 模型导出 ONNX

### 导出命令

```bash
cd mindspore-lite/examples/base_models/gliner_large-v2.5

python export_gliner_large-v2.5_onnx.py \
  --model-dir models/model_weight/gliner_large-v2.5 \
  --save-dir ./onnx \
  --opset 17
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录（含 `pytorch_model.bin`、`gliner_config.json`、tokenizer 文件） | `models/model_weight/gliner_large-v2.5` |
| `--save-dir` | ONNX 输出目录 | `./onnx` |
| `--opset` | ONNX opset 版本 | `17` |

### 产出文件

```text
./onnx/
├── model.onnx                # 单一 ONNX（含 DeBERTa + LSTM + SpanMarker）
├── gliner_config.json        # GLiNER 配置（max_width=12, ent/sep token 等）
├── tokenizer.json            # DeBERTa-v3 tokenizer
├── spm.model                 # SentencePiece 模型
├── tokenizer_config.json
├── special_tokens_map.json
└── added_tokens.json
```

### 导出注意事项（实际踩坑点）

GLiNER 上游依赖三处 Ascend 不友好的实现，导出脚本在 `model.export_to_onnx(...)` 调用前对源码做了 monkey patch（均在 `export_gliner_large-v2.5_onnx.py` 中）：

1. **`_SmallOpLSTM` → 原生 `nn.LSTM`**：上游 `LstmSeq2SeqEncoder` 用 `_SmallOpLSTM`，其 `_run_direction` 是 Python `for t in range(seq_len)` 循环，JIT trace 会把 `seq_len` 烧成常量（dummy batch 是 5 words）。脚本从 checkpoint 中读取 `rnn.lstm.*_ih/_hh` 权重，重建为 `nn.LSTM` 并替换；同时 patch `LstmSeq2SeqEncoder.forward` 去掉 `lengths=` kwarg + 预分配静态 `h0/c0`。
2. **DeBERTa 的 `make_log_bucket_position`**：用 `@torch.jit.script` 包装，内部用 `torch.sign` 产生 Sign 算子，Ascend 不支持。脚本预计算 `(_REL_POS_MAX_SEQ × _REL_POS_MAX_SEQ)` 的 bucketed 相对位置矩阵作为常量，运行时只做切片。
3. **DeBERTa 的 `build_rpos` 与 `transpose_for_scores`**：`@torch.jit.script` 内的 Python `if` 会 trace 成 24 个 If 子图，Ascend 拒绝其中的 Range 子图；`transpose_for_scores` 用元组拼接，产生 rank 推断失败的 Concat。脚本把 `build_rpos` 替换为恒等函数，`transpose_for_scores` 用 Python int 常量重写。
4. **`extract_prompt_features`**：`.max()` 返回 ambiguous rank 的标量，与 span_idx 维度拼接时报 Concat rank 不匹配。脚本统一加 `.max().reshape(())` 强制 rank-0。

由于 dummy batch 默认用 `[person, organization, country]` 3 标签，导出的 ONNX 在 `num_classes=3` 维度上是动态的；若需要其他数量的标签，需要在 `gliner/model.py::_build_dummy_batch` 修改 `labels` 默认值，或直接在导出脚本中传入 `labels=` kwarg。

---

## 3. MindSpore Lite 转换（ONNX → MindIR）

### 转换命令

说明：`converter_lite` 为 MindSpore Lite 版本包中提供的离线转换工具。

```bash
converter_lite --fmk=ONNX \
  --modelFile=./onnx/model.onnx \
  --outputFile=./onnx/model \
  --saveType=MINDIR \
  --optimize=ascend_oriented \
  --configFile=./gliner_large-v2.5.ini
```

### 参数说明

| 参数 | 说明 |
| --- | --- |
| `--modelFile` | 输入 ONNX |
| `--outputFile` | 输出前缀 |
| `--optimize=ascend_oriented` | Ascend 定向优化 |
| `--saveType=MINDIR` | 输出 MindIR |
| `--configFile` | 配置文件（指定输入 dtype、固定 shape、precision mode 等） |

### 配置文件

`gliner_large-v2.5.ini`（静态 shape，**生产推荐**）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,128;attention_mask:1,128;words_mask:1,128;text_lengths:1,1;span_idx:1,288,2;span_mask:1,288"
```

固定 shape 说明（必须写清楚，否则推理侧无法对齐）：

| 输入 | 静态 shape | 含义 |
| --- | --- | --- |
| `input_ids` / `attention_mask` / `words_mask` | `(1, 128)` | 序列长度固定 128 |
| `text_lengths` | `(1, 1)` | 实际 body words 数（运行时可变，1–24） |
| `span_idx` | `(1, 288, 2)` | 24 body words × 12 max_width 的 span 网格 |
| `span_mask` | `(1, 288)` | 仅 `s+k < text_lengths[0]` 的 span 为 True |

**为什么必须静态 shape**：上游 `_SmallOpLSTM` 用 Python 循环展开，我们替换为原生 `nn.LSTM`，但原生 LSTM 在动态 seq_len 下输出 4D 张量 `[-1, 2, -1, 384]`（两个动态维度），Ascend 多 batch 编译器报 `Multi-batch not support middle dynamic shape`。尝试过预分配静态 `h0/c0` 与减少动态维度（`config_dyn_seq.ini`、`config.ini`）均失败，因此放弃动态 shape，转而使用静态 shape + padding/截断。

### 产出文件

```text
./onnx/
└── model.mindir                 # ~1 GB，单文件（fp16 权重）
```

执行日志：

```log
CONVERT RESULT SUCCESS:0
```

---

## 4. MindSpore Lite 推理

### 推理命令

```bash
python infer_gliner_large-v2.5_mslite.py \
  --model-dir models/model_weight/gliner_large-v2.5 \
  --mindir-path ./onnx/model.mindir \
  --text "Cristiano Ronaldo plays for Al-Nassr FC and captains Portugal." \
  --labels person,organization,country \
  --threshold 0.5 \
  --device-id 0
```

### 参数说明

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model-dir` | 权重目录（用于加载 tokenizer 与 `gliner_config.json`） | `models/model_weight/gliner_large-v2.5` |
| `--mindir-path` | MindIR 文件 | `./onnx/model.mindir` |
| `--text` | 输入文本 | 内置 3 条示例 |
| `--text-file` | 每行一条文本的输入文件 | None |
| `--labels` | 逗号分隔的实体标签（**必须为 3 个，与导出 dummy batch 一致**） | `person,organization,country` |
| `--threshold` | sigmoid 置信度阈值 | `0.5` |
| `--flat-ner` | 禁止 span 重叠（默认开启） | True |
| `--device-id` | Ascend 设备 ID | `0` |

### 执行日志

```log
[infer] config: ent='<<ENT>>', sep='<<SEP>>', labels=['person', 'organization', 'country']
[infer] loading tokenizer from models/model_weight/gliner_large-v2.5
[infer] loading MindIR from ./onnx/model.mindir
WARNING:root:Ascend custom operator path not found

[infer] text: Cristiano Ronaldo Ronaldo dos Santos Aveiro plays for Al-Nassr FC and captains Portugal.
[infer] words (13): ['Cristiano', 'Ronaldo', 'dos', 'Santos', 'Aveiro', 'plays', 'for', 'Al-Nassr', 'FC', 'and', 'captains', 'Portugal', '.']
[infer] seq_len: 128, logits shape: (1, 24, 12, 3)
  - 'Portugal' [country] score=0.9993 chars=(71, 79)
  - 'Al-Nassr FC' [organization] score=0.9944 chars=(46, 57)
  - 'Cristiano Ronaldo dos Santos Aveiro' [person] score=0.9910 chars=(0, 35)
[infer] 模型推理: 15.82 ms | 端到端: 16.52 ms
```

说明（ascend_oriented 固定 shape 约束）：

- 推理脚本固定 3 标签 `[person, organization, country]`，与导出 dummy batch 一致；传入其他数量的标签会报错。
- body words 数运行时可变（1–24），脚本自动截断长文本，并对 `input_ids/attention_mask/words_mask` 做 seq_len=128 padding。`text_lengths` 反映真实 body words 数，`span_mask` 仅置位前 `num_body_words × max_width` 个 span。
- 若需要不同的标签集，需要重新跑导出脚本（修改 `gliner/model.py::_build_dummy_batch` 的 `labels` 默认值，或在导出脚本中传 `labels=` kwarg），再重新转换 MindIR。

---

## 5. 性能数据

测试环境：Atlas 300I Duo

固定文本（"Cristiano Ronaldo dos Santos Aveiro plays for Al-Nassr FC and captains Portugal."），50 次平均（3 次 warmup 后）：

| 指标 | MindSpore Lite (Ascend fp16) |
| --- | ---: |
| 模型推理 | 15.82 ms |
| 端到端（含预处理） | 16.52 ms |
| **吞吐量** | **60.6 req/s** |

---

## 6. 常见问题

1. 现象：`op[Expand], custom inputs shape [0] error!`
   - 原因：输入 `text_lengths` 为 0。
   - 解决方案：保证至少有 1 个 body word；空文本会被跳过或加 placeholder。

2. 现象：converter 报 `Multi-batch not support middle dynamic shape. CurrentShape: [-1,-1,-1,-1]`
   - 原因：原生 `nn.LSTM` 在动态 seq_len 下输出 `[-1, 2, -1, 384]`，含 2 个动态维度，Ascend 多 batch 编译器拒绝。
   - 解决方案：使用静态 shape（`gliner_large-v2.5.ini`），不要使用动态 shape 配置（`config.ini`/`config_dyn_seq.ini` 已废弃）。

3. 现象：`RuntimeError: Static MindIR has 3 classes baked in, but got N labels`
   - 原因：静态 MindIR 在转换时锁定了 `num_classes=3`（导出 dummy batch 用 3 标签）。
   - 解决方案：传入 3 标签 `--labels person,organization,country`，或重新导出 ONNX 并重新转换。

4. 现象：converter 很慢且有大量 warning（`SetupParamInitSubGraph` / `tiling offset out of range`）
   - 原因：DeBERTa 主干层数深 + ascend_oriented 编译优化重。
   - 解决方案：确认最终 `CONVERT RESULT SUCCESS:0`；转换约耗时 2–3 分钟，确保内存 ≥ 16 GB。

---

## 7. 参考资源

- 上游模型仓库：https://github.com/urchade/GLiNER
- HuggingFace 权重：https://huggingface.co/gliner-community/gliner_large-v2.5
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 8. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- 上游 GLiNER 模型与代码以 Apache License 2.0 发布。
