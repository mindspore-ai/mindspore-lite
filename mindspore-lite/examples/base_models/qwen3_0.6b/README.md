# Qwen3-0.6B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-0.6B 纯文本模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend NPU 上完成端到端推理部署。

Qwen3-0.6B 是 Qwen 系列中体积最小的对话模型，适合作为轻量级对话 / 抬头组件部署在 Atlas 300I Duo 等 NPU 上。本教程支持三种部署场景：

| 场景 | 说明 | 适用业务 |
|------|------|---------|
| **A. 通用 prefill+decode** | 完整对话生成，prefill 输出 KV cache，decode 自回归生成 | 通用对话 |
| **B. 单 token prefill** | 只需输出一个 token（如选择题判定），无 decode | 单 token 判定场景 |
| **C. 公共前缀（prefix+suffix）** | 有固定公共前缀（如 system prompt），prefix 算一次 KV cache，suffix 只算用户输入 | 有固定 system prompt 的场景 |

三种场景共享以下优化特性（默认使能）：

- **混合精度**：`allow_mix_precision` + `op_fp32.json` 黑名单，RmsNorm 敏感算子保 FP32，其余走 FP16
- **CANN 融合算子**：`RotaryMul`、`PromptFlashAttention`
- **slice_last**：prefill 输出 `[batch, 1, vocab]` logits，最小化 D2H 传输时延

## 模型架构

Qwen3-0.6B 是一个 28 层的 decoder-only Transformer：

| 项目                | 值     |
|-------------------|-------|
| 层数                | 28    |
| num_attention_heads | 16    |
| num_key_value_heads | 8（GQA）|
| head_dim          | 128   |
| hidden_size       | 1024  |
| vocab_size        | 151936 |

---

## 1. 环境准备

### 依赖版本

| 软件包            | 版本     |
|----------------|--------|
| Python         | 3.11   |
| torch          | 2.10.0 |
| transformers   | 4.51.3 |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 8.5    |
| mindspore-lite | 2.9.0  |

```bash
pip install transformers==4.51.3 torch==2.10.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

### 权重准备

从 HuggingFace 下载 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 权重，解压后放到本地目录（本文以 `./Qwen3-0.6B` 为例）。
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B

---

## 2. 场景 A：通用 prefill + decode

适用于通用对话生成场景。Prefill 一次性处理完整 prompt，输出首 token logits 与初始 KV cache；Decode 基于 KV cache 做自回归增量生成。

### 2.1 模型导出 ONNX

```bash
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu
```

> **短 seq_len 场景（如 chat template 的 s=33）可关闭 PFA 融合以避免 128 对齐 padding 带来的计算膨胀：**
> ```bash
> python export_qwen3_onnx.py --model-id ./Qwen3-0.6B --output-dir ./qwen3_onnx --device cpu \
>   --disable-fusion --enable-rotarymul
> ```

#### 导出参数说明

| 参数            | 说明                  | 默认值                  |
|---------------|-----------------------|------------------------|
| `--model-id`  | HuggingFace 模型路径或本地目录 | `Qwen/Qwen3-0.6B`     |
| `--output-dir`| 输出目录                | `./qwen3_onnx`         |
| `--device`    | 导出设备（cpu / cuda）    | `cpu`                  |
| `--enable-pfa` | QK^T+softmax+V 走 `Custom(PromptFlashAttention)` | **开启** |
| `--prefill-only-without-cache` | 只导出 prefill（无 KV cache 输出，无 decode），一般用于场景 B | 关闭 |
| `--enable-common-prefix` | 导出 prefix + suffix 模型，一般用于场景 C 公共前缀缓存 | 关闭 |
| `--disable-fusion` | 关闭所有融合（导出 non-fused 基线） | 关闭 |

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill.onnx   # Prefill（输出 [batch, 1, vocab] logits + KV cache）
└── qwen3_llm_decode.onnx    # Decode（自回归增量生成）
```

#### ONNX 模型输入输出 Shape

**LLM Prefill** — `qwen3_llm_prefill.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, seq_len)`     | int32   | 输入 token IDs     |
| 输入 | `attention_mask` | `(batch, seq_len)`     | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, seq_len)`     | int32   | 位置 ID            |
| 输出 | `logits`         | `(batch, 1, 151936)`   | float32 | 最后一个 token 的 logits |
| 输出 | `past_kv`        | `(56, batch, 8, seq_len, 128)` | float32 | 初始 KV cache（56 = 28 × 2） |

**LLM Decode** — `qwen3_llm_decode.onnx`

| 方向  | 名称               | Shape                          | Dtype   | 说明                     |
|-----|------------------|--------------------------------|---------|------------------------|
| 输入 | `input_ids`      | `(batch, 1)`                   | int32   | 单步 token               |
| 输入 | `attention_mask` | `(batch, total_seq_len)`       | int32   | 累积注意力掩码              |
| 输入 | `position_ids`   | `(batch, 1)`                   | int32   | 单步位置 ID              |
| 输入 | `past_key_values` | `(56, batch, 8, past_seq_len, 128)` | float32 | 上一步 KV cache          |
| 输出 | `logits`         | `(batch, 1, 151936)`           | float32 | 单步 logits             |
| 输出 | `past_kv`        | `(56, batch, 8, past_seq_len+1, 128)` | float32 | 更新后的 KV cache        |

### 2.2 ONNX 转 MindIR

```bash
Convert=mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill.ini

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_decode.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_decode \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_decode.ini
```

#### 转换参数说明

| 参数             | 说明                          |
|----------------|-----------------------------|
| `--fmk`        | 输入模型格式（ONNX）                |
| `--modelFile`  | 输入 ONNX 模型路径                |
| `--outputFile` | 输出 MindIR 路径（不带扩展名）         |
| `--optimize`   | 优化模式，必须指定 `ascend_oriented` |
| `--saveType`   | 输出格式（MINDIR）                |
| `--configFile` | 配置文件路径（**所有模型都必须指定**）    |

#### 配置文件

所有配置文件统一存放在 `configs/` 目录下：

```text
configs/
├── qwen3_llm_prefill.ini           # 场景 A/B Prefill（6 档动态分档）
├── qwen3_llm_decode.ini            # 场景 A Decode（5 档 KV cache 分档）
├── qwen3_llm_prefill_prefix.ini    # 场景 C Prefix（2 档）
├── qwen3_llm_prefill_suffix.ini    # 场景 C Suffix（2 档）
└── op_fp32.json                    # 混合精度黑名单
```

`configs/qwen3_llm_prefill.ini`（场景 A/B Prefill 用，6 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;position_ids:-1,-1"
ge.dynamicDims="1,128,1,128,1,128;1,256,1,256,1,256;1,512,1,512,1,512;1,1024,1,1024,1,1024;1,480,1,480,1,480;1,640,1,640,1,640"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./op_fp32.json"
```

`configs/qwen3_llm_decode.ini`（场景 A Decode 用，5 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_values:56,1,8,-1,128"
ge.dynamicDims="33,32;769,768;1025,1024;1537,1536;2049,2048"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

- `past_key_values:56,1,8,-1,128`：56 = 28 层 × 2（K+V），1 = batch，8 = num_kv_heads，-1 = 动态 KV 长度，128 = head_dim
- `ge.dynamicDims`：5 个分档对应 `past_kv_len ∈ {32, 768, 1024, 1536, 2048}`

`configs/op_fp32.json`（混合精度黑名单）：

```json
{
    "black-list": {
        "to-add": ["RealDiv", "SquareSumV1", "Square", "Sqrt", "ReduceMean"]
    }
}
```

> **AOE 子图调优**（可选）：如需启用 AOE 子图调优以进一步优化性能，在配置文件中添加 `[ascend_context]` 段并设置 `aoe_mode="subgraph tuning"` 即可。AOE 会自动搜索最优算子分块策略，但会增加转换时间。

#### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen3_onnx/
├── qwen3_llm_prefill_graph.mindir           # Prefill 主图（含动态分档）
├── qwen3_llm_prefill_variables/             # Prefill 权重
│   └── data_0
├── qwen3_llm_decode_graph.mindir            # Decode 主图
└── qwen3_llm_decode_variables/              # Decode 权重
    └── data_0
```

### 2.3 MindSpore Lite 推理

```bash
python infer_qwen3_0.6b_mindir.py \
  --mode prefill_decode \
  --prefill-model ./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --decode-model ./qwen3_onnx/qwen3_llm_decode_graph.mindir \
  --tokenizer ./Qwen3-0.6B \
  --prompt "你好，请介绍一下你自己。" \
  --max-new-tokens 128 \
  --prefill-buckets "128,256,512,1024" \
  --decode-buckets "32,768,1024,1536,2048" \
  --device ascend \
  --device-id 0
```

#### 推理参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|---------------------------|
| `--mode`           | 推理模式：`prefill_decode` / `prefill_only` / `common_prefix` | `prefill_only` |
| `--prefill-model`  | Prefill MindIR 模型路径（`*_graph.mindir`） | 必填                       |
| `--decode-model`   | Decode MindIR 模型路径（`*_graph.mindir`）  | 必填（场景 A）           |
| `--tokenizer`      | HuggingFace tokenizer 路径              | `Qwen/Qwen3-0.6B` |
| `--prompt`         | 输入文本                                  | `"你好，请介绍一下你自己。"`   |
| `--system-prompt`  | 系统提示词（场景 B/C）                      | `"You are a helpful assistant. Answer questions concisely."` |
| `--max-new-tokens` | 最大生成 token 数                          | `128`                     |
| `--max-length`     | 模型最大上下文长度                            | `2048`                    |
| `--prefill-buckets` | Prefill seq_len 分档（**必须与转换 ini 的 `ge.dynamicDims` 完全一致**） | `128,256,512,1024,480,640` |
| `--decode-buckets` | Decode KV cache 分档（**必须与转换 ini 的 `ge.dynamicDims` 完全一致**） | 5 档 |
| `--no-chat-template` | 关闭 chat template，进入 raw completion 模式 | 关闭                       |
| `--device`         | 推理设备（ascend/cpu）                    | `ascend`                  |
| `--device-id`      | Ascend 设备 ID                          | `0`                       |
| `--force-no-pre-alloc` | 关闭输出 buffer 预分配（部分精度模式下需要）     | 关闭                       |

> `--decode-buckets` 必须与转换时 `ge.dynamicDims` 中的 `past_kv_len` 列表**完全一致**，否则会触发 `aclmdlSetInputDynamicDims failed`。脚本会自动把 `past_kv` 与 `attention_mask` 补零到下一档边界，并在每步把模型追加在分档尾部的 K/V 切回真实长度。
>
> `--prefill-buckets` 必须与 `qwen3_llm_prefill.ini` 中 `ge.dynamicDims` 的 `seq_len` 列表**完全一致**。脚本会把 prompt tokenization 后的实际 seq_len 向上 pad 到最近的 prefill bucket（补 `pad_token` + `attention_mask=0`），Prefill 后再把 `past_kv` 与 `logits` 切回真实 seq_len 传给 Decode。

#### 推理示例输出

```text
Initializing MindSpore Lite context for ascend...
Loading prefill model from ./qwen3_onnx/qwen3_llm_prefill_graph.mindir...
Loading decode model from ./qwen3_onnx/qwen3_llm_decode_graph.mindir...
Loading tokenizer from ./Qwen3-0.6B...
[zero-copy] pre-allocated outputs: enabled (probe OK)

============================================================
Mode: prefill_decode
Input Prompt: 你好，请介绍一下你自己。
============================================================
[prefill] seq_len=13 -> bucket=128 (pad 115)
Running LLM prefill...
Prefill time: 42.92 ms
Running LLM decode...
Total decode time: 2487.42 ms, avg decode step: 19.59 ms, steps: 127
Total time: 2530.33 ms, throughput: 50.59 tok/s

============================================================
Generated Response:
好的，用户问我的介绍。我需要先确认用户的需求是什么。可能他们想了解我的功能，或者想进行交流。我应该保持友好和专业的态度，同时提供有用的信息。

首先，我应该简要介绍我的功能，比如处理各种请求、提供帮助等。然后，可以提到我的特点，比如多语言支持、快速响应等。同时，要确保回答清晰，避免使用过于技术化的术语，让用户容易理解。

另外，用户可能没有明确说明他们的需求，所以需要保持开放，邀请他们提出更多问题。这样可以确保回答既全面又实用，同时保持互动
============================================================
```

> 场景 A 默认开启 Qwen3 thinking 模式（`enable_thinking=True`），输出含思考过程（"好的，用户问我的介绍..."）。场景 B/C 关闭 thinking 模式，直接输出选项/答案。

### 2.4 性能数据

#### 测试环境

| 项目   | 配置                     |
|------|------------------------|
| 硬件   | Atlas 300I Duo |
| 模型   | Qwen3-0.6B（28 层，16 attn heads，8 KV heads GQA，head_dim=128） |
| 精度   | Prefill: allow_mix_precision + op_fp32.json 黑名单 + slice_last + AOE；Decode: force_fp32 |
| 推理脚本 | `infer_qwen3_0.6b_mindir.py`（zero-copy + 预分配输出 buffer） |

#### 各阶段推理输入 Shape 与性能

**LLM Prefill（slice_last + mix 精度 + AOE）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids` |
| input_ids Shape  | `(1, seq_len)`，运行时 pad 到最近 bucket 边界 |
| 输出 logits Shape | `(1, 1, 151936)`（slice_last，只输出最后 token） |
| 输出 past_kv Shape | `(56, 1, 8, seq_len_bucket, 128)` |
| 推理耗时（s=128 bucket） | **42.92 ms** |

**LLM Decode（force_fp32，单步）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids`, `past_key_values` |
| input_ids Shape  | `(1, 1)`                        |
| attention_mask Shape | `(1, past_kv_len+1)`         |
| past_key_values Shape | `(56, 1, 8, past_kv_len, 128)` |
| 输出 logits Shape | `(1, 1, 151936)`               |
| 单步平均耗时      | **19.59 ms** |

#### 端到端推理性能（128 tokens 生成）

| Prompt | Prompt seq_len | Prefill (ms) | Avg decode (ms) | Total (ms) | 吞吐 (tok/s) |
|--------|----------------|--------------|-----------------|------------|--------------|
| 你好，请介绍一下你自己。 | 13 | 42.92 | 19.59 | 2530.33 | **50.59** |

---

## 3. 场景 B：单 token 判定

适用于只需要一个 token 输出的场景（如选择题判定、分类等）。Prefill 只输出最后一个 token 的 logits `[batch, 1, vocab]`，不输出 KV cache、不导出 decode，最小化 D2H 传输。

### 3.1 模型导出 ONNX

```bash
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu \
  --prefill-only-without-cache
```

`--prefill-only-without-cache` 的作用：

- 不输出 KV cache（单输出：logits）
- 不导出 decode 模型

#### 产出

```text
qwen3_onnx/
└── qwen3_llm_prefill.onnx   # 输出 [batch, 1, 151936]（场景 B）
```

> 场景 A 和场景 B 导出的 prefill 模型文件名相同（`qwen3_llm_prefill.onnx`），但模型结构不同（场景 A 有 KV cache 输出，场景 B 没有）。如需同时使用两个场景，请导出到不同目录。

#### ONNX 模型输入输出 Shape

**LLM Prefill (slice_last, no cache)** — `qwen3_llm_prefill.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, seq_len)`     | int32   | 输入 token IDs     |
| 输入 | `attention_mask` | `(batch, seq_len)`     | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, seq_len)`     | int32   | 位置 ID            |
| 输出 | `logits`         | `(batch, 1, 151936)`   | float32 | 最后一个 token 的 logits |

> **slice_last 右 padding 处理**：当输入右 padding 时，模型通过 `attention_mask.sum(dim=1)` 计算真实 last token 位置，使用 `index_select` 提取真实最后 token 的 logits，避免取到 pad token。

### 3.2 ONNX 转 MindIR

```bash
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill.ini
```

配置文件与场景 A 相同（`configs/qwen3_llm_prefill.ini`），场景 B 使用 480, 640 分档。

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill_graph.mindir   # 场景 B（slice_last，无 KV cache）
└── qwen3_llm_prefill_variables/
    └── data_0
```

### 3.3 MindSpore Lite 推理

```bash
python infer_qwen3_0.6b_mindir.py \
  --mode prefill_only \
  --prefill-model ./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --tokenizer ./Qwen3-0.6B \
  --prompt "The sky is blue because of what physical phenomenon, choose from A, B, C, D? A) Rayleigh scattering B) Diffraction C) Reflection D) Refraction" \
  --prefill-buckets "480,640"
```

#### 推理示例输出

```text
Mode: prefill_only
Input Prompt: The sky is blue because of what physical phenomenon, ...
[prefill] seq_len=65 -> bucket=480 (pad 415)
Running LLM prefill...
Prefill time: 35.50 ms
Output logits shape: (1, 1, 151936)
Predicted token id: 32
Decoded token: 'A'
```

#### Benchmark 命令

```bash
# Scene B Benchmark（seq=480）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,480;attention_mask:1,480;position_ids:1,480"

# Scene B Benchmark（seq=640）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,640;attention_mask:1,640;position_ids:1,640"
```

### 3.4 性能数据

| seq_len | Execute (ms) | D2H (ms) | AvgRunTime (ms) |
|---------|--------------|----------|-----------------|
| 480 | 35.5 | 0.12 | 35.6 |
| 640 | 48.7 | 0.12 | 49.1 |

> D2H 从 ~50ms（无 slice_last）降至 0.12ms — 只拷贝 `[1, 1, 151936]` = 297KB。

---

## 4. 场景 C：公共前缀（prefix + suffix）

适用于有固定公共前缀（如 system prompt）的场景。Prefix 模型处理公共前缀 token，输出 KV cache；Suffix 模型接收 prefix KV + 用户输入，输出最后 token logits。后续请求可复用 prefix KV cache，只需运行 suffix 模型。

> **场景 C 注意**：Suffix 模型使用非 PFA 路径（手动 matmul + softmax + matmul，含 GQA repeat），因为 300I Duo 的 PFA 算子要求 `q_len == k_len`，而 suffix 的 q_len ≠ k_len（suffix_len ≠ prefix_len + suffix_len）。

### 4.1 模型导出 ONNX

```bash
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu \
  --enable-common-prefix
```

`--enable-common-prefix` 导出两个模型：

- **Prefix 模型**：输入公共前缀 token，输出 KV cache `[56, batch, 8, prefix_len, 128]`
- **Suffix 模型**：输入用户 suffix token + prefix KV cache，输出最后 token logits `[batch, 1, 151936]`

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill_prefix.onnx   # 输入 (input_ids, attention_mask, position_ids)，输出 KV cache
└── qwen3_llm_prefill_suffix.onnx   # 输入 (input_ids, attention_mask, position_ids, past_key_values)，输出 logits
```

#### ONNX 模型输入输出 Shape

**Prefix** — `qwen3_llm_prefill_prefix.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, prefix_len)`  | int32   | 公共前缀 token IDs |
| 输入 | `attention_mask` | `(batch, prefix_len)`  | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, prefix_len)`  | int32   | 位置 ID            |
| 输出 | `past_kv`        | `(56, batch, 8, prefix_len, 128)` | float32 | 公共前缀 KV cache |

**Suffix** — `qwen3_llm_prefill_suffix.onnx`

| 方向  | 名称               | Shape                          | Dtype   | 说明                     |
|-----|------------------|--------------------------------|---------|------------------------|
| 输入 | `input_ids`      | `(batch, suffix_len)`          | int32   | 用户输入 suffix token IDs |
| 输入 | `attention_mask` | `(batch, prefix_len + suffix_len)` | int32   | 累积注意力掩码              |
| 输入 | `position_ids`   | `(batch, suffix_len)`          | int32   | suffix 位置 ID          |
| 输入 | `past_key_values` | `(56, batch, 8, prefix_len, 128)` | float32 | Prefix 模型输出的 KV cache |
| 输出 | `logits`         | `(batch, 1, 151936)`           | float32 | 最后一个 token 的 logits  |

### 4.2 ONNX 转 MindIR

```bash
# Prefix 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_prefix.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill_prefix \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill_prefix.ini

# Suffix 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_suffix.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill_suffix \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill_suffix.ini
```

#### 配置文件

`configs/qwen3_llm_prefill_prefix.ini`（Prefix 用，2 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;position_ids:-1,-1"
ge.dynamicDims="1,480,1,480,1,480;1,768,1,768,1,768"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./op_fp32.json"
```

`configs/qwen3_llm_prefill_suffix.ini`（Suffix 用，2 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;position_ids:-1,-1;past_key_values:56,-1,8,-1,128"
ge.dynamicDims="1,480,1,1248,1,480,1,768;1,640,1,1408,1,640,1,768"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="./op_fp32.json"
```

> Suffix `ge.dynamicDims` 每 8 个值对应 4 个输入的 `-1` 维度：
> - `1,480` = input_ids 的 (batch, suffix_len)
> - `1,1248` = attention_mask 的 (batch, total_len)，其中 1248 = 768(prefix) + 480(suffix)
> - `1,480` = position_ids 的 (batch, suffix_len)
> - `1,768` = past_key_values 的 (batch, prefix_seq_len)

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill_prefix_graph.mindir   # 场景 C prefix 模型
├── qwen3_llm_prefill_prefix_variables/
├── qwen3_llm_prefill_suffix_graph.mindir   # 场景 C suffix 模型
└── qwen3_llm_prefill_suffix_variables/
```

### 4.3 MindSpore Lite 推理

```bash
python infer_qwen3_0.6b_mindir.py \
  --mode common_prefix \
  --prefix-model ./qwen3_onnx/qwen3_llm_prefill_prefix_graph.mindir \
  --suffix-model ./qwen3_onnx/qwen3_llm_prefill_suffix_graph.mindir \
  --tokenizer ./Qwen3-0.6B \
  --prefix-text "You are a helpful assistant. Answer questions concisely." \
  --prompt "The sky is blue because of what physical phenomenon, choose from A, B, C, D? A) Rayleigh scattering B) Diffraction C) Reflection D) Refraction" \
  --prefix-seq-len 768 \
  --suffix-buckets "480,640"
```

#### 推理参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|---------------------------|
| `--prefix-model`   | Prefix MindIR 模型路径（场景 C） | 必填（场景 C）           |
| `--suffix-model`   | Suffix MindIR 模型路径（场景 C） | 必填（场景 C）           |
| `--prefix-text`    | 公共前缀文本（场景 C） | 系统提示词 |
| `--prefix-seq-len` | Prefix 模型档位（场景 C） | `768` |
| `--suffix-buckets` | Suffix seq_len 分档（场景 C） | `480,640` |

> **固定 Shape 约束**：由于使用 `ascend_oriented` 编译，推理侧输入 shape 必须匹配转换时配置的 `ge.dynamicDims` 分档之一。推理脚本会自动将输入 pad 到最近的 bucket 边界。
>
> - 场景 C prefix：分档 480, 768
> - 场景 C suffix：分档 (suffix=480, total=1248, prefix=768) 和 (suffix=640, total=1408, prefix=768)

#### 推理示例输出

```text
Mode: common_prefix
Prefix text: You are a helpful assistant. Answer questions concisely.
[prefix] tokens=12, padded to 768
Running prefix model...
Prefix model time: 98.30 ms
Prefix KV cache shape: (56, 1, 8, 768, 128)

User prompt: The sky is blue because of what physical phenomenon, ...
[suffix] tokens=48, padded to 480
Running suffix model...
Suffix model time: 65.10 ms
Output logits shape: (1, 1, 151936)
Predicted token id: 32
Decoded token: 'A'
Total time: 163.40 ms
```

#### Benchmark 命令

```bash
# Prefix Benchmark（seq=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_prefix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,768;attention_mask:1,768;position_ids:1,768"

# Suffix Benchmark（suffix=480, total=1248, prefix=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_suffix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,480;attention_mask:1,1248;position_ids:1,480;past_key_values:56,1,8,768,128"

# Suffix Benchmark（suffix=640, total=1408, prefix=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_suffix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,640;attention_mask:1,1408;position_ids:1,640;past_key_values:56,1,8,768,128"
```

### 4.4 性能数据

| 模型 | seq_len | Execute (ms) | D2H (ms) | AvgRunTime (ms) |
|------|---------|--------------|----------|-----------------|
| Prefix | 768 | 98.3 | 17.4 | 122.1 |
| Suffix | 480 | 57.1 | 0.16 | 64.3 |
| Suffix | 640 | 81.0 | 0.12 | 87.3 |

> **首次请求**：Prefix(98.3ms) + Suffix(64.1ms) = 162.4ms
> **后续请求**（prefix KV 复用）：Suffix(64.1ms) = **64.1ms**

---

## 5 . 常见问题

1. **现象**：场景 C Suffix 模型转换报错 `attention mask must be NULL, when Qs is not equal to Kvs`
   - 原因：300I Duo 芯片的 PFA 算子要求 `q_len == k_len`，Suffix 模型 q_len=480 而 k_len=1248
   - 解决方案：Suffix 模型使用非 PFA 路径（手动 matmul + softmax + matmul，含 GQA repeat），导出脚本已自动处理

2. **现象**：场景 A prefill 转换报错（5 档动态分档 + 混合精度）
   - 原因：分档数量过多时，部分子图 tiling 失败
   - 解决方案：减少分档数量，使用 2 档（128, 768）即可转换成功

3. **现象**：模型输出 `midt` 标记（thinking 模式起始标记）
   - 原因：Qwen3 默认开启 thinking 模式
   - 解决方案：`apply_chat_template(enable_thinking=False)` 禁用 thinking 模式

4. **现象**：slice_last 模型右 padding 时输出错误 token
   - 原因：右 padding 时 `logits[:, -1:]` 取到 pad token 的 logits
   - 解决方案：通过 `attention_mask.sum(dim=1)` 计算真实 last token 位置，使用 `index_select` 提取真实最后 token

5. **现象**：场景 B/C 输出错误 token（如输出 `The` 而非 `A`）
   - 原因：推理脚本未添加 system prompt，模型未被告知"直接输出选项"
   - 解决方案：使用 `--system-prompt` 参数（场景 B）或将 system prompt 作为 `--prefix-text`（场景 C）

---

## 6. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-0.6B 官方文档](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime 文档](https://onnxruntime.ai/docs/)

---

## 7. 许可证

本教程遵循 Qwen3-0.6B 模型的许可证。
