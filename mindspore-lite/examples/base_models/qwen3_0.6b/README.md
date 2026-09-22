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
- **slice_last + Slice 前移 + ArgMax 入图**：只取最后一个真实 token 的 hidden state 过 lm_head（lm_head MatMul FLOPs 降低 `seq_len` 倍），并在图内完成 `argmax`。图输出为 `token_id [batch, 1, 1] INT64`（8 bytes），替代完整 logits（~600 KB），最大化削减 D2H 传输。仅支持 greedy decoding
- **免拷贝（Zero-Copy）Decode**：场景 A decode 使用 CANN Scatter 算子原位更新 KV cache（present 形状 == past 形状），推理侧 ping-pong 双 device 缓冲每步交换，KV 不回主机。每步主机流量仅 `input_ids(4B) + attention_mask(≤4KB) + position_ids(4B)` H2D + `token_id(8B)` D2H

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
| torch          | 2.8.0 |
| transformers   | 5.4.0 |
| onnx           | 1.19.1 |
| onnxruntime    | 1.24.2 |
| numpy          | 1.26.4 |
| CANN           | 8.5    |
| mindspore-lite | 2.9.0  |

```bash
pip install transformers==5.4.0 torch==2.8.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4
```

### 权重准备

从 HuggingFace 下载 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 权重，解压后放到本地目录（本文以 `./Qwen3-0.6B` 为例）。

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

---

## 2. 场景 A：通用 prefill + decode

适用于通用对话生成场景。Prefill 一次性处理完整 prompt，输出首 token_id（ArgMax 已入图）与初始 KV cache；Decode 基于 KV cache 做自回归增量生成（greedy decoding），使用 **Scatter 原位写入免拷贝**优化，KV cache 在 device 侧 ping-pong 流转，每步不回主机。

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
| `--prefill-only-without-cache` | 只导出 prefill（无 KV cache 输出，单输出 `token_id`），一般用于场景 B | 关闭 |
| `--enable-common-prefix` | 导出 prefix + suffix 模型，一般用于场景 C 公共前缀缓存 | 关闭 |
| `--disable-fusion` | 关闭所有融合（导出 non-fused 基线） | 关闭 |

> Slice 前移 + ArgMax 入图为默认行为，无需任何开关：prefill/decode 图均输出 `token_id [batch,1,1] int64`（场景 B/C 同理）。Decode 默认使用 Scatter 原位写入免拷贝（present 形状 == past 形状）。因此图内仅支持 greedy decoding；如需 top-k/top-p 采样或 beam search，请基于本文流程导出含完整 logits 的自定义模型。

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill.onnx   # Prefill（输出 [batch, 1, 1] token_id + KV cache）
└── qwen3_llm_decode.onnx    # Decode（Scatter 免拷贝 + ArgMax 入图）
```

#### ONNX 模型输入输出 Shape

**LLM Prefill** — `qwen3_llm_prefill.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, seq_len)`     | int32   | 输入 token IDs     |
| 输入 | `attention_mask` | `(batch, seq_len)`     | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, seq_len)`     | int32   | 位置 ID            |
| 输出 | `token_id`       | `(batch, 1, 1)`        | int64   | 最后一个真实 token 的 argmax 结果（ArgMax 已入图） |
| 输出 | `present_key_values` | `(56, batch, 8, seq_len, 128)` | float16 | 初始 KV cache（56 = 28 × 2） |

**LLM Decode（Scatter 免拷贝 + ArgMax 入图）** — `qwen3_llm_decode.onnx`

| 方向  | 名称               | Shape                          | Dtype   | 说明                     |
|-----|------------------|--------------------------------|---------|------------------------|
| 输入 | `input_ids`      | `(1, 1)`                       | int32   | 单步 token               |
| 输入 | `attention_mask` | `(1, gear)`                    | int32   | ones 在 `[0, valid_len+1)`，其余为 0 |
| 输入 | `position_ids`   | `(1, 1)`                       | int32   | 实际写入位置 = valid_len |
| 输入 | `past_key_values` | `(56, 1, 8, gear, 128)`       | float16 | 档位尺寸 padding buffer |
| 输出 | `token_id`       | `(1, 1, 1)`                    | int64   | ArgMax 入图（8B D2H） |
| 输出 | `present_key_values` | `(56, 1, 8, gear, 128)`    | float16 | **与输入同形状**（Scatter 原位写入） |

> **免拷贝设计**：`past_key_values` 为档位尺寸的 padding buffer，图内 CANN Scatter 算子（`reduce=update, axis=-2`）将新 K/V 原位写入 `position_ids` 位置，present 输出形状 == past 输入形状。推理侧 ping-pong 双 device 缓冲每步交换，KV 完全在 device 侧流转，不回主机。

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
├── qwen3_llm_prefill.ini           # 场景 A/B Prefill（4 档动态分档）
├── qwen3_llm_decode.ini            # 场景 A Decode（4 档免拷贝分档）
├── qwen3_llm_prefill_prefix.ini    # 场景 C Prefix（2 档）
├── qwen3_llm_prefill_suffix.ini    # 场景 C Suffix（8 档）
└── op_fp32.json                    # 混合精度黑名单
```

`configs/qwen3_llm_prefill.ini`（场景 A/B Prefill 用，4 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;position_ids:-1,-1"
ge.dynamicDims="1,128,1,128,1,128;1,512,1,512,1,512;1,1024,1,1024,1,1024;1,2048,1,2048,1,2048"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="configs/op_fp32.json"
```

`configs/qwen3_llm_decode.ini`（场景 A Decode 用，4 档免拷贝，档位 256/640/1152/2176）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_values:56,1,8,-1,128"
ge.dynamicDims="256,256;640,640;1152,1152;2176,2176"

[acl_init_options]
ge.exec.precision_mode=force_fp32
```

- `past_key_values:56,1,8,-1,128`：56 = 28 层 × 2（K+V），1 = batch，8 = num_kv_heads，-1 = 动态 KV 长度（档位值），128 = head_dim
- `ge.dynamicDims`：4 个分档对应 `gear ∈ {256, 640, 1152, 2176}`（每档的两个值分别是 `attention_mask` 长度和 `past_key_values` 的 KV 长度，均为档位值）
- 免拷贝设计：present 输出形状 == past 输入形状（padding 保持），档位内 KV shape 不变支持 ping-pong swap，档位切换时一次性拷贝有效前缀
- 档位选择：gear = prefill_bucket + 128（prefill 4 档 128/512/1024/2048 + 128 → decode 4 档 256/640/1152/2176），保证 prefill 后 KV cache 可就近落入 decode 档位

> 转换日志中出现 `Can't find OpAdapter for Scatter` 警告是正常的（ACL pass 无法识别 ONNX Custom 节点，但 ATC mapper 会接管），只需检查最终输出 `CONVERT RESULT SUCCESS:0`。

`configs/op_fp32.json`（混合精度黑名单）：

```json
{
    "black-list": {
        "to-add": ["RealDiv", "SquareSumV1", "Square", "Sqrt", "ReduceMean"]
    }
}
```

#### 产出

模型文件超过 2GB 时，会分成 `*_graph.mindir` 和 `*_variables/` 目录：

```text
qwen3_onnx/
├── qwen3_llm_prefill_graph.mindir           # Prefill 主图（含动态分档）
├── qwen3_llm_prefill_variables/             # Prefill 权重
│   └── data_0
├── qwen3_llm_decode_graph.mindir            # Decode 主图（Scatter 免拷贝）
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
| `--prefill-buckets` | Prefill seq_len 分档（**必须与转换 ini 的 `ge.dynamicDims` 完全一致**） | `128,512,1024,2048` |
| `--decode-buckets` | Decode KV cache 分档（**必须与转换 ini 的 `ge.dynamicDims` 完全一致**） | `256,640,1152,2176` |
| `--no-chat-template` | 关闭 chat template，进入 raw completion 模式 | 关闭                       |
| `--device`         | 推理设备（ascend/cpu）                    | `ascend`                  |
| `--device-id`      | Ascend 设备 ID                          | `0`                       |

> `--decode-buckets` 必须与转换时 `ge.dynamicDims` 中的档位列表**完全一致**，否则会触发 `aclmdlSetInputDynamicDims failed`。免拷贝推理使用 ping-pong 双 device 缓冲：每档位预分配两个 gear-sized KV buffer，decode 每步 Scatter 原位写入 + 交换缓冲，KV 完全在 device 侧流转。档位切换时一次性 D2H+H2D 拷贝有效前缀到新档位缓冲 + untimed prime（触发新档位 resize），最多 3 次/生成（256→640→1152→2176）。
>
> `--prefill-buckets` 必须与 `qwen3_llm_prefill.ini` 中 `ge.dynamicDims` 的 `seq_len` 列表**完全一致**。脚本会把 prompt tokenization 后的实际 seq_len 向上 pad 到最近的 prefill bucket（补 `pad_token` + `attention_mask=0`），Prefill 后把 KV cache 传给 Decode 的 device 缓冲；首 token 直接取图输出 `token_id`（ArgMax 已入图）。

#### 推理示例输出

```text
Initializing MindSpore Lite context for ascend...
Loading prefill model from ./qwen3_onnx/qwen3_llm_prefill_graph.mindir...
Loading decode model from ./qwen3_onnx/qwen3_llm_decode_graph.mindir...
Loading tokenizer from ./Qwen3-0.6B...

============================================================
Mode: prefill_decode
Input Prompt: 你好，请介绍一下你自己。
============================================================
[prefill] seq_len=13 -> bucket=128 (pad 115)
Running LLM prefill...
Prefill time: 34.81 ms
Running LLM decode (zero-copy ping-pong KV on device)...
[decode] start gear=256
[decode] prefill->decode KV handoff: H2D 7.47 ms
Total decode time: 1755.65 ms, avg decode step: 13.82 ms, steps: 127
  gear 256: 127 steps, avg 13.82 ms
Gear-switch/handoff one-time costs (copy+prime): 1 (total 7.47 ms, excluded from decode avg)
Total time: 1790.46 ms, throughput: 71.49 tok/s
Generated token ids: [151667, 198, 99692, 3837, 20002, 56007, 97611, 100157, ...]

============================================================
Generated Response:
好的，用户问我的介绍。我需要先确认用户的需求是什么。可能他们想了解我的功能，或者想进行互动。我应该保持友好和专业的态度，同时提供有用的信息。

首先，我应该简要介绍我的功能，比如处理各种请求、提供帮助等。然后，可以提到我的特点，比如多语言支持、快速响应等。同时，要确保回答清晰，避免使用过于技术化的术语，让用户容易理解。

另外，用户可能没有明确说明他们的需求，所以需要保持开放，鼓励他们提出问题。最后，检查回答是否符合所有要求，没有遗漏任何信息
============================================================
```

> - `avg decode step` 计时口径为完整步进开销（H2D 输入 + NPU 计算 + token_id D2H），KV cache 在 device 侧 ping-pong 不回主机。
> - 档位切换开销（D2H+H2D+prime）不计入 decode avg。
> - 场景 A 默认开启 Qwen3 thinking 模式（`enable_thinking=True`），输出含思考过程（"好的，用户问我的介绍..."）。场景 B/C 关闭 thinking 模式，直接输出选项/答案。

### 2.4 性能数据

#### 测试环境

| 项目   | 配置                     |
|------|------------------------|
| 硬件   | Atlas 300I Duo |
| 模型   | Qwen3-0.6B（28 层，16 attn heads，8 KV heads GQA，head_dim=128） |
| 精度   | Prefill: allow_mix_precision + op_fp32.json 黑名单 + Slice前移+ArgMax入图；Decode: force_fp32 + Scatter免拷贝 + ArgMax入图 |
| 推理脚本 | `infer_qwen3_0.6b_mindir.py`（ping-pong 双 device 缓冲 + Scatter 免拷贝） |

#### 各阶段推理输入 Shape 与性能

**LLM Prefill（Slice 前移 + ArgMax 入图 + mix 精度）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids` |
| input_ids Shape  | `(1, seq_len)`，运行时 pad 到最近 bucket 边界 |
| 输出 token_id Shape | `(1, 1, 1)` int64（ArgMax 入图，D2H 仅 8 bytes） |
| 输出 present_key_values Shape | `(56, 1, 8, seq_len_bucket, 128)` float16 |
| 稳态耗时（s=128 bucket） | **34 ms** |

**LLM Decode（force_fp32，Scatter 免拷贝 + ArgMax 入图，单步全开销口径）**

| 项目           | 值                               |
|--------------|----------------------------------|
| 输入名称        | `input_ids`, `attention_mask`, `position_ids`, `past_key_values` |
| input_ids Shape  | `(1, 1)`                        |
| attention_mask Shape | `(1, gear)`             |
| past_key_values Shape | `(56, 1, 8, gear, 128)`   |
| 输出 token_id Shape | `(1, 1, 1)` int64（ArgMax 入图，D2H 仅 8 bytes） |
| 输出 present_key_values Shape | `(56, 1, 8, gear, 128)` float16（与输入同形状） |
| 单步平均耗时（gear=256） | **13.82 ms**（KV 不回主机） |
| 单步平均耗时（gear=640） | **~20 ms**（KV 不回主机，KV 长度增长约 2.5×） |

> 免拷贝优化使每步主机流量从 ~14MB（KV D2H+H2D）降至 `input_ids(4B) + attention_mask(≤4KB) + position_ids(4B)` H2D + `token_id(8B)` D2H，KV 完全在 device 侧 ping-pong。

#### 端到端推理性能（128 tokens 生成，冷启动进程）

| Prompt | Prompt seq_len | Prefill (ms) | Avg decode (ms) | Total (ms) | 吞吐 (tok/s) |
|--------|----------------|--------------|-----------------|------------|--------------|
| 你好，请介绍一下你自己。 | 13 | 34.81 | 13.82 | 1755.65 | **71.49** |
| The sky is blue because of ... (A/B/C/D) | 44 | 35.34 | 12.73 | 1652.02 | **77.48** |

#### 各档位 Benchmark 性能（loopCount=10, warmUpLoopCount=3, GLOG_v=1）

GLOG_v=1 抓取 ACL `Model execute` 耗时（纯 NPU 计算耗时，排除 H2D/D2H 拷贝开销），反映模型真实计算性能。AvgRunTime 为 Benchmark 工具单次执行总耗时口径（含 H2D 输入 + NPU 计算 + D2H 输出）。

> 测试机器为：Atlas 300I Duo

| 模型 | 档位 | Model execute (ms) | AvgRunTime (ms) |
|------|------|--------------------|-----------------|
| Prefill | seq=128  | 17.62  | 32.55  |
| Prefill | seq=512  | 38.63  | 64.26  |
| Prefill | seq=1024 | 86.16  | 123.30 |
| Prefill | seq=2048 | 235.18 | 296.86 |
| Decode  | gear=256  | 14.19 | 34.08  |
| Decode  | gear=640  | 20.13 | 56.01  |
| Decode  | gear=1152 | 29.47 | 80.97  |
| Decode  | gear=2176 | 57.17 | 140.64 |

> 注：Decode 的 AvgRunTime 含 117MB KV cache 的 H2D 拷贝开销（~20ms），显著高于 Model execute；实际 infer 脚本使用 ping-pong 免拷贝，KV 不回主机，gear=256 单步仅 **13.82ms**（见上表端到端推理性能）。

#### 精度验证（greedy decoding，对比 torch npu fp16/fp32 eager）

| Prompt | 与 HF fp32 greedy 对比 |
|--------|----------------------|
| The sky is blue because of ... (A/B/C/D) | **128/128 token 完全一致（100%）** |
| 你好，请介绍一下你自己。 | **128/128 token 完全一致（100%）** |

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

- 不输出 KV cache（单输出：`token_id`，ArgMax 已入图）
- 不导出 decode 模型

#### 产出

```text
qwen3_onnx/
└── qwen3_llm_prefill.onnx   # 输出 [batch, 1, 1] token_id（场景 B）
```

> 场景 A 和场景 B 导出的 prefill 模型文件名相同（`qwen3_llm_prefill.onnx`），但模型结构不同（场景 A 有 KV cache 输出，场景 B 没有）。如需同时使用两个场景，请导出到不同目录。

#### ONNX 模型输入输出 Shape

**LLM Prefill (Slice 前移 + ArgMax 入图, no cache)** — `qwen3_llm_prefill.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, seq_len)`     | int32   | 输入 token IDs     |
| 输入 | `attention_mask` | `(batch, seq_len)`     | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, seq_len)`     | int32   | 位置 ID            |
| 输出 | `token_id`       | `(batch, 1, 1)`        | int64   | 最后一个真实 token 的 argmax 结果 |

> **右 padding 处理**：当输入右 padding 时，模型通过 `attention_mask.sum(dim=1)` 计算真实 last token 位置，使用 `index_select` 提取真实最后 token 的 hidden state 过 lm_head 并 `argmax`，避免取到 pad token。

### 3.2 ONNX 转 MindIR

```bash
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill.onnx \
  --outputFile=./qwen3_onnx/qwen3_llm_prefill \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill.ini
```

配置文件与场景 A 相同（`configs/qwen3_llm_prefill.ini`），场景 B 使用 128, 512, 1024, 2048 分档。

#### 产出

```text
qwen3_onnx/
├── qwen3_llm_prefill_graph.mindir   # 场景 B（ArgMax 入图，无 KV cache）
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
  --system-prompt "You are a helpful assistant. Answer questions concisely." \
  --prefill-buckets "128,512,1024,2048"
```

#### 推理示例输出

```text
============================================================
Mode: prefill_only
Input Prompt: The sky is blue because of what physical phenomenon, choose from A, B, C, D? A) Rayleigh scattering B) Diffraction C) Reflection D) Refraction
============================================================
[prefill] seq_len=65 -> bucket=128 (pad 63)
Running LLM prefill...
Prefill time: 15.85 ms
Output token_id shape: (1, 1, 1)
Predicted token id: 32
Decoded token: 'A'

============================================================
First token id:    32
Decoded token:     'A'
Prefill latency:   15.85 ms
============================================================
```

#### Benchmark 命令

```bash
# Scene B Benchmark（seq=128）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,128;attention_mask:1,128;position_ids:1,128"

# Scene B Benchmark（seq=512)
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_llm_prefill_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,512;attention_mask:1,512;position_ids:1,512"
```

### 3.4 性能数据

Benchmark 实测（loopCount=10, warmUpLoopCount=3），Prefill 模型已融合 Slice 前移 + ArgMax 入图优化：

> 测试机器为：Atlas 300I Duo

| bucket | Execute (ms) | D2H (ms) | AvgRunTime (ms) |
|--------|--------------|----------|-----------------|
| 128 | 13.96 | 0.049 | 14.52 |
| 512 | 34.94 | 0.031 | 34.54 |
| 1024 | 83.31 | 0.033 | 83.52 |
| 2048 | 233.57 | 0.043 | 233.01 |

> 默认开启 ArgMax fusion 后，D2H 从 ~0.12 ms（logits `[1,1,151936]` FP32 ≈ 600 KB）降至 ~0.04 ms（token_id `[1,1,1]` INT64 = 8 bytes）。ArgMax 入图为默认行为且三场景统一（无开关）；如需 top-k/top-p sampling 或 beam search，请基于本文流程导出含完整 logits 的自定义模型。

---

## 4. 场景 C：公共前缀（prefix + suffix）

适用于有固定公共前缀（如 system prompt）的场景。Prefix 模型处理公共前缀 token，输出 KV cache；Suffix 模型接收 prefix KV + 用户输入，输出最后 token 的 token_id（ArgMax 已入图）。后续请求可复用 prefix KV cache，只需运行 suffix 模型。

> **场景 C 注意**：场景 C 默认不开启 PFA 融合（attention 走原生 PyTorch 计算）。如需使能 PFA 融合，Suffix 模型需使用 `InnerPromptFlashAttention` 算子（支持 `q_len != k_len`），因为标准 `PromptFlashAttention` 算子在 300I Duo 上要求 `q_len == k_len`，而 suffix 的 `q_len != k_len`（suffix_len != prefix_len + suffix_len）。场景 A/B 仍使用标准 `PromptFlashAttention`。
> **使能 InnerPromptFlashAttention 前置条件**：需使用 MSLite 2.11 及之后的版本，并安装自定义算子包：
> ```bash
> # 安装自定义算子包
> bash <mslite-tar-path>/tools/custom_kernels/install.sh
> # 设置环境变量
> source <cann-path>/ascend-toolkit/latest/opp/vendors/mslite_custom_ops/bin/set_env.bash
> ```
> 安装完成后再执行 ONNX→MindIR 转换即可自动使能 InnerPromptFlashAttention 算子。

### 4.1 模型导出 ONNX

```bash
# 默认不开启 PFA 融合
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu \
  --enable-common-prefix

# 如需使能 PFA 融合（需安装自定义算子包，见上方前置条件）
python export_qwen3_onnx.py \
  --model-id ./Qwen3-0.6B \
  --output-dir ./qwen3_onnx \
  --device cpu \
  --enable-common-prefix \
  --enable-pfa
```

`--enable-common-prefix` 导出两个模型：

- **Prefix 模型**：输入公共前缀 token，输出 KV cache `[56, batch, 8, prefix_len, 128]`
- **Suffix 模型**：输入用户 suffix token + prefix KV cache，输出 token_id `[batch, 1, 1]`（ArgMax 已入图）

#### 产出

```text
qwen3_onnx/
├── qwen3_prefix.onnx   # 输入 (input_ids, attention_mask, position_ids)，输出 KV cache
└── qwen3_suffix.onnx   # 输入 (input_ids, attention_mask, position_ids, past_key_values)，输出 token_id
```

#### ONNX 模型输入输出 Shape

**Prefix** — `qwen3_prefix.onnx`

| 方向  | 名称               | Shape                  | Dtype   | 说明               |
|-----|------------------|------------------------|---------|------------------|
| 输入 | `input_ids`      | `(batch, prefix_len)`  | int32   | 公共前缀 token IDs |
| 输入 | `attention_mask` | `(batch, prefix_len)`  | int32   | 注意力掩码           |
| 输入 | `position_ids`   | `(batch, prefix_len)`  | int32   | 位置 ID            |
| 输出 | `past_kv`        | `(56, batch, 8, prefix_len, 128)` | float16 | 公共前缀 KV cache |

**Suffix** — `qwen3_suffix.onnx`

| 方向  | 名称               | Shape                          | Dtype   | 说明                     |
|-----|------------------|--------------------------------|---------|------------------------|
| 输入 | `input_ids`      | `(batch, suffix_len)`          | int32   | 用户输入 suffix token IDs |
| 输入 | `attention_mask` | `(batch, prefix_len + suffix_len)` | int32   | 累积注意力掩码              |
| 输入 | `position_ids`   | `(batch, suffix_len)`          | int32   | suffix 位置 ID          |
| 输入 | `past_key_values` | `(56, batch, 8, prefix_len, 128)` | float16 | Prefix 模型输出的 KV cache |
| 输出 | `token_id`       | `(batch, 1, 1)`                | int64   | 最后一个 token 的 argmax 结果 |

### 4.2 ONNX 转 MindIR

```bash
# Prefix 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_prefix.onnx \
  --outputFile=./qwen3_onnx/qwen3_prefix \
  --optimize=ascend_oriented \
  --saveType=MINDIR \
  --configFile=./configs/qwen3_llm_prefill_prefix.ini

# Suffix 转换
$Convert --fmk=ONNX \
  --modelFile=./qwen3_onnx/qwen3_suffix.onnx \
  --outputFile=./qwen3_onnx/qwen3_suffix \
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
ge.exec.modify_mixlist="configs/op_fp32.json"
```

`configs/qwen3_llm_prefill_suffix.ini`（Suffix 用，8 档）：

```ini
[acl_build_options]
input_format="ND"
input_shape="input_ids:-1,-1;attention_mask:-1,-1;position_ids:-1,-1;past_key_values:56,-1,8,-1,128"
ge.dynamicDims="1,32,1,800,1,32,1,768;1,64,1,832,1,64,1,768;1,96,1,864,1,96,1,768;1,128,1,896,1,128,1,768;1,256,1,1024,1,256,1,768;1,384,1,1152,1,384,1,768;1,512,1,1280,1,512,1,768;1,640,1,1408,1,640,1,768"

[acl_init_options]
ge.exec.precision_mode=allow_mix_precision
ge.exec.modify_mixlist="configs/op_fp32.json"
```

> Suffix `ge.dynamicDims` 每 8 个值对应 4 个输入的 `-1` 维度（total_len = prefix_len 768 + suffix_len）：
> - `1,32` = input_ids 的 (batch, suffix_len)，total_len=800=768+32
> - `1,64` = input_ids 的 (batch, suffix_len)，total_len=832=768+64
> - `1,96` = input_ids 的 (batch, suffix_len)，total_len=864=768+96
> - `1,128` = input_ids 的 (batch, suffix_len)，total_len=896=768+128
> - `1,256` = input_ids 的 (batch, suffix_len)，total_len=1024=768+256
> - `1,384` = input_ids 的 (batch, suffix_len)，total_len=1152=768+384
> - `1,512` = input_ids 的 (batch, suffix_len)，total_len=1280=768+512
> - `1,640` = input_ids 的 (batch, suffix_len)，total_len=1408=768+640

#### 产出

```text
qwen3_onnx/
├── qwen3_prefix_graph.mindir   # 场景 C prefix 模型
├── qwen3_prefix_variables/
├── qwen3_suffix_graph.mindir   # 场景 C suffix 模型
└── qwen3_suffix_variables/
```

### 4.3 MindSpore Lite 推理

```bash
python infer_qwen3_0.6b_mindir.py \
  --mode common_prefix \
  --prefix-model ./qwen3_onnx/qwen3_prefix_graph.mindir \
  --suffix-model ./qwen3_onnx/qwen3_suffix_graph.mindir \
  --tokenizer ./Qwen3-0.6B \
  --prefix-text "You are a helpful assistant. Answer questions concisely." \
  --prompt "The sky is blue because of what physical phenomenon, choose from A, B, C, D? A) Rayleigh scattering B) Diffraction C) Reflection D) Refraction" \
  --prefix-seq-len 768 \
  --suffix-buckets "128,256,384,512,640"
```

#### 推理参数说明

| 参数                 | 说明                                    | 默认值                     |
|--------------------|---------------------------------------|---------------------------|
| `--prefix-model`   | Prefix MindIR 模型路径（场景 C） | 必填（场景 C）           |
| `--suffix-model`   | Suffix MindIR 模型路径（场景 C） | 必填（场景 C）           |
| `--prefix-text`    | 公共前缀文本（场景 C） | 系统提示词 |
| `--prefix-seq-len` | Prefix 模型档位（场景 C） | `768` |
| `--suffix-buckets` | Suffix seq_len 分档（场景 C） | `128,256,384,512,640` |

> **固定 Shape 约束**：由于使用 `ascend_oriented` 编译，推理侧输入 shape 必须匹配转换时配置的 `ge.dynamicDims` 分档之一。推理脚本会自动将输入 pad 到最近的 bucket 边界。
>
> - 场景 C prefix：分档 480, 768
> - 场景 C suffix：默认分档 (suffix=128, total=896), (suffix=256, total=1024), (suffix=384, total=1152), (suffix=512, total=1280), (suffix=640, total=1408)，prefix 均为 768（转换配置 `qwen3_llm_prefill_suffix.ini` 包含 8 档，推理可使用其子集）

#### 推理示例输出

```text
============================================================
Mode: common_prefix
Prefix text: You are a helpful assistant. Answer questions concisely.
============================================================
[prefix] tokens=17, padded to 768
Running prefix model...
Prefix KV cache shape: (56, 1, 8, 768, 128)
Prefix model time: 68.18 ms

============================================================
User prompt: The sky is blue because of what physical phenomenon, choose from A, B, C, D? A) Rayleigh scattering B) Diffraction C) Reflection D) Refraction
============================================================
[suffix] tokens=48, padded to 128
Running suffix model...
Suffix model time: 20.69 ms
Output token_id shape: (1, 1, 1)
Predicted token id: 32
Decoded token: 'A'

============================================================
Prefix model time:  68.18 ms
Suffix model time:  20.69 ms
Total time:         88.86 ms
Predicted token id: 32
Decoded token:      'A'
============================================================
```

#### Benchmark 命令

```bash
# Prefix Benchmark（seq=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_prefix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,768;attention_mask:1,768;position_ids:1,768"

# Suffix Benchmark（suffix=32, total=800, prefix=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_suffix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,32;attention_mask:1,800;position_ids:1,32;past_key_values:56,1,8,768,128"

# Suffix Benchmark（suffix=512, total=1280, prefix=768）
$Benchmark \
  --modelFile=./qwen3_onnx/qwen3_suffix_graph.mindir \
  --device=Ascend \
  --inputShape="input_ids:1,512;attention_mask:1,1280;position_ids:1,512;past_key_values:56,1,8,768,128"
```

#### 性能数据

Benchmark 实测（loopCount=10, warmUpLoopCount=3），Suffix 模型已融合 Slice 前移 + ArgMax 入图优化：

> 测试机器为：Atlas 300I Duo

| 模型 | seq_len | total_len | Execute (ms) | D2H (ms) | AvgRunTime (ms) |
|------|---------|-----------|--------------|----------|-----------------|
| Prefix | 768 | 768 | 63.5 | 17.7 | 87.42 |
| Suffix | 32 | 800 | 16.32 | 0.040 | 21.92 |
| Suffix | 64 | 832 | 21.23 | 0.037 | 25.17 |
| Suffix | 96 | 864 | 21.04 | 0.038 | 27.79 |
| Suffix | 128 | 896 | 22.23 | 0.049 | 28.89 |
| Suffix | 256 | 1024 | 32.14 | 0.035 | 39.76 |
| Suffix | 384 | 1152 | 54.13 | 0.035 | 59.26 |
| Suffix | 512 | 1280 | 62.51 | 0.057 | 69.43 |
| Suffix | 640 | 1408 | 80.81 | 0.032 | 87.33 |

> **首次请求**：Prefix(63.5ms) + Suffix(16.32ms) = 79.82ms（免拷贝，suffix=64 档位）
> **后续请求**（prefix KV 复用）：Suffix(16.32ms) = **16.32ms**

---

## 5 . 常见问题

1. **现象**：场景 C Suffix 模型转换报错 `attention mask must be NULL, when Qs is not equal to Kvs`
   - 原因：300I Duo 芯片的标准 PFA 算子要求 `q_len == k_len`，Suffix 模型 q_len=128 而 k_len=896
   - 解决方案：Suffix 模型使用 `InnerPromptFlashAttention` 算子（支持 `q_len != k_len`），需安装 MSLite 自定义算子包（见场景 C 前置条件）

2. **现象**：场景 A prefill 转换报错（动态分档过多 + 混合精度）
   - 原因：分档数量过多时，部分子图 tiling 失败
   - 解决方案：减少分档数量（当前仓库配置为 4 档 128/512/1024/2048，已验证可正常转换）

3. **现象**：模型输出 `midt` 标记（thinking 模式起始标记）
   - 原因：Qwen3 默认开启 thinking 模式
   - 解决方案：`apply_chat_template(enable_thinking=False)` 禁用 thinking 模式

4. **现象**：prefill 模型右 padding 时输出错误 token
   - 原因：右 padding 时若简单取 `[:, -1:]` 会取到 pad token 的输出
   - 解决方案：通过 `attention_mask.sum(dim=1)` 计算真实 last token 位置，使用 `index_select` 提取真实最后 token 的 hidden state（已内置于导出图与推理脚本）

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
