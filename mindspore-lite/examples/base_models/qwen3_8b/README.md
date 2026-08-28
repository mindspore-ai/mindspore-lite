# Qwen3-8B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-8B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend 300I Duo NPU 上完成端到端推理部署。

Qwen3-8B 是一个纯文本大语言模型，采用标准 Transformer 全注意力架构。模型被拆分为 2 个 ONNX 文件：

1. **LLM Prefill**（`qwen3_8b_llm_prefill_rank*.onnx`）：一次性处理完整 prompt，输出 logits 与 decode 兼容的 KV cache
2. **LLM Decode**（`qwen3_8b_llm_decode_rank*.onnx`）：基于 past KV cache 做自回归增量生成

Ascend 300I Duo 单卡包含 2 个 NPU 芯（chip），因此：

- **1p = 单卡单芯**：`infer_qwen3_8b_mslite_1p.py`，单 mindir + 单图 `ge.dynamicDims` 8 档，KV 8 头/卡
- **2p = 单卡双芯**（卡内双芯 HCCS 互联，本机已验证）：`infer_qwen3_8b_mslite_tp.py`，HCCL 多进程，`ge.dynamicDims` 8 档，KV 4 头/rank

---

## 模型架构

### 架构参数

| 参数 | 值 |
|------|------|
| hidden_size | 4096 |
| num_attention_heads | 32 |
| num_hidden_layers | 36 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| vocab_size | 151936 |

### TP 切分策略

- **QKV 投影**：列并行，每卡持 `32/TP` 个 q 头、`8/TP` 个 kv 头
- **o_proj / down_proj / lm_head**：行并行，每卡计算局部结果后插入 `Custom(AllReduce)` 跨卡求和
- **KV cache shape**：`[36, 1, 8/TP, kv_len, 128]`（1p=8 头，2p=4 头，4p=2 头）

### 动态分档设计（1p/2p 均八档）

`kv_len = prefill_seq + 512`，每档最多可生成 512 个新 token。

**1p（八档）**：对应 `configs/qwen3_8b_llm_prefill.config` / `qwen3_8b_llm_decode.config`

| prefill_seq | 适用输入长度 | kv_len | 最大输出 tokens |
|------------:|:-----------:|-------:|:--------------:|
| 512  | ≤ 512 | 1024 | 512 |
| 896  | 513–896 | 1408 | 512 |
| 1024 | 897–1024 | 1536 | 512 |
| 1664 | 1025–1664 | 2176 | 512 |
| 2048 | 1665–2048 | 2560 | 512 |
| 2560 | 2049–2560 | 3072 | 512 |
| 2816 | 2561–2816 | 3328 | 512 |
| 3072 | 2817–3072 | 3584 | 512 |

**2p（八档）**：对应 `configs/qwen3_8b_llm_prefill.config` / `qwen3_8b_llm_decode.config`

| prefill_seq | 适用输入长度 | kv_len | 最大输出 tokens |
|------------:|:-----------:|-------:|:--------------:|
| 512  | ≤ 512 | 1024 | 512 |
| 896  | 513–896 | 1408 | 512 |
| 1024 | 897–1024 | 1536 | 512 |
| 1664 | 1025–1664 | 2176 | 512 |
| 2048 | 1665–2048 | 2560 | 512 |
| 2560 | 2049–2560 | 3072 | 512 |
| 2816 | 2561–2816 | 3328 | 512 |
| 3072 | 2817–3072 | 3584 | 512 |

**分档逻辑**：

1. 按 prompt 实际 token 数选**最小 ≥ real_len** 的 prefill 档位；
2. prefill 输入 pad 到选中档 seq（padding 在末尾）；
3. prefill KV 输出只 pad 到该档 kv_len（**不是** max 3584）；
4. decode 使用该 kv_len 档，attention_mask 长度 = kv_len；
5. 输出超过 kv_len 时截断并提示。

---

## 1. 环境准备

### 依赖版本（建议）

| 软件包            | 版本 |
|----------------|------|
| Python         | 3.11 |
| torch          | 2.8.0（仅导出 ONNX 时需要，推理不再使用） |
| transformers   | 4.51.0 |
| accelerate     | ≥ 0.26 |
| onnx           | 1.21.0 |
| mindspore-lite | 2.10.0 |
| CANN           | 9.0.1 |

```bash
pip install torch==2.8.0 transformers==4.51.0 accelerate onnx==1.21.0
```

### 环境变量

每次新终端执行（或写入 `env.sh` 后 `source`）：

```bash
source /path/to/ascend-toolkit/set_env.sh          # CANN
export MSLITE_HOME_PATH=/path/to/mindspore-lite-2.10.0-linux-aarch64
export ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/mslite_custom_ops
export CONV=$MSLITE_HOME_PATH/tools/converter/converter/converter_lite
export MODEL_ID=./Qwen3-8B
export DTYPE=fp16
export ASCEND_DEVICE_ID=0    # 1p 使用
```

### 权重获取

```bash
# 方式一：ModelScope（国内快）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='./Qwen3-8B')"

# 方式二：HuggingFace
huggingface-cli download Qwen/Qwen3-8B --local-dir ./Qwen3-8B
```

---

## 2. 模型导出 ONNX

### 一键导出 + 转换（推荐）

```bash
cd ./mindspore-lite/examples/base_models/qwen3_8b
source path/to/cann/set_env.sh
export MSLITE_HOME_PATH=path/to/mindspore_lite

bash export_and_convert.sh 1p    # 单卡：导出 + 转换
bash export_and_convert.sh 2p    # TP=2：导出 + 转换（本机已验证）
bash export_and_convert.sh 4p    # TP=4：导出 + 转换（未验证）
```

`export_and_convert.sh <1p|2p|4p>` 自动执行：

1. `export_qwen3_8b_onnx.py --tp-size N` 导出各 rank 的 prefill/decode ONNX；
2. `converter_lite --optimize=none --saveType=MINDIR` 转换为在线 GE MindIR（`optimize=none` 保留动态图，供运行时 `provider=ge` 使用）。

其中 2p 使用 `--tp-dynamic`：prefill 导出动态 seq 轴、decode 导出动态 KV 轴，一个 MindIR 即可服务 2p 的 8 档；1p 同样通过单个动态图 mindir + `model.resize()` 服务 8 档。

### 直接导出命令

```bash
# 1p：单 rank 动态轴
python3 export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_onnx \
  --device cpu \
  --dtype fp16 \
  --tp-size 1

# 2p：TP=2（--tp-dynamic 使一个 ONNX 服务 8 档）
python3 export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_tp_onnx \
  --device cpu \
  --dtype fp16 \
  --tp-size 2 \
  --tp-dynamic

# 4p：TP=4（未验证）
python3 export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_tp4_onnx \
  --device cpu \
  --dtype fp16 \
  --tp-size 4
```

### 导出参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3-8B` |
| `--output-dir` | 输出目录 | `./qwen3_8b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dtype` | 导出精度（fp16/bf16/fp32） | `fp16` |
| `--tp-size` | 张量并行规模（1/2/4） | `1` |
| `--tp-dynamic` | TP>=2 时导出动态 seq/KV 轴（配合 `ge.dynamicDims`） | `False` |
| `--dummy-seq-len` | 导出时 dummy 序列长度 | `8` |
| `--num-layers` | 调试用：只导出前 N 层（0=全部 36 层） | `0` |
| `--kv-cache-len` | 覆盖 decode 默认 KV 长度（默认 256） | `0` |
| `--use-dynamo` | 使用 torch dynamo 导出路径 | `False` |

### 产出（ONNX）

```text
qwen3_8b_onnx/            # 1p
├── prefill/qwen3_8b_llm_prefill_rank0.onnx
└── decode/qwen3_8b_llm_decode_rank0.onnx

qwen3_8b_tp_onnx/         # 2p
├── prefill/qwen3_8b_llm_prefill_rank{0,1}.onnx
└── decode/qwen3_8b_llm_decode_rank{0,1}.onnx

qwen3_8b_tp4_onnx/        # 4p
├── prefill/qwen3_8b_llm_prefill_rank{0..3}.onnx
└── decode/qwen3_8b_llm_decode_rank{0..3}.onnx
```

> rank 间图不同（真 TP 切分），每 rank 权重约 8.8GB（半个 transformer + 复制的 embed/lm_head）。

### ONNX 模型输入输出 Shape

**LLM Prefill** — `prefill/qwen3_8b_llm_prefill_rank*.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `input_ids` | `(1, seq)` | int64 | token IDs |
| 输入 | `attention_mask` | `(1, seq)` | int64 | 注意力掩码 |
| 输入 | `position_ids` | `(1, seq)` | int64 | 位置 ID |
| 输出 | `logits` | `(151936,)` | float32 | 最后一个真实 token 的 1D logits |
| 输出 | `present_key_cache` | `(36, 1, 8/TP, seq+512, 128)` | float16 | KV cache（graph 内 pad 512 空槽） |
| 输出 | `present_value_cache` | `(36, 1, 8/TP, seq+512, 128)` | float16 | KV cache |

**LLM Decode** — `decode/qwen3_8b_llm_decode_rank*.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `input_ids` | `(1, 1)` | int64 | 单步 token |
| 输入 | `attention_mask` | `(1, kv_len)` | int64 | 长度 = kv_len 的掩码 |
| 输入 | `position_ids` | `(1, 1)` | int64 | 单步位置 |
| 输入 | `past_key_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | 上一步 KV cache |
| 输入 | `past_value_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | 上一步 KV cache |
| 输出 | `logits` | `(1, 1, 151936)` | float32 | 单步 logits |
| 输出 | `present_key_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | 更新后的 KV cache |
| 输出 | `present_value_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | 更新后的 KV cache |

> `kv_len = seq + 512`；`8/TP` 为每 rank 的 KV 头数（1p=8、2p=4、4p=2）。

---

## 3. ONNX 转 MindIR

### 转换命令

`export_and_convert.sh` 已包含转换步骤，通常无需手动执行。以下为底层命令示例（1p）：

```bash
Convert=$MSLITE_HOME_PATH/tools/converter/converter/converter_lite

# Prefill 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0.onnx \
  --outputFile=qwen3_8b_onnx/prefill/qwen3_8b_llm_prefill_rank0 \
  --optimize=none \
  --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0.onnx \
  --outputFile=qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0 \
  --optimize=none \
  --saveType=MINDIR
```

2p/4p 只需对 `qwen3_8b_tp_onnx/` / `qwen3_8b_tp4_onnx/` 下的每个 rank 重复上述命令。

### 参数说明

| 参数 | 说明 |
|------|------|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出 MindIR 路径（不带扩展名） |
| `--optimize` | 必须指定 `none`（在线 GE 流程，保留动态图） |
| `--saveType` | 输出格式（MINDIR） |

### 运行时动态分档配置

1p/2p 的在线 GE 动态分档由 `ge.dynamicDims` 配置驱动，prefill/decode 各一份，位于：

- `configs/`

`infer_qwen3_8b_mslite_tp.py`（2p/4p）按设备数自动选择配置目录，也可用 `--bucket-cfg-dir` 覆盖；1p 固定使用 `configs/` 下的配置。

> 实际配置文件为 `configs/qwen3_8b_llm_prefill.config` / `configs/qwen3_8b_llm_decode.config`（1p）与 `configs/qwen3_8b_llm_prefill.config` / `configs/qwen3_8b_llm_decode.config`（2p）。

**1p prefill 配置**（`configs/qwen3_8b_llm_prefill.config`，KV 8 头/卡，8 档）：

```ini
[ascend_context]
plugin_custom_ops=All
model_cache_mode=mem_opt

[ge_session_options]
ge.constLifecycle=session
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=force_fp16
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=U292:1,-1;U293:1,-1;U294:1,-1
ge.dynamicDims=512,512,512;896,896,896;1024,1024,1024;1664,1664,1664;2048,2048,2048;2560,2560,2560;2816,2816,2816;3072,3072,3072
ge.dynamicNodeType=1
```

**1p decode 配置**（`configs/qwen3_8b_llm_decode.config`，KV 8 头/卡，8 档）：

```ini
[ascend_context]
plugin_custom_ops=All
model_cache_mode=mem_opt

[ge_session_options]
ge.constLifecycle=session
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=force_fp16
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=U292:1,1;U293:1,-1;U294:1,1;U295:36,1,8,-1,128;U296:36,1,8,-1,128
ge.dynamicDims=1024,1024,1024;1408,1408,1408;1536,1536,1536;2176,2176,2176;2560,2560,2560;3072,3072,3072;3328,3328,3328;3584,3584,3584
ge.dynamicNodeType=1
```

**2p prefill 配置**（`configs/qwen3_8b_llm_prefill.config`，KV 4 头/rank）：

```ini
[ascend_context]
plugin_custom_ops=All
model_cache_mode=mem_opt

[ge_session_options]
ge.constLifecycle=session
ge.exec.formatMode=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1
ge.dynamicDims=512,512,512;896,896,896;1024,1024,1024;1664,1664,1664;2048,2048,2048;2560,2560,2560;2816,2816,2816;3072,3072,3072
ge.dynamicNodeType=1
```

**2p decode 配置**（`configs/qwen3_8b_llm_decode.config`，KV 4 头/rank）：

```ini
[ascend_context]
plugin_custom_ops=All
model_cache_mode=mem_opt

[ge_session_options]
ge.constLifecycle=session
ge.exec.formatMode=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=must_keep_origin_dtype
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_cache:36,1,4,-1,128;past_value_cache:36,1,4,-1,128
ge.dynamicDims=1024,1024,1024;1408,1408,1408;1536,1536,1536;2176,2176,2176;2560,2560,2560;3072,3072,3072;3328,3328,3328;3584,3584,3584
ge.dynamicNodeType=1
```

### 产出（MindIR）

模型文件超过 2GB 时会拆分为 `*_graph.mindir` 与 `*_variables/` 目录：

```text
qwen3_8b_onnx/                      # 1p
├── prefill/qwen3_8b_llm_prefill_rank0_graph.mindir   (+ _variables/ 权重)
└── decode/qwen3_8b_llm_decode_rank0_graph.mindir     (+ _variables/ 权重)

qwen3_8b_tp_onnx/                   # 2p
├── prefill/qwen3_8b_llm_prefill_rank0_graph.mindir   (+ _variables/ 权重)
├── prefill/qwen3_8b_llm_prefill_rank1_graph.mindir   (+ _variables/ 权重)
├── decode/qwen3_8b_llm_decode_rank0_graph.mindir     (+ _variables/ 权重)
└── decode/qwen3_8b_llm_decode_rank1_graph.mindir     (+ _variables/ 权重)
```

---

## 4. MindSpore Lite 推理

### 一键推理

`infer.sh` 第一个参数为设备 ID 列表，第二个参数为模式（默认 `infer`）：

```bash
# 1p（单卡动态单图，8 档）
bash infer.sh 0                # 单档精度/功能验证
bash infer.sh 0 perf           # 八档性能扫描（一次编译，多档推理）

# 2p（TP=2，已验证，8 档）
bash infer.sh 0,1              # 单档精度/功能验证
bash infer.sh 0,1 perf         # 八档性能扫描（单进程多档，worker 复用）

# 4p（TP=4，未验证）
bash infer.sh 0,1,2,3          # 单档精度/功能验证
bash infer.sh 0,1,2,3 perf     # 八档性能扫描

# prof 模式（msprof 分阶段采集）
bash infer.sh 0 prof           # 1p 全档位采集
bash infer.sh 0,1 prof 1024    # 2p 只采集 1024 档
```

模式说明：

- `infer`：单 prompt → 命中一档 → 输出 + 核心点 + 显存峰值
- `perf`：逐档扫描（1p/2p 均 8 档：512/896/1024/1664/2048/2560/2816/3072），逐档 prefill/decode 计时 + 核心点汇总
- `prof`：msprof 包装，逐档 3 次 warmup + 1 次采集

### 直接调用推理脚本

**1p**（动态单图入口）：

```bash
python3 infer_qwen3_8b_mslite_1p.py \
  --device-id 0 \
  --single-prompt "你好，请用一句话介绍一下你自己" \
  --max-new-tokens 128
```

**2p/4p**（统一推理脚本，按设备数派发）：

```bash
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids 0,1 \
  --model-id ./Qwen3-8B \
  --prompt "你好，请用一句话介绍一下你自己" \
  --max-new-tokens 128 \
  --warmup 3
```

### 参数说明

**`infer_qwen3_8b_mslite_tp.py`（2p/4p 张量并行推理）**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--device-ids` | 逗号分隔设备 id（数量决定并行度：1/2/4） | 必填 |
| `--model-id` | 权重/tokenizer 路径 | `./Qwen3-8B` |
| `--prompt` | 输入 prompt | `你好，请用一句话介绍一下你自己` |
| `--max-new-tokens` | 最大生成 token 数 | `128` |
| `--warmup` | TP warmup 轮数（1p 忽略） | `3` |
| `--prefill-model` / `--decode-model` | 1p 显式指定 prefill/decode MindIR 路径 | `None`（自动解析） |
| `--prefill-ranks` / `--decode-ranks` | TP 显式指定各 rank MindIR 路径 | `None`（自动解析） |
| `--config-file` | TP 显式指定 HCCL config_file.ini | `None`（自动生成） |
| `--bucket-cfg-dir` | TP 动态分档配置目录（含 qwen3_8b_llm_prefill.config / qwen3_8b_llm_decode.config） | `configs` |
| `--prompt-tokens` | 强制合成 N 个 token 的 prompt（分档性能用） | `None` |
| `--perf-sweep` | 单进程多档性能扫描（worker 复用） | `False` |
| `--repeats` | perf sweep 每档重复轮数 | `3` |
| `--json-out` | 输出 perf dict 到 JSON 文件 | `None` |

**`infer_qwen3_8b_mslite_1p.py`（1p 动态单图）**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--device-id` | Ascend 设备 ID | `0` |
| `--max-new-tokens` | 最大生成 token 数 | `16` |
| `--buckets` | 逗号分隔的 prefill seq 档（`all` = 全 8 档） | `512,1024` |
| `--repeats` | 每档推理轮数（第 1 轮含懒编译，取第 2..N 轮稳态） | `3` |
| `--single-prompt` | 单次功能验证：跑一个真实 prompt | `None` |
| `--out` | 结果 JSON 输出路径 | `_dynamic_bucket_results.json` |
| `--prof-phase` | prof 模式：只跑 `prefill` 或 `decode` | `None` |

### 推理示例输出

> 以下为示意输出，实际数值以运行为准。

```text
=== [infer] devices=0,1 ===
Loading tokenizer from ./Qwen3-8B...
[Bucket] real_len=18 -> prefill_seq=512, kv_len=1024
Waiting for warmup...
All workers ready. Starting timed inference.
[Prefill: 18 tokens -> seq 512]
[rank0] 核心点 OK: prefill KV padded to kv_len == 1024 (seq bucket 512, NOT max 3584)
Generated Response: 你好！我是 Qwen，一个由阿里云开发的大语言模型……
--- Performance (TP=2 prefill+decode) ---
  Input tokens:     18
  Output tokens:    128
  Prefill (ms):     309.1
  Total Decode (ms): 17715.2
  Avg decode step:  138.4
```

---

## 5. 性能数据

### 测试环境

| 项目 | 说明 |
|------|------|
| 硬件 | Atlas 300I Duo（Ascend NPU） |
| 模型 | Qwen3-8B（fp16） |
| 精度 | fp16（GE dynamicDims 分档） |
| 数据来源 | 本机实测，可复现 |

### 1p 实测性能（300I DUO，fp16，单卡动态单图）

| prompt_tokens | prefill_seq | kv_len | prefill (ms) | decode/step (ms) | tok/s | 核心点 |
|--------------:|------------:|-------:|-------------:|------------------:|------:|:------:|
| 512  | 512  | 1024 | 341   | 94.7  | 10.56 | 通过 |
| 896  | 896  | 1408 | 608   | 96.7  | 10.34 | 通过 |
| 1024 | 1024 | 1536 | 677   | 97.2  | 10.29 | 通过 |
| 1664 | 1664 | 2176 | 1243  | 100.0 | 10.00 | 通过 |
| 2048 | 2048 | 2560 | 1450  | 101.5 | 9.85  | 通过 |
| 2560 | 2560 | 3072 | 1878  | 103.7 | 9.64  | 通过 |
| 2816 | 2816 | 3328 | 2123  | 104.9 | 9.53  | 通过 |
| 3072 | 3072 | 3584 | 2339  | 105.9 | 9.44  | 通过 |

> 1p 共 8 档（512/896/1024/1664/2048/2560/2816/3072），上表为实测数据（`KV_phys=3584 → 切到 1024，核心点OK=True`）。每档首次 predict 触发该档 GE profile 懒编译（约 5 min），属一次性成本；稳态 decode 约 0.095–0.106 s/step。

### 2p 实测性能（300I DUO，fp16，TP=2）

数据源：`bench_tp2_bucket_results/verified_results.json`（本机实测，可复现）。

| prompt_tokens | prefill_seq | kv_len | prefill (ms) | decode/step (ms) | tok/s | 核心点 |
|--------------:|------------:|-------:|-------------:|------------------:|------:|:------:|
| 512  | 512  | 1024 | 299   | 112  | 8.92 | 通过 |
| 896  | 896  | 1408 | 577   | 153  | 6.53 | 通过 |
| 1024 | 1024 | 1536 | 601   | 177  | 5.64 | 通过 |
| 1664 | 1664 | 2176 | 1146  | 246  | 4.06 | 通过 |
| 2048 | 2048 | 2560 | 1267  | 271  | 3.69 | 通过 |
| 2560 | 2560 | 3072 | 1514  | 303  | 3.30 | 通过 |
| 2816 | 2816 | 3328 | 1765  | 333  | 3.00 | 通过 |
| 3072 | 3072 | 3584 | 1837  | 338  | 2.95 | 通过 |

> 2p 配置共 8 档（含 896/2560）；上表为原 6 档实测数据，896/2560 档可用 `bash infer.sh 0,1 perf` 复现。

**结论**：

- **核心点 PASS**：每档 prefill KV 输出 length == 该档 kv_len（日志打印 `核心点 OK: prefill KV out len == kv_len`），**绝不 pad 到最大 3584**。
- prefill 随 seq 线性增长（约 0.64 ms/token）；decode 随 kv_len 增长（更长 attention_mask）。
- **显存峰值**：2p 冷加载首次 +6.9~8.4 GB/活动卡；稳态复跑复用常驻权重（增量小）。

---

## 6. A8W8 量化导出与推理（TP=2）

> quant-dir 支持 AMCT A8W8 量化权重产物（manifest.json + npy）。

### 量化导出（每 rank 独立目录 + 内置 postproc）

```bash
python3 export_qwen3_8b_onnx.py \
  --model-id Qwen3-8B \
  --output-dir <out> --device cpu --dtype fp16 \
  --tp-size 2 --tp-dynamic \
  --quant-dir amct_probe/a8w8_smoke35_fp16
```

- `--quant-dir`：AMCT pack 目录（36 层 `a8w8_smoke35_fp16` 216 模块；单层调试 `a8w8_1layer` + `--num-layers 1`）
- 产物自动 postproc（216 个 QBMM scale → uint64），**不再需要单独 postproc 步骤**
- 导出后 allow_nz 仅作用于 fp16 MatMul（down_proj/lm_head），QBMV3 是 Custom 不受影响

### 转换（与 fp16 相同）

```bash
$CONV --fmk=ONNX --modelFile=<out>/rank{r}/{prefill,decode}/qwen3_8b_llm_*.onnx \
      --outputFile=<out>/rank{r}/{prefill,decode}/... --optimize=none --saveType=MINDIR
```

### 推理（force_fp16 模板；显式指定量化 mindir）

```bash
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids 0,1 --model-id <tok> --prompt "你好，请用一句话介绍一下你自己" \
  --prefill-ranks <out>/rank0/prefill/..._rank0_graph.mindir,<out>/rank1/prefill/..._rank1_graph.mindir \
  --decode-ranks <out>/rank0/decode/..._rank0_graph.mindir,<out>/rank1/decode/..._rank1_graph.mindir \
  --bucket-cfg-dir configs \
  --json-out perf_quant.json
```

### profiling 采集（infer_qwen3_8b_mslite_tp_prof.py）

基于真实推理流程（双图 build + 真实输入 + decode 自回归）包裹 aclprof：

- timed prefill predict 前后启停 → `<prof_dir>/bucket_<seq>/prefill_r<rank>/`
- decode 循环前后启停 → `decode_r<rank>/`（build/warmup 不进数据）
- 采集后自动 `msprof --export=on` 解析 → `op_summary_*.csv`（算子耗时/占比）等

```bash
python3 infer_qwen3_8b_mslite_tp_prof.py \
  --device-ids 0,1 --model-id <tok> \
  --prefill-ranks <pf0>,<pf1> --decode-ranks <dc0>,<dc1> \
  --bucket-cfg-dir configs --buckets 896,2560 \
  --prof-dir prof_data --json-out prof.json
```

### 2p 量化实测性能（300I DUO，fp16+A8W8，TP=2）

| prompt_tokens | prefill_seq | kv_len | prefill (ms) | decode/step (ms) | tok/s | 核心点 |
|--------------:|------------:|-------:|-------------:|------------------:|------:|:------:|
| 896  | 896  | 1408 | 325   | 117  | 8.54 | 通过 |
| 1664 | 1664 | 2176 | 635   | 156  | 6.41 | 通过 |
| 2560 | 2560 | 3072 | 1038  | 208  | 4.80 | 通过 |
| 2816 | 2816 | 3328 | 1120  | 215  | 4.65 | 通过 |

### 量化推理现状（实测）

- Reshape_16 声明问题已修复后 decode warmup 通过、完整推理跑通（896 档 prefill ~390ms / decode ~135ms/step）
- prefill 输出 KV 声明仍为 FLOAT32（dc 输入 FLOAT16）→ 免拷贝 attach 退化 load_kv（一次性 H2D，功能正常）
- **输出文本仍乱码**：GE 9.1.0 动态图"输入无关"问题（量化/非量化都有，见遗留问题）

---

## 7. 目录与脚本总览

| 文件 | 用途 |
|------|------|
| `export_and_convert.sh` | 导出 ONNX + 转换 MindIR（`bash export_and_convert.sh 1p/2p/4p`） |
| `export_qwen3_8b_onnx.py` | 统一导出脚本（`--tp-size 1/2/4`，2p 加 `--tp-dynamic`） |
| `infer.sh` | 一键推理：`bash infer.sh <设备>`（精度）/ `perf`（1p/2p/4p 八档性能）/ `prof`（msprof 采集） |
| `infer_qwen3_8b_mslite_tp.py` | 2p/4p 张量并行推理脚本 |
| `infer_qwen3_8b_mslite_1p.py` | 1p 动态单图分档推理脚本（被 `infer.sh` 1p 路径调用） |
| `_npu_mem.py` | 1p 显存峰值采样器（被 `infer_qwen3_8b_mslite_1p.py` 调用） |
| `configs/` | 动态分档配置 |
| `infer_qwen3_8b_mslite_tp_prof.py` | 2p profiling 采集（真实推理包裹 aclprof + 自动解析） |
| `prof_ctrl.py` | aclprof ctypes 封装（profiling 启停控制） |
| `bench_tp2_bucket_results/` | 2p 分档性能实测结果（含 `verified_results.json`） |

---

## 8. 关键坑与注意事项（实测）

1. **必须配置 `ge.dynamicNodeType=1`**：在线 GE 解析动态 dim 的开关，缺失时 dynamicDims 不生效（1p 会报 `GetCurDynamicDims: input count of user:0 should be equal to data count of graph:3`）。
2. **`ge.inputShape` 使用 ONNX 输入名即可**（如 `input_ids` / `attention_mask` / `past_key_cache`）；C++ `UpdateGraphInputs` 位置 fallback 会自动映射 GE 内部 `Uxxx` 名（日志 `fall back to positional` WARNING 属正常）。
3. **每档首次 predict 触发该档 GE profile 懒编译**：1p 约 5 min/档，2p 约 10–15 min/档；跨档 TBE 内核不共享，`--repeats` 第 1 轮为 warmup，第 2..N 轮才是稳态。
4. **TP worker 精度模式必须通过 Context 设置**（`enforce_fp32`），不能放进 config 文件——`[acl_init_options]` 会导致 GE 在 Context 创建时急切绑定端口 16666，与其它 rank 冲突。
5. **TP warmup dummy 必须使用选中档 seq**（不能硬编 64），否则不命中已编译档。
6. **1p GE 强制 KV 输出 = max 档 3584**：单图动态下无法让 KV 只 pad 到 kv_len，必须 D2H 后显式切 `[0..kv_len]` 丢弃脏尾部（否则 decode 喂脏 KV → 输出垃圾/重复）。这是 1p 核心点能否达成的关键。
7. **TP 设备映射漂移**：同一 `--device-ids 0,1`，不同 run 可能落到物理 dev {0,1} 或 {2,3}，需留意实际落卡。
8. **4p 未验证**：300I Duo 双卡 PCIe 拓扑下 4p HCCL 有已知精度问题。
9. **短输出退化**：用填充文本 prompt 做性能扫描时，输出可能退化成重复 token（如 `f`），这是填充文本/短 max_new_tokens 的共性，不影响性能测量与核心点验证；默认短档已使用真实对话 prompt。
10. **量化 decode 报 `Reshape_16 ... cannot be divided from [4096]`**：根因是 QBMV3 symbolic 输出类型声明为激活输入形状 [M,K]（误把 K 当输出列数），gate/up 声明翻倍 → SwiGlu 链声明错位 → GE ReshapeInfer 失败（任何 precision_mode 都触发）。修复：输出类型声明为 [M, N]（取 w 列数）；SwiGlu symbolic 用 `len(sizes())` 替代已失效的 `dim()`。
11. **onnx 1.17 `save_model(save_as_external_data=True)` 会放大 fp16 external data（3~4 倍）**，converter 报 `memcpy_s dst/src size` 错。导出后处理（allow_nz）只写 onnx 头部（`load_external_data=False` + `onnx.save`），禁止再对已导出的 onnx 用 save_model external 模式重写。
12. **per-rank GE cache**：cfg 配 `ge.graph_compiler_cache_dir=./ge_cache`，脚本按 rank 生成 `./ge_cache/rank{r}`（目录必须预先存在，否则 E13026）；cache key 不含 session 参数（precision_mode 等），改参数后需清 ge_cache/kernel_meta。

---

## 9. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-8B（ModelScope）](https://www.modelscope.cn/models/Qwen/Qwen3-8B)
- [Qwen3-8B（HuggingFace）](https://huggingface.co/Qwen/Qwen3-8B)

---

## 10. 许可证

本教程遵循 Qwen3-8B 模型及相关依赖的许可证要求。
