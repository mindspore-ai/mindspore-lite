# Qwen3-8B ONNX 导出与 MindSpore Lite 推理部署教程

本教程详细介绍如何将 Qwen3-8B 模型导出为 ONNX 格式，转换为 MindSpore Lite MindIR 格式，并在 Ascend 300I Duo NPU 上完成端到端推理部署。

Qwen3-8B 是一个纯文本大语言模型，采用标准 Transformer 全注意力架构。当前采用**公共前缀（common prefix）架构**，每个 rank 导出 3 个 ONNX 图：

1. **Prefix**（`qwen3_8b_prefix_rank*.onnx`）：对公共前缀（如系统 prompt）**只跑一次**并产出 KV cache，不含 lm_head（前缀只缓存、不采样）
2. **Suffix**（`qwen3_8b_suffix_rank*.onnx`）：接收 prefix KV + 用户后缀 token，输出 logits 与完整 KV cache
3. **Decode**（`qwen3_8b_llm_decode_rank*.onnx`）：基于 past KV cache 做自回归增量生成

Ascend 300I Duo 单卡包含 2 个 NPU 芯（chip），因此：

- **1p = 单卡单芯**：`infer_qwen3_8b_mslite_tp.py --device-ids 0`，单进程三图（prefix/suffix/decode）
- **2p = 单卡双芯**（卡内双芯 HCCS 互联，本机已验证）：`infer_qwen3_8b_mslite_tp.py --device-ids 0,1`，HCCL 多进程，每 rank KV 4 头

> 4p：导出与推理脚本当前仅支持 1p/2p；4p 为未验证路径，暂不提供。

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
- **KV cache shape**：`[36, 1, 8/TP, kv_len, 128]`（1p=8 头，2p=4 头）

### 公共前缀分桶设计

每图各 2 档（`ge.dynamicDims`），组合后覆盖两类输入规模；每档最多可生成 512 个新 token（`MAX_OUTPUT_TOKENS=512`）。

**Prefix（2 档）** — `configs/qwen3_8b_llm_prefill_prefix.config`

| prefix 档 | 说明 |
|---:|---|
| 768 | 默认公共前缀档（`COMMON_PREFIX_BUCKET=768`） |
| 256 | 短前缀档 |

**Suffix（2 档）** — `configs/qwen3_8b_llm_prefill_suffix.config`（dynamicDims 五元组：suffix_len、kv_len、total、prefix、prefix）

| suffix 档 | prefix 档 | total（seq） | 说明 |
|---:|---:|---:|---|
| 896 | 768 | 1664 | 默认后缀档（`COMMON_SUFFIX_BUCKET=896`） |
| 2560 | 256 | 2816 | 长后缀档 |

**Decode（2 档）** — `configs/qwen3_8b_llm_decode.config`，`kv_len = total + 512`

| kv_len | 说明 |
|---:|---|
| 2176 | 768 + 896 + 512（默认组合） |
| 3328 | 256 + 2560 + 512（长后缀组合） |

**运行流程**：

1. prefix 图按公共前缀**只执行一次**，KV 落在设备侧常驻 Tensor（跨多次请求复用）；
2. suffix 图接收 prefix KV（设备侧零拷贝 attach）+ 用户后缀，pad 到选中 suffix 档，输出末位 logits + prefix+suffix 全量 KV；
3. 推理脚本把"右 pad 的 prefix + 左 pad 的 suffix"压缩为连续 KV，并预留 512 空槽供 decode 写入；
4. decode 图用该 kv_len 档自回归，输出超过 kv_len 时截断并提示。

> `configs/qwen3_8b_llm_prefill.config`（8 档单图 prefill）为旧版流程保留配置，当前脚本默认不再使用。

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
export CONV=$MSLITE_HOME_PATH/tools/converter/converter/converter_lite
```

> `infer.sh` 会自动设置 `ASCEND_CUSTOM_OPP_PATH=$ASCEND_OPP_PATH/vendors/mslite_custom_ops` 与 `HCCL_NPU_SOCKET_PORT_RANGE=21500-21600`，无需手工导出。

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
```

完整参数（均可省略，走默认值）：

```bash
bash export_and_convert.sh <1p|2p> [model_id] [dtype] [prefix_len] [suffix_len] [output_dir]
# 例：bash export_and_convert.sh 2p ./Qwen3-8B fp16 768 896
```

| 位置参数 | 默认值 | 说明 |
|---|---|---|
| `<1p\|2p>` | 必填 | 并行规模 |
| `[model_id]` | `./Qwen3-8B` | 权重目录 |
| `[dtype]` | `fp16` | 导出精度 |
| `[prefix_len]` | `768` | 公共前缀 dummy 长度（300IDUO 对齐默认 768） |
| `[suffix_len]` | `896` | 后缀 dummy 长度（真实长度仍为动态轴） |
| `[output_dir]` | 1p=`./qwen3_8b_onnx`，2p=`./qwen3_8b_tp2_onnx` | 输出目录 |

`export_and_convert.sh` 自动执行：

1. `export_qwen3_8b_onnx.py --common-prefix --prefix-len N --suffix-len M --tp-size N` 导出各 rank 的 prefix/suffix/decode ONNX（2p 追加 `--tp-dynamic`：动态 seq / 动态 KV 轴，一个 MindIR 服务全部档位）；
2. `converter_lite --optimize=none --saveType=MINDIR` 转换为在线 GE MindIR（`optimize=none` 保留动态图，供运行时 `provider=ge` 使用）。

### 直接导出命令

```bash
# 1p：公共前缀三图
python3 export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_onnx \
  --device cpu --dtype fp16 --tp-size 1 \
  --common-prefix --prefix-len 768 --suffix-len 896

# 2p：TP=2（--tp-dynamic 使一个 ONNX 服务所有档位）
python3 export_qwen3_8b_onnx.py \
  --model-id ./Qwen3-8B \
  --output-dir ./qwen3_8b_tp2_onnx \
  --device cpu --dtype fp16 --tp-size 2 \
  --common-prefix --prefix-len 768 --suffix-len 896 \
  --tp-dynamic
```

### 导出参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | HuggingFace 模型路径或本地目录 | `./Qwen3-8B` |
| `--output-dir` | 输出目录 | `./qwen3_8b_onnx` |
| `--device` | 导出设备（cpu/cuda） | `cpu` |
| `--dtype` | 导出精度（fp16/bf16/fp32） | `fp16` |
| `--tp-size` | 张量并行规模（1/2） | `1` |
| `--common-prefix` | 导出公共前缀三图（prefix/suffix/decode） | `False` |
| `--prefix-len` | 公共前缀 dummy 长度（310IDUO 对齐默认 768） | `768` |
| `--suffix-len` | 后缀 dummy 长度（真实长度仍为动态） | `16` |
| `--tp-dynamic` | TP>=2 时导出动态 seq/KV 轴（配合 `ge.dynamicDims`） | `False` |
| `--quant-dir` | AMCT A8W8 量化权重目录（manifest.json + npy） | `None` |
| `--dummy-seq-len` | 导出时 dummy 序列长度 | `8` |
| `--num-layers` | 调试用：只导出前 N 层（0=全部 36 层） | `0` |
| `--kv-cache-len` | 覆盖 decode 默认 KV 长度（默认 256） | `0` |
| `--use-dynamo` | 使用 torch dynamo 导出路径 | `False` |

### 产出（ONNX）

```text
qwen3_8b_onnx/             # 1p
├── prefix/qwen3_8b_prefix_rank0.onnx
├── suffix/qwen3_8b_suffix_rank0.onnx
└── decode/qwen3_8b_llm_decode_rank0.onnx

qwen3_8b_tp2_onnx/         # 2p（每 rank 一套三图）
├── rank0/{prefix,suffix,decode}/qwen3_8b_*_rank0.onnx
└── rank1/{prefix,suffix,decode}/qwen3_8b_*_rank1.onnx
```

> rank 间图不同（真 TP 切分），每 rank 权重约 8.8GB（半个 transformer + 复制的 embed/lm_head）。

### ONNX 模型输入输出 Shape

**Prefix** — `prefix/qwen3_8b_prefix_rank*.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `input_ids` | `(1, prefix_len)` | int64 | 公共前缀 token IDs |
| 输入 | `attention_mask` | `(1, prefix_len)` | int64 | 注意力掩码 |
| 输入 | `position_ids` | `(1, prefix_len)` | int64 | 位置 ID |
| 输出 | `present_key_cache` | `(36, 1, 8/TP, prefix_len, 128)` | float16 | 前缀 KV cache（无 logits/lm_head） |
| 输出 | `present_value_cache` | `(36, 1, 8/TP, prefix_len, 128)` | float16 | 前缀 KV cache |

**Suffix** — `suffix/qwen3_8b_suffix_rank*.onnx`

| 方向 | 名称 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| 输入 | `input_ids` | `(1, suffix_len)` | int64 | 用户后缀 token IDs |
| 输入 | `attention_mask` | `(1, prefix_len+suffix_len)` | int64 | 完整 prefix+suffix 掩码（保留 prefix padding） |
| 输入 | `position_ids` | `(1, suffix_len)` | int64 | 后缀位置 ID（从 prefix_len 起） |
| 输入 | `past_key_cache` | `(36, 1, 8/TP, prefix_len, 128)` | float16 | prefix 图输出的 KV（零拷贝 attach） |
| 输入 | `past_value_cache` | `(36, 1, 8/TP, prefix_len, 128)` | float16 | prefix 图输出的 KV |
| 输出 | `logits` | `(1, suffix_len, 151936)` | float32 | 取末位有效 token 的 logits |
| 输出 | `present_key_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | prefix+suffix 全量 KV |
| 输出 | `present_value_cache` | `(36, 1, 8/TP, kv_len, 128)` | float16 | prefix+suffix 全量 KV |

**Decode** — `decode/qwen3_8b_llm_decode_rank*.onnx`

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

> `kv_len = prefix 档 + suffix 档 + 512`；`8/TP` 为每 rank 的 KV 头数（1p=8、2p=4）。

---

## 3. ONNX 转 MindIR

### 转换命令

`export_and_convert.sh` 已包含转换步骤，通常无需手动执行。以下为底层命令示例（1p）：

```bash
Convert=$MSLITE_HOME_PATH/tools/converter/converter/converter_lite

# Prefix 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_8b_onnx/prefix/qwen3_8b_prefix_rank0.onnx \
  --outputFile=qwen3_8b_onnx/prefix/qwen3_8b_prefix_rank0 \
  --optimize=none --saveType=MINDIR

# Suffix 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_8b_onnx/suffix/qwen3_8b_suffix_rank0.onnx \
  --outputFile=qwen3_8b_onnx/suffix/qwen3_8b_suffix_rank0 \
  --optimize=none --saveType=MINDIR

# Decode 转换
$Convert --fmk=ONNX \
  --modelFile=qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0.onnx \
  --outputFile=qwen3_8b_onnx/decode/qwen3_8b_llm_decode_rank0 \
  --optimize=none --saveType=MINDIR
```

2p 只需对 `qwen3_8b_tp2_onnx/rank{0,1}/` 下的每个 rank 重复上述三组命令。

### 参数说明

| 参数 | 说明 |
|------|------|
| `--fmk` | 输入模型格式（ONNX） |
| `--modelFile` | 输入 ONNX 模型路径 |
| `--outputFile` | 输出 MindIR 路径（不带扩展名） |
| `--optimize` | 必须指定 `none`（在线 GE 流程，保留动态图） |
| `--saveType` | 输出格式（MINDIR） |

### 运行时动态分档配置

公共前缀三图各配一份 `ge.dynamicDims` 配置：

- **1p**：`configs/`（KV 8 头/卡）
- **2p**：`configs/tp2/`（KV 4 头/rank；推理时按设备数自动选择，`--common-config-dir` 可覆盖）

文件清单（两级目录一致）：

| 文件 | 用途 |
|---|---|
| `qwen3_8b_llm_prefill_prefix.config` | prefix 图 2 档（768 / 256） |
| `qwen3_8b_llm_prefill_suffix.config` | suffix 图 2 档（896+768 / 2560+256） |
| `qwen3_8b_llm_decode.config` | decode 图 2 档（kv_len 2176 / 3328） |
| `qwen3_8b_llm_prefill.config` | 旧版单图 prefill 8 档（保留，当前流程不用） |

**prefix 配置**（`configs/qwen3_8b_llm_prefill_prefix.config`）：

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
ge.inputShape=U285:1,-1;U286:1,-1;U287:1,-1
ge.dynamicDims=768,768,768;256,256,256
ge.dynamicNodeType=1
```

**suffix 配置**（`configs/qwen3_8b_llm_prefill_suffix.config`，1p 为 8 头、tp2 为 4 头）：

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
ge.inputShape=U292:1,-1;U293:1,-1;U294:1,-1;U295:36,1,8,-1,128;U296:36,1,8,-1,128
ge.dynamicDims=896,1664,896,768,768;2560,2816,2560,256,256
ge.dynamicNodeType=1
```

**decode 配置**（`configs/qwen3_8b_llm_decode.config`）：

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
ge.dynamicDims=2176,2176,2176;3328,3328,3328
ge.dynamicNodeType=1
```

> 2p 的 HCCL 配置由推理脚本在运行时把上述 `[ge_graph_options]` 合并进按 rank 生成的临时 HCCL config（含按 rank 隔离的 `ge.graph_compiler_cache_dir`），无需手工编辑。

### 产出（MindIR）

模型文件超过 2GB 时会拆分为 `*_graph.mindir` 与 `*_variables/` 目录：

```text
qwen3_8b_onnx/                                        # 1p
├── prefix/qwen3_8b_prefix_rank0_graph.mindir          (+ _variables/ 权重)
├── suffix/qwen3_8b_suffix_rank0_graph.mindir          (+ _variables/ 权重)
└── decode/qwen3_8b_llm_decode_rank0_graph.mindir      (+ _variables/ 权重)

qwen3_8b_tp2_onnx/                                    # 2p
├── rank0/{prefix,suffix,decode}/..._rank0_graph.mindir
└── rank1/{prefix,suffix,decode}/..._rank1_graph.mindir
```

---

## 4. MindSpore Lite 推理

### 一键推理

`infer.sh` 需先 source CANN 环境（要求 `ASCEND_OPP_PATH` 已设置），第一个参数为 `--device-ids`（设备数决定 1p/2p 与模型目录），其余参数原样透传给 `infer_qwen3_8b_mslite_tp.py`：

```bash
# 1p（单卡，模型目录 qwen3_8b_onnx）
bash infer.sh --device-ids 0

# 2p（TP=2，模型目录 qwen3_8b_tp2_onnx）
bash infer.sh --device-ids 0,1

# 自定义公共前缀 / 后缀 / 输出长度
bash infer.sh --device-ids 0,1 \
  --common-prefix-text "你是一个专业的客服助手，" \
  --suffix-prompt "请介绍一下退款流程" \
  --max-new-tokens 128 \
  --json-out perf.json
```

默认值：`--model-id ./Qwen3-8B`、`--common-config-dir configs`（1p）或 `configs/tpN`（TP）、`--common-prefix-text "你好，"`、`--suffix-prompt "请用一句话介绍一下你自己"`、`--max-new-tokens 64`。

### 直接调用推理脚本

```bash
# 1p：需显式指定模型目录（infer.sh 会自动补）
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids 0 \
  --common-model-dir ./qwen3_8b_onnx \
  --model-id ./Qwen3-8B \
  --common-prefix-text "你好，" \
  --suffix-prompt "请用一句话介绍一下你自己" \
  --max-new-tokens 64

# 2p/4p：按设备数派发，配置目录默认 configs/tp2
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids 0,1 \
  --common-model-dir ./qwen3_8b_tp2_onnx \
  --model-id ./Qwen3-8B \
  --max-new-tokens 64
```

### 参数说明

**`infer_qwen3_8b_mslite_tp.py`（1p/2p 公共前缀推理，统一入口）**

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--device-ids` / `--device-id` | 逗号分隔设备 id（数量决定并行度：1/2） | 必填 |
| `--model-id` | 权重/tokenizer 路径 | `./Qwen3-8B` |
| `--common-model-dir` | 公共前缀三图模型根目录（`infer.sh` 自动注入） | 必填 |
| `--common-config-dir` | 配置目录 | `None`（1p=`configs`，TP=`configs/tpN`） |
| `--common-prefix-text` | 公共前缀文本（只 prefill 一次） | `你好，` |
| `--suffix-prompt` | 用户后缀 prompt | `请用一句话介绍一下你自己` |
| `--max-new-tokens` | 最大生成 token 数 | `64` |
| `--json-out` | 输出 perf dict 到 JSON 文件 | `None` |

> 公共前缀超过 768 token 会报错（`common prefix has N tokens; maximum is 768`）；后缀过长时选用 2560 档（前缀须 ≤256 token）。

### 推理示例输出

> 以下为示意输出，实际数值以运行为准。

```text
=== TP_SIZE=1  devices=0 ===

============================================================
Input Prompt: 你好，请用一句话介绍一下你自己
============================================================
Generated Response: 你好！我是通义千问……

--- Performance (1p common prefix) ---
{'prefix_actual_len': 4, 'prefix_bucket': 768, 'suffix_actual_len': 12,
 'suffix_bucket': 896, 'prefix_ms_once': 210.5, 'suffix_ms': 118.3,
 'prefill_total_ms': 328.8, 'decode_first_ms': 96.2, 'decode_min_ms': 94.1,
 'decode_avg_ms': 95.0, 'decode_total_ms': 6080.0, 'decode_steps': 64,
 'output_len': 64, 'truncated': False, ...}
Prefill: total=328.8 ms (prefix=210.5 ms, suffix=118.3 ms)
Decoder: total=6080.0 ms, steps=64, first=96.2 ms, avg=95.0 ms, min=94.1 ms
```

> prefix 仅首个请求执行一次（`prefix_ms_once`），同一公共前缀下的后续请求只付 suffix + decode 的成本。

---

## 5. 性能数据

### 测试环境

| 项目 | 说明 |
|------|------|
| 硬件 | Atlas 300I Duo（Ascend NPU） |
| 模型 | Qwen3-8B（fp16） |
| 精度 | fp16（GE dynamicDims 分档） |
| 数据来源 | 本机实测，可复现 |

> 以下两表为**旧版单图 prefill（8 档）流程**的历史实测数据；公共前缀三图流程下 prefix/suffix 的拆分耗时见推理日志（`prefix_ms_once` / `suffix_ms`），decode 单步耗时可对照同 kv_len 档位参考。

### 1p 实测性能（300I DUO，fp16，单卡动态单图，旧版流程）

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

> 旧版 1p 共 8 档（512/896/1024/1664/2048/2560/2816/3072）。每档首次 predict 触发该档 GE profile 懒编译（约 5 min），属一次性成本；稳态 decode 约 0.095–0.106 s/step。

### 2p 实测性能（300I DUO，fp16，TP=2，旧版流程）

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

**结论**：

- **核心点 PASS**：每档 prefill KV 输出 length == 该档 kv_len（公共前缀流程下由推理脚本压缩为连续 KV 后供 decode 零拷贝使用）。
- prefill 随 seq 线性增长（约 0.64 ms/token）；decode 随 kv_len 增长（更长 attention_mask）。
- **公共前缀收益**：同一系统前缀的多轮请求只需一次 prefix prefill，后续请求 prefill 成本 ≈ suffix（远小于全量）。

---

## 6. A8W8 量化导出与推理（TP=2）

> quant-dir 支持 AMCT A8W8 量化权重产物（manifest.json + npy）。

### 量化导出（每 rank 独立目录 + 内置 postproc）

```bash
python3 export_qwen3_8b_onnx.py \
  --model-id Qwen3-8B \
  --output-dir <out> --device cpu --dtype fp16 \
  --tp-size 2 --tp-dynamic \
  --common-prefix --prefix-len 768 --suffix-len 896 \
  --quant-dir amct_probe/a8w8_smoke35_fp16
```

- `--quant-dir`：AMCT pack 目录（36 层 `a8w8_smoke35_fp16` 216 模块；单层调试 `a8w8_1layer` + `--num-layers 1`）
- 产物自动 postproc（216 个 QBMM scale → uint64），**不再需要单独 postproc 步骤**
- 导出后 allow_nz 仅作用于 fp16 MatMul（down_proj/lm_head），QBMV3 是 Custom 不受影响

### 转换（与 fp16 相同）

对 `<out>/rank{r}/{prefix,suffix,decode}/` 下每个 ONNX 重复：

```bash
$CONV --fmk=ONNX --modelFile=<out>/rank{r}/{prefix,suffix,decode}/qwen3_8b_*.onnx \
      --outputFile=<out>/rank{r}/{prefix,suffix,decode}/... --optimize=none --saveType=MINDIR
```

### 推理

量化 MindIR 走同一公共前缀流程：把 `--common-model-dir` 指向量化产物目录（`rank{r}/{prefix,suffix,decode}` 布局一致），配置目录沿用 `configs/tp2`：

```bash
python3 infer_qwen3_8b_mslite_tp.py \
  --device-ids 0,1 --model-id <tok> \
  --common-model-dir <out> \
  --common-prefix-text "你好，" \
  --suffix-prompt "请用一句话介绍一下你自己" \
  --max-new-tokens 64 --json-out perf_quant.json
```

### 2p 量化实测性能（300I DUO，fp16+A8W8，TP=2，旧版流程）

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

## 7. profiling 采集（infer_qwen3_8b_mslite_tp_prof.py）

基于真实推理流程（多图 build + 真实输入 + decode 自回归）包裹 aclprof：

- timed prefill predict 前后启停 → `<prof_dir>/bucket_<seq>/prefill_r<rank>/`
- decode 循环前后启停 → `decode_r<rank>/`（build/warmup 不进数据）
- 采集后自动 `msprof --export=on` 解析 → `op_summary_*.csv`（算子耗时/占比）等

> 注意：`infer_qwen3_8b_mslite_tp_prof.py` 仍基于旧版 prefill/decode 双图流程，公共前缀三图流程的 profiling 请以其为模板改造。

---

## 8. 目录与脚本总览

| 文件 | 用途 |
|------|------|
| `export_and_convert.sh` | 导出 ONNX + 转换 MindIR（`bash export_and_convert.sh 1p/2p [model_id] [dtype] [prefix_len] [suffix_len] [out_dir]`） |
| `export_qwen3_8b_onnx.py` | 统一导出脚本（`--tp-size 1/2`、`--common-prefix` 三图、2p 加 `--tp-dynamic`） |
| `infer.sh` | 一键推理启动器：`bash infer.sh --device-ids 0[,1] [options]`（透传参数，自动选模型目录） |
| `infer_qwen3_8b_mslite_tp.py` | 1p/2p 公共前缀推理脚本（按设备数自动派发；`CommonPrefixRunner`） |
| `infer_qwen3_8b_mslite_tp_prof.py` | profiling 采集（真实推理包裹 aclprof + 自动解析，旧版双图流程） |
| `prof_ctrl.py` | aclprof ctypes 封装（profiling 启停控制） |
| `configs/` | 1p 动态分档配置（prefix/suffix/decode + 旧版 prefill） |
| `configs/tp2/` | 2p 动态分档配置（KV 4 头/rank） |

---

## 9. 关键坑与注意事项（实测）

1. **必须配置 `ge.dynamicNodeType=1`**：在线 GE 解析动态 dim 的开关，缺失时 dynamicDims 不生效（会报 `GetCurDynamicDims: input count of user:0 should be equal to data count of graph:3`）。
2. **`ge.inputShape` 使用 ONNX 输入名即可**（如 `input_ids` / `attention_mask` / `past_key_cache`）；C++ `UpdateGraphInputs` 位置 fallback 会自动映射 GE 内部 `Uxxx` 名（日志 `fall back to positional` WARNING 属正常）。
3. **每档首次 predict 触发该档 GE profile 懒编译**：1p 约 5 min/档，2p 约 10–15 min/档；跨档 TBE 内核不共享，首轮为 warmup，后续轮次才是稳态。
4. **TP worker 精度模式必须通过 Context 设置**（`enforce_fp32`），不能放进 config 文件——`[acl_init_options]` 会导致 GE 在 Context 创建时急切绑定端口 16666，与其它 rank 冲突。
5. **TP warmup dummy 必须使用选中档 shape**（不能硬编 64），否则不命中已编译档。
6. **KV 零拷贝 buffer 必须覆盖最大档**：GE 动态图只暴露当前逻辑输出 shape，零拷贝输出 Tensor 要按最大 bucket（如 prefix 768 / decode 3328）分配；转换后图描述可能报 FLOAT32 而 GE force_fp16 物理产出 FP16，**不要用 `get_outputs()[].dtype`** 决定 buffer dtype。
7. **prefix/suffix 压缩拼接**：prefix 右 pad、suffix 左 pad，decode 需要连续 KV——推理脚本把两者压缩到 `[0..valid)` 并预留 512 空槽，否则 decode 喂脏 KV 会输出垃圾/重复。
8. **TP 设备映射漂移**：同一 `--device-ids 0,1`，不同 run 可能落到物理 dev {0,1} 或 {2,3}，需留意实际落卡。
9. **短输出退化**：用填充文本 prompt 做性能扫描时，输出可能退化成重复 token（如 `f`），这是填充文本/短 max_new_tokens 的共性，不影响性能测量与核心点验证。
10. **量化 decode 报 `Reshape_16 ... cannot be divided from [4096]`**：根因是 QBMV3 symbolic 输出类型声明为激活输入形状 [M,K]（误把 K 当输出列数），gate/up 声明翻倍 → SwiGlu 链声明错位 → GE ReshapeInfer 失败（任何 precision_mode 都触发）。修复：输出类型声明为 [M, N]（取 w 列数）；SwiGlu symbolic 用 `len(sizes())` 替代已失效的 `dim()`。
11. **onnx 1.17 `save_model(save_as_external_data=True)` 会放大 fp16 external data（3~4 倍）**，converter 报 `memcpy_s dst/src size` 错。导出后处理（allow_nz）只写 onnx 头部（`load_external_data=False` + `onnx.save`），禁止再对已导出的 onnx 用 save_model external 模式重写。
12. **per-rank GE cache**：推理脚本把 cfg 的 `[ge_graph_options]` 合并进按 rank 生成的 HCCL 临时 config，`ge.graph_compiler_cache_dir` 按 rank 隔离（`ge_cache/rank{r}`，目录必须预先存在，否则 E13026）；cache key 不含 session 参数（precision_mode 等），改参数后需清 ge_cache/kernel_meta。

---

## 10. 参考资源

- [MindSpore Lite 文档](https://www.mindspore.cn/lite)
- [Qwen3-8B（ModelScope）](https://www.modelscope.cn/models/Qwen/Qwen3-8B)
- [Qwen3-8B（HuggingFace）](https://huggingface.co/Qwen/Qwen3-8B)

---

## 11. 许可证

本教程遵循 Qwen3-8B 模型及相关依赖的许可证要求。
