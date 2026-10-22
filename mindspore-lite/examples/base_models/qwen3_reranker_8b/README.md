# Qwen3-Reranker-8B ONNX 导出与 MindSpore Lite 推理部署教程

本目录提供 Qwen3-Reranker-8B 的 ONNX 导出、MindIR 转换与 MindSpore Lite 推理脚本，
已在 **Atlas 300I Duo（CANN 9.1.0 + MindSpore Lite 2.11.0）** 上用
T2Reranking（中文重排基准）完成端到端精度对齐验证。

Qwen3-Reranker-8B 是重排序（Cross-Encoder）模型：输入 Query 与 Document，输出
相关性分数。本实现与官方一致采用二分类打分：

- 取最后一个 token 位置的 logits
- 取 `yes` / `no` 两个 token 的 logits 做 softmax
- `P(yes)` 作为相关性分数（`yes`=9693，`no`=2152）

推荐流程为 **CP2（Ulysses 序列并行）+ 公共前缀 + 免拷贝 KV**：

1. **Prefix** 图：共享前缀（系统提示等）**只算一次**，产出 36 层 K/V 缓存
   `[36, 1, bucket, 8, 128]`；prefix 与 rank 无关（无集合通信）；
2. **Suffix** 图：接收 prefix KV + 请求后缀，输出全局最后 token 的 logits；
   prefix 的 K/V 通过**零拷贝**（device 张量直传）交给 suffix，不进 host、不落 npy。

目录下另有 **A8W8 量化开关**（第 6 节，默认关闭）；本示例只提供 CP2 双 rank 路径。

---

## 模型架构

### 架构参数

| 参数 | 值 |
|---|---|
| `architectures` | Qwen3ForCausalLM |
| `hidden_size` | 4096 |
| `num_hidden_layers` | 36 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8（GQA） |
| `head_dim` | 128 |
| `intermediate_size` | 12288 |
| `vocab_size` | 151669（GE 按 32 对齐到 151680） |
| 原始权重精度 | bfloat16 |

### CP2 切分策略（Ulysses）

Context Parallel（Ulysses）切 **序列维度**，权重每 rank 完整复制：

```text
每 rank:   输入切片 [1, S/2]（自己的序列段）
attention: 本段 q/k/v -> AllGather(seq) 全序列注意力（本 rank 负责 16 q / 4 kv 头）
           -> AllGather(heads) 32 头 -> 切回本 rank 段 -> o_proj/MLP（全权重，本地）
lm_head:   全权重；host 取拥有全局最后 token 的 rank（左 padding 下恒为 rank1）
```

通信只用 `Custom(AllGather)`（HcomAllGather），每层 2 个（seq 方向 + head 方向）。

### 公共前缀分桶设计

**Prefix（固定 1 桶，静态 shape）** — 转换时用 `--inputShape` 固定：

| 项 | 值 |
|---|---|
| 真实共享前缀 | ≤ 512 token（T2Reranking 示例前缀 56 token） |
| 桶 | 512（128 对齐；PFA tiling 要求） |
| 产出 | K/V `[36, 1, 512, 8, 128]` fp16，两 rank 相同 |

**Suffix（动态 4 档）** — `ge.dynamicDims`（五元组：suffix_len、mask_len=prefix+桶、
suffix_len、prefix、prefix）：

| 档 | suffix 桶 | mask 长 | 说明 |
|---:|---:|---:|---|
| 512 | 512 | 1024 | 默认档（真实 368 token 进此档） |
| 1024 | 1024 | 1536 | |
| 1536 | 1536 | 2048 | |
| 2048 | 2048 | 2560 | 上限 |

---

## 1. 环境准备

### 依赖版本（建议）

| 软件包 | 版本 / 路径 |
|---|---|
| Python | 3.11 |
| torch | ≥ 2.7（**仅导出**；`dynamo=False` 强制 legacy exporter） |
| transformers | 5.x（导出与推理都要用 tokenizer） |
| onnx | 1.19.x（仅导出） |
| **mindspore-lite（运行时包）** | 2.11.0 aarch64。<br>需含 `runtime/lib/` 与 `tools/converter/converter/converter_lite` |
| **mindspore_lite（Python 模块）** | 与上面运行时包**同源**的 wheel：`mindspore_lite-2.11.0-cp311-cp311-linux_aarch64.whl`。**仅推理需要**，导出不需要 |
| CANN | 9.1.0 |

> ⚠ **运行时包与 config 必须匹配**：`configs/*.config` 的 `ge.inputShape` 用的是
> ONNX 真实名（`input_ids` 等）。务必使用上面这个包；换用其它版本包若报
> `Failed to update graph inputs` 或 `Failed to run multi-dimension process`，
> 说明包与本配置的命名/顺序不匹配，请先核对 `MSLITE_HOME`。

### 环境变量

`export_and_convert.sh` 内部会自动 source CANN 与 `mslite_custom_ops` 供应商 OPP；
推理脚本需自行准备环境（命令见 §4「运行推理」）；路径均可用环境变量覆盖：

```bash
export CANN=/path/to/cann-9.1.0
export MSLITE_HOME=/path/to/mindspore-lite-2.11.0-linux-aarch64
export PYTHONPATH=/path/to/pynew:$PYTHONPATH     # 可选：mindspore_lite 模块目录（未装进解释器时需要）
export PY=/path/to/python3.11
export DEVICES=2,3                              # 双 rank 设备（默认 2,3）
```

### 权重获取

```bash
# 方式一：ModelScope（国内快）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-Reranker-8B', local_dir='./Qwen3-Reranker-8B')"

# 方式二：HuggingFace
huggingface-cli download Qwen/Qwen3-Reranker-8B --local-dir ./Qwen3-Reranker-8B
```

---

## 2. 模型导出 ONNX（CP2 + 公共前缀）

### 一键导出 + 转换（推荐）

```bash
cd ./mindspore-lite/examples/base_models/qwen3_reranker_8b

bash export_and_convert.sh            # 导出 + 转换（默认非量化 fp16）
bash export_and_convert.sh ./Qwen3-Reranker-8B fp16 512 384 ./output
```

| 位置参数 | 默认值 | 说明 |
|---|---|---|
| `[model_id]` | `$B/Qwen3-Reranker-8B` | 权重目录 |
| `[dtype]` | `fp16` | 导出精度 |
| `[prefix_len]` | `512` | 公共前缀桶（128 对齐） |
| `[suffix_len]` | `384` | 后缀 dummy 长度（真实长度动态） |
| `[out_dir]` | `$SELF/output` | 产物目录 |

`export_and_convert.sh` 自动执行：

1. `export_qwen3_reranker_8b_onnx_cp.py --mode prefix/suffix` 导出双 rank 的 prefix + suffix ONNX（同一脚本也支持 `--mode whole` 全图 CP，历史入口）；
2. `converter_lite --optimize=none --saveType=MINDIR` 转换：**prefix 静态**（`--inputShape` 固定）、**suffix 动态**（不带 inputShape）；
3. 放置配置：把随示例交付的 `configs/{prefix,suffix}.config` 复制到产物目录（prefix 静态、suffix 4 档动态）。

> 首次跑约 25–40 分钟（大头是推理阶段的 GE 双 rank 特化编译）；之后命中
> GE 编译缓存约 **4 分钟**。

### 直接导出命令

```bash
# prefix + suffix（双 rank；--mode 可选 whole/prefix/suffix/both）
python3 export_qwen3_reranker_8b_onnx_cp.py \
  --model-id ./Qwen3-Reranker-8B --output-dir ./output/export \
  --dtype fp16 --cp-size 2 --prefix-len 512 --suffix-len 384 --mode both
```

### 导出参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--model-id` | 权重目录 | 必填 |
| `--quant-dir` | A8W8 权重包目录（空 = 非量化 fp16） | `None` |
| `--quant-format` | 包格式（auto/amct/safetensors） | `auto` |
| `--output-dir` | 导出目录 | 必填 |
| `--dtype` | 导出精度（fp16/fp32） | `fp16` |
| `--cp-size` | Context Parallel 规模 | `2` |
| `--prefix-len` | 前缀桶 | `128` |
| `--suffix-len` | 后缀 dummy 长 | `384` |
| `--mode` | prefix/suffix/both | `both` |

### 产出（ONNX）

```text
output/export/rank{r}/{prefix,suffix}/qwen3_reranker_8b_{prefix,suffix}_rank{r}.onnx
# 外部权重按 tensor 名分文件（fp16，prefix ~9.8GB / suffix ~11GB，双 rank）
```

### ONNX 模型输入输出 Shape

**Prefix** — 输入 `input_ids/attention_mask/position_ids` 各 `(1, 512)` int64；
输出 `k_out/v_out` 各 `(36, 1, 512, 8, 128)` fp16（36 层 K/V，无 logits/lm_head）。

**Suffix** — 输入 `input_ids/position_ids` `(1, S)`、`attention_mask` `(1, 512+S)`、
`past_key_cache/past_value_cache` `(36, 1, 512, 8, 128)`；输出 `logits` `(1, 1, 151669)`
（全局最后 token；GE 写 151680，推理侧按 32 对齐截回）。

> 每 rank 的 suffix 图在**图内**完成 CP 切分（`CP_RANK` 常量行切片），host 喂
> 相同的 padding 后缀即可。

---

## 3. ONNX 转 MindIR

### 转换命令

```bash
# prefix：静态（固定 512，部署时绝不 resize）
$CONV --fmk=ONNX --modelFile=…/prefix/…_rank0.onnx --outputFile=…_rank0 \
      --optimize=none --saveType=MINDIR \
      --inputShape="input_ids:1,512;attention_mask:1,512;position_ids:1,512"

# suffix：动态（不带 --inputShape，运行时按桶 resize）
$CONV --fmk=ONNX --modelFile=…/suffix/…_rank0.onnx --outputFile=…_rank0 \
      --optimize=none --saveType=MINDIR
```

### 运行时动态分档配置（随示例预生成）

suffix 的 GE 分档配置**已预生成并随示例交付**：`configs/qwen3_reranker_8b_suffix.config`

```ini
[ge_graph_options]
ge.inputShape=input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1;past_key_cache:36,1,-1,8,128;past_value_cache:36,1,-1,8,128
ge.dynamicDims=512,1024,512,512,512;1024,1536,1024,512,512;1536,2048,1536,512,512;2048,2560,2048,512,512
ge.dynamicNodeType=1
```

要点：

- `ge.inputShape` 用 **ONNX 图 input 名**，且**顺序必须与图的输入顺序一致**（勿按字母序重排，
  否则 GE 报 `Failed to run multi-dimension process`）；
- `ge.dynamicDims` 每档五元组，列次序与 `inputShape` 一致：
  `input_ids`(suffix 桶)、`attention_mask`(prefix 桶 + suffix 桶)、`position_ids`(suffix 桶)、
  `past_key_cache`/`past_value_cache`(prefix 桶)；
- prefix 配置（`configs/qwen3_reranker_8b_prefix.config`）静态构建，**不含** `ge.inputShape`；
- ⚠ 若改模型结构或导出契约（层数 / KV 头数 / `head_dim` / 桶大小），需同步修改这两个配置。

### 产出（MindIR）

```text
output/mindir/rank{r}/{prefix,suffix}/qwen3_reranker_8b_{prefix,suffix}_rank{r}_graph.mindir
output/mindir/rank{r}/{prefix,suffix}/…_variables/data_0   # 编译后的权重（fp16）
```

### GE 缓存说明（模型缓存 / 编译缓存）

GE 在**首次 `predict`** 时做图特化编译（双 rank 并行）。缓存完全由 **mslite / GE 侧**管理，
脚本不再注入任何缓存配置；`.config` 里 `[ascend_context]` 段的键会被**原样透传**进每个 rank
生成的 `config_<tag>.ini`，因此缓存也在这段里配置：

1. **`model_cache_mode=mem_opt`**（目前写在 `configs/qwen3_reranker_8b_suffix.config`）：
   mslite 据此**自动**把编译缓存目录设为**运行目录**下的 `model_build_cache_<rank>/`
   （不存在会自动创建），并启用「小图 + 从缓存加载已编译图」的快路径 —— 这就是命中缓存后
   起跑只要几分钟、且显存占用更低的原因。
2. **`model_cache_dir=<dir>`**（可选，写在同一段）：想固定缓存位置时加上这一行 ——
   否则缓存会跟着运行目录走，并在运行目录留下 `model_build_cache_<rank>/`（每个约 16 GB）。
   该键等价于 GE 的 `ge.graph_compiler_cache_dir`。

> ⚠ **缓存 key 由框架按「rank + 图序号 + 图结构」生成，不含 `precision_mode` /
> `externalWeight` 等 session 参数**：改这些参数后请手工清缓存（删掉
> `model_build_cache_<rank>/` 或 `model_cache_dir` 指向的目录）再跑，避免复用到旧产物；
> 换 `model_cache_dir`、换运行目录同样等于换一份缓存。
>
> 冷编译耗时参考：双 rank 一次特化编译首次约 20–25 分钟（本机实测冷启动整程 9 分钟），
> 命中编译缓存后起跑约 3–5 分钟。

---

## 4. MindSpore Lite 推理（零拷贝 KV）

### 测试数据（T2Reranking，必须自备）

推理输入**不随示例交付**，运行前必须显式指定两件套（缺 `--prefix-text-file` / `--queries` 会直接报错退出）：

| 环境变量 | 文件 | 用途 / 规格 |
|---|---|---|
| `PREFIX_TEXT` | `prefix_official.txt` | 共享前缀文本文件（`--prefix-text-file`，逐字节读入）；T2 官方模板前缀 56 token（桶 512） |
| `QUERIES` | `t2rerank_20_official.tsv` | 打分输入，每行 `query<tab>doc`；40 行 = T2Reranking 20 条 × (正例, 负例) |

数据来源：**T2Reranking**（中文重排基准，见 §7）。输入按 Qwen3-Reranker **官方模板**构造
（`<Instruct>/<Query>/<Document>` + assistant 后缀，文档取 `title + text`）：共享前缀放
`PREFIX_TEXT`，Query/Document 放在 tsv 的行内。

**口径约定（关键，写错会出现"分数还行但逐 logit 错位"）**：

- **tsv 是行式格式**：字段内的换行写成 `\n` 转义（推理脚本会还原；`\t` / `\\` 同理），
  因此官方模板那种多行 prompt 也能放进一行；
- 共享前缀**必须用文件**传入（`PREFIX_TEXT` → `--prefix-text-file`）：shell 的 `"$(cat f)"`
  会吃掉尾随换行，导致 prefix 少 1 token、后续 token 的位置全部错位；请用 `--prefix-text-file` 原样读入文件。

### 运行推理

```bash
# ① 环境（每个新 shell 一次；路径按实际环境改，或用 §1 的环境变量）
source $CANN/set_env.sh
export LD_LIBRARY_PATH=$MSLITE_HOME/runtime/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/pynew:$PYTHONPATH        # 可选：mindspore_lite 模块目录

# ② 打分（T2Reranking 40 对；两件套见上表）
T2=/path/to/t2rerank_20
python3 infer_qwen3_reranker_8b_mslite_cp.py \
  --device-ids 0,1 --model-id /path/to/Qwen3-Reranker-8B \
  --mindir-root ./output/mindir --cfg-dir ./output/configs \
  --prefix-bucket 512 --prefix-text-file $T2/prefix_official.txt \
  --suffix-buckets 512,1024,1536,2048 \
  --queries $T2/t2rerank_20_official.tsv --warmup 1 \
  --work-dir ./output/zc_run --json-out ./output/scores_cp2.json
```

产物：`output/scores_cp2.json`（逐对 `P(yes)` + 每对延迟）、`output/zc_run/logits_ge.npy`
（逐对 GE logits，可与自备的 CPU eager 参考做离线比对）。GE 编译缓存的目录与失效规则
见 §3「GE 缓存说明」（默认落在**运行目录**的 `model_build_cache_<rank>/`）。

**免拷贝原理**：每个 rank worker **同进程** build prefix + suffix 两张图；
prefix `predict(feeds, outputs=[t_k, t_v])` 把 K/V 写进预分配的
`mslite.Tensor(device="ascend:N")` 设备张量；suffix `predict(...)` 把这**同一对
device 张量**当输入直接喂进去（`_zc_feed()` 里 `i>=3` 槽位）——全程零 H2D/D2H。

### 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--device-ids` | 双 rank 设备（逗号分隔） | `2,3` |
| `--model-id` | 权重/tokenizer 路径 | 必填 |
| `--mindir-root` | `export_and_convert.sh` 的 mindir 根目录 | 必填 |
| `--cfg-dir` | 含 `{prefix,suffix}.config` 的目录 | 必填 |
| `--prefix-bucket` | 前缀桶 | `512` |
| `--prefix-text` | 共享前缀文本（字符串；注意 shell `$(cat)` 会吃掉尾随换行） | 二选一 |
| `--prefix-text-file` | 共享前缀文本**文件**（逐字节读入，推荐） | 二选一 |
| `--suffix-buckets` | 4 档桶 | `512,1024,1536,2048` |
| `--queries` | `query<tab>doc` 每行一对（T2Reranking 打分输入） | 必填 |
| `--warmup` | 每档桶首次进入时的 warmup 次数 | `1` |
| `--json-out` | 结果 JSON | 必填 |

### 推理示例输出

以下为 T2Reranking 官方模板（40 对）实跑输出（省略中间日志，本仓库只包含了mindir推理脚本，不包含torch CPU推理与输出对比脚本）：

```text
{"scores": [... 40 个 P(yes) ...], "suffix_latency_ms": [...],
 "prefix_real_len": 56, "prefix_bucket": 512}
CPU eager score(首对): 0.1306
GE score(首对):        0.1322
max|ge-eager|:  全局 0.4800  逐对均值 0.1681  (good<=0.7, bad~16.8)
≤0.7 的对数:    40/40
✓ precision OK
```

---

## 5. 精度验证与性能参考

### 测试环境

| 项目 | 说明 |
|---|---|
| 硬件 | Atlas 300I Duo |
| 软件 | CANN 9.1.0 + MindSpore Lite 2.11.0（aarch64）+ py311 |
| 模型 | Qwen3-Reranker-8B，fp16（非量化默认路径） |
| 数据 | T2Reranking 20 条 = 40 对（官方模板，自备，见 §4） |

### 精度（T2Reranking 20 条 = 40 对，官方模板）

输入按 **Qwen3-Reranker 官方模板**构造（`<Instruct>/<Query>/<Document>` + assistant 后缀，
文档取 `title + text`），共享前缀由 `PREFIX_TEXT` 指定（T2 为 `prefix_official.txt`）。

| 指标 | mindir (GE) | CPU eager（fp32 参考） |
|---|---|---|
| 成对排序（每 query 正例 > 负例） | **16/20 = 80%** | 16/20 = 80% |
| AUC | **0.7675** | 0.7650 |
| pos 均分 / neg 均分 | 0.4917 / 0.2570 | 0.4889 / 0.2577 |
| max\|logit−eager\|（40 对） | **40/40 ≤ 0.7**（最大 0.48，均值 0.168） | — |
| 与 eager 的分数相关 / 最大分差 | 0.9997 / 0.022 | — |

> **mindir 无损**：4 个错判 query（q2/q5/q16/q18）与 eager 完全一致 —— 80% 是模型本身
> 在该 prompt/数据上的水位，GE 相对 eager **未引入任何额外错判**。
> 注意 `P(yes)` 未做校准（正例均分仅 0.49），跨 query 比较请用排序类指标，
> 不要用 0.5 硬阈值。
> 表中的 CPU eager 参考与 `max|logit−eager|` 为**离线比对**结果：本示例只交付推理链路，
> 校验脚本与 eager 参考文件不随示例交付。

### 性能参考（实测）

非量化 fp16 + 零拷贝路径下 suffix 各档延迟（Atlas 300I Duo 双 rank；每档 warmup 1 次 +
重复 5 次、双 rank 各 5 次共 10 次统计；suffix 输入为脚本内合成的等长填充，
共享前缀真实长 254 token，pad到512长度）：

| prefix (token) | suffix (token) | mean (ms) |
|---:|---:|---:|
| 512| 512 | 200.1 |
| 512| 1024 | 409.7 |
| 512| 1536 | 674.7 |
| 512| 2048 | 911.3 |

> 上表为稳态延迟参考值：首次运行需先完成 GE 特化编译，之后命中编译缓存即可。
> **延迟扫描入口未随示例交付**（推理脚本只做 T2Reranking 打分）；如需复测，请在与上表
> 相同的环境下自行压测（同款运行时包、相同 `DEVICES`、warmup/repeats），数值仅供量级参考。

---

## 6. A8W8 量化开关（默认关闭）

导出脚本内置 A8W8 量化开关，**默认关闭（非量化 fp16）**；推理入口不感知量化
（量化只体现在导出的 MindIR 权重里，推理命令与 §4 完全一致）：

| 开关（导出脚本入参） | 默认 | 说明 |
|---|---|---|
| `--quant-dir`（一键脚本用环境变量 `QUANT` 转发） | 空 | A8W8 权重包目录；空 = 非量化 fp16 |
| `--quant-mode`（一键脚本用环境变量 `DSH_QUANT_MODE` 转发） | `off` | `dyn` = 动态 per-token A8W8，`static` = 静态 per-tensor A8W8 |
| `DSH_QUANT_DOWN_PROJ` | `auto` | down_proj 是否量化（auto = 包里有就量化） |
| `DSH_FUSE_SWIGLU` | `0` | SwigluDual 融合。默认 0（内置 SwiGlu，任意环境可跑）；`1` **仅量化路径**可用，非量化 fp16 下 SwigluDual 执行有偏差、精度会坏 |

```bash
# 一键脚本：环境变量只是入口，内部会转成 --quant-mode 传给导出脚本
QUANT=/path/to/w8a8_pack DSH_QUANT_MODE=dyn bash export_and_convert.sh

# 直接调用导出脚本：量化模式是脚本入参
python3 export_qwen3_reranker_8b_onnx_cp.py \
  --model-id /path/to/Qwen3-Reranker-8B --output-dir ./output/export \
  --device cpu --dtype fp16 --cp-size 2 --mode both \
  --quant-dir /path/to/w8a8_pack --quant-mode dyn

# 推理命令不变：见 §4「运行推理」（量化只影响导出的 MindIR 权重）
```

量化路径的算子约束（`input_names`/`output_names` 属性名、per-token `scale`
必须 float32、`bias` 补 `zeros(int32)` 占位防段错误。

## 7. 参考资源

- [MindSpore Lite Ascend 推理](https://www.mindspore.cn/lite/)
- [Qwen3-Reranker-8B ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Reranker-8B)
- [Qwen3-Reranker-8B HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-8B)
- [T2Reranking（中文重排基准，精度回归用）](https://huggingface.co/datasets/mteb/T2Reranking)

## 8. 许可证

本工具遵循 Qwen3 模型的许可证要求。
