# Qwen3.5-9B LiteBoost 端到端用例

本目录是 Qwen3.5-9B 适配器的 pytest 用例：精度用例把服务化
`Qwen35GraphRunner` 的 pooled prefill 输出与**离线生成的参考数据**逐位置
比对，性能用例跑 128/256/512/1024/2048 的 prefill + decode sweep 并检查
时延是否在合理范围。两条用例共用同一份 torchrun 驱动代码，但由 `--mode`
分开，不会为同一个 checkpoint 付两次加载代价。

参考数据（`accuracy_reference.npz`）是**未加速** HuggingFace 模型在 host
上用 float32 / eager 跑出来的，由 `accuracy_runner.py --mode reference`
（原始 torch pipeline，不走 LiteBoost）一次性离线生成；首次运行精度用例前
需按下方“生成参考数据”跑一遍，文件缺失时精度用例 skip。用例本身只跑加速
路径，不会在测试时加载原始模型，也不会和进程内的第二份模型做 A/B。

Runner 在启动时（`warmup()`）为每个 bucket 捕获一次 prefill 图，并捕获一张
decode 图；之后每个请求只拷贝静态输入、replay 图，不再做任何图捕获。因此
时延就是长驻服务里单请求的时延，不含首次捕获开销。

## 目录布局

| 文件 | 作用 |
|------|------|
| `test_qwen3_5_9b.py` | pytest 用例：`--mode accuracy` 比参考数据，`--mode perf` 判时延 |
| `conftest.py` | `QWEN35_9B_MODEL_DIR` / `QWEN35_9B_TP` / `QWEN35_9B_MODE` fixture、`--mode` 反选、torchrun 封装 |
| `accuracy_runner.py` | torchrun 入口：`--mode reference` 生成参考数据，`--mode fused` 跑加速路径 |
| `perf_runner.py` | torchrun 入口：prefill/decode sweep，也可单独当 benchmark 用 |
| `accuracy_reference.npz` | 参考数据（约 4 MB，由 `accuracy_runner.py --mode reference` 生成） |

## 前置条件

- 昇腾 NPU，**CANN 9.0.0**（8.5.2 缺少 decode 路径用到的
  `aclnnRecurrentGatedDeltaRule` 参数，会在第一个 decode step 报
  `gk parameter is not supported`）。
- Qwen3.5-9B bf16 权重目录，通过 `QWEN35_9B_MODEL_DIR` 指定。
- 单卡可直接运行；双卡用 `QWEN35_9B_TP=2`，由 conftest 自动起 torchrun。
  双卡方案由 `QWEN35_9B_MODE` 选择：
  - `tp`（默认）：注意力头 / GDN head / MLP 行切分 + 行并行 all-reduce，
    prefill 和 decode 都被切分；
  - `cp`（Ulysses 序列并行）：只把 8 个 full-attention 层的 forward 换成
    head-split + `all_gather_seq`，GDN/MLP 在各 rank 上复制。decode 序列长度
    为 1、小于 world size，CP 自动退回每 rank 静态 full-attention，**只有
    prefill 被并行**，decode 无收益（实测每 token 与 1P 持平）。

缺少权重或卡数不够时用例 skip，不会失败。

## 运行方式

在 `mindspore-lite/lite_boost` 目录下：

```bash
# 精度：加速路径 vs 参考数据
QWEN35_9B_MODEL_DIR=/path/to/Qwen3.5-9B \
    python -m pytest test/models/qwen3_5_9b --mode accuracy -m ascend_a2 -s

# 性能：一次 sweep 覆盖 5 个 bucket
QWEN35_9B_MODEL_DIR=/path/to/Qwen3.5-9B \
    python -m pytest test/models/qwen3_5_9b --mode perf -m ascend_a2 -s

# 两组都跑（默认 --mode all）
QWEN35_9B_MODEL_DIR=/path/to/Qwen3.5-9B \
    python -m pytest test/models/qwen3_5_9b -m ascend_a2 -s

# 双卡 TP（默认 QWEN35_9B_MODE=tp）
QWEN35_9B_MODEL_DIR=/path/to/Qwen3.5-9B QWEN35_9B_TP=2 \
    python -m pytest test/models/qwen3_5_9b --mode perf -m ascend_a2 -s

# 双卡 CP（Ulysses 序列并行，只影响 full-attention prefill）
QWEN35_9B_MODEL_DIR=/path/to/Qwen3.5-9B QWEN35_9B_TP=2 QWEN35_9B_MODE=cp \
    python -m pytest test/models/qwen3_5_9b --mode accuracy -m ascend_a2 -s
```

可在 `ASCEND_RT_VISIBLE_DEVICES` 里指定用哪两张卡（默认取 `0..TP-1`）。

两个 runner 也能脱离 pytest 手动跑（`--json-out` 输出机器可读结果）：

```bash
torchrun --nproc_per_node 2 test/models/qwen3_5_9b/perf_runner.py \
    --model-dir /path/to/Qwen3.5-9B --json-out perf.json
```

优化开关默认 `--opts mid`（4 个新增融合算子 + `fuse_rms_norm`、
`fuse_rms_norm_gated`、`fuse_inproj_ba`、`opt_a_log_precompute`），可用
`--opts none|new|mid` 切换；精度用例同样默认 `mid`，可单独传 `--opts`。

## 精度用例

### 参考数据内容

`accuracy_runner.py --mode reference` 用 transformers 在 host 上加载
**未加速**模型（float32、`attn_implementation="eager"`、不 import
LiteBoost），对固定 64-token prompt 做一次 `use_cache=False` 的前向，保存：

| 键 | 形状 | 说明 |
|----|------|------|
| `input_ids` | `[64]` | 固定 prompt，用例两侧共用，避免 tokenizer 版本漂移 |
| `positions` | `[4]` | `(15, 31, 47, 63)`，均匀覆盖早/中/末位上下文 |
| `logits` | `[4, 248320]` | 上述位置的完整 logits（float32） |
| `top1` | `[64]` | 每个位置的 top-1 token id |
| `model`, `dtype` | 标量 | 生成来源，便于回溯 |

只保存 4 个位置的完整 logits（约 4 MB）而不是全量 `[64, 248320]`：全量
会把仓库撑到 60 MB 以上，而 4 个位置加上 64 个 top-1 已能同时约束连续值
（cosine）和离散决策（top-1），并覆盖到末位——真正决定下一个 token 的
那个位置。

生成参考数据（首次运行精度用例前必做；换 checkpoint 或 transformers 版本后
需重新生成）：

```bash
python test/models/qwen3_5_9b/accuracy_runner.py --mode reference \
    --model-dir /path/to/Qwen3.5-9B \
    --out test/models/qwen3_5_9b/accuracy_reference.npz
```

### 加速侧

`--mode fused` 走和线上请求完全相同的路径：
`boost_qwen3_5_9b(graph_mode=True, bucketed_prefill=True)` →
`init_bucket_pipelines` → `pooled_prefill`，然后从
`model._pool_outs[bucket]` 取 hidden states 过 `lm_head`，保存同样的
`logits` / `top1` 两个键。用例复用参考数据的 `input_ids`，两侧看到的是
同一个 prompt。

### 判据

| 指标 | 默认阈值 | 说明 |
|------|----------|------|
| 4 个位置 cosine 均值 | >= 0.99 | 连续值一致性 |
| 4 个位置 top-1 | 全部一致 | 严格项：决定下一个 token |
| 全部 64 个位置 top-1 一致率 | >= 0.90 | 粗判据；bf16 near-tie 会翻转个别位置 |

1P 实测（Ascend A2 280T、CANN 9.0.0、bf16、`--opts mid`）：
4 个位置 cosine 为 0.999959 / 0.999973 / 0.999981 / 0.999785（均值
0.999924），采样位置 top-1 4/4 一致，全 prompt top-1 62/64。不一致的
2 个位置是 bf16 下 top-1/top-2 极接近的 near-tie，这也是全 prompt 判据
放宽到 0.90 的原因；末位与其余采样位置严格一致。

双卡实测（同机同配置，`QWEN35_9B_TP=2`）：

| 方案 | cosine 均值 | 采样位置 top-1 | 全 prompt top-1 |
|------|------------:|---------------:|----------------:|
| TP-2 | 0.999957 | 4/4 | 62/64 |
| CP-2 | 0.999926 | 4/4 | 63/64 |

两种方案的 logits 在相同输入下并不逐位相同（all-reduce 与 all-gather
的求和顺序不同），但都落在同一阈值内。多卡精度运行时
`accuracy_runner.py` 会先设 `HCCL_DETERMINISTIC=true` 再初始化通信域，
保证同一方案多次运行可复现；性能路径不设该变量，以免把 deterministic
算法带来的额外开销算进时延。

## 性能用例

`perf_runner.py` 一次启动完成所有捕获，然后对每个长度发一个请求
（prefill + 32 个 decode token），输出：

- `prefill_ms`：cache 复位、静态输入拷贝、prefill 图 replay 之和
  （`reset + copy + replay`），与 403 patch 的 `timing["prefill"]` 同口径；
- `first_token_ms`：TTFT，即 `timing["prefill_total"]`，额外包含输入
  预处理与 `lm_head`；
- `decode_ms_per_tok`：稳态 decode 每 token 时间，首 token 不计入（它来自
  prefill），含每步输入拷贝、图 replay、stream 同步和 argmax。

用例判据（回归护栏，不是性能门禁）：

| 用例 | 判据 |
|------|------|
| `test_prefill_decode_latency` | `prefill_ms < QWEN35_9B_PREFILL_CAP_MS`（默认 1000）、`decode_ms_per_tok < QWEN35_9B_DECODE_CAP_MS`（默认 200）、decode 满 32 token |
| `test_prefill_scales_with_length` | 128 < 512 < 2048 的 `prefill_ms` 单调递增 |
| `test_decode_latency_is_stable` | 各 bucket 间 decode 每 token 时延的 max/min <= 1.5 |
| `test_graph_capture_completes` | 5 个 bucket 都有 capture 记录且全部 > 0，decode 图也已捕获；benchmark 上报的 `parallel_mode` 与 `QWEN35_9B_MODE` 一致（否则 `MODE=cp` 会静默测到 TP） |

阈值可用 `QWEN35_9B_PREFILL_CAP_MS` / `QWEN35_9B_DECODE_CAP_MS` 覆盖，
换机器时只调这两个环境变量。

### 1P 实测结果

单卡 Ascend A2 280T、CANN 9.0.0、bf16、`--opts mid`，默认 bucket
`128,256,512,1024,2048`，同一进程按默认 sweep 跑一遍（单次采样）：

启动捕获（只发生一次）：`prefill_init=2.46s`（5 张 bucket 图合计，其中
128 的 warmup 占 0.78 s）、`decode_init=0.31s`。

| length | bucket | prefill_ms | first_token_ms | decode_ms_tok | decode_tok_s |
|--------|--------|-----------:|---------------:|--------------:|-------------:|
| 128    | 128    | 48.19  | 52.49  | 32.49 | 30.78 |
| 256    | 256    | 61.71  | 65.21  | 32.37 | 30.90 |
| 512    | 512    | 86.67  | 90.10  | 32.36 | 30.90 |
| 1024   | 1024   | 141.75 | 145.32 | 32.36 | 30.90 |
| 2048   | 2048   | 238.40 | 242.33 | 32.37 | 30.89 |

decode 时延与 prefill bucket 无关（32.36–32.49 ms/tok），符合
“decode 图与起始长度无关”的设计；prefill 随长度增长且 2048 相对 1024
的斜率明显变大（长序列 attention 占比上升）。这些数字是单次采样的横向
参考，不是严格 benchmark：跨进程 A/B 噪声可达 20–30 ms，正式结论应重复
多次取中位数并固定进程/卡。

### 双卡实测结果

同一台 Ascend A2 280T、CANN 9.0.0、bf16、`--opts mid`，各配置一个进程
跑完整个 sweep（单次采样）：

| length | 1P prefill | TP-2 prefill | CP-2 prefill | 1P decode | TP-2 decode | CP-2 decode |
|--------|-----------:|-------------:|-------------:|----------:|------------:|------------:|
| 128    | 48.19  | 70.00  | 82.12  | 32.49 | 34.50 | 32.46 |
| 256    | 61.71  | 66.53  | 71.78  | 32.37 | 34.36 | 32.35 |
| 512    | 86.67  | 86.48  | 97.39  | 32.36 | 34.34 | 32.37 |
| 1024   | 141.75 | 125.11 | 154.36 | 32.36 | 34.35 | 32.36 |
| 2048   | 238.40 | 196.29 | 251.76 | 32.37 | 34.38 | 32.35 |

（prefill / decode 单位分别为 ms 和 ms/tok，均为单次采样。）

9B 这个尺寸单卡算力本来就够，双卡收益仅出现在长序列：TP-2 从 1024 起
反超 1P（2048 为 196 vs 238 ms），短 bucket 被 all-reduce 通信吃掉；
CP-2 的 attention 分片要到 2048 才勉强抵平通信（251.76 vs 238.40），
整体不值得用。decode 侧 TP-2 比 1P 慢约 2 ms/tok（每层两次 all-reduce），
CP-2 与 1P 持平（decode 序列长 1，CP 退回 1P 的静态 full-attention）。
验收用例只保证“多卡能跑通且时延在合理范围”，不设多卡加速门禁。

## 服务化约束

- prefill 只支持 pooled bucket 图：prompt 长度必须不超过最大 bucket，
  超出时 `pooled_prefill` / Runner 直接报错，不会自动回退到
  `graph_generate` 的溢出路径。
- prefill 与 decode 共用同一份 pooled cache，其容量为
  `max(buckets) + max_new_tokens + 16`（`max_new_tokens` 含首 token）。
  请求要求的总长度超出容量时 `pooled_decode` 报错，需在
  `init_bucket_pipelines(..., max_new_tokens=...)` / Runner 构造时配大。
- decode 图捕获后不能在同一进程再走 `graph_generate` / multi-round 等
  使用不同 `max_cache_len` 的路径，否则全局 decode mask 指针失效；
  `pooled_decode` 会校验 cache 容量并报错。
- 纯文本请求开始时 Runner 会清空 `rope_deltas`，避免上一张图的 mRoPE
  delta 泄漏到本请求；图文请求由 prefill 重新计算。
- 多模态 prompt 的展开由调用方负责：Runner 只接收最终 `input_ids` 以及
  `pixel_values` / `extra_token_embeds`，按最终长度选 bucket 并 replay，
  不替调用方解析占位符。当前用例只覆盖文本，多模态需调用方自备素材。
- 生成默认在命中 checkpoint 的 EOS（Runner 构造时从
  `config.text_config.eos_token_id` 读取）时提前结束，传
  `stop_token_ids=()` 可关闭提前停止。`timing` 给出 `finish_reason`
  （`stop` / `length`）、`stop_token` 和实际执行的 `decode_steps`。
  异步 replay（`opt_no_decode_sync=True`）每 8 步查看一次输出，命中最坏
  多跑当前 block 的剩余步数。性能 sweep 为了固定 token 数，显式使用
  `stop_token_ids=()`。
- 没有使用 prefix cache，也没有 multi-round / 变长 batch 组合；每个长度
  只跑一组 prompt。服务 Runner 一次只处理一个请求（静态输入/输出和共享
  cache 都是单份），并发请求需要外部串行或每请求独立模型。
