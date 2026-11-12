# Qwen3.5-9B LiteBoost 推理（bf16 / 1P / TP / CP / 图模式）

`Qwen3_5ForConditionalGeneration`（32 层，每 4 层一个 full attention，其余为 GDN
线性注意力；hidden 4096）的 NPU 推理适配：加载 Hugging Face bf16 权重，叠加
LiteBoost 的 CANN 融合算子（short-conv+SiLU、GDN q/k 归一化、`npu_rms_norm` /
`npu_swiglu` / `npu_prompt_flash_attention`），对 128/256/512/1024/2048 五档
prefill 与 decode 做 NPUGraph 捕获与回放，支持 1P 与 TP / CP 多卡推理。

本文档按「部署环境 → 下载权重 → 跑推理脚本 → 跑测试用例」的顺序编写，命令默认在
仓库的 `mindspore-lite/lite_boost` 目录下执行。性能与精度数据来自
Ascend A2 280T 实测。

---

## 0. 部署环境

| 组件 | 验证版本 | 说明 |
|------|----------|------|
| 硬件 | Ascend A2 280T | 1P 单卡放 19 GiB 权重；TP-2 / CP-2 用双卡 |
| CANN | **9.0.0** | 8.5.2 缺少 decode 路径用到的 `aclnnRecurrentGatedDeltaRule` 参数，第一个 decode step 报 `gk parameter is not supported` |
| Python | 3.12 | |
| PyTorch | 2.10.0 | |
| torch_npu | 2.10.0.post6 | |
| transformers | 5.19.0 | 需要 `Qwen3_5ForConditionalGeneration` 与 `eager` attention 实现 |
| lite_boost | 0.3.0（本仓库） | |

> 环境的构建与安装见 [lite_boost/CLAUDE.md](../../../CLAUDE.md)，本文不重复。

## 1. 下载权重

模型 ID：`Qwen/Qwen3.5-9B`（bf16，约 19 GiB）。

```bash
# 方式一：Hugging Face
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-9B --local-dir /data/models/Qwen3.5-9B

# 方式二：ModelScope（国内网络）
pip install -U modelscope
modelscope download --model Qwen/Qwen3.5-9B --local_dir /data/models/Qwen3.5-9B
```

无需格式转换：加载时用 `AutoModelForImageTextToText` + `attn_implementation="eager"`，
切分与算子替换都由 lite_boost 完成。

## 2. 运行推理 / benchmark 脚本

`test/models/qwen3_5_9b/perf_runner.py` 加载权重、施加选定的优化预设，启动时为
五档 prefill 各捕获一张图、再捕获一张 1-token decode 图，然后每档跑
「prefill + 32 步 decode」。它和性能用例共用同一个脚本：

```bash
source /path/to/ascend-toolkit/set_env.sh   # CANN 9.0.0 环境，路径以实际安装为准
cd mindspore-lite/lite_boost/test/models/qwen3_5_9b

# 1P
python perf_runner.py --model-dir /data/models/Qwen3.5-9B

# TP-2（--parallel-mode=tp，默认）/ CP-2（--parallel-mode=cp）
torchrun --nproc_per_node=2 perf_runner.py \
    --model-dir /data/models/Qwen3.5-9B --parallel-mode tp
```

正常输出（1P，rank 0）：

```text
Device: npu:0, world_size=1, parallel_mode=tp, opts=mid, lengths=[128, 256, 512, 1024, 2048]
STARTUP: prefill_init=2.46s decode_init=0.31s wall=3.74s
...
 prefill  bucket  prefill(ms)   TTFT(ms)  decode(ms/tok)  decode(tok/s)
     128     128        48.25      52.48           32.36          30.79
     256     256        62.06      65.74           32.22          31.04
     512     512        87.20      90.60           32.23          31.03
    1024    1024       141.89     145.30           32.23          31.03
    2048    2048       238.86     242.48           32.25          31.01
```

口径说明：

- `prefill(ms)`：cache 复位 + 静态输入拷贝 + prefill 图回放之和
  （`reset + copy + replay`），不含启动时的捕获开销；
- `TTFT(ms)`：`prefill_total`，额外包含输入预处理与 `lm_head`；
- `decode(ms/tok)`：稳态 decode 每 token 时间，首 token 来自 prefill 不计入；
  每步包含输入拷贝、图 replay、stream 同步与 argmax。
- 启动捕获只发生一次（五档 prefill 图合计约 2.5 s，decode 图约 0.3 s，
  其中 decode 图捕获前有 3 步 warmup），不计入性能数字。
- `--json-out` 写出 `{lengths, buckets, world_size, parallel_mode, opts, startup,
  results[]}`，用例读的就是这份 JSON。

常用参数：

| 参数 | 说明 |
|------|------|
| `--model-dir` | HF bf16 权重目录（必填） |
| `--lengths` / `--buckets` | 逗号分隔的 prefill 档位，默认 `128,256,512,1024,2048` |
| `--opts` | 优化预设：`none` / `new` / `mid`（默认） |
| `--parallel-mode` | 多卡方案：`tp`（默认）/ `cp` |
| `--max-new-tokens` | 每档 decode 步数，默认 32 |
| `--num-warmup` | 图捕获前的 eager warmup 步数，默认 2 |
| `--device-id` | 1P 模式的 NPU 编号 |
| `--json-out` | 结果写 JSON |

## 3. 运行测试

精度与性能用例在同一个文件
`lite_boost/test/models/qwen3_5_9b/test_qwen3_5_9b.py`，用 `--mode` 选择跑哪一半
（两种 mode 互斥，避免同一个 checkpoint 加载两次）：

| mode | 内容 | 是否需要权重 |
|------|------|--------------|
| `accuracy` | pooled prefill 的 logits 与免 LiteBoost 基线逐位置对比 | 需要 |
| `perf` | 128/256/512/1024/2048 五档 prefill + decode 时延 | 需要 |
| `all` | 默认值，两者都跑 | 需要 |

精度基线沿用 boogu-image 的 `reference.npz` 做法：`accuracy_reference.npz` 是
**免 LiteBoost** 的原始模型在 host 上用 float32 / eager 对固定 64-token prompt
的输出（约 4 MB）。它由 `accuracy_runner.py --mode reference` 用未加速的
torch 模型（float32 / eager）在 host 上离线生成，首次跑精度用例前生成一次，
文件缺失时精度用例 skip。用例只跑融合路径，不会在测试时再加载一遍未加速
模型。文件里存：

- `input_ids`：两边共用同一串 token，不受 tokenizer 版本影响；
- 4 个位置（15/31/47/63）的完整 logits：算 cosine；
- 全部 64 个位置的 top-1：算逐位置一致率。

判据分两层：4 个采样位置的 cosine 均值 ≥ 0.99、top-1 必须 4/4 一致（这些位置决定
下一个 token）；全 prompt 一致率只作粗筛（≥ 0.90），因为 bf16 下 top-1/top-2
接近的位置会翻转。

多卡精度在初始化通信域前设 `HCCL_DETERMINISTIC=true`（`accuracy_runner.py`
自动设置），保证同一方案重复运行可复现；性能路径不设，避免把 deterministic
算法的额外开销算进时延。

生成参考数据（CPU，float32，约数分钟；首次或换 checkpoint 后执行）：

```bash
cd mindspore-lite/lite_boost/test/models/qwen3_5_9b
python accuracy_runner.py --mode reference \
    --model-dir /data/models/Qwen3.5-9B \
    --out accuracy_reference.npz
```

```bash
source /path/to/ascend-toolkit/set_env.sh   # CANN 9.0.0 环境，路径以实际安装为准
cd mindspore-lite/lite_boost/test

export QWEN35_9B_MODEL_DIR=/data/models/Qwen3.5-9B
export QWEN35_9B_TP=1                 # 双卡填 2
export QWEN35_9B_MODE=tp              # 双卡可改 cp
export ASCEND_RT_VISIBLE_DEVICES=0    # 双卡填 0,1

# 1) 端到端 logits vs 免 LiteBoost 基线（约 35 s）
python -m pytest models/qwen3_5_9b --mode accuracy -m ascend_a2 -s

# 2) 五档性能（约 45 s）
python -m pytest models/qwen3_5_9b --mode perf -m ascend_a2 -s

# 3) 精度 + 性能全量（约 70 s）
python -m pytest models/qwen3_5_9b -m ascend_a2 -s
```

缺少权重或卡数不足时用例 **skip** 而不是 fail（`QWEN35_9B_MODEL_DIR` 未设置、
`ASCEND_RT_VISIBLE_DEVICES` 少于 `QWEN35_9B_TP` 都算）。

期望结果：

- `test_end_to_end_logits_match_reference`：1P cosine 0.999924（四个位置
  0.999959 / 0.999973 / 0.999981 / 0.999785）、采样位置 top-1 4/4、全 prompt
  62/64；TP-2 cosine 0.999957、4/4、62/64；CP-2 cosine 0.999926、4/4、63/64。
- `test_prefill_decode_latency[128..2048]`：prefill < 1000 ms、
  decode < 200 ms/tok、decode 满 32 token。
- `test_prefill_scales_with_length`：128 < 512 < 2048 的 prefill 单调递增。
- `test_decode_latency_is_stable`：各档 decode 每 token 时延 max/min ≤ 1.5。
- `test_graph_capture_completes`：五档都有 capture 记录且全部 > 0，decode 图也已
  捕获；benchmark 上报的 `parallel_mode` 必须与 `QWEN35_9B_MODE` 一致。

时延上限可按机器放宽：

```bash
export QWEN35_9B_PREFILL_CAP_MS=1500
export QWEN35_9B_DECODE_CAP_MS=300
```

## 4. 优化内容

```text
boost_qwen3_5_9b()
├── 多卡：TP（头/MLP 切分 + 行并行 all-reduce）或 CP（Ulysses 序列并行）（model.py）
├── 算子融合：short-conv+SiLU / GDN q-k 归一化 / RMSNorm / gated RMSNorm /
│             gate_up+SWIGLU / in_proj_b+a / full-attn QKV 投影（model.py）
└── 图捕获：5 档 prefill 图 + 1-token decode 图（graph_decode.py）
```

### 4.1 张量并行（TP）

`boost_qwen3_5_9b(..., mode="tp")` 原地切分，不需要离线转换权重：

| 模块 | 切分方式 | 每卡形状（TP=2） |
|------|----------|------------------|
| full-attn `q/k/v_proj` | 列并行（按 head，16 q head / 4 kv head） | q: 8 head，k/v: 2 head（head_dim 256） |
| full-attn `o_proj` | 行并行 + all-reduce | `[4096, 8×256]` |
| GDN `in_proj_qkv` + conv | 列并行（16 key head / 32 value head，conv_dim 随切） | 8 key / 16 value head |
| GDN `in_proj_z/b/a` | 列并行 | — |
| GDN `out_proj` | 行并行 + all-reduce | — |
| GDN `A_log` / `dt_bias` | 按 value head 切 | 16 个 value head |
| MLP `gate/up/down_proj` | gate/up 列并行，down 行并行 + all-reduce | intermediate 6144 |

- attention 要求 `num_attention_heads(16)` 与 `num_key_value_heads(4)` 都能被 TP
  整除，GDN 要求 key/value head 能整除，所以 **TP ∈ {1, 2, 4}**。
- 每个 decoder 层只在 `o_proj` / `out_proj` / `down_proj` 后各做一次行并行
  all-reduce，其它通信为零。

### 4.2 上下文并行（CP）

`mode="cp"` 走 Ulysses：只把 8 个 full-attention 层的 forward 换成
`all_gather_seq`（按 head 切序列、attention 后按序列拼回），GDN 与 MLP 在每个
rank 上复制计算。

- 只并行 full-attention 的 prefill；decode 序列长度为 1、小于 world size，自动退回
  每 rank 的静态 full-attention，因此 CP 的 decode 时延与 1P 持平。
- CP 不减少权重显存，收益只在长序列 attention：实测 2048 档仅与 1P 打平。

### 4.3 算子融合

| 原始计算 | 融合后 | 覆盖范围 |
|----------|--------|----------|
| GDN short-conv（`conv1d + bias + silu`）与 conv_state 切片 | `FusedCausalConvSilu`（C++ 自定义算子，AIV） | 24 个 linear-attention 层，仅 1P |
| chunk GDN 里 q/k 的 L2 归一化 | `FusedQkL2NormChunkGatedDeltaRule`（归一化融合进 chunk 递推） | 24 层，1P / 多卡均可用 |
| `RMSNorm`（统计 + 缩放） | `torch_npu.npu_rms_norm` | 每层 input / q_norm / k_norm + final norm |
| gated RMSNorm（norm + silu(gate)） | `npu_rms_norm` + FP32 SiLU + mul | 24 个 GDN 层 |
| `gate_proj` + `up_proj` 两个 GEMM + `silu(gate)·up` | 权重拼接成单 GEMM + `npu_swiglu` | 32 个 decoder 层 |
| GDN `in_proj_b` + `in_proj_a` | 两个小 GEMM 合成一次 | 24 层 |
| `A_log` 的 `exp` 预计算 | 构造时预计算并缓存 | 24 层 |
| full-attn 的 `q_proj`+gate / `k_proj` / `v_proj` | 权重拼接成一次 QKV GEMM + gate GEMM | 8 个 full-attention 层，仅 1P |
| prefill / decode attention | `npu_prompt_flash_attention`（静态 mask） | 8 个 full-attention 层 |

`--opts mid`（默认）= 上表全部可用项；`new` 只开四个新增融合
（`opt_fused_conv_silu` / `opt_full_attn_proj_fuse` /
`opt_chunk_qk_l2norm_sink` / `opt_chunk_out_zero_pad`）；`none` 全关。

标「仅 1P」的融合在 `world_size > 1` 时会打印 warning 并自动跳过，多卡走参考路径。
适配层还提供未进预设的实验开关（`opt_w8a16_down_proj` / `opt_w8a16_gate_up` /
`opt_inproj_4way` 等），默认关闭。

### 4.4 图模式与服务化路径

- **prefill**：`init_bucket_pipelines` 为每档 bucket 捕获一张整图，真实 token 拷入
  静态输入后 pad 到档位；图内完成 32 层（含融合算子）与 KV / GDN 状态写入。
- **decode**：`init_decode_pipeline` 在 prefill 之后捕获一张 1-token 图
  （warmup 3 步再捕获，避免把首次编译带进图），之后每步只拷 token id / position
  并 replay，host 侧只做 argmax 与取回。
- `Qwen35GraphRunner` 把上面两步包成服务接口：
  `warmup()` 捕获一次 → 每个请求 `prefill()` + `decode()`。请求之间共享 pooled
  cache 与图静态 buffer，因此一次只处理一个请求。
- cache 容量为 `max(buckets) + max_new_tokens + 16`；超出时 `pooled_decode` 报错，
  需在启动时把 `max_new_tokens` 配大。

## 5. 性能数据（Ascend A2 280T，bf16，`--opts mid`）

单次采样，`128,256,512,1024,2048` 五档，prefill 单位 ms、decode 单位 ms/tok：

| length | 1P prefill | TP-2 prefill | CP-2 prefill | 1P decode | TP-2 decode | CP-2 decode |
|--------|-----------:|-------------:|-------------:|----------:|------------:|------------:|
| 128    | 48.25  | 70.00  | 59.18  | 32.36 | 34.50 | 32.45 |
| 256    | 62.06  | 66.53  | 72.15  | 32.22 | 34.36 | 32.24 |
| 512    | 87.20  | 86.48  | 98.18  | 32.23 | 34.34 | 32.24 |
| 1024   | 141.89 | 125.11 | 154.58 | 32.23 | 34.35 | 32.23 |
| 2048   | 238.86 | 196.29 | 251.89 | 32.25 | 34.38 | 32.23 |

结论：

- decode 时延与 prefill 档位无关（32.22–32.36 ms/tok），符合「decode 图与起始
  长度无关」的设计。
- 9B 单卡算力足够：TP-2 从 1024 档起才反超 1P（2048 档 196 vs 239 ms），短档
  被 all-reduce 吃掉；TP-2 每 token 多约 2 ms（每层两次行并行通信）。
- CP-2 的 attention 分片要到 2048 档才勉强抵平通信，decode 与 1P 持平，整体不
  值得部署。
- 128 档两道多卡采样波动较大（CP-2 在 59–82 ms 之间），跨进程噪声可达
  20–30 ms，正式对比应重复多次取中位数并固定进程/卡。

## 6. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `QWEN35_9B_MODEL_DIR` | — | 测试用权重目录，未设置时用例 skip |
| `QWEN35_9B_TP` | `1` | 测试用并行度（1 / 2） |
| `QWEN35_9B_MODE` | `tp` | 多卡方案：`tp` / `cp` |
| `QWEN35_9B_PREFILL_CAP_MS` | `1000` | 性能用例 prefill 上限 |
| `QWEN35_9B_DECODE_CAP_MS` | `200` | 性能用例 decode 上限 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0..TP-1` | 用例启动 torchrun 时使用的卡 |
| `HCCL_DETERMINISTIC` | 精度路径自动置 `true` | 固定集合通信归约顺序，多卡精度可复现 |

## 7. 常见问题

**第一个 decode step 报 `aclnnRecurrentGatedDeltaRule ... gk parameter is not supported`**
CANN 版本低于 9.0.0，或没有 `source` 9.0.0 的 `set_env.sh`。prefill 不经过该算子，
所以只跑精度（prefill-only）时不会暴露这个问题。

**性能用例全部 skip**
`QWEN35_9B_MODEL_DIR` 没设或目录不存在，或者 `ASCEND_RT_VISIBLE_DEVICES` 的卡数
小于 `QWEN35_9B_TP`。

**`QWEN35_9B_MODE=cp` 但性能数字和 TP 一样**
检查 benchmark 上报的 `parallel_mode`；`test_graph_capture_completes` 会断言它
与配置一致，若不一致说明 runner 没有收到 `--parallel-mode`。

**多卡时 fused short-conv / QKV 投影没有生效**
这两项融合仅支持 1P，`world_size > 1` 时适配层打印 warning 并回退参考路径，属于
预期行为。

**一个进程里混用不同 test 路径**
decode 图捕获后不能再走 `graph_generate` / multi-round 等使用不同 `max_cache_len`
的路径，否则全局 decode mask 指针失效；服务路径只允许 pooled prefill/decode。
