# Qwen3-30B-A3B LiteBoost 推理（bf16 / TP / 图模式）

`Qwen3MoeForCausalLM`（48 层，128 专家、top-8，hidden 2048）的 NPU 推理适配：
加载 Hugging Face bf16 权重，叠加 LiteBoost 的 CANN 融合算子
（`npu_add_rms_norm` / `npu_swiglu` / MoE 全链路 / `aclnnQkvRmsNormRopeCache`），
对 128/256/512/1024/2048 五档 prefill 与 decode 分别做 NPUGraph 捕获与回放，
支持 1p 与 TP 多卡推理。

本文档按「部署环境 → 下载权重 → 跑推理脚本 → 跑测试用例」的顺序编写，命令默认在
仓库的 `mindspore-lite/lite_boost` 目录下执行。性能与测试数据来自
4 × Ascend A2 280T 实测。

---

## 0. 部署环境

| 组件 | 验证版本 | 说明 |
|------|----------|------|
| 硬件 | Ascend A2 280T，4 卡 | TP=4 每卡权重 14.2 GB；TP=2 每卡 30.6 GB，需 ≥ 33 GB/卡的机器（如 300I Duo 44 GB） |
| CANN | **9.0.0** | 8.5.2 没有 `aclnnQkvRmsNormRopeCache`，prefill 融合算子无法解析 |
| Python | 3.12 | |
| PyTorch | 2.10.0 | |
| torch_npu | 2.10.0.post6 | |
| transformers | 5.19.0 | 需要 `experts_implementation` 参数（Qwen3MoE eager 专家实现） |
| lite_boost | 0.3.0（本仓库） | |

> 环境的构建与安装见 [lite_boost/CLAUDE.md](../../../CLAUDE.md)，本文不重复。

## 1. 下载权重

模型 ID：`Qwen/Qwen3-30B-A3B`（bf16，约 57 GiB）。

```bash
# 方式一：Hugging Face
pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3-30B-A3B --local-dir /data/models/Qwen3-30B-A3B

# 方式二：ModelScope（国内网络）
pip install -U modelscope
modelscope download --model Qwen/Qwen3-30B-A3B --local_dir /data/models/Qwen3-30B-A3B
```

目录里应有：

```text
config.json                          generation_config.json
model-00001-of-00016.safetensors ... model-00016-of-00016.safetensors
model.safetensors.index.json         tokenizer.json  tokenizer_config.json
vocab.json  merges.txt
```

无需预先做格式转换，加载时 `attn_implementation="eager"`、
`experts_implementation="eager"`，TP 切分与算子替换都由 lite_boost 完成。

## 2. 运行推理 / benchmark 脚本

`test/models/qwen3_30b_a3b/perf_runner.py` 会加载权重、施加全部融合
（RMSNorm / FFN / MoE / router），然后对五档 prefill 各捕获一次图、回放一次，
并在每档后跑 32 步 decode。它和性能测试放在一起，`--mode perf` 用的就是同一个
脚本：

```bash
source /path/to/ascend-toolkit/set_env.sh   # CANN 9.0.0 环境，路径以实际安装为准
cd mindspore-lite/lite_boost/test/models/qwen3_30b_a3b

torchrun --nproc_per_node=4 perf_runner.py \
    --model-dir /data/models/Qwen3-30B-A3B \
    --graph \
    --json-out /tmp/qwen3_30b_a3b_perf.json
```

`--nproc_per_node` 就是 TP 并行度，`ASCEND_RT_VISIBLE_DEVICES` 至少要有同样多的卡。
正常输出（rank 0）：

```text
Device: npu:0, world_size=4, graph=True, lengths=[128, 256, 512, 1024, 2048]
  load (CPU): ...
  boost: ...
  to(npu:0): ...
[bench] prefill 128 tokens + 32 decode tokens ...
  prefill 50.04 ms, decode 14.14 ms/tok, reply='filling those contexts dominates ...'
...
==============================================================================
 prefill  bucket  prefill(ms)  decode(ms/tok)  decode(tok/s)
     128       128        50.04           14.14          70.72
     256       256        55.67           14.23          70.27
     512       512        64.99           14.12          70.82
    1024      1024        87.48           14.15          70.67
    2048      2048       129.57           14.19          70.47
==============================================================================
```

口径说明：

- `prefill(ms)` 是**图回放**耗时，不含捕获；首档捕获约 2–4 s（冷启动），其余档约
  0.4 s，单独记录在 JSON 的 `prefill_capture_s`，不计入性能数字。
- `decode(ms/tok)` 是每档生成 32 token 的稳态平均，1-token 前向 + lm_head +
  argmax + KV 追加都在图内，host 侧只做输入拷贝。
- `--json-out` 写出 `{lengths, world_size, graph, results[], session}`，测试用例
  读的就是这份 JSON。

常用参数：

| 参数 | 说明 |
|------|------|
| `--model-dir` | HF bf16 权重目录（必填） |
| `--tokenizer-dir` | tokenizer 目录，默认同 `--model-dir` |
| `--lengths` | 逗号分隔的 prefill 档位，默认 `128,256,512,1024,2048` |
| `--max-new-tokens` | 每档 decode 步数，默认 32 |
| `--graph` | 开图捕获与回放；去掉即 eager 路径（仅用于对照，见「常见问题」） |
| `--session` / `--prefix-len` | 额外跑共享前缀会话（默认前缀 512 token） |
| `--device-id` | 1p 模式的 NPU 编号 |
| `--json-out` | 结果写 JSON |

> 1p（`--nproc_per_node=1`）要求单卡能放下 61 GB 权重；Ascend A2 280T（32 GB）与
> 300I Duo（44 GB）都放不下，因此本文硬件上实际部署请用 TP。

## 3. 运行测试

精度与性能用例在同一个文件
`lite_boost/test/models/qwen3_30b_a3b/test_qwen3_30b_a3b.py`，用 `--mode` 选择
跑哪一半（两种 mode 互斥，避免重复加载 61 GB 权重）：

| mode | 内容 | 是否需要权重 |
|------|------|--------------|
| `accuracy` | 全量融合 TP 的 prefill logits 与免 LiteBoost 基线对比 | 需要 |
| `perf` | 128/256/512/1024/2048 五档 prefill + decode 时延 | 需要 |
| `all` | 默认值，两者都跑 | 需要 |

精度基线沿用 boogu-image 的 `reference.npz` 做法：`accuracy_reference.npz` 是
**免 LiteBoost** 的原始 `Qwen3MoeForCausalLM`（float32 / CPU / 不切分）对固定
256-token prompt 的输出，随用例一起入库；用例只跑融合路径，读这份数据判定，不会
再去加载一遍未加速的模型。文件里存：

- `input_ids`：两边共用同一串 token，不受 tokenizer 版本影响；
- 4 个采样位置（63/127/191/255）的完整 logits：算 cosine；
- 全部 256 个位置的 top-1 与 `top1 - top2` margin：算逐位置一致率。

自然文本里约四成位置的一二名 logit 只差不到 1.0，bf16 舍入就足以让它们互换，
所以一致率拆成两层：**margin ≥ 1.0 的位置必须 100% 一致**（这些是模型真正做了
判断的位置），全量一致率只作粗筛（阈值 0.95）。退化到单次运行会把 1 个近 tie
翻转放大成 1.6% 的波动，这正是之前 0.9844 的来历。

bf16 + TP 的 all-reduce 归约顺序本身不确定（与随机种子无关，推理路径没有随机
算子）：同一进程连续两次 prefill 的 logits 就能差到 cosine 0.998。用例因此用
`HCCL_DETERMINISTIC=true` 启动融合路径，并连跑 3 轮，要求三轮 top-1 逐位相同；
否则一次运行的 0.9xxx 无法区分是稳定结果还是舍入抖动。开启后实测同进程内与
跨进程的 logits 均逐位一致。

换 checkpoint 时才需要重新生成（CPU，约数分钟）：

```bash
cd mindspore-lite/lite_boost/test/models/qwen3_30b_a3b
python accuracy_runner.py --mode reference \
    --model-dir /data/models/Qwen3-30B-A3B \
    --out accuracy_reference.npz
```

```bash
source /path/to/ascend-toolkit/set_env.sh   # CANN 9.0.0 环境，路径以实际安装为准
cd mindspore-lite/lite_boost/test

export QWEN3_30B_A3B_MODEL_DIR=/data/models/Qwen3-30B-A3B
export QWEN3_30B_A3B_TP=4
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

# 1) 端到端 logits vs 免 LiteBoost 基线（TP=4，约 2 min）
python -m pytest models/qwen3_30b_a3b/test_qwen3_30b_a3b.py --mode accuracy -s

# 2) 五档性能（TP=4，约 2 min）
python -m pytest models/qwen3_30b_a3b/test_qwen3_30b_a3b.py --mode perf -s

# 3) 精度 + 性能全量
python -m pytest models/qwen3_30b_a3b/test_qwen3_30b_a3b.py --mode all -s
```

用例都带 `L1 + ascend_a2` 标记，接入 CI 时按原有约定加 `-m "L1 and ascend_a2"`；
`--mode` 与 `-m` 可以叠加。缺少权重或卡数不足时用例会 **skip** 而不是 fail
（`QWEN3_30B_A3B_MODEL_DIR` 未设置、`ASCEND_RT_VISIBLE_DEVICES` 少于
`QWEN3_30B_A3B_TP` 都算）。

期望结果：

- `test_end_to_end_logits_match_reference`：cosine 0.999642（四个采样位置
  0.999642 / 0.999816 / 0.999970 / 0.999141）、全量 top-1 一致率 0.9648、
  margin ≥ 1.0 的 146 个位置一致率 1.0000、3 轮 top-1 完全一致
  （阈值 cosine ≥ 0.99、全量 ≥ 0.95、decisive = 1.0、轮间必须逐位一致）
- `test_prefill_decode_latency[128..2048]`：prefill < 2000 ms、decode < 200 ms/tok
- `test_graph_capture_completes`：五档 `prefill_capture_s` 全部 > 0（证明真的捕获了
  图，没有静默回退到 eager）

时延上限可用环境变量按机器放宽：

```bash
export QWEN3_30B_A3B_PREFILL_CAP_MS=3000
export QWEN3_30B_A3B_DECODE_CAP_MS=300
```

## 4. 优化内容

实现分三层，`--graph` 下三层全部生效：

```text
boost_qwen3_30B_A3B()
├── TP 并行：attention 列/行并行 + MoE 专家内切分 + 行并行 all-reduce（model.py）
├── 算子融合：RMSNorm / add+RMSNorm / QKV-RMSNorm-RoPE-Cache / SWIGLU /
│             gating_top_k / 整条 MoE 链（model.py + layers/moe.py；
│             QKV-RMSNorm-RoPE 在图路径的 graph_decode.py 里）
└── 图捕获：5 档 prefill 图 + 1-token decode 图（graph_decode.py）
```

### 4.1 张量并行（TP）

`boost_qwen3_30B_A3B(..., mode="tp")` 原地切分，不需要离线转换权重，每卡拿到等分
的权重后直接上卡：

| 模块 | 切分方式 | 每卡形状（TP=4） |
|------|----------|------------------|
| `q_proj` / `k_proj` / `v_proj` | 列并行（按 head） | q: 8 head，k/v: 1 head（head_dim 128） |
| `o_proj` | 行并行 | `[2048, 8×128]` |
| MoE 专家 | **专家内**切分：128 个专家每卡都保留，切 FFN 的 intermediate 维 | gate_up `[128, 2×192, 2048]`，down `[128, 2048, 192]` |
| router（`gate`） | 复制，各卡冗余计算 | `[128, 2048]` |
| `embed_tokens` / `lm_head` | 默认复制；`LB_TP_VOCAB_PARALLEL=1` 时按词表切 | — |

- attention 要求 `num_attention_heads(32)` 和 `num_key_value_heads(4)` 都能被 TP 整除，
  所以 **TP ∈ {1, 2, 4}**。
- MoE 走专家内 TP 而不是专家并行（EP）：token 不需要跨卡调度、没有 all-to-all，
  路由在每个 rank 上冗余算一遍，块尾只做一次 `[T, hidden]` 的 all-reduce；代价是
  每张卡都要读全部 128 个专家的分片权重。对本模型最实际的收益是显存：
  bf16 权重共约 61 GB，切分后每卡约 30.6 GB @ TP2、15.3 GB @ TP4。
- 行并行的 all-reduce：默认用 `npu_mm_all_reduce_base` 把 matmul 与 all-reduce
  合成一个算子；`--graph` 下推理脚本置 `LB_TP_SEPARATE_MM_AR=1`，改用
  `F.linear + dist.all_reduce`。
- `LB_TP_VOCAB_PARALLEL=1` 额外按词表切 `embed_tokens` / `lm_head`（151936 可被
  TP 整除），省约 0.6 GB/卡，代价是每步多两次集合通信。

### 4.2 算子融合

| 原始计算 | 融合后 | 覆盖范围 |
|----------|--------|----------|
| HF `RMSNorm`（统计 + 缩放） | `torch_npu.npu_rms_norm` | 每层 input / q_norm / k_norm，加 final norm，共 145 处（post-attention 的 48 处并进下一行） |
| attention 输出残差 add + post-attention RMSNorm | `torch_npu.npu_add_rms_norm`（一次返回归一化结果与残差） | 48 个 decoder 层 |
| Q/K/V 三个 GEMM 之后的 q_norm + k_norm + RoPE + KV cache 写入 | `aclnnQkvRmsNormRopeCache`（C++ 封装 `lite_boost.ops.qkv_rms_norm_rope_cache`，CANN 9.0） | 仅图路径：prefill 每层一次，decode 图内每层一次 |
| prefill attention | `npu_fused_infer_attention_score`（FIA，BSND，档位静态 mask） | 图路径；非图路径用 `npu_fusion_attention`（sparse_mode=3） |
| decode attention | `npu_incre_flash_attention`（IFA，BSND，按当前长度屏蔽） | 图路径（`LB_IFA_MASK=2`） |
| router 的 softmax + top-8 + 归一化 | `npu_moe_gating_top_k`（renorm 融合进核） | 48 层 |
| MoE 专家（128 个） | `npu_moe_init_routing_v2` → `npu_grouped_matmul` → `npu_swiglu` → `npu_grouped_matmul` → `npu_moe_token_unpermute` | 48 层，重排/分组 GEMM/激活/回填全部单算子 |
| `gate_proj` + `up_proj` 两个 GEMM | 权重拼接成一个 GEMM + `npu_swiglu` | 稠密 FFN（本 checkpoint 48 层全是 MoE，`mlp_only_layers` 为空，不会触发；为同族配置保留） |

MoE 链的细节：

- `npu_moe_init_routing_v2` 按 top-8 路由把 token 重排成专家连续布局，一次给出
  `permuted / argsort / 每专家 token 数`；
- 两次 `npu_grouped_matmul` 分别算 128 个专家的 gate_up 和 down，group_list 用
  计数模式（`group_list_type=1`），避免额外 cumsum；
- 中间激活是 `npu_swiglu`（silu(gate)·up），最后 `npu_moe_token_unpermute` 按路由
  权重把结果回填到原 token；
- 单 token 的 decode 在图路径下默认走 slot-major bmm（top-8 展开成 8 个 bmm，
  形状静态、好捕获）；普通 boost 路径默认就走 grouped matmul，图路径下置
  `LB_MOE_DECODE_GROUPED=1` 也可以切过去；
- `LB_MOE_PACKED_LAYOUT=1` 时把专家权重从 `[E, out, in]` 重排成 `[E, in, out]`，
  省掉每次调用前的两次 transpose（默认关）；
- CANN 算子缺失或 dtype 不满足时自动回退 `torch.sort` 路由或逐专家 bmm 的路径，
  不会静默算错。

### 4.3 图模式（NPUGraph）

- **prefill**：128/256/512/1024/2048 五档各捕获一张整图，缓存于 `_PREFILL_GRAPHS`。
  `input_ids` 用静态缓冲区，真实 token 拷入后 pad 到档位；图内完成 48 层
  （含融合 QKV-RoPE、MoE）与 KV cache 写入，lm_head 只算最后一个 token。
- **decode**：prefill 结束后捕获一张 1-token 图，内容为 1-token 前向 + KV 追加 +
  分 8 块计算的 lm_head + argmax；每步只把新的 token id / position 拷进静态输入，
  host 侧不做任何计算。捕获前先 warmup 3 步，避免把首次编译带进图。
- **KV cache 用 BSND 布局**（`LB_CACHE_BSND=1` 默认），prefill/decode 的 attention
  都不需要 BNSD↔BSND 转置。
- `--session` 额外支持公共前缀：共享前缀只 prefill 一次并缓存 KV，suffix 用独立的
  档位图，decode 图跨请求复用。

## 5. 性能数据（4 × Ascend A2 280T，TP=4，bf16 + 图回放）

| prefill tokens | prefill (ms) | decode (ms/token) | decode (token/s) |
|---------------:|-------------:|------------------:|-----------------:|
| 128  | 50.04  | 14.14 | 70.72 |
| 256  | 55.67  | 14.23 | 70.27 |
| 512  | 64.99  | 14.12 | 70.82 |
| 1024 | 87.48  | 14.15 | 70.67 |
| 2048 | 129.57 | 14.19 | 70.47 |

decode 时延与历史上下文长度基本无关；prefill 时延随档位近似线性增长。

## 6. 环境变量

推理脚本在 `--graph` 下会设置 `LB_PREFILL_GRAPH=1` / `LB_IFA_MASK=2` /
`LB_TP_SEPARATE_MM_AR=1`；其余开关可按需覆盖：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LB_MOE_FUSED` | `2` | MoE 融合路径，`2` = `npu_grouped_matmul` 的 CANN 全链路 |
| `LB_MOE_DECODE_GROUPED` | 普通路径 `1` / 图路径 `0` | 单 token decode 是否也走 grouped matmul；图路径默认走 slot-major bmm |
| `LB_MOE_PACKED_LAYOUT` | `0` | 置 `1` 时把专家权重重排为 `[E, in, out]`，省掉每步两次 transpose |
| `LB_ADD_RMSNORM` | `1` | 残差 + RMSNorm 融合为 `npu_add_rms_norm` |
| `LB_PREFILL_QKVROPE` | `1` | prefill 使用 `qkv_rms_norm_rope_cache` 融合算子 |
| `LB_DECODE_QKVROPE` | `0` | decode 使用融合 QKV-RoPE（`--graph` 下由图路径自动开启） |
| `LB_TP_VOCAB_PARALLEL` | `0` | 切分 embed_tokens / lm_head（词表 151936） |
| `LB_TP_SEPARATE_MM_AR` | — | 图模式下置 `1`，行并行 all-reduce 走独立通信算子 |
| `LB_PREFILL_GRAPH` / `LB_IFA_MASK` | — | 图捕获由推理脚本设置（`1` / `2`） |
| `LB_DECODE_ARGMAX_IN_GRAPH` | `1` | argmax 放进 decode 图内 |
| `HCCL_DETERMINISTIC` | `false` | 置 `true` 固定集合通信归约顺序；精度用例自动开启，bf16 TP 结果才可复现 |
| `QWEN3_30B_A3B_MODEL_DIR` | — | 测试用权重目录 |
| `QWEN3_30B_A3B_TP` | `2` | 测试用 TP 并行度 |
| `QWEN3_30B_A3B_PREFILL_CAP_MS` / `QWEN3_30B_A3B_DECODE_CAP_MS` | `2000` / `200` | 性能用例上限 |

## 7. 常见问题

**`aclnnQkvRmsNormRopeCache` / `qkv_rms_norm_rope_cache` 找不到**
CANN 版本低于 9.0.0，或没有 `source set_env.sh`。融合 QKV-RoPE 通过
`libopapi.so` 在运行时解析符号，构建和运行都要用 9.0.0 的环境。

**TP=2 时第一个 all-reduce OOM**
bf16 权重每卡 30.6 GB，32 GB 卡放得下权重但没有激活余量。改用 TP=4（每卡
14.2 GB），或换 ≥ 33 GB/卡的硬件。

**`please set the atten_mask_shape to [2048,2048]`**
CANN 9.0 的压缩 sparse-mode attention 要求 mask 固定为 2048×2048；用例里就是这么
构造的，自己写调用时不要传 `seq × seq` 的小 mask。

**测试全部 skip**
`QWEN3_30B_A3B_MODEL_DIR` 没设或目录不存在，或者 `ASCEND_RT_VISIBLE_DEVICES`
的卡数小于 `QWEN3_30B_A3B_TP`。

**想对比 eager 与图模式**
去掉 `--graph` 走 eager，但注意两点：eager 不会把输入 pad 到档位，2048 档在
32 GB 卡上会 OOM；且同一进程里连续调用 `graph_generate` 会因 cache 未复位报
`index_copy_ ... Self and source must have same slice shapes`。做 A/B 时请一个
进程只跑一个档位、只调一次，并先预热。
