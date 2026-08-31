# RecurrentGatedDeltaRule

## 1 功能说明

`RecurrentGatedDeltaRule`（RGDR）是 Gated Delta Rule（门控 Delta 规则）的**逐 token 递归前向算子**，面向 MindSpore Lite 昇腾 Atlas 300I Duo 端侧推理的 **decode / 多 token 预测（MTP）** 场景。它与同目录的 [chunk_gated_delta_rule](../chunk_gated_delta_rule/README.md) 互补：后者按 `chunkSize=64` 分块做 chunkwise 前向（适合 prefill 长序列），本算子逐 token 递推（适合 decode 短序列，单 batch 每次至多 8 个 token）。

### 1.1 算子语义

给定查询 `Q`、键 `K`、值 `V`、门控 `β`、初始状态 `S`、序列长度索引、可选衰减门 `g`（代码内称 `gama`）、可选键门 `gk`（代码内称 `gamaK`）以及缩放因子 `scale`，对序列中每个 token 逐步递推：

```text
# 每个 token t（对当前 V-tile 内的每个 v 切片）：
gama   = exp(g[t])                  # 可选，默认 1（状态衰减门）
gamaK  = exp(gk[t])                 # 可选（标量逐头广播，或逐头向量），默认 1（键衰减门）

state  = state * gama * gamaK       # 状态衰减（element-wise，state 形状 [Dv, Dk]）
memory = <state[v], K[t]>           # K 与 state 对应行的点积，长度 Dk
delta  = (V[t][v] - memory) * beta[t]
state[v] = state[v] + K[t] * delta  # 秩 1 状态更新（outer 更新）
out[t][v] = <state[v], Q[t]> * scale  # 输出注意力值
```

其中 `state` 内部以 `[Dk, vStep]` tile 排布（`stateInUb[v*alignK_]` 为一行，长 `alignK_=ceil(Dk,16)`），公共输出 state 形状为 `[B, Nv, Dv, Dk]`（`Dv` 为外维、`Dk` 连续）。`<·,·>` 为 FP32 点积（`Mul + WholeReduceSum` + 标量累加），秩 1 更新 `K*delta` 用 `Muls + Add`。

### 1.2 实现要点

- **逐 token 标量递推**：状态更新天然串行（token t 依赖 t-1 的 state），无法跨 token 并行；算子把 `O(Dk)` 点积与秩 1 更新放到 Vector 流水（`DotFp32`/`Muls`/`Add`），用 `WholeReduceSum` 把 64 元素折叠到一个标量，再标量累加尾部。Atlas 300I Duo Cube 仅 FP16、本算子递推尺度小，未走 Cube 路径。
- **MTP 多 token 一次处理**：`MAX_MTP=8`，单 batch 一次最多送 8 个 token；`CopyInQKV` 一次把 `[seqLen, Dk/Dv]` 整段 DMA 进 UB（`DataCopyPadCustom` 处理非 16 对齐尾部），再 cast FP16→FP32。
- **V 维 tiling**：`Dv` 过大时按 `vStep`（FP16 块对齐，由 UB 反解）拆多 V-tile，每个 V-tile 独立加载 state 行、完成全序列递推后写回；`tiling` 在 `{(1,1),(1,2),(2,2)}`（stateOut/attnOut 双缓冲深度）组合中按「`repeatTime=ceil(Dv/vStep)` 最小、缓冲深度最大」择优（`SelectBufferProfile`/`IsBetterProfile`）。
- **state 预取流水**：`PrefetchState` 在处理当前 V-tile 时预取下一 V-tile 的 state，`LoadPrefetchedState` 做 FP16→FP32 cast，隐藏 MTE2 延迟；`QueueAttnOutput/QueueStateOutput` 在双缓冲模式下延迟一拍写出，提升 MTE3 吞吐。
- **推测解码 / 状态槽索引**：`ssm_state_indices[t]` 把序列内每个 token 映射到一个状态槽地址，配合 `num_accepted_tokens[b]` 选择「接受 token」对应的状态索引（`stateTokenIdx = seq0 + acceptedTokenNum - 1`）做 state 读写定位，支持 spec-decode 接受/回滚场景。
- **核间切分**：work item = (batch, value-head)，`blockDim = min(B*Nv, coreNum)`，每核按模 `blockDim` 取自己负责的 (batch, head) 对；`coreNum` 上限 `8`（Atlas 300I Duo 可调度 AICore 上限）。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| query | 必选 | float16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | 查询张量；`T` 为所有 batch 拼接后的总 token 数。内部 cast FP32 并乘 `scale`。 |
| key | 必选 | float16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | 键张量，与 query 共享 `[T, Nk, Dk]` 布局；参与 `memory=K·state` 与秩 1 更新 `state+=K*delta`。 |
| value | 必选 | float16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 值张量；`Nv` 为 V 头数（支持 GQA，需 `Nv≥Nk` 且为 `Nk` 倍数），`Dv` 为每头值维。 |
| beta | 必选 | float16 | `[T, Nv]` | `[seq, v_head]` | 门控系数 β，每 token 每 V 头一个标量。 |
| state | 必选 | float16 | `[S, Nv, Dv, Dk]` | `[state_slot, v_head, v_dim, qk_dim]` | 初始/输入递归状态。第 0 维 `S` 为状态槽总数（由 `ssm_state_indices` 索引）；内部按 `[Dv, Dk]` tile 处理，`Dk` 为连续内维。输出 `state` 形状与之相同。 |
| actual_seq_lengths | 必选 | int32 或 int64 | `[B]` 或 `[B+1]` | `[batch]` 或 `[cumsum]` | 每个 batch 的序列长度。tiling 记录 `cuSeqlensIsInt64` 区分 dtype；`cuSeqlensIsPrefix` 标记 `[B+1]` 前缀和样式（当前 tiling 固定写 0，即按 `[B]` 实际长度样式处理，`seq0` 由前序 batch 累加得到）。 |
| ssm_state_indices | 必选 | int32 或 int64 | `[T]` | `[seq]` | 每个 token 对应的状态槽索引，用于推测解码的状态读写定位。tiling 记录 `ssmStateIndicesIsInt64`。 |
| g（gama） | 可选 | float32 | `[T, Nv]` | `[seq, v_head]` | 状态衰减门 g（log 域）；提供时取 `gama=exp(g[t])` 对 state 做 element-wise 衰减。默认 1（不衰减）。 |
| gk（gamaK） | 可选 | float32 | `[T, Nv]`（标量）或 `[T, Nv, Dk]`（向量） | `[seq, v_head,(, qk_dim)]` | 键衰减门 gk。标量模式逐头广播到 `Dk`（`gamaKScalar=1`），向量模式逐头逐维（`gamaKScalar=0`）；取 `exp(gk[t])` 作用于 state。**注意**：当 `g` 与 `gk`（标量）同时提供时，tiling 会将 `hasGamaK` 置 0（即忽略 `gk`，仅用 `g`）。 |
| num_accepted_tokens | 可选 | int32 | `[B]` | `[batch]` | 推测解码中每个 batch 接受的 token 数；提供时取 `stateTokenIdx = seq0 + accepted - 1` 作为初始 state 槽索引，否则用 `seq0`。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| out | 必选 | float16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 每个 token 的注意力输出 `out=Q·state`（FP32 中间结果 cast 回 FP16），`AutoContiguous`。 |
| state | 必选 | float16 | `[S, Nv, Dv, Dk]` | `[state_slot, v_head, v_dim, qk_dim]` | 更新后的递归状态，形状与输入 `state` 相同（`infershape` 直接镜像输入第 4 维 4D 形状）。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| scale_value | 可选 | float | 1.0 | Q 的缩放因子，作用于 `Q*scale`（`out=Q·state` 的 Q 分量）。 |

### 2.4 Tiling 数据结构

host 侧 `RecurrentGatedDeltaRuleTilingData` 与 device 侧镜像结构体（`op_host/recurrent_gated_delta_rule_tiling.h` 与 `op_kernel/recurrent_gated_delta_rule_tiling_data.h`），`#pragma pack(8)` + `alignas(8)` 字节一致。

| 字段 | 类型 | 含义 |
|------|------|------|
| vectorCoreNum | uint32_t | AICore 核数（上限 8） |
| ubCalSize | uint32_t | UB 总容量（字节） |
| ubRestBytes | uint32_t | 扣除各 TQue 后分配给 `tmpBuff` 的字节数 |
| t | uint32_t | 总 token 数 T |
| nk | uint32_t | Q/K 头数 Nk |
| dk | uint32_t | 键头维 Dk |
| nv | uint32_t | V 头数 Nv |
| dv | uint32_t | 值头维 Dv |
| sBlockNum | uint32_t | state 第 0 维（状态槽数 S） |
| b | uint32_t | batch 数 B（来自 `actual_seq_lengths`，0 视作 1） |
| vStep | uint32_t | V 维 tile 宽（FP16 块对齐，UB 反解） |
| stateOutBufferNum | uint32_t | state 输出 TQue 双缓冲深度（1 或 2） |
| attnOutBufferNum | uint32_t | attn 输出 TQue 双缓冲深度（1 或 2） |
| scale | float | 缩放因子 |
| hasGama | uint32_t | 是否提供 `g`（1/0） |
| hasGamaK | uint32_t | 是否提供 `gk`（1/0；与 `g` 同时标量提供时被置 0） |
| hasAcceptedTokens | uint32_t | 是否提供 `num_accepted_tokens`（1/0） |
| gamaKScalar | uint32_t | `gk` 是否为逐头标量（1）或逐头向量（0） |
| cuSeqlensIsPrefix | uint32_t | `actual_seq_lengths` 是否为 `[B+1]` 前缀和样式（当前固定 0） |
| cuSeqlensIsInt64 | uint32_t | `actual_seq_lengths` 是否 int64（1）或 int32（0） |
| ssmStateIndicesIsInt64 | uint32_t | `ssm_state_indices` 是否 int64（1）或 int32（0） |
| reserved | uint32_t | 保留字段 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | 不支持 Atlas 800I A2 |
| 输入数据类型 | query/key/value/beta/state = float16；actual_seq_lengths/ssm_state_indices = int32 或 int64；g/gk = float32；num_accepted_tokens = int32 | FP16 输入内部 cast FP32 递推；索引输入按 tiling 标志分别用 int32/int64 读取。 |
| 输出数据类型 | out / state = float16 | 固定 FP16 输出。 |
| 输入格式 | `FORMAT_ND` | `DynamicFormatFlag(true)` + `DynamicRankSupportFlag(true)` + `DynamicShapeSupportFlag(true)`，支持动态 shape。 |
| 单 batch 序列长度 | `1 ≤ seqLen ≤ 8`（`MAX_MTP=8`） | kernel 中 `if (seqLen > MAX_MTP) return;`，超过 8 直接返回（不报错但不产出）；面向 decode/MTP，**非 prefill**。 |
| Dk（键头维） | ≥ 1，内部按 16 上取整 `alignK_` | 非 16 对齐由 `DataCopyPadCustom` 补零。 |
| Dv（值头维） | 受 UB 容量约束 | `vStep` 按 FP16 块对齐反解；`vStep < 16` 时 tiling 返回 `GRAPH_FAILED`。 |
| 头数关系 | GQA：`Nv ≥ Nk` 且 `Nv` 为 `Nk` 倍数 | `qkOffset` 用 `head_i / (Nv/Nk)` 映射 V 头到 Q/K 头；`Nv < Nk` 时除零/下溢，不支持。 |
| batch / 状态槽 | B ≥ 1（0 视作 1）；state 第 0 维 S 由 `ssm_state_indices` 取值范围决定 | 多 (batch, head) 在核间按模切分。 |
| 核数 | 最多 8 | `coreNum` 超 8 截断；`blockDim = min(B*Nv, coreNum)`。 |
| workspace | 16 MiB（`SYSTEM_WORKSPACE_BYTES`） | 固定系统 workspace。 |
| tilingKey | 固定 0 | 单一 kernel 路径（`TILING_KEY_IS(0)`），dtype `half/half`。 |

### 3.2 不支持的场景

- **非 Atlas 300I Duo 硬件**：部署到这些平台找不到编译产物。
- **长序列 / prefill**：单 batch `seqLen > 8` 时 kernel 直接 `return`（不产出有效结果）；长序列请用 [chunk_gated_delta_rule](../chunk_gated_delta_rule/README.md)。
- **`Nv < Nk` 的 GQA**：`head_i / (Nv/Nk)` 在 `Nv<Nk` 时下溢，不支持；仅支持 `Nv≥Nk`。
- **`actual_seq_lengths` 前缀和样式 `[B+1]`**：tiling 当前固定 `cuSeqlensIsPrefix=0`，即只按 `[B]` 实际长度样式处理（`seq0` 由前序 batch 累加）；前缀和样式不会被启用。
- **极小 `Dv` 装不下最小 V-tile**：UB 不足以容纳 `vStep=16` 所需缓冲时 `SelectBufferProfile` 失败，tiling `GRAPH_FAILED`。
- **训练/反向**：仅前向（`KERNEL_TYPE_AICORE`），无反向梯度。
- **混合精度/任意 dtype**：仅上述固定类型组合，不支持 bfloat16 / int8 / FP32 输入输出。
- **`g` 与 `gk`（标量）同时提供**：tiling 会显式将 `hasGamaK=0` 忽略 `gk`，仅用 `g`；若需 `gk` 独立生效，应只提供 `gk`（向量或标量）而不提供 `g`，或以向量形式提供 `gk`。

### 3.3 数值约束与精度说明

- **递推全 FP32**：state、Q、K、V 均 cast FP32 后递推，点积用 `Mul + WholeReduceSum`（64 元素折叠为标量）+ 标量累加，秩 1 更新用 `Muls + Add`；最终 out/state cast 回 FP16。
- **逐 token 串行**：状态更新强串行依赖（token t 需 t-1 的 state），无法跨 token 并行，性能受 `Dk` 与序列长度线性约束。
- **`exp(g)/exp(gk)` 数值范围**：`g/gk` 为 log 域，`Exp` 在 Vector 上执行；状态衰减 `state *= gama*gamaK`，当 `g` 很负时 state 快速衰减趋零（符合预期）。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Vector 流水、`WholeReduceSum`/`DataCopyPad`/`Muls`/`Add` 原语、TQue 双缓冲），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
