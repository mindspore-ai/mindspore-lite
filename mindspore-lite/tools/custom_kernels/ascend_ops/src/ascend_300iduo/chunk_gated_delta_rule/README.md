# ChunkGatedDeltaRule

## 1 功能说明

`ChunkGatedDeltaRule`（CGDR）是 Gated Delta Rule（门控 Delta 规则）线性注意力/线性 RNN 前向算子，面向昇腾 Atlas 300I Duo 推理场景。

### 1.1 算子语义

给定查询 `Q`、键 `K`、值 `V`、门控 `β`、初始递归状态 `S₀`、可选 log 衰减门控 `g` 以及缩放因子 `scale`，算子按固定 `chunkSize=64` 将序列切分为若干 chunk，在每个 chunk 内完成下述递推并输出每个 token 的注意力值 `out` 与更新后的最终状态 `final_state`：

```text
# 1. 系数预处理
k_beta      = K * beta                                  # [T, Hqk, Dk]
g_cumsum    = cumsum(g, dim=t)                           # [T, Hv]   (g 可选)
exp_g       = exp(g_cumsum)
decay[i][j] = exp(g_cumsum[i] - g_cumsum[j])             # 因果下三角

# 2. chunk 内递归注意力矩阵（下三角，对角为 1）
attn        = -(k_beta @ K^T) * decay                    # chunk 内 intra-attn，下三角
attn        = (I - attn)^{-1}                            # 递归三角求逆（逐 chunk）
attn_i      = (Q @ K^T) * decay                          # QK 注意力（下三角+对角）

# 3. chunk 内状态交互
k_cumdecay  = attn @ (k_beta * exp_g)                    # 整理 K，存回 chunkKFp32
v_beta      = V * beta                                   # [T, Hv, Dv]
v_new       = attn @ v_beta  -  k_cumdecay @ state       # chunk 内新值

# 4. 输出与状态更新
out         = (Q * exp_g) @ state + attn_i @ v_new        # 每个输出 token
state       = exp_g_last * state + (K * exp(g_last - g))^T @ v_new   # 跨 chunk 携带
```

其中 `state` 维度为 `[Dk, Dv]`（每头），在 chunk 之间以 **FP32** 形式保存在 workspace 中，仅在最后一个 chunk 转回 FP16 写入 `final_state`，以避免逐 chunk 量化累积误差使结果依赖 `chunkSize`。

### 1.2 实现要点

- **Cube + Vector 三级流水**：Atlas 300I Duo 的 Cube 单元仅支持 FP16 输入（无 `f32f32f32` mmad），但所有递推逻辑（state 交互、`exp(g)` 衰减、衰减掩码）天然是 FP32。本算子采用 **FP16 高/低半部分拆分 + FP32 L0C 累加** 的补偿策略，在 Cube 上完成稠密矩阵乘（`attn@K^T`、`Q@K^T`、`attn@v_beta`、`k_cumdecay@state`、`Q@state`、`attn_i@v_new`、`K^T@v_new`），将 `O(chunk³)` 的算术从标量流水移至 Cube，同时保留接近 FP32 的数值稳定性。
- **`tilingKey` 多桶特化**：按 `Dk` 落入的 64/80/96/128 Cube 桶位编译 4 份 kernel（`tilingKey=0/1/2/3`），尾部维度由 kernel 零填充到桶宽，使每个合法 `Dk` 都走融合快路径；超过 128 的形状回退到 `kSpecializedDk=96` 的通用实现。
- **递归三角求逆向量化**：64×64 的 `(I-A)⁻¹` 严格下三角求逆按 2×2 分块（32→16）分解，16×16 对角块在 Vector 上用 `Gather+Brcb+MulAddDst` 做外积更新，跨块稠密积由补偿 Cube 完成。
- **UB 缓冲复用**：所有 FP32 临时张量共享一个 `tmpBuff`，按存活区间手工排布偏移；当单头 `Dv` 需要拆多个 V-tile 时，注意力矩阵 `chunkScoresFp32` 与状态 `stateInFp32` 复用同一 UB 区（状态在 scores 末次使用后才加载）。
- **多核负载均衡**：一个 work item = (batch, value-head, V-tile) 三元组，按 `actual_seq_lengths` 真实长度做近似均匀的核间切分（`ComputeAvgload/IsCurrentBlock`），小头多 batch 输入也能打满所有 AIV 核。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| query | 必选 | float16 | `[T, Hqk, Dk]` | `[seq, qk_head, qk_dim]` | 查询张量，`T` 为所有 batch 拼接后的总序列长度，`Hqk` 为 Q/K 头数，`Dk` 为每头键维度。需与 key 同布局。 |
| key | 必选 | float16 | `[T, Hqk, Dk]` | `[seq, qk_head, qk_dim]` | 键张量，与 query 共享 `[T, Hqk, Dk]` 布局；chunk 内同时参与 `attn=-(k_beta@K^T)*decay` 与 `attn_i=(Q@K^T)*decay`。 |
| value | 必选 | float16 | `[T, Hv, Dv]` | `[seq, v_head, v_dim]` | 值张量，`Hv` 为 V 头数（支持 GQA，`Hv` 可小于 `Hqk`），`Dv` 为每头值维度。 |
| beta | 必选 | float16 | `[T, Hv]` | `[seq, v_head]` | 门控系数 β，每 token 每 V 头一个标量。用于生成 `k_beta=K*beta` 与 `v_beta=V*beta`；`Hv` 维与 value 的头维对齐。 |
| initial_state | 必选 | float16 | `[B, Hv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 初始递归状态 `S₀`。`B` 为 batch 数；内存按 `[B, Nv, Dv, Dk]` 排布，内部转置为 `[Dk, Dv]` tile 处理。`B` 由该输入推导（`TilingData.b`）。 |
| actual_seq_lengths | 必选 | int32 | `[B]` | `[batch]` | 每个 batch 的真实序列长度；`T = Σ actual_seq_lengths[b]`。kernel 据此跳过 padding、做核间切分。 |
| g | 可选 | float32 | `[T, Hv]` | `[seq, v_head]` | log 衰减门控 g（“gate”）。提供时计算 `g_cumsum=cumsum(g)` 与 `exp(g_cumsum)`，参与衰减掩码与状态更新；缺省时按全 0 处理（退化为无衰减的 Delta Rule）。注意：当前 def 中 `g` 声明为 `ge::DT_FLOAT`，与其它输入的 FP16 不同。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| out | 必选 | float16 | `[T, Hv, Dv]` | `[seq, v_head, v_dim]` | 每个 token 的注意力输出。由 `out=(Q*exp_g)@state + attn_i@v_new` 得到，`AutoContiguous` 保证连续存储。 |
| final_state | 必选 | float16 | `[B, Hv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 更新后的递归状态，形状与 `initial_state` 相同（由 `infershape` 直接镜像）。仅在最后一个 chunk 由 FP32 workspace 转回 FP16 写出。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| scale_value | 可选 | float | 1.0 | QK 注意力点的缩放因子（通常取 `1/√Dk`）。作用于 `Q@K^T` 与 `(Q*exp_g)@state` 两处。 |

### 2.4 Tiling 数据结构

host 侧 `ChunkGatedDeltaRuleTilingData` 与 device 侧镜像结构体（`op_host/chunk_gated_delta_rule_tiling.h` 与 `op_kernel/chunk_gated_delta_rule_tiling_data.h`），二者必须**字节一致**且位于全局命名空间（CANN tiling 宏要求），`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| vectorCoreNum | uint32_t | AIV 核数 |
| ubCalSize | uint32_t | UB 总容量（字节） |
| ubRestBytes | uint32_t | 扣除 `stateOutQueue` 后分配给 `tmpBuff` 的字节数 |
| t | uint32_t | 总序列长度 T |
| hqk | uint32_t | Q/K 头数 |
| dk | uint32_t | 键头维 Dk |
| hv | uint32_t | V 头数 |
| dv | uint32_t | 值头维 Dv |
| chunkSize | uint32_t | 分块大小，固定 64 |
| numChunks | uint32_t | T 上取整后的 chunk 数 |
| b | uint32_t | batch 数（来自 `initial_state`） |
| padSize | uint32_t | T 到 chunkSize 倍数的 padding 量 |
| hasGamma | uint32_t | 是否提供可选输入 `g`（1/0） |
| scaleValue | float | 缩放因子 |
| vStep | uint32_t | V 维 tile 宽（按 UB 容量反解，FP32 块对齐） |
| debug | uint32_t | 调试标志（当前固定 0；host 侧可由 `CGDR_DEBUG_LOG=1` 打印 tiling 日志到 stderr） |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | 不支持 Atlas 800I A2 等其它 SoC（Atlas 800I A2 有独立的 CGDR 实现）。 |
| 输入数据类型 | query/key/value/beta/initial_state = float16；actual_seq_lengths = int32；g（可选）= float32 | 所有 FP16 输入在内部转 FP32 递推；FP16 拆分补偿仅对原本来自 FP16 的 K/Q/V/state 精确无损（见 3.3）。 |
| 输出数据类型 | out / final_state = float16 | FP32 中间结果最终 cast 回 FP16。 |
| 输入格式 | `FORMAT_ND` | def 中 ND + `DynamicFormatFlag(true)` + `DynamicRankSupportFlag(true)`，支持动态 shape 与动态秩。 |
| chunkSize | 固定 64 | `kDefaultChunkSize=64`，与 Atlas 800I A2 端口对齐；不可配置。 |
| Dk（键头维） | ≤ 128 且需落入 64/80/96/128 桶位走快路径 | `SelectCubeDk` 将 Dk 上取整到 ≥ Dk 的最小桶宽（64/80/96/128）；超过 128 回退 `kSpecializedDk=96` 通用路径（性能下降）。 |
| Dv（值头维） | 受 UB 容量约束 | `SolveVStep` 按 FP32 块对齐（8 元素）反解最大可容纳 `vStep`；`Dv` 过大时自动拆多 V-tile，`vStep` 不得小于 16（`FP16_NUM_PER_BLOCK`），否则 tiling 失败返回 `GRAPH_FAILED`。 |
| 头数关系 | 支持 GQA：`Hv` 可小于 `Hqk`，`nvPerNk = max(NV_/NK_, 1)`（`NV_≥NK_` 时取 `NV_/NK_`，否则按 1 复用同一 Q/K 头） | Q/K 头与 V 头通过 `head_i / nvPerNk` 映射。 |
| batch | ≥ 1，由 `initial_state` 的第 0 维推导 | 多 batch 在核间切分；`actual_seq_lengths` 控制每 batch 真实长度，padding 跳过。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`；tiling 运行时按实际 shape 反解 `vStep`/`blockDim`/`tilingKey`。 |
| workspace | ≥ 32 MiB | `kWorkspaceBytes=32MB` 下限；实际取 `max(32MB, state_workspace + blockDim*raw_matmul_stage)`。`state_workspace = B*Hv*Dk*ceil(Dv,8)*4B` 用于跨 chunk FP32 state 暂存；`raw_matmul_stage` 为 Cube 暂存 A_hi/A_lo/B_hi 的 GM 暂存区（每核 2 slot × 64KB）。 |

### 3.2 不支持的场景

- **非 Atlas 300I Duo 硬件**：未在 Atlas 800I A2 等添加 `AICoreConfig`，部署到这些平台会找不到编译产物。
- **`Dk > 128` 的快路径**：`SelectCubeDk` 对 `Dk>128` 返回 0，kernel 走 `kSpecializedDk=96` 的通用回退路径（`tilingKey=2`），稠密积改为逐行 `Axpy`/标量 `DotFp32`，性能显著下降；不报错但非推荐。
- **`g` 输入类型非 float32**：def 固定 `g` 为 `ge::DT_FLOAT`；若上游以 FP16 传入 `g`，需在接入层先转换，否则类型校验失败。
- **极小 `Dv` 无法容纳最小 V-tile**：当 UB 不足以容纳 `vStep=16`（FP32 块对齐下限）所需的 `tmpBuff+outQueue` 时，`SolveVStep` 返回 `false`，tiling 直接 `GRAPH_FAILED`，算子无法编译。
- **FP32 Cube 路径**：Atlas 300I Duo Cube 的 mmad 不支持 `f32f32f32`（仅 FP16 输入）。因此 Atlas 800I A2 上“原生 FP32 Cube”的 CGDR 路径无法直接移植到 Atlas 300I Duo ；本实现以 FP16 拆分补偿替代。
- **`SetAtomicAdd` 跨核累加**：dav_m200（Atlas 300I Duo）上 `SetAtomicAdd` 不可用，故状态更新不依赖原子加，而是按 (batch, value-head, V-tile) 切分保证核间写互不重叠。
- **混合精度/任意 dtype**：仅支持上述固定类型组合，不支持 bfloat16 / int8 / FP32 输入输出。
- **训练/反向**：本算子仅前向（`KERNEL_TYPE_AIC_ONLY`），不含反向梯度。

### 3.3 数值约束与精度说明

- **FP16 拆分补偿的精确性**：`K`、`Q`、`V`、`state` 原始即为 FP16，`StageHalfWithResidual` 取 `hi=cast_fp16(x)`、`lo=x-hi`，二者 cast 回 FP16 再入 Cube；由于源是 FP16，`hi` 精确、`lo` 为残差，四个交叉项 `A_hi·B_hi + A_lo·B_hi + A_hi·B_lo + A_lo·B_lo` 在 FP32 L0C 累加，逼近原始 FP32 递推。
- **`exp(g)` 数值范围**：`g` 为 log 衰减，其前缀和真实 Qwen 激活下常低于 -100。算子刻意避免 `exp(g_i - g_j) = exp(g_i)·exp(-g_j)` 的因子分解（会溢出），改为先按行广播构造完整因果矩阵、清上三角、再统一 `Exp`，并用 host/Vector 原语保证稳定。
- **state 跨 chunk 精度**：chunk 间 state 以 FP32 保存在 workspace，仅末 chunk cast FP16，避免逐 chunk 量化误差使结果依赖 `chunkSize`。
- **`g_cumsum` 前缀和**：单 chunk（≤64）内用 UB 双缓冲 ping-pong 的 `Add(offset)` 并行前缀和替代逐元素标量链。

### 3.4 调试

- 设环境变量 `CGDR_DEBUG_LOG=1` 可在 host tiling 阶段向 stderr 打印一行 tiling 摘要（`T/chunkSize/numChunks/hqk/dk/hv/dv/scaleValue/ubSize/vStep/blockDim/...`），便于定位核切分与 UB 规划。默认关闭。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Cube/Vector 流水、`Mmad`/`LoadData`/`Nd2Nz`/`TransDataTo5HD`/`WholeReduceSum` 等原语），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
