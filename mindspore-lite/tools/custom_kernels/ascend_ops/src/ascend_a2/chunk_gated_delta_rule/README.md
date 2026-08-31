# ChunkGatedDeltaRule

## 1 功能说明

`ChunkGatedDeltaRule`（CGDR）是 Gated Delta Rule（门控 Delta 规则）线性注意力/线性 RNN 前向算子，面向 MindSpore Lite 昇腾 Atlas 800I A2云侧推理场景。算子从 [CANN ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule) 移植，接口（TND layout、输入输出名/顺序/dtype）与开源 ops-transformer 完全一致。

### 1.1 算子语义

给定查询 `Q`、键 `K`、值 `V`、门控 `β`、初始递归状态 `S₀`、可选 log 衰减门控 `g` 以及缩放因子 `scale`，算子按固定 `chunkSize=64` 将序列切分为若干 chunk，在每个 chunk 内完成下述递推，输出每个 token 的注意力值 `out` 与更新后的最终状态 `final_state`：

```text
# 1. 系数预处理
k_beta      = K * beta                                  # [T, Nk, Dk]
g_cumsum    = cumsum(g, dim=t)                           # [T, Nv]   (g 可选，缺省为 0)
exp_g       = exp(g_cumsum)
decay[i][j] = exp(g_cumsum[i] - g_cumsum[j])             # 因果下三角

# 2. chunk 内递归注意力矩阵（严格下三角，对角为 0）
attn        = -(k_beta @ K^T) * decay                    # chunk 内 intra-attn
attn        = (I - attn)^{-1}                            # 递归三角求逆（逐 chunk）
attn_i      = (Q @ K^T) * decay                          # QK 注意力（下三角 + 对角）

# 3. chunk 内状态交互
k_cumdecay  = -1 * k_beta * exp_g                        # = -K*beta*exp(g_cum)
v_beta      = V * beta                                    # [T, Nv, Dv]
v_new       = attn @ v_beta  -  k_cumdecay @ state        # chunk 内新值
kg          = K * exp(g_last - g_cum)                     # 跨 chunk 衰减后的 K

# 4. 输出与状态更新
out         = (Q * exp_g) @ state + attn_i @ v_new        # 每个输出 token
state       = kg^T @ v_new + exp(g_last) * state          # 跨 chunk 携带
```

其中 `state` 维度为 `[Dv, Dk]`（每头），在 chunk 之间以输入 dtype（BF16 或 FP16）携带保存于 `final_state` 输出张量中（Atlas 800I A2 上 FP32 state 路径被 tiling 拒绝，仅 ascend950 支持）。

### 1.2 实现要点

- **arch22 三阶段流水（Cube + Vector，MIX_AIC_1_2）**：kernel 为 `KERNEL_TYPE_MIX_AIC_1_2` 构建（1 AIC : 2 AIV），三阶段在 `op_kernel/arch22/` 下分别由 `chunk_gated_delta_rule_stage1.h` / `stage2.h` / `stage3.h` 实现，通过 `CrossCoreSetFlag/WaitFlag` 在 Cube 与 Vector 间同步：
  - **Stage 1（chunk 内预处理 + 递归三角求逆）**：AIV 负责 Q/K 连续化（从 TND layout Gather 成连续块）、`g_cumsum/exp(g_cum)`、衰减掩码 `decay=exp(g_cum[i]-g_cum[j])` 构造、`v_beta=V*beta`、`q_prime=Q*scale*exp_g`、`k_cumdecay=-K*beta*exp_g`、`kg=K*exp(g_last-g_cum)`，以及 64×64 的 `(I-A)⁻¹` 递归三角求逆（对角 32×32 块用 `Gather+Broadcast+MulAddDst` 外积更新，跨块稠密积由 AIC 完成）；AIC 负责 `K@K^T`、`Q@K^T`、`attn@k_cumdecay`、`attn@v_beta` 四个 Cube 矩阵乘。产物：`qkt`、`kCumDecay`、`vInner`、`qPrime`、`kg`、`gCumExp`。
  - **Stage 2（chunk 间状态递归 + `out` 的 Q@state 项）**：按 chunk 顺序串行推进，AIV 把上一 chunk 的 state 用 `exp(g_last)` 缩放后写回，AIC 在 Cube 上完成 `v_inner += k_cumdecay @ state^T`（`v_new` 的减项）、`out += q_prime @ state^T`（`out` 的第一项 `(Q*exp_g)@state`）、`state_new = v_inner^T @ kg`（状态更新）；跨核 flag 保证“AIC 读旧 state → AIV 写缩放 state → AIC 写新 state”的有序性。
  - **Stage 3（`out` 的 `attn_i@v_new` 项 + 因果 mask）**：AIV 用下三角含对角 mask（`stageThreeMask`）对 `qkt` 做 `scale * decay * mask` 得到 `masked_qkt`，AIC 在 Cube 上完成 `masked_qkt @ vInner` 并累加到 `out`，补上 `out` 公式的第二项 `attn_i@v_new`。
- **arch22 原生 BF16/FP16 Cube**：Atlas 800I A2 的 Cube 单元原生支持 BF16 与 FP16 输入的 `MMAD`（`MatmulImpl`，ND 格式），所有递推中的稠密矩阵乘（`K@K^T`、`Q@K^T`、`attn@k_cumdecay`、`attn@v_beta`、`q_prime@state`、`k_cumdecay@state`、`v_inner^T@kg`、`masked_qkt@vInner`）均直接在 Cube 上以输入 dtype 完成，无需 Atlas 300I Duo 版本的 FP16 高低半拆分补偿。
- **BF16/FP16 双 dtype 编译（`DataTypeList`）**：def 中 7 个可变张量（query/key/value/beta/initial_state/out/final_state）用 `DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})` + `FormatList({ge::FORMAT_ND, ge::FORMAT_ND})` 声明，配合 `DynamicFormatFlag(true)`，opbuild 会为每种 dtype 组合生成一份 kernel binary（`2^7 = 128` 份）。但 tiling 在运行时拒绝混合 dtype（`CheckLowDtype` 要求 7 张量同 BF16 或同 FP16），因此实际只会选中全 BF16 与全 FP16 两份 binary。device 入口按 `tilingData.isFp16` 分派 `CGDR<half, float>` 或 `CGDR<bfloat16_t, float>`，二者 `highType=float`。
- **`tilingKey` 分桶**：BF16 与 FP16 **共用 tilingKey=0**（`TILING_KEY_CGDR_BF16_STATE`），因为二者使用同一份 tiling 结构、仅 `isFp16` 标志不同；为 FP16 单独引入 tilingKey（如 2）不会在 CANN autogen 的 `tiling_struct_expr_map` 中注册，会在 convert 阶段抛 `KeyError`。FP32 state 走 tilingKey=1（`TILING_KEY_CGDR_FP32_STATE`），但在 Atlas 800I A2 上被 tiling 拒绝（不支持 FP32 state），故该分支不可达。
- **自适应 Cube tile base**：matmul tiling 在 `Dk` 与 `Dv` 均 ≤ 64 时用 64 base（小模型优化，避免 128 base 对 64³ 级 matmul 浪费 L0 带宽），否则用 128 base；base 与 dtype 无关（BF16/FP16 同 tiling）。
- **与 Atlas 300I Duo 版本的核心差异**：
  - Atlas 800I A2 原生支持 BF16 与 FP16 Cube，稠密矩阵乘直接走 `MatmulImpl`，**无需** Atlas 300I Duo 的 FP16 高/低半拆分 + FP32 L0C 累加补偿。
  - Atlas 800I A2 通过 `DataTypeList` 原生支持 BF16 输入输出；Atlas 300I Duo 仅 FP16。
  - Atlas 800I A2 采用 arch22 三阶段流水（Stage1 预处理+求逆、Stage2 状态递归+`Q@state`、Stage3 `attn_i@v_new`）+ `MIX_AIC_1_2` Cube/Vector 协同；Atlas 300I Duo 为基于 AIV 的补偿 Cube 方案。
  - Atlas 800I A2 上 FP32 state **不支持**（tiling 显式拒绝，仅 ascend950 有），state 跨 chunk 以输入 dtype（BF16/FP16）携带于 `final_state` 输出张量；Atlas 300I Duo 则在 workspace 中以 FP32 保存跨 chunk state、仅末 chunk cast 回 FP16。
  - Atlas 800I A2 的 tilingKey 仅按 dtype 类别（0=低精度 state / 1=FP32 state）分桶；Atlas 300I Duo 按 `Dk` 落入 64/80/96/128 Cube 桶位分 4 份 kernel。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| query | 必选 | BF16 或 FP16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | 查询张量，`T` 为所有 batch 拼接后的总序列长度（`T = Σ actual_seq_lengths[b]`），`Nk` 为 Q/K 头数，`Dk` 为每头键维度。与 key 同布局。 |
| key | 必选 | BF16 或 FP16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | 键张量，与 query 共享 `[T, Nk, Dk]` 布局；chunk 内同时参与 `attn=-(k_beta@K^T)*decay` 与 `attn_i=(Q@K^T)*decay`。 |
| value | 必选 | BF16 或 FP16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 值张量，`Nv` 为 V 头数（支持 GQA，`Nv` 可与 `Nk` 不同，但需 `Nv % Nk == 0`），`Dv` 为每头值维度。 |
| beta | 必选 | BF16 或 FP16 | `[T, Nv]` | `[seq, v_head]` | 门控系数 β，每 token 每 V 头一个标量。用于生成 `k_beta=K*beta` 与 `v_beta=V*beta`；`Nv` 维与 value 的头维对齐。 |
| initial_state | 必选 | BF16 或 FP16 | `[B, Nv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 初始递归状态 `S₀`。`B` 为 batch 数，由该输入推导（`tilingData.b`）。Atlas 800I A2 上 dtype 必须与 7 张量统一（BF16 或 FP16），FP32 仅 ascend950 支持。 |
| actual_seq_lengths | 必选 | int32 | `[B]` | `[batch]` | 每个 batch 的真实序列长度；`T = Σ actual_seq_lengths[b]`。kernel 据此跳过 padding、做核间切分。 |
| g | 可选 | float32 | `[T, Nv]` | `[seq, v_head]` | log 衰减门控 g（“gate”）。提供时计算 `g_cumsum=cumsum(g)` 与 `exp(g_cumsum)`，参与衰减掩码与状态更新；缺省时按全 0 处理（退化为无衰减的 Delta Rule）。注意：`g` 声明为 `ge::DT_FLOAT`，与其它 7 张量的 BF16/FP16 不同。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| out | 必选 | BF16 或 FP16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 每个 token 的注意力输出。由 `out=(Q*exp_g)@state + attn_i@v_new` 得到，前者在 Stage2 由 `q_prime@state^T` 累加、后者在 Stage3 由 `masked_qkt@vInner` 累加。`AutoContiguous` 保证连续存储。dtype 由 `query` 推导（`InferDataType`）。 |
| final_state | 必选 | BF16 或 FP16 | `[B, Nv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 更新后的递归状态，形状与 `initial_state` 相同（由 `infershape` 直接镜像）。dtype 由 `initial_state` 推导。Atlas 800I A2 上跨 chunk 以输入 dtype 携带（不升 FP32）。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| scale_value | 可选 | float | 1.0 | QK 注意力点的缩放因子（通常取 `1/√Dk`）。作用于 `Q@K^T` 与 `(Q*exp_g)@state` 两处（`q_prime = Q*scale*exp_g`、Stage3 `qkt *= scale`）。 |

### 2.4 Tiling 数据结构

host 侧 `ChunkGatedDeltaRuleTilingData` 与 device 侧镜像结构体（`op_host/chunk_gated_delta_rule_tiling.h` 与 `op_kernel/chunk_gated_delta_rule_tiling_data.h`），二者必须**字节一致**且位于全局命名空间（CANN tiling 宏 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA` 在本 repo 的 customize 构建下要求全局作用域，命名空间限定会使 autogen emit 的 tiling 符号无法解析），`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| aiCoreNum | int64_t | AICore 核数（`blockDim`） |
| t | int64_t | 总序列长度 T |
| nk | int64_t | Q/K 头数 |
| dk | int64_t | 键头维 Dk |
| nv | int64_t | V 头数 |
| dv | int64_t | 值头维 Dv |
| b | int64_t | batch 数（来自 `initial_state` 第 0 维） |
| hasGamma | int64_t | 是否提供可选输入 `g`（1/0） |
| chunkSize | int64_t | 分块大小，固定 64 |
| maxGroupLength | int64_t | 单 chunk 组最大长度 = `p * aiCoreNum * chunkSize`（`p=2`） |
| interWorkspaceSz | int64_t | 跨 chunk 中间张量 workspace 字节数 |
| stageWorkspaceSz | int64_t | Stage1 临时 workspace 字节数 |
| stageOneParaNum | int64_t | Stage1 单核并行度（固定 4） |
| scale | float | 缩放因子 `scale_value` |
| matmulTilingFp32 | TCubeTiling | 低精度（BF16/FP16）→ 低精度 Cube matmul tiling（arch22 / Atlas 800I A2 主路径） |
| matmulTilingFp32C | TCubeTiling | BF16 → FP32 Cube matmul tiling（ascend950 FP32-state 路径专用，Atlas 800I A2 上未使用） |
| stateIsFp32 | int64_t | 是否为 FP32 state（1/0）；Atlas 800I A2 上恒为 0 |
| isFp16 | int64_t | 1 = 7 张量为 FP16；0 = BF16（Atlas 800I A2 路径，device 入口据此分派模板） |
| stateStride0 | int64_t | `final_state` 第 0 维 stride（batch 间 stride） |
| stateStride1 | int64_t | `final_state` 第 1 维 stride（= `Dk*Dv`，缺省时由推导） |

另在 `ChunkGatedDeltaRule` 命名空间内有 `ChunkGroup` 结构体（仅 kernel 类引用）：`startPos` / `length` / `chunkSize` / `coreStart` / `coreEnd`。

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 800I A2 | Atlas 300I Duo有独立版本，不可混用。 |
| 输入数据类型 | query/key/value/beta/initial_state/out/final_state = **全 BF16 或全 FP16**（7 张量统一）；actual_seq_lengths = int32；g（可选）= float32 | `CheckLowDtype` 要求 7 张量同 BF16 或同 FP16，拒绝混合 dtype 组合；`CheckStateDtype` 要求 state/final_state 与低精度 dtype 一致；`CheckAuxDtype` 要求 `g`（若提供）为 FLOAT。 |
| 输出数据类型 | out / final_state = 与对应输入同 dtype（BF16 或 FP16） | `InferDataType`：out 跟 query、final_state 跟 initial_state。 |
| 输入格式 | `FORMAT_ND` | def 中 ND + `DynamicFormatFlag(true)` + `DynamicRankSupportFlag(true)`，支持动态 shape 与动态秩；tiling 仅拒绝 `FORMAT_FRACTAL_NZ`，其它 ND 派生布局放行。 |
| chunkSize | 固定 64 | `DoOpTiling` 中 `c = 64`，不可配置。 |
| Dk（键头维） | ≤ 128 | `CheckDerivedDimConstraints` 要求 `dk ≤ 128`。 |
| Dv（值头维） | ≤ 128 | 要求 `dv ≤ 128`。 |
| 头数 Nk / Nv | 均 ≤ 64，且 `Nv % Nk == 0` | 支持 GQA：`Nv` 可大于 `Nk`，但 `Nv` 必须是 `Nk` 的整数倍。 |
| T / B | `T > 0`，`B > 0` | `B` 由 `initial_state` 第 0 维推导；`T` 由 `query` 第 0 维推导且需 `T = Σ actual_seq_lengths[b]`。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`；tiling 运行时按实际 shape 计算 workspace、matmul tiling、blockDim。 |
| workspace | ≥ 16 MiB | `SYS_WORKSPACE_SIZE = 16MB` 下限，外加 `interWorkspaceSz`（跨 chunk 中间张量：`gCumExp`/`kCumDecay`/`vInner`/`qPrime`/`attnInter`/`kg`/`qkt`/`highState`/mask）与 `stageWorkspaceSz`（Stage1 每核临时区）。 |
| 任务类型 | `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV） | 三阶段流水由 Cube（`MatmulImpl`）与 Vector 协同，`CrossCore` flag 同步。 |

### 3.2 不支持的场景

- **非 Atlas 800I A2 硬件**：部署到这些平台会找不到编译产物。Atlas 300I Duo 请用 [`ascend_300iduo/chunk_gated_delta_rule`](../../ascend_300iduo/chunk_gated_delta_rule/README.md)。
- **混合 dtype**：7 个可变张量必须全 BF16 或全 FP16。`DataTypeList` 虽生成 `2^7=128` 份 binary，但 tiling 的 `CheckLowDtype` 在运行时拒绝混合组合，仅全 BF16 / 全 FP16 binary 会被选中。
- **FP32 state（Atlas 800I A2）**：`CheckStateDtype` 中 `stateDtype == ge::DT_FLOAT && SOC_VERSION_IS_NOT_950` 直接 `GRAPH_FAILED`；FP32 state 路径（含 `matmulTilingFp32C`、`stateBf16Wk`）仅 ascend950 可用，Atlas 800I A2 上不可达。
- **`g` 非 float32**：def 固定 `g` 为 `ge::DT_FLOAT`；若上游以 BF16/FP16 传入 `g`，需在接入层先转换，否则 `CheckAuxDtype` 失败。
- **`Dk` 或 `Dv` > 128**：`CheckDerivedDimConstraints` 直接 `GRAPH_FAILED`。
- **`Nk` 或 `Nv` > 64**：同上，`GRAPH_FAILED`。
- **`Nv % Nk != 0`**：GQA 头数关系不满足时 `GRAPH_FAILED`。
- **`FRACTAL_NZ` 格式**：`AnalyzeFormat` 拒绝；其它非 ND 派生布局放行。
- **训练 / 反向**：本算子仅前向（`KERNEL_TYPE_MIX_AIC_1_2`），不含反向梯度。

### 3.3 数值与精度说明

- **BF16 vs FP16**：Atlas 800I A2 Cube 原生支持两种低精度输入的 `MMAD`，L0C 以 FP32 累加；BF16 尾数位较少（7 位）但动态范围大，FP16 尾数位较多（10 位）但范围窄。对于 `g` 真实激活下 `exp(g_cum)` 跨度较大的场景，BF16 通常更稳定；FP16 在 `|g_cum|` 较大时 `exp` 可能溢出。
- **state 跨 chunk 精度（Atlas 800I A2）**：Atlas 800I A2 上 state 以输入 dtype（BF16 或 FP16）在 `final_state` 输出张量中跨 chunk 携带（Stage2 的 `CalStateNew` 直接以低精度写出）；Stage2 内部 `CalGCumExp` 会把 state 先 cast 到 FP32 做 `exp(g_last)` 缩放再 cast 回低精度，但 chunk 间的 state 存储是低精度。这与 Atlas 300I Duo 的“workspace 内跨 chunk FP32 保存、仅末 chunk cast 回 FP16”策略不同——Atlas 800I A2 因不支持 FP32 state 路径而无法做该补偿，精度上对极端长序列可能有微小差异。
- **`exp(g)` 数值稳定**：算子刻意避免 `exp(g_i - g_j) = exp(g_i)·exp(-g_j)` 的因子分解（会溢出），改为先按行广播构造完整因果差值矩阵 `g_cum[i] - g_cum[j]`、乘以下三角 mask 清上三角、再统一 `Exp`（Stage1 `GammaCompute`、Stage3 `CalMaskedQKT`）。
- **`(I-A)⁻¹` 递归三角求逆**：64×64 的严格下三角矩阵求逆按 2×2 分块（32×32 对角块）分解，对角块在 Vector 上用 `Gather + Broadcast + MulAddDst` 做外积更新，跨块稠密积由 AIC Cube 完成（`AttnInverseMMCompute`），保证 `I-A` 对角为 1 时求逆的数值稳定。

## 4 参考资源

- **算子来源**：copy from <https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule>（接口与开源 ops-transformer 一致，TND layout，Atlas 800I A2 only）
- **CANN 算子开发指南**：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>（Cube/Vector 流水、`MatmulImpl`/`MMAD`/`DataCopyPad`/`Broadcast`/`Gather`/`MulAddDst`/`CrossCore` 等原语）
