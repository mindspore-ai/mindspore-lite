# FusedQkL2NormChunkGatedDeltaRule

## 1 功能说明

`FusedQkL2NormChunkGatedDeltaRule`（FusedCGDR）是 Gated Delta Rule（门控 Delta 规则）chunk 级前向算子，面向 MindSpore Lite 昇腾 Atlas 800I A2 云侧推理场景。

它在 [ChunkGatedDeltaRule](../chunk_gated_delta_rule/README.md) 的 chunk 递推之上，把调用方原本在 Python 侧做的两步逐元素处理搬进 kernel：

1. **q/k 逐行 L2 归一化**（`QkL2Norm`）：原实现是 `F.normalize(q, p=2, dim=-1)` / `F.normalize(k, ...)`，在 24 层 linear-attention 各调用两次，host 上拆成 `LpNorm + Div + ClampMin + BroadcastTo + Slice` 一整串小算子；现在 q/k 以**原始值**传入，kernel 在 Stage1 把每个 token 的行向量归一化后再参与后续 Cube 计算。
2. **输出 padding 行清零**（`out_zero_pad_rows`）：多 batch / 定长分桶场景下 `T` 会被补齐到块大小，`[actual_seq_len, T)` 的输出行是无效数据，原实现在 Python 侧用 `masked_fill` 逐层清零；现在可选由 kernel 在收尾阶段直接写 0。

之所以做成**独立新算子**而不是给 `ChunkGatedDeltaRule` 加开关：融合改变了算子的输入语义（q/k 由“已归一化”变为“原始值”），也改变了输出写后行为（多写一段 padding 区），两者混在一个 op 里会让原有调用方在升级后静默变语义。独立算子让两条路径（纯 CGDR / 融合 CGDR）各自可验证、可回退。

kernel 主体移植自 [CANN ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_qk_l2norm_chunk_gated_delta_rule)，接口（TND layout、输入输出名/顺序/dtype）与开源 ops-transformer 一致，融合部分是本次在本仓库新增。

### 1.1 算子语义

给定**原始**查询 `Q`、**原始**键 `K`、值 `V`、门控 `β`、初始递归状态 `S₀`、可选 log 衰减门控 `g` 以及缩放因子 `scale`，算子按固定 `chunkSize=64` 将序列切分为若干 chunk，在每个 chunk 内完成下述递推，输出每个 token 的注意力值 `out` 与更新后的最终状态 `final_state`：

```text
# 0. 行 L2 归一化（融合项 1）
Qn          = Q / max(||Q||₂, 1e-6)                       # 逐 token、逐头，按 Dk 求范数
Kn          = K / max(||K||₂, 1e-6)

# 1. 系数预处理
k_beta      = Kn * beta                                  # [T, Nk, Dk]
g_cumsum    = cumsum(g, dim=t)                           # [T, Nv]   (g 可选，缺省为 0)
exp_g       = exp(g_cumsum)
decay[i][j] = exp(g_cumsum[i] - g_cumsum[j])             # 因果下三角

# 2. chunk 内递归注意力矩阵（严格下三角，对角为 0）
attn        = -(k_beta @ Kn^T) * decay                   # chunk 内 intra-attn
attn        = (I - attn)^{-1}                            # 递归三角求逆（逐 chunk）
attn_i      = (Qn @ Kn^T) * decay                        # QK 注意力（下三角 + 对角）

# 3. chunk 内状态交互
k_cumdecay  = -1 * k_beta * exp_g                        # = -Kn*beta*exp(g_cum)
v_beta      = V * beta                                    # [T, Nv, Dv]
v_new       = attn @ v_beta  -  k_cumdecay @ state        # chunk 内新值
kg          = Kn * exp(g_last - g_cum)                    # 跨 chunk 衰减后的 K

# 4. 输出与状态更新
out         = (Qn * exp_g) @ state + attn_i @ v_new       # 每个输出 token
state       = kg^T @ v_new + exp(g_last) * state          # 跨 chunk 携带

# 5. padding 行清零（融合项 2，可选）
out[actual_seq_len:T] = 0
```

其中 `state` 维度为 `[Dv, Dk]`（每头），在 chunk 之间以输入 dtype（BF16 或 FP16）携带保存于 `final_state` 输出张量中（Atlas 800I A2 上 FP32 state 路径被 tiling 拒绝，仅 ascend950 支持）。

### 1.2 实现要点

- **arch22 三阶段流水（Cube + Vector，MIX_AIC_1_2）**：kernel 为 `KERNEL_TYPE_MIX_AIC_1_2` 构建（1 AIC : 2 AIV），三阶段在 `op_kernel/arch22/` 下分别由 `fused_qk_l2norm_chunk_gated_delta_rule_stage1.h` / `stage2.h` / `stage3.h` 实现，通过 `CrossCoreSetFlag/WaitFlag` 在 Cube 与 Vector 间同步：
  - **Stage 1（q/k 归一化 + chunk 内预处理 + 递归三角求逆）**：AIV 负责 Q/K 连续化（从 TND layout Gather 成连续块）、`QKNormalizeRows` 行归一化、`g_cumsum/exp(g_cum)`、衰减掩码 `decay=exp(g_cum[i]-g_cum[j])` 构造、`v_beta=V*beta`、`q_prime=Qn*scale*exp_g`、`k_cumdecay=-Kn*beta*exp_g`、`kg=Kn*exp(g_last-g_cum)`，以及 64×64 的 `(I-A)⁻¹` 递归三角求逆（对角 32×32 块用 `Gather+Broadcast+MulAddDst` 外积更新，跨块稠密积由 AIC 完成）；AIC 负责 `Kn@Kn^T`、`Qn@Kn^T`、`attn@k_cumdecay`、`attn@v_beta` 四个 Cube 矩阵乘。产物：`qkt`、`kCumDecay`、`vInner`、`qPrime`、`kg`、`gCumExp`。
  - **Stage 2（chunk 间状态递归 + `out` 的 Q@state 项）**：按 chunk 顺序串行推进，AIV 把上一 chunk 的 state 用 `exp(g_last)` 缩放后写回，AIC 在 Cube 上完成 `v_inner += k_cumdecay @ state^T`（`v_new` 的减项）、`out += q_prime @ state^T`（`out` 的第一项 `(Qn*exp_g)@state`）、`state_new = v_inner^T @ kg`（状态更新）；跨核 flag 保证“AIC 读旧 state → AIV 写缩放 state → AIC 写新 state”的有序性。
  - **Stage 3（`out` 的 `attn_i@v_new` 项 + 因果 mask）**：AIV 用下三角含对角 mask（`stageThreeMask`）对 `qkt` 做 `scale * decay * mask` 得到 `masked_qkt`，AIC 在 Cube 上完成 `masked_qkt @ vInner` 并累加到 `out`，补上 `out` 公式的第二项 `attn_i@v_new`。
- **行 L2 归一化在 Stage1 的 Gather 之后就地完成（融合项 1，`QKNormalizeRows`）**：kernel 每读入一个子 chunk 的 q 或 k（`QKPreProcess`），先把有效行 cast 到 FP32、求平方、按行归约出 `Σx²`、开方、低精度往返一次、`max(·, 1e-6)` 夹紧，再把分母广播到 `[rows, Dk]` 在 FP32 下做除法、单次低精度舍入写回原 buffer；后续所有使用 q/k 的步骤天然读到归一化后的值，`kg`（未提供 `g` 时由 Stage1 直接落盘的那条路径）同样写出归一化值。归约用 64 lane 的 `WholeReduceSum`（raw vcadd）分半累加，`dkAligned/64` 个半段循环相加，因此 Dk=64/128/256 都适用；相邻 8 行一组批量处理（`NORM_GROUP_ROWS`），把指令开销摊到组上，无标量往返。
- **padding 行清零在收尾阶段完成（融合项 2，`ZeroOutPadRows`）**：`Process()` 在三个 Stage 全部结束后，若 tiling 里 `outZeroPadRows=1`，由 AIV 把 `out` 的第 `[actual_seq_len, T)` 行按 1024 元素一块 `DataCopyPad` 写 0。Stage1/2/3 只写 `[0, actual_seq_len)`，所以这段区域否则是未初始化数据（Device 内存不保证为 0），必须显式清零才能替代 Python 侧的 `masked_fill`。
- **arch22 原生 BF16/FP16 Cube**：Atlas 800I A2 的 Cube 单元原生支持 BF16 与 FP16 输入的 `MMAD`（`MatmulImpl`，ND 格式），所有递推中的稠密矩阵乘（`Kn@Kn^T`、`Qn@Kn^T`、`attn@k_cumdecay`、`attn@v_beta`、`q_prime@state`、`k_cumdecay@state`、`v_inner^T@kg`、`masked_qkt@vInner`）均直接在 Cube 上以输入 dtype 完成，无需 Atlas 300I Duo 版本的 FP16 高低半拆分补偿。
- **BF16/FP16 双 dtype 编译（`DataTypeList`）**：def 中 7 个可变张量（query/key/value/beta/initial_state/out/final_state）用 `DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})` + `FormatList({ge::FORMAT_ND, ge::FORMAT_ND})` 声明，配合 `DynamicFormatFlag(true)`，opbuild 会为每种 dtype 组合生成一份 kernel binary（`2^7 = 128` 份）。但 tiling 在运行时拒绝混合 dtype（`CheckLowDtype` 要求 7 张量同 BF16 或同 FP16），因此实际只会选中全 BF16 与全 FP16 两份 binary。device 入口按 `tilingData.isFp16` 分派 `FusedCGDR<half, float>` 或 `FusedCGDR<bfloat16_t, float>`，二者 `highType=float`。
- **`tilingKey` 分桶**：BF16 与 FP16 **共用 tilingKey=0**（`TILING_KEY_FusedCGDR_BF16_STATE`），因为二者使用同一份 tiling 结构、仅 `isFp16` 标志不同；为 FP16 单独引入 tilingKey（如 2）不会在 CANN autogen 的 `tiling_struct_expr_map` 中注册，会在 convert 阶段抛 `KeyError`。FP32 state 走 tilingKey=1（`TILING_KEY_FusedCGDR_FP32_STATE`），但在 Atlas 800I A2 上被 tiling 拒绝（不支持 FP32 state），故该分支不可达。
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
| query | 必选 | BF16 或 FP16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | **原始**查询张量（未归一化，归一化由本算子完成），`T` 为所有 batch 拼接后的总序列长度（`T = Σ actual_seq_lengths[b]`），`Nk` 为 Q/K 头数，`Dk` 为每头键维度。与 key 同布局。 |
| key | 必选 | BF16 或 FP16 | `[T, Nk, Dk]` | `[seq, qk_head, qk_dim]` | **原始**键张量，与 query 共享 `[T, Nk, Dk]` 布局且同样由本算子归一化；归一化后同时参与 `attn=-(k_beta@Kn^T)*decay`、`attn_i=(Qn@Kn^T)*decay` 与 `kg`。 |
| value | 必选 | BF16 或 FP16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 值张量，`Nv` 为 V 头数（支持 GQA，`Nv` 可与 `Nk` 不同，但需 `Nv % Nk == 0`），`Dv` 为每头值维度。 |
| beta | 必选 | BF16 或 FP16 | `[T, Nv]` | `[seq, v_head]` | 门控系数 β，每 token 每 V 头一个标量。用于生成 `k_beta=Kn*beta` 与 `v_beta=V*beta`；`Nv` 维与 value 的头维对齐。 |
| initial_state | 必选 | BF16 或 FP16 | `[B, Nv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 初始递归状态 `S₀`。`B` 为 batch 数，由该输入推导（`tilingData.b`）。Atlas 800I A2 上 dtype 必须与 7 张量统一（BF16 或 FP16），FP32 仅 ascend950 支持。 |
| actual_seq_lengths | 必选 | int32 | `[B]` | `[batch]` | 每个 batch 的真实序列长度；`T = Σ actual_seq_lengths[b]`。kernel 据此跳过 padding、做核间切分，并在 `out_zero_pad_rows=1` 时决定清零范围。 |
| g | 可选 | float32 | `[T, Nv]` | `[seq, v_head]` | log 衰减门控 g（“gate”）。提供时计算 `g_cumsum=cumsum(g)` 与 `exp(g_cumsum)`，参与衰减掩码与状态更新；缺省时按全 0 处理（退化为无衰减的 Delta Rule）。注意：`g` 声明为 `ge::DT_FLOAT`，与其它 7 张量的 BF16/FP16 不同。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| out | 必选 | BF16 或 FP16 | `[T, Nv, Dv]` | `[seq, v_head, v_dim]` | 每个 token 的注意力输出。由 `out=(Qn*exp_g)@state + attn_i@v_new` 得到，前者在 Stage2 由 `q_prime@state^T` 累加、后者在 Stage3 由 `masked_qkt@vInner` 累加；`out_zero_pad_rows=1` 时 `[actual_seq_len, T)` 的行在收尾阶段被清零。`AutoContiguous` 保证连续存储。dtype 由 `query` 推导（`InferDataType`）。 |
| final_state | 必选 | BF16 或 FP16 | `[B, Nv, Dv, Dk]` | `[batch, v_head, v_dim, qk_dim]` | 更新后的递归状态，形状与 `initial_state` 相同（由 `infershape` 直接镜像）。dtype 由 `initial_state` 推导。Atlas 800I A2 上跨 chunk 以输入 dtype 携带（不升 FP32）。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| scale_value | 可选 | float | 1.0 | QK 注意力点的缩放因子（通常取 `1/√Dk`）。作用于 `Q@K^T` 与 `(Q*exp_g)@state` 两处（`q_prime = Qn*scale*exp_g`、Stage3 `qkt *= scale`）。**不参与归一化**：归一化在 scale 之前完成。 |
| out_zero_pad_rows | 可选 | bool | false | 为 true 时，kernel 在全部 Stage 结束后把 `out` 的 `[actual_seq_len, T)` 行清零，替代调用方的 `masked_fill`。单 batch、`T == actual_seq_len` 时置 false 可省掉这段写。 |

### 2.4 Tiling 数据结构

host 侧 `FusedQkL2NormChunkGatedDeltaRuleTilingData` 与 device 侧镜像结构体（`op_host/fused_qk_l2norm_chunk_gated_delta_rule_tiling.h` 与 `op_kernel/fused_qk_l2norm_chunk_gated_delta_rule_tiling_data.h`），二者必须**字节一致**且位于全局命名空间（CANN tiling 宏 `REGISTER_TILING_DEFAULT` / `GET_TILING_DATA` 在本 repo 的 customize 构建下要求全局作用域，命名空间限定会使 autogen emit 的 tiling 符号无法解析），`#pragma pack(8)` + `alignas(8)`。

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
| outZeroPadRows | int64_t | 1 = 收尾时清零 `out` 的 `[actual_seq_len, T)` 行；0 = 不动该区域 |
| scale | float | 缩放因子 `scale_value` |
| matmulTilingFp32 | TCubeTiling | 低精度（BF16/FP16）→ 低精度 Cube matmul tiling（arch22 / Atlas 800I A2 主路径） |
| matmulTilingFp32C | TCubeTiling | BF16 → FP32 Cube matmul tiling（ascend950 FP32-state 路径专用，Atlas 800I A2 上未使用） |
| stateIsFp32 | int64_t | 是否为 FP32 state（1/0）；Atlas 800I A2 上恒为 0 |
| isFp16 | int64_t | 1 = 7 张量为 FP16；0 = BF16（Atlas 800I A2 路径，device 入口据此分派模板） |
| stateStride0 | int64_t | `final_state` 第 0 维 stride（batch 间 stride） |
| stateStride1 | int64_t | `final_state` 第 1 维 stride（= `Dk*Dv`，缺省时由推导） |

另在 `FusedQkL2NormChunkGatedDeltaRule` 命名空间内有 `ChunkGroup` 结构体（仅 kernel 类引用）：`startPos` / `length` / `chunkSize` / `coreStart` / `coreEnd`。

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 800I A2 | Atlas 300I Duo 有独立版本，不可混用。 |
| 输入数据类型 | query/key/value/beta/initial_state/out/final_state = **全 BF16 或全 FP16**（7 张量统一）；actual_seq_lengths = int32；g（可选）= float32 | `CheckLowDtype` 要求 7 张量同 BF16 或同 FP16，拒绝混合 dtype 组合；`CheckStateDtype` 要求 state/final_state 与低精度 dtype 一致；`CheckAuxDtype` 要求 `g`（若提供）为 FLOAT。 |
| 输出数据类型 | out / final_state = 与对应输入同 dtype（BF16 或 FP16） | `InferDataType`：out 跟 query、final_state 跟 initial_state。 |
| 输入格式 | `FORMAT_ND` | def 中 ND + `DynamicFormatFlag(true)` + `DynamicRankSupportFlag(true)`，支持动态 shape 与动态秩；tiling 仅拒绝 `FORMAT_FRACTAL_NZ`，其它 ND 派生布局放行。 |
| chunkSize | 固定 64 | `DoOpTiling` 中 `c = 64`，不可配置。 |
| Dk（键头维） | ≤ 128，且 16 元素对齐后为 64/128/256 之一的 2 的幂 | 除原有 `dk ≤ 128` 外，行归一化的 `WholeReduceSum` 以 64 lane 为半段、按 `dkAligned/64` 段累加，要求 `dkAligned = align(dk, 16)` 是 [64, 256] 内的 2 的幂，否则 tiling 直接 `GRAPH_FAILED`。Qwen3.5-9B 的 `Dk=128` 天然满足。 |
| Dv（值头维） | ≤ 128 | 要求 `dv ≤ 128`。 |
| 头数 Nk / Nv | 均 ≤ 64，且 `Nv % Nk == 0` | 支持 GQA：`Nv` 可大于 `Nk`，但 `Nv` 必须是 `Nk` 的整数倍。 |
| T / B | `T > 0`，`B > 0` | `B` 由 `initial_state` 第 0 维推导；`T` 由 `query` 第 0 维推导且需 `T = Σ actual_seq_lengths[b]`。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`；tiling 运行时按实际 shape 计算 workspace、matmul tiling、blockDim。 |
| workspace | ≥ 16 MiB | `SYS_WORKSPACE_SIZE = 16MB` 下限，外加 `interWorkspaceSz`（跨 chunk 中间张量：`gCumExp`/`kCumDecay`/`vInner`/`qPrime`/`attnInter`/`kg`/`qkt`/`highState`/mask）与 `stageWorkspaceSz`（Stage1 每核临时区）。 |
| 任务类型 | `KERNEL_TYPE_MIX_AIC_1_2`（1 AIC : 2 AIV） | 三阶段流水由 Cube（`MatmulImpl`）与 Vector 协同，`CrossCore` flag 同步。 |

### 3.2 不支持的场景

- **非 Atlas 800I A2 硬件**：部署到这些平台会找不到编译产物。
- **混合 dtype**：7 个可变张量必须全 BF16 或全 FP16。`DataTypeList` 虽生成 `2^7=128` 份 binary，但 tiling 的 `CheckLowDtype` 在运行时拒绝混合组合，仅全 BF16 / 全 FP16 binary 会被选中。
- **FP32 state（Atlas 800I A2）**：`CheckStateDtype` 中 `stateDtype == ge::DT_FLOAT && SOC_VERSION_IS_NOT_950` 直接 `GRAPH_FAILED`；FP32 state 路径（含 `matmulTilingFp32C`、`stateBf16Wk`）仅 ascend950 可用，Atlas 800I A2 上不可达。
- **`g` 非 float32**：def 固定 `g` 为 `ge::DT_FLOAT`；若上游以 BF16/FP16 传入 `g`，需在接入层先转换，否则 `CheckAuxDtype` 失败。
- **`Dk` 或 `Dv` > 128**：`CheckDerivedDimConstraints` 直接 `GRAPH_FAILED`；`dkAligned` 非 2 的幂（或不在 [64, 256]）同样被 tiling 拒绝。
- **`Nk` 或 `Nv` > 64**：同上，`GRAPH_FAILED`。
- **`Nv % Nk != 0`**：GQA 头数关系不满足时 `GRAPH_FAILED`。
- **`FRACTAL_NZ` 格式**：`AnalyzeFormat` 拒绝；其它非 ND 派生布局放行。
- **训练 / 反向**：本算子仅前向（`KERNEL_TYPE_MIX_AIC_1_2`），不含反向梯度。

### 3.3 数值与精度说明

- **q/k 归一化数值路径**：平方和以 FP32 累加（`Cast → Mul → WholeReduceSum`），`Sqrt` 后先往返一次低精度再参与 `max(·, 1e-6)`，夹紧在 FP32 下比较、再往返一次低精度，然后广播分母、FP32 除法、最后一次低精度舍入。这与 host 侧 `F.normalize`（`LpNorm` 保持输入 dtype + `Div` 在 FP32 计算后一次 RNE 舍入）逐步对齐；全零行经 1e-6 夹紧后归一化为 0 而不是 NaN。补零的尾部 lane 在拷贝时被显式写 0，因此 `Σx²` 与只对真实 `Dk` 求和一致。
- **BF16 vs FP16**：Atlas 800I A2 Cube 原生支持两种低精度输入的 `MMAD`，L0C 以 FP32 累加；BF16 尾数位较少（7 位）但动态范围大，FP16 尾数位较多（10 位）但范围窄。对于 `g` 真实激活下 `exp(g_cum)` 跨度较大的场景，BF16 通常更稳定；FP16 在 `|g_cum|` 较大时 `exp` 可能溢出。
- **state 跨 chunk 精度（Atlas 800I A2）**：Atlas 800I A2 上 state 以输入 dtype（BF16 或 FP16）在 `final_state` 输出张量中跨 chunk 携带（Stage2 的 `CalStateNew` 直接以低精度写出）；Stage2 内部 `CalGCumExp` 会把 state 先 cast 到 FP32 做 `exp(g_last)` 缩放再 cast 回低精度，但 chunk 间的 state 存储是低精度。这与 Atlas 300I Duo 的“workspace 内跨 chunk FP32 保存、仅末 chunk cast 回 FP16”策略不同——Atlas 800I A2 因不支持 FP32 state 路径而无法做该补偿，精度上对极端长序列可能有微小差异。
- **`exp(g)` 数值稳定**：算子刻意避免 `exp(g_i - g_j) = exp(g_i)·exp(-g_j)` 的因子分解（会溢出），改为先按行广播构造完整因果差值矩阵 `g_cum[i] - g_cum[j]`、乘以下三角 mask 清上三角、再统一 `Exp`（Stage1 `GammaCompute`、Stage3 `CalMaskedQKT`）。
- **`(I-A)⁻¹` 递归三角求逆**：64×64 的严格下三角矩阵求逆按 2×2 分块（32×32 对角块）分解，对角块在 Vector 上用 `Gather + Broadcast + MulAddDst` 做外积更新，跨块稠密积由 AIC Cube 完成（`AttnInverseMMCompute`），保证 `I-A` 对角为 1 时求逆的数值稳定。

### 3.4 与“`F.normalize` + 纯 CGDR”路径的等价性

- 本算子与 [ChunkGatedDeltaRule](../chunk_gated_delta_rule/README.md) 共用 chunk 递推主体，差异只有 Stage1 的原地行归一化和收尾的 padding 清零；因此相同输入下必须与 `F.normalize(q/k, p=2, dim=-1)` + 纯 CGDR 的输出一致。这也是把它做成独立算子、而不是给原算子加开关的原因：原路径保持可逐比特回归。
- 归一化发生在 `scale` 之前：`q_prime = Qn * scale * exp_g`、Stage3 的 `qkt *= scale` 都作用在归一化后的 Q/K 上，与 host 侧 `F.normalize` 后再乘 `scale` 的顺序一致。
- 验证用例 [test/ops/test_fused_qk_l2norm_chunk_gated_delta_rule.py](../../../../../../lite_boost/test/ops/test_fused_qk_l2norm_chunk_gated_delta_rule.py) 覆盖 GQA 16×32、MHA 16×16、小 GQA 4×8 三组形状：对 `out` 与 `final_state` 均按 `rtol=2e-3, atol=0` 与 “`F.normalize` + 纯 CGDR” 比对（bf16 下 host `F.normalize` 与 kernel 内归一化只差一次舍入）；`out_zero_pad_rows=True` 时断言 `out[valid:]` 全 0 且有效行非全 0；可选输入 `g` 缺省时与 `hasGamma=0` 路径同样比对。

## 4 参考资源

- **算子来源**：copy from <https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_qk_l2norm_chunk_gated_delta_rule>（接口与开源 ops-transformer 一致，TND layout，Atlas 800I A2 only）；行归一化与 padding 清零为本次新增融合项。
- **验证用例**：[test/ops/test_fused_qk_l2norm_chunk_gated_delta_rule.py](../../../../../../lite_boost/test/ops/test_fused_qk_l2norm_chunk_gated_delta_rule.py)。
- **调用入口**：`torch.ops.lite_boost.fused_qk_l2norm_chunk_gated_delta_rule(query, key, value, beta, initial_state, actual_seq_lengths, g=None, scale_value=1.0, out_zero_pad_rows=False) -> (out, final_state)`，注册见 `lite_boost/src/ops/register_ops.cc`。
- **CANN 算子开发指南**：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>（Cube/Vector 流水、`MatmulImpl`/`MMAD`/`DataCopyPad`/`Broadcast`/`WholeReduceSum`/`Gather`/`MulAddDst`/`CrossCore` 等原语）
