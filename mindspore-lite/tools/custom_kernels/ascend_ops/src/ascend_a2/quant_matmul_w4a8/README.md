# QuantMatmulW4a8

## 1 功能说明

`QuantMatmulW4a8` 是 **W4A8 量化矩阵乘**算子（权重 INT4、激活 INT8），面向 MindSpore Lite 昇腾 Atlas 800I A2云侧推理场景，用于 LLM Linear 层的 INT4×INT8 量化加速。它基于 CANN ops-nn 的 `quant_batch_matmul_v4`（V5 MSD 实现）改造而来，差异在于新增 `output_bias` 输出偏置并在反量化阶段应用逐通道权重 scale。

### 1.1 算子语义

给定 INT8 激活 `act`、INT4 权重 `weight`、逐通道权重 scale `scale`、预乘权重 scale 的偏置 `bias`、逐 token 激活 scale `x_scale`、输出偏置 `output_bias`，计算：

```text
# Phase 1（AIV）：激活 in-place 拆分 int8→int4b_t（每行拆成高 4 位/低 4 位两行）
act_i4 = split_int8_to_int4b(act)          # [M, K] int8 → [2M, K/2] int4b_t（高半/低半交错）

# Phase 2（AIC Cube + AIV 反量化）：
mm_out = act_i4 @ weight^T                   # INT4×INT4 矩阵乘，FP16/L0C 累加（Cube）
          + bias                              # bias 已在 host 端预乘 w_scale，作为 y_offset
result = mm_out × w_scale                     # 逐通道权重 scale（VEC，Cube 用 SetQuantScalar(1.0) 恒等）
result = result × x_scale                     # 逐 token 激活 scale（VEC，BroadCast+Brcb）
result = result + output_bias                 # 输出偏置（VEC，×x_scale 之后施加）
out    = bf16(result)                          # Cast FP32→BF16 输出
```

输出 `out` 为 `[M, N]` 的 BF16 张量。注意 host 侧 `bias` 输入语义为「已在 host 端预乘 `w_scale`」的偏置（kernel 内作为 `y_offset` 直接相加，不再乘 scale）。

### 1.2 实现要点

- **AIC + AIV 双核协同（MSD 模式）**：Atlas 800I A2 上 AIC 做 Cube 矩阵乘、AIV 做反量化，二者通过 `CrossCoreSetFlag/CrossCoreWaitFlag`（同步 ID `SYNC_AIC_TO_AIV=5`、`SYNC_AIV_TO_AIC=3`）乒乓流水；`SetScheduleMode(1)` 独占核模式。
- **Phase 1 in-place 拆分**：AIV 把 `[M,K]` int8 激活就地拆成 `[2M, K/2]` int4b_t —— 高 4 位行先 `Cast→Muls(1/16)→CAST_FLOOR` 取高半，低 4 位行用 `And(0x0F0F)` 取低半再 `Adds(-8)` 还原有符号值；拆分结果直接写回 `act` 自身（无独立 split buffer，V5 模式）。
- **Phase 2 Cube INT4×INT4**：用 `MatmulImpl`（A=ND int4b_t、B=ND int4b_t bTrans=true、C=ND half、`CFG_MDL`），`SetQuantScalar(0x3F800000`即1.0`) 让 Cube 输出恒等（权重 scale 留到 VEC 施加）；`SetSingleShape` 按 tile 动态设置。
- **反量化 4 步（VEC）**：`Cast half→float` → `Muls(×16)` 还原 int4 到 int8 量级并 `Add` 高低半 → `Add(y_offset)` 加偏置 → `Mul(w_scale)` 逐通道权重反量化 → `BroadCast+Brcb(x_scale)` 逐 token scale → `Add(output_bias)` → `Cast→bf16`。`alignBaseN` 需为 64 的倍数（256B 对齐）。
- **乒乓 workspace**：每核 `parallNum=4` 路 ping-pong，`workspace = 16MB(系统) + 4 × usedCoreNum × baseM×baseN × 2B`，供 Cube 与 VEC 重叠 `mmOut` 中间结果。
- **tiling 自洽**：独立 tiling 类（不依赖 ops-nn `TilingBaseClass`），用 `matmul_tiling::MultiCoreMatmulTiling` 求 `baseM/baseN/baseK`，但 `usedCoreNum` 由 tile 计数直接覆写（V5 模式，因 API 可能固定返回 1）；`baseM≤120`、`baseN≤128`、`baseK≤64`，`M` 方按 `2M`（拆分后行数）规划。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| act | 必选 | int8 | `[M, K]` | `[tokens, in_features]` | 量化激活（对称量化）。Phase 1 被 AIV 就地拆分为 int4b_t（`[2M, K/2]`），拆分结果写回 `act` 自身，故输入张量需可写。`K` 必须是 32 的倍数。 |
| weight | 必选 | int4b_t（声明 int32 存储） | `[N, K]` | `[out_features, in_features]` | INT4 权重，ND 排布、`bTrans=true`（kernel 内做 `weight^T`）。`N` 必须 ≥ 16。 |
| scale | 必选 | float | `[N]` | `[out_features]` | 逐通道权重反量化 scale（`w_scale`）。在 VEC 阶段乘到 Cube 输出上；Cube 端用 `SetQuantScalar(1.0)` 恒等。 |
| bias | 必选 | float | `[N]` | `[out_features]` | 偏置，**语义为已在 host 端预乘 `w_scale`**，kernel 内作为 `y_offset` 直接相加（不再乘 scale）。 |
| x_scale | 必选 | float | `[M]` | `[tokens]` | 逐 token 激活反量化 scale。在 VEC 阶段 `BroadCast` 到 `[curVecBaseM, alignBaseN]` 后相乘。 |
| output_bias | 必选 | float | `[N]` | `[out_features]` | 输出偏置，在 `×x_scale` 之后、`Cast→bf16` 之前施加。必选输入，若无需输出偏置可传入全 0。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| out | 必选 | bfloat16 | `[M, N]` | `[tokens, out_features]` | 反量化后的 BF16 输出。`infershape` 取 `M=act.dim(0)`、`N=weight.dim(0)`；`dtype` 固定 BF16。 |

### 2.3 属性

本算子**无属性**（`op_def` 未声明任何 `Attr`）。`scale_value` 等均在输入张量中显式传入。

### 2.4 Tiling 数据结构

`QuantMatmulW4a8TilingData`（host 与 kernel 共享，`#pragma pack(8)` + `alignas(8)`），其中 `TCubeTiling matmulTiling` 为 CANN 标准 Cube tiling。

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint8_t | 实际使用核数（由 tile 计数覆写） |
| vBaseM | uint32_t | VEC 单次处理的 M 行数（`min(ubCalSize/baseN, baseM/2)`） |
| ubRestBytes | uint32_t | `tmpBuff` 字节数（`ubCalSize × sizeof(float) × 4`） |
| parallNum | uint32_t | Cube/VEC 乒乓路数（固定 4） |
| ubCalSize | uint32_t | UB 计算单元尺寸（固定 8192） |
| mSize | uint32_t | M（原始行数） |
| kSize | uint32_t | K（已按 32 上取整） |
| nSize | uint32_t | N |
| groupSize | uint32_t | 量化分组大小（K_C per-channel 模式下为 0） |
| matmulTiling | TCubeTiling | Cube 矩阵乘 tiling（baseM/baseN/baseK/stepM/stepN/Ka/Kb/usedCoreNum 等） |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 800I A2 | Atlas 800I A2 的 AIC+AIV 双核 MSD 模式。 |
| 输入数据类型 | act=int8；weight=int4b_t（int32 存储）；scale/bias/x_scale/output_bias=float | 固定类型组合。 |
| 输出数据类型 | out=bfloat16 | 固定 BF16 输出。 |
| 输入格式 | `FORMAT_ND` | weight 为 ND + `bTrans=true`；不支持 NZ 权重（`weightNz=false`，模板虽留 NZ 分支但本算子实例化为 ND）。 |
| K（in_features） | 必须为 32 的倍数 | `SetMatmulTiling` 中 `K % 32 != 0` 直接返回失败；否则按 32 上取整。 |
| N（out_features） | 必须 ≥ 16 | `N < 16` 时 tiling 失败；`baseN` 按 16 对齐（`bn = (bn/16)*16`）。 |
| M（tokens） | ≥ 1；`baseM` 受 UB 约束（`maxBm = (ubSize - 16*bn)/(12*bn)`，最小 2） | tiling 用 `2M`（拆分后行数）规划 M 方；`M` 方最后一个 tile 处理尾部。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)` + `DynamicRankSupportFlag(true)`。 |
| workspace | ≥ 16 MiB + 乒乓区 | `16MB(系统) + parallNum(4) × usedCoreNum × baseM×baseN × 2B`。 |
| 调度模式 | 独占核（`SetScheduleMode(1)`） | MSD 双核协同需独占核，`CrossCoreSetFlag` 才生效。 |
| 量化模式 | K_C（逐 token 叠加逐通道） | `QuantType::K_C`、`groupSize=0`；K_G（逐组）分支模板保留但本算子未使用。 |

### 3.2 不支持的场景

- **非 Atlas 800I A2 硬件**：Atlas 300I Duo 无 int4b_t Cube 与 AIC/AIV MSD 双核模式，不支持。
- **K 非 32 倍数**：tiling 直接 `GRAPH_FAILED`。
- **N < 16**：tiling 直接 `GRAPH_FAILED`（`baseN` 无法对齐）。
- **NZ 权重排布**：模板虽有 `weightNz=true` 分支，但本算子固定实例化为 `weightNz=false`（ND）；NZ 权重不被支持。
- **逐组量化（K_G）**：算子固定 `K_C`（逐通道）；`groupSize=0`，K_G 路径未启用。
- **可写性**：`act` 输入会被 in-place 拆分覆写，若上游张量只读/共享，需先拷贝副本。
- **训练/反向**：仅前向，无反向梯度。
- **其它 dtype**：不支持 FP16/FP32 激活或权重、不支持非对称量化偏置在 kernel 内乘 scale（`bias` 必须在 host 端预乘 `w_scale`）。

### 3.3 数值约束与精度说明

- **int4 还原**：拆分时低半 `Adds(-8)` 还原有符号；反量化时 `Muls(×16)` 把 int4 量级还原到 int8 量级，再合并高低半。
- **scale 施加顺序**：`mm_out → +bias(y_offset) → ×w_scale → ×x_scale → +output_bias → bf16`。`bias` 已含 `w_scale` 预乘，故先加 `bias` 再乘 `w_scale` 不会重复缩放 `bias`。
- **BF16 输出**：最终 `Cast(ROUND_RINT)` FP32→BF16。

## 4 参考资源

- **算子来源**：modified from CANN ops-nn `quant_batch_matmul_v4`（V5 MSD 实现）
  - kernel：<https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_kernel/quant_batch_matmul_v4.cpp>
  - tiling：<https://gitcode.com/cann/ops-nn/blob/master/matmul/quant_batch_matmul_v4/op_host/quant_batch_matmul_v4_tiling.cpp>
- **CANN 算子开发**：Ascend C 算子开发指南（`MatmulImpl`/`lib/matmul_intf.h`、`CrossCoreSetFlag`/`CrossCoreWaitFlag` 双核同步、`DataCopyPad`、`BroadCast`/`Brcb`），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
