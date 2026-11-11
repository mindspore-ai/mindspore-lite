# FusedAddRmsNormSmoothQuant

## 1 功能说明

`FusedAddRmsNormSmoothQuant` 是残差加 + RMSNorm + 逐通道 smooth + 逐 token 动态量化的纯 Vector（AICore）融合算子，面向昇腾 Atlas 300I Duo 上的 W8A8 推理场景。算子将 layer-norm 站点的 `Add + RmsNorm + Mul(smooth) + DynamicQuant + TransData` 五算子访存受限链压缩为单 kernel，残差 `r = x1 + x2` 直写输出，并顺带消除下游 gate/up GEMM 的重复量化。算子名保留 `NZ` 后缀，但端口按 ND 注册（NZ 注册实测净负，见 3.2）。

### 1.1 算子语义

给定残差 carry-in `x1`、子层输出 `x2`、RMSNorm 权重 `gamma`、可选逐通道 smooth 缩放 `smooth_scale`，前导维展平为 `M` 行，算子按下式输出残差 `res_out`、W8A8 激活 `y` 与逐 token 反量化 scale `token_scale`：

```text
r[m,j]        = x1[m,j] + x2[m,j]                  # 残差，直写 res_out
rms[m]        = sqrt(mean_j(r[m,:]^2) + epsilon)   # RMSNorm 分母
n[m,j]        = r[m,j] / rms[m] * gamma[j] * smooth_scale[j]
absmax[m]     = max_j |n[m,j]|                     # 逐 token 绝对值最大
token_scale[m] = absmax[m] / 127                   # 300IDUO DynamicQuant 输出约定
y[m,j]        = round_int8( n[m,j] * 127 / absmax[m] )   # absmax ≤ 1e-12 时整行置 0
```

与 CANN 内置 `AddRmsNormQuantV2` 的区别：后者为静态 per-channel scales 语义（scales 为 REQUIRED fp32 常量），与逐 token 动态量化（absmax/127）不匹配，且无 smooth_scale 融入、无双输出喂 gate/up GEMM 的通道，故本算子保持动态量化语义手写实现。

### 1.2 实现要点

- **残差直写双输出**：`r` 在第一遍算出后驻留 UB 行缓冲，`res_out` 与量化激活共用同一次拷入，原链 Add+RmsNorm 的 5 遍数据访问压成 3 遍（读 x1/x2 + 写 res_out/y）。
- **8 行组条带分核**：fp32 `[M]` 输出按 8 行一组写出，每次 DataCopy 恰为一个 32B 块、GM 偏移 32B 对齐；最后一组不足 8 行时整组写出，越界部分落在 32B 粒度分配的 padding 内。行不相交，无需跨核同步（无 atomics）。
- **整行驻留 UB**：逐 token absmax 必须看完整行才能产出任何 `y` 元素，故 `r` 与归一化激活两个 fp16 行缓冲（各 2H B/行）常驻 UB（预算 200KB）。常见 H 档：H=2048→R=8、H=4096→R=4、H=8192→R=1，均单 chunk（C=1）整行驻留，减半树每行只需跑一遍。
- **`gs` 预融合**：`gs[j] = gamma[j]·smooth_scale[j]` 每核算一次常驻 UB（fp16 乘 → fp32 常驻），Init 时借尚未启用的 x-tile 暂存拷入，不设专用缓冲。
- **三遍行块流水**：块内按 (rowBlock R ∈ {8,4,2,1}, colChunk C) 切块，pass1 批量拷入并算行平方和 → invRms 相位（Muls/Adds/Sqrt/Reciprocal 4 条指令批量覆盖块内全部行槽）→ pass2 归一化落 fp16 → absmax 减半树 → pass3 量化尾 + 整块 MTE3 拷出并预取下一块。
- **零块内 V barrier**：同 pipe 相邻依赖指令间的 PipeBarrier 是纯 scalar 发射开销（V 指令队列按序执行），整块 V 区是一条按序指令流，仅在两处 V→标量回读（invRms、absmax）前放 barrier（约 3 个/块）。
- **手动减半树替代 ReduceSum/ReduceMax**：Level-2 reduce 在 count ≤ 4096 fp32 时走内部标量同步路径，逐行调用的同步开销淹没访存收益；改用行平方和折 8 段再折标量、`|act|` 折半树收 8 lane，每块仅一次 V_S 回读。
- **端口保持 ND 注册**：NZ 注册变体（x1/x2/res_out/y 四 GEMM 侧端口 FRACTAL_NZ）已实现并 6/6 位等价验证，但 msprof task 口径实测全部形状净负（NZ 分形 gather/scatter 增量 > 站点可消 TransData），def 保持 ND；kernel/tiling 的 NZ 分支保留为已验证死代码（见 3.2）。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| x1 | 必选 | float16 | `[M..., H]` | `[token..., hidden]` | 残差 carry-in（上一 norm 站点的 res_out 或 embedding）。前导维由 tiling 展平为 M，支持 `[M,H]` / `[B,S,H]` 等 row-major 形状。 |
| x2 | 必选 | float16 | `[M..., H]` | `[token..., hidden]` | 子层输出。上游 QBMV3 的 NZ 输出经 GE 自动插入的 NZ→ND TransData 进入本端口。需与 x1 同形（图契约，kernel 不做 shape 校验）。 |
| gamma | 必选 | float16 | `[H]` | `[hidden]` | RMSNorm 权重。与 smooth_scale 每核预融合为 `gs` 常驻 UB。 |
| smooth_scale | 可选 | float16 | `[H]` | `[hidden]` | 逐通道 smooth 缩放。缺省视为全 1（`gs = gamma`）。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| res_out | 必选 | float16 | `[M..., H]` | `[token..., hidden]` | 残差 `r = x1 + x2`，一次写出，喂下一站点 x1。形状镜像 x1（infershape）。`AutoContiguous` 保证连续存储。 |
| y | 必选 | int8 | `[M..., H]` | `[token..., hidden]` | W8A8 激活。喂 QuantBatchMatmulV3 时由 GE 自动插入 int8 ND→NZ TransData。 |
| token_scale | 必选 | float32 | `[M]` | `[token]` | 逐 token 反量化 scale = absmax/127，与 300IDUO DynamicQuant 输出约定一致，匹配 QBMV3 的 ND scale 输入。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| epsilon | 可选 | float | 1e-6 | RMSNorm epsilon，加在均方后开方前。 |

### 2.4 Tiling 数据结构

host 侧 `FusedAddRmsNormSmoothQuantTilingData` 与 device 侧镜像结构体（`op_host/fused_add_rms_norm_smooth_quant_tiling.h` 与 `op_kernel/fused_add_rms_norm_smooth_quant_tiling_data.h`），二者必须**字节一致**（32B）且位于全局命名空间（CANN tiling 宏要求），`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际启动的 block dim（8 行组数与 AIV 核数取小） |
| m | uint32_t | 展平后的逻辑行数 M（NZ pad 行不计入，token_scale 不得写越界） |
| n | uint32_t | 列数 H（hidden） |
| colChunk | uint32_t | 每列块列数 C（32 的倍数） |
| rowBlock | uint32_t | 每块行数 R ∈ {8,4,2,1}（整除 8 行组） |
| hasSmooth | uint32_t | 是否提供可选输入 smooth_scale（1/0） |
| epsilon | float | RMSNorm epsilon |
| nzInfo | uint32_t | NZ 布局位打包（bit0 `isNz`：x1/x2/res_out/y 是否 FRACTAL_NZ 存储；bits1..12 `m1t`：NZ 每 16 行块数 ≤ 4096，受 uint16 stride 上限约束）。保持结构体 32B；0 值解码为 ND。 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo（dav_m200，def 按 `ascend310 P` 注册） | 其它 SoC 未适配。 |
| 输入数据类型 | x1/x2/gamma/smooth_scale = float16 | 数值路径与被替换算子链逐算子对齐（见 3.3）。 |
| 输出数据类型 | res_out = float16；y = int8；token_scale = float32 | 固定类型组合。 |
| 输入/输出格式 | `FORMAT_ND`（全端口） | def 中 ND + `DynamicFormatFlag(true)` + `DynamicShapeSupportFlag(true)`；`DynamicRankSupportFlag` 关闭（要求已知的维度数）；H 由 tiling 按实际 shape 反解 R/C。 |
| H（hidden） | `H % 32 == 0` 且 `H ≥ 32` | dav_m200 无 DataCopyPad，fp16 入 / fp16 与 int8 出的每列块拷贝与行基址需 32B 对齐，tiling 显式拒绝非法 H。常见 hidden（4096/2048…）均满足。 |
| H 上限 | H ≤ 8192 单 chunk 整行驻留（R=1）；H=16384 退回 C=8 多 chunk（仍正确，未调优） | 整行驻留受 200KB UB 预算约束。 |
| 形状契约 | x1/x2 同形 `[M..., H]`；gamma/smooth_scale 为 `[H]` | 图契约；kernel 不做 shape 校验。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`。 |
| workspace | 0 | 无 device workspace，x1/x2 直接流式读入。 |

### 3.2 不支持的场景

- **NZ 端口注册**：四 GEMM 侧端口改 FRACTAL_NZ 的变体已实现并 6/6 位等价验证（快照位比 + 3× 重复执行位比全部逐位一致），但 msprof task 口径实测全部形状净负：NZ kernel 的分形 gather/scatter 代价（mte2 87→569 µs、mte3 22→523 µs @1536×4096，散射 store ~30 GB/s）超过站点仅有的 2 个可消 TransData（res_out 在 ND 世界本无 TransData 对位物，NZ 化反而白付散射代价）。故 def 全端口保持 ND，kernel/tiling 的 NZ 分支为已验证的不可达死代码；若未来解开 MTE3 散射瓶颈，将四端口翻回 FRACTAL_NZ 即可零内核改动重启零 TransData 链。
- **`H % 32 != 0`**：tiling 显式拒绝（dav_m200 无 DataCopyPad，32B 对齐不可满足）。
- **混合精度/任意 dtype**：仅支持上述固定类型组合，不支持 bfloat16 / int8 / float32 输入。
- **训练/反向**：仅推理前向（纯 Vector AICore 任务），不含反向梯度。

### 3.3 数值约束与精度说明

- **`r` 的 fp16 直接 Add 位等价**：两个 fp16 的精确和 fp32 可精确表示，单条 Add 指令只舍入一次到 fp16，与被替换的 fp16 Add 输出**逐位一致**（实测门禁：全部精度用例 × 独立运行逐位通过）。该等价要求加法器 IEEE round-to-nearest-even，换硬件首跑必须重过 res_out 位等价检查。
- **`rms` 输入口径**：对舍入后的 fp16 `r` 求 fp32 平方和（与链中 rmsnorm 看到的输入一致），减半树折 8 段再折标量。
- **`n` 少一次舍入**：链中 rmsnorm、smooth-mul 各舍入一次，本算子 `fp32(r)·invRms·gs` 落 fp16 一次舍入，差 < 1 ULP fp16，int8 容差内不可见。
- **量化尾同构**：absmax（fp16 Abs + 减半树）与量化尾（fp16 Muls + CAST_ROUND）与 300IDUO DynamicQuant 内核逐行处理完全同构，epsilon=1e-12 同 CANN。
- **极小 absmax 行为**：`absmax ≤ 1e-12` 时 `y` 整行置 0；`absmax < 127/65504` 时 `(half)(127/absmax)` 溢出为 inf（与 300IDUO DynamicQuant 同样的暴露，正常激活幅度下无影响）。
- **零块内 barrier 依赖 V pipe 按序执行**：整块 V 区按序指令流零 barrier 依赖该硬件行为，已由多次独立精度运行验证确定性；若未来硬件/编译器改动发射顺序，现象将是精度偶发出错，排查时优先回读该假设。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Vector 原语、`DataCopy`/`Muls`/`Adds`/`Sqrt`/`Reciprocal`/`Cast`、事件与 barrier 模型），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
- **CANN 内置算子语义**：`AddRmsNormQuantV2`（静态 per-channel scales，本算子未对接的原因）、300IDUO `DynamicQuant`（token_scale = absmax/127 输出约定）。
