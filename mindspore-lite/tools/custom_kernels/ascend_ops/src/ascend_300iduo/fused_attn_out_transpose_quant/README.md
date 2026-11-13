# FusedAttnOutTransposeQuant

## 1 功能说明

`FusedAttnOutTransposeQuant` 是注意力输出 head-major → token-major 折叠 + 逐通道 smooth + 逐 token 动态量化的纯 Vector（AICore）融合算子，面向昇腾 Atlas 300I Duo 上的 W8A8 推理场景。算子把注意力输出的本地尾部处理——两次 Transpose、Slice、smooth 乘与逐 token 动态量化——整体并入单 kernel（单设备契约：输入 `[H,M,D]`，输出 `[M,H·D]`），原链对同一份激活的 6 遍以上搬运压成单遍。

### 1.1 算子语义

给定 FA/PFA 的 head-major 输出 `attn`（前导维折叠进 `H`，`[B,H,S,D]` 亦可）与可选逐通道 smooth 缩放 `smooth_scale`，算子按下式输出 token-major 折叠的 W8A8 激活 `y`（即下游 GEMM 的 x1）与逐 token 反量化 scale `token_scale`：

```text
a[m, h*D+d] = attn[h, m, d] * smooth_scale[h*D+d]   # 无 smooth 时 a = attn 位等价
                                                       # （折叠是纯布局操作，不触碰数值）
absmax[m]     = max_j |a[m, j]|                       # 逐 token、跨 head 跨 lane 归约
token_scale[m] = absmax[m] / 127                      # 300IDUO DynamicQuant 输出约定
y[m, j]       = round_int8( a[m, j] * 127 / absmax[m] )   # absmax ≤ 1e-12 时整行置 0
```

其中 `m` 为 token 行、`h` 为 head、`d` 为 head 维；`y` 的行宽 `n = H·D`，即折叠后的隐藏宽。

### 1.2 实现要点

- **2D DataCopyParams 直折装载（性能核心）**：每 token 行 1 条描述符（blockCount=H、blockLen=8×32B=256B、srcGap=(m-1)·8、dstGap=0，blockLen/gap 单位 32B），R 条描述符即把 Transpose#1/#2 + Slice + TransData 的多段 gather 拼接为一次落 token-major fold 位的 DMA；相比每 head 1 条 plain 拷贝，描述符发射数降 8×（H 条 → R 条/块）。m > 8192 时 uint16 srcGap 溢出，自动回退 plain 每 head 拷贝路径（精度等价、发射数变多）。
- **absmax 原生 3 阶段归约**：复用 CANN 生产内核 `quant_matmul_dequant` 在本 arch 的逐 token absmax 归约形状——Abs 全 tile → 逐行 `vcgmax`（counter 模式，每 16 元素 32B 块压 1 个 max 到源前缀；n > 2048 时再一阶 vcgmax）→ 逐行 `vcmax`（norm 位掩码 ≤128 lanes、`ORDER_VALUE_INDEX`，落 `[value,index]` 对取偶 lane）。相比 count 形对折树（活 lane 每折减半，12 折 @n=4096 ≈ 11.5 遍死数据 ≈ 72% 向量浪费），原生归约 ≈ 1.07 遍，向量管时长约降 63%。
- **8 行组条带分核**：token_scale 的 fp32 `[M]` 输出按整 8 行组刷出（整组可合并为一条 32B/64B MTE3 拷贝）；末组不足 8 行时按行用 4B 标量写，不越出 `[M]` 逻辑范围；rowBlock R ∈ {16,8,4,2,1}（16/8 为整组块，4/2/1 整除 8，块边界不跨组）。
- **零 barrier 执行模型**：装载用 MTE2_V Set/Wait 消费点即同步（零 barrier 装载）；量化尾（逐行 Duplicate scaleStage + fp16 Muls + CAST_ROUND）零块内 barrier——V pipe 按序执行，同 pipe 相邻依赖无需 barrier；仅 absmax 归约的 mask 态切换处放 PipeBarrier（V 指令执行时才读掩码寄存器）。
- **MTE3 节流与预取**：每 fence ≤ 2 次拷贝（y 整块一次 + 完整 8 行组 scale 一次，300IDUO copy-loss 定律：≤4 描述符/fence）；handback 后立即预取下一块 fold 拷贝，使其在 absmax/量化/MTE3 尾部下排水。
- **handback 位置分档**：有 smooth 时在 ApplySmooth 后立即释放 fold 缓冲并预取（其最后读者是 ApplySmooth）；无 smooth 时 fold 缓冲就是 act 本体，handback 推迟到量化循环后。
- **折叠不用 TransDataTo5HD/interleave 的原因**：本 arch 的 `TransDataTo5HD` 是指针表转置（永远交换 stream×position），无法拼接 head；`[D,16]` interleave 对跨 head 跨 lane 的逐 token归约无益，且 int8 输出不能过 vnchwconv、输出必须 natural，徒增 9 次/行的转置调用。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| attn | 必选 | float16 | `[H, M, D]` | `[head, token, head_dim]` | FA/PFA 的 head-major 输出；前导维折叠进 H（`[B,H,S,D]` 亦可，行主且前导积 = H 为图契约，kernel 不做 shape 校验）。D 必须为 128（见 3.1）。 |
| smooth_scale | 可选 | float16 | `[H*D]` | `[folded_dim]` | 逐通道 smooth 缩放（折叠维度上每元素一个系数）。缺省视为全 1（a = attn 位等价）。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | int8 | `[M, H*D]` | `[token, folded_dim]` | token-major 折叠的 W8A8 激活（下游 GEMM 的 x1）。infershape 由 attn `[..., M, D]` 推导。`AutoContiguous` 保证连续存储。 |
| token_scale | 必选 | float32 | `[M]` | `[token]` | 逐 token 反量化 scale = absmax/127，与 300IDUO DynamicQuant 输出约定一致。 |

### 2.3 属性

本算子无属性。

### 2.4 Tiling 数据结构

host 侧 `FusedAttnOutTransposeQuantTilingData` 与 device 侧镜像结构体（`op_host/fused_attn_out_transpose_quant_tiling.h` 与 `op_kernel/fused_attn_out_transpose_quant_tiling_data.h`），二者必须**字节一致**（32B）且位于全局命名空间（CANN tiling 宏要求），`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际启动的 block dim（8 行组数与 AIV 核数取小） |
| m | uint32_t | 展平后的 token 行数 M |
| h | uint32_t | head 数 H（attn 前导维之积） |
| d | uint32_t | head 维 D（固定 kHeadDim=128，tiling 硬校验） |
| n | uint32_t | 行宽 n = H·D |
| rowBlock | uint32_t | 每块行数 R ∈ {16,8,4,2,1}（按平台 UB 预算择优：min(平台 UB, 200KB)） |
| hasSmooth | uint32_t | 是否提供可选输入 smooth_scale（1/0） |
| fold2d | uint32_t | 1 = 2D DataCopyParams 折叠装载（m ≤ 8192 @D=128）；0 = plain 每 head 拷贝回退 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo（dav_m200，def 按 `ascend300IDUO` 注册） | 原生 `vcgmax`/`vcmax` 归约依赖 dav_m200 微架构。其它 SoC 未适配。 |
| 输入数据类型 | attn/smooth_scale = float16 | 数值路径与被替换链逐算子对齐（见 3.3）。 |
| 输出数据类型 | y = int8；token_scale = float32 | 固定类型组合。 |
| 输入/输出格式 | `FORMAT_ND`（全端口） | def 中 ND + `DynamicFormatFlag(true)` + `DynamicShapeSupportFlag(true)`；`DynamicRankSupportFlag` 关闭（要求已知的维度数）。 |
| D（head 维） | 固定 128 | 2D 装载的 256B（=D×2B）切片游程保证每条拷贝与行基址 32B 对齐；D=64 时奇数 head 切片步距破坏对齐、D=256 时非仿射切片步距（除非 R=1）。tiling 显式拒绝其它 D。 |
| M（token 数） | ≤ 8192 走 2D 折叠装载 | uint16 srcGap 上限 `(m-1)·8 ≤ 65535`；超过自动回退 plain 每 head 拷贝路径（精度等价、发射数变多）。 |
| n（行宽 H·D） | n ≤ 32768 且（n ≤ 2048 或 n % 256 == 0） | 原生归约形状界：二阶 vcgmax 需 16 倍数 maxima 数、vcmax 位掩码 ≤128 lanes。已验证 head 数 1/2/8/32（n=128/256/1024/4096）均可通过；其余由 tiling 显式拒绝。 |
| 布局契约 | attn 行主 `[..., M, D]` 且前导积 = H | 图契约（FA 输出）；kernel 不做 shape 校验。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`；tiling 运行时按实际 shape 反解 R 与 fold2d。 |
| workspace | 0 | 无 device workspace，attn 直接流式读入。 |

### 3.2 不支持的场景

- **`D != 128`**：tiling 显式拒绝。其它 head 维需 interleave/deinterleave 机制（未实现）。
- **超界 n**（n > 32768，或 n > 2048 且 n % 256 != 0）：无树回退路径，算子直接不可用；新增形状需先过 tiling 界检查。
- **混合精度/任意 dtype**：仅支持上述固定类型组合，不支持 bfloat16 / float32 / int8 输入。
- **训练/反向**：仅推理前向（纯 Vector AICore 任务），不含反向梯度。

### 3.3 数值约束与精度说明

- **无 smooth 时位等价**：折叠是纯布局操作，a = attn 逐位一致。
- **smooth 数值路径**：fp16 vector Mul（fp32 内部精度、单次 fp16 舍入），与链中 fp16 Mul 数值等价（位等价门禁验证）。
- **absmax 精确性**：fp16 max 精确且无序，原生归约与树归约算术完全相同；逐行 counter ≤ 4096（tile 级 counter 会挂死）。
- **量化尾同构**：fp16 Muls + CAST_ROUND 与 300IDUO DynamicQuant 内核逐行处理同构（量化误差门限：int8 ±1、scale 相对误差 < 5e-3）。
- **极小 absmax 行为**：`absmax ≤ 1e-12` 时 `y` 整行置 0；`absmax < 127/65504` 时 `(half)(127/absmax)` 溢出为 inf（与 300IDUO DynamicQuant 同样的暴露，正常激活幅度下无影响）。
- **vcgmax/vcmax 语义边界**：counter 模式延展单 repeat（逐行 counter ≤ 4096）、终阶取 `ORDER_VALUE_INDEX` 的 `[value,index]` 对偶数 lane（`ORDER_ONLY_VALUE` 在 arch 2002 非法）、stride 元组沿用生产内核形状。该语义为同 arch 生产内核源码考古结论（无官方文档），CANN 版本升级后必须重跑精度门禁。
- **mask 态纪律**：V 指令执行时才读掩码寄存器，相位切换处 barrier 按 CANN 生产内核同款布置，编译器升级需复验。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Vector 原语、`DataCopyParams` 2D 拷贝、`vcgmax`/`vcmax` 归约、事件与 barrier 模型），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
- **CANN 内置算子语义**：300IDUO `DynamicQuant`（token_scale = absmax/127 输出约定）、生产内核 `quant_matmul_dequant`（本 arch 逐 token absmax 原生归约形状实证）。
