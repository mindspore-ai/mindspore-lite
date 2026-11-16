# FusedGeluTanhNZ

## 1 功能说明

`FusedGeluTanhNZ` 是面向 MindSpore Lite 昇腾 Atlas 300I Duo 推理场景的 tanh 近似 Gelu（`gelu_pytorch_tanh`）单核融合算子，**仅支持 FP16 输入/输出、FRACTAL_NZ 单格式注册**。

算子定义文件头部（[fused_gelu_tanh_nz.cpp](op_host/fused_gelu_tanh_nz.cpp)）说明了融合动机：导出侧把 tanh 式标量物化为 fp32 常数后，转换器把链中张量升为 fp32，部署态分解链每块跑 10 个内核、3 次 dtype 翻转、2 次 TransData 重排，GM 流量 ~1105MB/块，而单遍只需 67MB。

- **NZ-only 单格式注册**：部署位点（BatchMatMul → 激活 → BatchMatMul）全 NZ，算子零 TransData 直连；ND 图中 GE 自动在两侧插 TransData（仅布局、保位）。单格式同时是本仓库 CANN 8.5.0 工具链的硬约束——Format 列表会同时打断 aclnn autogen 与 kernel opbuild。
- **布局无关扁平遍历**：纯元素级映射，kernel 不感知格式，按展平元素数分 chunk 流式处理。
- **无属性、无 workspace、无 optional 输入**；前导维任意（DynamicRankSupport，tiling 只看展平元素数）。

### 1.1 算子语义

```text
y = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3)))
```

- 与 `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))` **数学恒等**（tanh(y)=2/(1+e^(−2y))−1 代入）；CANN `GeluV2(tanh)` TBE 实现与 `gelu_quant_base.h::ComputeGeluTanh` 用同款 exp 重排式与同款常数 −√(8/π)。
- 常数 fp32 常驻：kBeta=0.044715f、kNegAlpha=−1.5957691216057308f（fp16 段的 Muls 内联 cast，fp32 段直接用）。
- `x` 为 fp16 `[M..., N]` FRACTAL_NZ（上游输出），`y` 为同形 fp16 FRACTAL_NZ（下游输入）；逐元素映射，无归约轴。

### 1.2 实现要点

- **混合 dtype 算术链**：多项式段（x²→x³→·β→+x）与末乘在 fp16（原生 ALU 指令，2B/tensor），特殊函数段（·(−√(8/π))→Exp→+1→Reciprocal）在 fp32（原生指令；**fp16 vexp/vdiv 在 dav_m200 是编译器仿真、实测净变慢**），两次 Cast 衔接，72B/元素 UB 流量（全 fp32 链 100B；分解链 GM 流量放大 1105MB/67MB=16.5×）。
- **布局无关扁平遍历**：NZ storage（[L,nBlk,mBlk,16,16] 稠密行主序）与 ND storage 都是稠密 [totalElems] 缓冲，x/y 同布局则扁平位置一一对应——kernel 无需感知格式，直接按 chunk 流式处理。totalElems 取 **GetStorageShape() 全维乘积**（不是 OriginShape）：这正是 kernel 实际行走的缓冲大小，NZ 的 16 对齐 padding 也被覆盖（padding 上的结果无人下游读取）。
- **256 元素对齐契约**：chunk 边界全部是 256（一个 16×16 fp16 fractal）的倍数，dav_m200 无 DataCopyPad，32B 对齐的 DataCopy 由此保证；tiling 显式拒绝 totalElems % 256 ≠ 0。
- **chunk 条带分核**：`[blockIdx*totalChunks/coreNum, (blockIdx+1)*…)` 整数分界，核间 chunk 不相交、无 atomics；1024×4096 → 342 chunk / 8 核（12288 元素/chunk）。
- **UB 预算**：2×x fp16（2×2B）+ 2×y fp16（2×2B）+ tmp fp16 2B + tmp fp32 4B + 提升到循环外的常量张量 fp16 2B = **16B/元素 × 12288 = 196608B = 192KB**（dav_m200 单核 UB 上限）；x/y 双缓冲让 MTE2 预取 chunk i+1、MTE3 排空 chunk i−1 与 V 计算 chunk i 重叠。
- **禁止 xH_ 原地复用**：把输出 staging 复用 xH_（fp32 cast 后即死，理论上安全）实测每核第 1 个 chunk 正确、之后全错——`WaitFlag<MTE3_V>` 只序 V 流水线，下一迭代的 MTE2 拷入是 MTE2 队列指令，任何 V 侧 wait 都管不住它；必须独立 yH_ + 四握手分区。
- **事件通道 FIFO 配对**：四通道（MTE2_V / V_MTE3 / MTE3_V / V_MTE2）各 1:1 set/wait、跨迭代 FIFO——copy-in(i)↔V(i)、V(i)↔copy-out(i)、copy-out(i)↔V(i+2)（yH_[i%2] 可重写）、V(i)↔copy-in(i+2)（xH_[i%2] 可重写）；收尾 drain min(nChunks,2) 次保证通道平衡。每个 MTE3 burst 一个 1D DataCopy 自带 gate（300IDUO MTE3 copy-loss 律：每 four-flag gate ≤4 拷、全 1D、不在队首），首个 MTE3 工作在首个计算之后（队首非 store）。
- **逐算子保守 PipeBarrier**：相邻混合吞吐量向量指令非 RAW 安全需 drain（与 CANN 官方 gelu_quant_base.h 同纪律），实测瓶颈不在 barrier。
- **KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AICORE)**：300IDUO 纯 vector 内核以统一 AICore 任务下发。
- host 侧 tiling（[fused_gelu_tanh_nz_tiling.cpp](op_host/fused_gelu_tanh_nz_tiling.cpp)）：totalElems = 全部 storage 维乘积；chunkElems = min(12288, totalElems)（256 对齐钳制）；chunk 条带分核，blockDim = min(totalChunks, aivNum)；workspace = 0；192KB UB 预算 static_assert，另按平台上报的 UB 容量对 chunk 足迹在 tiling 内运行时复检（不足即 GRAPH_FAILED，不让 InitBuffer 拿到越界张量）。infershape（[fused_gelu_tanh_nz_infershape.cpp](op_host/fused_gelu_tanh_nz_infershape.cpp)）y 镜像 x 维度、dtype fp16。kernel 入口 [op_kernel/fused_gelu_tanh_nz.cpp](op_kernel/fused_gelu_tanh_nz.cpp)。

## 2 参数说明

> 说明：真实维度为张量在 GM 中的实际存储形状；逻辑维度为算子语义上的轴含义。`M...` 为任意前导维（乘积记 m），`N` 为尾维；`totalElems` = storage 维乘积。

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| x | 必选 | float16 | FRACTAL_NZ（[M..., N] 的 16×16 分形存储） | [M..., N] | 上游输出。前导维任意（DynamicRankSupport），tiling 只看展平元素数 totalElems，不感知分形结构；声明 AutoContiguous，kernel 按扁平流访问，不接受非连续视图。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | float16 | FRACTAL_NZ，与 x 同形 | [M..., N] | 下游输入。infershape 镜像 x 全部维度，dtype fp16，AutoContiguous。 |

### 2.3 属性

无属性。

### 2.4 Tiling 数据结构

host 侧 `FusedGeluTanhNZTilingData`（[fused_gelu_tanh_nz_tiling.h](op_host/fused_gelu_tanh_nz_tiling.h)，与 [op_kernel/fused_gelu_tanh_nz_tiling_data.h](op_kernel/fused_gelu_tanh_nz_tiling_data.h) 字节镜像；必须驻留全局命名空间，CANN tiling 宏要求；32 字节，host/kernel 两侧 static_assert 锁定）：

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际 LaunchBlockDim = min(totalChunks, aivNum) |
| totalElems | uint64_t | 展平元素数（storage 维乘积；64 位，乘积溢出即拒绝） |
| chunkElems | uint32_t | 每 chunk 元素数（256 的倍数，≤ 12288，UB 预算钳制） |
| reserved0~2 | uint32_t ×3 | 保留，补齐 32 字节布局 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | def 仅注册 Atlas 300I Duo 配置。 |
| 输入/输出数据类型 | float16 | FP16-only；x/y 均为 fp16。 |
| 输入/输出格式 | FRACTAL_NZ（单格式注册） | 部署位点全 NZ 零 TransData；ND 图 GE 自动两侧插 TransData（仅布局、保位）。 |
| totalElems | % 256 == 0 且乘积不溢出 uint64 | NZ fractal 整块 + 32B DataCopy 对齐（tiling 显式拒绝）；4096×4096 / 1024×4096 等形状均满足。 |
| x/y 形状格式 | 同形同格式 | 图契约；kernel 不做 shape 校验。 |
| 前导维 | 任意（含一维） | DynamicRankSupport；tiling 只看展平元素数。 |
| workspace | 0 | 全部驻留 UB。 |

### 3.2 不支持的场景

- **非 fp16 输入/输出**：图契约限定 fp16（分解链的 fp32 中段正是本算子要消除的）。
- **动态 shape（-1 档）**：tiling 依赖具体 totalElems，直接拒绝。
- **totalElems % 256 ≠ 0 或 storage 维乘积溢出 uint64**：tiling GRAPH_FAILED（NZ fractal 整块/32B 拷贝对齐、64 位计数不容忍截断）。
- **x/y 不同形或不同格式**：图契约，kernel 不做 shape 校验（错配属图错误）。
- **非 Atlas 300I Duo 平台**：无编译产物。

### 3.3 数值/精度说明

- **数值安全域**：fp32 尾段下 x³ @ x=±65504 ≈ 2.8e14 远在 fp32 范围内；fp16 多项式段在 |x|>40.3 处 x³ 溢出 inf，但 inf 传播（u=±inf→z=∓∞→E=0/∞）与全 fp32 饱和路径落到**同值**（E floor 2⁻⁹ → y=x/1.002）——实测五形状与全 fp32 链版本**逐值位等价**，模拟逐值命中。
- **精度边界（平台行为）**：300IDUO VEC fp32 Exp 是**输出侧 floor**——真值 < 2⁻⁹ 的结果一律返回 ~2⁻⁹（隐含 e≈0.002 在 z=−36.8…−2×10¹³ 全程恒定，输入 clamp 无法规避），正尾 x≳2.9（exp 重排式 z<−6.2）处 y = x/(1+2⁻⁹)，相对偏差 0.195% → 折合输出量级**恰好 4 个 fp16 ULP**（0.00195×2¹¹ = 4.0；EDGE 档 x=65504 实测 |d| = 128 = 4×ULP(65504) 精确触界）。CANN GeluV2(tanh)/SwiGlu/gelu_quant 用同一 exp 重排式（同款 floor），属平台行为而非本算子缺陷；AscendC 8.5.0 无 Tanh/Sigmoid/Exp2 指令（头文件已核），无法绕开。bench 门禁据此定 4×ULP。
- **舍入次数**：fp16 多项式段 + fp32 特殊函数段的混合链，出口单次 fp16 舍入；实测与全 fp32 链版本逐值位等价（分解链则多跑 3 次 dtype 翻转及对应舍入）。

## 4 参考资源

- **CANN 算子开发指南**（Ascend C 算子开发，Vector 原语、UB 流水/事件通道）
- **CANN 内置 GeluV2（approximate="tanh"）与 gelu_quant_base.h::ComputeGeluTanh**：同款 exp 重排式与常数的官方先例。
- **本仓库同族算子**：fused_rms_norm_nz（NZ 单格式注册先例）、chunk_gated_delta_rule（strided DataCopy 32B 口径先例）。
