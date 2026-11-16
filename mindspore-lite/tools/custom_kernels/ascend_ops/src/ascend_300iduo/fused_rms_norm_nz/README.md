# FusedRmsNormNZ

## 1 功能说明

`FusedRmsNormNZ` 是面向 MindSpore Lite 昇腾 Atlas 300I Duo 推理场景的单核 RMSNorm 融合算子，**FP16 边界 + FP32 统计、FRACTAL_NZ 原生**。

算子定义文件头部（[fused_rms_norm_nz.cpp](op_host/fused_rms_norm_nz.cpp)）说明了融合动机：CANN 内置 RmsNorm 是 fp16-ND 语义（拒绝 fp16 统计），不满足 FP32 统计 + FRACTAL_NZ 数据的需求，只能分解为 6 内核行归一链（Square+ReduceMean、Add(eps)、RealDiv、Mul、Mul_1、Cast）；分解还会把上游 GEMM 的 epilogue 拖到 f32（双倍写带宽）并多付 6 次 kernel launch。

- **NZ-only 单格式注册**（gelu 先例）：部署位点（GEMM epilogue → norm → GEMM）全 NZ，算子零 TransData 直连；ND 生产者经 GE 自动插入的 TransData 连接。tiling 自动判定 NZ/ND 存储，内核保留 ND 平坦分支为防御（NZ-only 注册下不可达）。
- **FP32 统计红线**：统计与归一乘全在 UB 内以 fp32 进行，出口单次 fp16 舍入。

### 1.1 算子语义

逐行：

```text
y = x * rsqrt(mean(x^2) + epsilon) * gamma
```

- `x` fp16 `[M..., N]` FRACTAL_NZ（上游 GEMM 输出），`gamma` fp16 ND `[N]`（逐道缩放），`y` 与 x 同形同格式。
- 统计在 UB 内以 fp32 累积（平方取 f32(x)——与链的 Square on Cast(x) 完全同值），出口单次 fp16 舍入——与链的舍入次数一致。
- `epsilon` 属性（可选，默认 1e-6）加在 mean(x²) 上。

### 1.2 实现要点

- **NZ 行寻址**：逻辑 [P…, M, N] fp16 的 NZ 存储为 [P…, n1, m1t, 16, 16]（列分形在外、16×16 分形内 [m0, n0] 行主序；fp16 的 C0=16）。行 r 的数据 = N/16 段 16 元素（32B），段距 m1t·512B；行基址 `(p·n1·m1t + m1)·256 + m0·16`。
- **gather/scatter**：每行每块 ONE 次 strided DataCopyParams 拷贝——load `blockCount=len/16, blockLen=1, srcStride=m1t·16−1, dstStride=0`，store 反向。**blockLen 与 stride 的单位是 32 字节不是元素**。UB 内即得行连续槽位，后续全部是平坦行主序向量作业，无任何布局体操。ND 路径为平凡平坦拷贝（store 整块一次 MTE3）。
- **行块执行模型（v2，整行驻留）**：R 行（rowBlock）一块，**一行全部 N 列驻留 UB**（无 colChunk 切列）：x f16 [R,n] + work f32 [R,n] + tmp f32 [R,n] + y f16 [R,n] = 12·R·n 字节 + 三个 [R,8] fp32 槽位 tile（96R）+ gamma 表（6n）。每块：MTE2 批量 gather（每行 ONE 次 strided DataCopy）→ pass0（f32 视图 Cast + 平方 Mul，一次平坦 count 覆盖整块）→ **归约梯子（v3）** → 逐行 lane 折叠（奇活数先折尾）→ count-1 累加进 32B 跨距槽位 → **批量 invRms 向量链**（slot tile 上 Muls(1/n)+Adds(eps)+Sqrt+Reciprocal seed + 两轮 Newton rsqrt 精炼，15 条指令/块；pad lane 落在 v=eps 不动点，从不读取）→ 单次 V_S 读回 → pass2（驻留行 ×invRms ×gs，乘法顺序同链，仅相邻依赖对加栅）→ 批量 MTE3（**drain 延迟到下一块 pass2**）+ 下一块 gather 在 pass0 Cast 后即预取（MTE2/MTE3 与向量全程重叠）。
- **v2 相对 v1 的关键差异**：(1) 去 colChunk——v1 每行按列块分 C 次进出同步（2048 列 C=2），v2 整行一次驻留；(2) 平坦树替代逐 chunk 树——树指令数从 O(C·log) 降为 O(log)，且跨行垃圾尾免置零；(3) MTE3 延迟 drain + pass0 后预取，MTE2/MTE3 与向量全程重叠（v1 串行段更多）。实测内核 396→222 µs@2048²（1.79×），位等价精度不变。
- **归约梯子（v3）**：`BlockReduceSum`（原生 vcgadd，产品 softmax 同款整 tile in-place 形态：repeatTime=1 + counter mask + srcRepStride=8）替换 v2 平坦垃圾尾减半树：每级把 tile 内每 8 元组累加落到前 1/8 槽位、per-row span 逐级 ÷8（n=2048：2048→256→32→4 三级；n=1536：1536→192→24→3；n=4096：4096→512→64→8→1，落点 1 = 免 lane 折叠）。归约元素 112.6k→16.4k/块（−85%，跨行垃圾尾消失），lane 折叠从落点 redLand 继续（v2 折叠序列泛化）；n%8≠0（tiling 契约外）回退 v2 树路径。梯子计划（级数/落点）由内核 Init 从 n_ 推导，**tiling 数据布局不变**；counter 上限 R·n ≤ 16384 沿用（n=2048 时 14336）。实测稳态内核 244.9→176.5 µs@2048²（−28%），vec pipe 占比 0.86→0.56。
- **零块内 V barrier**：同 pipe RAW 链 + count 型指令自括号（gelu v5 / fused_qk_norm_rope v3 / BSH v2 定律），归约后算术链 15 道 + pass2 逐行 14 道 PipeBarrier 全拆（MTE↔V 握手 ~4 次/块保留）；梯子自身的 mask count 状态切换保留保守 bracket（SetMaskCount 前置 + 每级后一道 + SetMaskNorm/ResetMask 收尾，n=2048 为 4 道/块）。
- **不使用 ReduceSum**（repeatTimes ≤ 64 时 Level-2 内部 V_S→标量→S_V 隐藏同步）。
- host 侧 tiling（[fused_rms_norm_nz_tiling.cpp](op_host/fused_rms_norm_nz_tiling.cpp)）：NZ/ND 经存储维判定（末二维 16,16 且 dim[d−4]·16 == N vs 末维 == N，gamma 长度交叉校验）；R 候选 {16,12,8,7,6,4,3,2,1} 按 UB 镜像 `BlockBytes = R·(12n+96)+6n+512 ≤ 196KB` 与平坦 count 上限 `R·n ≤ 16384` 枚举取最大（与内核 ClampRowBlock 同式防御钳制）；行块条带分核；pad 行（m1t·16 > 逻辑 M）garbage-in-garbage-out（gelu 同立场），pad 列被 N%32 契约排除。**N 上界 11116**（18·N+96+512 ≤ 196KB，整行驻留所致；N 为 32 的倍数，实际最大 11104；v1 colChunk 可切更大 N，v2 对超限 N 直接 GRAPH_FAILED 拒绝）。infershape（[fused_rms_norm_nz_infershape.cpp](op_host/fused_rms_norm_nz_infershape.cpp)）y 镜像 x 维度、dtype fp16。kernel 入口 [op_kernel/fused_rms_norm_nz.cpp](op_kernel/fused_rms_norm_nz.cpp)。

## 2 参数说明

> 说明：真实维度为张量在 GM 中的实际存储形状；逻辑维度为算子语义上的轴含义。`M...` 为任意前导维（乘积记 m），`N` 为行长度（== gamma 长度），`R` 为行块行数；NZ 存储维 `[P..., n1, m1t, 16, 16]` 中 `n1 = N/16`、`m1t = ceil(M/16)`。

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| x | 必选 | float16 | FRACTAL_NZ（[M..., N] 的分形存储 [P…, n1, m1t, 16, 16]；ND 生产者经 GE 插 TransData） | [M..., N] | 上游 GEMM 输出。逻辑尾维必须 == N；tiling 按存储维自动判定 NZ/ND（isNz）。 |
| gamma | 必选 | float16 | ND `[N]` | `[N]` | 逐道缩放权重；长度即 N（tiling 交叉校验）。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | float16 | FRACTAL_NZ，与 x 同形 | [M..., N] | infershape 镜像 x 全部维度，dtype fp16，AutoContiguous；存储格式随 x（NZ/ND）。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| epsilon | 可选 | float | 1e-6 | RMSNorm epsilon，加在 mean(x²) 上。 |

### 2.4 Tiling 数据结构

host 侧 `FusedRmsNormNZTilingData`（[fused_rms_norm_nz_tiling.h](op_host/fused_rms_norm_nz_tiling.h)，与 [op_kernel/fused_rms_norm_nz_tiling_data.h](op_kernel/fused_rms_norm_nz_tiling_data.h) 字节镜像；必须驻留全局命名空间，CANN tiling 宏要求）：

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际 LaunchBlockDim（行块条带分核） |
| m | uint32_t | 展平行数（NZ：prefix·m1t·16，含 pad 行） |
| n | uint32_t | 行长 N（== gamma 长度，32 的倍数） |
| colChunk | uint32_t | 保留字段：kernel 不使用，tiling 写入 n；仅用于保持 tiling 结构体字节布局不变。 |
| rowBlock | uint32_t | 每块行数 R（1..16，按 UB/counter 上限枚举取最大） |
| isNz | uint32_t | 1 = x/y 为 FRACTAL_NZ 存储，0 = ND |
| epsilon | float | RMSNorm epsilon |
| m1t | uint32_t | NZ 分形行块数（存储维 d−3）；ND 为 0 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | def 仅注册 Atlas 300I Duo AICore 配置。 |
| 数据类型 | x/gamma/y = float16 | FP16 边界；统计在内核 fp32。 |
| x/y 格式 | FRACTAL_NZ（注册口径）；kernel 兼容 ND 存储防御分支 | NZ-only 单格式注册（CANN 8.5.0 opbuild 不支持多值 Format 列表）；部署位点全 NZ 零 TransData。 |
| N | % 32 == 0 且 ≥ 32，≤ 11116（32 的倍数，实际最大 11104） | dav_m200 无 DataCopyPad，所有拷贝/NZ 段 32B 对齐（如 N=2048 ✓）；上界为整行驻留 18·N+96+512 ≤ 196KB。 |
| M | NZ 时 m1t·16−1 ≤ 65535（M ≤ 65536） | DataCopyParams stride 为 uint16（32B 块单位）；tiling 拒绝越界。 |
| N/16 | ≤ 65535 | 整行 scatter 的 blockCount 上限。 |
| gamma | 恒 [N] fp16 ND | 图契约；kernel 不做 shape 校验（tiling 已交叉校验）。 |
| x 逻辑尾维 | == N | 行归一图契约。 |
| 前导维 | 任意 | DynamicRankSupport；m 为前导维乘积（NZ 下按存储维展开，含 pad 行）。 |
| 平坦 count | R·n ≤ 16384 | counter 模式上限，超限自动降 R。 |

### 3.2 不支持的场景

- **N % 32 ≠ 0 或 N < 32**：tiling GRAPH_FAILED（32B 拷贝/NZ 段对齐）。
- **N > 11116**：v2 整行驻留 UB 放不下，tiling 直接拒绝（v1 colChunk 可切更大 N；超大 N 场景用 v1 存量或回退）。
- **NZ 时 M > 65536 或 N/16 > 65535**：DataCopyParams uint16 stride / blockCount 越界，tiling 拒绝。
- **x 逻辑尾维 ≠ N（gamma 长度）**：行归一图契约，tiling 交叉校验拒绝。
- **动态 shape（-1 档）**：tiling 依赖具体 m/n 数值。
- **非 fp16 / gamma 非 ND**：图契约。
- **非 Atlas 300I Duo 平台**：无编译产物。

### 3.3 数值/精度说明

- **fp32 统计红线**：统计与归一乘全在 UB 内 fp32（平方取 f32(x)，与链的 Square on Cast(x) 完全同值）；出口单次 fp16 舍入，与链的舍入次数一致——这正是 fp32 统计红线的直接原因。
- **invRms 精修**：Sqrt+Reciprocal seed + 两轮 Newton rsqrt 精炼（批量向量链）；pad lane 落在 v=eps 不动点，从不读取。
- **v3 归约结合序**：BlockReduceSum 的 8-ary 结合序 ≠ v2 减半树，非 v2 位等价，精度与 v2 同 ≤1 ULP 类。
- **乘法顺序**：pass2 驻留行 ×invRms ×gs，乘法顺序同链（(x·invRms)·gamma）。
- **pad 行口径**：m1t·16 > 逻辑 M 的 pad 行 garbage-in-garbage-out（gelu 同立场）；pad 列被 N%32 契约排除。

## 4 参考资源

- **CANN 算子开发指南**（Ascend C 算子开发，BlockReduceSum/vcgadd、counter mask、NZ 分形存储）
- **CANN 内置 RmsNorm**：fp16-ND 语义（本算子存在的原因——需要 fp32 统计 + NZ 数据）。
- **本仓库同族算子**：fused_gelu_tanh_nz（NZ 单格式注册与 MTE3 延迟 drain 先例）、chunk_gated_delta_rule（strided DataCopy 32B 口径先例）。
