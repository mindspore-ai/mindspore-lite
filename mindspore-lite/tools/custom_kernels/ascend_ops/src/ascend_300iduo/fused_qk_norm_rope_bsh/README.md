# FusedQKNormRopeBSH

## 1 功能说明

`FusedQKNormRopeBSH` 是面向 MindSpore Lite 昇腾 Atlas 300I Duo 推理场景的 qk_norm（逐头 RMSNorm）+ NeoX RoPE 融合算子，**仅支持 FP16、全端口 ND 注册**，处理 BSH 布局扁平行 q/k，纯 vector 内核（AICore 任务）。

算子定义文件头部（[fused_qk_norm_rope_bsh.cpp](op_host/fused_qk_norm_rope_bsh.cpp)）说明了融合动机：分解态 qk-norm+rope 链是 18 算子 fp32 链。相比 BNSD 变体 fused_qk_norm_rope 的差异：

- 无 v 路径（attention 直接以 BSH 消费 q/k）；
- q/k 以 [M, H·D] 扁平行进出、出同行（无 BNSD 转置、无去交织转置输出）；
- q|k 头组按行在内核内拼接，rms 归约/norm 广播/gamma 缩放全部以平坦连续向量算子跑在一个 [rows, G, 16·D] tile 上。

全端口 ND 注册：部署位点生产者/消费者均 ND 直连，零 TransData。D 由 `len(gamma_q)` 判定；tiling 交叉校验 q/k/cos/sin 的 numel 契约，不匹配 loud 拒绝（GRAPH_FAILED），无静默错数据。

### 1.1 算子语义

对 q/k 各自的每个头 h 与行 m（x 代表对应头切片）：

```text
# 1. 逐头 RMSNorm（统计 fp32）
rms = sqrt(mean_d(x^2) + epsilon)            # epsilon 属性，默认 1e-6
xn  = x * rsqrt(mean_d(x^2) + epsilon) * gamma[d]

# 2. NeoX rotate-half（cos/sin 为预计算半表 [D/2]）
out[0:D/2] = xn1*cos - xn2*sin
out[D/2:D] = xn2*cos + xn1*sin
```

其中 `M...` 为任意前导维（乘积记 m，token 扁平行）、`Hq`/`Hkv` 为 q/k 头数、`D` 为每头维度（由 gamma_q 长度判定）、`gamma` 为逐道 norm 权重、`cos/sin` 为 rotary 半表。q 与 k 各自独立做 norm+RoPE。数值与 fused_qk_norm_rope 对齐：fp32 norm 累积、raw Sqrt/Reciprocal 链、只在最末端做一次 fp16 舍入（优于网络链的 fp16 中间乘 x*invRms）。

### 1.2 实现要点

- **拼接 tile**：每行 q|k 的 16 头组 tile 拼接，slot s = r·G+g 落 s·span（span = 16·D，G = ceil(Hq/16)+ceil(Hkv/16)）——rms 归约/norm 广播/gamma 缩放全部以平坦连续向量算子跑在 [rows·G·span] 一块 tile 上。
- **交织视图 + 逐 slot 减半树（v2 核心）**：每 slot scatter_vnchwconv 交织成 [D, 16] fp32 视图（p = d·16 + lane，**lane = 头**）；树按**逐 slot** 折叠（`for h = span/2 → kGroupHeads, h >>= 1: for s: Add(sq[s·span], sq[s·span], sq[s·span+h], h)`）——批量折叠每 fold 计算整 span 长度（垃圾落死亡区，55312 元素/块），逐 slot 树只算 live 部分 **8128 元素/块（−39% fp32 vec 工作）**。位等价的构造性论证：fold h 读 `[s·span, +h) ∪ [s·span+h, +h)`，两段恰好是 fold 2h live 区写过的值——逐 slot 树与批量树每个 live 输出看到**相同的 (p, p+h) 输入对序列**（归纳可证），算术逐位一致；实测 10/10 max_ulp 与批量版逐对相等。
- **invRms 链**：每 slot 4-op count=16（Muls(1/D)+Adds(eps)+Sqrt+Reciprocal）相位批处理。
- **norm 广播**：每 slot 2 次 stride-0 Mul（偶 32B 块读 invRms 块 0、奇块读块 1）；**gamma**：每行 ONE plain Mul over G·span vs 拼接 [G, span] fp32 表（[gamEq×Gq | gamEk×Gk]，每核建一次）。
- **RoPE**：每 (row, group) 6 条 plain 算子（A·cos、B·sin、Sub、B·cos、A·sin、Add）vs 每行 [D/2, 16] Brcb16 cos/sin 展开表（B 半复用）。
- **dummy 头**：H%16≠0 的尾组 dummy lane 垃圾严格 in-lane，MTE3 按整 BSH 行（hq·d / hkv·d 连续）写出，dummy 不出内核。
- **同步模型（v2）：零块内 V PipeBarrier**（v1 27 → 0）——V 管道按发射序执行，同管道 RAW/WAR/WAW 链不需要 PipeBarrier（含 repeat-form Brcb 与 TransDataTo5HD 指针表散射先例）；跨管道事件全部保留：E0(MTE2_V)/E1/E2(V_MTE3/MTE3_V 四旗定律)/E3(V_MTE2 预取 arm) 严格 1:1 set/wait。
- **MTE 模式**：MTE2 每块一次发全部（cos/sin 行 + 全部 (row,group) nat tile）+ 单次 drain；MTE3 **v2 分侧合并**：out16 重排为 [q 区 | k 区]（q slot 落前 rows·qG·span、k slot 落其后），q 侧 `hq·d` 为 16 倍数时 ONE 大 copy（主形状 q=16 头 ✓），k 侧 hkv=8 乱尾 → per-row chunk-4 回退；格挡按 300IDUO MTE3 law（>4 未格挡 DMA 描述符丢拷贝；定律限制的是描述符数不是字节数，大 copy 安全）。MTE2-in 目标与 MTE3-out 源不相交 → 无 MTE3→MTE2 序。
- **rotary 表**：repeatTimes 为 uint8，按行分块使 rows·D/2/8 ≤ 255。
- host 侧 tiling（[fused_qk_norm_rope_bsh_tiling.cpp](op_host/fused_qk_norm_rope_bsh_tiling.cpp)）：R 候选 {8,4,2,1}（UB 镜像精确到字节：PerRow = 224·G·d+66·d，Fixed = 164·d+64·G·d+512，R·PerRow+Fixed ≤ 200KB）+ 平坦 count 上限 R·G·span ≤ 16384（本芯片已验证 counter 8192，16384 为绝对上限）；行块条带分核；部署形状 (2048,16,8,128) 取 R=2（G=2），(2048,32,8,128) 取 R=1（G=3）。infershape（[fused_qk_norm_rope_bsh_infershape.cpp](op_host/fused_qk_norm_rope_bsh_infershape.cpp)）q_out/k_out 分别镜像 q/k 维度、dtype fp16。kernel 入口 [op_kernel/fused_qk_norm_rope_bsh.cpp](op_kernel/fused_qk_norm_rope_bsh.cpp)。

## 2 参数说明

> 说明：真实维度为张量在 GM 中的实际存储形状；逻辑维度为算子语义上的轴含义。`M...` 为任意前导维（乘积记 m），`Hq`/`Hkv` 为 q/k 头数，`D` 为每头维度（2 幂，32..128），`G = ceil(Hq/16)+ceil(Hkv/16)` 为每行拼接头组数，`span = 16·D`，`R` 为行块行数。

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| q | 必选 | float16 | ND `[M..., Hq·D]` | `[m..., hq·d]`（BSH 扁平行，头主序） | 查询张量。前导维任意；尾维需被 D 整除（D 由 gamma_q 长度判定）。 |
| k | 必选 | float16 | ND `[M..., Hkv·D]` | `[m..., hkv·d]` | 键张量。与 q 独立 norm+RoPE；展平 numel 需与 q 的 m·hkv·d 契约一致。 |
| gamma_q | 必选 | float16 | ND `[D]` | `[d]` | qk_norm 的 q 逐道权重；其长度即 D。 |
| gamma_k | 必选 | float16 | ND `[D]` | `[d]` | qk_norm 的 k 逐道权重；需与 gamma_q 等长。 |
| cos | 必选 | float16 | ND `[M..., D/2]` | `[m..., d/2]` | 预计算 rotary cos 半表（NeoX rotate-half 半表形态）。 |
| sin | 必选 | float16 | ND `[M..., D/2]` | `[m..., d/2]` | 预计算 rotary sin 半表。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| q_out | 必选 | float16 | ND `[M..., Hq·D]`，镜像 q | 同 q | norm+RoPE 后的 q；infershape 镜像 q 全部维度，dtype fp16，AutoContiguous。 |
| k_out | 必选 | float16 | ND `[M..., Hkv·D]`，镜像 k | 同 k | norm+RoPE 后的 k；infershape 镜像 k 全部维度。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| epsilon | 可选 | float | 1e-6 | RMSNorm epsilon，加在 mean(x²) 上。 |

### 2.4 Tiling 数据结构

host 侧 `FusedQKNormRopeBSHTilingData`（[fused_qk_norm_rope_bsh_tiling.h](op_host/fused_qk_norm_rope_bsh_tiling.h)，与 [op_kernel/fused_qk_norm_rope_bsh_tiling_data.h](op_kernel/fused_qk_norm_rope_bsh_tiling_data.h) 字节镜像；40 字节布局与 fused_qk_norm_rope 字段集一致，跨算子 tiling 审计可 diff）：

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际 LaunchBlockDim（行块条带分核） |
| m | uint32_t | 展平 token 行数（q 前导维乘积） |
| hq / hkv | uint32_t | q / k 头数 |
| d | uint32_t | 每头维度（2 幂，32..128，由 gamma_q 判定） |
| rowBlock | uint32_t | 每块行数 R（候选 {8,4,2,1} 按 UB/counter 上限取最大） |
| epsilon | float | RMSNorm epsilon |
| invD | float | host 预计算 1/D |
| reserved0 / reserved1 | uint32_t ×2 | 保留 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | def 仅注册 Atlas 300I Duo AICore 配置。 |
| 数据类型 | 全端口 float16 | 图契约。 |
| 格式 | 全端口 ND | 部署位点生产者/消费者均 ND，零 TransData。 |
| D | 2 幂且 ∈ [32, 128]（部署 128） | <32 破坏 32B 对齐拷贝/Brcb 源；D=256 时 G=2 的 fp32 拼接驻留在 R=1 已 207KB > 200KB。 |
| Hq / Hkv | 任意（16 头/组） | 尾组 dummy 不出内核；H%16≠0 有废向量功（部署 16/32 整除无影响）。 |
| M | 任意 | 行块条带分核；repeatTimes uint8 由 rotary 表分块满足。 |
| 平坦 count | R·G·span ≤ 16384 | counter 模式绝对上限，超限自动降 R。 |
| 前导维 | 任意 | DynamicRankSupport；m 为前导维乘积。 |

### 3.2 不支持的场景

- **D 非 2 幂、D < 32 或 D > 128**：tiling GRAPH_FAILED（32B 对齐/UB 预算）。
- **q/k 尾维不被 D 整除，或 k/cos/sin 展平 numel 与 m·hkv·d / m·(D/2) 契约不匹配**：tiling loud 拒绝（GRAPH_FAILED），无静默错数据。
- **gamma_k 长度 ≠ gamma_q 长度**：tiling 拒绝。
- **所有 R 候选均超 UB/counter 上限的形状**：tiling 拒绝。
- **动态 shape（-1 档）**：tiling 依赖具体维数值。
- **非 fp16 / 非 ND 端口**：图契约；kernel 不做 shape 校验（tiling 已交叉校验）。
- **非 Atlas 300I Duo 平台**：无编译产物。

### 3.3 数值/精度说明

- **fp32 norm 累积**：rms 统计在 fp32（fp32 统计红线），raw Sqrt/Reciprocal 链（与 fused_qk_norm_rope 数值对齐）。
- **单次 fp16 舍入**：只在最末端舍入一次，优于网络链（链在 fp16 做 x*invRms 中间乘）。
- **逐 slot 减半树位等价**：与批量减半树每个 live 输出看到相同 (p, p+h) 输入对序列（归纳可证），实测 10/10 max_ulp 逐对相等。
- **RoPE 半表**：cos/sin 为 NeoX rotate-half 半表 [D/2]，内核展开成 [D/2, 16] Brcb 表，B 半复用。

## 4 参考资源

- **CANN 算子开发指南**（Ascend C 算子开发，Vector 原语、Brcb/scatter_vnchwconv、事件通道
- **fused_qk_norm_rope（BNSD 变体）**：本算子的数值与同步定律来源（团队算子库）。
