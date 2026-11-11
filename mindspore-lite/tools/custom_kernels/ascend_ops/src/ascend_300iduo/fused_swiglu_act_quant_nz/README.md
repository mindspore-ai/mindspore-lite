# FusedSwiGLUActQuantNZ

## 1 功能说明

`FusedSwiGLUActQuantNZ` 是 FFN SwiGLU 激活 + 逐通道 smooth + 逐 token 动态量化的纯 Vector（AICore）融合算子，面向昇腾 Atlas 300I Duo 上的 W8A8 推理场景。算子直读两个 QuantBatchMatmulV3 输出（gate/up 分支），将每层 FFN 尾部的 `ConcatD + SwiGlu + Mul(smooth) + DynamicQuant + TransData` 五算子链压成单 kernel，宽拼接中间量不再存在。`NZ` 后缀名实相符：gate/up/y 三个 GEMM 侧端口为 FRACTAL_NZ 注册（2026-09-20 实测定稿，见 3.1），QBMV3 的 NZ 输出直连、`y` 以 QBMV3 原生布局直出，站点上 3 个 TransData 全部消除。

### 1.1 算子语义

给定 gate 分支 `gate`、up 分支 `up`（均为 QBMV3 的 NZ 输出，前导维展平为 `M` 行）与可选逐通道 smooth 缩放 `smooth_scale`，算子按下式输出 W8A8 激活 `y` 与逐 token 反量化 scale `token_scale`：

```text
act[m,j]      = silu(gate[m,j]) * up[m,j] * smooth_scale[j]   # silu 作用于 gate 分支
                                                                # （与 CANN te_swiglu 一致：
                                                                #   sigma(a) -> a*sigma(a) -> *b）
absmax[m]     = max_j |act[m,j]|                                # 逐 token 绝对值最大
token_scale[m] = absmax[m] / 127                                # 300IDUO DynamicQuant 输出约定
y[m,j]        = round_int8( act[m,j] * 127 / absmax[m] )        # absmax ≤ 1e-12 时整行置 0
```

### 1.2 实现要点

- **NZ 端口注册（实测采纳）**：CANN 8.5.0 opbuild 每端口只允许单一格式，gate/up/y 按 FRACTAL_NZ 注册（gelu/rms_norm_nz 先例）。部署站点（QBMV3 → swiglu → QBMV3，两侧边均为 NZ）零 TransData 直连；内核内分形 gather/scatter 与 Vector pipe 重叠，优于独立 TransData 图算子在关键路径上串行——「慢搬移但藏得住」胜过「快搬移但串行」。
- **silu 头部 FP32 + 尾链 FP16**：头部 `fp16 取负（精确）→ Cast↑f32（精确）→ Exp<f32>`；尾链 `e16=RN(e32) → d16=1+e16 → r16=vrec(d16) → s16=g·r → t16=s·u → act16=t·smooth` 全 fp16。链共 9 拍（有 smooth）/ 8 拍（无 smooth）。
- **cc==n 单块行融合**：单列块档整块 `[R,n]` 的 gate/up tile 与 act 行切片均连续，整块并作一次宽拍集跑链（9 拍 × R·n 宽）——逐元素算术不变，与逐行链逐位一致，发射/barrier 费从 9×R 拍/块降到 9 拍/块；乱尾单行块走逐行链。
- **cc<n 乒乓双缓冲**：多列块档 gate/up 双对 tile 乒乓，下一列块的拷入先于当前块的链发出，拷入排空隐藏在链的 Vector 工作之下。
- **非 2 幂树 fold-to-1 absmax**：行缓冲 fp16 Abs + 原地折半 Max 树，span=n 起折到 1（奇数档悬尾元素折入槽 0，无需 2 幂补零暂存），每块一次 V_S、每行 1 次 GetValue（手写树避开 ReduceMax Level-2 的内部标量同步）。
- **8 行组条带分核**：fp32 `[M]` 输出按 8 行组刷出，整组为一次 32B DataCopy、GM 偏移 32B 对齐；最后一组不足 8 行时按行用 4B 粒度 GM 标量写，不越出 `[M]` 逻辑范围。行不相交，无需跨核同步（无 atomics）。
- **`vrec<half>` 快速倒数**：dav_m200 唯一原生倒数指令（约 1 指令单位），设备统计标定为快速倒数（ε≈9.8e-4 ≈ 2 ulp fp16，非 RN 精确值）；被替换链的 `Reciprocal<f32>` 同为快速近似实现，语义等价层级一致。
- **tiling 字段防御**：`rowBlock` 非法（0、超过上限或不整除 8 行组）时 clamp 为 1，避免块循环不推进；退化到逐行路径，结果仍正确。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| gate | 必选 | float16 | `[M..., N]`（FRACTAL_NZ） | `[token..., inter]` | gate 分支（QBMV3 输出 1）。N 即 inter 维。前导维由 tiling 展平为 M。NZ 存储解读：fp16 `[N/16, M/16, 16, 16]` fractal，段 512B。需与 up 同形（图契约，kernel 不做 shape 校验）。 |
| up | 必选 | float16 | `[M..., N]`（FRACTAL_NZ） | `[token..., inter]` | up 分支（QBMV3 输出 2），与 gate 共享布局。 |
| smooth_scale | 可选 | float16 | `[N]` | `[inter]` | 逐通道 smooth 缩放（ND）。缺省视为全 1，跳过 smooth 乘法分支。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | int8 | `[M..., N]`（FRACTAL_NZ） | `[token..., inter]` | W8A8 激活，以 QuantBatchMatmulV3 x1 原生布局直出，直连下游 QBMV3。形状镜像 gate（infershape）。`AutoContiguous` 保证连续存储。 |
| token_scale | 必选 | float32 | `[M]` | `[token]` | 逐 token 反量化 scale = absmax/127，与 300IDUO DynamicQuant 输出约定一致，匹配 QBMV3 的 ND scale 输入。 |

### 2.3 属性

本算子无属性。

### 2.4 Tiling 数据结构

host 侧 `FusedSwiGLUActQuantNZTilingData` 与 device 侧镜像结构体（`op_host/fused_swi_glu_act_quant_nz_tiling.h` 与 `op_kernel/fused_swi_glu_act_quant_nz_tiling_data.h`），二者必须**字节一致**（32B）且位于全局命名空间（CANN tiling 宏要求），`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| coreNum | uint32_t | 实际启动的 block dim（8 行组数与 AIV 核数取小） |
| m | uint32_t | 展平后的逻辑行数 M（NZ pad 行不计入，token_scale 不得写越界） |
| n | uint32_t | 每分支列数 N（inter 维，32 的倍数） |
| colChunk | uint32_t | 每 UB chunk 列数 cc（32 的倍数，host 侧反解；cc=n 走单块融合路径，cc<n 走乒乓逐行路径） |
| hasSmooth | uint32_t | 是否提供可选输入 smooth_scale（1/0） |
| rowBlock | uint32_t | 每块行数 R ∈ {8,4,2,1}（整除 8 行组） |
| isNz | uint32_t | 1 = gate/up/y 为 FRACTAL_NZ 存储（当前 def 恒为 1；0 为 ND 解码路径） |
| m1t | uint32_t | NZ 每 16 行块数 ≤ 4096（uint16 stride 上限 m1t·16-1 ≤ 65535 约束） |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo（dav_m200，def 按 `ascend300IDUO` 注册） | 其它 SoC 未适配。 |
| 输入数据类型 | gate/up/smooth_scale = float16 | silu 头 FP32、尾链 FP16 的数值路径见 3.3。 |
| 输出数据类型 | y = int8；token_scale = float32 | 固定类型组合。 |
| 端口格式 | gate/up/y = `FORMAT_FRACTAL_NZ`；smooth_scale/token_scale = `FORMAT_ND` | NZ 注册为实测决策：精度与 ND 版 6/6 逐位一致（快照位比 + 3× 重复执行位比），msprof task 口径 NZ kernel 慢于 ND kernel（分形 gather/scatter 代价），但消除站点 3 个 TransData 后净收益为正（内核内搬移与 Vector pipe 重叠 vs 独立 TransData 串行）。def 开启 `DynamicFormatFlag(true)` 与 `DynamicShapeSupportFlag(true)`；`DynamicRankSupportFlag` 关闭（本算子要求已知 rank）。 |
| N（inter 维） | `N % 32 == 0` 且 `N ≥ 32` | dav_m200 无 DataCopyPad，fp16 入 / int8 出的每列块拷贝与行基址需 32B 对齐，tiling 显式拒绝非法 N。常见 inter 维（6144/12288/2048…）均满足。 |
| 形状契约 | gate/up 同形 `[M..., N]` | 图契约（QBMV3 双输出）；kernel 不做 shape 校验。 |
| 上游生产者 | gate/up 须来自 QuantBatchMatmulV3（NZ 原生输出） | ND 生产者接入时 GE 自动插入 TransData（付 3 个 TD 代价）；smooth_scale/token_scale 侧无此约束。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)`；tiling 运行时按实际 shape 反解 (R, cc)。 |
| workspace | 0 | 无 device workspace，行数据直接从 QBMV3 输出流式读入。 |

### 3.2 不支持的场景

- **`N % 32 != 0`**：tiling 显式拒绝（dav_m200 无 DataCopyPad，32B 对齐不可满足）。
- **ND 上游接入**：当前 def 为 NZ-only 注册（CANN 8.5.0 opbuild 每端口单格式）；上游为 ND 生产者时只能经 GE 插入的 TransData 接入，无零 TransData 通道。
- **混合精度/任意 dtype**：仅支持上述固定类型组合，不支持 bfloat16 / float32 / int8 输入。
- **训练/反向**：仅推理前向（纯 Vector AICore 任务），不含反向梯度。

### 3.3 数值约束与精度说明

- **silu 头部 FP32 精确段**：fp16 取负与 Cast↑f32 均精确，`Exp<f32>` 与被替换链同 pipe，e32 与 fp32 路径逐位相同。
- **尾链 FP16 有界偏差**：相比 fp32 尾链，act 前多 3 次 fp16 舍入（e16/s16/t16）+ `vrec<half>` 快速倒数（ε≈9.8e-4 ≈ 2 ulp fp16）替代 `Reciprocal<f32>`（后者同为快速近似，探针 ε 最高 ~2.8e-3）。同输入设备探针：99.3–99.5% 位等价，翻转全部 |d|=1 且 ± 对称，无 |d|≥2，token_scale 相对偏移 ≤ 5.3e-3；act 尾误差典型 ~2e-3、最坏 ~4.5e-3，距 int8 翻转阈 1.2e-2（|y|=127 处 1.5/127）裕度约 2.7×。精度门限（int8 ±1 且 scale 相对误差 < 5e-3）全部用例通过。
- **absmax 精确性**：fp16 Abs + 折半 Max 树——max 精确且无序，与 ReduceMax 同结果；悬尾元素折入槽 0，非 2 幂 N 无需补零。
- **量化尾同构**：fp16 Muls + CAST_ROUND 与 300IDUO DynamicQuant 内核逐行处理完全同构，epsilon=1e-12 同 CANN。
- **极小 absmax 行为**：`absmax ≤ 1e-12` 时 `y` 整行置 0；`absmax < 127/65504` 时 `(half)(127/absmax)` 溢出为 inf（与 300IDUO DynamicQuant 同样的暴露，正常激活幅度下无影响）。
- **公式语义对齐**：silu 作用于 gate 分支，与 CANN `te_swiglu`（`swi_glu_single.hpp` 设 beta=-1.0，序列 `σ(a)→a·σ(a)→·b`）一致。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Vector 原语、`DataCopyParams` 2D 拷贝、FRACTAL_NZ 布局、事件与 barrier 模型），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
- **CANN 内置算子语义**：`te_swiglu`（silu 位置与分支角色）、300IDUO `DynamicQuant`（token_scale = absmax/127 输出约定）。
