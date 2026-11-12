# FusedCausalConvSilu

## 1 功能说明

`FusedCausalConvSilu` 是 Gated DeltaNet（GDN，Qwen3.5 混合模型中的线性注意力层）short-conv 的前向算子，面向昇腾 Atlas 800I A2 推理场景：一次 kernel 完成 **depthwise 因果 Conv1d（K=4）+ bias + SiLU**，同时把最后 4 个输入 tap 作为 `conv_state` 返回，供 decode 阶段续接。

### 1.1 算子语义

输入 `x` 维度为 `[B, T, S]`（BTS 布局），卷积权重 `weight` 为 `[K, C]`，`bias` 为 `[C]`，`seq_len` 为 int32 向量（第 0 个元素是真实有效行数）。对每个输出通道 `c`、每个 token `t`：

```text
# 因果卷积：t 时刻只看 x[t-3 .. t]，t<0 的历史补 0
conv[t][c] = bias[c] + Σ_{k=0}^{3} weight[k][c] * x[t - 3 + k][c]

# 两次 bf16 舍入：卷积结果先落回 bf16（与框架 conv1d 的输出 dtype 一致），
# 再在 FP32 中计算 SiLU 并舍入输出
y[t][c]    = bf16( bf16(conv[t][c]) * sigmoid( bf16(conv[t][c]) ) )

# decode 续接所需的状态：真实序列末尾 4 行“原始输入”（SiLU 之前）
conv_state[j][c] = x[max(real_len - 4 + j, 0)][c],  j = 0..3
```

只有 `x` 的前 `C` 个通道参与计算；`S ≥ C` 时尾部通道被忽略（融合后 `mixed_qkv` 的布局对齐会带出多余列）。`conv_state` 存的是**输入行**而不是卷积结果：decode 的下一个 token 会把它与当前 token 拼成新的 4 tap 窗口再送进同一个算子（或 `causal_conv1d_update`）。

### 1.2 实现要点

- **单算子替代三段 host 路径**：框架侧是 `conv1d(x) → +bias → F.silu`，再单独从 `x` 里切出 4 行更新 conv cache；每层 linear-attention 每次 prefill 都要付两次 kernel launch、一次完整 UB 往返和若干 view/copy。本算子把卷积、bias、SiLU 与 state 提取合并在一个 AIV kernel 内完成。
- **AIV-only 向量实现，无 Cube、无 workspace**：卷积是 depthwise（逐通道）4-tap 乘加，天然是向量运算；整图没有矩阵乘，因此 kernel 声明 `KERNEL_TYPE_AIV_ONLY`，tiling 直接把 workspace 置 0。
- **通道 × token 二维分块**：通道方向按 `TILE_C=512` 切 tile，token 方向按 `TR=16` 行切 chunk；每个 work item 负责一个 `(channel tile, token chunk)`，按 `u += cores` 轮询分给所有 AIV 核。chunk 需要前 3 行的 halo，因此每个 item 载入 `rows + 3` 行窗口（满 chunk 为 19 行）。
- **UB 单缓冲、手工偏移**：权重（bf16/fp32 各一份）、bias、4-tap 累加器、SiLU 中间量（`exp/sigmoid` 的分母、分子）、输出行、输入窗口、`conv_state` 行全部共用同一个 `TBuf<VECCALC>`，在 `InitScratch` 里一次性按存活区间排布偏移，避免多 buffer 的容量膨胀。窗口行按 bf16 与 fp32 双份存放，bf16 行只在载入时使用。
- **4 tap 展开为逐 tap Mul + Add**：每个输出行只依赖窗口里的 4 行，直接 4 次 `Mul`（tap × 权重）与 4 次 `Add` 累加，不做通用卷积展开；累加器与权重均为 FP32。
- **两次低精度舍入对齐框架语义**：卷积 + bias 的 FP32 结果先 `CAST_RINT` 到 bf16 再转回 FP32，然后才计算 SiLU（`exp(-y) → 1+exp → 1/(1+exp) → y*sigmoid`），最后再 `CAST_RINT` 到 bf16 输出。这复刻了框架“`conv1d` 返回 bf16 张量、`F.silu` 在其上计算”的数值顺序，而不是把整个链路留在 FP32。
- **`conv_state` 只写一次**：每个通道 tile 有多个 token chunk，但 state 只由 `chunk == 0` 的 work item 写，按 `real_len` clamp 后从 `x` 读 4 行原始输入；`real_len` 先夹紧到 `[1, T]`，`real_len < 4` 时索引 clamp 到第 0 行（与参考实现一致，不会读到负索引）。
- **padding 行照常卷积**：`[real_len, T)` 的行仍参与计算（kernel 不跳过），模型侧在卷积后按真实长度切片，因此不影响正确性；只有 `conv_state` 严格取自有效前缀。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| x | 必选 | bfloat16 | `[B, T, S]` | `[batch, seq, in_channels]` | 待卷积的输入。`B` 必须为 1；`S` 为输入通道数，只有前 `C = weight.shape[1]` 列参与计算，尾部多余列被忽略。`T` 为 token 总数（不含 padding 语义，padding 由 `seq_len` 标出）。 |
| weight | 必选 | bfloat16 | `[K, C]` | `[kernel_size, out_channels]` | depthwise 卷积权重，`K` 固定为 4，`C` 为输出通道数。调用方由 `conv1d.weight.squeeze(1).transpose(0, 1)` 得到，即 `[C, 1, K] → [K, C]`。 |
| bias | 必选 | bfloat16 | `[C]` | `[out_channels]` | 逐通道偏置。框架 `conv1d.bias` 为 `None` 时调用方传全 0 向量（适配层 `_conv_bias_z`），kernel 始终做 bias 加。 |
| seq_len | 必选 | int32 | `[8]` | `[8]` | 真实有效 token 数，**只读取第 0 个元素**；其余 7 个元素是给调用方复用静态向量的填充槽，kernel 不读取。值会 clamp 到 `[1, T]`。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | bfloat16 | `[B, T, C]` | `[batch, seq, out_channels]` | 每个 token、每个通道的 `silu(conv(x) + bias)`，按 1.1 的两次 bf16 舍入语义计算。`AutoContiguous` 保证连续存储。 |
| conv_state | 必选 | bfloat16 | `[B, K, C]` | `[batch, kernel_size, out_channels]` | 真实序列末尾 4 行原始输入，供 decode 阶段拼出下一个 4 tap 窗口。按 `conv_state.transpose(1, 2)` 交给 `cache_params.update_conv_state`（内部布局 `[B, C, K]`）。 |

### 2.3 属性

无。算子没有属性参数，`tilingKey` 恒为 0。

### 2.4 Tiling 数据结构

host 侧 `FusedCausalConvSiluTilingData`（`op_host/fused_causal_conv_silu_tiling.h` → `op_kernel/fused_causal_conv_silu_tiling_data.h`）与 device 侧镜像结构体，`#pragma pack(8)` + `alignas(8)`。

| 字段 | 类型 | 含义 |
|------|------|------|
| T | int64_t | token 总数（`x` 第 1 维） |
| C | int64_t | 输出通道数（`weight` 第 1 维） |
| S | int64_t | 输入通道数（`x` 第 2 维） |
| K | int64_t | 卷积核长度，固定 4 |
| usedCoreNum | int64_t | 参与计算的 AIV 核数（= `GetCoreNumAiv()`，同时作为 `blockDim`） |

编译期常量在 `fused_causal_conv_silu_tiling_data.h`：`FUSED_CONV_TILE_C = 512`（通道 tile 宽）、`FUSED_CONV_TR = 16`（token chunk 行数），kernel 侧窗口宽度为 `TR + K - 1 = 19`。

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 800I A2 | def 只在 A2 上注册了 AICoreConfig；Atlas 300I Duo 无对应编译产物。 |
| 输入数据类型 | x/weight/bias = bfloat16；seq_len = int32 | tiling 逐输入校验 dtype，任何不匹配都返回 `GRAPH_FAILED`。 |
| 输出数据类型 | y / conv_state = bfloat16 | `InferDataType` 直接沿用 `x` 的 dtype。 |
| 输入格式 | `FORMAT_ND` | def 声明 ND + `DynamicFormatFlag(true)`；`AutoContiguous` 保证输出连续。 |
| batch | 固定 1 | `x` 第 0 维必须为 1，`B > 1` 在 tiling 直接拒绝。 |
| T | ≥ 1 | `seq_len[0]` clamp 到 `[1, T]`，因此 `T` 至少 1 行。 |
| K | 固定 4 | tiling 中 `K_TAPS = 4`，权重第 0 维必须等于 4。 |
| C | 512 的正整数倍，且 16 对齐 | 通道 tile 为 512，`C % 512 == 0`；同时要求 `C ≥ 16` 且 `C % 16 == 0`（512 的倍数已隐含）。 |
| S | ≥ C 且 16 对齐 | `S < C` 或 `S % 16 != 0` 直接 `GRAPH_FAILED`；`S > C` 时尾部列不参与卷积。 |
| bias / seq_len 长度 | bias = `[C]`，seq_len = `[8]` | 长度不符在 tiling 拒绝（`valO.GetDim(0) != 8`）。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag` + `DynamicShapeSupportFlag` + `DynamicRankSupportFlag`，tiling 运行时按实际 `T/C/S` 计算 work item 数。 |
| workspace | 0 | 纯 AIV 向量算子，`workspaces[0] = 0`。 |
| 任务类型 | `KERNEL_TYPE_AIV_ONLY` | 不使用 Cube/AIC。 |
| 多卡 | 仅 1p | 适配层在 `world_size > 1`（TP/CP）时跳过该算子并打印 warning，多卡 prefill 走 `causal_conv1d_fn` 参考路径。 |

### 3.2 不支持的场景

- **`B > 1`**：batch 必须为 1；多 batch prefill 不支持的请求应拆成单 batch 调用或退回框架路径。
- **`K != 4`**：算子按固定 4 tap 展开（`K_TAPS`/`NUM_TAP_ROWS`），不是通用卷积。
- **`C` 不是 512 的倍数**：通道 tile 固定 512，`C=1024/2048/...` 之外的宽度（如 384、768）在 tiling 直接 `GRAPH_FAILED`；Qwen3.5-9B 的 `conv_dim = 2*key_dim + value_dim` 恰好满足。
- **`S < C` 或 `S % 16 != 0`**：尾部通道虽可忽略，但输入行仍按 32 字节对齐读取，未对齐的 `S` 会越界/错位。
- **fp16 / fp32 输入输出**：def 只声明 bf16；混合精度或 FP32 场景不可用。
- **非 ND 格式**：kernel 按连续的 `[T, S]` / `[T, C]` 线性寻址，不处理 `FRACTAL_NZ` 等分形布局。
- **TP / CP 多卡**：融合 kernel 面向单卡；多卡时适配层主动关闭（见 3.1），避免在未验证的通信路径上使用。
- **训练 / 反向**：仅前向，无梯度。

### 3.3 数值约束与精度说明

- **两次舍入点与框架对齐**：框架 `conv1d` 的输出是 bf16 张量，`F.silu` 在其上计算。本算子先在 FP32 中累加 4 tap 与 bias，再 `CAST_RINT` 回 bf16、转回 FP32 算 SiLU，最后再 `CAST_RINT` 输出；因此与“bf16 卷积结果 + bf16 SiLU 输出”的参考实现逐值对齐，唯一差异是 sigmoid 的求值方式，最多 1 个 bf16 ulp（`2^-8`）。算子测试用 `rtol=4e-3, atol=1e-3` 覆盖这一差异。
- **`conv_state` 精确相等**：state 是从 `x` 直接拷贝的原始输入行，不做任何算术，测试用 `torch.equal` 断言逐比特一致。
- **零历史 padding**：`t < K-1` 的行等价于框架侧对序列左侧补 3 行 0；kernel 在 `t0 == 0` 时把窗口前 3 行显式写 0。
- **`real_len < K` 的 state**：索引按 `max(real_len - K + j, 0)` clamp 到第 0 行，与参考实现的 `rows = [max(real_len - K + tap, 0) ...]` 一致；不会读到负地址，也不会引入未初始化数据。
- **padding 行不影响有效输出**：`[real_len, T)` 的行照常卷积，模型在卷积后按真实长度切片；`conv_state` 只从有效前缀取。

## 4 参考资源

- **框架参考**：Transformers `Qwen3_5GatedDeltaNet` 的 short-conv，即 `mixed_qkv → conv1d → F.silu → conv_state` 路径；算子在其上做等价融合。
- **验证用例**：[lite_boost/test/ops/test_fused_causal_conv_silu.py](../../../../../../lite_boost/test/ops/test_fused_causal_conv_silu.py)，覆盖单 token/单 chunk/chunk halo/多通道 tile、`real_len < K` 的 state、以及 `S > C` 的多余通道。
- **调用入口**：`torch.ops.lite_boost.fused_causal_conv_silu(x, weight, bias, seq_len) -> (y, conv_state)`，注册见 `lite_boost/src/ops/register_ops.cc`，插件实现见 `lite_boost/src/ops/plugin/fused_causal_conv_silu.cc`（调用 `aclnnFusedCausalConvSilu`）。
- **CANN 算子开发**：Ascend C 算子开发指南（AIV 向量流水、`DataCopy`/`Cast`/`Mul`/`Add`/`Exp`/`Div`/`Duplicate` 等原语），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
