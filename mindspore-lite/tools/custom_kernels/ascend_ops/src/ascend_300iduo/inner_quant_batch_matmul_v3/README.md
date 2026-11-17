# InnerQuantBatchMatmulV3

## 1 功能说明

`InnerQuantBatchMatmulV3` 是 W8A8 量化批量矩阵乘（per-token 反量化 epilogue）算子，面向昇腾 Atlas 300I Duo 上的 W8A8 推理场景。内核算术与 CANN 内置 300IDUO `QuantBatchMatmulV3` 的 per-token 路径一致（int32 L0C 累加 + fp32 epilogue，位等价由构造保证），host tiling 做两级修正：① k0 的 L1 预算按内核真实的四缓冲布局修正；② 对 `[1536, 12288, 4096]` 档加 shape-keyed swizzle 覆盖表（`Nz/2 -> Zn/4`）。实测 K=12288 档 1.10-1.11×，其余档中性。

### 1.1 算子语义

给定 int8 激活 `x1`、int8 权重 `x2`（NZ 存储、`transpose_x2=true`）、per-channel 反量化 `scale`、可选偏置 `bias`（int32）与必选 per-token 反量化 `pertoken_scale`，算子按下式输出 fp16 结果：

```text
acc[m,n]      = sum_k x1[m,k] * x2[n,k]            # L0C int32 累加，精确可结合
acc[m,n]     += bias[n]                            # 仅 offset 端口缺席时生效，int32 精确加
y[m,n]        = fp16( fp32(acc[m,n]) * scale[n] * pertoken_scale[m] )
                                                   # 乘法次序固定：先 scale，后 pertoken
```

其中 `offset` 端口接受但不参与 per-token 计算；它在本路径在场时 `bias` 同步丢弃（与内置 per-token 语义一致）。公式与 aclnn `QuantMatmulWeightNz` 的 per-token 形态一致（有 `x1Scale` 时无 `x2Offset` 项）。

### 1.2 实现要点

- **内核 vendor 复刻**：`op_kernel/inner_quant_batch_matmul_v3_pertoken_arch20.h` 是内置 300IDUO per-token arch20 内核类的逐字拷贝，6 个模板实例按 tiling 字段运行时分发（与内置 INVOKE 宏相同）；位等价由构造保证——int32 累加精确可结合，k 重分段只重组精确整数加法。
- **te 注册合同**：vendor 侧必须用 `REGISTER_TILING_DEFAULT(结构名)` + `GET_TILING_DATA`（te kernel-source 扫描路径），内置的 `GET_TILING_DATA_WITH_STRUCT` 只对内置 op_host 注册表生成；tiling 结构（`op_kernel/..._tiling_data.h`）与内置逐字段逐字节对齐（68B pack8）。
- **tiling 一级修正（k0 的 L1 预算）**：内置 `k0Max = 131072/(m0+n0)` 把 a/b ping 当一个共享池；本算子改为 `131072/max(m0,n0)`，对应内核实际打开的 4 个独立 131072-int8 L1 缓冲（a-ping/a-pong/b-ping/b-pong）。
- **tiling 二级优化（shape-keyed swizzle 覆盖表）**：search 链后置覆盖表，仅 `[1536, 12288, 4096] -> Zn/4`（B-chunk 跨核 L2 共享），其余 shape 沿用搜索值；swizzle 是纯调度变化（工作项→核的指派序），不改任何数值路径。
- **tiling 扫描通道**：`IQBMM_FORCE_TILING=m0,n0,k0[,swD,swC]` 环境变量可强制指定 tiling，优先级最高（调试/扫描用，免重建逐点扫）。
- **infershape origin-only**：输出 `y` 的 origin 为 `[batch.., M, N]`，NZ 存储布局由 def 的 `FRACTAL_NZ` 端口声明让 GE 推导（与 `rms_norm_nz` 同款做法）。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| x1 | 必选 | int8 | `[.., M, K]` | `[batch, m, k]` | 激活（`transpose_x1=false`），FRACTAL_NZ 存储。 |
| x2 | 必选 | int8 | `[.., N, K]` | `[batch, n, k]` | W8 权重（`transpose_x2=true`），FRACTAL_NZ 存储。 |
| scale | 必选 | float32 | `[N]` | `[n]` | per-channel 权重反量化（x2Scale），ND。 |
| offset | 可选 | float32 | `[N]` | `[n]` | 接受但 per-token 路径忽略；**在场时 bias 同步丢弃**（内置语义镜像）。 |
| bias | 可选 | int32 | `[N]` 或 `[B, 1, N]` | `[n]` 或 `[batch, 1, n]` | L0C int32 域精确加（见 §3.2 bias5 现状）。 |
| pertoken_scale | 必选 | float32 | `[M]` | `[m]` | per-row 激活反量化（x1Scale）；本算子必填（见 §3）。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| y | 必选 | float16 | `[.., M, N]` | `[batch, m, n]` | W8A8 输出，FRACTAL_NZ 存储，`AutoContiguous`。 |

### 2.3 属性

全部属性均为 INT 类型（INT 是本仓库 Custom 链路无损透传的已证类型）。

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| quant_dtype | 必选 | int32 | 无 | y 固定 fp16（`ge::DT_FLOAT16`），tiling 校验。 |
| transpose_x1 | 可选 | int32 | 0 | 仅支持 0。 |
| transpose_x2 | 可选 | int32 | 0 | def 默认 0，但运行时仅接受 1（NZ per-token combo），tiling 显式校验；调用方须显式置 1。 |
| group_size | 可选 | int32 | 0 | 无 group 量化。 |

### 2.4 Tiling 数据结构

host 侧 `QuantMatmulPertokenTilingDataArch20`（`op_host/inner_quant_batch_matmul_v3_tiling.h`）与 device 侧镜像结构体（`op_kernel/inner_quant_batch_matmul_v3_tiling_data.h`）必须**字节一致**，且与 CANN 内置逐字段对齐；二者位于全局命名空间（CANN tiling 宏要求），采用 `#pragma pack(8)` + `alignas(8)`，共 `16 × uint32 + 2 × bool = 68` 字节（对齐到 72）。内核经 `GET_TILING_DATA` 逐字读取该结构体，布局不可漂移。

| 字段 | 类型 | 含义 |
|------|------|------|
| batchSize | uint32_t | batch 数（x1/x2 前导维折叠后）。 |
| m | uint32_t | token 总数 M（所有 batch 折叠）。 |
| k | uint32_t | K 维，int8 NZ 存储要求 32 的倍数。 |
| n | uint32_t | 输出列数 N（= scale 的 numel）。 |
| m0 | uint32_t | 基础块 M 尺寸。 |
| k0 | uint32_t | 基础块 K 尺寸（32 的倍数）。 |
| n0 | uint32_t | 基础块 N 尺寸。 |
| mLoop | uint32_t | M 方向基础块循环次数。 |
| kLoop | uint32_t | K 方向基础块循环次数。 |
| nLoop | uint32_t | N 方向基础块循环次数。 |
| coreLoop | uint32_t | 核间主循环次数（= batchSize × mLoop × nLoop）。 |
| blockDim | uint32_t | 使用核数（= min(coreLoop, coreNum)）。 |
| swizzleDirect | uint32_t | swizzle 方向（L2 局部性优化）。 |
| swizzleCount | uint32_t | swizzle 分组计数。 |
| biasDtype | uint32_t | bias 的 `ge::DataType` 编码（缺省 `DT_INT32`）。 |
| withBias | bool | bias 端口是否生效（offset 缺席时为 true）。 |
| biasWithBatch | bool | bias 是否带 batch 维（`[B, 1, N]` 形态）。 |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | 仅 Ascend300IDUO（300I Duo） | tiling 显式校验 `SocVersion`，其它 SoC 直接失败。 |
| 格式/精度组合 | x1/x2 = int8 FRACTAL_NZ；scale/pertoken_scale = float32 ND；y = float16 FRACTAL_NZ | 固定组合，def 中单值声明（双值 `DataType` 列表会破坏 opbuild 的 ops-info 生成）。 |
| 形状 | 静态 shape；`k` 必须是 32 的倍数（int8 NZ 存储）；x1/x2 前导 batch 维一致（tiling 内折叠） | scale numel = n、pertoken_scale numel = m，tiling 显式校验。 |
| pertoken_scale | 必填 | per-token 变体专用；非 per-token 场景使用内置算子。 |

### 3.2 不支持的场景

- **`transpose_x1=1` / `transpose_x2=0`**：其它 combo 未 vendor，tiling 显式拒绝。
- **fp32 bias**：def 只收 int32（W8A8 的 quant bias 本就是 int32）；内置虽接受 fp32 bias，但双值 `DataType` 列表会触发 opbuild 的 ops-info 生成问题。
- **`withBias` 变体（offset 缺席 + bias 生效）**：内核/tiling/def 侧已就绪，但当前 ONNX Custom 发射路径不可达（中位 optional 端口的省略绑定在 converter 上不通过）；主口径 all6（offset 在场、bias 丢弃）已验证，若需该路径须在导出器侧定约。
- **非 per-token 场景**：请使用内置算子（本算子为 per-token 专用变体）。

### 3.3 数值约束与精度说明

- **位级可复现链**：int32 累加（精确）→ [+ int32 bias]（精确）→ fp32 RN → ×`scale[n]`（fp32 RN）→ ×`pertoken_scale[m]`（fp32 RN）→ fp16 RN；与 numpy int32-exact 参考逐位一致（9.2 栈 7/7），与内置同输入逐位一致。
- **k 重分段无精度影响**：只重组精确整数加法，不改变累加结果；swizzle 只改工作项到核的指派序，epilogue 数学链不动。
- **CANN 版本**：CANN 9.2 完成设备侧精度/性能验证（7/7 全绿）；CANN 8.5 仅作编译目标（设备路径不可用）。
- **非法输入**：numel 不匹配（scale ≠ n、pertoken_scale ≠ m）、非 32 倍数 k、非 NZ 组合均由 tiling 显式拒绝，不会静默降级。

## 4 参考资源

- **CANN 9.2 内置算子源**（内核来源）：`opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/quant_batch_matmul_v3/arch20/quant_batch_matmul_v3_pertoken_arch20.h`；tiling 链来源：`cann_ops/ops-nn/matmul/transpose_batch_mat_mul/op_host/op_tiling/pp_matmul_common_tiling.h`
- **CANN 算子开发**：Ascend C 算子开发指南（`Mmad`/`LoadData`/`DataCopy`/`SetFlag` 等原语），<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
- **aclnn 接口**：`aclnnQuantMatmulWeightNz` 文档
