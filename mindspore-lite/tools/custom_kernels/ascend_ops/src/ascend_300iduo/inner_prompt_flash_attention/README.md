# InnerPromptFlashAttention

## 1 功能说明

`InnerPromptFlashAttention`（内部 PFA）是面向 MindSpore Lite 昇腾 Atlas 300I Duo端侧推理场景的 Prompt Flash Attention 前向融合算子，**仅支持 FP16 输入**。它实现标准 Prompt Flash Attention 的前向计算，是 CANN 内置 `PromptFlashAttention` 算子在 Atlas 300I Duo 上的裁剪与增强端口实现。

算子定义文件头部（[inner_prompt_flash_attention_def.cpp](op_host/inner_prompt_flash_attention_def.cpp)）明确说明：相比 CANN 内置 PFA，本版本的核心增强点为

- 支持 `S1 != S2`（query 序列长度 != key/value 序列长度）时的 `attenMask`；
- 支持 GQA/MQA（`num_key_value_heads < num_heads`）与 `S1 != S2 + mask` 的组合场景。

kernel 入口（[inner_prompt_flash_attention.cpp](op_kernel/inner_prompt_flash_attention.cpp)）头部进一步说明：本实现是上游 `ops-transformer` kernel 入口的裁剪副本，**仅保留 Atlas 推理系列（Atlas 300I Duo）的 FP16 路径**（由 `InnerPromptFlashAttentionS1s2Bns1X310` 实现），刻意丢弃了上游的 QINT8/GQA prefill 路径（`PromptAttentionPrefill`），以避免引入 `unpad_flash_attention_common.h` / `prompt_attention_prefill.h` 及重型 act-template 迭代头文件——FP16 aclnn 路径不会命中这些 tilingKey。

### 1.1 算子语义

给定查询 `Q`、键 `K`、值 `V`，可选的偏置 `pse_shift`、注意力掩码 `atten_mask`、每 batch 真实长度 `actual_seq_lengths` / `actual_seq_lengths_kv`，以及缩放因子 `scale_value`，算子按如下前向公式计算 `attention_out`：

```text
# 1. QK^T 点积并缩放
scores = (Q @ K^T) * scale_value            # [B, N, S1, S2]（GQA 下 K/V 头按 headNumRatio 复用）

# 2. 叠加偏置与掩码
if pse_shift is not None:
    scores = scores + pse_shift              # 位置编码偏置（仅 Atlas 800I A2 支持，Atlas 300I Duo 不支持 pse 非空）
if atten_mask is not None:
    scores = scores + atten_mask_as_fp16     # bool mask 转 fp16 后以 0/-10000 叠加（见 1.2）

# 3. softmax（带 online max/sum 累积，沿 S2 轴归一）
attn   = softmax(scores)                     # [B, N, S1, S2]

# 4. attention @ V
out    = attn @ V                            # [B, N, S1, D]
```

其中 `B` 为 batch、`N` 为 query 头数、`S1` 为 query 序列长度、`S2` 为 key/value 序列长度、`D` 为每头维度。GQA 场景下 K/V 头数 `N_kv` 可小于 `N`，kernel 内按 `headNumRatio = N / N_kv` 复用同一 K/V 头。

### 1.2 实现要点

- **s1s2_bns1 布局分块计算**：kernel 以 `(B, N, S1, S2)` 为逻辑计算轴，按 `singleProcessSOuterSize`（S1 外层切分，典型 128）× `singleProcessSInnerSize`（S2 内层切分，典型 128/512/1024）做矩形基本块。多核按 `(N, B, S1Block)` 三元组切分（`InnerPromptFlashAttentionSplitNS`），稀疏/窗口场景按真实计算量负载均衡（`InnerPromptFlashAttentionSplitNSNew`）。详见 [inner_prompt_flash_attention_s1s2_bns1_x310.h](op_kernel/inner_prompt_flash_attention_s1s2_bns1_x310.h) 与 [inner_prompt_flash_attention_s1s2_bns1_x310_base.h](op_kernel/inner_prompt_flash_attention_s1s2_bns1_x310_base.h)。
- **tiling 模板注册机制**：通过 [fia_tiling_templates_registry.h](op_host/fia_tiling_templates_registry.h) 的 `FiaTilingRegistry` 单例按 `NpuArch`（`DAV_2201/2002/3003/3113`）与优先级（91）注册 `InnerPromptFlashAttentionTiling` 模板类；host 侧 `TilingInnerPromptFlashAttention` 入口调用 `FiaTilingRegistry::GetInstance().DoTilingImpl` 遍历已注册模板，第一个返回非 `GRAPH_PARAM_INVALID` 的模板生效。
- **NZ 域 softmax**：因 Atlas 300I Duo Cube 输出为 NZ（Fractal）格式，`bmm1`（Q@K^T）结果直接落在 `VECCALC` 上的 NZ tensor，后续 softmax 在 NZ 域完成（[kernel_operator_softmax_compute_nz.h](op_kernel/kernel_operator_softmax_compute_nz.h)），通过 `BlockReduceMax` + `TransDataTo5HD` 做 NZ 块内 reduce，避免 NZ↔ND 反复转置。高精度模式（`inner_precise=0`）下 softmax/max/sum 用 FP32，高性模式用 FP16。
- **data_copy_transpose**：`bmm2`（attn@V）结果为 NZ，输出需转回 ND 并按 `BNSD→BSH`/`BSND` 等 layout 排布，由 [kernel_data_copy_transpose.h](op_kernel/kernel_data_copy_transpose.h) 的 `DataCopyTransposeOut` 完成，支持 `NZ2ND_0213`、`NZ2ND_012_WITH_N` 等多种转置枚举（[data_copy_transpose_tiling_def.h](op_host/data_copy_transpose_tiling_def.h)）。
- **多 tilingKey 特化**：Atlas 300I Duo 仅分发 4 个 FP16 tilingKey（见 [inner_prompt_flash_attention_tilingkey.h](op_kernel/inner_prompt_flash_attention_tilingkey.h)）：
  - `12288` → BNSD 高性能（`QINT8_KVFP16_OUTBF16_HIGHLEVELAPI_MDL_NOTAIL_CUBEVECTORDIFF_BNSD_310TILING`，注：名称虽含 QINT8 但 kernel 实际以 `PFATypeNZ<PFALayoutNZ::BNSD, half, int8_t, half>` 实例化，mask 用 int8_t 表示 bool）
  - `22288` → BSH 高性能
  - `12888` → BNSD 高精度（`QFP4E1M2_KVFP16_OUTBF16_..._310TILING`，额外传 `ModeNZ::HighPrecisionNZ`）
  - `22888` → BSH 高精度
- **`S1 != S2` 增强**：[inner_prompt_flash_attention_s1s2_bns1_x310.h](op_kernel/inner_prompt_flash_attention_s1s2_bns1_x310.h) 的 `ComputeEachCore` 中，当 `preTokens >= S1` 且 `nextTokens == 0 || nextTokens >= S2`（全注意力窗口）时，短路 S2 内层循环 `[0, maxInnerLoopTimes)`，避免原 band-mapping 算术在 `S1 < S2` 时漏算 KV 尾块（代码注释 `[S1!=S2 change ②]`）；同时 host tiling 解除了原 `s == seqInnerSize` 的强约束（注释 `[S1!=S2 change ⑤]`），仅保留 `S1 % 128 == 0` 与 mask 下 `S1 % 16 == 0 && S2 % 16 == 0` 的硬件对齐约束。
- **混合核型**：kernel 声明 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`，Cube 与 Vector 协同，`Bmm1/Bmm2` 走 Cube（NZ format），softmax/elewise 走 Vector。

## 2 参数说明

### 2.1 输入

> 说明：真实维度为张量在 GM 中的实际存储形状；逻辑维度为算子语义上的轴含义。`B`=batch、`N`=query 头数（`num_heads`）、`N_kv`=kv 头数（`num_key_value_heads`，缺省等于 `N`）、`S1`=query 序列长、`S2`=kv 序列长、`D`=每头维度、`H = N*D`（BSH）或 `H_kv = N_kv*D`。`input_layout` 属性决定各 layout 的轴顺序，下表“真实维度”以 `input_layout` 实际取值为准。

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| query | 必选 | float16 | BSH: `[B, S1, H]`；BNSD: `[B, N, S1, D]`；BSND: `[B, S1, N, D]`；SH: `[S1, H]`；NSD: `[N, S1, D]` | `[batch, q_seq, q_head*qk_dim]` 或 `[batch, q_head, q_seq, qk_dim]` | 查询张量。Atlas 300I Duo 实际仅走 BSH/BNSD 两条 layout（见 2.3 input_layout）。`H = N*D`，需 `N <= 256`、`D <= 512`。 |
| key | 必选 | float16 | 与 query 同 layout 但头数为 `N_kv`：BSH `[B, S2, N_kv*D]`；BNSD `[B, N_kv, S2, D]`；BSND `[B, S2, N_kv, D]` | `[batch, kv_seq, kv_head*qk_dim]` | 键张量。需与 value 同 dtype、同 dimNum；`N % N_kv == 0` 且 `N/N_kv <= 64`（GQA）。 |
| value | 必选 | float16 | 同 key | 同 key | 值张量。输出 D 维取 value 的 D（与 query 的 D 需相等，见 `CheckD`）。 |
| pse_shift | 可选 | float16 | `[B或1, N, S1, S2]`（4 维） | `[batch, head, q_seq, kv_seq]` | 位置编码偏置，叠加在 QK^T*scale 之后。**Atlas 300I Duo 不支持 pse 非空**（`CheckPseShiftTypeAndShape` 直接报错）。仅 Atlas 800I A2 路径使用。 |
| atten_mask | 可选 | bool | 2/3/4 维：`(S1,S2)` / `(B,S1,S2)` / `(B,N,S1,S2)`；稀疏模式 2/3/4 时需 `(2048,2048)` | `[batch, head, q_seq, kv_seq]` | 注意力掩码。bool 类型，kernel 内 cast 为 fp16 后以 `0/-10000` 叠加（`AttenMaskTransND2NZ` 中 `Muls(..., -10000.0)`）。Atlas 300I Duo 上 mask 类型必须为 `DT_BOOL`。 |
| actual_seq_lengths | 可选 | int64 | `[B]` 或 `[1]` | `[batch]` | 每 batch query 的真实序列长度，元素值 `∈ [0, S1]`。`ValueDepend(OPTIONAL)`。缺省时按 `S1` 处理。 |
| actual_seq_lengths_kv | 可选 | int64 | `[B]` 或 `[1]` | `[batch]` | 每 batch key/value 的真实序列长度，元素值 `∈ [0, S2]`（PA 场景仅要求 `>= 0`）。缺省时按 `S2` 处理。 |
| deq_scale1 | 可选 | uint64 | — | — | bmm1 输出反量化 scale。Atlas 300I Duo FP16 路径不使用（仅 INT8 in 场景）。 |
| quant_scale1 | 可选 | float | — | — | bmm1 输出量化 scale。Atlas 300I Duo FP16 路径不使用。 |
| deq_scale2 | 可选 | uint64 | — | — | bmm2 输出反量化 scale。仅 output=int8 场景使用。 |
| quant_scale2 | 可选 | float | `[1]`（per-tensor）或 `[H]`（per-channel，`H=N*D`） | `[1]` 或 `[head]` | 输出量化 scale。仅 `outputType==DT_INT8` 时必填；per-channel 需 `D` 32B 对齐。 |
| quant_offset2 | 可选 | float | 同 quant_scale2 | 同 quant_scale2 | 输出量化 offset。与 quant_scale2 同形状、同 dtype。 |

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 逻辑维度 | 说明 |
|--------|----------|----------|----------|----------|------|
| attention_out | 必选 | float16 | 同 query 布局：BSH `[B, S1, N*D]`；BNSD `[B, N, S1, D]`；BSND `[B, S1, N, D]`；SH `[S1, N*D]`；NSD `[N, S1, D]` | `[batch, q_seq, q_head*qk_dim]` 等 | 注意力输出。infershape 直接镜像 query 形状（BSH 下 H 由 value 的 `H_kv` × `headNumRatio` 推导）。当 `quant_scale2` 非空时输出 dtype 推断为 `DT_INT8`。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| num_heads | 必选 | int | 1（占位，实际必填） | query 头数 `N`。需 `<= 256`，且与 query shape 的头维一致。 |
| scale_value | 可选 | float | 1.0 | QK^T 缩放因子，通常取 `1/√D`。 |
| pre_tokens | 可选 | int | 214748647 | 稀疏窗口前向 token 数（band mask 上界）。Atlas 300I Duo 上要求 `preTokens >= S1`（否则报错）。 |
| next_tokens | 可选 | int | 0 | 稀疏窗口后向 token 数。Atlas 300I Duo 上要求 `nextTokens == 0` 或 `nextTokens >= S2`。 |
| input_layout | 可选 | string | "BSH" | 输入布局。`SetInputLayout` 支持取值：`SH`/`BSH`/`NSD`/`BSND`/`BNSD`/`BNSD_BSND`（复用 BNSD）/`TND`/`NTD_TND`。**Atlas 300I Duo kernel 实际仅分发 BSH 与 BNSD 两条路径**（tilingKey 12288/22288/12888/22888）；其它 layout 虽在 tiling/infershape 有校验逻辑，但 Atlas 300I Duo 上无对应 kernel 产物。 |
| num_key_value_heads | 可选 | int | 0 | kv 头数 `N_kv`。`0` 表示等于 `num_heads`（MHA）。需满足 `N % N_kv == 0` 且 `N/N_kv <= 64`（GQA 分组上限 64）。 |
| sparse_mode | 可选 | int | 0 | 稀疏模式：`0`=NO_MASK、`1`=ALL_MASK、`2`=LEFT_UP（左上三角，需 mask 为 `(2048,2048)`）、`3`=RIGHT_DOWN（右下三角）、`4`=BAND（带状）。Atlas 300I Duo BaseApi 路径仅支持 `sparse_mode ∈ {20,21,22}`（NONE/NORM/ALIBI 内部编码，见 `CheckBaseAPISupportScenarios`）。 |
| inner_precise | 可选 | int | 1 | 精度模式：`0`=HIGH_PRECISION（高精度，softmax/bmm1 用 FP32）、`1`=HIGH_PERFORMANCE（高性能，全程 FP16）、`4`=APPROXIMATE_COMPUTATION（近似，仅 Atlas 300I Duo 支持，softmax 降为 FP16）。bit0 区分高精度/高性能，bit1 控制 invalid line 修正（`isRowInvalid`）。FP16 输入且 pse 非空时强制升为高精度（Atlas 300I Duo 不支持 pse 故不触发）。 |

### 2.4 Tiling 数据结构

host 侧 `InnerPromptFlashAttentionTilingData`（[inner_prompt_flash_attention_tiling.h](op_host/inner_prompt_flash_attention_tiling.h)）主要嵌套子结构（字段过多，仅列关键）：

| 子结构 | 关键字段 | 含义 |
|--------|----------|------|
| `PromptAttentionBaseParams` | batchSize/headNumSize/seqSize/headSize/scaleValue/preTokens/nextTokens/sparseMode/headNumRatio/layoutType/usePseShift/useMask/innerPrecise(isRowInvalid 编码)/deqScaleFlag/... | 基础参数，描述 B/N/S/D/H、scale、稀疏窗口、layout、各可选输入开关 |
| `PromptAttentionSingleCoreParams` | singleProcessSOuterSize/singleProcessSInnerSize/multiSmaxsInnerLoopTimes/actualCoreNums/attenMaskBatch/pseShiftBatch | 单核切分参数（S1/S2 基本块、内层循环次数、实际用核数） |
| `PromptAttentionSingleCoreTensorSize` | mmResUbSize/bmm2ResUbSize/softmaxMaxSize/softmaxSumSize/softmaxExpSize/spmTmpSize/scmTmpSize/selectSpaceUbSize/... | UB 各 buffer 大小规划 |
| `PromptAttentionSeqParams` | CoreHeadNumTail/actualS1/actualCoreNums/singleCoreHeadNumSize/coreSeqPosStart/coreSeqPosEnd（均 64 元素数组） | 多核 N/B/S 切分起止索引 |
| `PromptAttentionInitOutputParams` | singleCoreSize/totalOutputSize/needInit/isOneN | 输出初始化切分（空 KV 场景） |
| `TCubeTiling bmm1TilingDataRect` / `bmm2TilingDataRect` | — | bmm1/bmm2 的 matmul tiling |
| `SoftMaxTiling softmaxTilingDataRect` / `softmaxFlashTilingDataRect` | — | softmax tiling（普通/flash） |
| `CopyTransposeTiling transposeTilingDataRect` | dstShapeB/N/S/H/... | 输出 NZ→ND 转置 tiling |

## 3 约束说明

### 3.1 支持的限定条件

| 约束项 | 支持范围 | 说明 |
|--------|----------|------|
| 硬件平台 | Atlas 300I Duo | 不支持 Atlas 800I A2 有独立实现，本算子不分发 Atlas 800I A2 tilingKey。 |
| 输入数据类型 | query/key/value = **float16**（FP16-only）；pse_shift = float16；atten_mask = **bool**；actual_seq_lengths(_kv) = int64；deq_scale1/2 = uint64；quant_scale1/2/offset2 = float | `CheckIOType`：`inputType != DT_FLOAT16` 时走 INT8/BF16 分支（Atlas 300I Duo 无对应 kernel）。`DT_FLOAT` 输入/输出直接报错。Atlas 300I Duo 上 mask 必须 `DT_BOOL`（`CheckMaskType`）。 |
| 输出数据类型 | attention_out = float16（默认）；当 quant_scale2 非空时 = int8 | `InferDataType`：`quant_scale2` 非空 → `DT_INT8`。Atlas 300I Duo FP16 路径下输出为 FP16。 |
| 输入格式 | `FORMAT_ND` | def 中 ND + `DynamicFormatFlag(true)` + `DynamicRankSupportFlag(true)`，支持动态 shape 与动态秩。 |
| input_layout | **BSH / BNSD**（Atlas 300I Duo 实际分发） | `SetInputLayout` 解析支持 SH/BSH/NSD/BSND/BNSD/BNSD_BSND/TND/NTD_TND，但 Atlas 300I Duo kernel 仅注册 BSH(22288/22888) 与 BNSD(12288/12888) 四个 tilingKey。其它 layout 在 Atlas 300I Duo 上无编译产物。 |
| S1（query 序列长） | **必须 128 对齐**；`S1 <= 65536` | Atlas 300I Duo 尾块约束：`s % 128 != 0` 直接 `GRAPH_FAILED`，因 kernel tail-block 路径在非 128 对齐时输出 NaN/无效。 |
| S2（kv 序列长） | `S2 <= 65536`；mask 存在时需 `S2 % 16 == 0` | S2 无 128 对齐硬约束（内层 softmax 循环覆盖尾块）。mask 存在时 S1/S2 均需 16 对齐。 |
| S1 != S2 | **支持**（本算子增强点） | host 解除 `s == seqInnerSize` 约束；kernel 全注意力窗口下短路 S2 循环。但 mask 存在时 S1/S2 仍需 16 对齐。 |
| D（每头维度） | `D <= 512`；BSH 下 `H = N*D <= 512*256` | `GetAndCheckEmptyQueryShape`：`d > 512` 报错。BaseApi 高性能路径（`CheckBaseAPISupportScenarios` 置 `atbRunFlag_=true`）要求 query 为 4D（BSH/BNSD）且 `dim(3) == BLOCK_SIZE == 16`（即 `D == 16`）；不满足时退回常规 tiling 路径，不报错。 |
| N（query 头数） | `N <= 256` | `CheckInputDimAndHeadNum`：`nQ > 256` 报错。 |
| GQA 头数关系 | `N % N_kv == 0` 且 `N/N_kv <= 64` | `SetTilingHeadNumRatio`：分组数 `G` 上限 64。`num_key_value_heads=0` 等价于 `=num_heads`（MHA）。 |
| B（batch） | `B <= 65536` | `GetAndCheckEmptyQueryShape`：`b > BLIMIT(65536)` 报错。 |
| 动态 shape | 支持 | `DynamicCompileStaticFlag(true)` + `DynamicShapeSupportFlag(true)` + `DynamicRankSupportFlag(true)`。`jitCompile.flag = static_false,dynamic_false`（静态/动态均不 JIT，预编译产物）。 |
| 稀疏窗口 | `preTokens >= S1` 且 `nextTokens == 0 || nextTokens >= S2` | Atlas 300I Duo 强制要求校验；即 Atlas 300I Duo 仅支持全注意力（无窗口）或右下三角（`sparse_mode=3`，`nextTokens = S2 - S1`）等满足该不等式的组合。 |
| actual_seq_lengths | 元素 `∈ [0, S1]`，长度为 1 或 `>= B` | `CheckActualSeqLength`。TND layout 下要求单调不减且长度 `<= 4096`（`MAX_VAR_LEN_SEQ_LEN`）。 |
| actual_seq_lengths_kv | 元素 `∈ [0, S2]`（PA 场景仅 `>= 0`） | 同上。 |
| 精度模式 | `inner_precise ∈ {0,1,4}` | `0`=高精度（FP32 softmax）、`1`=高性能（FP16）、`4`=近似（仅 Atlas 300I Duo）。`> 4` 仅 warn 不报错。FP16+pse 非空强制升为 0（Atlas 300I Duo 不支持 pse）。 |
| workspace | Atlas 300I Duo：系统 workspace（`GetLibApiWorkSpaceSize`） | Atlas 300I Duo 不额外申请 PFA workspace（`GetPFAWorkSpaceSize` 对 Atlas 300I Duo 直接返回 `defaultSysWorkspaceSize`）。空 KV 场景固定 `16 MiB`。 |

### 3.2 不支持的场景

- **非 Atlas 300I Duo 硬件**：部署到 Atlas 800I A2 等平台找不到编译产物。Atlas 800I A2 有独立 PFA 实现，本算子不分发其 tilingKey。
- **bf16 / fp32 / int8 输入**：Atlas 300I Duo kernel 仅注册 FP16 四个 tilingKey。`inputType == DT_FLOAT` 直接报错；bf16/int8 虽在 tiling/CheckIOType 有分支，但 Atlas 300I Duo 无对应 kernel 产物。
- **pse_shift 非空（Atlas 300I Duo）**：`CheckPseShiftTypeAndShape` 在 Atlas 300I Duo 上直接报错。
- **PageAttention（blockTable 非空，Atlas 300I Duo）**：`RunBigKernelTilingWithParams` 中 `enablePA` 直接报错。
- **KV prefix（shared_prefix 非空）**：与 PA 互斥逻辑中，prefix 路径仅在 Atlas 800I A2 走通，Atlas 300I Duo 无对应 kernel。
- **非 BSH/BNSD layout（Atlas 300I Duo）**：SH/BSND/NSD/TND/NTD_TND 虽有 tiling 校验，但 Atlas 300I Duo 无 kernel tilingKey 注册。
- **mask 类型非 bool（Atlas 300I Duo）**：`CheckMaskType` 在 Atlas 300I Duo 上要求 `maskDataType == DT_BOOL`，fp16/int8/uint8 mask 均报错。FP32 mask 任何平台都不支持。
- **S1 非 128 对齐（Atlas 300I Duo）**：`s % 128 != 0` 直接 `GRAPH_FAILED`，因 tail-block 输出 NaN。
- **mask 存在且 S1/S2 非 16 对齐（Atlas 300I Duo）**：`useMask != 0 && (s % 16 != 0 || seqInnerSize % 16 != 0)` 报错。
- **S1 或 S2 > 65536（Atlas 300I Duo）**：校验报错。
- **`preTokens < S1` 或 `nextTokens` 非 0 且 `< S2`（Atlas 300I Duo）**：报错，Atlas 300I Duo 仅支持满足全注意力窗口不等式的组合。
- **GQA 分组 > 64**：`N/N_kv > 64` 报错。
- **头数 > 256 / D > 512 / B > 65536**：均报错。
- **MSD（per-token 反量化）路径（Atlas 300I Duo）**：`enableMsd` 触发的 `QFP16_KVFP8_*` tilingKey 在 Atlas 300I Duo 无 kernel 产物（仅 Atlas 800I A2）。
- **PA perchannel antiquant（Atlas 300I Duo）**：`CheckPAAntiquantSupportScenarios` 要求 `sparse_mode == RIGHT_DOWN(3)` 且 mask 非空，但 Atlas 300I Duo 上 PA 本身就不支持。
- **tiling 下沉（`tiling_schedule_optimize=True` / `reduce-overhead` 模式）+ TND layout**：`actualSequenceLengthQ/KV` 的 `GetData()` 为 null 时报错。
- **aclnn 单算运行时无 TilingParse**：`ConvertContextToPFAParams` 中 fallback 重建 `InnerPromptFlashAttentionCompileInfo`（thread_local static），因 nnopbase 对 custom vendor op 不调用 `TilingPrepareForInnerPromptFlashAttention`。

### 3.3 数值/精度说明

- **PrecisionReduceFlag(true)**：def 中 `OpAICoreConfig.PrecisionReduceFlag(true)`，允许框架在精度无损前提下做 dtype 降维优化。
- **inner_precise 行为**：
  - `0`（HIGH_PRECISION）：`bmm1` 输出 FP32，softmax/max/sum 用 FP32（`softmaxDataTypeSize = FLOAT32SIZE`），`SoftmaxBasicComputeFirstTail` 走 `SoftmaxFlashV2<float>`；对应 tilingKey 加 600。
  - `1`（HIGH_PERFORMANCE，默认）：全程 FP16，softmax 用 `SoftmaxFlashV2Tmp<half>`；tilingKey 不加 600。
  - `4`（APPROXIMATE_COMPUTATION）：仅 Atlas 300I Duo 支持，`softmaxDataTypeNZ_ = FLOAT16SIZE`，Atlas 800I A2 报错。
- **mask 叠加语义**：bool mask 在 `AttenMaskTransND2NZ` 中先 cast 为 fp16，再 `Muls(..., -10000.0)`，即 `True→0`（保留）、`False→-10000`（屏蔽）。FP16 mask 不支持 invalid line 修正（`isRowInvalid`）。
- **NZ 域 softmax 精度**：NZ 格式下 `BlockReduceMax` 按 16 元素块内 reduce，再 `TransDataTo5HD` 重排，等价于 ND 域行 reduce，无精度损失。
- **bmm1/bmm2 矩阵类型**：Atlas 300I Duo 上 `bmm1` A/B 来自 GM ND，输出 `VECCALC` NZ；`bmm2` A 来自 `VECCALC` NZ，B 来自 GM ND，输出 `VECCALC` ND。高精度下 bmm1 输出 FP32，`ComputeEachCoreSInnerLoop` 中额外 `Cast` 回 FP16 再喂 bmm2。
- **GQA 复用**：K/V 头按 `headNumRatio = N/N_kv` 在 N 维复用同一 K/V 块，`bmm1.SetOrgShape(..., strideQ, strideK=strideQ/ratio)`。

## 4 参考资源

- **CANN ops-transformer flash_attention_score README 格式参考**：<https://gitcode.com/cann/ops-transformer/blob/master/attention/flash_attention_score/README.md>
- **CANN 算子开发指南**（Ascend C 算子开发，Cube/Vector 流水、`Matmul`/`Nd2Nz`/`SoftmaxFlashV2`/`TransDataTo5HD` 等原语）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
