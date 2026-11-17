# InnerBatchMatMul

## 1 功能说明

`InnerBatchMatMul` 是面向 Atlas 300I Duo 的**手写 Cube 通道**批量矩阵乘算子，
用于替代 CANN 内置 `BatchMatMulV3` 在 fp16/NZ 场景下的通用实现。它在算子内部直接编排
`GM → L1 → L0A/L0B → L0C → UB → GM` 三级流水，绕开高阶 MatMul API 的调度开销，并在大 K 形态下用 2×2
输出分块把 A/B 的 GM 重读各减半。

典型使用场景：Transformer 类模型的线性投影（QKV、输出投影、FFN 等），fp16、M/N/K 均 16 对齐。

### 1.1 算子语义

```text
y = x1 @ x2          # 不传 bias
y = x1 @ x2 + bias   # bias 为行广播 [batch..., 1, N] 或全尺寸 [batch..., M, N]
```

- `x1`：`[batch..., M, K]`，**FRACTAL_NZ** 存储（`[B, K/16, M/16, 16(m), 16(k)]`）；
- `x2`：`[batch..., K, N]`，**FRACTAL_NZ** 存储（`[B, N/16, K/16, 16(k), 16(n)]`）；
- `bias`：可选，**FRACTAL_NZ**；批量维按 numpy 规则右对齐广播；
- `y`：`ND`，`[batch..., M, N]`，`AutoContiguous`。

FP32 累加在 Cube 内部完成，结果一次 `Cast` 回 FP16（与内置 `BatchMatMulV3` 实测逐 bit 一致）。

### 1.2 实现要点

- **NZ 直拷替代 Nd2Nz**：输入已是 NZ 时 A/B 用 strided `DataCopy` 装入 L1（实测 ~75 GB/s/核），不做格式转换（Nd2Nz 仅 ~13 GB/s/核）。
- **L1/L0 双缓冲 + L0 乒乓**：A/B 的 L1 槽按 `buf & 1` 轮转，L0A/L0B 用两个 `AllocEventID<M_MTE1>` 令牌做
  `Wait → LoadL0 → Mmad → Set`，使 `LoadL0` 与上一个 `Mmad` 重叠。
- **MTE2 逐槽令牌**：2×2 主循环用两个 `MTE2_MTE1` 事件分别标记"该 L1 槽数据已落地"，循环改为
  "先发下一块 → 只等本槽"，替代原先每 chunk 一次的 `Sync<MTE2_MTE1>` 全屏障；覆盖安全性由上一轮末尾的
  `Sync<MTE1_MTE2>` 保证（小 M 路径早已采用同一模式）。
- **B 整 K 常驻 L1**：`K * baseN * 2 ≤ 768KB` 时 B 整列只装一次、跨 m-tile 复用；否则切 `strideAssign=1`
  （同波 `usedCoreNum` 个核共享同一 B 列，L2 复用）。
- **2×2 输出分块（大 K）**：`K > 1536` 且核间均衡时，一个超大瓦片覆盖 `2*baseM × 2*baseN`（256×256），
  L0C 用 4 个累加器（合计 256KB），A/B 各被复用 2 次 → GM 重读各减半。代价是 L0C 无乒乓，写回在瓦片末尾串行；
  只在「大 K + 每核瓦片数不增加」时启用，写回占比很小。
- **延迟写回（tileMul=1 / 小 M）**：L0C 分两块半区并逐 tile 翻转，瓦片 t 的 `L0C→UB(fp32)→Cast(fp16)→NZ2ND→GM`
  推迟到瓦片 t+1 开头，与下一瓦片计算重叠。
- **写回向量链**：`Nz2NdByVec` 用带 dst 步长的 `Muls` 一趟完成 NZ→ND 摊平（每 repeat 8 行），再一次 2D `DataCopy` 写 GM。
- **bias 走 Cube 累加器**：行广播用一次 `K=16` 的附加 `Mmad`（`A'≡1`、`B'` 第 0 行 = bias）把 bias 累加进 L0C；
  全尺寸用 `A'=块对角单位阵`、`B'=bias` 的 `Mmad`（`cmatrixInitVal=true` 首块清零）把 bias 直接写进 L0C ——
  两者都不需要额外的 GM 读/写或向量加。
- **批量维**：6 维右对齐广播，A/B/C/bias 的 batch stride 由 tiling 预计算，kernel 只做一次下标分解；
  A-stationary（grpSched）调度按 batch 逐层展开。

## 2 参数说明

### 2.1 输入

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 格式 | 说明 |
|--------|----------|----------|----------|------|------|
| x1 | 必选 | float16 | `[batch..., M, K]` | FRACTAL_NZ | 左矩阵；`M`、`K` 需 16 对齐；批量维可广播。ND 上游由 GE 自动插一次 `TransData`。 |
| x2 | 必选 | float16 | `[batch..., K, N]` | FRACTAL_NZ | 右矩阵；`K`、`N` 需 16 对齐；批量维可广播。 |
| bias | 可选 | float16 | 行广播 `[batch..., 1, N]` 或全尺寸 `[batch..., M, N]` | FRACTAL_NZ | 批量维右对齐广播；不传时按无 bias 计算。 |

> 可选输入的构图方式：ONNX Custom 节点传 `input_names=["x1","x2","bias"]` + `optional_input_names=["bias"]`，
> 无 bias 时 `input_index=[0,1]`（缺失槽位由转换器 `attr_to_args_pass` 补 `None`）。

### 2.2 输出

| 参数名 | 是否必选 | 数据类型 | 真实维度 | 格式 | 说明 |
|--------|----------|----------|----------|------|------|
| y | 必选 | float16 | `[batch..., M, N]` | ND | 输出矩阵，行主序连续。 |

### 2.3 属性

| 属性名 | 是否必选 | 数据类型 | 默认值 | 说明 |
|--------|----------|----------|--------|------|
| adj_x1 | 可选 | int | 0 | 仅支持 0（不支持转置乘）；非 0 时 tiling 返回 `GRAPH_FAILED`。 |
| adj_x2 | 可选 | int | 0 | 仅支持 0；非 0 时 tiling 返回 `GRAPH_FAILED`。 |
| base_m | 可选 | int | 0 | 调参用：手工指定 M 方向基础瓦片（16 的倍数）；0=自动。受 L0A 双缓冲约束（`tileM*kL1 ≤ 16384`）与 `tileM*tileN ≤ 32768`，有效上限 128，超出按上限收敛。 |
| base_n | 可选 | int | 0 | 调参用：手工指定 N 方向基础瓦片；0=自动。有效范围 16~256（受 L0B 与 UB 容量约束）。 |

> `adj_x1/adj_x2` 按 **INT** 声明：ONNX 没有 BOOL 属性类型，若按 Bool 声明，converter 落在 Custom 上的
> INT 真值会让 `GetAttrPointer<bool>` 拿不到值而被静默当作 false（按非转置计算）。

### 2.4 Tiling 数据结构

`InnerBatchMatMulTilingData`（host/device 共享 POD）：

| 字段 | 含义 |
|------|------|
| m / n / k / batchCount | 逻辑规模与广播后的 batch 总数 |
| coreNum / usedCoreNum / tilePerCore | 可用核数 / 实际启用核数 / 每核瓦片数 |
| baseM / baseN / tileMul | 基础瓦片边长；`tileMul=2` 表示 2×2 超大瓦片 |
| mTiles / nTiles / totalTiles | 瓦片数 |
| outBatchDim[6] / x1·x2·c BatchStride[6] | 6 维右对齐 batch 与各输入的 batch 步长（广播维为 0） |
| strideAssign | 1 = 跨核步进（同波核共享 B 列，L2 复用） |
| grpSched | 1 = A-stationary 分相位调度（每核固定 mi 扫完所有 ni） |
| biasMode | 0 = 无；1 = 行广播；2 = 全尺寸 |
| biasBatchStride[6] | bias 各 batch 维的元素步长（NZ 存储口径） |
| smallM | 1 = 小 M（M≤64）专用路径 |
| useCube | 保留字段（恒为 1） |

## 3 约束说明

### 3.1 支持的限定条件

- 数据类型：**仅 float16**（Cube 只支持 fp16，fp32 累加在 Cube 内部完成）；
- 形状：`M % 16 == 0 && N % 16 == 0 && K % 16 == 0`；在此前提下 M/N 的瓦片尾块（如 64/128/192 行）
  与 K 的 chunk 尾块（如 80/144/1552）均已支持，不要求是 256 或 64 的整数倍；
- bias：行广播 `[batch..., 1, N]`（含 `[1,1,N]`）或全尺寸 `[batch..., M, N]`，末维必须等于 `N`，
  批量维数不得超过输出的批量维数；`batch > 1` 的行广播/全尺寸 bias 均已支持；
- 批量维：最多 6 维，numpy 规则右对齐广播；输入 rank 上限 8（5~6 个批量维要求输入 rank 7/8）；
- 布局：`x1/x2/bias` 为 FRACTAL_NZ，输出 `ND`（由 op def 声明，GE 负责上游 `TransData`）。

### 3.2 不支持的场景

| 场景 | 原因 / 行为 |
|------|------|
| 非 fp16（bf16 / int8） | Cube 无对应 mmad（fp32 输入会被 GE 自动插 `Cast` 降为 fp16，见 §3.3） |
| M / N / K 非 16 对齐 | NZ 直拷与 base tile 均要求 16 对齐；tiling 返回 `GRAPH_FAILED` |
| `M < 16`（单 token GEMV） | 未做专用 tiling；`GRAPH_FAILED` |
| `adj_x1 / adj_x2` 非 0 | 未实现转置通路；tiling 显式返回 `GRAPH_FAILED`（不会静默按非转置计算） |
| 1 维 bias `[N]` 或其它非 `[..,1,N]` / `[..,M,N]` 形状 | 形状解析要求末维 = N、倒数第二维为 1 或 M；`GRAPH_FAILED` |
| bias 批量维多于输出批量维 | 不可广播；`GRAPH_FAILED` |
| 动态 shape | tiling 按静态 `OriginShape` 计算，动态档未验证 |

### 3.3 数值约束与精度说明

- Cube 内 FP32 累加，输出一次 `Cast`（`CAST_NONE`）回 FP16，与内置 `BatchMatMulV3` 实测**逐 bit 一致**；
- bias 在 FP32 累加器内相加（行广播 / 全尺寸均走 `Mmad`），与 FP64 golden 的相对误差 ≈2.9e-4（FP16 舍入量级）；
- 算子只登记 fp16：上游若给 fp32，GE 会自动插入 `Cast` 降为 fp16 再进本算子（有精度损失，不是报错）。

### 3.4 平台声明与动态形状的说明

op def 声明了 `DynamicShapeSupportFlag/DynamicRankSupportFlag(true)`：这是为了让**动静混合图**能通过 GE
编译期检查——动态维由 GE 的分档（`ge.dynamicDims`）在编译期落实为具体形状后才会调用本算子 tiling；
运行期真正未知的维度、以及 rank 未知的图本算子不支持（tiling / infershape 直接 `GRAPH_FAILED`）。
关闭该声明会使使用动态档的上层模型构建失败，故保留。

## 4 参考资源

- **CANN 算子开发**：Ascend C 算子开发指南（Cube/Vector 流水、`Mmad`/`LoadData`/`DataCopy` 等原语），
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition>
- **数据布局**：FRACTAL_NZ 格式与 `TransData` 转换规则，参见 CANN 文档"数据排布格式"章节。
