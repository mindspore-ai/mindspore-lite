# MsRmsNorm 算子设计

> 最后更新：2026-08-17
> Tiling ABI：`op_host/ms_rms_norm_tiling_data.h`
> 目标平台：Kirin 9020（dav-l310）

## 1. 恢复出的契约

```text
y = fp16(fp32(x) * rsqrt(mean(fp32(x)^2, axis=-1) + epsilon) * gamma)
```

`gamma` 缺省时省略最后一次乘法。平方、求和、均值和 rstd 使用 FP32，输出
转换为 FP16。

| 方向 | 名称 | dtype | format | shape |
|---|---|---|---|---|
| Input 0 | x | fp16 | ND | 非空 `[..., K]` |
| Input 1 | w/gamma | fp16 | ND | 可选 `[K]` |
| Output 0 | y | fp16 | ND | 同 x |

- `M = product(x.shape[:-1])`，一维 x 时 `M=1`。
- `0 < K <= 8192`，且因为当前 GM↔UB 使用普通 `DataCopy`，K 必须是
  16 个 FP16 元素（32B）的整数倍。
- `epsilon` 必须有限且非负；默认值为 `1e-6`。测试可以显式覆盖为 `1e-5`。
- `hasGamma` 来自指定 tiling，是 gamma 可选语义的直接证据；通用 ND 和
  最后一维归一化由 `originM/originK` 合理推导。

## 2. Tiling ABI 与公式

指定头文件的字段名称、类型和顺序保持不变。

| 字段 | 单位 | Host 公式 / Kernel 用途 |
|---|---:|---|
| `originM` | 行 | 前导维乘积 |
| `originK` | 元素 | x 最后一维 |
| `epsilon` | 标量 | 属性值或默认 `1e-6` |
| `reciprocalOfHLength` | 标量 | `1.0f / originK`，避免 Kernel 整数转浮点 |
| `hasGamma` | 布尔值 | 可选 gamma 是否实例化 |
| `blockM` | 行/Block | `ceil(originM / min(originM, physicalAivCores))` |
| `splitM` | Block | `ceil(originM / blockM)`，同时作为 `BlockDim` |
| `splitK` | 元素 | `originK`，当前采用整行 GM↔UB |
| `loopK` | tile | `0`，唯一的数据 tile 作为 tail 处理 |
| `tailK` | 元素 | `originK`，唯一且非空的 tail tile |
| `reduceSplitK` | 元素 | `min(originK, 128)`，FP16 向量寄存器宽度 |
| `reduceLoopK` | 向量 | `(originK - 1) / reduceSplitK` |
| `reduceTailK` | 元素 | `originK - reduceLoopK * reduceSplitK` |

Kernel 在访问 GM 前验证两条覆盖等式，避免损坏的 tiling 导致越界：

```text
loopK * splitK + tailK == originK
reduceLoopK * reduceSplitK + reduceTailK == originK
```

## 3. 并行与 UB 策略

Host 通过平台接口读取物理 AIV 核数，以
`min(originM, physicalAivCores)` 作为目标 Block 数。Kirin 9020 当前是单物理
Vector Core，因此 prefill 也只发射一个有效 Block，避免把逻辑行分成大量
串行调度的伪并行 Block。平台查询不可用时保守回退到 1 核。

Kernel 不再每行单独调用归一化接口，而是在 UB 容量内一次处理多行。
对 FP16 的行宽 `K`：

```text
rowBytes = 2 * K
rowsByUb = floor((120 KiB - rowBytes(gamma) - 32 B(tmp)) /
                 (2 * rowBytes(x + y)))
tileRows = min(blockM, 32, max(1, rowsByUb))
```

每个 tile 只执行一次连续 GM→UB 搬入、一次多行 `RmsNorm`和一次连续
UB→GM 搬出。gamma 在每个物理 Block 中只搬入一次。`reduce*` 仍描述高阶接口
内部的 FP16 寄存器归约分解，Kernel 在调用前继续验证覆盖范围。

Qwen2.5-0.5B 的 `K=896` 可容纳 32 行/tile；`K=2048` 由 UB 公式自动下调到
14 行/tile。对支持上限 `K=8192`，公式仍保证 x、y、gamma 和临时区的合计
占用不超过 120 KiB。

## 4. 名称和发布链

| 层 | 名称 |
|---|---|
| OpDef / tiling 注册 | `MsRmsNorm` |
| ONNX | `custom::MsRmsNorm` |
| shell / 全局函数 | `ms_rms_norm.cpp` / `ms_rms_norm` |
| 固定实现 | `ms_rms_norm_impl.cpp` / `ms_rms_norm_impl` |
| 平台 | `kirin9020` |

旧的 `ms_rms_norm_tiling.h` 已删除，Host、动态 shell 和固定实现只使用
`ms_rms_norm_tiling_data.h` 的 ABI。

## 5. 精度与限制

- Golden 独立使用 Torch FP32 中间计算，最后转换到 FP16。
- 设备精度门限沿用组合容差：`atol=1e-2`、`rtol=1e-3`，并报告
  `max_abs_diff / failed / fail_ratio`。
- 当前只发布 FP16 和 Kirin 9020；未声明 FP32/BF16 或其他 SoC。
- `K > 8192` 暂不支持。当前 DDK L310 的分块更新接口可以累加平方和，
  但对应分块 Normalize 路径存在工具链类型不匹配，不能作为可发布实现。
- 非 32B 对齐的 K 暂不支持。若后续放开，需要同时引入 Host padding tiling、
  `DataCopyPad` 和尾部 GM 安全测试，不能只删除校验。

## 6. 主要风险与对应措施

| 风险 | 处理 |
|---|---|
| 把旧的 4 字段 tiling 继续传给 Kernel | shell 显式逐项传递新 ABI 的全部字段 |
| K 过大导致整行 UB 超限 | Host 明确拒绝 K>8192，不静默越界 |
| 可选 gamma 为空仍访问 GM | `hasGamma` 同时控制 GM buffer 和 Normalize 配置 |
| 使用逻辑行数作 BlockDim，在单核平台产生伪并行 | Host 按实际 AIV 核数分块 |
| 多行 tile 过大导致 UB 越界 | Kernel 根据 120 KiB 容量和行宽运行时计算 `tileRows` |
| tail 为 0 产生非法 DMA | Host 固定 `loopK=0, tailK=originK` |
| 普通 DataCopy 处理非对齐尾块 | Host 明确拒绝 K 非 16 倍数 |
