# RecurrentGatedDeltaRule

RecurrentGatedDeltaRule 是 lite_boost 组件中面向华为昇腾 NPU 的线性注意力递推推理算子，封装 CANN `aclnnRecurrentGatedDeltaRule` 后端，为混合线性注意力模型（如 Qwen3.5）的 decode（逐 token 推理）阶段提供硬件加速。

与传统 Softmax 注意力不同，Gated Delta Rule 采用固定大小的递推状态矩阵替代无限增长的 KV Cache，将序列混合的复杂度从 O(N²) 降低至 O(N)，实现高效的常量内存推理。

---

## 1. 算法原理

### 1.1 背景：线性注意力的递推视角

标准线性注意力（Linear Attention）将 Softmax 注意力中的指数核替换为线性核，使注意力计算可表示为递推形式：

```text
O_i = (Σ_{j=1}^{i} φ(k_j) ⊗ v_j) @ q_i
```

等价于维护一个递推状态矩阵 `S = Σ k_j ⊗ v_j`，每步通过 `o_i = S @ q_i` 读取输出。但朴素线性注意力在检索和长上下文任务上显著弱于 Softmax 注意力。

### 1.2 Gated Delta Rule 的改进

Gated Delta Rule（门控增量规则）源自论文 *"Gated Delta Networks: Improving Mamba2 with Delta Rule"* (Yang et al., ICLR 2025, arXiv:2412.06464)，通过两个互补机制解决线性注意力的局限性：

**门控机制（Gating）**：通过指数衰减因子对递推状态进行选择性遗忘：

```text
S_i = S_{i-1} * exp(g_i) * exp(gk_i)
```

- `exp(g)`：全局衰减门，`g < 0` 使 `exp(g) ∈ (0, 1)`，控制整体状态衰减速率。
- `exp(gk)`：Key 维度门控，沿每个 key 维度独立衰减，提供更精细的记忆控制粒度。

**Delta Rule（增量更新规则）**：借鉴 Delta 学习规则，不直接覆盖记忆，而是先计算当前记忆的"误差"，再沿 key 方向进行定向修正：

```text
kv_mem_i = S_i @ k_i                    # 检索：当前状态对 key 的响应
delta_i  = (v_i - kv_mem_i) * beta_i     # 误差：目标值与已有记忆之差
S_i      = S_i + k_i^T @ delta_i         # 更新：沿 key 方向修正
```

- `beta`：更新步长，取值 `(0, 1)`，越大则新信息覆盖程度越强。

### 1.3 完整递推流程

decode 阶段对每个 token 依次执行以下四步：

```text
输入：S_{i-1}（上一步递推状态）, q_i, k_i, v_i, g_i, gk_i, beta_i

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. 状态衰减  │──▶│ 2. 记忆检索  │──▶│ 3. Delta 更新│──▶│ 4. 输出计算  │
│ S *= exp(g)  │   │ kv_mem = S@k │   │ S += k^T@δ   │   │ o = S^T@q    │
│ S *= exp(gk) │   │              │   │ δ=(v-mem)*β  │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘

输出：o_i, S_i
```

### 1.4 与 Softmax 注意力的对比

| 特性 | Softmax Attention | Gated Delta Rule |
|------|-------------------|------------------|
| 训练复杂度 | O(N²) | O(N)（chunk 并行） |
| 推理内存 | O(N) KV Cache | O(D_k × D_v) 固定状态 |
| 长上下文支持 | 受 KV Cache 限制 | 固定状态，不受限 |
| 记忆精度 | 精确（全序列可见） | 近似（递推压缩） |
| 检索性能 | 优秀 | 显著优于朴素线性注意力 |

---

## 2. 实现架构

### 2.1 三层架构

```text
┌─────────────────────────────────────────────────┐
│  Python 绑定层 (recurrent_gated_delta_rule.py)  │
│  - BNSD ↔ TND 布局转换                          │
│  - cu_seqlen 累积序列长度计算                    │
│  - state 矩阵布局适配 (Dk,Dv ↔ Dv,Dk)           │
├─────────────────────────────────────────────────┤
│  C++ 算子层 (register_ops.cc + .cc/.h)          │
│  - TORCH_LIBRARY 注册自定义算子                  │
│  - PrivateUse1 (NPU) dispatch                   │
│  - EXEC_NPU_CMD 调用 CANN 后端                   │
├─────────────────────────────────────────────────┤
│  CANN 后端 (aclnnRecurrentGatedDeltaRule)        │
│  - 昇腾 NPU 硬件加速执行                        │
│  - Workspace 自动分配与异步执行                  │
└─────────────────────────────────────────────────┘
```

### 2.2 布局转换说明

用户侧使用 BNSD 布局 `[Batch, Heads, SeqLen, Dim]`，CANN 算子要求 TND 布局 `[T_total, Heads, Dim]`。

**输入转换（BNSD → TND）**：

```text
4D: [B, H, T, D] → transpose(1,2) → [B, T, H, D] → reshape(-1, H, D) → [B*T, H, D]
3D: [B, H, T]    → transpose(1,2) → [B, T, H]    → reshape(-1, H)    → [B*T, H]
```

**输出转换（TND → BNSD）**：

```text
[T_total, H, D] → reshape(B, T, H, D) → transpose(1,2) → [B, H, T, D]
```

**状态矩阵布局**：

- Python 侧：`[B, H, D_k, D_v]`（key 维度在前，value 维度在后）
- CANN 侧：`[B, H, D_v, D_k]`（value 维度在前，key 维度在后）
- 转换方式：`transpose(-1, -2)`

### 2.3 cu_seqlen 累积序列长度

cu_seqlen 用于在展平的 T_total 维度中定位每个 batch 的起止位置：

```text
actual_seq_lengths = [4, 3, 5] → cu_seqlen = [0, 4, 7, 12]
  - batch 0: tokens [0, 4)
  - batch 1: tokens [4, 7)
  - batch 2: tokens [7, 12)
```

---

## 3. 接口说明

### 3.1 函数签名

```python
recurrent_gated_delta_rule(
    query,                  # [B, H_q, T, D_k]  bfloat16
    key,                    # [B, H_q, T, D_k]  bfloat16
    value,                  # [B, H_v, T, D_v]  bfloat16
    beta,                   # [B, H_v, T]       bfloat16
    state,                  # [B, H_v, D_k, D_v] bfloat16
    actual_seq_lengths,     # [B]               int32
    ssm_state_indices,      # [B]               int32
    g,                      # [B, H_v, T]       float32
    gk,                     # [B, H_v, T, D_k]  float32
    num_accepted_tokens,    # [B]               int32
    scale_value=1.0,        # float
)
-> (output, state_out)      # ([B, H_v, T, D_v], [B, H_v, D_k, D_v]) bfloat16
```

### 3.2 参数说明

| 参数 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `query` | `[B, H_q, T, D_k]` | bfloat16 | 查询张量，需 L2 归一化 |
| `key` | `[B, H_q, T, D_k]` | bfloat16 | 键张量，需 L2 归一化 |
| `value` | `[B, H_v, T, D_v]` | bfloat16 | 值张量 |
| `beta` | `[B, H_v, T]` | bfloat16 | Delta 更新步长，取值 (0, 1) |
| `state` | `[B, H_v, D_k, D_v]` | bfloat16 | 递推状态矩阵，首次调用初始化为零 |
| `actual_seq_lengths` | `[B]` | int32 | 各 batch 的实际序列长度 |
| `ssm_state_indices` | `[B]` | int32 | 各 batch 在全局状态池中的索引 |
| `g` | `[B, H_v, T]` | float32 | 全局衰减门，**必须为负值** |
| `gk` | `[B, H_v, T, D_k]` | float32 | Key 维度门控，**必须为负值** |
| `num_accepted_tokens` | `[B]` | int32 | 已接受 token 数（speculative decoding 场景） |
| `scale_value` | - | float | 注意力缩放因子，默认 1.0 |

### 3.3 返回值

| 输出 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `output` | `[B, H_v, T, D_v]` | bfloat16 | 注意力输出 |
| `state_out` | `[B, H_v, D_k, D_v]` | bfloat16 | 更新后的递推状态，传给下一步推理 |

其中：

- B = batch_size
- H_q = 查询头数，H_v = 值头数（支持 GQA/MQA，H_q >= H_v）
- T = 序列长度（decode 阶段通常为 1~8）
- D_k = key/query 维度，D_v = value 维度

---

## 4. 使用示例

### 4.1 基本用法（单 token decode）

```python
import torch
import lite_boost.ops as lite_ops

# Qwen3.5-2B decode 配置
B, H, T, Dk, Dv = 1, 64, 1, 64, 512

# 初始化输入张量
query  = torch.randn(B, H, T, Dk, device="npu:0", dtype=torch.bfloat16)
key    = torch.randn(B, H, T, Dk, device="npu:0", dtype=torch.bfloat16)
value  = torch.randn(B, H, T, Dv, device="npu:0", dtype=torch.bfloat16)
beta   = torch.rand(B, H, T, device="npu:0", dtype=torch.bfloat16) * 0.9 + 0.05
state  = torch.zeros(B, H, Dk, Dv, device="npu:0", dtype=torch.bfloat16)
g      = -(torch.rand(B, H, T, device="npu:0") + 0.01)       # 负值
gk     = -(torch.rand(B, H, T, Dk, device="npu:0") + 0.01)   # 负值

actual_seq_lengths  = torch.tensor([T], dtype=torch.int32, device="npu:0")
ssm_state_indices   = torch.tensor([0], dtype=torch.int32, device="npu:0")
num_accepted_tokens = torch.tensor([T], dtype=torch.int32, device="npu:0")

# 执行递推推理
output, state_out = lite_ops.recurrent_gated_delta_rule(
    query, key, value, beta, state,
    actual_seq_lengths, ssm_state_indices,
    g, gk, num_accepted_tokens,
    scale_value=1.0 / (Dk ** 0.5),
)
# output:    [1, 64, 1, 512]  — 当前 token 的注意力输出
# state_out: [1, 64, 64, 512] — 更新后的递推状态
```

### 4.2 多步 decode 推理循环

```python
# 初始化递推状态
state = torch.zeros(B, H, Dk, Dv, device="npu:0", dtype=torch.bfloat16)

# 逐 token 推理循环
for step in range(num_decode_steps):
    # 获取当前 token 的 query, key, value, beta, g, gk
    q_step = get_query(step)    # [B, H, 1, Dk]
    k_step = get_key(step)      # [B, H, 1, Dk]
    v_step = get_value(step)    # [B, H, 1, Dv]
    b_step = get_beta(step)     # [B, H, 1]
    g_step = get_g(step)        # [B, H, 1]
    gk_step = get_gk(step)      # [B, H, 1, Dk]

    # 执行单步递推
    output, state = lite_ops.recurrent_gated_delta_rule(
        q_step, k_step, v_step, b_step, state,
        actual_seq_lengths, ssm_state_indices,
        g_step, gk_step, num_accepted_tokens,
        scale_value=1.0 / (Dk ** 0.5),
    )
    # output: 当前 token 的输出，state: 传递给下一步
```

---

## 5. CANN 算子约束

使用本算子时，输入数据需满足以下 CANN 后端约束：

| 约束项 | 范围 | 说明 |
|--------|------|------|
| 序列长度 Li | `0 < Li <= 8` | 每批次最大支持 8 个 token |
| Key 头数 Nk | `0 < Nk <= 256` | 查询/键的注意力头数 |
| Value 头数 Nv | `Nk <= Nv <= 256` | 值头数，需满足 `Nv % Nk == 0` |
| Key 维度 Dk | `0 < Dk <= 512` | Key/Query 的单头维度 |
| Value 维度 Dv | `0 < Dv <= 512` | Value 的单头维度 |
| query/key 值域 | `[0, 1]` | 需 L2 归一化 |
| g（衰减门） | `< 0` | 确保 exp(g) ∈ (0, 1) |
| gk（key 门控） | `< 0` | 确保 exp(gk) ∈ (0, 1] |
| beta（步长） | `(0, 1)` | Delta 更新步长 |

**违反约束的后果**：计算结果不保证正确性，可能导致精度异常或运行时错误。

---

## 6. 环境依赖与配置

### 6.1 硬件要求

- 华为昇腾 NPU（Atlas 800I A2 及以上）
- 已安装 CANN 软件包和 `torch_npu`

### 6.2 共享库加载

`recurrent_gated_delta_rule` 依赖 `liblite_boost_ops.so` 共享库，加载优先级如下：

1. **环境变量** `LITE_BOOST_OPS_LIB`：直接指定 `.so` 路径
2. **相对路径**：`python/ops/../lib/liblite_boost_ops.so` 或 `lite_boost_ops.so`
3. **系统路径**：在 `sys.path` 中搜索 `lite_boost/lib/` 下的 `.so` 文件

若加载失败，将抛出 `FileNotFoundError`。

### 6.3 自定义算子注册

底层算子通过 PyTorch custom op 机制注册：

```text
TORCH_LIBRARY(lite_boost, m)
└─ recurrent_gated_delta_rule → aclnnRecurrentGatedDeltaRule
```

NPU 后端实现绑定在 `PrivateUse1` dispatch key 上，仅在 NPU 设备上生效。

---

## 7. 支持的模型

| 模型 | 配置 | 说明 |
|------|------|------|
| Qwen3.5-0.8B | H=64, Dk=64, Dv=512 | 多模态 VL 模型 |
| Qwen3.5-2B | H=64, Dk=64, Dv=512 | 混合线性注意力 |
| Qwen3.5-4B | H=64, Dk=64, Dv=512 | 混合线性注意力 |

Qwen3.5 采用混合架构，部分层使用标准 Softmax 注意力，部分层使用 Gated Delta Rule 线性注意力。本算子用于加速其中的线性注意力层的 decode 推理。

---

## 8. 注意事项

1. **设备限制**：`recurrent_gated_delta_rule` 仅在 NPU 上可用，不能在 CPU/GPU 上运行。
2. **序列长度**：decode 阶段每批次序列长度不超过 8。Prefill 阶段的并行计算请使用 chunk-level 算子。
3. **L2 归一化**：`query` 和 `key` 必须 L2 归一化（每个 head 向量的 L2 范数为 1），否则计算结果不保证正确性。
4. **衰减门符号**：`g` 和 `gk` 必须为负值，确保 `exp(g) ∈ (0, 1)`，实现正确的状态衰减。若误传正值，状态将指数增长导致数值溢出。
5. **GQA/MQA 支持**：支持 H_q >= H_v 的分组查询注意力模式，多个查询头共享同一组 key/value 头。
6. **状态传递**：`state_out` 需在下一步递推时作为 `state` 输入传入，形成状态传递链。首次调用前应将 `state` 初始化为与 `[B, H_v, D_k, D_v]` 形状匹配的全零张量。

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `lite_boost/python/ops/recurrent_gated_delta_rule.py` | Python 绑定层 |
| `lite_boost/python/ops/__init__.py` | 算子导出注册 |
| `lite_boost/src/ops/plugin/recurrent_gated_delta_rule.cc` | C++ 算子实现 |
| `lite_boost/src/ops/plugin/recurrent_gated_delta_rule.h` | C++ 算子头文件 |
| `lite_boost/src/ops/register_ops.cc` | PyTorch 自定义算子注册 |
| `lite_boost/test/ops/test_recurrent_gated_delta_rule.py` | 测试用例（精度 + 性能） |

---

## 10. 参考文献

- Yang, S., Kautz, J., & Hatamizadeh, A. (2025). **Gated Delta Networks: Improving Mamba2 with Delta Rule**. *ICLR 2025*. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)
- Yang, S., Wang, B., Shen, Y., & Others (2024). **DeltaNet: Error-Driven Linear Attention with the Delta Rule**.
- Katharopoulos, G., Vyas, A., Pappas, N., & Fleuret, F. (2020). **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention**. *ICML 2020*.

---

## 11. Ascend 310P support

On Atlas 300I Duo (Ascend 310P), LiteBoost loads the
`mslite_custom_ops` implementation because CANN does not provide the native
RGDR op on this product. The 310P path uses FP16 inputs, in-place FP16 state,
and FP16 output. The A2 native path continues to use BF16.

Qwen3.5-4B uses `Nk=16`, `Nv=32`, and `Dk=Dv=128` for recurrent
attention. RGDR maps each key/query head to `Nv/Nk` consecutive value heads;
therefore `Nv` must be divisible by `Nk`.

The 310P test marker is `ascend_300iduo`. Its L0 cases cover the Qwen3.5-4B
single-token decode shape and batch size two.
