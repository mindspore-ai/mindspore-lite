# Ulysses 序列并行 (Context Parallel / USP)

## 原理

Ulysses Sequence Parallel (USP) 是一种面向 Transformer 自注意力层的序列维度并行策略。与 Tensor Parallel（切分参数）和 Pipeline Parallel（切分层）不同，USP **不切分模型参数**——每卡持有完整权重，仅在激活上通过 `all_to_all` 通信。

### 核心流程

```text
                          all_to_all                 all_to_all
                         (scatter heads,            (scatter seq,
                        gather seq)                gather heads)
                         ┌──────┐                  ┌──────┐
  rank=0: [B, S/P, H, D]─┤      ├─[B, S, H/P, D]──┤      ├─[B, S/P, H, D]
  rank=1: [B, S/P, H, D]─┤      ├─[B, S, H/P, D]──┤      ├─[B, S/P, H, D]
      ...                 │ a2a  │      ...        │ a2a  │     ...
  rank=N: [B, S/P, H, D]─┤      ├─[B, S, H/P, D]──┤      ├─[B, S/P, H, D]
                         └──────┘                  └──────┘
  各卡有部分序列           每卡有完整序列            各卡回到局部序列
   的全部 head         (H/P)个 head 做全局注意    的全部 head
```

1. **计算前切分**：序列沿 seq_len 均匀切分为 P 份（P = world_size），每卡持有 `[B, S/P, H, D]`
2. **前向 all_to_all**：scatter head(2) → gather seq(1)，每卡拿到完整序列的 H/P 个 head
3. **全局注意力**：每卡在完整序列上独立计算 attention
4. **反向 all_to_all**：scatter seq(1) → gather head(2)，恢复到 `[B, S/P, H, D]`

### 约束

- `num_heads % world_size == 0`，注意力头数必须能被卡数整除
- 序列长度需对齐 `world_size`（通过尾部 padding 实现）
- `world_size == 1` 时退化为单卡，零通信开销

## lite_boost 通用基础设施

### 通信基元：`all_to_all_4d`

`lite_boost/parallel/context_parallel.py` 提供 4D tensor 的 `all_to_all` 通信：

```python
# 前向：scatter head(2), gather seq(1) — 默认
q = all_to_all_4d(q)  # [B, S/P, H, D] → [B, S, H/P, D]

# 反向：scatter seq(1), gather head(2)
x = all_to_all_4d(x, scatter_idx=1, gather_idx=2)  # [B, S, H/P, D] → [B, S/P, H, D]
```

### 通用适配模式

对每个模型采用**三步原地替换**：

1. **替换 flash_attention** → NPU 兼容版（FA3 → FA2 → `npu_prompt_flash_attention`）
2. **替换 self_attn.forward** → `usp_attn_forward`，插入 `all_to_all` 通信对
3. **替换 model.forward** → `usp_dit_forward`，入口切分序列 / 出口 all_gather 合并

```python
for block in model.blocks:
    block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
model.forward = types.MethodType(usp_dit_forward, model)
```

### CP 并行总览

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  usp_dit_forward()                                                       │
│                                                                          │
│  输入 x: list of [T_i, C, H, W]  (各视频片段)                             │
│    ↓  embedding + padding → x: [B, S_pad, D]  (S_pad 对齐 world_size)    │
│    ↓                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  序列切分 (chunk dim=1)                                              │ │
│  │    rank=0: [B, S_pad/P, D]                                          │ │
│  │    rank=1: [B, S_pad/P, D]                                          │ │
│  │    ...                                                               │ │
│  │    rank=P-1: [B, S_pad/P, D]  (含尾部 zero padding)                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│    ↓  各卡独立执行所有 DiT blocks                                          │
│    ↓  ┌───────────────────────────────────────────────────────────────┐  │
│    ↓  │ 每个 block.self_attn = usp_attn_forward                         │  │
│    ↓  │   QKV → RoPE → a2a 前向 → Flash Attn → a2a 反向 → O 投影       │  │
│    ↓  │   内部 a2a 实现跨卡交换：                                        │  │
│    ↓  │     [S_pad/P, H]  ──a2a──→  [S_pad, H/P]  (拿全局序列)          │  │
│    ↓  │     [S_pad, H/P]  ──a2a──→  [S_pad/P, H]  (恢复局部)            │  │
│    ↓  └───────────────────────────────────────────────────────────────┘  │
│    ↓  head + 输出                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  all_gather (dim=1)                                                  │ │
│  │    [B, S_pad/P, D] × P  ──→  [B, S_pad, D]                          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│    ↓  剥离尾部 padding → [B, S, D]                                        │
│    ↓  unpatchify → list of tensors                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

**各卡持有完整模型权重**，仅序列激活在卡间通信。除 attention 内的 `all_to_all` 外，其余计算（embedding、FFN、head、unpatchify）均无通信。

### `usp_attn_forward` 详细流程

```text
QKV 投影 → RoPE → all_to_all 前向 → 剥离 Pad → Flash Attn → 重插 Pad → all_to_all 反向 → 输出投影
```

### Padding 生命周期

当 S 不能整除 P 时，在序列**末尾**追加 N_pad 个零 token，使 S_pad = S + N_pad 能被 P 整除。

```text
┌──────────────────────────────────────────────────────────────────┐
│ 入口 embedding 后：末尾 padding                                    │
│                                                                  │
│   输入: [B, S, H, D]   ──pad──→  [B, S_pad, H, D]                │
│   chunk 切分: rank=k 拿第 k·(S_pad/P) 到 (k+1)·(S_pad/P) 区间    │
│              padding 落在最后一卡                                 │
└──────────────────────────────────────────────────────────────────┘
            ↓ 各卡 QKV 投影 + RoPE
            ↓ all_to_all 前向
┌──────────────────────────────────────────────────────────────────┐
│ 每卡拿到完整序列:  [B, S_pad, H/P, D]                              │
│ 末尾 N_pad 个 token 为 padding（值为零，RoPE 后仍为零）             │
└──────────────────────────────────────────────────────────────────┘
            ↓ 截尾: dim=1 末尾窄化 N_pad
            ↓    → [B, S, H/P, D]  (纯真实序列)
            ↓ Flash Attention
            ↓ 补尾: dim=1 末尾补零 N_pad
            ↓    → [B, S_pad, H/P, D]
            ↓ all_to_all 反向
┌──────────────────────────────────────────────────────────────────┐
│ 回到 chunk 形状:  [B, S_pad/P, H, D]                              │
│ padding 随维度切分自然回到各卡                                      │
└──────────────────────────────────────────────────────────────────┘
            ↓ 所有 block 完成后
            ↓ all_gather
┌──────────────────────────────────────────────────────────────────┐
│ 合并完整序列: [B, S_pad, D]                                       │
│ 出口剥离尾部 N_pad → [B, S, D]                                    │
└──────────────────────────────────────────────────────────────────┘
```

> 反向 `all_to_all` 后各卡已恢复到 chunk 形状，**无需在 attention 层做 `all_gather`**。`all_gather` 仅在 `usp_dit_forward` 出口做一次。
