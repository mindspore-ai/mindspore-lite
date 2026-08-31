lite_boost.layers.flash_attention
=================================

.. py:function:: lite_boost.layers.flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.float32, version=None)

    对 `q`、`k`、`v` 计算Flash注意力，按可用性自动选择FA3、FA2或NPU（``npu_prompt_flash_attention``）后端。

    支持varlen序列（`q_lens`/`k_lens`），两者均为 ``None`` 时按定长全序列计算。本接口仅支持A2，不支持300I Duo。`dropout_p`、`causal`、`window_size`、`deterministic`、`version` 为GPU端flash_attn（FA2/FA3）后端专有参数，NPU后端不生效（始终计算全局注意力、无dropout）。

    符号说明：`B` 为batch大小，`N` 为头数，`D` 为head_dim（NPU上 :math:`D \le 256`），`S` 为输入 `q`/`k`/`v` 的序列维长度，`lq` 为查询序列长度（``q.size(1)``，与 `S` 数值相同），`lk` 为键序列长度（``k.size(1)``）。返回形状沿用 `lq` 以强调输出序列维与输入 `q` 一致。varlen模式下 `q_lens`/`k_lens` 给出每序列的真实长度（可短于满长）；均为 ``None`` 时按满长（`lq`/`lk`）计算。

    参数：
        - **q** (Tensor) - shape为 :math:`(B, S, N, D)` 的Query张量，须在NPU设备上且 :math:`D \le 256`。支持float16、float32、bfloat16。
        - **k** (Tensor) - shape为 :math:`(B, S, N, D)` 的Key张量，dtype与 `q` 一致。
        - **v** (Tensor) - shape为 :math:`(B, S, N, D)` 的Value张量，dtype与 `q` 一致。
        - **q_lens** (list[int]或Tensor[int32], 可选) - varlen模式下每序列的查询长度，长度须等于 `B`，可短于满长。默认值： ``None`` 。
        - **k_lens** (list[int]或Tensor[int32], 可选) - varlen模式下每序列的键长度，长度须等于 `B`，可短于满长。默认值： ``None`` 。
        - **dropout_p** (float, 可选) - dropout概率，仅GPU端flash_attn FA2后端生效。默认值： ``0.`` 。
        - **softmax_scale** (float, 可选) - 注意力缩放因子， ``None`` 时为 :math:`1/\sqrt{D}`。默认值： ``None`` 。
        - **q_scale** (float, 可选) - 对 `q` 的预缩放（``q = q * q_scale``）。默认值： ``None`` 。
        - **causal** (bool, 可选) - 因果掩码，仅GPU端flash_attn FA2后端生效（NPU后端为全局注意力， ``True`` 不生效）。默认值： ``False`` 。
        - **window_size** (Tuple[int, int], 可选) - 滑窗限制，仅GPU端flash_attn FA2后端生效。默认值： ``(-1, -1)`` 。
        - **deterministic** (bool, 可选) - 确定性模式，仅GPU端flash_attn FA3/FA2后端生效。默认值： ``False`` 。
        - **dtype** (torch.dtype, 可选) - 目标计算dtype，须为float16、bfloat16、float32之一；非half输入先转换。默认值： ``torch.float32`` 。
        - **version** (int, 可选) - ``3`` 强制FA3，不可用时告警降级FA2； ``None`` 自动选择。仅GPU端flash_attn FA3/FA2后端生效。默认值： ``None`` 。

    返回：
        Tensor, shape为 :math:`(B, lq, N, D)`，dtype与输入 `q` 一致。

    异常：
        - **ValueError** - `dtype` 不在{float16, bfloat16, float32}中，或 `q` 不在NPU设备上，或head_dim大于256时抛出。
        - **RuntimeError** - 无可用注意力后端（未安装 ``flash_attn`` 且NPU融合注意力不可用）时抛出。
