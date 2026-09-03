lite_boost.ops.chunk_gated_delta_rule
=====================================

.. py:function:: lite_boost.ops.chunk_gated_delta_rule(query, key, value, beta, initial_state, actual_seq_lengths, g=None, scale_value=1.0)

    分块（prefill）Gated Delta Rule算子（A2）。

    包装 ``torch.ops.lite_boost.chunk_gated_delta_rule``，对应A2上CANN的AscendC算子 ``aclnnChunkGatedDeltaRule``。将用户友好的BNSD布局转换为CANN算子要求的TND布局，``query``/``key``/``value``/``beta``/``initial_state`` 转换为低精度dtype（bf16或fp16，跟随输入dtype；fp32/其他默认bf16），可选门 ``g`` 保持float32，``actual_seq_lengths`` 直接透传（T = sum(actual_seq_lengths)）。

    在每个时间步 :math:`t`，Gated Delta Rule按如下公式计算新的递推状态和注意力输出：

    .. math::

        S_t = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1} k_t) k_t^{\top}

    .. math::

        o_t = S_t q_t \cdot scale

    其中 :math:`\alpha_t = \exp(g_t)` 为衰减因子（省略 ``g`` 时禁用衰减，即 :math:`\alpha_t = 1`），:math:`\beta_t` 为Delta更新步长。本算子是上述递推的分块（按块并行）实现，在长序列场景比逐token形式计算效率更高，适用于prefill阶段；输出每一步的结果以及最终状态。

    本接口仅支持A2，不支持300I Duo。

    参数：
        - **query** (Tensor) - 查询张量，shape :math:`(B, N_k, T, D_k)` 。计算前转换为低精度dtype。
        - **key** (Tensor) - 键张量，shape :math:`(B, N_k, T, D_k)` 。计算前转换为低精度dtype。
        - **value** (Tensor) - 值张量，shape :math:`(B, N_v, T, D_v)` 。计算前转换为低精度dtype。
        - **beta** (Tensor) - Delta更新步长，shape :math:`(B, N_v, T)` ，取值范围(0, 1)。计算前转换为低精度dtype。
        - **initial_state** (Tensor) - 输入递推状态，shape :math:`(B, N_v, D_k, D_v)` ，内部转换为算子的value在前布局 ``[B, N_v, D_v, D_k]`` 。
        - **actual_seq_lengths** (Tensor) - 每batch的有效token数，shape :math:`(B)` ，dtype=int32。总序列长度 T = sum(actual_seq_lengths)。BNSD到TND的展平假定每个batch的序列长度一致。
        - **g** (Tensor, 可选) - 全局衰减门，shape :math:`(B, N_v, T)` ，dtype=float32， **必须为负值** 。 ``None`` 表示禁用衰减门（hasGamma=0路径）。默认值： ``None`` 。
        - **scale_value** (float, 可选) - 施加在 `query` 上的注意力缩放因子。默认值： ``1.0`` 。

    返回：
        tuple[Tensor, Tensor]

        - **out** (Tensor) - 注意力输出，shape :math:`(B, N_v, T, D_v)` ，dtype与输入低精度转换结果一致（默认bfloat16）。
        - **final_state** (Tensor) - 更新后的递推状态，shape :math:`(B, N_v, D_k, D_v)` ，dtype与 `out` 一致。

    异常：
        - **RuntimeError** - 输入张量形状、dtype或设备不符合要求，或CANN算子执行失败时抛出。

    .. note::

        - 本接口仅支持A2。300I Duo上注册的是另一签名的 ``ascend_300iduo`` 算子，本绑定不兼容，请勿在300I Duo上使用。
        - 所有输入张量必须在同一NPU设备上。
        - CANN算子通过DataTypeList同时接受bf16和fp16（q/k/v/beta/state），跟随输入dtype，fp32/其他输入默认bf16；可选门 ``g`` 始终为float32。
