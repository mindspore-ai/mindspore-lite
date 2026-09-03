lite_boost.ops.recurrent_gated_delta_rule
=========================================

.. py:function:: lite_boost.ops.recurrent_gated_delta_rule(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices, g, gk, num_accepted_tokens, scale_value=1.0)

    基于CANN aclnn后端的递推式线性注意力decode算子。

    实现Gated Delta Rule的逐token递推前向计算，更新递推状态矩阵并输出注意力结果。主要用于混合线性注意力模型（如Qwen3.5）的decode阶段推理加速。

    算法流程（对每个batch中的每个token依次执行）。状态衰减为

    .. math::

        S = S * \exp(g) * \exp(gk)

    记忆检索为

    .. math::

        kv\_mem = S^{\top} k

    Delta更新为

    .. math::

        S = S + k^{\top} ((v - kv\_mem) * \beta)

    输出计算为

    .. math::

        o = S^{\top} q

    其中 :math:`S` 是shape为 :math:`(N_v, D_k, D_v)` 的递推状态矩阵，存储了线性注意力的key-value关联信息。

    参数：
        - **query** (Tensor) - 查询张量，shape :math:`(B, N_k, T, D_k)` ，dtype=bfloat16。必须L2归一化（每个head向量的L2范数为1，值域[0, 1]）。其中B=batch_size，N_k=查询头数，T=序列长度，D_k=key维度。
        - **key** (Tensor) - 键张量，shape :math:`(B, N_k, T, D_k)` ，dtype=bfloat16。必须L2归一化（同query）。
        - **value** (Tensor) - 值张量，shape :math:`(B, N_v, T, D_v)` ，dtype=bfloat16。N_v=值头数（须为N_k的整数倍），D_v=value维度。
        - **beta** (Tensor) - Delta更新步长，shape :math:`(B, N_v, T)` ，dtype=bfloat16。取值范围(0, 1)。控制每次delta更新的幅度：beta越大，新信息覆盖旧记忆的程度越强；beta越小，倾向于保留已有记忆。
        - **state** (Tensor) - 递推状态池，shape :math:`(state\_slots, N_v, D_k, D_v)` ，dtype=bfloat16。state_slots为池中状态槽的个数，每个槽独立存储一个序列累积的key-value关联，各token经 `ssm_state_indices` 映射到对应槽位（普通推理时每个batch占用一个槽，state_slots通常等于B）。D_k为key维度（行），D_v为value维度（列）。首次调用时可初始化为零张量。
        - **actual_seq_lengths** (Tensor) - 实际序列长度，shape :math:`(B)` ，dtype=int32。用于变长序列推理。每个元素表示对应batch中的有效token数。例如 ``[4, 3, 5]`` 表示3个batch的序列长度分别为4、3、5。
        - **ssm_state_indices** (Tensor) - 状态槽索引，shape :math:`(T\_total)` ，dtype=int32，每个展平后的token据此在全局状态池（ `state` 的第0维，共state_slots个槽）中选择一个状态槽。
        - **g** (Tensor) - 全局衰减门，shape :math:`(B, N_v, T)` ，dtype=float32。 **必须为负值** 。 ``exp(g)`` 作为状态衰减因子，值域(0, 1)。g越负，历史信息遗忘越快。例如g=-1时，每步保留约37%的历史状态。
        - **gk** (Tensor) - key维度门控，shape :math:`(B, N_v, T, D_k)` ，dtype=float32。 **必须为负值** 。 ``exp(gk)`` 对每个key维度独立施加衰减，实现更细粒度的记忆控制。与全局门g的区别在于gk在D_k维度上逐元素操作。
        - **num_accepted_tokens** (Tensor) - 已接受token数，shape :math:`(B)` ，dtype=int32。在speculative decoding等场景中用于标记实际接受（非拒绝）的token数量。普通推理时与 ``actual_seq_lengths`` 相同。
        - **scale_value** (float, 可选) - 注意力缩放因子。通常设为 ``1.0 / sqrt(D_k)`` ，与标准注意力缩放一致。query在计算前会乘以此缩放因子。默认值： ``1.0`` 。

    返回：
        tuple[Tensor, Tensor]

        - **out** (Tensor) - 注意力输出，shape :math:`(B, N_v, T, D_v)` ，dtype=bfloat16。每个token位置的线性注意力计算结果。
        - **state_out** (Tensor) - 更新后的递推状态池，shape与 `state` 一致，dtype=bfloat16。需在下一步递推时作为 ``state`` 输入传入，形成状态传递链。

    异常：
        - **RuntimeError** - 输入张量形状、dtype或设备不符合要求，或CANN算子执行失败时抛出。

    .. note::

        - 本算子仅支持 **decode阶段** （逐token推理），序列长度T不应超过8。Prefill阶段的并行计算请使用chunk-level算子。
        - 支持分组递推头（grouped recurrent heads），即N_v为N_k的整数倍。
        - 所有输入张量必须在同一NPU设备上。
        - CANN算子内部状态存储为 :math:`(state\_slots, N_v, D_v, D_k)` 布局（value维度在前），本函数会自动进行布局转换。
