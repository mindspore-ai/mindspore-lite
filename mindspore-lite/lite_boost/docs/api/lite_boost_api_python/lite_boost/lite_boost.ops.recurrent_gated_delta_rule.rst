lite_boost.ops.recurrent_gated_delta_rule
=========================================

.. py:function:: lite_boost.ops.recurrent_gated_delta_rule(query, key, value, beta, state, actual_seq_lengths, ssm_state_indices, g, gk, num_accepted_tokens, scale_value=1.0)

    基于CANN aclnn后端的递推式线性注意力decode算子。

    实现Gated Delta Rule的逐token递推前向计算，更新递推状态矩阵并输出注意力结果。主要用于混合线性注意力模型（如Qwen3.5）的decode阶段推理加速。

    算法流程（对每个batch中的每个token依次执行）：

    1. 状态衰减：  S = S * exp(g) * exp(gk)
    2. 记忆检索：  kv_mem = S^T @ k
    3. Delta更新：S = S + k^T @ ((v - kv_mem) * beta)
    4. 输出计算：  o = S^T @ q

    其中S是递推状态矩阵 ``[H, D_k, D_v]`` ，存储了线性注意力的key-value关联信息。

    参数：
        - **query** (Tensor) - 查询张量，shape ``[B, H_q, T, D_k]`` ，dtype=bfloat16。必须L2归一化（每个head向量的L2范数为1，值域[0, 1]）。其中B=batch_size，H_q=查询头数，T=序列长度，D_k=key维度。
        - **key** (Tensor) - 键张量，shape ``[B, H_q, T, D_k]`` ，dtype=bfloat16。必须L2归一化（同query）。在GQA/MQA场景中，多个查询头共享同一组key。
        - **value** (Tensor) - 值张量，shape ``[B, H_v, T, D_v]`` ，dtype=bfloat16。H_v=值头数（可与H_q不同以支持GQA/MQA），D_v=value维度。
        - **beta** (Tensor) - Delta更新步长，shape ``[B, H_v, T]`` ，dtype=bfloat16。取值范围(0, 1)。控制每次delta更新的幅度：beta越大，新信息覆盖旧记忆的程度越强；beta越小，倾向于保留已有记忆。
        - **state** (Tensor) - 递推状态矩阵，shape ``[B, H_v, D_k, D_v]`` ，dtype=bfloat16。存储线性注意力的累积key-value关联。D_k为key维度（行），D_v为value维度（列）。首次调用时可初始化为零张量。
        - **actual_seq_lengths** (Tensor) - 实际序列长度，shape ``[B]`` ，dtype=int32。用于变长序列推理。每个元素表示对应batch中的有效token数。例如 ``[4, 3, 5]`` 表示3个batch的序列长度分别为4、3、5。
        - **ssm_state_indices** (Tensor) - SSM状态索引，shape ``[B]`` ，dtype=int32。指示每个batch item在全局状态池中的索引位置，用于多batch场景下的状态管理。通常为 ``[0, 1, 2, ..., B-1]`` 。
        - **g** (Tensor) - 全局衰减门，shape ``[B, H_v, T]`` ，dtype=float32。 **必须为负值** 。 ``exp(g)`` 作为状态衰减因子，值域(0, 1)。g越负，历史信息遗忘越快。例如g=-1时，每步保留约37%的历史状态。
        - **gk** (Tensor) - key维度门控，shape ``[B, H_v, T, D_k]`` ，dtype=float32。 **必须为负值** 。 ``exp(gk)`` 对每个key维度独立施加衰减，实现更细粒度的记忆控制。与全局门g的区别在于gk在D_k维度上逐元素操作。
        - **num_accepted_tokens** (Tensor) - 已接受token数，shape ``[B]`` ，dtype=int32。在speculative decoding等场景中用于标记实际接受（非拒绝）的token数量。普通推理时与 ``actual_seq_lengths`` 相同。
        - **scale_value** (float, 可选) - 注意力缩放因子。通常设为 ``1.0 / sqrt(D_k)`` ，与标准注意力缩放一致。query在计算前会乘以此缩放因子。默认值： ``1.0`` 。

    返回：
        tuple[Tensor, Tensor]

        - **out** (Tensor) - 注意力输出，shape ``[B, H_v, T, D_v]`` ，dtype=bfloat16。每个token位置的线性注意力计算结果。
        - **state_out** (Tensor) - 更新后的递推状态，shape ``[B, H_v, D_k, D_v]`` ，dtype=bfloat16。需在下一步递推时作为 ``state`` 输入传入，形成状态传递链。

    异常：
        - **RuntimeError** - 输入张量形状、dtype或设备不符合要求，或CANN算子执行失败时抛出。

    .. note::

        - 本算子仅支持 **decode阶段** （逐token推理），序列长度T不应超过8。Prefill阶段的并行计算请使用chunk-level算子。
        - 支持GQA (Grouped Query Attention) / MQA (Multi-Query Attention)模式，即H_q可以大于H_v，多个查询头共享同一组key/value头。
        - 所有输入张量必须在同一NPU设备上。
        - CANN算子内部状态存储为 ``[B, H_v, D_v, D_k]`` 布局（value维度在前），本函数会自动进行布局转换。
