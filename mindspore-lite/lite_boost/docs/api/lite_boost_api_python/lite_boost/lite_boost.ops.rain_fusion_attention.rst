lite_boost.ops.rain_fusion_attention
=====================================

.. py:function:: lite_boost.ops.rain_fusion_attention(query, key, value, select_idx, select_num_idx, block_shape, attn_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, block_table=None, q_input_layout='TND', kv_input_layout='TND', num_key_value_heads=1, mask_type=0, scale_value=1.0, inner_precise=1, block_size=0)

    块稀疏融合注意力前向计算。

    调用NPU原生 ``aclnnRainFusionAttention`` 算子执行块级稀疏融合注意力。通过 ``select_idx`` 和 ``select_num_idx`` 指定每个query块需要关注的KV块集合，支持灵活的非均匀稀疏注意力模式。输入张量支持 ``float16`` 和 ``bfloat16`` 数据类型。

    参数：
        - **query** (Tensor) - Query张量。输入布局由 ``q_input_layout`` 决定。
        - **key** (Tensor) - Key张量。输入布局由 ``kv_input_layout`` 决定。
        - **value** (Tensor) - Value张量。输入布局由 ``kv_input_layout`` 决定。
        - **select_idx** (Tensor) - 稀疏选择索引矩阵。shape ``[q_blocks, num_heads, kv_blocks]`` ，dtype ``torch.int64`` 。每行按有效KV块索引升序排列在前，多余位置填充 ``-1`` 。
        - **select_num_idx** (Tensor) - 每query块、每head对应的有效KV块数量。shape ``[q_blocks, num_heads]`` ，dtype ``torch.int64`` 。
        - **block_shape** (list[int]) - 块内tile尺寸，格式 ``[block_rows, block_cols]`` ，通常设为 ``[128, 128]`` 。
        - **attn_mask** (Tensor, 可选) - 注意力mask张量。默认值： ``None`` 。
        - **actual_seq_lengths** (list[int], 可选) - 每个batch的实际Q序列长度。layout为 ``"TND"`` 时需要提供此参数以正确计算序列边界。默认值： ``None`` 。
        - **actual_seq_lengths_kv** (list[int], 可选) - 每个batch的实际KV序列长度。默认值： ``None`` 。
        - **block_table** (Tensor, 可选) - PagedAttention场景下的block table。默认值： ``None`` 。
        - **q_input_layout** (str, 可选) - Query输入布局。可选值为 ``"TND"`` 或 ``"BNSD"`` 。默认值： ``"TND"`` 。
        - **kv_input_layout** (str, 可选) - Key/Value输入布局。可选值为 ``"TND"`` 或 ``"BNSD"`` 。默认值： ``"TND"`` 。
        - **num_key_value_heads** (int, 可选) - KV head数量，用于GQA/MQA。默认值： ``1`` 。
        - **mask_type** (int, 可选) - Mask类型。 ``0`` 表示causal mask。默认值： ``0`` 。
        - **scale_value** (float, 可选) - 注意力缩放因子。建议设为 ``head_dim ** -0.5`` 。默认值： ``1.0`` 。
        - **inner_precise** (int, 可选) - 精度模式。 ``0`` 为高精度， ``1`` 为高性能。默认值： ``1`` 。
        - **block_size** (int, 可选) - Block size。 ``0`` 表示自动推断。默认值： ``0`` 。

    返回：
        tuple[Tensor, Tensor]

        - **attention_out** (Tensor) - 注意力计算结果，shape与 ``query`` 相同。
        - **softmax_lse** (Tensor) - Softmax log-sum-exp值，shape为 ``[T, N, H]`` ，用于调试和梯度回传。

    异常：
        - **TypeError** - ``query`` 、 ``key`` 、 ``value`` 、 ``select_idx`` 或 ``select_num_idx`` 不是Tensor时抛出。

