lite_boost.ops.sparse_attention
===============================

.. py:function:: lite_boost.ops.sparse_attention(q, k, v, attn_mask=None, scale=None, is_causal=False, head_num=1, input_layout='BNSD', inner_precise=0, sparse_type=None, txt_len=0, block_size=128, latent_shape_q=None, latent_shape_k=None, keep_sink=True, keep_recent=True, cdf_threshold=1.0, sparsity=0.0)

    高层稀疏注意力入口。

    支持两种模式：

    - **dense** ( ``sparse_type=None`` )：直接调用 ``torch_npu.npu_fusion_attention`` 。
    - **block-sparse** ( ``sparse_type="rf_v2"`` )：Token重排 → 块级池化 → Top-k稀疏Mask生成 → ``rain_fusion_attention`` → 逆重排。

    官方接口还支持 ``"rf_v3"`` 和 ``"ada_bsa"`` 模式，当前lite_boost尚未实现，待后续支持。输入张量支持 ``float16`` 和 ``bfloat16`` 数据类型。

    参数：
        - **q** (Tensor) - Query张量。
        - **k** (Tensor) - Key张量。
        - **v** (Tensor) - Value张量。
        - **attn_mask** (Tensor, 可选) - 保留参数（当前rf_v2路径未使用）。默认值： ``None`` 。
        - **scale** (float, 可选) - 注意力缩放因子。 ``None`` 时自动设为 ``head_dim ** -0.5`` 。默认值： ``None`` 。
        - **is_causal** (bool, 可选) - 是否使用因果注意力掩码。在dense模式( ``sparse_type=None`` )下生效，rf_v2稀疏路径暂不支持。默认值： ``False`` 。
        - **head_num** (int, 可选) - 注意力头数。默认值： ``1`` 。
        - **input_layout** (str, 可选) - 输入布局。可选值为 ``"BSND"`` 或 ``"BNSD"`` 。默认值： ``"BNSD"`` 。
        - **inner_precise** (int, 可选) - 精度模式。 ``0`` 为高精度， ``1`` 为高性能。默认值： ``0`` 。
        - **sparse_type** (str, 可选) - 稀疏类型。 ``None`` 为dense； ``"rf_v2"`` 为block-sparse。 ``"rf_v3"`` 和 ``"ada_bsa"`` 待后续支持。默认值： ``None`` 。
        - **txt_len** (int, 可选) - 文本前缀token数量。当 ``>0`` 时，文本token会被分离处理，不参与空间重排（rf_v2专有）。默认值： ``0`` 。
        - **block_size** (int, 可选) - 池化和注意力计算的块大小（rf_v2专有）。默认值： ``128`` 。
        - **latent_shape_q** (list[int], 可选) - Query的潜在空间维度 ``(t, h, w)`` 。rf_v2路径要求提供此参数。例如单帧64×64（4096 tokens）设为 ``(1, 64, 64)`` 。默认值： ``None`` 。
        - **latent_shape_k** (list[int], 可选) - Key/Value的潜在空间维度 ``(t, h, w)`` 。未提供时复用 ``latent_shape_q`` （rf_v2专有）。默认值： ``None`` 。
        - **keep_sink** (bool, 可选) - 是否保留sink token。属于 ``ada_bsa`` 模式专有参数，当前rf_v2路径不读取。默认值： ``True`` 。
        - **keep_recent** (bool, 可选) - 是否保留recent token。属于 ``ada_bsa`` 模式专有参数，当前rf_v2路径不读取。默认值： ``True`` 。
        - **cdf_threshold** (float, 可选) - CDF阈值。属于 ``ada_bsa`` 模式专有参数，当前rf_v2路径不读取。默认值： ``1.0`` 。
        - **sparsity** (float, 可选) - 稀疏率，取值范围 ``[0, 1]`` 。 ``0.0`` 表示全量注意力（不稀疏）， ``0.5`` 表示剪枝50% 的KV块（rf_v2专有）。默认值： ``0.0`` 。
        - **_kwargs** - 其他关键字参数（预留，当前未使用）。

    返回：
        Tensor - 注意力计算结果，shape与输入 ``q`` 一致。

    异常：
        - **ValueError** - ``input_layout`` 不是 ``"BSND"`` 或 ``"BNSD"`` 时抛出。
        - **ValueError** - ``sparse_type`` 不是 ``None`` 或 ``"rf_v2"`` 时抛出。

