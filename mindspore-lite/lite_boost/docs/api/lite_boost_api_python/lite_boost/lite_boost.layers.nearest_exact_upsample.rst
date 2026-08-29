lite_boost.layers.nearest_exact_upsample
========================================

.. py:function:: lite_boost.layers.nearest_exact_upsample(x, size=None, scale_factor=None)

    对 `x` 执行最近邻精确插值上采样。

    行为与 ``torch.nn.functional.interpolate(mode="nearest-exact")`` 一致。在A2等平台上所有支持的数据类型均原生执行，在300I Duo上bfloat16输入通过float32中间转换计算，因最近邻插值使用整数索引，结果与直接计算逐位一致。

    参数：
        - **x** (Tensor) - shape为 :math:`(B, C, H, W)` 的输入张量。支持float16、float32、bfloat16。
        - **size** (Union[int, tuple[int]]，可选) - 输出空间尺寸。`size` 和 `scale_factor` 二选一。默认值： ``None`` 。
        - **scale_factor** (Union[float, tuple[float]]，可选) - 空间维度的放大倍数。`size` 和 `scale_factor` 二选一。默认值： ``None`` 。

    返回：
        Tensor, 与 `x` 数据类型相同，shape由 `size` 或 `scale_factor` 决定。

    异常：
        - **ValueError** - 输入不是四维 (B, C, H, W) 时抛出。
        - **ValueError** - `size` 和 `scale_factor` 同时提供或同时未提供时抛出。
