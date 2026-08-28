lite_boost.layers.rms_norm
==========================

.. py:function:: lite_boost.layers.rms_norm(x, gamma, eps=1e-6)

    对 `x` 的最后一维执行逐行RMS归一化。

    计算 :math:`y = x / RMS(x) * gamma`，其中 :math:`RMS(x) = \sqrt{mean(x^2)}`，:math:`mean(x^2)` 是 `x` 在最后一维上的平方均值。通常用于替代Wan系列VAE ``RMS_norm`` 层中展开的 ``F.normalize(x, dim) * sqrt(dim) * gamma`` 计算链。

    参数：
        - **x** (Tensor) - shape为 :math:`(N, C)` 的二维输入张量，归一化维度是最后一维。支持float16、float32，在A2等平台上额外支持bfloat16。
        - **gamma** (Tensor) - 逐列缩放因子，shape为 :math:`(C,)`，与 `x` 的最后一维匹配。
        - **eps** (float，可选) - 为数值稳定性添加到分母的值。默认值： ``1e-6`` 。

    返回：
        Tensor, 与 `x` 相同shape和数据类型的归一化结果。

    异常：
        - **ValueError** - 输入不是二维（仅支持 (N, C) 输入），或 `gamma` 不是一维且与 `x` 最后一维不匹配时抛出。
        - **ValueError** - 在300I Duo上 `x` 为bfloat16，或 `x` 的最后一维小于16时抛出。
