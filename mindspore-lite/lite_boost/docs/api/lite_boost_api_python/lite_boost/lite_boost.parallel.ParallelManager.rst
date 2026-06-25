lite_boost.parallel.ParallelManager
=====================================

.. py:class:: lite_boost.parallel.ParallelManager(target)

    对支持的模型进行原地修改，使其支持分布式并行推理。

    :class:`ParallelManager` 封装一个支持的模型或流水线对象，并对其进行原地补丁替换，以实现多NPU设备的并行推理。根据检测到的模型组件，自动应用以下两种并行策略：

    - **Ulysses序列并行（USP）** 用于DiT模型 - 补丁替换 ``forward`` 方法和注意力层，通过 ``all_to_all`` 通信实现序列维度并行，每张卡持有完整模型权重，仅对序列的一个切片进行计算。
    - **数据并行（DP）时间切片** 用于VAE模型 - 将 ``vae.encode`` 和 ``vae.decode`` 替换为DP时间切片版本，沿时间维度将视频切分为重叠的帧片段，分发到各卡独立处理，最后收集拼接为完整结果。

    当传入流水线对象（如 ``WanT2V`` 或 ``WanTI2V``）时，两种策略同时生效，DiT模型应用USP，VAE应用DP。

    模型在原地修改后原样返回，因此所有已有的属性和方法（ ``.to`` 、``.cpu`` 、``.eval`` 等）均可正常使用。

    参数：
        - **target** (object) - 需要并行化的支持流水线对象，支持的类包括 ``WanT2V`` 和 ``WanTI2V`` 。

    返回：
        object，与输入相同的实例，已原地修改为USP补丁后的forward和注意力方法（DiT）以及DP补丁后的encode/decode方法（VAE）。

    异常：
        - **RuntimeError** - 模型类型不被lite_boost支持时抛出。
