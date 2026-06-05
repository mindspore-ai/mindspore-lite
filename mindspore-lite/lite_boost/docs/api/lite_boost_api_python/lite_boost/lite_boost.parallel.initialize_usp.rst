lite_boost.parallel.initialize_usp
====================================

.. py:function:: lite_boost.parallel.initialize_usp()

    初始化并行推理所需的HCCL分布式环境。

    该函数配置NPU运行时设置，并通过读取以下环境变量初始化HCCL分布式进程组：

    - ``RANK`` - 当前进程的本地rank，默认值 ``0`` 。
    - ``WORLD_SIZE`` - 分布式进程总数，默认值 ``1`` 。
    - ``MASTER_ADDR`` - 主节点IP地址，默认值 ``"127.0.0.1"`` 。
    - ``MASTER_PORT`` - 主节点端口，默认值 ``29502`` 。
    - ``NUM_THREADS`` - 每个进程的CPU线程数，默认值 ``24`` 。

    若分布式进程组尚未初始化，该函数将使用 ``hccl`` 后端进行初始化。初始化完成后，将 ``RANK`` 对应的NPU设备设置为当前活跃设备。

    .. note::

        该函数必须在构造 :class:`ParallelManager` 之前调用，通常在分布式训练或推理脚本的入口处执行。

    异常：
        - **RuntimeError** - HCCL进程组初始化失败时抛出。
