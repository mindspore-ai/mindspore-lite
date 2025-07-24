mindspore_lite.LLMEngine
==================================

.. py:class:: mindspore_lite.LLMEngine(role: LLMRole, cluster_id: int, batch_mode="auto")

    `LLMEngine` 类定义了一个MindSpore Lite的LLMEngine，用于加载和管理大语言模型，以及响应调度和推理请求。

    参数：
        - **role** (LLMRole) - LLMEngine对象所属的角色。
        - **cluster_id** (int) - LLMEngine对象所属的集群id。
        - **batch_mode** (str，可选) - 决定batching请求是由框架生成（"auto"模式）还是用户手动生成（"manual"模式）。默认值：``"auto"``。

    异常：
        - **TypeError** - `role` 不是LLMRole类型。
        - **TypeError** - `cluster_id` 不是int类型。

    .. py:method:: add_model(self, model_paths: Union[Tuple[str], List[str]], options: Dict[str, str], postprocess_model_path=None)

        在LLMEngine中添加一个模型。

        参数：
            - **model_paths** (Union[Tuple[str], List[str]]) - 模型路径。
            - **options** (Dict[str, str]) - LLMEngine对象的初始化选项。
            - **postprocess_model_path** (Union[str, None]，可选) - 后处理模型路径。默认值：``None``。

        异常：
            - **TypeError** - `model_paths` 不是list或者tuple。
            - **TypeError** - `model_paths` 是list或者tuple，但其中的元素不是str类型。
            - **TypeError** - `options` 不是dict。
            - **RuntimeError** - 添加模型失败。
        
    .. py:method:: batch_mode
        :property:

        获取LLMEngine对象的批处理模式。

    .. py:method:: cluster_id
        :property:

        获取LLMEngine对象的集群id。

    .. py:method:: complete_request(llm_req: LLMReq)

        完成推理请求。

        参数：
            - **llm_req** (LLMReq) - LLMEngine请求。

        异常：
            - **TypeError** - `llm_req` 不是LLMReq类型。
            - **RuntimeError** - LLMEngine对象未初始化。

    .. py:method:: fetch_status()

        获取LLMEngine状态。

        返回：
            LLMEngine状态，类型为LLMEngineStatus。

        异常：
            - **RuntimeError** - LLMEngine对象未初始化。

    .. py:method:: finalize()

        析构LLMEngine。

    .. py:method:: init(options: Dict[str, str])

        初始化LLMEngine。

        参数：
            - **options** (Dict[str, str]) - LLMEngine对象的初始化选项。

        异常：
            - **TypeError** - `options` 不是dict。
            - **RuntimeError** - 初始化LLMEngine失败。
    
    .. py:method:: link_clusters(clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=-1)

        连接集群。

        参数：
            - **clusters** (Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]]) - 集群。
            - **timeout** (int，可选) - 超时秒数。默认值：``-1``。

        返回：
            (Status, tuple[Status])，分别表示所有集群的连接状态，和每个集群的连接状态。

        异常：
            - **TypeError** - `clusters` 不是list或者tuple，或者其中的内容不是LLMClusterInfo类型。
            - **RuntimeError** - LLMEngine对象未初始化或者初始化失败。

    .. py:method:: role
        :property:

        获取LLMEngine对象的角色。

    .. py:method:: unlink_clusters(clusters: Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]], timeout=-1)

        断连集群。

        参数：
            - **clusters** (Union[List[LLMClusterInfo], Tuple[LLMClusterInfo]]) - 集群。
            - **timeout** (int，可选) - 超时秒数。默认值：``-1``。

        返回：
            (Status, tuple[Status])，分别表示所有集群的断连状态，和每个集群的断连状态。

        异常：
            - **TypeError** - `clusters` 不是list或者tuple，或者其中的内容不是LLMClusterInfo类型。
            - **RuntimeError** - 断连失败。
