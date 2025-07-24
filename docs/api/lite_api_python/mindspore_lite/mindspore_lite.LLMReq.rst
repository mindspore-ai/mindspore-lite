mindspore_lite.LLMReq
======================

.. py:class:: mindspore_lite.LLMReq(prompt_cluster_id: int, req_id: int, prompt_length: int)

    LLMEngine的请求类，用于表示多轮推理任务。

    参数：
        - **prompt_cluster_id** (int) - 该推理任务的集群id。
        - **req_id** (int) - 该推理任务的请求id。
        - **prompt_length** (int) - 该推理任务的提示词长度。

    .. py:method:: next_req_id

        获取下一个请求id。

    .. py:method:: req_id
        :property:

        该推理任务的请求id。

    .. py:method:: prompt_length
        :property:

        该推理任务的提示词长度。

    .. py:method:: prompt_cluster_id
        :property:

        LLMEngine中该推理任务的提示词集群id。

    .. py:method:: decoder_cluster_id
        :property:

        LLMEngine中该推理任务的解码器集群id。

    .. py:method:: prefix_id
        :property:

        LLMEngine中该推理任务的解码器集群id前缀。

    .. py:method:: sequence_length
        :property:

        LLMEngine中该推理任务的解码器序列长度。
