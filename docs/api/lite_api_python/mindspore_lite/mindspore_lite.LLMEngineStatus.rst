mindspore_lite.LLMEngineStatus
===============================

.. py:class:: mindspore_lite.LLMEngineStatus(status)

    LLMEngine的状态类，用于表示推理任务状态。

    .. py:method:: empty_max_prompt_kv
        :property:

        获取该LLMEngine的prompt KV cache空计数。

    .. py:method:: num_free_blocks
        :property:

        获取PagedAttention空闲块数量。

    .. py:method:: num_total_blocks
        :property:

        获取PagedAttention块总数。

    .. py:method:: block_size
        :property:

        获取PagedAttention块大小。
