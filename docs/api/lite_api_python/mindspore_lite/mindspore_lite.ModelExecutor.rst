mindspore_lite.ModelExecutor
============================

.. py:class:: mindspore_lite.ModelExecutor(executor=None)

    `ModelExecutor` 类包装多个mindspore_lite模型，并实现其推理调度。

    参数：
        - **executor** (_c_lite_wrapper.ModelExecBind, 可选) - pybind11包装的ModelExecutor类。默认值：``None``。

    .. py:method:: get_inputs()

        获取模型的所有输入Tensor。

        返回：
            list[Tensor]，模型的输入Tensor列表。

    .. py:method:: get_outputs()

        获取模型的所有输出Tensor信息。

        返回：
            list[TensorMeta]，模型的输出TensorMeta列表。

    .. py:method:: predict(inputs, outputs=None)

        推理模型。

        参数：
            - **inputs** (list[Tensor]) - 包含所有输入Tensor的顺序列表。
            - **outputs** (list[Tensor]，可选) - 包含所有输出Tensor的顺序列表。默认值：``None``。

        返回：
            list[Tensor]，模型的输出Tensor列表。

        异常：
            - **TypeError** - `inputs` 不是list类型。
            - **TypeError** - `outputs` 不是list类型。
            - **TypeError** - `inputs` 是list类型，但元素不是Tensor类型。
            - **TypeError** - `outputs` 是list类型，但元素不是Tensor类型。
            - **RuntimeError** - 预测推理模型失败。
