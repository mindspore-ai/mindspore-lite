mindspore_lite.LiteInfer
=============================

.. py:class:: mindspore_lite.LiteInfer(model_or_net, *net_inputs, context=None, model_group_id=None, config=None)

    `LiteInfer` 类接受训练模型作为输入直接执行推理。

     参数：
        - **model_or_net** (Model, Cell) - MindSpore模型或者MindSpore的nn.Cell。
        - **net_inputs** (Union[Tensor, Dataset, List, Tuple, Number, Bool]) - 表示 `net` 的输入。如果网络有多个输入，则将它们设置在一起。当其类型为 `Dataset` 时，表示 `net` 的预处理行为，数据预处理操作将被序列化，此时需要手动调整数据集脚本的batch大小来影响 `net` 输入的batch大小。目前仅支持从数据集中解析“image”列。
        - **context** (Context，可选) - 定义执行过程中用于传递选项的上下文，``None`` 表示使用CPU的上下文。默认值：``None``。
        - **model_group_id** (int，可选) - 用于绑定模型id至模型群组。默认值：``None``。
        - **config** (dict，可选) - 当后端为“lite”时使用。配置信息包含两部分，config_path（'configPath'，str）和config_item（str，dict）。config_item优先级高于config_path。设置用于推理的rank table文件，配置文件的内容如下：

        .. code-block::
            
            [ascend_context]
            rank_table_file=[path_a](storage initial path of the rank table file)
            
        当设置了

        .. code-block::

            config = {"ascend_context" : {"rank_table_file" : "path_b"}}

        配置中的path_b将会被用于编译模型。默认值：``None``。
    
    异常：
        - **ValueError** - `model_or_net` 不是MindSpore模型或者MindSpore的nn.Cell。
    
    .. py:method:: get_inputs()

        获取模型的所有输入张量。详情见 :func:`mindspore_lite.model.get_inputs`。
    
    .. py:method:: predict(inputs)

        模型推理。详情见 :func:`mindspore_lite.model.predict`。
    
    .. py:method:: resize(inputs, dims)

        调整输入的形状。详情见 :func:`mindspore_lite.model.resize`。
