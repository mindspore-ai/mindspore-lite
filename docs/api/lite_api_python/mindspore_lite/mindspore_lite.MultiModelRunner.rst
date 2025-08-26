mindspore_lite.MultiModelRunner
===============================

.. py:class:: mindspore_lite.MultiModelRunner()

    `MultiModelRunner` 用于创建包含多个Model的mindir，并提供调度多个模型的方式。

    .. py:method:: build_from_file(model_path, model_type, context=None, config_path="", config_dict: dict = None)

        从文件加载并构建模型。

        参数：
            - **model_path** (str) - 定义输入模型文件的路径，例如："/home/user/model.mindir"。模型应该使用.mindir作为后缀。
            - **model_type** (ModelType) - 定义输入模型文件的类型。选项有 ``ModelType::MINDIR`` 。有关详细信息，请参见 `模型类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.ModelType.html>`_ 。
            - **context** (Context，可选) - 定义上下文，用于在执行期间传递选项。默认值： ``None``，表示设置target为cpu的Context。
            - **config_path** (str，可选) - 定义配置文件的路径，用于在构建模型期间传递用户定义选项。在以下场景中，用户可能需要设置参数。例如："/home/user/config.txt"。默认值： ``""`` 。

              - **用法1** - 进行混合精度推理的设置，配置文件内容及说明如下：

                .. code-block::

                    [execution_plan]
                    [op_name1]=data_type:float16（名字为op_name1的算子设置数据类型为float16）
                    [op_name2]=data_type:float32（名字为op_name2的算子设置数据类型为float32）

              - **用法2** - 在使用GPU推理时，进行TensorRT设置，配置文件内容及说明如下：

                .. code-block::

                    [ms_cache]
                    serialize_path=[serialization model path]（序列化模型的存储路径）
                    [gpu_context]
                    input_shape=input_name:[input_dim]（模型输入维度，用于动态shape）
                    dynamic_dims=[min_dim~max_dim]（模型输入的动态维度范围，用于动态shape）
                    opt_dims=[opt_dim]（模型最优输入维度，用于动态shape）

            - **config_dict** (dict，可选) - 配置参数字典，当使用该字典配置参数时，优先级高于配置文件。默认值：``None``。

              推理配置rank table。配置文件中的内容及说明如下：

              .. code-block::

                  [ascend_context]
                  rank_table_file=[path_a]（使用路径a的rank table）

              同时配置参数字典中如下：

              .. code-block::

                  config_dict = {"ascend_context" : {"rank_table_file" : "path_b"}}

              那么配置参数字典中路径b的rank table将覆盖配置文件中的路径a的rank table。

        异常：
            - **TypeError** - `model_path` 不是str类型。
            - **TypeError** - `model_type` 不是ModelType类型。
            - **TypeError** - `context` 既不是Context类型也不是 ``None`` 。
            - **TypeError** - `config_path` 不是str类型。
            - **RuntimeError** - `model_path` 文件路径不存在。
            - **RuntimeError** - `config_path` 文件路径不存在。
            - **RuntimeError** - 从 `config_path` 加载配置文件失败。
            - **RuntimeError** - 从文件加载并构建模型失败。
    
    .. py:method:: get_model_executor()

        获取runner创建的ModelExecutor对象

        返回：
            ModelExecutor对象的列表
