mindspore_lite.LLMRole
=======================

.. py:class:: mindspore_lite.LLMRole()

    LLMEngine的角色。当LLMEngine通过KVCache提升推理性能时，生成过程包括一次完整推理和n次增量推理，涉及完整模型和增量模型。当完整模型和增量模型部署在不同节点上时，完整模型所在节点的角色为“Prompt”，增量模型所在节点的角色为“Decoder”。
