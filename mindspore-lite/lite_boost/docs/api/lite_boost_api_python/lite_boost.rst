lite_boost
==========

LiteBoost是MindSpore Lite面向昇腾硬件的推理加速工具包，提供高性能自定义算子、多卡并行推理、量化稀疏等推理加速能力。

并行
----

.. mscnautosummary::
    :toctree: lite_boost

    lite_boost.parallel.initialize_usp
    lite_boost.BoostManager

层
--

.. mscnautosummary::
    :toctree: lite_boost

    lite_boost.layers.rms_norm
    lite_boost.layers.nearest_exact_upsample
    lite_boost.layers.rope_apply
    lite_boost.layers.flash_attention

算子
----

.. mscnautosummary::
    :toctree: lite_boost

    lite_boost.ops.rain_fusion_attention
    lite_boost.ops.sparse_attention
    lite_boost.ops.recurrent_gated_delta_rule
    lite_boost.ops.chunk_gated_delta_rule
