lite_boost
==========

LiteBoost is an inference acceleration toolkit for Ascend hardware, built on top of MindSpore Lite. It provides high-performance custom operators, multi-card parallel inference, quantization and sparsity, and other inference acceleration capabilities.

Parallel
--------

.. autosummary::
    :toctree: lite_boost
    :nosignatures:
    :template: classtemplate.rst

    lite_boost.parallel.initialize_usp
    lite_boost.BoostManager

Layers
--------

.. autosummary::
    :toctree: lite_boost
    :nosignatures:
    :template: classtemplate.rst

    lite_boost.layers.rms_norm
    lite_boost.layers.nearest_exact_upsample

Operators
------------

.. autosummary::
    :toctree: lite_boost
    :nosignatures:
    :template: classtemplate.rst

    lite_boost.ops.rain_fusion_attention
    lite_boost.ops.sparse_attention
    lite_boost.ops.recurrent_gated_delta_rule
