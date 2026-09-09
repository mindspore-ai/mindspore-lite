#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""BooguImage boost package: TP in ``tp.py``, Ulysses SP in ``usp.py``."""

__all__ = [
    "boost_booguimage",
    "shard_boogu_transformer",
    "patch_transformer_forwards",
    "boost_sp_single_stream",
    "sp_single_stream_block_forward",
    "tp_single_stream_processor_call",
    "tp_single_stream_processor_call_flash",
    "tp_double_stream_processor_call",
    "tp_double_stream_processor_call_flash",
    "tp_feed_forward_forward",
    "tp_vae_decode",
    "tp_pipeline_call",
]

from .boost import boost_booguimage
from .tp import shard_boogu_transformer, patch_transformer_forwards
from .usp import boost_sp_single_stream, sp_single_stream_block_forward
from .tp import (
    tp_single_stream_processor_call,
    tp_single_stream_processor_call_flash,
    tp_double_stream_processor_call,
    tp_double_stream_processor_call_flash,
    tp_feed_forward_forward,
    tp_vae_decode,
    tp_pipeline_call,
)
