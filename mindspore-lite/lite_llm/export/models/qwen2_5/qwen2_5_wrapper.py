# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Qwen2.5 NNRT ONNX wrapper (bias projections, no per-head Q/K norm).

Qwen2.5 attention uses the default base behaviour: projections carry q/k/v
biases (handled by ``nn.Linear`` automatically — no special wrapper code),
and there is no per-head Q/K RMSNorm.  The class exists so the exporter names
a concrete subclass and remains a seam for future Qwen2 variants.
"""

from models._base.nnrt_decoder_wrapper import NnrtDecoderWrapper


class Qwen2NnrtWrapper(NnrtDecoderWrapper):
    """NNRT wrapper for Qwen2.5 — uses the base attention path unchanged."""
