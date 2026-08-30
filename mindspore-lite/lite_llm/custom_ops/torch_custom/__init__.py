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
"""Torch framework adapter layer for Kirin custom operators.

Each operator gets one module here (``ms_<op>.py``) defining its
``torch.autograd.Function`` (eager reference + ONNX symbolic), named after the
operator (``Ms<Op>``, e.g. ``MsRmsNorm``). Consumers may import the package as
a whole (``from torch_custom import MsRmsNorm``) or the module directly
(``from torch_custom.ms_rms_norm import MsRmsNorm``).
"""

from __future__ import annotations

from .ms_rms_norm import MsRmsNorm
from .ms_add_softmax import MsAddSoftmax
from .ms_group_matmul import MsGroupMatmul
from .ms_rotary_pos_emb import MsRotaryPosEmb
from .ms_scatter_nd import MsScatterND
from .ms_add_rms_norm import MsAddRmsNorm
from .ms_quant4_n0_group32 import MsQuant4N0Group32
from .ms_float_cast_int import MsFloatCastInt

__all__ = [
    'MsRmsNorm', 'MsAddSoftmax', 'MsGroupMatmul', 'MsRotaryPosEmb', 'MsScatterND', 'MsAddRmsNorm',
    'MsQuant4N0Group32', 'MsFloatCastInt',
]
