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
"""Shared NNRT ONNX wrapper base for Qwen-style decoder LLMs.

Two orthogonal variation axes:

* architecture (Qwen2 / Qwen3 / MiniCPM / ...) — subclass
  ``NnrtDecoderWrapper`` (select an attention adapter via ``attn_module``,
  override its ``apply_qk_norm`` hook for per-head Q/K norm variants);
* fused-operator set (``torch_custom`` kernel combination per model spec /
  target) — subclass ``NnrtOpSet`` and override the differing primitives,
  then inject via ``wrapper(op_set=...)``.

The adapter hierarchy mirrors the HF ``*ForCausalLM`` module layout so
traced node / initializer names stay identical to the pre-refactor exports
(the GGUF loaders map GGUF tensors by ONNX node name).

Adding a new model or supporting a new transformers version no longer
requires touching transformers internals — only the wrapper subclass, if
at all.
"""

from .nnrt_decoder_wrapper import (
    NnrtAttention,
    NnrtDecoderCore,
    NnrtDecoderLayer,
    NnrtDecoderWrapper,
    NnrtOpSet,
    NnrtRmsNorm,
)

__all__ = [
    "NnrtAttention",
    "NnrtDecoderCore",
    "NnrtDecoderLayer",
    "NnrtDecoderWrapper",
    "NnrtOpSet",
    "NnrtRmsNorm",
]
