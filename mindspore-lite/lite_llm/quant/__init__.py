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
"""Offline quantization tools for MindSpore Lite on-device LLM.

Produces quantized weights (e.g. pure Q4_0 / Q3_0 GGUF via
``mslite-llm-quantize-gguf``) that ``mslite-llm-export`` consumes.
This package must not import ``export`` / ``models`` / ONNX graph rewrites.

Each scheme lives in its own module (e.g. ``.q3_0``) and requires an explicit
import; this ``__init__`` imports no scheme eagerly.
"""
