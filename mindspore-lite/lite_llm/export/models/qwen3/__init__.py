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
"""Qwen3 (MiniMind-3) model exporter: patch + skeleton export + GGUF injection."""

from .qwen3_exporter import Qwen3Onnx, export_qwen3
from .qwen3_gguf_loader import gguf_loader
from .qwen3_patch import apply_qwen3_patch, patch_gguf_config_mapping

__all__ = ["Qwen3Onnx", "export_qwen3", "gguf_loader", "apply_qwen3_patch", "patch_gguf_config_mapping"]
