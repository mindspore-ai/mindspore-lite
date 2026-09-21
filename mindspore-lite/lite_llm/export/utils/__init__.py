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
"""Export utilities and the source-checkout custom operator bootstrap."""

import importlib.util
import os
import sys


def ensure_custom_ops():
    """Resolve the vendored adapters only when no installed package is visible."""
    if importlib.util.find_spec("torch_custom") is not None:
        return
    custom_ops = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "custom_ops"))
    if os.path.isdir(custom_ops) and custom_ops not in sys.path:
        sys.path.insert(0, custom_ops)
