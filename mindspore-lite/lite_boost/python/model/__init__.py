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
"""
Model adapter registry for ParallelManager.

Each supported model registers its adapter key and setup function here.
To add a new model:
1. Create a directory under model/ (e.g. model/my_model/)
2. Implement boost_xxx(model) in model/my_model/boost.py
3. Add model class name to SUPPORTED_MODELS
4. Add dispatch branch in setup_model()
"""
SUPPORTED_MODELS = {
    'WanModel': 'wan2_1',
    'VaceWanModel': 'wan2_1',
}


def detect_model_type(model) -> str:
    """Detect model class name and map to adapter key."""
    cls_name = model.__class__.__name__
    if cls_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[cls_name]
    raise ValueError(
        f"Unsupported model type: {cls_name}. "
        f"Supported: {list(SUPPORTED_MODELS.keys())}"
    )


def setup_model(model):
    """Dispatch model setup based on detected model type.

    Returns:
        model: The same model instance, modified in-place.
    """
    model_type = detect_model_type(model)

    if model_type == 'wan2_1':
        from .wan2_1.boost import boost_wan2_1
        boost_wan2_1(model)
    else:
        raise ValueError(f"Unknown model adapter: {model_type}")
