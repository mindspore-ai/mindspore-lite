#!/usr/bin/env python3
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
"""
Model adapter registry for ParallelManager.

Each supported model registers its class name here.  Pipeline wrapper
classes (e.g. WanT2V) can also be registered so that ParallelManager
works when passed a pipeline object directly.

To add a new model:
1. Create a directory under model/ (e.g. model/my_model/)
2. Implement boost_xxx(target) in model/my_model/boost.py
3. Add model/pipeline class name to SUPPORTED_MODELS
4. Add dispatch entry to _BOOST_REGISTRY
"""
import importlib

# Class name → adapter key.
# For WanModel the default is wan2_2; the 'vace' model_type routes
# to wan2_1 via a runtime check in detect_model_type().
SUPPORTED_MODELS = {
    'WanModel':         'wan2_2',
    'VaceWanModel':     'wan2_1',
    'WanT2V':           'wan2_2',
    'WanTI2V':          'wan2_2',
    'WanI2V':           'wan2_2',
    'WanS2V':           'wan2_2',
}

_BOOST_REGISTRY = {
    'wan2_1': ('.wan2_1.boost', 'boost_wan2_1'),
    'wan2_2': ('.wan2_2.boost', 'boost_wan2_2'),
}


def detect_model_type(model) -> str:
    """Detect model class name and map to adapter key."""
    cls_name = model.__class__.__name__

    # Wan2.1 VACE: WanModel with model_type='vace' routes to wan2_1
    if cls_name == 'WanModel' and getattr(model, 'model_type', None) == 'vace':
        return 'wan2_1'

    try:
        return SUPPORTED_MODELS[cls_name]
    except KeyError:
        raise ValueError(
            f"Unsupported model type: {cls_name}. "
            f"Supported: {list(SUPPORTED_MODELS.keys())}"
        ) from None


def setup_model(model):
    """Dispatch model setup based on detected model type.

    Returns:
        model: The same model instance, modified in-place.
    """
    module_key = detect_model_type(model)
    pkg, fn = _BOOST_REGISTRY[module_key]
    mod = importlib.import_module(pkg, package=__name__)
    getattr(mod, fn)(model)
