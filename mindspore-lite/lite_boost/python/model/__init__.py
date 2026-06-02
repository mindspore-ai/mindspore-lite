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

Wan-series models (class name starts with 'Wan') are auto-detected:
  - class in _WAN22_CLASSES or has low_noise_model → wan2_2
  - otherwise → wan2_1
"""
import importlib

# Wan2.2-specific attribute: some Wan2.2 pipelines have low_noise_model.
_WAN22_MARKER_ATTR = 'low_noise_model'

# Wan2.2 classes that do NOT have low_noise_model (e.g. WanTI2V, WanS2V).
_WAN22_CLASSES = frozenset({'WanTI2V', 'WanS2V'})

_BOOST_REGISTRY = {
    'wan2_1': ('.wan2_1.boost', 'boost_wan2_1'),
    'wan2_2': ('.wan2_2.boost', 'boost_wan2_2'),
}


def detect_model_type(model) -> str:
    """Detect model type for Wan-series models."""
    cls_name = model.__class__.__name__

    if not cls_name.startswith('Wan'):
        raise ValueError(
            f"Unsupported model type: {cls_name}. "
            f"Expected a Wan-series pipeline or model."
        )

    if cls_name in _WAN22_CLASSES or hasattr(model, _WAN22_MARKER_ATTR):
        return 'wan2_2'
    return 'wan2_1'


def setup_model(model):
    """Dispatch model setup based on detected model type."""
    module_key = detect_model_type(model)
    pkg, fn = _BOOST_REGISTRY[module_key]
    mod = importlib.import_module(pkg, package=__name__)
    getattr(mod, fn)(model)
