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
lite_boost package
"""

# Lazily install the AscendC custom-op .run packages shipped in this wheel into
# $ASCEND_HOME_PATH on first use (stamp-guarded, idempotent; shares the stamp with
# the mindspore_lite wheel since both bundle the same .run). Wrapped so any failure
# leaves `import lite_boost` fully functional. Self-contained: must not import
# mindspore_lite.
try:
    from lite_boost._ascend_custom_ops import ensure_installed as _ensure_ascend_ops
    _ensure_ascend_ops()
except Exception:  # pylint: disable=broad-except
    pass
