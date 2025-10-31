# Copyright 2025 Huawei Technologies Co., Ltd
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
utils for test
"""
import time
import contextlib
import pytest


class ScopeTimeRecord:
    """
    time record
    """

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.finish_time = time.perf_counter()
        self.duration = (self.finish_time - self.start_time) * 1000

@contextlib.contextmanager
def expect_error(errors, *args, **kwargs):
    if errors:
        with pytest.raises(errors, *args, **kwargs) as exc_info:
            yield exc_info
    else:
        yield
