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
"""aclprof ctypes wrapper for phase-separated profiling (mindspore lite ge)."""
import ctypes
import os

_LIB = None

# dataTypeConfig bitmask (acl_prof.h)
ACL_PROF_ACL_API = 0x0001
ACL_PROF_TASK_TIME = 0x0002
ACL_PROF_AICORE_METRICS = 0x0004
ACL_PROF_HCCL_TRACE = 0x0020
ACL_PROF_TASK_TIME_L0 = 0x0800
ACL_AICORE_PIPE_UTILIZATION = 1
ACL_SUCCESS = 0

# CANN 9.1.0 (env91.sh): real implementation in libmsprofiler.so.
# Prefer the active CANN install from ASCEND_HOME_PATH (set by the CANN env.sh);
# the remaining entries are hard-coded fallbacks.
_ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH")
_LIB_PATHS = [
    os.path.join(_ASCEND_HOME, "aarch64-linux/lib64/libmsprofiler.so"),
]


def _lib():
    """Load and cache the msprofiler CDLL (lazy init)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    for c in _LIB_PATHS:
        try:
            _LIB = ctypes.CDLL(c)
            break
        except OSError:
            continue
    if _LIB is None:
        raise RuntimeError("libacl_prof.so not found")
    _LIB.aclprofInit.restype = ctypes.c_int
    _LIB.aclprofInit.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    _LIB.aclprofFinalize.restype = ctypes.c_int
    _LIB.aclprofStart.restype = ctypes.c_int
    _LIB.aclprofStart.argtypes = [ctypes.c_void_p]
    _LIB.aclprofStop.restype = ctypes.c_int
    _LIB.aclprofStop.argtypes = [ctypes.c_void_p]
    _LIB.aclprofCreateConfig.restype = ctypes.c_void_p
    _LIB.aclprofCreateConfig.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
        ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64,
    ]
    _LIB.aclprofDestroyConfig.restype = ctypes.c_int
    _LIB.aclprofDestroyConfig.argtypes = [ctypes.c_void_p]
    return _LIB


def prof_init(result_path):
    lib = _lib()
    p = os.path.abspath(result_path)
    os.makedirs(p, exist_ok=True)
    pb = p.encode("utf-8")
    ret = lib.aclprofInit(pb, len(pb))
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"aclprofInit failed rc={ret} path={p}")
    return ret


def prof_start(device_id):
    """Start aclprof collection on the device; return the config handle."""
    lib = _lib()
    devs = (ctypes.c_uint32 * 1)(device_id)
    cfg = lib.aclprofCreateConfig(
        devs, 1, ACL_AICORE_PIPE_UTILIZATION, None,
        ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_HCCL_TRACE | ACL_PROF_TASK_TIME_L0)
    if not cfg:
        raise RuntimeError("aclprofCreateConfig failed")
    ret = lib.aclprofStart(cfg)
    if ret != ACL_SUCCESS:
        lib.aclprofDestroyConfig(cfg)
        raise RuntimeError(f"aclprofStart failed rc={ret}")
    return cfg


def prof_stop(cfg):
    lib = _lib()
    ret = lib.aclprofStop(cfg)
    lib.aclprofDestroyConfig(cfg)
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"aclprofStop failed rc={ret}")
    return ret


def prof_finalize():
    ret = _lib().aclprofFinalize()
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"aclprofFinalize failed rc={ret}")
    return ret
