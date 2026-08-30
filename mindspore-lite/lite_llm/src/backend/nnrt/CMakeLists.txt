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
# NNRT (Kirin NPU) backend: only NNRT-specific definitions live here.  This
# file is included via add_subdirectory(nnrt) from ../CMakeLists.txt when
# MSLITE_LLM_ENABLE_NNRT is on.  mslite_llm_nnrt_backend is an OBJECT library
# appended to MSLITE_LLM_BACKENDS; the engine links every object in that list.

file(GLOB MSLITE_LLM_NNRT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

add_library(mslite_llm_nnrt_backend OBJECT ${MSLITE_LLM_NNRT_SOURCES})
target_compile_features(mslite_llm_nnrt_backend PUBLIC cxx_std_17)
# The objects are merged into the shared engine (mslite_llm), so compile PIC.
set_target_properties(mslite_llm_nnrt_backend
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The engine include root (src/) is where "backend/common/..." style includes
# resolve; include/ carries the public C API headers used by the backends.
target_include_directories(mslite_llm_nnrt_backend PRIVATE
    ${MSLITE_LLM_ROOT}/src
    ${MSLITE_LLM_ROOT}/include
)
# The NNRT executor dlopens libneural_network_core.so + libhiai_foundation.so
# at runtime — no compile-time linkage to mindspore_lite_ndk required.
# mindspore-lite/src provides common/log.h (real-log path on OHOS).
target_include_directories(mslite_llm_nnrt_backend PRIVATE
    ${MSLITE_LLM_REPO_ROOT}/mindspore-lite/src
)

# Register this backend object in the engine-wide backend list (one PARENT_SCOPE
# hop reaches ../CMakeLists.txt, which forwards it to src/CMakeLists.txt).
list(APPEND MSLITE_LLM_BACKENDS mslite_llm_nnrt_backend)
set(MSLITE_LLM_BACKENDS ${MSLITE_LLM_BACKENDS} PARENT_SCOPE)
