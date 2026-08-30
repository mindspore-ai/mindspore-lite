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
# backend subsystem: common/ holds the engine<->backend contract (Backend
# abstraction + factory); per-backend implementations live in their own
# subdirectory with a CMakeLists.txt that only adds backend-specific
# definitions (sources, includes) to mslite_llm_backend.
#
# mslite_llm_backend is an OBJECT library: both the shared and the static
# engine libraries in ../CMakeLists.txt link it, so the backend sources are
# compiled once and the objects are merged into each consumer.

file(GLOB MSLITE_LLM_BACKEND_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
)

add_library(mslite_llm_backend OBJECT ${MSLITE_LLM_BACKEND_SOURCES})
target_compile_features(mslite_llm_backend PUBLIC cxx_std_17)
# The objects are merged into the shared engine (mslite_llm), so compile PIC.
set_target_properties(mslite_llm_backend
    PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The engine include root (src/) is where "backend/common/..." style includes
# resolve; include/ carries the public C API headers used by the backends.
target_include_directories(mslite_llm_backend PRIVATE
    ${MSLITE_LLM_ROOT}/src
    ${MSLITE_LLM_ROOT}/include
)

# Backend-per-switch: each backend directory is added only when enabled
# (the nnrt backend is compiled in only when MSLITE_LLM_ENABLE_NNRT is on).
if(MSLITE_LLM_ENABLE_NNRT)
    add_subdirectory(nnrt)
endif()

# Append the common contract object to the backend list and propagate the
# whole list (including what backend subdirectories appended above) to the
# engine directory.
list(APPEND MSLITE_LLM_BACKENDS mslite_llm_backend)
set(MSLITE_LLM_BACKENDS ${MSLITE_LLM_BACKENDS} PARENT_SCOPE)
