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
# lite_llm engine: mslite_llm shared library (libmindspore-lite-llm.so).
#
# Sources are collected per subsystem with file(GLOB ... CONFIGURE_DEPENDS), so
# new .cc files are picked up automatically on the next build. The NNRT backend
# directory is compiled in only when MSLITE_LLM_ENABLE_NNRT is on
# (backend-per-switch, directory per backend API).

# Module version is read once in the top-level CMakeLists.txt from
# export/version.txt (single source of truth, also used by the wheel and the
# release archive).

# Backends are a subsystem of their own: common/ contract + per-backend
# implementations, exposed to both engine libraries as OBJECT libraries.
# MSLITE_LLM_BACKENDS collects every backend OBJECT library; each backend
# directory appends its own object (see backend/nnrt/CMakeLists.txt) and the
# engine links the whole list.
set(MSLITE_LLM_BACKENDS "")
add_subdirectory(backend)

file(GLOB LLM_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/pipeline/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/manifest/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler/*.cc
)

add_library(mslite_llm SHARED ${LLM_SOURCES})
set_target_properties(mslite_llm PROPERTIES
    OUTPUT_NAME mindspore-lite-llm
)
target_compile_features(mslite_llm PUBLIC cxx_std_17)

# Merge the backend OBJECT libraries (each compiled once, linked into both
# engines): the list is filled by backend/ and its subdirectories.
target_link_libraries(mslite_llm PRIVATE ${MSLITE_LLM_BACKENDS})

# Note: unit tests link against internal C++ classes (MslPackageReader etc.),
# so the shared library currently exports all symbols; the MSLLM_API-marked
# C ABI remains the stable public surface.  A version script restricting the
# dynamic symbol table to the C API would require rewriting the tests.

target_include_directories(mslite_llm PUBLIC
    ${MSLITE_LLM_ROOT}/include
)

target_include_directories(mslite_llm PRIVATE
    ${MSLITE_LLM_ROOT}/src
    ${MSLITE_LLM_ROOT}/include
)

# Static engine library: the device example links it statically so the
# packaged mslite-chat has no runtime .so dependency on the engine (the
# BinRunner memory loader on HarmonyOS cannot dlopen engine deps).
add_library(mslite_llm_static STATIC ${LLM_SOURCES})
set_target_properties(mslite_llm_static PROPERTIES
    OUTPUT_NAME mindspore-lite-llm
)
target_compile_features(mslite_llm_static PUBLIC cxx_std_17)

target_link_libraries(mslite_llm_static PRIVATE ${MSLITE_LLM_BACKENDS})

target_include_directories(mslite_llm_static PUBLIC
    ${MSLITE_LLM_ROOT}/include
)

target_include_directories(mslite_llm_static PRIVATE
    ${MSLITE_LLM_ROOT}/src
)

# ─── Install rules (consumed by build.sh packaging) ────────────────────
# Engine libs land in lib/, public headers in include/; the example binary
# (bin/) is installed by examples/CMakeLists.txt.  build.sh runs
# `cmake --install` into a staging dir and collects the wheel / .run
# packages itself — C++/Python/op.run joint packaging is not CPack's job.
install(TARGETS mslite_llm LIBRARY DESTINATION lib)
install(TARGETS mslite_llm_static ARCHIVE DESTINATION lib)
install(DIRECTORY ${MSLITE_LLM_ROOT}/include/llm DESTINATION include)
