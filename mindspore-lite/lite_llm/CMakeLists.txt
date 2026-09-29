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
cmake_minimum_required(VERSION 3.12)

# ─── Options ───────────────────────────────────────────────────────────────
# Backend switches: NNRT (Kirin NPU) is the only backend currently compiled in;
# future backends get their own switch (MSLITE_LLM_ENABLE_QNN / _METAL / _GPU).
# build.sh passes an explicit -D based on the build form (off on host, on with
# -b nnrt); the default here makes a bare `cmake -S .` produce a usable engine.
option(MSLITE_LLM_ENABLE_NNRT "Enable the NNRT (Kirin NPU) backend" ON)
message(STATUS "MSLITE_LLM_ENABLE_NNRT = ${MSLITE_LLM_ENABLE_NNRT}")

# The nnrt backend targets the Kirin NPU, i.e. a HarmonyOS cross-compile; derive
# the cross-compile flag from the backend instead of a separate option.  The
# NDK check must run before project(): the toolchain file is loaded there.
if(MSLITE_LLM_ENABLE_NNRT)
    set(MSLITE_LLM_CROSS_COMPILE_HARMONYOS ON)
    if(NOT DEFINED ENV{OHOS_NDK} OR NOT IS_DIRECTORY "$ENV{OHOS_NDK}")
        message(FATAL_ERROR
            "MSLITE_LLM_ENABLE_NNRT is on but OHOS_NDK is not "
            "set or not a directory. Set OHOS_NDK to the native root "
            "(<SDK>/native) and rerun configure.")
    endif()
    set(MSLITE_LLM_TOOLCHAIN "$ENV{OHOS_NDK}/build/cmake/ohos.toolchain.cmake")
    if(NOT EXISTS "${MSLITE_LLM_TOOLCHAIN}")
        message(FATAL_ERROR "OHOS toolchain not found: ${MSLITE_LLM_TOOLCHAIN}")
    endif()
    if(NOT CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_TOOLCHAIN_FILE "${MSLITE_LLM_TOOLCHAIN}" CACHE FILEPATH
            "OHOS NDK toolchain")
    endif()
endif()

project(mslite_llm LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Module-wide compile options (engine, backends, tests, examples): the
# per-target copies live here once instead of repeated in every subdirectory.
add_compile_options(
    -Wall -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-copy
)
# Per-config flags: keep each generator expression single-argument — a genex
# value containing a space is not split for add_compile_options().
add_compile_options($<$<CONFIG:Release>:-O2>)
add_compile_options($<$<CONFIG:Debug>:-g>)
add_compile_options($<$<CONFIG:Debug>:-O0>)

# ─── Path anchors ──────────────────────────────────────────────────────────
# Sub-module CMakeLists.txt resolve repo paths from these anchors instead of
# recomputing relative paths from their own directory.
set(MSLITE_LLM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(MSLITE_LLM_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# Module version: single source of truth is export/version.txt (used by the
# engine build, the wheel and the release archive name).
file(READ ${MSLITE_LLM_ROOT}/export/version.txt MSLITE_LLM_VERSION)
string(STRIP ${MSLITE_LLM_VERSION} MSLITE_LLM_VERSION)

# ─── securec (memcpy_s / memset_s / snprintf_s) ────────────────────────────
# Reuse the outer repo's recipe (cmake/external_libs/securec.cmake): same
# openEuler libboundscheck v1.1.16, plus its CMake patch (install rules,
# Windows branch) and the shared .mslib hash cache. lite_llm therefore
# requires the outer repo layout (cmake/ + third_party/ two levels up).
set(TOP_DIR "${MSLITE_LLM_REPO_ROOT}")
find_program(Patch_EXECUTABLE patch)   # mindspore_add_pkg applies securec.patch001 with it
if(NOT Patch_EXECUTABLE)
    message(FATAL_ERROR "'patch' tool not found: required to apply securec.patch001")
endif()
set(BUILD_LITE ON)                     # securec.cmake's cross branches are BUILD_LITE-gated
if(MSLITE_LLM_ENABLE_NNRT)
    set(TOOLCHAIN_NAME ohos)           # activates the OHOS recipe via $ENV{OHOS_NDK}
endif()
include("${TOP_DIR}/cmake/utils.cmake")
include("${TOP_DIR}/cmake/external_libs/securec.cmake")
# Restore the outer-repo switches set above: they are meaningful only to
# securec.cmake. If BUILD_LITE leaked onward, gtest.cmake (included later by
# tests/) would provision gtest 1.8.1 instead of the 1.12.1 baseline — and
# MslGoldenTest's GTEST_SKIP() (test_msl_golden.cc:199, skips when
# golden_v1.msl is not provisioned) requires gtest >= 1.10.
unset(BUILD_LITE)
unset(TOOLCHAIN_NAME)

# ─── Sub-modules ──────────────────────────────────────────────────────────
# src/      — mslite_llm static library (engine)
# examples/ — public API sample CLI (mslite-chat), always built; also the
#             device verification entry for ST (loads a .msl, runs inference)
# tests/    — gtest unit tests, optional
add_subdirectory(src)
add_subdirectory(examples)

if(MSLITE_LLM_BUILD_TESTS)
    # enable_testing() must be in the top-level directory for ctest at the
    # build root to pick up add_test() calls from the tests/ subdirectory.
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Packaging ──────────────────────────────────────────────────────────────
# No CPack here: the release archive bundles C++ (engine .so/.a + headers +
# mslite-chat), the Python export wheel, and the AscendC operator .run
# packages — three kinds of artifacts CPack cannot compose.  build.sh runs
# `cmake --install` into a staging dir (install() rules above land under
# lib/ include/ bin/) and then collects the wheel (output/tool) and the .run
# packages (custom_ops/output) into the same staging tree before tar-ing it.
