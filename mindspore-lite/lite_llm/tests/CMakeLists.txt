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
# lite_llm tests: gtest unit tests, reusing the main repo's third-party gtest
# provisioning (cmake/external_libs/gtest.cmake) instead of a hand-rolled
# system fallback.
#
# ST (model-based end-to-end: export -> real-device inference) is not part of
# the CTest suite; run it via pytest, see tests/st/conftest.py.

# Reuse the main repo's third-party build infrastructure: mindspore_add_pkg
# (cmake/utils.cmake) provisions gtest from source (downloaded once, cached in
# $MSLIBS_CACHE_PATH or <build>/.mslib).  TOP_DIR anchors the strip/cache
# logic used inside utils.cmake.  gtest.cmake builds gtest with
# -D_GLIBCXX_USE_CXX11_ABI=0 unless ENABLE_GLIBCXX is set; default it to ON
# here so the provisioned gtest matches this module's ABI (the main repo sets
# the variable itself when integrating, which is then respected).
if(NOT DEFINED ENABLE_GLIBCXX)
    set(ENABLE_GLIBCXX ON)
endif()
include(${MSLITE_LLM_REPO_ROOT}/cmake/utils.cmake)
set(TOP_DIR ${MSLITE_LLM_REPO_ROOT})
# gtest cross-compilation for the OHOS target is not wired up (the main repo's
# gtest.cmake is reused as-is; a custom CMAKE_OPTION seed would leak into the
# shared third-party configure and break the Android STL linkage).  Cross
# builds therefore run the host gtest, which is fine for the unit-test
# harness used here.
include(${MSLITE_LLM_REPO_ROOT}/cmake/external_libs/gtest.cmake)

# golden_v1.msl is gitignored (deterministically regenerable via
# data/gen_golden.py).  Generate it at configure time when missing so a
# fresh checkout works for both this CTest suite and the pytest suite;
# an existing file is left untouched.  msl_pack.py is stdlib-only, so any
# Python3 interpreter suffices.
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/golden_v1.msl")
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        message(STATUS "Generating tests/data/golden_v1.msl via gen_golden.py")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/data/gen_golden.py"
            RESULT_VARIABLE _golden_gen_result
        )
        if(NOT _golden_gen_result EQUAL 0)
            message(WARNING "gen_golden.py failed (${_golden_gen_result}); "
                "llm_msl_golden will fail until tests/data/golden_v1.msl exists")
        endif()
    else()
        message(WARNING "Python3 interpreter not found; cannot auto-generate "
            "tests/data/golden_v1.msl — llm_msl_golden will fail until it exists")
    endif()
endif()

# Every test executable links the shared entry (ut/gtest_main.cc) plus the
# provisioned gtest; there is no gtest_main library involved.
set(MSLITE_LLM_GTEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/ut/gtest_main.cc)

# Unit test harness: executable + engine link + shared include dirs.
# add_test name derives from the target ("test_foo" -> "llm_foo").
function(mslite_llm_add_unit_test target_name source_file)
    add_executable(${target_name} ${source_file} ${MSLITE_LLM_GTEST_MAIN})
    target_link_libraries(${target_name} PRIVATE mslite_llm mindspore::gtest)
    target_include_directories(${target_name} PRIVATE
        ${MSLITE_LLM_ROOT}/include
        ${MSLITE_LLM_ROOT}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    string(REGEX REPLACE "^test_" "" llm_test_name ${target_name})
    add_test(NAME llm_${llm_test_name} COMMAND ${target_name})
endfunction()

mslite_llm_add_unit_test(test_sampler ut/test_sampler.cc)
mslite_llm_add_unit_test(test_model_resource_loader ut/test_model_resource_loader.cc)
mslite_llm_add_unit_test(test_llm_lifecycle ut/test_llm_lifecycle.cc)
mslite_llm_add_unit_test(test_llm_chat_template ut/test_llm_chat_template.cc)
mslite_llm_add_unit_test(test_llm_engine ut/test_llm_engine.cc)

# Golden-file test: embeds the golden package path; no unit-test fixture include.
add_executable(test_msl_golden ut/test_msl_golden.cc ${MSLITE_LLM_GTEST_MAIN})
target_link_libraries(test_msl_golden PRIVATE mslite_llm mindspore::gtest)
target_include_directories(test_msl_golden PRIVATE ${MSLITE_LLM_ROOT}/src)
target_compile_definitions(test_msl_golden PRIVATE
    MSL_GOLDEN_FILE="${CMAKE_CURRENT_SOURCE_DIR}/data/golden_v1.msl")
add_test(NAME llm_msl_golden COMMAND test_msl_golden)
