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
# lite_llm examples: public API sample CLI, always built with the engine.

add_executable(mslite-chat mslite-chat.cc)
target_include_directories(mslite-chat PRIVATE ${MSLITE_LLM_ROOT}/src)
# The engine is linked statically. Backends self-register via static-init
# objects (MSLLM_REGISTER_BACKEND) with no other external reference, so a
# plain archive link would drop their .o files (linker pulls archive members
# on demand) and CreateBackend would find an empty registry. --whole-archive
# forces every engine/backend object in, keeping the registrations alive.
if(MSLITE_LLM_CROSS_COMPILE_HARMONYOS)
  # OHOS: statically link the engine and libc++ — the packaged binary must
  # run from the BinRunner memory loader (no runtime dlopen of the engine
  # .so, and the device has no libc++_shared.so in the app sandbox).
  target_link_libraries(mslite-chat PRIVATE
    -Wl,--whole-archive mslite_llm_static -Wl,--no-whole-archive)
  target_link_options(mslite-chat PRIVATE -static-libstdc++)
else()
  target_link_libraries(mslite-chat PRIVATE
    -Wl,--whole-archive mslite_llm_static -Wl,--no-whole-archive)
endif()
# Relocatable library lookup: dev tree keeps the .so in build/src, the packaged
# archive keeps it in lib/, so resolve relative to the binary.
set_target_properties(mslite-chat PROPERTIES
    BUILD_RPATH "$ORIGIN/../src"
    INSTALL_RPATH "$ORIGIN/../lib")
# Packaged into bin/ by build.sh (cmake --install staging); INSTALL_RPATH
# keeps the packaged binary resolving the sibling lib/.
install(TARGETS mslite-chat RUNTIME DESTINATION bin)
