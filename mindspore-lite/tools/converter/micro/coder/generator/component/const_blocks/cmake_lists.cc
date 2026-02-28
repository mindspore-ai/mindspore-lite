/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "coder/generator/component/const_blocks/cmake_lists.h"

namespace mindspore::lite::micro {
const char *bench_cmake_lists_cortex = R"RAW(
cmake_minimum_required(VERSION 3.18)
project(benchmark)

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(HEADER_PATH ${PKG_PATH}/runtime)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)

add_compile_definitions(NOT_USE_STL)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
  add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build benchmark with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build benchmark release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-incompatible-pointer-types -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-incompatible-pointer-types -Wno-missing-braces -Wno-overloaded-virtual \
    ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--gc-sections")

add_subdirectory(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
set(SRC_FILES
        benchmark/benchmark.c
        benchmark/calib_output.c
        benchmark/load_input.c
        benchmark/data.c
)
add_library(benchmark STATIC ${SRC_FILES})

)RAW";

const char *bench_cmake_lists = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(benchmark)

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(HEADER_PATH ${PKG_PATH}/runtime)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)
option(ENABLE_FP16 "enable fp32/fp16 datatype transform for inputs or outputs" OFF)

add_compile_definitions(NOT_USE_STL)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
  if(ENABLE_FP16)
    add_compile_definitions(ENABLE_FP16)
  endif()
endif()

if(PLATFORM_ARM64)
  add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build benchmark with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build benchmark release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,--gc-sections")

add_subdirectory(src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
file(GLOB SRC_FILES
    benchmark/*.c)
add_executable(benchmark ${SRC_FILES})
target_link_libraries(benchmark net -lm -pthread)

)RAW";

const char *bench_cmake_lists_riscv = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(micro_lib)

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-g++")
set(CMAKE_C_FLAGS "-march=rv32imafc -mabi=ilp32f ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-march=rv32imafc -mabi=ilp32f ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}, CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("OP_LIB: ${OP_LIB}")
message("WRAPPER_LIB: ${WRAPPER_LIB}")
message("MS_ROOT_DIR: ${MS_ROOT_DIR}")

set(HEADER_PATH ${PKG_PATH}/runtime)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)

add_subdirectory(src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MS_ROOT_DIR}/include)
include_directories(${MS_ROOT_DIR})
include_directories(${OP_HEADER_PATH})
set(
    SOURCE_FILE
    src/model.c
    src/tensor.c
    src/allocator.c
    src/context.c
    src/model0/model0.c
    src/model0/net0.c
    src/model0/weight0.c
)
add_library(micro_runtime STATIC ${SOURCE_FILE})
target_link_libraries(micro_runtime net -lm -pthread)
)RAW";

const char *bench_cmake_lists_riscv_debug = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(micro_lib)

set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-g++")
set(CMAKE_C_FLAGS "-march=rv32imafc -mabi=ilp32f ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-march=rv32imafc -mabi=ilp32f ${CMAKE_CXX_FLAGS}")
message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}, CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("OP_LIB: ${OP_LIB}")
message("WRAPPER_LIB: ${WRAPPER_LIB}")
message("MS_ROOT_DIR: ${MS_ROOT_DIR}")

set(HEADER_PATH ${PKG_PATH}/runtime)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)

add_subdirectory(src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MS_ROOT_DIR}/include)
include_directories(${MS_ROOT_DIR})
include_directories(${OP_HEADER_PATH})
set(
    SOURCE_FILE
    src/model.c
    src/tensor.c
    src/log.c
    src/allocator.c
    src/context.c
    src/model0/model0.c
    src/model0/net0.c
    src/model0/weight0.c
)
add_library(micro_runtime STATIC ${SOURCE_FILE})
target_link_libraries(micro_runtime net -lm -pthread)
)RAW";

const char *src_cmake_lists = R"RAW(
cmake_minimum_required(VERSION 3.14)
project(net)

if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

get_filename_component(PKG_PATH ${PKG_PATH} ABSOLUTE BASE_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(OP_LIB ${PKG_PATH}/tools/codegen/lib/cpu/libnnacl.a)
set(WRAPPER_LIB ${PKG_PATH}/tools/codegen/lib/cpu/libwrapper.a)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)
set(HEADER_PATH ${PKG_PATH}/runtime)

message(STATUS "operator lib path: ${OP_LIB}")
message(STATUS "operator header path: ${OP_HEADER_PATH}")

add_compile_definitions(NOT_USE_STL)

include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)

include(net.cmake)

option(PLATFORM_ARM64 "build android arm64" OFF)
option(PLATFORM_ARM32 "build android arm32" OFF)
option(ENABLE_FP16 "enable fp32/fp16 datatype transform for inputs or outputs" OFF)

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
  add_compile_definitions(ENABLE_NEON)
  add_compile_definitions(ENABLE_ARM)
  if(ENABLE_FP16)
    add_compile_definitions(ENABLE_FP16)
  endif()
endif()

if(PLATFORM_ARM64)
  add_compile_definitions(ENABLE_ARM64)
endif()

if(PLATFORM_ARM32)
  add_compile_definitions(ENABLE_ARM32)
  add_definitions(-mfloat-abi=softfp -mfpu=neon)
endif()

set(CMAKE_C_FLAGS "${CMAKE_ENABLE_C99} ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "build net library with debug info")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDebug -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDebug -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=default")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
else()
    message(STATUS "build net library release version")
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O3 -Wall -Werror -fstack-protector-strong -Wno-attributes \
    -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

function(create_library)
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND rm -rf tmp
            COMMAND mkdir tmp
            COMMAND cd tmp && ar -x ${OP_LIB}
            COMMAND cd tmp && ar -x ${WRAPPER_LIB}
            COMMAND echo "raw static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}
            COMMENT "unzip raw static library ${library_name}"
            )

    foreach(object_file ${OP_SRC})
        add_custom_command(TARGET net POST_BUILD COMMAND mv ./tmp/${object_file} .)
    endforeach()
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND ar cr ${library_name} *.o
            COMMAND ranlib ${library_name}
            COMMAND echo "new static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND rm -rf tmp && rm -rf *.o
            COMMENT "generate specified static library ${library_name}"
            )
endfunction(create_library)
string(CONCAT library_name "lib" net ".a")
create_library()
)RAW";

const char *src_cmake_lists_riscv = R"RAW(
if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)
set(HEADER_PATH ${PKG_PATH}/runtime)

message(STATUS "operator lib path: ${OP_LIB}")
message(STATUS "operator header path: ${OP_HEADER_PATH}")

add_compile_definitions(NOT_USE_STL)

include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MS_ROOT_DIR})
include_directories(${MS_ROOT_DIR}/include)

include(net.cmake)
function(create_library)
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND rm -rf tmp
            COMMAND mkdir tmp
            COMMAND cd tmp && ar -x ${OP_LIB}
            COMMAND cd tmp && ar -x ${WRAPPER_LIB}
            COMMAND echo "raw static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}
            COMMENT "unzip raw static library ${library_name}"
            )

    foreach(object_file ${OP_SRC})
        add_custom_command(TARGET net POST_BUILD COMMAND mv ./tmp/${object_file} .)
    endforeach()
    add_custom_command(TARGET net
            POST_BUILD
            COMMAND ar cr ${library_name} *.o
            COMMAND ranlib ${library_name}
            COMMAND echo "new static library ${library_name} size:"
            COMMAND ls -lh ${library_name}
            COMMAND rm -rf tmp && rm -rf *.o
            COMMENT "generate specified static library ${library_name}"
            )
endfunction(create_library)
string(CONCAT library_name "lib" net ".a")
create_library()

)RAW";

}  // namespace mindspore::lite::micro
