cmake_minimum_required(VERSION 3.12)
project(Micro)
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fstack-protector-strong -Wno-strict-aliasing")

set(CMAKE_VERBOSE_MAKEFILE on)
set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${NNACL_DIR}/..)

include(${TOP_DIR}/cmake/utils.cmake)

include_directories(${MICRO_DIR}/coder/)
set(WRAPPER_DIR ${MICRO_DIR}/coder/wrapper/)
file(GLOB WRAPPER_SRC
        ${WRAPPER_DIR}/base/*.c
        ${WRAPPER_DIR}/fp32/*.c
        ${WRAPPER_DIR}/int8/*.c
        )

# generate static library
add_library(wrapper STATIC ${WRAPPER_SRC})
