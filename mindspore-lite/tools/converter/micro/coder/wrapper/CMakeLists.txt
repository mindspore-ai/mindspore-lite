cmake_minimum_required(VERSION 3.14)
project(wrapper)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "-DDebug -g -fvisibility=default ${CMAKE_C_FLAGS}")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -Werror -O3 -fstack-protector-strong -Wno-atomic-alignment -fomit-frame-pointer \
        ${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    add_compile_definitions(_GNU_SOURCE)
    add_compile_definitions(ENABLE_NEON)
    add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
elseif(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
endif()

set(MICRO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(KERNEL_DIR "${TOP_DIR}/mindspore-lite/src/litert/kernel/cpu/")
include("${MICRO_DIR}/cmake/package_wrapper.cmake")
include_directories("${KERNEL_DIR}")

# generate static library
add_library(wrapper_mid OBJECT ${WRAPPER_SRC})

add_library(wrapper STATIC $<TARGET_OBJECTS:wrapper_mid>)
target_link_libraries(wrapper_mid -lpthread)
