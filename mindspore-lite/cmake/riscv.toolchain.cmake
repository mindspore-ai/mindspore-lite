# GNU toolchain from https://github.com/riscv-collab/riscv-gnu-toolchain
# put at /opt/riscv
# HiSpark BiSheng-llvm-riscv-x86-linux toolchain from https://developers.hisilicon.com/cn/developerTool
set(CMAKE_SYSTEM_NAME Linux)
set(ENABLE_RISCV32_TOOLCHAIN ${RISCV32})
set(ENABLE_HISPARK_TOOLCHAIN ${RISCV32})

if(ENABLE_RISCV32_TOOLCHAIN STREQUAL ON)
    set(CMAKE_SYSTEM_PROCESSOR riscv)
else()
    set(CMAKE_SYSTEM_PROCESSOR riscv64)
endif()
message("CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")

if(ENABLE_RISCV32_TOOLCHAIN STREQUAL ON AND ENABLE_HISPARK_TOOLCHAIN STREQUAL ON)
    if(DEFINED ENV{HISPARK_RISCV_TOOLCHAIN_PATH})
        set(TOOLCHAIN_PATH "$ENV{HISPARK_RISCV_TOOLCHAIN_PATH}/bin")
    elseif(DEFINED RISCV_TOOLCHAIN_PATH)
        set(TOOLCHAIN_PATH "${RISCV_TOOLCHAIN_PATH}")  # Supports CMake parameter input
    else()
        set(TOOLCHAIN_PATH "/opt/riscv")
    endif()
else()
    set(TOOLCHAIN_PATH "/opt/riscv")
endif()
message(STATUS "TOOLCHAIN_PATH path: ${TOOLCHAIN_PATH}")


if(ENABLE_RISCV32_TOOLCHAIN STREQUAL ON AND ENABLE_HISPARK_TOOLCHAIN STREQUAL ON)
    set(CMAKE_C_COMPILER   "${TOOLCHAIN_PATH}/clang")
    set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PATH}/clang++")
else()
    set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-g++)
endif()

if(ENABLE_RISCV32_TOOLCHAIN STREQUAL ON)
    set(RISCV_ARCH_FLAGS "-march=rv32imfc -mabi=ilp32f")  # Hard float support[9](@ref)
    set(CMAKE_C_FLAGS_INIT "${RISCV_ARCH_FLAGS} -Os -fno-exceptions")
    set(CMAKE_CXX_FLAGS_INIT "${RISCV_ARCH_FLAGS} -Os -fno-exceptions")
    if(ENABLE_HISPARK_TOOLCHAIN STREQUAL ON)
        set(CMAKE_C_FLAGS_INIT   "${CMAKE_C_FLAGS_INIT} \
            -D_FORTIFY_SOURCE=2 -Wall -Werror -Wno-attributes -Wno-deprecated-declarations -mllvm \
            -enable-unroll-and-jam=true -mllvm -allow-unroll-and-jam=true -Wno-missing-braces -fomit-frame-pointer \
            -mllvm -enable-loop-fusion=true -mllvm -enable-small-loop-unroll=false -mllvm -unroll-and-jam-count=9 \
            -fstrict-aliasing -ffunction-sections  -fdata-sections -O3 -mllvm -unroll-runtime -mllvm \
            -enable-loop-flatten=true -mcpu=linx-rv32 -DNDEBUG -Wno-atomic-alignment -fstack-protector-all")
        set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} -Os -fno-exceptions")
        set(CMAKE_EXE_LINKER_FLAGS_INIT
            "${RISCV_ARCH_FLAGS} -Wl,--gc-sections -static"
        )
    endif()
else()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv -mabi=lp64d -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv -mabi=lp64d -pthread")
endif()

if(ENABLE_RISCV32_TOOLCHAIN STREQUAL ON)
    execute_process(
        COMMAND ${CMAKE_C_COMPILER} -v
        ERROR_VARIABLE RISCV_COMPILER_VERSION
        OUTPUT_QUIET
    )
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
else()
    option(PLATFORM_RISCV64 "build riscv64" ON)
    #enable rvv
    option(ENABLE_RVV "enable rvv" ON)
    #define for c
    add_definitions(-DENABLE_RVV=${ENABLE_RVV})
    #eliminate array-bounds warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-template-body")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=array-bounds")
    #eliminate free-nonheap-object warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=free-nonheap-object")
endif()

set(CMAKE_C_OUTPUT_EXTENSION_REPLACE 0)   # 0 means preserve extension
set(CMAKE_C_OUTPUT_EXTENSION ".c.o")
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 0) # Same for C++

set(CMAKE_SYSROOT /opt/riscv/sysroot)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
