# GNU toolchain from https://github.com/riscv-collab/riscv-gnu-toolchain
# put at /opt/riscv
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(TOOLCHAIN_PATH "/opt/riscv")
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/bin/riscv64-unknown-linux-gnu-g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64imafdc_zicsr -pthread ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64imafdc_zicsr -pthread ${CXX_FLAGS}")

#set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_SYSROOT /opt/riscv/sysroot)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
