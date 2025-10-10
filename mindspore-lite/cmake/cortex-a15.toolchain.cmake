set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

find_program(arm-linux-gnueabihf-gcc_EXE arm-linux-gnueabihf-gcc)
if(NOT arm-linux-gnueabihf-gcc_EXE)
    message(FATAL_ERROR "Required C COMPILER arm-linux-gnueabihf-gcc not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find C COMPILER PATH: ${arm-linux-gnueabihf-gcc_EXE}")
endif()

find_program(arm-linux-gnueabihf-g++_EXE arm-linux-gnueabihf-g++)
if(NOT arm-linux-gnueabihf-g++_EXE)
    message(FATAL_ERROR "Required CXX COMPILER arm-linux-gnueabihf-g++ not found, "
            "please install the package and try building MindSpore again.")
else()
    message("Find CXX COMPILER PATH: ${arm-linux-gnueabihf-g++_EXE}")
endif()

set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mtune=cortex-a15 -mfpu=neon -mfloat-abi=hard")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mtune=cortex-a15 -mfpu=neon -mfloat-abi=hard")

# used for flatc compile
find_path(GCC_PATH gcc)
find_path(GXX_PATH g++)
if(NOT ${GCC_PATH} STREQUAL "GCC_PATH-NOTFOUND" AND NOT ${GXX_PATH} STREQUAL "GXX_PATH-NOTFOUND")
    set(FLATC_GCC_COMPILER ${GCC_PATH}/gcc)
    set(FLATC_GXX_COMPILER ${GXX_PATH}/g++)
endif()
