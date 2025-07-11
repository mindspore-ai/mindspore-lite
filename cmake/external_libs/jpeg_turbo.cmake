if(TARGET_AOS_ARM)
    set(CMAKE_C_COMPILER          "$ENV{CC}")
    set(CMAKE_SYSTEM_PROCESSOR    "aarch64")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/libjpeg-turbo/repository/archive/3.0.1.tar.gz")
    set(SHA256 "5b9bbca2b2a87c6632c821799438d358e27004ab528abf798533c15d50b39f82")
else()
    set(REQ_URL "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/3.0.1.tar.gz")
    set(SHA256 "5b9bbca2b2a87c6632c821799438d358e27004ab528abf798533c15d50b39f82")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 \
        -O2")
else()
    if(MSVC)
        set(jpeg_turbo_CFLAGS "-O2")
    else()
        set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2")
        if(TARGET_AOS_ARM)
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -march=armv8.2-a -mtune=cortex-a72")
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -Wno-uninitialized -march=armv8.2-a -mtune=cortex-a72")
        else()
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -Wno-maybe-uninitialized")
        endif()
    endif()
endif()

set(jpeg_turbo_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack,-s")


set(jpeg_turbo_USE_STATIC_LIBS ON)
set(CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_RPATH=TRUE -DWITH_SIMD=ON)
if(BUILD_LITE)
    set(jpeg_turbo_USE_STATIC_LIBS OFF)
    if(ANDROID_NDK)  #  compile android on x86_64 env
        if(PLATFORM_ARM64)
            set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                              -DANDROID_NATIVE_API_LEVEL=19
                              -DANDROID_NDK=$ENV{ANDROID_NDK}
                              -DANDROID_ABI=arm64-v8a
                              -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                              -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
        if(PLATFORM_ARM32)
            set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                              -DANDROID_NATIVE_API_LEVEL=19
                              -DANDROID_NDK=$ENV{ANDROID_NDK}
                              -DANDROID_ABI=armeabi-v7a
                              -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                              -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
    elseif(TARGET_AOS_ARM)
        set(CMAKE_OPTION ${CMAKE_OPTION} -DCMAKE_C_COMPILER=${C_COMPILER}
                -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
    endif()
endif()

mindspore_add_pkg(jpeg_turbo
        VER 3.0.1
        LIBS jpeg turbojpeg
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION}
        )
include_directories(${jpeg_turbo_INC})
add_library(mindspore::jpeg_turbo ALIAS jpeg_turbo::jpeg)
add_library(mindspore::turbojpeg ALIAS jpeg_turbo::turbojpeg)
