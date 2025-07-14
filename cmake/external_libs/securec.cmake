set(securec_USE_STATIC_LIBS ON)

if(BUILD_LITE)
    if(MSVC)
        set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(securec_CFLAGS "${CMAKE_C_FLAGS}")
        set(securec_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    else()
        set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(securec_CFLAGS "${CMAKE_C_FLAGS}")
        set(securec_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    endif()
else()
    if(MSVC)
        # add "/EHsc", for vs2019 warning C4530 about securec
        set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    else()
        set(securec_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    endif()
endif()

if(TARGET_OHOS)
    set(securec_CFLAGS "${securec_CFLAGS} -Wno-unused-command-line-argument")
endif()

# libboundscheck-v1.1.16
set(REQ_URL "https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.zip")
set(SHA256 "5119bda1ee96440c1a45e23f0cb8b079cc6697e052c4a78f27d0869f84ba312b")

string(REPLACE "/mindspore-lite" "" MS_TOP_DIR ${CMAKE_SOURCE_DIR})

if(BUILD_LITE)
    if(ANDROID_NDK)
        if(PLATFORM_ARM64)
            if(MSLITE_ENABLE_AOS)
                set(CMAKE_OPTION ${CMAKE_OPTION} -DCMAKE_C_COMPILER=${C_COMPILER}
                        -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
                        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
            else()
                set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                        -DANDROID_NATIVE_API_LEVEL=19
                        -DANDROID_NDK=$ENV{ANDROID_NDK}
                        -DANDROID_ABI=arm64-v8a
                        -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                        -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
            endif()
        endif()
        if(PLATFORM_ARM32)
            set(CMAKE_OPTION  -DCMAKE_TOOLCHAIN_FILE=$ENV{ANDROID_NDK}/build/cmake/android.toolchain.cmake
                    -DANDROID_NATIVE_API_LEVEL=19
                    -DANDROID_NDK=$ENV{ANDROID_NDK}
                    -DANDROID_ABI=armeabi-v7a
                    -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-clang
                    -DANDROID_STL=c++_shared -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
    else()
        if(PLATFORM_ARM64 AND MSLITE_ENABLE_AOS)
            set(CMAKE_OPTION ${CMAKE_OPTION} -DCMAKE_C_COMPILER=${C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CXX_COMPILER}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE})
        endif()
    endif()
endif()

mindspore_add_pkg(securec
        VER 1.1.16
        LIBS securec
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION} -DTARGET_OHOS_LITE=OFF
        PATCHES ${TOP_DIR}/third_party/patch/securec/securec.patch001
        )

include_directories(${securec_INC})
include_directories(${securec_INC}/../)
add_library(mindspore::securec ALIAS securec::securec)