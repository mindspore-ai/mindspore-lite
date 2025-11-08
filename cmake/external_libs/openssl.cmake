if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/openssl-3.5.4.tar.gz")
  set(SHA256 "758b69feed5787dc12d34b0eb29b60d3c9d73d5a64760c62d93a6d26b344d65d")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.4.tar.gz")
  set(SHA256 "758b69feed5787dc12d34b0eb29b60d3c9d73d5a64760c62d93a6d26b344d65d")
endif()

if(PLATFORM_ARM64 AND ANDROID_NDK_TOOLCHAIN_INCLUDED)
    set(openssl_USE_STATIC_LIBS OFF)
    set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK})
    set(PATH
        ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:
        ${ANDROID_NDK_ROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:
        $ENV{PATH})
    mindspore_add_pkg(openssl
            VER 3.5.4
            LIBS ssl crypto
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./Configure android-arm64 -D__ANDROID_API__=29 no-zlib no-afalgeng
            )
elseif(PLATFORM_ARM32 AND ANDROID_NDK_TOOLCHAIN_INCLUDED)
    set(openssl_USE_STATIC_LIBS OFF)
    set(ANDROID_NDK_ROOT $ENV{ANDROID_NDK})
    set(PATH
        ${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:
        ${ANDROID_NDK_ROOT}/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin:
        $ENV{PATH})
    mindspore_add_pkg(openssl
            VER 3.5.4
            LIBS ssl crypto
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./Configure android-arm -D__ANDROID_API__=19 no-zlib no-afalgeng
            )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
    set(openssl_CFLAGS -fvisibility=hidden)
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64" OR "arm64")
        mindspore_add_pkg(openssl
                VER 3.5.4
                LIBS ssl crypto
                URL ${REQ_URL}
                SHA256 ${SHA256}
                CONFIGURE_COMMAND ./config no-zlib no-shared no-afalgeng
                )
    elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
        mindspore_add_pkg(openssl
                VER 3.5.4
                LIB_PATH lib64
                LIBS ssl crypto
                URL ${REQ_URL}
                SHA256 ${SHA256}
                CONFIGURE_COMMAND ./config no-zlib no-shared no-afalgeng
        )
    endif()
else()
    MESSAGE(FATAL_ERROR "openssl does not support compilation for the current environment.")
endif()
include_directories(${openssl_INC})
add_library(mindspore::ssl ALIAS openssl::ssl)
add_library(mindspore::crypto ALIAS openssl::crypto)
