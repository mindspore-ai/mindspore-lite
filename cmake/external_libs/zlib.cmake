if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/zlib/repository/archive/v1.3.1.tar.gz")
    set(SHA256 "4208eb8a8ba5703831123b06c9bbadf780e678a9972eb00ecce57e6f87b30f36")
else()
    set(REQ_URL "https://github.com/madler/zlib/archive/v1.3.1.tar.gz")
    set(SHA256 "17e88863f3600672ab49182f217281b6fc4d3c762bde361935e436a95214d05c")
endif()

if(BUILD_LITE)
    set(ZLIB_PATCH_ROOT ${TOP_DIR}/third_party/patch/zlib)
else()
    set(ZLIB_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/zlib)
endif()

mindspore_add_pkg(zlib
        VER 1.3.1
        LIBS z
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release)

include_directories(${zlib_INC})
add_library(mindspore::z ALIAS zlib::z)
