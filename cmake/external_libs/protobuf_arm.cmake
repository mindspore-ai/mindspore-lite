set(protobuf_arm_USE_STATIC_LIBS ON)
if(BUILD_LITE)
    if(MSVC)
        set(protobuf_arm_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(protobuf_arm_CFLAGS "${CMAKE_C_FLAGS}")
        set(protobuf_arm_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
    else()
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_arm_CXXFLAGS "${protobuf_arm_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
        set(protobuf_arm_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    else()
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_arm_CXXFLAGS "${protobuf_arm_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
    endif()
    set(protobuf_arm_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v33.1.tar.gz")
    set(SHA256 "9d3e214f7a30abe4c05349163c20d93cb28ed7a3c6dae4b24a4a1671e53744c1")
else()
    set(REQ_URL "https://codeload.github.com/protocolbuffers/protobuf/tar.gz/refs/tags/v33.1")
    set(SHA256 "0c98bb704ceb4e68c92f93907951ca3c36130bc73f87264e8c0771a80362ac97")
endif()

# Pre-download abseil-cpp for protobuf build. Protobuf v33.1 uses FetchContent
# internally to download absl from GitHub (GIT_REPOSITORY). Pre-download and
# point FETCHCONTENT_SOURCE_DIR_ABSL to the extracted source, so protobuf's
# FetchContent skips the download entirely. This avoids patching protobuf's
# cmake/abseil-cpp.cmake.
set(absl_for_protobuf_arm_ver "20250512.1")
if(ENABLE_GITEE OR ENABLE_GITEE_EULER)
    set(absl_for_protobuf_arm_url
        "https://gitee.com/mirrors/abseil-cpp/repository/archive/${absl_for_protobuf_arm_ver}.tar.gz")
else()
    set(absl_for_protobuf_arm_url
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/${absl_for_protobuf_arm_ver}.tar.gz")
endif()
set(absl_for_protobuf_arm_sha256 "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db")
# Use __download_pkg which supports LOCAL_LIBS_SERVER mirror (tools.mindspore.cn).
__download_pkg(absl_for_protobuf_arm ${absl_for_protobuf_arm_url} ${absl_for_protobuf_arm_sha256})
# Git archives create a single top-level directory; strip it.
file(GLOB _absl_entries "${absl_for_protobuf_arm_SOURCE_DIR}/*")
list(LENGTH _absl_entries _absl_entry_count)
if(_absl_entry_count EQUAL 1 AND IS_DIRECTORY "${_absl_entries}")
    file(GLOB _absl_subdir_contents "${_absl_entries}/*")
    file(COPY ${_absl_subdir_contents} DESTINATION "${absl_for_protobuf_arm_SOURCE_DIR}")
    file(REMOVE_RECURSE "${_absl_entries}")
endif()
set(absl_for_protobuf_arm_dir "${absl_for_protobuf_arm_SOURCE_DIR}")

if(APPLE)
    mindspore_add_pkg(protobuf_arm
            VER 33.1
            LIBS protobuf
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_PATH .
            CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -Dprotobuf_WITH_ZLIB=OFF
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -DCMAKE_CXX_STANDARD=17
            -DFETCHCONTENT_SOURCE_DIR_ABSL=${absl_for_protobuf_arm_dir}
            )
else()
    mindspore_add_pkg(protobuf_arm
            VER 33.1
            LIBS protobuf
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_PATH .
            CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -Dprotobuf_WITH_ZLIB=OFF
            -DCMAKE_CXX_STANDARD=17
            -DFETCHCONTENT_SOURCE_DIR_ABSL=${absl_for_protobuf_arm_dir}
            )
endif()

include_directories(${protobuf_arm_INC})
set(_ms_protobuf_arm_prefix "")
if(DEFINED protobuf_arm_DIRPATH AND EXISTS "${protobuf_arm_DIRPATH}")
    set(_ms_protobuf_arm_prefix "${protobuf_arm_DIRPATH}")
elseif(DEFINED protobuf_arm_ROOT AND EXISTS "${protobuf_arm_ROOT}")
    set(_ms_protobuf_arm_prefix "${protobuf_arm_ROOT}")
elseif(DEFINED protobuf_arm_BASE_DIR AND EXISTS "${protobuf_arm_BASE_DIR}")
    set(_ms_protobuf_arm_prefix "${protobuf_arm_BASE_DIR}")
endif()
if(_ms_protobuf_arm_prefix AND EXISTS "${_ms_protobuf_arm_prefix}/lib64/cmake/protobuf/protobuf-config.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${_ms_protobuf_arm_prefix}")
    find_package(absl CONFIG REQUIRED PATHS "${_ms_protobuf_arm_prefix}/lib64/cmake/absl" NO_DEFAULT_PATH)
    find_package(utf8_range CONFIG REQUIRED PATHS "${_ms_protobuf_arm_prefix}/lib64/cmake/utf8_range" NO_DEFAULT_PATH)
    find_package(protobuf CONFIG REQUIRED PATHS "${_ms_protobuf_arm_prefix}/lib64/cmake/protobuf" NO_DEFAULT_PATH)
    add_library(mindspore::protobuf_arm ALIAS protobuf::libprotobuf)
else()
    add_library(mindspore::protobuf_arm ALIAS protobuf_arm::protobuf)
endif()
unset(_ms_protobuf_arm_prefix)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()
