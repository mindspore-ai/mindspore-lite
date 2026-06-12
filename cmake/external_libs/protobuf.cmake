set(protobuf_USE_STATIC_LIBS ON)
set(ENABLE_NATIVE_PROTOBUF "off")
if(EXISTS ${TOP_DIR}/mindspore-lite/providers/protobuf/native_protobuf.cfg)
    set(ENABLE_NATIVE_PROTOBUF "on")
    file(STRINGS ${TOP_DIR}/mindspore-lite/providers/protobuffer/native_protobuffer.cfg native_protobuffer_path)
endif()
if(BUILD_LITE)
    if(MSVC)
        set(protobuf_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(protobuf_CFLAGS "${CMAKE_C_FLAGS}")
        set(protobuf_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
        if(DEBUG_MODE)
            set(protobuf_Debug ON)
        endif()
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
        set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        if(MSVC)
            set(protobuf_CXXFLAGS "/DWIN32 /D_WINDOWS /W3 /GR /EHsc")
            set(protobuf_CFLAGS "${CMAKE_C_FLAGS}")
            set(protobuf_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
            set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
            set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
            if(DEBUG_MODE)
                set(protobuf_Debug ON)
            endif()
        else()
            set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
                -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        endif()
    else()
        set(protobuf_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_CXXFLAGS "${protobuf_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
    endif()
    set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v33.1.tar.gz")
    set(SHA256 "9d3e214f7a30abe4c05349163c20d93cb28ed7a3c6dae4b24a4a1671e53744c1")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v33.1.tar.gz")
    set(SHA256 "0c98bb704ceb4e68c92f93907951ca3c36130bc73f87264e8c0771a80362ac97")
endif()

# Pre-download abseil-cpp for protobuf build. Protobuf v33.1 uses FetchContent
# internally to download absl from GitHub (GIT_REPOSITORY). Pre-download and
# point FETCHCONTENT_SOURCE_DIR_ABSL to the extracted source, so protobuf's
# FetchContent skips the download entirely. This avoids patching protobuf's
# cmake/abseil-cpp.cmake.
set(absl_for_protobuf_ver "20250512.1")
if(ENABLE_GITEE OR ENABLE_GITEE_EULER)
    set(absl_for_protobuf_url
        "https://gitee.com/mirrors/abseil-cpp/repository/archive/${absl_for_protobuf_ver}.tar.gz")
else()
    set(absl_for_protobuf_url
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/${absl_for_protobuf_ver}.tar.gz")
endif()
set(absl_for_protobuf_sha256 "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db")
# Use __download_pkg which supports LOCAL_LIBS_SERVER mirror (tools.mindspore.cn).
__download_pkg(absl_for_protobuf ${absl_for_protobuf_url} ${absl_for_protobuf_sha256})
# Git archives create a single top-level directory; strip it.
file(GLOB _absl_entries "${absl_for_protobuf_SOURCE_DIR}/*")
list(LENGTH _absl_entries _absl_entry_count)
if(_absl_entry_count EQUAL 1 AND IS_DIRECTORY "${_absl_entries}")
    file(GLOB _absl_subdir_contents "${_absl_entries}/*")
    file(COPY ${_absl_subdir_contents} DESTINATION "${absl_for_protobuf_SOURCE_DIR}")
    file(REMOVE_RECURSE "${_absl_entries}")
endif()
set(absl_for_protobuf_dir "${absl_for_protobuf_SOURCE_DIR}")

if(MSVC)
mindspore_add_pkg(protobuf
        VER 33.1
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_PATH .
        CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CXX_STANDARD=17
            -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
            -Dprotobuf_WITH_ZLIB=OFF
            -DFETCHCONTENT_SOURCE_DIR_ABSL=${absl_for_protobuf_dir}
        )
elseif(WIN32)
mindspore_add_pkg(protobuf
        VER 33.1
        LIBS protobuf
        EXE protoc
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_PATH .
        CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CXX_STANDARD=17
            -Dprotobuf_WITH_ZLIB=OFF
            -DFETCHCONTENT_SOURCE_DIR_ABSL=${absl_for_protobuf_dir}
        )
else()
mindspore_add_pkg(protobuf
        VER 33.1
        LIBS protobuf
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_PATH .
        CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CXX_STANDARD=17
            -Dprotobuf_WITH_ZLIB=OFF
            -DFETCHCONTENT_SOURCE_DIR_ABSL=${absl_for_protobuf_dir}
        )
endif()
include_directories(${protobuf_INC})
include_directories(${CMAKE_BINARY_DIR}/proto_py)
set(_ms_protobuf_prefix "")
if(DEFINED protobuf_DIRPATH AND EXISTS "${protobuf_DIRPATH}")
    set(_ms_protobuf_prefix "${protobuf_DIRPATH}")
elseif(DEFINED protobuf_ROOT AND EXISTS "${protobuf_ROOT}")
    set(_ms_protobuf_prefix "${protobuf_ROOT}")
elseif(DEFINED protobuf_BASE_DIR AND EXISTS "${protobuf_BASE_DIR}")
    set(_ms_protobuf_prefix "${protobuf_BASE_DIR}")
endif()
set(_ms_protobuf_cmake_dir "${_ms_protobuf_prefix}/lib64/cmake")
if(NOT EXISTS "${_ms_protobuf_cmake_dir}/protobuf/protobuf-config.cmake" AND _ms_protobuf_prefix)
    set(_ms_protobuf_cmake_dir "${_ms_protobuf_prefix}/lib/cmake")
endif()
if(_ms_protobuf_prefix AND EXISTS "${_ms_protobuf_cmake_dir}/protobuf/protobuf-config.cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${_ms_protobuf_prefix}")
    find_package(absl CONFIG REQUIRED PATHS "${_ms_protobuf_cmake_dir}/absl" NO_DEFAULT_PATH)
    find_package(utf8_range CONFIG REQUIRED PATHS "${_ms_protobuf_cmake_dir}/utf8_range" NO_DEFAULT_PATH)
    if(WIN32 AND NOT MSVC)
        # MinGW: protobuf::protoc is already created by mindspore_add_pkg (has EXE protoc).
        # find_package(protobuf CONFIG) would conflict on protobuf::protoc target,
        # so create mindspore::protobuf as an interface library wrapping protobuf + absl.
        add_library(mindspore::protobuf INTERFACE IMPORTED)
        target_link_libraries(mindspore::protobuf INTERFACE
            protobuf::protobuf
            absl::absl_check absl::absl_log absl::algorithm absl::base
            absl::bind_front absl::bits absl::btree absl::cleanup
            absl::cord absl::core_headers absl::debugging absl::die_if_null
            absl::dynamic_annotations absl::flags absl::flat_hash_map
            absl::flat_hash_set absl::function_ref absl::hash absl::layout
            absl::log_initialize absl::log_globals absl::log_severity
            absl::memory absl::node_hash_map absl::node_hash_set
            absl::random_distributions absl::random_random absl::span
            absl::status absl::statusor absl::strings absl::synchronization
            absl::time absl::type_traits absl::utility
            utf8_range::utf8_validity
            utf8_range::utf8_range)
        if(protobuf_INC)
            target_include_directories(mindspore::protobuf INTERFACE ${protobuf_INC})
        endif()
    else()
        # Linux/MSVC: protobuf::protoc is not yet defined.
        # Use find_package(protobuf CONFIG) to import protobuf::libprotobuf (which
        # also creates protobuf::protoc needed by common_protobuf_generate's DEPENDS).
        find_package(protobuf CONFIG REQUIRED PATHS "${_ms_protobuf_cmake_dir}/protobuf" NO_DEFAULT_PATH)
        add_library(mindspore::protobuf ALIAS protobuf::libprotobuf)
    endif()
else()
    add_library(mindspore::protobuf ALIAS protobuf::protobuf)
endif()
unset(_ms_protobuf_cmake_dir)
unset(_ms_protobuf_prefix)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
# recover original value
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()

if(ENABLE_NATIVE_PROTOBUF)
    find_program(PROTOC protoc PATHS ${native_protobuffer_path}/bin NO_DEFAULT_PATH)
    find_library(PROTOBUF_LIB protobuf
        PATHS ${native_protobuffer_path}/lib ${native_protobuffer_path}/lib64
        NO_DEFAULT_PATH)
    set(protobuf_LIBPATH ${native_protobuffer_path}/lib)
    set(protobuf_INC ${native_protobuffer_path}/include)

    include_directories(${protobuf_INC})
    message("protobuf_INC : ${protobuf_INC}")
    set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
endif()
function(common_protobuf_generate path c_var h_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})

    foreach(file ${ARGN})
        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        file(RELATIVE_PATH rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${file_dir})
        set(_ms_proto_input ${abs_file})

        set(_ms_proto_out_cc "${path}/${file_name}.pb.cc")
        set(_ms_proto_out_h "${path}/${file_name}.pb.h")
        set(_ms_proto_out_dir "${path}")

        list(APPEND ${c_var} "${_ms_proto_out_cc}")
        list(APPEND ${h_var} "${_ms_proto_out_h}")
        if(ENABLE_NATIVE_PROTOBUF)
            if(WIN32 AND NOT MSVC)
                add_custom_command(
                OUTPUT "${_ms_proto_out_cc}" "${_ms_proto_out_h}"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_ms_proto_out_dir}"
                COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${protobuf_LIBPATH}" ${PROTOC} -I${file_dir}
                --cpp_out=${_ms_proto_out_dir} ${_ms_proto_input}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
            else()
                add_custom_command(
                OUTPUT "${_ms_proto_out_cc}" "${_ms_proto_out_h}"
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory "${_ms_proto_out_dir}"
                COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=${protobuf_LIBPATH}" ${PROTOC} -I${file_dir}
                --cpp_out=${_ms_proto_out_dir} ${_ms_proto_input}
                DEPENDS ${PROTOC} ${abs_file}
                COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
            endif()
        else()
        if(WIN32 AND NOT MSVC)
            add_custom_command(
                    OUTPUT "${_ms_proto_out_cc}" "${_ms_proto_out_h}"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${_ms_proto_out_dir}"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${_ms_proto_out_dir} ${_ms_proto_input}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        else()
            add_custom_command(
                    OUTPUT "${_ms_proto_out_cc}" "${_ms_proto_out_h}"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${_ms_proto_out_dir}"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${_ms_proto_out_dir} ${_ms_proto_input}
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        endif()
        endif()
    endforeach()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

function(ms_protobuf_generate c_var h_var)
    common_protobuf_generate(${CMAKE_BINARY_DIR}/proto ${c_var} ${h_var} ${ARGN})
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

function(ms_protobuf_generate_py c_var h_var py_var)
    if(NOT ARGN)
        message(SEND_ERROR "Error: ms_protobuf_generate() called without any proto files")
        return()
    endif()

    set(${c_var})
    set(${h_var})
    set(${py_var})

    foreach(file ${ARGN})
        get_filename_component(abs_file ${file} ABSOLUTE)
        get_filename_component(file_name ${file} NAME_WE)
        get_filename_component(file_dir ${abs_file} PATH)
        set(_ms_proto_input ${abs_file})

        if(WIN32 AND NOT MSVC)
            set(_ms_proto_py_out_cc "proto_py/proto/${file_name}.pb.cc")
            set(_ms_proto_py_out_h "proto_py/proto/${file_name}.pb.h")
            set(_ms_proto_py_out_py "proto_py/proto/${file_name}_pb2.py")
        else()
            set(_ms_proto_py_out_cc "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.cc")
            set(_ms_proto_py_out_h "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.h")
            set(_ms_proto_py_out_py "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py")
        endif()

        list(APPEND ${c_var} "${_ms_proto_py_out_cc}")
        list(APPEND ${h_var} "${_ms_proto_py_out_h}")
        list(APPEND ${py_var} "${_ms_proto_py_out_py}")
        if(WIN32)
            if(WIN32 AND NOT MSVC)
                add_custom_command(
                        OUTPUT "${_ms_proto_py_out_cc}"
                        "${_ms_proto_py_out_h}"
                        "${_ms_proto_py_out_py}"
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/proto_py/proto"
                        COMMAND protobuf::protoc -I${file_dir}
                                --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/proto_py/proto ${_ms_proto_input}
                        COMMAND protobuf::protoc -I${file_dir}
                                --python_out=${CMAKE_CURRENT_BINARY_DIR}/proto_py/proto ${_ms_proto_input}
                        COMMAND perl -pi.bak -e "s/import (.+_pb2.*)/from . import \\1/"
                                "${CMAKE_CURRENT_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                        COMMAND ${CMAKE_COMMAND} -E copy
                                "${CMAKE_CURRENT_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                                "${TOP_DIR}/mindspore/mindspore/python/mindspore/train/"
                        COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
            else()
                add_custom_command(
                        OUTPUT "${_ms_proto_py_out_cc}"
                        "${_ms_proto_py_out_h}"
                        "${_ms_proto_py_out_py}"
                        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto_py/proto"
                        COMMAND protobuf::protoc -I${file_dir}
                                --cpp_out=${CMAKE_BINARY_DIR}/proto_py/proto ${_ms_proto_input}
                        COMMAND protobuf::protoc -I${file_dir}
                                --python_out=${CMAKE_BINARY_DIR}/proto_py/proto ${_ms_proto_input}
                        COMMAND perl -pi.bak -e "s/import (.+_pb2.*)/from . import \\1/"
                                "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                                "${TOP_DIR}/mindspore/mindspore/python/mindspore/train/"
                        DEPENDS protobuf::protoc ${abs_file}
                        COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
            endif()
        else()
            add_custom_command(
                    OUTPUT "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.cc"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}.pb.h"
                    "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/proto_py/proto"
                    COMMAND protobuf::protoc -I${file_dir} --cpp_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND protobuf::protoc -I${file_dir} --python_out=${CMAKE_BINARY_DIR}/proto_py/proto ${abs_file}
                    COMMAND perl -pi -e "s/import (.+_pb2.*)/from . import \\1/"
                            "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                    COMMAND cp "${CMAKE_BINARY_DIR}/proto_py/proto/${file_name}_pb2.py"
                            "${TOP_DIR}/mindspore/mindspore/python/mindspore/train/"
                    DEPENDS protobuf::protoc ${abs_file}
                    COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)
        endif()
    endforeach()
    set_source_files_properties(${${c_var}} ${${h_var}} ${${py_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
    set(${py_var} ${${py_var}} PARENT_SCOPE)
endfunction()
