message(STATUS "Compiling metadef proto file")
message(STATUS "[ME] build_path: ${BUILD_PATH}")

function(ge_protobuf_generate c_var h_var)
    common_protobuf_generate(${CMAKE_BINARY_DIR}/proto/ge/proto ${c_var} ${h_var} ${ARGN})
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)
endfunction()

if(MSLITE_ENABLE_ACL)
    set(METADEF_PATH "${TOP_DIR}/metadef")
else()
    set(METADEF_PATH "${CMAKE_SOURCE_DIR}/metadef")
endif()

if(BUILD_LITE)
    file(GLOB_RECURSE GE_PROTO_FILE ${TOP_DIR}/metadef/proto/*.proto)
else()
    file(GLOB_RECURSE GE_PROTO_FILE ${METADEF_PATH}/proto/*.proto)
endif()

set(TMP_FILE_NAME_LIST)
foreach(file ${GE_PROTO_FILE})
    get_filename_component(file_name ${file} NAME_WE)
    list(FIND TMP_FILE_NAME_LIST ${file_name} OUT_VAR)
    if(NOT ${OUT_VAR} EQUAL "-1")
        list(REMOVE_ITEM GE_PROTO_FILE ${file})
    endif()
    list(APPEND TMP_FILE_NAME_LIST ${file_name})
endforeach()
ge_protobuf_generate(GE_PROTO_SRCS GE_PROTO_HDRS ${GE_PROTO_FILE})
add_library(ge_proto SHARED ${GE_PROTO_SRCS})
if(NOT MSVC)
    set_target_properties(ge_proto PROPERTIES COMPILE_FLAGS "-Wno-unused-veriable -Wno-array-bounds")
endif()

