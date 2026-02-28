file(GLOB_RECURSE MINDIR_EXPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${MINDIR_EXPORTER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

file(GLOB PROTO_FILE ""
        ${CORE_DIR}/proto/mind_ir.proto
        ${CCSRC_DIR}/utils/*.proto
        )
ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(mindir_proto_mid OBJECT ${PROTO_SRCS})

add_compile_definitions(MINDIR_EXPORT_TENSOR_LAYOUT_CLIP)
add_compile_definitions(COMMON_DLL)

set(IR_DUMP_ONNX ${CCSRC_DIR}/utils/ir_dump/onnx)
add_subdirectory(${IR_DUMP_ONNX} ir_dump_onnx)
add_dependencies(_mindspore_utils_ir_dump_onnx_obj mindir_proto_mid)

add_library(mindir_serializer_mid OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/mindir_serializer.cc
        )
add_dependencies(mindir_serializer_mid fbs_src fbs_inner_src)
file(STRINGS "${TOP_DIR}/version.txt" VERSION)
add_definitions(-DVERSION=\"${VERSION}\")
add_library(ccsrc_debug_common_mid_ OBJECT
        ${CMAKE_CURRENT_SOURCE_DIR}/export_depend/mindir_exporter.cc
        )
add_dependencies(ccsrc_debug_common_mid_ mindir_proto_mid)

target_link_libraries(mindir_serializer_mid
        _mindspore_utils_ir_dump_onnx_obj
        ccsrc_debug_common_mid_
        mindir_proto_mid
        )
