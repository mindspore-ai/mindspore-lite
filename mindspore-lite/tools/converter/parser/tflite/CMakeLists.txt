if(MSLITE_MERGE_DISABLE)
    file(GLOB_RECURSE TFLITE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
else()
    include(${TOP_DIR}/mindspore-lite/cmake/merge.cmake)
    merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/tflite/tflite_op_parser.cc)
    file(GLOB_RECURSE TFLITE_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/tflite/tflite_op_parser.cc)
endif()

set_property(SOURCE ${TFLITE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE
        FLATBUFFERS_LOCALE_INDEPENDENT=0
        )
add_library(tflite_parser_mid OBJECT
        ${TFLITE_SRC_LIST}
        )
add_dependencies(tflite_parser_mid tflite_fbs_src)
add_dependencies(tflite_parser_mid fbs_src)
add_dependencies(tflite_parser_mid fbs_inner_src)
if(ENABLE_NATIVE_FLATBUFFER)
    target_link_libraries(tflite_parser_mid ${FLAT_BUFFERS})
else()
    target_link_libraries(tflite_parser_mid mindspore::flatbuffers)
endif()