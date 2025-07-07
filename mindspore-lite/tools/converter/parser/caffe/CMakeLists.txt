include(${TOP_DIR}/mindspore-lite/cmake/merge.cmake)
merge_parser(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/tools/converter/parser/caffe/caffe_op_parser.cc)
file(GLOB_RECURSE CAFFE_SRC_LIST ${CMAKE_BINARY_DIR}/tools/converter/parser/caffe/caffe_op_parser.cc)

set_property(SOURCE ${CAFFE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(caffe_parser_mid OBJECT ${CAFFE_SRC_LIST})

add_dependencies(caffe_parser_mid proto_mid)
add_dependencies(caffe_parser_mid fbs_src)
add_dependencies(caffe_parser_mid fbs_inner_src)
