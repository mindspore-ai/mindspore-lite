set(KERNEL_REG_DIR ${TOP_DIR}/mindspore-lite/src/registry)
set(CONVERTER_DIR ${TOP_DIR}/mindspore-lite/tools/converter)
file(GLOB CONVERT_REG_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB KERNEL_REG_SRC ${KERNEL_REG_DIR}/*.cc)
set(REG_SRC ${CONVERT_REG_SRC}
        ${KERNEL_REG_SRC}
        ${KERNEL_REG_DIR}/../litert/cxx_api/context.cc
        ${KERNEL_REG_DIR}/../litert/cxx_api/types.cc
        ${KERNEL_REG_DIR}/../litert/cxx_api/tensor/tensor_impl.cc
        ${KERNEL_REG_DIR}/../litert/cxx_api/tensor_utils.cc
        ${KERNEL_REG_DIR}/../tensor.cc
        ${KERNEL_REG_DIR}/../litert/inner_allocator.cc
        ${KERNEL_REG_DIR}/../common/string_util.cc
        ${KERNEL_REG_DIR}/../common/utils.cc
        ${KERNEL_REG_DIR}/../extendrt/delegate/tensorrt/distribution/distribution_base.cc
        ${KERNEL_REG_DIR}/../extendrt/delegate/plugin/tensorrt_executor_plugin.cc
        ${KERNEL_REG_DIR}/../extendrt/kernel/ascend/plugin/ascend_allocator_plugin.cc
        ${CONVERTER_DIR}/converter_context.cc
        ${TOP_DIR}/mindspore/mindspore/ops/kernel/cpu/nnacl/tensor_c_utils.c
        ${TOP_DIR}/mindspore-lite/src/common/file_utils.cc
        )
set_property(SOURCE ${REG_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(mslite_converter_plugin SHARED ${REG_SRC})
target_link_libraries(mslite_converter_plugin mindspore::glog)
add_dependencies(mslite_converter_plugin fbs_src)
add_dependencies(mslite_converter_plugin fbs_inner_src)

if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    target_link_libraries(mslite_converter_plugin mindspore_core mindspore_ops)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release" AND (NOT WIN32))
    add_custom_command(TARGET mslite_converter_plugin POST_BUILD COMMAND strip
            ${CMAKE_CURRENT_BINARY_DIR}/libmslite_converter_plugin.so)
endif()
