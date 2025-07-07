include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")

link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
file(GLOB ACL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/api/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mapper/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer/*.cc
        ${TOP_DIR}/mindspore-lite/src/extendrt/utils/serialization.cc
        ${TOP_DIR}/mindspore-lite/src/extendrt/cxx_api/serialization.cc
        )

set(ACL_SRC ${ACL_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/acl_pass.cc)

set(ENABLE_ACL on)
set(MODE_ASCEND_ACL off)
add_subdirectory(${TOP_DIR}/mindspore/mindspore/ccsrc/backend/ge_backend/graph_ir _mindspore_transform_graph_ir)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cxx_api_lite/cxx_api mslite_shared_lib)


set_property(SOURCE ${ACL_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(ascend_pass_plugin SHARED ${ACL_SRC})
target_link_libraries(ascend_pass_plugin mslite_shared_lib)
add_dependencies(ascend_pass_plugin mslite_shared_lib)
add_dependencies(ascend_pass_plugin fbs_inner_src)
add_dependencies(ascend_pass_plugin mindspore_converter)
target_link_libraries(ascend_pass_plugin mindspore_converter lite_src_common_mid)