include_directories(${TOP_DIR}/mindspore)
include_directories(${TOP_DIR}/mindspore-lite/src)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")

find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl_cblas libacl_cblas.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

file(GLOB_RECURSE ASCEND_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "src/*.cc"
        "api/*.cc"
        "model/*.cc"
        "profiling/*.cc"
        )
set(ASCEND_SRC ${ASCEND_SRC} ${TOP_DIR}/mindspore-lite/src/litert/kernel/ascend/src/acl_mem_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model/acl_allocator.cc
        )
file(GLOB_RECURSE ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "../acl/*.cc")

set_property(SOURCE ${ASCEND_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

set_property(SOURCE ${ACL_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(ascend_kernel_plugin_mid OBJECT ${ASCEND_SRC} ${ACL_SRC})
add_library(ascend_kernel_plugin SHARED $<TARGET_OBJECTS:ascend_kernel_plugin_mid>)
add_dependencies(ascend_kernel_plugin fbs_inner_src)
add_dependencies(ascend_kernel_plugin mindspore-extendrt)
target_link_libraries(ascend_kernel_plugin mindspore-extendrt _mindspore_ascend_symbol_obj)

if("${MSLITE_REGISTRY_DEVICE}" STREQUAL "SD3403" AND PLATFORM_ARM64)
    find_library(acl_retr libacl_retr.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_runtime libruntime.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(ascend_kernel_plugin ${ge_graph} ${acl} ${acl_retr} ${acl_cblas} ${acl_runtime})
else()
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ge_compiler libge_compiler.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libplatform libplatform.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libcompress libcompress.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libopskernel libopskernel.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicore_utils libaicore_utils.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_CANN_RUNTIME_PATH}
        ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(ascend_kernel_plugin ${ge_graph} ${ge_compiler} ${acl_cblas} ${acl_dvpp} ${acl_runtime}
        ${libplatform} ${libcompress} ${libopskernel} ${libaicore_utils} ${libaicpu_engine_common} ${acl})
endif()
