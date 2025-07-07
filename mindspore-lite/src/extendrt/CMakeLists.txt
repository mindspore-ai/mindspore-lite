string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-fvisibility-inlines-hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fvisibility=hidden" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    if(PLATFORM_ARM64)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    endif()
    set(ENABLE_CPU on)
    set(LOAD_PLUGIN_STATIC on)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    remove_definitions(-DBUILD_LITE_INFERENCE)

    set(MSLITE_EXTEND_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/numa_adapter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/model_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/delegate_session.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/tensor_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/serialization.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/func_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/comm_group_info.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/factory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/litert_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/ascend_ge_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/plugin/ascend_acl_executor_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate_graph_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/delegate/graph_executor/litert/func_graph_reuse_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/execution_flow.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/execution_plan.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../infer/primitive_type.cc
        )

    if(MSLITE_ENABLE_BFC_MEMORY)
        set(MSLITE_EXTEND_RUNTIME_SRC ${MSLITE_EXTEND_RUNTIME_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_allocator.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_mem_manager.cc
            )
    endif()

    include(${TOP_DIR}/cmake/external_libs/pocketfft.cmake)
    set_property(SOURCE ${MSLITE_EXTEND_RUNTIME_SRC} PROPERTY COMPILE_DEFINITIONS
            LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
            SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
    add_library(mindspore-extendrt SHARED ${MSLITE_EXTEND_RUNTIME_SRC} ${MSLITE_EXTEND_CPU_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt lite_src_common_mid)
    target_link_libraries(mindspore-extendrt lite_src_common_mid)
    add_dependencies(mindspore-extendrt fbs_src fbs_inner_src)
    add_subdirectory(cxx_api)
    add_subdirectory(delegate/graph_executor/litert)
    add_library(mindspore-extendrt_static STATIC ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static lite_src_common_mid)
    target_link_libraries(mindspore-extendrt_static lite_src_common_mid)
    add_dependencies(mindspore-extendrt_static fbs_src fbs_inner_src)
    target_link_libraries(mindspore-extendrt mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt mindspore_core mindspore_ops mindspore::protobuf)
    target_link_libraries(mindspore-extendrt_static mindspore_infer_shared_lib_obj)
    target_link_libraries(mindspore-extendrt_static mindspore_core mindspore_ops mindspore::protobuf)
    add_dependencies(mindspore-extendrt_static msplugin-ge-litert)
    target_link_libraries(mindspore-extendrt_static msplugin-ge-litert)

    add_subdirectory(${LITE_DIR}/src/executor unified_executor)

    if(NOT WIN32)
        target_link_libraries(mindspore-extendrt dl)
        target_link_libraries(mindspore-extendrt_static dl)
    endif()

    if(MSLITE_ENABLE_ACL)
        add_subdirectory(delegate/ascend_ge)
        add_subdirectory(delegate/ascend_acl)
    endif()
    if(MSLITE_ENABLE_CONVERTER)
        add_subdirectory(convert)
        target_link_libraries(mindspore-extendrt -Wl,--no-as-needed mindspore_converter)
    endif()
else()
    set(MSLITE_EXTEND_RUNTIME_SRC ${MODEL_LOADER_FRAMEWORK_SRC})
    add_library(mindspore-extendrt OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt fbs_src fbs_inner_src)
    add_library(mindspore-extendrt_static OBJECT ${MSLITE_EXTEND_RUNTIME_SRC})
    add_dependencies(mindspore-extendrt_static fbs_src fbs_inner_src)
endif()

set_target_properties(mindspore-extendrt PROPERTIES OUTPUT_NAME "mindspore-lite")
