include_directories(${CCSRC_DIR})

file(STRINGS "${TOP_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")
add_compile_definitions(ENABLE_SECURITY)

file(GLOB LLM_ENGINE_PLUGIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llm_engine_plugin.cc
    ${CCSRC_DIR}/utils/config_manager.cc
    )

add_library(llm_engine_plugin SHARED ${LLM_ENGINE_PLUGIN_SRC})

find_library(llm_engine libllm_engine.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ge_graph libgraph.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

target_link_libraries(llm_engine_plugin ${llm_engine} ${ge_graph} ${acl}
                      mindspore_converter mindspore_core mindspore_ops mindspore_graph_ir)

target_link_libraries(llm_engine_plugin mindspore-extendrt)
