file(GLOB GRAPH_PASS
        ${CMAKE_CURRENT_SOURCE_DIR}/dtype_trans_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/isolated_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/topological_sort_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dropout_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infershape_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_quant_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/infer_quant_param_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/convert_fp32_to_fp16_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/set_unused_quant_param_to_default_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tensor_name_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_node_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tensor_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/const_node_reorder_pass.cc
        )
set_property(SOURCE ${GRAPH_PASS} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(graph_pass_mid OBJECT ${GRAPH_PASS})

add_dependencies(graph_pass_mid fbs_src)
add_dependencies(graph_pass_mid fbs_inner_src)
