file(GLOB FUSION_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pattern.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fusion_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_cast_fusion_pass.cc
        )
set_property(SOURCE ${FUSION_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(fusion_mid OBJECT ${FUSION_SRC})

add_dependencies(fusion_mid fbs_src)
add_dependencies(fusion_mid fbs_inner_src)
target_link_libraries(fusion_mid mindspore::securec)
