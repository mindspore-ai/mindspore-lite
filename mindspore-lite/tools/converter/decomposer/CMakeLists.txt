file(GLOB DECOMPOSER
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )
set_property(SOURCE ${DECOMPOSER} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
    add_library(decomposer_mid OBJECT ${DECOMPOSER})

    add_dependencies(decomposer_mid fbs_src)
    add_dependencies(decomposer_mid fbs_inner_src)
endif()