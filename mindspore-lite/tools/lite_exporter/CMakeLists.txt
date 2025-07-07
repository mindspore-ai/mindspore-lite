file(GLOB_RECURSE ANF_EXPORTER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${ANF_EXPORTER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(lite_exporter_mid OBJECT
        ${ANF_EXPORTER_SRC_LIST}
        )

add_dependencies(lite_exporter_mid fbs_src)
add_dependencies(lite_exporter_mid fbs_inner_src)
