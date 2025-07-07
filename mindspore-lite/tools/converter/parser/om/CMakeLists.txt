file(GLOB_RECURSE OM_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
set_property(SOURCE ${OM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(om_parser_mid OBJECT
        ${OM_SRC_LIST}
        )
add_dependencies(om_parser_mid proto_mid)
add_dependencies(om_parser_mid fbs_src)
add_dependencies(om_parser_mid fbs_inner_src)
