file(GLOB_RECURSE PREPROCESS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *.cc
        )
set_property(SOURCE ${PREPROCESS_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
add_library(preprocess_mid OBJECT
        ${PREPROCESS_SRC_LIST}
        )
add_dependencies(preprocess_mid fbs_src)
add_dependencies(preprocess_mid fbs_inner_src)
if(MSLITE_DEPS_OPENCV)
target_link_libraries(preprocess_mid
        mindspore::opencv_core
        mindspore::opencv_imgcodecs
        mindspore::opencv_imgproc)
endif()