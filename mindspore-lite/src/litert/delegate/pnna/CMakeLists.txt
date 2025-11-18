include_directories($ENV{DSP_SDK_PATH}/pnna_core/include)
file(GLOB_RECURSE PNNA_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc)
add_library(pnna_kernel_mid OBJECT ${PNNA_RUNTIME_SRC})
add_dependencies(pnna_kernel_mid fbs_src)
