file(GLOB_RECURSE COMMON_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp_allocator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp_runtime_wrapper.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp_runtime.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dsp_subgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/custom_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/converter/adapter/dsp/infer/*.cc
    )

file(GLOB_RECURSE DSP_KERNEL_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/${MSLITE_REGISTRY_DEVICE}/*.cc
    )

add_library(dsp_kernel_mid OBJECT ${DSP_KERNEL_SRC} ${COMMON_SRC})
add_dependencies(dsp_kernel_mid fbs_src)
target_include_directories(dsp_kernel_mid PRIVATE $ENV{DSP_SDK_PATH})
