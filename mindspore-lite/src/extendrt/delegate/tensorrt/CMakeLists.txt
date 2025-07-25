set(CUDA_PATH $ENV{CUDA_HOME})
include_directories(${OPS_DIR}/kernel/gpu)
set(CUDA_VERSION 11.1)
set(CUDA_LIB_PATH ${CUDA_PATH}/lib64)
include_directories(${CUDA_PATH})
include_directories(${CUDA_PATH}/include)
find_package(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN/")
add_compile_definitions(GPU_TENSORRT)
set(TENSORRT_PATH $ENV{TENSORRT_PATH})
set(TENSORRT_LIB_PATH ${TENSORRT_PATH}/lib)
include_directories(${TENSORRT_PATH}/include)

include_directories(${OPS_DIR}/kernel/cpu)
include_directories(${CCSRC_DIR}/../)
include_directories(${OPS_DIR}/kernel/gpu/cuda_impl/cuda_ops)

if(DEFINED ENV{MS_ENABLE_CUDA_DISTRIBUTION})
    set(MS_ENABLE_CUDA_DISTRIBUTION $ENV{MS_ENABLE_CUDA_DISTRIBUTION})
else()
    set(MS_ENABLE_CUDA_DISTRIBUTION "off")
endif()

set(NCCL_MPI_SRC_STUB
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/distribution_collective.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/distribution/distribution_base.cc
    )

# nccl mpi
if(MS_ENABLE_CUDA_DISTRIBUTION STREQUAL "on")
    message("enable cuda gpu distribution collective")
    file(GLOB NCCL_MPI_SRC LIST_DIRECTORIES false
        ${CMAKE_CURRENT_SOURCE_DIR}/distribution/*.cc
        ${CCSRC_DIR}/plugin/device/gpu/hal/device/distribution/collective_wrapper.cc
        ${CCSRC_DIR}/plugin/device/gpu/hal/device/distribution/mpi_wrapper.cc
        ${CCSRC_DIR}/plugin/device/gpu/hal/device/distribution/nccl_wrapper.cc
        )
    list(REMOVE_ITEM NCCL_MPI_SRC ${NCCL_MPI_SRC_STUB})

    add_compile_definitions(LITE_CUDA_DISTRIBUTION)
    include(${TOP_DIR}/cmake/external_libs/ompi.cmake)
    include(${TOP_DIR}/cmake/external_libs/nccl.cmake)

    add_library(gpu_distribution_collective OBJECT ${NCCL_MPI_SRC})
    add_library(mindspore::nccl ALIAS nccl::nccl)
    add_library(mindspore::ompi ALIAS ompi::mpi)
    target_link_libraries(gpu_distribution_collective PRIVATE mindspore::ompi mindspore::nccl)
else()
    add_library(gpu_distribution_collective OBJECT ${NCCL_MPI_SRC_STUB})
endif()
add_dependencies(gpu_distribution_collective fbs_src)

file(GLOB TENSORRT_RUNTIME_SRC LIST_DIRECTORIES false
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_impl/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../extendrt/delegate/delegate_utils.cc
    ${OPS_DIR}/kernel/gpu/cuda_impl/cuda_ops/cuda_device_info.cc
    ${OPS_DIR}/kernel/cpu/nnacl/nnacl_common.c
    ${TOP_DIR}/mindspore-lite/src/common/file_utils.cc
    )

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache)

#set(TENSORRT_RUNTIME_SRC
#    ${TENSORRT_RUNTIME_SRC}
#    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/embedding_cache_manager.cc
#    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/load_host_cache_model.cc
#    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/lfu_cache.cc
#    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/embedding_cache.cc
#    ${CMAKE_CURRENT_SOURCE_DIR}/../parameter_cache/gpu/gpu_cache_mem.cc
#    )

link_libraries(${CUDA_LIB_PATH}/libcudnn.so)
link_libraries(${CUDA_LIB_PATH}/libcublasLt.so)

add_library(libcudart SHARED IMPORTED)
set_target_properties(libcudart PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcudart.so)

add_library(libnvinfer SHARED IMPORTED)
set_target_properties(libnvinfer PROPERTIES IMPORTED_LOCATION ${TENSORRT_LIB_PATH}/libnvinfer.so)

add_library(libcublas SHARED IMPORTED)
set_target_properties(libcublas PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/libcublas.so)
add_library(tensorrt_plugin SHARED ${TENSORRT_RUNTIME_SRC})

add_dependencies(tensorrt_plugin fbs_src)

target_link_libraries(
    tensorrt_plugin
    libcudart
    libcublas
    libnvinfer
)
if(SUPPORT_TENSORRT AND (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    add_library(libcuda SHARED IMPORTED)
    set_target_properties(libcuda PROPERTIES IMPORTED_LOCATION ${CUDA_LIB_PATH}/stubs/libcuda.so)
    target_link_libraries(
        tensorrt_plugin
        libcuda
    )
endif()

add_subdirectory(cuda_impl)

target_link_libraries(tensorrt_plugin cuda_kernel_mid gpu_distribution_collective)
target_link_libraries(tensorrt_plugin mindspore-extendrt mindspore_core mindspore_ops)
