set(TOP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(TEST_DIR ${TOP_DIR}/mindspore-lite/test)
set(LITE_DIR ${TOP_DIR}/mindspore-lite)

include_directories(${TOP_DIR})
include_directories(${TEST_DIR})
include_directories(${NNACL_DIR}/../)
# micro coder headers include each other module-relatively ("coder/..."), e.g. from the coder UTs.
include_directories(${LITE_DIR}/tools/converter/micro)

include(${TOP_DIR}/cmake/external_libs/gtest.cmake)
include(${TOP_DIR}/cmake/external_libs/mockcpp.cmake)

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    add_definitions(-DRUNTIME_CONVERT)
endif()

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    file(GLOB_RECURSE TEST_UT_SRC
            ${TEST_DIR}/main.cc
            ${TEST_DIR}/common/common_test.cc
        )
else()
    file(GLOB_RECURSE TEST_UT_SRC
            ${TEST_DIR}/main.cc
            ${TEST_DIR}/common/common_test.cc
            ${TEST_DIR}/ut/src/infer_test.cc
            ${TEST_DIR}/ut/src/flatbuffers_verifier_test.cc
            ${TEST_DIR}/ut/src/utils_test.cc
            ${TEST_DIR}/ut/src/scheduler_test.cc
            ${TEST_DIR}/ut/src/model_obfuscation_deprecated_test.cc
            ${TEST_DIR}/ut/src/runtime/dynamic_mem_manager_test.cc
            ${TEST_DIR}/ut/src/registry/registry_test.cc
            ${TEST_DIR}/ut/src/registry/registry_custom_op_test.cc
            ${TEST_DIR}/ut/src/common/ops/pooling_populate_test.cc
            ${TEST_DIR}/st/multiple_device_test.cc
            ${TEST_DIR}/st/mindrt_parallel_runtime_test.cc
            ${TEST_DIR}/st/mix_data_type_test.cc
            ${TEST_DIR}/ut/nnacl/fp32/*.cc
            ${TEST_DIR}/ut/nnacl/infer/*.cc
            ${TEST_DIR}/ut/src/runtime/kernel/arm/common/*.cc
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32/*.cc
            ${TEST_DIR}/ut/src/runtime/kernel/arm/string/*.cc
            ${TEST_DIR}/ut/src/api/context_c_test.cc
            ${TEST_DIR}/ut/src/api/model_c_test.cc
            ${TEST_DIR}/ut/src/api/tensor_c_test.cc
            )
endif()

if(MSLITE_ENABLE_RUNTIME_CONVERT)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/runtime_convert_tests.cc)
endif()

# UT for Status check
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/api/model_test.cc)
endif()

if(MSLITE_ENABLE_RUNTIME_PASS)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/runtime_pass_tests.cc)
endif()

if(MSLITE_ENABLE_TRAIN)
    file(GLOB_RECURSE TEST_TRAIN_UT_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32_grad/*.cc
            ${TEST_DIR}/ut/src/runtime/kernel/arm/cxx_api/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_TRAIN_UT_SRC})
endif()

if(MSLITE_ENABLE_SPARSE_COMPUTE)
    file(GLOB_RECURSE SPARSE_KERNEL_UT
            ${TEST_DIR}/ut/src/runtime/kernel/arm/fp32-sparsity/*.cc
            )
    list(APPEND TEST_UT_SRC ${SPARSE_KERNEL_UT})
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    if(PLATFORM_ARM)
        file(GLOB_RECURSE TEST_GPU_UT_SRC
                ${TEST_DIR}/ut/src/runtime/kernel/opencl/*.cc
                ${TEST_DIR}/ut/src/registry/registry_gpu_custom_op_test.cc
                )
        list(APPEND TEST_UT_SRC ${TEST_GPU_UT_SRC})
    endif()
endif()

if(MSLITE_ENABLE_DSP)
    file(GLOB_RECURSE TEST_GPU_UT_SRC
        ${TEST_DIR}/ut/src/runtime/kernel/dsp/*.cc
        )
    list(APPEND TEST_UT_SRC ${TEST_GPU_UT_SRC})
endif()

if(MSLITE_GPU_BACKEND STREQUAL cuda)
    set(CUDA_PATH $ENV{CUDA_HOME})
    include_directories(${CUDA_PATH}/include)
    include_directories(${OPS_DIR}/kernel/gpu)
    include_directories(${CUDA_PATH})
    file(GLOB_RECURSE TEST_GPU_UT_SRC
            ${TEST_DIR}/ut/src/extendrt/kernel/cuda/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_GPU_UT_SRC})
endif()

if(MSLITE_ENABLE_INT8 AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    file(GLOB_RECURSE TEST_INT8_UT_SRC
            ${TEST_DIR}/ut/src/runtime/kernel/arm/int8/*.cc
            ${TEST_DIR}/ut/nnacl/int8/*.cc
            ${TEST_DIR}/ut/nnacl/fp32/*.cc
            )
    list(APPEND TEST_UT_SRC ${TEST_INT8_UT_SRC})
endif()

if(MSLITE_ENABLE_FP16 AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    file(GLOB_RECURSE TEST_FP16_SRC
            ${TEST_DIR}/ut/nnacl/fp16/*.cc)
    list(APPEND TEST_UT_SRC ${TEST_FP16_SRC})
    if(SUPPORT_TRAIN)
        file(GLOB_RECURSE TEST_TRAIN_FP16_SRC
                ${TEST_DIR}/ut/src/runtime/kernel/arm/fp16_grad/*.cc)
        list(APPEND TEST_UT_SRC ${TEST_TRAIN_FP16_SRC})
    endif()
endif()

if(MSLITE_ENABLE_CONVERTER)
    if(MSLITE_ENABLE_TOOLS)
        set(TEST_CONVERTER_UT_SRC
                ${TEST_UT_SRC}
                ${LITE_DIR}/tools/benchmark/run_benchmark.cc
                ${LITE_DIR}/tools/benchmark/benchmark_base.cc
                ${LITE_DIR}/tools/benchmark/benchmark_unified_api.cc
                ${LITE_DIR}/tools/benchmark/benchmark_c_api.cc
                ${LITE_DIR}/tools/benchmark/benchmark.cc
                ${TEST_DIR}/st/benchmark_test.cc
                )
    endif()
    add_definitions(-DPRIMITIVE_WRITEABLE)

#    ================================================
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        file(GLOB_RECURSE TEST_CONVERTER_UT_SRC
                ${TEST_DIR}/ut/tools/converter/adapter/acl/mapper/*.cc
                ${TEST_DIR}/ut/tools/converter/parser/onnx/*.cc
                ${TEST_DIR}/ut/tools/converter/registry/pass_registry_position_ascend.cc
                )
    endif()
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        file(GLOB_RECURSE TEST_CONVERTER_UT_SRC
            ${TEST_DIR}/ut/tools/converter/decomposer/svd_test.cc
            ${TEST_DIR}/ut/tools/converter/micro_coder/*.cc
            ${TEST_DIR}/ut/tools/converter/registry/*.cc
            ${TEST_DIR}/ut/tools/converter/parser/onnx/*.cc
            ${TEST_DIR}/ut/tools/converter/parser/tflite/*.cc
            ${TEST_DIR}/ut/tools/converter/quantizer/*.cc
            ${TEST_DIR}/ut/tools/converter/api/*.cc
            ${TEST_DIR}/st/converter_test.cc
            ${TEST_DIR}/st/delegate_test.cc
            ${TEST_DIR}/st/mindrt_parallel_test.cc
            ${TEST_DIR}/st/sub_graph_test.cc
            ${TEST_DIR}/ut/src/dynamic_library_loader_test.cc
            ${TEST_DIR}/ut/tools/optimizer/fusion/*.cc
            ${TEST_DIR}/ut/tools/optimizer/graph/*.cc
            )
    endif()
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        list(REMOVE_ITEM TEST_CONVERTER_UT_SRC ${TEST_DIR}/st/mindrt_parallel_test.cc)
        list(REMOVE_ITEM TEST_UT_SRC ${TEST_DIR}/st/benchmark_test.cc)
        list(REMOVE_ITEM TEST_CONVERTER_UT_SRC ${TEST_DIR}/st/sub_graph_test.cc)
    endif()

    set(TEST_CONVERTER_UT_SRC
        ${TEST_CONVERTER_UT_SRC}
        ${TEST_DIR}/main.cc
        ${TEST_DIR}/ut/utils/build_func_graph.cc
        ${TEST_DIR}/common/common_test.cc
        )
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        set(TEST_CONVERTER_UT_SRC
            ${TEST_CONVERTER_UT_SRC}
            ${TEST_DIR}/common/import_from_meta_graphT.cc
        )
    endif()
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    list(APPEND TEST_UT_SRC ${TEST_DIR}/ut/src/runtime/kernel_executor_tests.cc)
endif()

set(TEST_LITE_SRC ${TEST_LITE_SRC} ${LITE_DIR}/src/litert/cxx_api/kernel.cc)

set(TEST_SRC
        ${TEST_UT_SRC}
        ${TEST_LITE_SRC}
        )

if(MSLITE_ENABLE_SERVER_INFERENCE OR MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    list(APPEND TEST_SRC ${TEST_DIR}/ut/src/api/model_parallel_runner_test.cc)
endif()

# Converter parser unit tests (resource-free, run against the converter library).
# The big TEST_CONVERTER_UT_SRC list assembled inside the ENABLE_CONVERTER branch
# is not consumed by any target, so register these into lite-test explicitly.
if(MSLITE_ENABLE_CONVERTER)
  list(APPEND TEST_SRC
    ${TEST_DIR}/ut/tools/converter/parser/onnx/onnx_pool_parser_test.cc
    ${TEST_DIR}/ut/tools/converter/parser/tflite/tflite_add_parser_test.cc
    ${TEST_DIR}/ut/tools/converter/parser/tflite/tflite_util_test.cc
    ${TEST_DIR}/ut/tools/converter/preprocess/image_preprocess_test.cc)
  set_source_files_properties(
    ${TEST_DIR}/ut/tools/converter/parser/onnx/onnx_pool_parser_test.cc
    ${TEST_DIR}/ut/tools/converter/parser/tflite/tflite_add_parser_test.cc
    ${TEST_DIR}/ut/tools/converter/parser/tflite/tflite_util_test.cc
    ${TEST_DIR}/ut/tools/converter/preprocess/image_preprocess_test.cc
    PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}"
      COMPILE_OPTIONS "-Wno-error")
endif()

add_executable(lite-test ${TEST_SRC})
add_dependencies(lite-test fbs_src fbs_inner_src)

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    target_link_libraries(lite-test mindspore-extendrt dl mindspore::gtest)
else()
    target_link_libraries(lite-test mindspore-lite dl mindspore::gtest)
endif()

# Converter parser unit tests registered above link against the converter library
# and its direct dependencies (the linker does not resolve symbols through
# shared-library DT_NEEDED entries during static link). Everything is linked by
# target name so the build stays independent of per-config output locations.
if(MSLITE_ENABLE_CONVERTER)
    set(MS_UT_CONVERTER_DEPS mindspore::glog mindspore::protobuf)
    if(TARGET mindspore::opencv_core)
        list(APPEND MS_UT_CONVERTER_DEPS mindspore::opencv_core
            mindspore::opencv_imgcodecs mindspore::opencv_imgproc)
    endif()
    target_link_libraries(lite-test
        mindspore_converter
        mindspore_core
        mindspore_ops
        mslite_converter_plugin
        ${MS_UT_CONVERTER_DEPS})
    # CMAKE_SKIP_BUILD_RPATH is ON globally (secure_option.cmake), so no rpath is
    # baked into the binary: generate a wrapper exporting the library paths. All
    # runtime dirs are resolved from the linked targets with generator expressions.
    set(LITE_UT_RUNTIME_DIRS
        "$<TARGET_FILE_DIR:mindspore-lite>"
        "$<TARGET_FILE_DIR:mindspore_converter>"
        "$<TARGET_FILE_DIR:mindspore_core>"
        "$<TARGET_FILE_DIR:mindspore_ops>"
        "$<TARGET_FILE_DIR:mslite_converter_plugin>"
        "$<TARGET_FILE_DIR:mindspore::glog>"
        "$<TARGET_FILE_DIR:mindspore::gtest>")
    if(TARGET mindspore::opencv_core)
        list(APPEND LITE_UT_RUNTIME_DIRS "$<TARGET_FILE_DIR:mindspore::opencv_core>")
    endif()
    list(JOIN LITE_UT_RUNTIME_DIRS ":" LITE_UT_LD_PATHS)
    # The CONTENT below is written verbatim into build/test/lite-test.sh (CMake only
    # writes the file); the export itself runs when that script is executed by bash.
    file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/test/lite-test.sh
        CONTENT "#!/bin/bash
export LD_LIBRARY_PATH=\"${LITE_UT_LD_PATHS}:\$LD_LIBRARY_PATH\"
exec ${CMAKE_BINARY_DIR}/test/lite-test \"\$@\"
")
endif()

# test file using gmock
target_link_libraries(lite-test mindspore::gmock)
add_library(gmock_tests OBJECT ${TEST_DIR}/st/optimize_allocator_test.cc)
add_dependencies(gmock_tests fbs_src fbs_inner_src)
target_compile_options(gmock_tests PRIVATE -fno-rtti)
target_link_libraries(lite-test gmock_tests)

if(MSLITE_ENABLE_TRAIN)
    target_link_libraries(lite-test mindspore-lite-train)
endif()

if(PLATFORM_ARM AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    if(NOT TARGET_FT78 AND NOT TARGET_FT04)
        target_link_libraries(lite-test log)
    endif()
else()
    target_link_libraries(lite-test mindspore::securec pthread)
endif()

if(MSLITE_ENABLE_NPU)
    target_link_libraries(lite-test c++_shared)
endif()

if(MSLITE_ENABLE_MINDRT)
    add_library(mindrt_test_mid OBJECT ${TEST_DIR}/ut/src/lite_mindrt_test.cc)
    add_dependencies(mindrt_test_mid fbs_src fbs_inner_src)
    if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        target_compile_options(mindrt_test_mid PRIVATE -fno-rtti)
    endif()
    target_link_libraries(lite-test mindrt_test_mid)
endif()

if(MSLITE_ENABLE_CONVERTER AND (NOT MSLITE_ENABLE_RUNTIME_CONVERT))
    add_executable(lite-test-converter ${TEST_CONVERTER_UT_SRC})
    target_compile_definitions(lite-test-converter PRIVATE PRIMITIVE_WRITEABLE USE_GLOG)
    target_link_libraries(lite-test-converter dl mindspore::gtest)
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        if(PLATFORM_ARM)
            target_link_libraries(lite-test-converter log)
        else()
            target_link_libraries(lite-test-converter mindspore::securec pthread)
        endif()
        add_dependencies(lite-test-converter fbs_src fbs_inner_src)
        target_link_libraries(lite-test-converter
                converter_runtime_mid
                cpu_ops_mid
                nnacl_mid
                cpu_kernel_mid
                lite_exporter_mid
                mslite_converter_plugin
                graph_pass_mid
                fusion_mid
                quantizer_mid
                decomposer_mid
                proto_mid
                # ccsrc_src_mid
                ccsrc_src_new_mid
                converter_src_mid
                mindspore::protobuf
                mindspore::json
                -Wl,--whole-archive mindspore_core -Wl,--no-whole-archive
                mindspore_ops
                mslite_primitive_mid
                mindspore::glog
                mindspore::flatbuffers
                preprocess_mid
                config_parser_mid
                coder_mid
                ccsrc_debug_common_mid_
                mindir_proto_mid
                _mindspore_utils_ir_dump_onnx_obj
                mindir_serializer_mid
                )
    else()
        target_link_libraries(lite-test-converter
                converter_runtime_mid
                ascend_pass_plugin
                )
    endif()

    if(SUPPORT_TRAIN)
        target_link_libraries(lite-test-converter train_cpu_kernel_mid)
    endif()
endif()

if(MSLITE_ENABLE_CONVERTER AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    target_link_libraries(lite-test-converter tflite_parser_mid caffe_parser_mid
                                    onnx_parser_mid tf_parser_mid)
endif()


if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    target_link_libraries(lite-test kernel_executor)
endif()

if(TARGET_FT78)
    set(PNNACORE_DRIVERS "$ENV{DSP_SDK_PATH}/pnna_core/drivers")
    target_link_options(lite-test PRIVATE
        "-Wl,-rpath-link,${PNNACORE_DRIVERS}"
        "-Wl,-rpath,${PNNACORE_DRIVERS}"
    )
endif()
