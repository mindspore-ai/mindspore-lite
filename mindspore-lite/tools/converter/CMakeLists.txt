add_definitions(-DPRIMITIVE_WRITEABLE)
add_definitions(-DUSE_GLOG)
set(USE_GLOG on)
set(MSLITE_MERGE_DISABLE on)
if(MSLITE_ENABLE_MODEL_ENCRYPTION AND MSLITE_DEPS_OPENSSL)
    add_compile_definitions(ENABLE_OPENSSL)
endif()
set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)

if(ENABLE_GPU)
    add_compile_definitions(ENABLE_GPU)
endif()

include_directories(${NNACL_DIR}/..)

file(GLOB_RECURSE CONVERTER_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/offline_packing_optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_packed_node.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_funcgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/converter_metagraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/anf_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/anf_transform_for_ge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graphdef_transform.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/export_model.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/optimizer_manager.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/session/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/graph_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/meta_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/node_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/tensor_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/string_util.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/protobuf_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/func_graph_subgraph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/flag_parser.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/storage.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/meta_graph_serializer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/parse_config_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/statistic_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/func_graph_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/custom_ascend_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/parser_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/tf_bidirection_gru_cf_fusion.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unused_node_remove_pass.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv1d_inout_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/conv2d_transpose_input_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/einsum_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/inputs_adjust.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/parser/unify_format.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter/acl/plugin/acl_pass_plugin.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/import/convert_extend_ops/*.cc
        ${SRC_DIR}/extendrt/delegate/plugin/ascend_ge_executor_plugin.cc
        ${SRC_DIR}/common/quant_utils.cc
        ${SRC_DIR}/common/dynamic_library_loader.cc
        ${SRC_DIR}/train/train_populate_parameter.cc
        ${SRC_DIR}/common/config_file.cc
        ${SRC_DIR}/common/config_infos.cc
        ${SRC_DIR}/common/primitive_t_utils.cc
        ../optimizer/*.cc
        )

if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_subdirectory(adapter/dpico)
endif()
add_subdirectory(../lite_exporter lite_exporter)
add_subdirectory(../mindir_exporter mindir_serializer)
add_subdirectory(parser/caffe)
add_subdirectory(parser/tflite)
add_subdirectory(parser/onnx)
add_subdirectory(parser/tf)
add_subdirectory(parser/om)
if(ENABLE_CONVERT_PYTORCH_MODEL)
    add_subdirectory(parser/pytorch)
endif()

add_subdirectory(legacy_optimizer)
add_subdirectory(quantizer)
add_subdirectory(decomposer)
add_subdirectory(registry)
add_subdirectory(preprocess)
add_subdirectory(config_parser)
add_subdirectory(micro/coder)

add_subdirectory(${CORE_DIR} mindspore_core)
add_subdirectory(${OPS_DIR} mindspore_ops)

if(MSLITE_ENABLE_ACL)
    set(MODE_ASCEND_ACL ON)
    add_subdirectory(adapter/acl)
    link_directories(${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
endif()

file(GLOB CXX_API_SRCS
        ${SRC_DIR}/litert/cxx_api/*.cc
        ${SRC_DIR}/litert/cxx_api/model/*.cc
        ${SRC_DIR}/litert/cxx_api/graph/*.cc
        ${SRC_DIR}/litert/cxx_api/tensor/*.cc)

if(MSLITE_ENABLE_RUNTIME_PASS)
        file(GLOB KERNEL_ONLINE_FUSION_SRC ${SRC_DIR}/litert/pass/online_fusion/*.cc)
endif()

file(GLOB FORMAT_PASS_SRCS ${SRC_DIR}/litert/pass/format_pass/*.cc)
set(LITE_SRC ${API_SRC}
        ${CXX_API_SRCS}
        ${FORMAT_PASS_SRCS}
        ${SRC_DIR}/common/ops/ops_def.cc
        ${SRC_DIR}/common/ops/ops_utils.cc
        ${SRC_DIR}/common/ops/anf_utils.cc
        ${SRC_DIR}/common/utils.cc
        ${SRC_DIR}/common/file_utils.cc
        ${SRC_DIR}/common/mmap_utils.cc
        ${SRC_DIR}/common/context_util.cc
        ${SRC_DIR}/common/graph_util.cc
        ${SRC_DIR}/common/string_util.cc
        ${SRC_DIR}/common/log.cc
        ${SRC_DIR}/common/prim_util.cc
        ${SRC_DIR}/common/tensor_util.cc
        ${SRC_DIR}/common/decrypt.cc
        ${SRC_DIR}/litert/allocator.cc
        ${SRC_DIR}/litert/inner_allocator.cc
        ${SRC_DIR}/litert/runtime_allocator.cc
        ${SRC_DIR}/litert/infer_manager.cc
        ${SRC_DIR}/litert/runtime_shape_fusion_pass.cc
        ${SRC_DIR}/litert/runtime_pass.cc
        ${SRC_DIR}/litert/inner_context.cc
        ${SRC_DIR}/litert/thread_pool_reuse_manager.cc
        ${SRC_DIR}/tensor.cc
        ${SRC_DIR}/litert/tensor_category.cc
        ${SRC_DIR}/litert/schema_tensor_wrapper.cc
        ${SRC_DIR}/tensorlist.cc
        ${SRC_DIR}/litert/kernel_registry.cc
        ${SRC_DIR}/registry/register_kernel_impl.cc
        ${SRC_DIR}/litert/lite_kernel.cc
        ${SRC_DIR}/executor/kernel_exec.cc
        ${SRC_DIR}/litert/kernel_exec_util.cc
        ${SRC_DIR}/litert/scheduler.cc
        ${SRC_DIR}/executor/sub_graph_kernel.cc
        ${SRC_DIR}/litert/sub_graph_split.cc
        ${KERNEL_ONLINE_FUSION_SRC}
        ${SRC_DIR}/litert/lite_session.cc
        ${SRC_DIR}/litert/runtime_packed_node_pass.cc
        ${SRC_DIR}/litert/executor.cc
        ${SRC_DIR}/litert/lite_model.cc
        ${SRC_DIR}/litert/model_manager.cc
        ${SRC_DIR}/errorcode.cc
        ${SRC_DIR}/litert/weight_decoder.cc
        ${SRC_DIR}/litert/pack_weight_manager.cc
        ${SRC_DIR}/litert/huffman_decode.cc
        ${SRC_DIR}/infer/primitive_type.cc
        ${LITE_DIR}/src/extendrt/mock/lite_runtime/populate/base_operator_populate_register.cc
        ${SRC_DIR}/control_flow/control_flow_scheduler.cc
        ${SRC_DIR}/control_flow/control_subgraph_creator.cc
        ${SRC_DIR}/extendrt/delegate/ascend_acl/ascend_allocator_plugin.cc
)

if(MSLITE_ENABLE_CUSTOM_KERNEL)
    file(GLOB KERNEL_REG_SRC ${SRC_DIR}/registry/*.cc)
    set(LITE_SRC ${LITE_SRC} ${KERNEL_REG_SRC})
endif()

if(MSLITE_ENABLE_MODEL_PRE_INFERENCE)
    set(LITE_SRC
            ${LITE_SRC}
            ${SRC_DIR}/common/thread_utils.cc
            ${SRC_DIR}/common/random_data_generator.cc
            )
endif()

set(MODEL_LOADER_FRAMEWORK_SRC
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${SRC_DIR}/extendrt/mindir_loader/model_loader.cc
)

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_FUSION_INFERENCE)
    add_compile_definitions(ENABLE_CLOUD_INFERENCE)
    add_compile_definitions(SUPPORT_TRAIN)
    set(SUPPORT_TRAIN on)

    set(MINDIR_MODEL_SRC
            ${MINDIR_MODEL_SRC}
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model.cc
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_util.cc
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_convertor.cc
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/mindir_model_loader.cc
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/kernel_mod_util.cc
            )

    set(MINDIR_KERNEL_SRC
            ${MINDIR_KERNEL_SRC}
            ${SRC_DIR}/extendrt/mindir_loader/mindir_model/inner_kernel.cc
            ${SRC_DIR}/extendrt/utils/tensor_utils.cc
            )
endif()

set(LITE_SRC
        ${LITE_SRC}
        ${MODEL_LOADER_FRAMEWORK_SRC}
        ${MINDIR_MODEL_SRC}
        ${MINDIR_KERNEL_SRC}
        )


if(MSLITE_ENABLE_BFC_MEMORY)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/extendrt/dynamic_mem_allocator.cc
        ${SRC_DIR}/extendrt/dynamic_mem_manager.cc
        ${SRC_DIR}/extendrt/numa_adapter.cc
        )
endif()

if(MSLITE_ENABLE_SHARING_MODEL_WEIGHT)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/litert/pack_weight.cc
        )
endif()

if(MSLITE_ENABLE_DYNAMIC_THREAD_DISTRIBUTE)
    set(LITE_SRC
        ${LITE_SRC}
        ${SRC_DIR}/litert/thread_cost_model.cc
        )
endif()

if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB CONTROL_FLOW_KERNEL_SRC
            ${SRC_DIR}/control_flow/kernel/*.cc
            )
    set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_KERNEL_SRC})
endif()

if(MSLITE_ENABLE_MINDRT)
    add_compile_definitions(ENABLE_MINDRT)
    include_directories(${CORE_DIR}/mindrt)
    include_directories(${CORE_DIR}/mindrt/include)

    set(LITE_SRC
            ${LITE_SRC}
            ${MINDRT_SRC}
            ${SRC_DIR}/litert/lite_mindrt.cc
            ${SRC_DIR}/litert/parallel_lite_actor.cc
            ${SRC_DIR}/litert/mindrt_executor.cc
            ${SRC_DIR}/control_flow/control_actor_creator.cc
            )

    if(MSLITE_ENABLE_CONTROLFLOW)
        file(GLOB CONTROL_FLOW_ACTOR_SRC
                ${SRC_DIR}/control_flow/actor/*.cc
                )
        set(LITE_SRC ${LITE_SRC} ${CONTROL_FLOW_ACTOR_SRC})
    endif()
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    file(GLOB_RECURSE OPENCL_RUNTIME_SRC
            ${SRC_DIR}/litert/kernel/gpu/opencl/*.cc
            )
    set(LITE_SRC
            ${LITE_SRC}
            ${OPENCL_RUNTIME_SRC}
            )
endif()

include(${TOP_DIR}/cmake/utils.cmake)
__download_pkg(onnx https://gitee.com/mirrors/ONNX/repository/archive/v1.6.0
    8470ee37fa44510c0ffca591bd17a0b4160ae23f5989014030456778fcb740a1)
__download_pkg(tensorflow https://gitee.com/mirrors/tensorflow/repository/archive/v2.4.1
    bc32b7046e1ad83158d99d890aafe384ef94b1d03a2c0a86859c72e264a9d885)
__download_pkg(caffe https://gitee.com/mirrors/caffe/repository/archive/1.0
    17ae8ce8f1b4ba426e500c30efa93bc279209bea99c6ceafde49d2746bf3ef93)

message("tf package path:${tensorflow_SOURCE_DIR}")
message("onnx package path:${onnx_SOURCE_DIR}")
message("caffe package path:${caffe_SOURCE_DIR}")

file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/attr_value.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/function.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/graph.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/node_def.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/op_def.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/resource_handle.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/tensor.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/tensor_shape.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/types.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/core/framework/versions.proto
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow)
file(COPY ${tensorflow_SOURCE_DIR}/tensorflow/lite/schema/schema.fbs
    DESTINATION ${TOP_DIR}/third_party/proto/tensorflow/lite)
file(COPY ${onnx_SOURCE_DIR}/onnx/onnx.proto DESTINATION ${TOP_DIR}/third_party/proto/onnx)
file(COPY ${caffe_SOURCE_DIR}/src/caffe/proto/caffe.proto DESTINATION ${TOP_DIR}/third_party/proto/caffe)

set(TENSORFLOW_PATCH_FILE ${TOP_DIR}/third_party/patch/tensorflow/tensorflow.patch)
set(CAFFE_PATCH_FILE ${TOP_DIR}/third_party/patch/caffe/caffe.patch)

execute_process(COMMAND ${Patch_EXECUTABLE} -Np1 -i ${TENSORFLOW_PATCH_FILE} -d ${TOP_DIR}
        RESULT_VARIABLE Result)
if(NOT Result EQUAL "0")
    message(FATAL_ERROR "Failed patch: ${_LF_PATCH_FILE}")
endif()

execute_process(COMMAND ${Patch_EXECUTABLE} -Np1 -i ${CAFFE_PATCH_FILE} -d ${TOP_DIR}
        RESULT_VARIABLE Result)
if(NOT Result EQUAL "0")
    message(FATAL_ERROR "Failed patch: ${_LF_PATCH_FILE}")
endif()

file(GLOB PROTO_FILE ""
        ${TOP_DIR}/third_party/proto/tensorflow/*.proto
        ${TOP_DIR}/third_party/proto/onnx/onnx.proto)
file(GLOB CAFFE_PROTO_FILE "" ${TOP_DIR}/third_party/proto/caffe/caffe.proto)
if(MSLITE_ENABLE_ACL AND MSLITE_ENABLE_ACL_QUANT_PARAM)
    set(PROTO_PATCH_FILE ${TOP_DIR}/third_party/patch/caffe/caffe_proto_acl_quant_param.patch001)
    message("patching caffe.proto < ${PROTO_PATCH_FILE}")
    execute_process(COMMAND ${Patch_EXECUTABLE} -Np1 -i ${PROTO_PATCH_FILE} -d ${TOP_DIR}
            -o ${TOP_DIR}/build/caffe.proto
            RESULT_VARIABLE Result)
    if(NOT Result EQUAL "0")
        message(FATAL_ERROR "Failed patch: ${_LF_PATCH_FILE}")
    endif()
    file(GLOB CAFFE_PROTO_FILE "" ${TOP_DIR}/build/caffe.proto)
endif()
set(PROTO_FILE ${PROTO_FILE} ${CAFFE_PROTO_FILE})

ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_library(proto_mid OBJECT ${PROTO_SRCS})
target_compile_options(proto_mid PRIVATE "-Wno-array-bounds")
target_compile_options(core_proto_obj PRIVATE "-Wno-array-bounds")

set(TFLITE_FBS_FILES
        ${TOP_DIR}/third_party/proto/tensorflow/lite/schema.fbs
        )
ms_build_flatbuffers_lite(TFLITE_FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/parser/tflite/ tflite_fbs_src
        ${CMAKE_BINARY_DIR}/schema "inner")

set_property(SOURCE ${CONVERTER_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
set_property(SOURCE ${LITE_SRC} PROPERTY COMPILE_DEFINITIONS
        LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
        SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

add_library(converter_runtime_mid OBJECT ${LITE_SRC})
add_dependencies(converter_runtime_mid fbs_src fbs_inner_src)
target_compile_options(converter_runtime_mid PRIVATE "-Wno-stringop-overflow")
add_library(mindspore_converter SHARED $<TARGET_OBJECTS:converter_runtime_mid>)

add_library(converter_src_mid OBJECT ${CONVERTER_SRC})
add_dependencies(converter_src_mid fbs_src fbs_inner_src)


target_compile_definitions(converter_src_mid PRIVATE BACKEND_DLL)
target_compile_definitions(converter_src_mid PRIVATE BACKEND_COMMON_DLL)
target_compile_definitions(quantizer_mid PRIVATE BACKEND_DLL)

add_dependencies(mindspore_converter nnacl_mid)

if((NOT WIN32) AND MSLITE_ENABLE_DPICO_ATC_ADAPTER)
    add_dependencies(mindspore_converter dpico_atc_adapter)
endif()

if(MSLITE_GPU_BACKEND STREQUAL opencl)
    include_directories(${SRC_DIR}/litert/kernel/opencl)
    target_link_libraries(mindspore_converter opencl_kernel_mid)
endif()

if(MSLITE_ENABLE_FP16)
    target_link_libraries(mindspore_converter
        nnacl_fp16_mid
        )
endif()

target_link_libraries(mindspore_converter
        converter_src_mid
        cpu_ops_mid
        nnacl_mid
        cpu_kernel_mid
        cast_for_converter_mid
        tflite_parser_mid
        tf_parser_mid
        caffe_parser_mid
        onnx_parser_mid
        om_parser_mid
        lite_exporter_mid
        graph_pass_mid
        fusion_mid
        quantizer_mid
        proto_mid
        ${SECUREC_LIBRARY}
        mindspore::json
        mindspore_core
        mindspore_ops
        mslite_converter_plugin
        mindspore::glog
        mindspore::protobuf
        preprocess_mid
        config_parser_mid
        coder_mid
        mindir_serializer_mid
        )
if(PYTHON_VERSION MATCHES "3.11")
  target_link_libraries(mindspore_converter ${PYTHON_LIBRARIES})
endif()

if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
    target_link_libraries(mindspore_converter decomposer_mid)
endif()

if(SUPPORT_TRAIN)
    target_link_libraries(mindspore_converter train_cpu_kernel_mid)
endif()

if(ENABLE_CONVERT_PYTORCH_MODEL)
    target_link_libraries(mindspore_converter pytorch_parser_mid)
endif()

target_link_libraries(mindspore_converter
        ccsrc_debug_common_mid_
        mindir_proto_mid
        _mindspore_transform_express_ir_obj)

if(NOT MSVC)
    target_link_libraries(mindspore_converter pthread)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_converter dl)
endif()

file(STRINGS "${TOP_DIR}/version.txt" VERSION)
add_definitions(-DVERSION=\"${VERSION}\")
set(CCSRC_SRC_NEW ${CCSRC_DIR}/backend/common/optimizer/graph_optimizer.cc
    ${CCSRC_DIR}/backend/common/optimizer/pattern_engine.cc
    ${CCSRC_DIR}/utils/convert_utils.cc
    ${CCSRC_DIR}/common/debug/mindir_exporter.cc
    ${CCSRC_DIR}/backend/common/optimizer/visitor.cc
    ${CCSRC_DIR}/common/debug/common.cc
    ${CCSRC_DIR}/utils/compile_cache_context.cc
    )
if(NOT WIN32)
    set(CCSRC_SRC_NEW ${CCSRC_SRC_NEW}
        ${CCSRC_DIR}/utils/anfalgo.cc
        ${CCSRC_DIR}/utils/parallel_context.cc
        ${CCSRC_DIR}/utils/utils.cc)
endif()
if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    set(CCSRC_SRC_NEW ${CCSRC_SRC_NEW}
    ${OPS_DIR}/kernel/common/kernel_factory.cc
    ${OPS_DIR}/kernel/common/kernel_tensor.cc
    ${OPS_DIR}/kernel/common/device_address.cc
    ${OPS_DIR}/kernel/common/format_utils.cc
    ${OPS_DIR}/kernel/common/kernel.cc)
endif()
add_library(ccsrc_src_new_mid OBJECT ${CCSRC_SRC_NEW})
target_compile_definitions(ccsrc_src_new_mid PRIVATE BACKEND_DLL)
target_compile_definitions(ccsrc_src_new_mid PRIVATE COMMON_DLL)
target_compile_definitions(ccsrc_src_new_mid PRIVATE OPS_KERNEL_COMMON_DLL)
target_link_libraries(mindspore_converter ccsrc_src_new_mid)
add_subdirectory(converter_lite)
