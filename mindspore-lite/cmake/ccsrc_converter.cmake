# Compile ccsrc files in converter independently
if(MSLITE_ENABLE_CONVERTER)
    add_definitions(-DPRIMITIVE_WRITEABLE)
    add_definitions(-DUSE_GLOG)
    set(USE_GLOG on)
    if(MSLITE_ENABLE_MODEL_ENCRYPTION AND MSLITE_DEPS_OPENSSL)
        add_compile_definitions(ENABLE_OPENSSL)
    endif()

    if(ENABLE_GPU)
        add_compile_definitions(ENABLE_GPU)
    endif()

    set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
    set(TOOLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
    set(CCSRC_SRC
            ${CCSRC_DIR}/backend/backend_manager/backend_jit_config.cc
            ${CCSRC_DIR}/backend/common/optimizer/pattern_engine.cc
            ${CCSRC_DIR}/backend/common/optimizer/visitor.cc
            ${CCSRC_DIR}/backend/common/optimizer/graph_optimizer.cc
            ${CCSRC_DIR}/backend/operator/ops_backend_infer_function.cc
            ${OPS_DIR}/kernel/common/kernel.cc
            ${OPS_DIR}/kernel/common/kernel_tensor.cc
            ${OPS_DIR}/kernel/common/kernel_factory.cc
            ${OPS_DIR}/kernel/common/format_utils.cc
            ${CCSRC_DIR}/utils/convert_utils.cc
            )

    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        set(CCSRC_SRC ${CCSRC_SRC}
                ${CCSRC_DIR}/ps/ps_context.cc
                ${CCSRC_DIR}/common/thread_pool.cc
                ${CCSRC_DIR}/debug/profiler/profiler.cc
                ${CCSRC_DIR}/common/pynative/abstract_converter.cc
                ${CCSRC_DIR}/plugin/device/cpu/kernel/cpu_kernel.cc
                ${CCSRC_DIR}/distributed/cluster/dummy_cluster_context.cc
                ${OPS_DIR}/kernel/common/kernel_utils.cc
                ${OPS_DIR}/kernel/common/common_utils.cc
                ${CCSRC_DIR}/kernel/framework_utils.cc
                ${CCSRC_DIR}/kernel/philox_random.cc
                ${CCSRC_DIR}/kernel/kash/kernel_pack.cc
                ${OPS_DIR}/kernel/common/kernel_build_info.cc
                ${OPS_DIR}/kernel/common/oplib/oplib.cc
                ${CCSRC_DIR}/kernel/kernel_info.cc
                ${CCSRC_DIR}/runtime/device/res_manager/utils/convert_tensor_utils.cc
                ${CCSRC_DIR}/utils/ms_device_shape_transfer.cc
                ${CCSRC_DIR}/runtime/device/kernel_runtime_manager.cc
                ${CCSRC_DIR}/runtime/hardware/device_context_manager.cc
                ${CCSRC_DIR}/common/runtime_conf/runtime_conf.cc
                ${CCSRC_DIR}/utils/comm_manager.cc
                ${CCSRC_DIR}/backend/common/session/exec_order_builder.cc
                ${CCSRC_DIR}/backend/common/session/kernel_graph.cc
                ${CCSRC_DIR}/backend/common/session/anf_runtime_algorithm.cc
                ${CCSRC_DIR}/runtime/device/res_manager/hal_res_manager.cc
                ${CCSRC_DIR}/runtime/device/res_manager/multi_stream_controller.cc
                ${SRC_DIR}/extendrt/utils/tensor_utils.cc
                )
    endif()

    if(NOT WIN32)
        set(CCSRC_SRC ${CCSRC_SRC}
                ${CCSRC_DIR}/utils/anfalgo.cc
                ${CCSRC_DIR}/utils/utils.cc
                ${CCSRC_DIR}/utils/parallel_context.cc
                )
    endif()

    if(ENABLE_GPU)
        add_compile_definitions(ENABLE_GPU)
    endif()

    if(MSLITE_ENABLE_GRAPH_KERNEL)

        if(AKG_USE_LLVM)
            add_compile_definitions(AKG_USE_LLVM)
            message(STATUS "Converter support Graph Kernel CPU backend")
        endif()

        if(AKG_ENABLE_D)
            add_compile_definitions(AKG_ENABLE_D)
            message(STATUS "Converter support Graph Kernel Ascend backend")
        endif()

        if(AKG_USE_CUDA)
            add_compile_definitions(AKG_USE_CUDA)
            message(STATUS "Converter support Graph Kernel CUDA backend")
        endif()

        add_compile_definitions(MSLITE_ENABLE_GRAPH_KERNEL)
        file(GLOB_RECURSE GRAPH_KERNEL_SRC
                ${TOOLS_DIR}/graph_kernel/common/*.cc
                ${TOOLS_DIR}/graph_kernel/converter/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/core/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/expander/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/expanders/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/model/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/split_model/*.cc
                ${CCSRC_DIR}/backend/common/graph_kernel/graph_kernel_flags.cc
                ${CCSRC_DIR}/kernel/graph_kernel/graph_kernel_json_generator.cc
                ${CCSRC_DIR}/backend/common/optimizer/optimizer.cc
                )
        set_property(SOURCE ${GRAPH_KERNEL_SRC}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_GRAPH_KERNEL)
        set(CCSRC_SRC
                ${CCSRC_SRC}
                ${GRAPH_KERNEL_SRC}
                )
    endif()
    set_property(SOURCE ${CCSRC_SRC} PROPERTY COMPILE_DEFINITIONS
            LOG_HDR_FILE_REL_PATH="mindspore-lite/../mindspore/mindspore/core/include/utils/log_adapter.h"
            SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)
    add_library(ccsrc_src_mid OBJECT ${CCSRC_SRC})
    add_dependencies(ccsrc_src_mid fbs_src fbs_inner_src)
    if(MSLITE_ENABLE_CLOUD_INFERENCE)
        add_dependencies(ccsrc_src_mid mindspore-lite-proto)
    endif()
    target_compile_definitions(ccsrc_src_mid PRIVATE BACKEND_DLL)
    target_compile_definitions(ccsrc_src_mid PRIVATE COMMON_DLL)
    target_compile_definitions(ccsrc_src_mid PRIVATE OPS_KERNEL_COMMON_DLL)
endif()
