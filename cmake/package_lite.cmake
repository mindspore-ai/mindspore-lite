include(CMakePackageConfigHelpers)

set(RUNTIME_PKG_NAME ${PKG_NAME_PREFIX}-${RUNTIME_COMPONENT_NAME})

set(CONVERTER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/converter)
set(OBFUSCATOR_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/obfuscator)
set(CROPPER_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/cropper)
set(BUILD_DIR ${TOP_DIR}/build)
set(TEST_CASE_DIR ${TOP_DIR}/mindspore-lite/test/build)
set(EXTENDRT_BUILD_DIR ${BUILD_DIR}/src/extendrt)
set(EXECUTOR_BUILD_DIR ${BUILD_DIR}/src/extendrt/unified_executor)

set(RUNTIME_DIR ${RUNTIME_PKG_NAME}/runtime)
set(RUNTIME_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include)
set(RUNTIME_LIB_DIR ${RUNTIME_PKG_NAME}/runtime/lib)
set(PROVIDERS_LIB_DIR ${RUNTIME_PKG_NAME}/providers)
set(MIND_DATA_INC_DIR ${RUNTIME_PKG_NAME}/runtime/include/dataset)
set(TURBO_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/libjpeg-turbo)
set(GLOG_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/glog)
set(DNNL_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/dnnl)
set(SECUREC_DIR ${RUNTIME_PKG_NAME}/runtime/third_party/securec)
set(MINDSPORE_LITE_LIB_NAME libmindspore-lite)
set(MINDSPORE_LITE_EXTENDRT_LIB_NAME libmindspore-lite)
set(MINDSPORE_CORE_LIB_NAME libmindspore_core)
set(MINDSPORE_OPS_LIB_NAME libmindspore_ops)
set(MINDSPORE_GE_LITERT_LIB_NAME libmsplugin-ge-litert)
set(MINDSPORE_LITE_ASCEND_NATIVE_PLUGIN libascend_native_plugin)
set(MINDSPORE_LITE_EXECUTOR_LIB_NAME liblite-unified-executor)
set(BENCHMARK_NAME benchmark)
set(MSLITE_NNIE_LIB_NAME libmslite_nnie)
set(MSLITE_PROPOSAL_LIB_NAME libmslite_proposal)
set(MICRO_NNIE_LIB_NAME libmicro_nnie)
set(DPICO_ACL_ADAPTER_LIB_NAME libdpico_acl_adapter)
set(BENCHMARK_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark)

set(MINDSPORE_LITE_TRAIN_LIB_NAME libmindspore-lite-train)
set(BENCHMARK_TRAIN_NAME benchmark_train)
set(BENCHMARK_TRAIN_ROOT_DIR ${RUNTIME_PKG_NAME}/tools/benchmark_train)
file(GLOB JPEGTURBO_LIB_LIST ${jpeg_turbo_LIBPATH}/*.so*)

include(${TOP_DIR}/cmake/package_micro.cmake)

function(__install_white_list_ops)
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/ops/base_operator.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/ops
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/op_def/nn_op_name.h
            ${TOP_DIR}/mindspore/mindspore/ops/op_def/op_name.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/op_def
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/ops_utils/op_constants.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/ops_utils
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/infer/adam.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/all.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/apply_momentum.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/assert.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/audio_spectrogram.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/batch_to_space.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/batch_to_space_nd.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/broadcast.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/clip.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/attention.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/constant_of_shape.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/crop.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/custom_extract_features.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/custom_normalize.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/custom_predict.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/depend.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/depth_to_space.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/detection_post_process.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/eltwise.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fake_quant_with_min_max_vars.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fake_quant_with_min_max_vars_per_channel.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fake_quant_with_min_max_vars.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fft_real.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fft_imag.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fill.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/fused_batch_norm.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/hashtable_lookup.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/instance_norm.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/leaky_relu.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/lp_normalization.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/lrn.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/lsh_projection.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/lstm.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/switch_layer.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/mfcc.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/mod.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/non_max_suppression.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/prior_box.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/quant_dtype_cast.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/resize.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/reverse_sequence.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/rfft.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/roi_pooling.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/sgd.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/sigmoid_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/skip_gram.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/softmax_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/space_to_batch.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/space_to_batch_nd.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/space_to_depth.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/sparse_softmax_cross_entropy_with_logits.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/sparse_to_dense.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/squared_difference.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/stack.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/switch.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_list_from_tensor.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_list_get_item.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_list_reserve.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_list_set_item.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_list_stack.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/unique.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/unsqueeze.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/unstack.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/where.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/scatter_nd_update.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/gru.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/invert_permutation.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/size.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/random_standard_normal.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/crop_and_resize.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/uniform_real.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/splice.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/call.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/custom.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/split_with_overlap.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/ragged_range.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_array.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_array_read.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tensor_array_write.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/affine.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/all_gather.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/reduce_scatter.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/dynamic_quant.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/random_normal.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tuple_get_item.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/tuple_get_item.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/scale.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/sub.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/conv2d_transpose.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/conv2d.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/topk.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/reduce.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/max_pool.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/make_tuple.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/return.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/pad.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/infer
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/activation.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/add_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/adder_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/arg_max_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/arg_min_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/avg_pool_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/conv2d_backprop_filter_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/conv2d_backprop_input_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/conv2d_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/conv2d_transpose_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/div_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/embedding_lookup_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/exp_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/full_connection.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/layer_norm_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/l2_normalize_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/mat_mul_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/max_pool_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/mul_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/pad_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/partial_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/pow_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/prelu_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/reduce_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/scale_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/slice_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/sub_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/tile_fusion.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/cxx_api/topk_fusion.h
            DESTINATION ${CONVERTER_ROOT_DIR}/include/infer/cxx_api
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
    file(GLOB GEN_OPS_NAME_H ${TOP_DIR}/mindspore/mindspore/ops/op_def/auto_generate/gen_ops_name_*.h)
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/op_def/auto_generate/gen_lite_ops.h
            ${GEN_OPS_NAME_H}
            DESTINATION ${CONVERTER_ROOT_DIR}/include/op_def/auto_generate
            COMPONENT ${RUNTIME_COMPONENT_NAME}
            )
endfunction()

function(__install_ascend_tbe_and_aicpu)
    set(TBE_CUSTOM_OPP_DIR ${BUILD_DIR}/tools/kernel_builder/ascend/tbe_and_aicpu/makepkg/packages)
    set(TBE_OPP_DST_DIR ${RUNTIME_PKG_NAME}/tools/custom_kernels/ascend/tbe_and_aicpu)
    install(DIRECTORY ${TBE_CUSTOM_OPP_DIR} DESTINATION ${TBE_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TBE_CUSTOM_OPP_DIR}/../install.sh DESTINATION
                  ${TBE_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TBE_CUSTOM_OPP_DIR}/../set_env.bash DESTINATION
                  ${TBE_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
endfunction()

function(__install_ascend_ascendc)
    set(ASCEMDC_CUSTOM_OPP_DIR ${BUILD_DIR}/tools/kernel_builder/ascend/ascendc/makepkg/packages)
    set(ASCENDC_OPP_DST_DIR ${RUNTIME_PKG_NAME}/tools/custom_kernels/ascend/ascendc)
    install(DIRECTORY ${ASCEMDC_CUSTOM_OPP_DIR} DESTINATION ${ASCENDC_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${CMAKE_BINARY_DIR}/ascendc_scripts/install.sh DESTINATION
            ${ASCENDC_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${CMAKE_BINARY_DIR}/ascendc_scripts/set_env.bash DESTINATION
            ${ASCENDC_OPP_DST_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
endfunction()

# full mode will also package the files of lite_cv mode.
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full"
        AND NOT(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    # full header files
    install(FILES
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/constants.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/data_helper.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/execute.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/iterator.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/samplers.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/transforms.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/vision_lite.h
            ${TOP_DIR}/mindspore-lite/minddata/dataset/liteapi/include/datasets.h
            DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})

    if(PLATFORM_ARM64)
        if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/vision_ascend.h
                    DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/minddata/kernels-dvpp-image/utils/libdvpp_utils.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${securec_LIBPATH}/libsecurec.a DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${securec_LIBPATH}/libsecurec.a DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
            install(FILES ${TOP_DIR}/mindspore-lite/minddata/dataset/include/dataset/vision_ascend.h
                    DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/minddata/kernels-dvpp-image/utils/libdvpp_utils.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.4.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.3.0 DESTINATION ${TURBO_DIR}/lib
                RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${securec_LIBPATH}/libsecurec.a DESTINATION ${SECUREC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()

    # lite_cv header files
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/kernels/image/lite_cv
            DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "wrapper"
        AND NOT(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "vision.h" EXCLUDE)
    if(PLATFORM_ARM64)
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TURBO_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libjpeg.so.62.4.0 DESTINATION ${TURBO_DIR}/lib RENAME libjpeg.so.62
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${jpeg_turbo_LIBPATH}/libturbojpeg.so.0.3.0 DESTINATION ${TURBO_DIR}/lib RENAME libturbojpeg.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite"
        AND NOT(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/include/ DESTINATION ${MIND_DATA_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(PLATFORM_ARM64)
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so DESTINATION ${TURBO_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libjpeg.so.62.4.0
                DESTINATION ${TURBO_DIR}/lib RENAME libjpeg.so.62 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/third_party/libjpeg-turbo/lib/libturbojpeg.so.0.3.0
                DESTINATION ${TURBO_DIR}/lib RENAME libturbojpeg.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite_cv" AND
        NOT(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    if(PLATFORM_ARM64)
        install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    elseif(PLATFORM_ARM32)
        install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(DIRECTORY ${TOP_DIR}/mindspore-lite/minddata/dataset/kernels/image/lite_cv
                DESTINATION ${MIND_DATA_INC_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(FILES ${BUILD_DIR}/minddata/libminddata-lite.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
endif()

if(WIN32)
    install(FILES ${TOP_DIR}/build/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
else()
    install(FILES ${BUILD_DIR}/.commit_id DESTINATION ${RUNTIME_PKG_NAME}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()
if(NOT PLATFORM_MCU)
    install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${RUNTIME_INC_DIR}/third_party
            COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()

if(ANDROID_NDK)
    set(glog_name libmindspore_glog.so)
else()
    set(glog_name libmindspore_glog.so.0.4.0)
endif()

if(PLATFORM_ARM64)
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
            install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                    DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    install(FILES ${TOP_DIR}/mindspore-lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${BUILD_DIR}/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${BUILD_DIR}/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
            install(FILES ${EXECUTOR_BUILD_DIR}/${MINDSPORE_LITE_EXECUTOR_LIB_NAME}.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core mindspore_ops DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
                install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            __install_ascend_tbe_and_aicpu()
            __install_ascend_ascendc()
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore-lite/tools/obfuscator/lib/android-aarch64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/mindspore/core/include/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/format.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/mindspore/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER OR TARGET_HIMIX)
        __install_micro_wrapper()
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(TARGET_HIMIX)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3559A")
                    install(FILES ${BUILD_DIR}/tools/benchmark/nnie/${MSLITE_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${BUILD_DIR}/tools/benchmark/nnie_proposal/${MSLITE_PROPOSAL_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${BUILD_DIR}/nnie_micro/${MICRO_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            elseif(TARGET_MIX210)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "SD3403" AND (NOT MSLITE_ENABLE_ACL))
                    install(FILES ${BUILD_DIR}/tools/benchmark/dpico/${DPICO_ACL_ADAPTER_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_ENABLE_CONVERTER)
            install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION
                    ${CONVERTER_ROOT_DIR}/include/registry COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/base
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/ir
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            __install_white_list_ops()
            install(DIRECTORY ${BUILD_DIR}/schema/
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                    COMPONENT ${RUNTIME_COMPONENT_NAME}
                    FILES_MATCHING PATTERN "*.h" PATTERN "schema_generated.h" EXCLUDE)
            install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${glog_LIBPATH}/../include/glog/
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(DIRECTORY ${securec_INC}
                    DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/converter/libmindspore_converter.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/converter/registry/libmslite_converter_plugin.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(DIRECTORY ${TOP_DIR}/third_party/proto/ DESTINATION ${CONVERTER_ROOT_DIR}/third_party/proto
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${CONVERTER_ROOT_DIR}/lib
                    RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(TARGETS mindspore_core mindspore_ops DESTINATION ${CONVERTER_ROOT_DIR}/lib
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(MSLITE_ENABLE_OPENCV)
                install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                        DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_core.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                        DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgcodecs.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                        DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgproc.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            if(MSLITE_ENABLE_ACL)
                set(LITE_ACL_DIR ${BUILD_DIR}/tools/converter/adapter/acl)
                if(MSLITE_ENABLE_RUNTIME_CONVERT)
                    install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
                            RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(TARGETS mindspore_core mindspore_ops DESTINATION ${CONVERTER_ROOT_DIR}/lib
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
                install(FILES ${LITE_ACL_DIR}/libascend_pass_plugin.so DESTINATION ${CONVERTER_ROOT_DIR}/lib
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${LITE_ACL_DIR}/_mindspore_transform_graph_ir/libmindspore_graph_ir.so DESTINATION
                        ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${LITE_ACL_DIR}/_mindspore_transform_graph_ir/libmindspore_graph_ir.so DESTINATION
                        ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()

            if(MSLITE_ENABLE_DPICO_ATC_ADAPTER)
                install(FILES ${BUILD_DIR}/tools/converter/adapter/dpico/libdpico_atc_adapter.so
                        DESTINATION ${CONVERTER_ROOT_DIR}/providers/SD3403 COMPONENT ${RUNTIME_COMPONENT_NAME})
                if(MSLITE_ENABLE_TOOLS)
                    install(TARGETS ${BECHCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()

            if(MSLITE_ENABLE_RUNTIME_GLOG)
                install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                        COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
                install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${GLOG_DIR}
                        RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            if(MSLITE_ENABLE_RUNTIME_CONVERT)
                install(FILES ${BUILD_DIR}/tools/converter/registry/libmslite_converter_plugin.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                if(MSLITE_ENABLE_OPENCV)
                    install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_core.so.4.5
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                            DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgproc.so.4.5
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
            if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
                AND MSLITE_ENABLE_GRAPH_KERNEL AND CMAKE_SYSTEM_NAME MATCHES "Linux")
                if(EXISTS ${BUILD_DIR}/akg)
                    set(AKG_PATH ${BUILD_DIR}/akg)
                    file(REMOVE_RECURSE ${AKG_PATH}/build/akg/lib)
                    install(DIRECTORY  ${AKG_PATH}/build/akg
                            DESTINATION ${BUILD_DIR}/package/mindspore_lite
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${AKG_PATH}/${AKG_PKG_PATH}
                            DESTINATION ${RUNTIME_PKG_NAME}/tools/akg
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${AKG_PATH}/${AKG_PKG_PATH}.sha256
                            DESTINATION ${RUNTIME_PKG_NAME}/tools/akg
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES ${AKG_PATH}/build/libakg.so
                            DESTINATION ${BUILD_DIR}/package/mindspore_lite/lib
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
        endif()
    endif()
    if(MSLITE_ENABLE_TESTCASES)
        install(FILES ${BUILD_DIR}/test/lite-test DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${BUILD_DIR}/src/ DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.so")
        install(DIRECTORY ${BUILD_DIR}/minddata/ DESTINATION ${TEST_CASE_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.so")
        install(FILES ${JPEGTURBO_LIB_LIST} DESTINATION ${TEST_CASE_DIR})
        if(SUPPORT_NPU)
            install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${TEST_CASE_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
                install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                        DESTINATION ${TEST_CASE_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
    endif()
elseif(PLATFORM_ARM32)
    if(SUPPORT_NPU)
        install(FILES ${DDK_LIB_PATH}/libhiai.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${DDK_LIB_PATH}/libhiai_ir_build.so DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(EXISTS "${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so")
            install(FILES ${DDK_LIB_PATH}/libhiai_hcl_model_runtime.so
                    DESTINATION ${RUNTIME_DIR}/third_party/hiai_ddk/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    install(FILES ${TOP_DIR}/mindspore-lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${BUILD_DIR}/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${BUILD_DIR}/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
            install(FILES ${EXECUTOR_BUILD_DIR}/${MINDSPORE_LITE_EXECUTOR_LIB_NAME}.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core mindspore_ops DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
                install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            __install_ascend_tbe_and_aicpu()
            __install_ascend_ascendc()
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore-lite/tools/obfuscator/lib/android-aarch32/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore/mindspore/core/include/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/format.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/mindspore/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED OR MSLITE_ENABLE_CONVERTER OR TARGET_OHOS_LITE OR TARGET_HIMIX)
        __install_micro_wrapper()
    endif()
    if(MSLITE_ENABLE_TOOLS AND NOT TARGET_OHOS_LITE)
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME
                    DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(TARGET_HIMIX)
                if(${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3516D" OR ${MSLITE_REGISTRY_DEVICE}  STREQUAL "Hi3519A")
                    install(FILES ${BUILD_DIR}/tools/benchmark/nnie/${MSLITE_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${BUILD_DIR}/tools/benchmark/nnie_proposal/${MSLITE_PROPOSAL_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                    install(FILES
                            ${BUILD_DIR}/nnie_micro/${MICRO_NNIE_LIB_NAME}.so
                            DESTINATION ${PROVIDERS_LIB_DIR}/${MSLITE_REGISTRY_DEVICE}
                            COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
            endif()
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
elseif(WIN32)
    get_filename_component(CXX_DIR ${CMAKE_CXX_COMPILER} PATH)
    file(GLOB LIB_LIST ${CXX_DIR}/libstdc++-6.dll ${CXX_DIR}/libwinpthread-1.dll
            ${CXX_DIR}/libssp-0.dll ${CXX_DIR}/libgcc_s_*-1.dll)
    if(MSLITE_ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/converter_lite/converter_lite.exe
                DESTINATION ${CONVERTER_ROOT_DIR}/converter COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/libmindspore_converter.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/tools/converter/registry/libmslite_converter_plugin.dll
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        file(GLOB GLOG_LIB_LIST ${glog_LIBPATH}/../bin/*.dll)
        install(FILES ${GLOG_LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core mindspore_ops DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_OPENCV)
            file(GLOB_RECURSE OPENCV_LIB_LIST
                    ${opencv_LIBPATH}/../bin/libopencv_core*
                    ${opencv_LIBPATH}/../bin/libopencv_imgcodecs*
                    ${opencv_LIBPATH}/../bin/libopencv_imgproc*
                    )
            install(FILES ${OPENCV_LIB_LIST} DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(NOT MSVC AND NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            __install_micro_wrapper()
            __install_micro_codegen()
        endif()
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(MSVC)
            install(FILES ${TOP_DIR}/build/mindspore/tools/benchmark/${BENCHMARK_NAME}.exe
                    DESTINATION ${BENCHMARK_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        else()
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        file(GLOB GLOG_LIB_LIST ${glog_LIBPATH}/../bin/*.dll)
        install(FILES ${GLOG_LIB_LIST} DESTINATION ${GLOG_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${TOP_DIR}/mindspore-lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    install(FILES ${TOP_DIR}/build/mindspore/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/build/mindspore/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/mindspore/core/include/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/format.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/mindspore/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(MSVC)
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.lib DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll.lib DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
    else()
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/build/mindspore/src/${MINDSPORE_LITE_LIB_NAME}.dll DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${LIB_LIST} DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
elseif(PLATFORM_MCU)
    __install_micro_wrapper()
    __install_micro_codegen()
    install(DIRECTORY ${TOP_DIR}/mindspore/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
else()
    install(FILES ${TOP_DIR}/mindspore-lite/include/kernel_interface.h DESTINATION ${RUNTIME_INC_DIR}
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION ${RUNTIME_INC_DIR}/registry
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "register_kernel_interface.h"
            PATTERN "register_kernel.h")
    if(SUPPORT_TRAIN)
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.so DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_TRAIN_LIB_NAME}.a DESTINATION
                ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    install(FILES ${BUILD_DIR}/schema/model_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${BUILD_DIR}/schema/ops_types_generated.h DESTINATION ${RUNTIME_INC_DIR}/schema
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore/mindspore/core/include/ir/dtype/type_id.h DESTINATION ${RUNTIME_INC_DIR}/ir/dtype
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/format.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/types.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(DIRECTORY ${TOP_DIR}/mindspore/include/api/ DESTINATION ${RUNTIME_INC_DIR}/api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "ops*" EXCLUDE)
    install(DIRECTORY ${TOP_DIR}/mindspore/include/c_api/ DESTINATION ${RUNTIME_INC_DIR}/c_api
            COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        install(FILES ${BUILD_DIR}/src/extendrt/${MINDSPORE_LITE_EXTENDRT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${EXTENDRT_BUILD_DIR}/delegate/graph_executor/litert/${MINDSPORE_GE_LITERT_LIB_NAME}.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
            install(FILES ${EXECUTOR_BUILD_DIR}/${MINDSPORE_LITE_EXECUTOR_LIB_NAME}.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_DEPS_MKLDNN)
                install(FILES ${onednn_LIBPATH}/libdnnl.so.2.2 DESTINATION ${DNNL_DIR}
                        RENAME libdnnl.so.2 COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        install(TARGETS mindspore_core mindspore_ops DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/extendrt/convert/libruntime_convert_plugin.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
                install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_ge/libascend_ge_plugin.so
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            __install_ascend_tbe_and_aicpu()
            __install_ascend_ascendc()
            if(MSLITE_ASCEND_TARGET)
                install(TARGETS ascend_native_plugin
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                if(TARGET ascend_native_kernels_impl)
                        install(TARGETS ascend_native_kernels_impl
                        DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
                endif()
                install(TARGETS hccl_plugin
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
        if(MSLITE_GPU_BACKEND STREQUAL tensorrt)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/tensorrt/libtensorrt_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    else()
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.so DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/src/${MINDSPORE_LITE_LIB_NAME}.a DESTINATION ${RUNTIME_LIB_DIR}
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_ACL)
            install(FILES ${BUILD_DIR}/src/extendrt/delegate/ascend_acl/libascend_acl_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
    endif()
    if(MSLITE_ENABLE_MODEL_OBF)
        install(FILES ${TOP_DIR}/mindspore-lite/tools/obfuscator/bin/linux-x64/msobfuscator
                DESTINATION ${OBFUSCATOR_ROOT_DIR} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${TOP_DIR}/mindspore-lite/tools/obfuscator/lib/linux-x64/libmsdeobfuscator-lite.so
                DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG)
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
    endif()
    if(MSLITE_ENABLE_CONVERTER)
        install(FILES ${TOP_DIR}/mindspore-lite/include/kernel_interface.h DESTINATION ${CONVERTER_ROOT_DIR}/include
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/mindspore-lite/include/registry/ DESTINATION ${CONVERTER_ROOT_DIR}/include/registry
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${API_HEADER}  DESTINATION ${CONVERTER_ROOT_DIR}/include/api
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_BASE_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/base
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${MINDAPI_IR_HEADER} DESTINATION ${CONVERTER_ROOT_DIR}/include/mindapi/ir
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        __install_white_list_ops()
        install(DIRECTORY ${BUILD_DIR}/schema/ DESTINATION ${CONVERTER_ROOT_DIR}/include/schema
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h" PATTERN "schema_generated.h" EXCLUDE)
        install(DIRECTORY ${flatbuffers_INC}/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${TOP_DIR}/third_party/proto/ DESTINATION ${CONVERTER_ROOT_DIR}/third_party/proto
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/glog
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(DIRECTORY ${securec_INC}
                DESTINATION ${CONVERTER_ROOT_DIR}/include/third_party/securec
                COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
        install(TARGETS converter_lite RUNTIME DESTINATION ${CONVERTER_ROOT_DIR}/converter
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/tools/converter/libmindspore_converter.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${BUILD_DIR}/tools/converter/registry/libmslite_converter_plugin.so
                DESTINATION ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${CONVERTER_ROOT_DIR}/lib
                RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        install(TARGETS mindspore_core mindspore_ops DESTINATION ${CONVERTER_ROOT_DIR}/lib
                COMPONENT ${RUNTIME_COMPONENT_NAME})
        if(MSLITE_ENABLE_OPENCV)
            install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_core.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgcodecs.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                    DESTINATION ${CONVERTER_ROOT_DIR}/lib RENAME libopencv_imgproc.so.4.5
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()

        if(MSLITE_ENABLE_ACL)
            set(LITE_ACL_DIR ${BUILD_DIR}/tools/converter/adapter/acl)
            if(MSLITE_ENABLE_RUNTIME_CONVERT)
                install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
                        RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(TARGETS mindspore_core mindspore_ops DESTINATION ${RUNTIME_LIB_DIR}
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
            install(FILES ${LITE_ACL_DIR}/libascend_pass_plugin.so DESTINATION ${CONVERTER_ROOT_DIR}/lib
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${LITE_ACL_DIR}/_mindspore_transform_graph_ir/libmindspore_graph_ir.so DESTINATION
                    ${CONVERTER_ROOT_DIR}/lib COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${LITE_ACL_DIR}/_mindspore_transform_graph_ir/libmindspore_graph_ir.so DESTINATION
                    ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()

        if(MSLITE_ENABLE_DPICO_ATC_ADAPTER)
            install(FILES ${BUILD_DIR}/tools/converter/adapter/dpico/libdpico_atc_adapter.so
                    DESTINATION ${CONVERTER_ROOT_DIR}/providers/SD3403 COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(MSLITE_ENABLE_TOOLS)
                install(TARGETS ${BECHCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()

        if(MSLITE_ENABLE_RUNTIME_GLOG)
            install(DIRECTORY ${glog_LIBPATH}/../include/glog/ DESTINATION ${RUNTIME_INC_DIR}/third_party/glog
                    COMPONENT ${RUNTIME_COMPONENT_NAME} FILES_MATCHING PATTERN "*.h")
            install(FILES ${glog_LIBPATH}/${glog_name}
                    DESTINATION ${GLOG_DIR} RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(MSLITE_ENABLE_RUNTIME_CONVERT)
            install(FILES ${BUILD_DIR}/tools/converter/registry/libmslite_converter_plugin.so
                    DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})

            if(MSLITE_ENABLE_OPENCV)
                install(FILES ${opencv_LIBPATH}/libopencv_core.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_core.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgcodecs.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgcodecs.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${opencv_LIBPATH}/libopencv_imgproc.so.4.5.2
                        DESTINATION ${RUNTIME_LIB_DIR} RENAME libopencv_imgproc.so.4.5
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
        if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            __install_micro_wrapper()
            __install_micro_codegen()
        endif()
    endif()
    if(MSLITE_ENABLE_TOOLS)
        if(MSLITE_ENABLE_GRAPH_KERNEL AND CMAKE_SYSTEM_NAME MATCHES "Linux")
            if(EXISTS ${BUILD_DIR}/akg)
                set(AKG_PATH ${BUILD_DIR}/akg)
                file(REMOVE_RECURSE ${AKG_PATH}/build/akg/lib)
                install(DIRECTORY  ${AKG_PATH}/build/akg
                        DESTINATION ${BUILD_DIR}/package/mindspore_lite
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${AKG_PATH}/${AKG_PKG_PATH}
                        DESTINATION ${RUNTIME_PKG_NAME}/tools/akg
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${AKG_PATH}/${AKG_PKG_PATH}.sha256
                        DESTINATION ${RUNTIME_PKG_NAME}/tools/akg
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
                install(FILES ${AKG_PATH}/build/libakg.so
                        DESTINATION ${BUILD_DIR}/package/mindspore_lite/lib
                        COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
        if(NOT MSLITE_COMPILE_TWICE)
            install(TARGETS ${BENCHMARK_NAME} RUNTIME DESTINATION ${BENCHMARK_ROOT_DIR}
                    COMPONENT ${RUNTIME_COMPONENT_NAME})
        endif()
        if(SUPPORT_TRAIN)
            install(TARGETS ${BENCHMARK_TRAIN_NAME} RUNTIME DESTINATION ${BENCHMARK_TRAIN_ROOT_DIR} COMPONENT
                    ${RUNTIME_COMPONENT_NAME})
        endif()
        if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
            install(TARGETS cropper RUNTIME DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/cropper/cropper_mapping_cpu.cfg
                    DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/cropper/cropper_mapping_gpu.cfg
                    DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            install(FILES ${BUILD_DIR}/tools/cropper/cropper_mapping_npu.cfg
                    DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            if(SUPPORT_TRAIN)
                install(FILES ${BUILD_DIR}/tools/cropper/cropper_mapping_cpu_train.cfg
                        DESTINATION ${CROPPER_ROOT_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
            endif()
        endif()
    endif()
endif()

if(MSLITE_ENABLE_KERNEL_EXECUTOR)
    file(GLOB GEN_OPS_NAME_H ${TOP_DIR}/mindspore/mindspore/ops/op_def/auto_generate/gen_ops_name_*.h)
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/ops/op_def/auto_generate/gen_lite_ops.h
            ${GEN_OPS_NAME_H}
            ${TOP_DIR}/mindspore/mindspore/core/include/ops/base_operator.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/custom.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/conv2d.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/conv2d_transpose.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/max_pool.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/pad.h
            ${TOP_DIR}/mindspore/mindspore/ops/infer/topk.h
            DESTINATION ${RUNTIME_INC_DIR}/ops
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/format.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_id.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/types.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/macros.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/shared_ptr.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/type_traits.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/base/base.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/base
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/ir/common.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/ir/primitive.h
            ${TOP_DIR}/mindspore/mindspore/core/include/mindapi/ir/value.h
            DESTINATION ${RUNTIME_INC_DIR}/mindapi/ir
            COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${TOP_DIR}/mindspore-lite/src/litert/cxx_api/kernel_executor/kernel_executor.h DESTINATION
            ${RUNTIME_INC_DIR}/api COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(TARGETS kernel_executor DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(TARGETS mindspore_core mindspore_ops DESTINATION ${RUNTIME_LIB_DIR} COMPONENT ${RUNTIME_COMPONENT_NAME})
    install(FILES ${glog_LIBPATH}/${glog_name} DESTINATION ${RUNTIME_LIB_DIR}
        RENAME libmindspore_glog.so.0 COMPONENT ${RUNTIME_COMPONENT_NAME})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CPACK_GENERATOR ZIP)
else()
    set(CPACK_GENERATOR TGZ)
endif()

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_COMPONENTS_ALL ${RUNTIME_COMPONENT_NAME})
set(CPACK_PACKAGE_FILE_NAME ${PKG_NAME_PREFIX})

if(WIN32)
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output)
else()
    set(CPACK_PACKAGE_DIRECTORY ${TOP_DIR}/output/tmp)
endif()
set(CPACK_PACKAGE_CHECKSUM SHA256)
include(CPack)
