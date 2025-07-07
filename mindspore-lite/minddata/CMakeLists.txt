find_package(Patch)
if(NOT Patch_FOUND)
  message(FATAL_ERROR "Patch not found, please set environment variable MS_PATCH_PATH to path where Patch is located, "
  "usually found in GIT_PATH/usr/bin on Windows")
endif()

include(${TOP_DIR}/cmake/external_libs/jpeg_turbo.cmake)
include(${TOP_DIR}/cmake/external_libs/zlib.cmake)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb -fno-inline-functions -fno-omit-frame-pointer \
  -D_LIBCPP_INLINE_VISIBILITY='' -D_LIBCPP_DISABLE_EXTERN_TEMPLATE=1 -DHALF_ENABLE_CPP11_USER_LITERALS=0 \
  -D_FORTIFY_SOURCE=2 -Wno-cpp")
if(TARGET_AOS_ARM)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-sign-compare -Wno-overloaded-virtual \
        -Wno-unused-variable")
    set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -std=c++17 -Wall -fPIC -march=armv8.2-a -funsafe-math-optimizations \
        -ftree-vectorize -flax-vector-conversions")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Werror -Wno-return-std-move -Wno-unused-private-field \
      -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration \
      -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override")
    set(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -I/usr/local/include -std=c++17 -Wall -fPIC")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTION_CXX_FLAGS}")

if(PLATFORM_ARM)
    if(TARGET_AOS_ARM)
        set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wno-sign-compare -Wno-overloaded-virtual \
          -Wno-unused-variable -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2 -O2")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Werror -Wno-return-std-move -Wno-unused-private-field \
          -Wno-unused-lambda-capture -Wno-sign-compare -Wno-overloaded-virtual -Wno-unneeded-internal-declaration \
          -Wno-unused-variable -Wno-pessimizing-move -Wno-inconsistent-missing-override \
          -DHALF_ENABLE_CPP11_USER_LITERALS=0 -D_FORTIFY_SOURCE=2")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O2 -Wl,--allow-shlib-undefined -DHALF_ENABLE_CPP11_USER_LITERALS=0 \
      -D_FORTIFY_SOURCE=2")
endif()
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # debug mode do not change flags
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
      -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
      -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    if(TARGET_AOS_ARM)
        set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -fstack-protector-strong -Wno-attributes \
        -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -fstack-protector-strong -Wno-attributes \
        -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong -Wno-attributes \
        -Wno-deprecated-declarations -Wno-missing-braces ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-fPIC -fPIE -D_FORTIFY_SOURCE=2 -O2 -Wall -Werror -fstack-protector-strong \
        -Wno-attributes -Wno-deprecated-declarations -Wno-missing-braces -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
    endif()
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-non-virtual-dtor")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sequence-point")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
if(PLATFORM_ARM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-virtual-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS} -s")

if(MSLITE_ENABLE_RUNTIME_GLOG AND (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
    add_definitions(-DUSE_GLOG)
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

message(STATUS "Compile minddata in [${MSLITE_MINDDATA_IMPLEMENT}] mode")
if(MSLITE_MINDDATA_IMPLEMENT STREQUAL "full")
    include_directories("./")
    include_directories("dataset")
    include_directories("dataset/kernels/image")
    include_directories("dataset/liteapi")
    include_directories("${TOP_DIR}/mindspore-lite")
    include_directories("${TOP_DIR}")
    include_directories("${TOP_DIR}/mindspore/mindspore/ops/kernel/cpu")

    if(MSLITE_ENABLE_ACL)
        include_directories(${CCSRC_DIR})
    endif()

    set(LITE_SRC_FILES
        ${TOP_DIR}/mindspore-lite/src/litert/cxx_api/types.cc
        ${TOP_DIR}/mindspore-lite/src/litert/cxx_api/tensor_utils.cc
        ${TOP_DIR}/mindspore-lite/src/litert/cxx_api/tensor/tensor_impl.cc
        ${TOP_DIR}/mindspore-lite/src/tensor.cc
        ${TOP_DIR}/mindspore/mindspore/ops/kernel/cpu/nnacl/tensor_c_utils.c
        ${TOP_DIR}/mindspore-lite/src/common/utils.cc
        ${TOP_DIR}/mindspore-lite/src/common/string_util.cc)

    set(MINDDATA_API_SRC_FILES
        dataset/api/execute.cc
        dataset/api/transforms.cc
        dataset/api/datasets.cc
        dataset/api/samplers.cc
        dataset/api/iterator.cc
        dataset/api/data_helper.cc
        dataset/api/vision.cc)

    set(MINDDATA_UTIL_SRC_FILES
        dataset/util/memory_pool.cc
        dataset/util/path.cc
        dataset/util/status.cc
        dataset/util/service.cc
        dataset/util/json_helper.cc
        dataset/util/cond_var.cc
        dataset/util/task_manager.cc
        dataset/util/services.cc
        dataset/util/task.cc
        dataset/util/circular_pool.cc
        dataset/util/lock.cc
        dataset/util/wait_post.cc
        dataset/util/intrp_service.cc
        dataset/util/arena.cc)

    set(MINDDATA_CORE_SRC_FILES
        dataset/core/config_manager.cc
        dataset/core/data_type.cc
        dataset/core/de_tensor.cc
        dataset/core/global_context.cc
        dataset/core/tensor.cc
        dataset/core/tensor_helpers.cc
        dataset/core/tensor_row.cc
        dataset/core/tensor_shape.cc
        dataset/core/client.cc)

    set(MINDDATA_KERNELS_SRC_FILES
        dataset/kernels/tensor_op.cc)

    set(MINDDATA_KERNELS_IMAGE_SRC_FILES
        dataset/kernels/image/affine_op.cc
        dataset/kernels/image/lite_image_utils.cc
        dataset/kernels/image/center_crop_op.cc
        dataset/kernels/image/crop_op.cc
        dataset/kernels/image/decode_op.cc
        dataset/kernels/image/gaussian_blur_op.cc
        dataset/kernels/image/hwc_to_chw_op.cc
        dataset/kernels/image/normalize_op.cc
        dataset/kernels/image/resize_op.cc
        dataset/kernels/image/resize_preserve_ar_op.cc
        dataset/kernels/image/rgb_to_bgr_op.cc
        dataset/kernels/image/rgb_to_gray_op.cc
        dataset/kernels/image/rotate_op.cc
        dataset/kernels/image/random_affine_op.cc
        dataset/kernels/image/math_utils.cc
        dataset/kernels/image/exif_utils.cc)

    set(MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES
        dataset/kernels/image/lite_cv/canny.cc
        dataset/kernels/image/lite_cv/gaussian_blur.cc
        dataset/kernels/image/lite_cv/image_process.cc
        dataset/kernels/image/lite_cv/lite_mat.cc
        dataset/kernels/image/lite_cv/warp_affine.cc)

    set(MINDDATA_KERNELS_DATA_SRC_FILES
        dataset/kernels/data/compose_op.cc
        dataset/kernels/data/data_utils.cc
        dataset/kernels/data/duplicate_op.cc
        dataset/kernels/data/one_hot_op.cc
        dataset/kernels/data/random_apply_op.cc
        dataset/kernels/data/random_choice_op.cc
        dataset/kernels/data/type_cast_op.cc)

    set(MINDDATA_KERNELS_IR_SRC_FILES
        dataset/kernels/ir/validators.cc)

    set(MINDDATA_KERNELS_IR_DATA_SRC_FILES
        dataset/kernels/ir/data/transforms_ir.cc)

    set(MINDDATA_KERNELS_IR_VISION_SRC_FILES
        dataset/kernels/ir/vision/affine_ir.cc
        dataset/kernels/ir/vision/auto_contrast_ir.cc
        dataset/kernels/ir/vision/bounding_box_augment_ir.cc
        dataset/kernels/ir/vision/center_crop_ir.cc
        dataset/kernels/ir/vision/crop_ir.cc
        dataset/kernels/ir/vision/cutmix_batch_ir.cc
        dataset/kernels/ir/vision/cutout_ir.cc
        dataset/kernels/ir/vision/decode_ir.cc
        dataset/kernels/ir/vision/equalize_ir.cc
        dataset/kernels/ir/vision/gaussian_blur_ir.cc
        dataset/kernels/ir/vision/hwc_to_chw_ir.cc
        dataset/kernels/ir/vision/invert_ir.cc
        dataset/kernels/ir/vision/mixup_batch_ir.cc
        dataset/kernels/ir/vision/normalize_ir.cc
        dataset/kernels/ir/vision/normalize_pad_ir.cc
        dataset/kernels/ir/vision/pad_ir.cc
        dataset/kernels/ir/vision/random_affine_ir.cc
        dataset/kernels/ir/vision/random_color_adjust_ir.cc
        dataset/kernels/ir/vision/random_color_ir.cc
        dataset/kernels/ir/vision/random_crop_decode_resize_ir.cc
        dataset/kernels/ir/vision/random_crop_ir.cc
        dataset/kernels/ir/vision/random_crop_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_horizontal_flip_ir.cc
        dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_posterize_ir.cc
        dataset/kernels/ir/vision/random_resized_crop_ir.cc
        dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_resize_ir.cc
        dataset/kernels/ir/vision/random_resize_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_rotation_ir.cc
        dataset/kernels/ir/vision/random_select_subpolicy_ir.cc
        dataset/kernels/ir/vision/random_sharpness_ir.cc
        dataset/kernels/ir/vision/random_solarize_ir.cc
        dataset/kernels/ir/vision/random_vertical_flip_ir.cc
        dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.cc
        dataset/kernels/ir/vision/rescale_ir.cc
        dataset/kernels/ir/vision/resize_ir.cc
        dataset/kernels/ir/vision/resize_preserve_ar_ir.cc
        dataset/kernels/ir/vision/resize_with_bbox_ir.cc
        dataset/kernels/ir/vision/rgba_to_bgr_ir.cc
        dataset/kernels/ir/vision/rgba_to_rgb_ir.cc
        dataset/kernels/ir/vision/rgb_to_bgr_ir.cc
        dataset/kernels/ir/vision/rgb_to_gray_ir.cc
        dataset/kernels/ir/vision/rotate_ir.cc
        dataset/kernels/ir/vision/swap_red_blue_ir.cc
        dataset/kernels/ir/vision/uniform_aug_ir.cc)

    set(MINDDATA_ENGINE_SRC_FILES
        dataset/engine/tree_adapter_lite.cc
        dataset/engine/tree_modifier.cc
        dataset/engine/runtime_context.cc
        dataset/engine/tree_adapter.cc
        dataset/engine/execution_tree.cc
        dataset/engine/dataset_iterator.cc
        dataset/engine/data_schema.cc)

    set(MINDDATA_ENGINE_CONSUMERS_SRC_FILES
        dataset/engine/consumers/pull_based_tree_consumer.cc
        dataset/engine/consumers/tree_consumer.cc)

    set(MINDDATA_ENGINE_IR_DATASETOPS_SRC_FILES
        dataset/engine/ir/datasetops/dataset_node.cc
        dataset/engine/ir/datasetops/epoch_ctrl_node.cc
        dataset/engine/ir/datasetops/batch_node.cc
        dataset/engine/ir/datasetops/map_node.cc
        dataset/engine/ir/datasetops/root_node.cc
        dataset/engine/ir/datasetops/repeat_node.cc
        dataset/engine/ir/datasetops/project_node.cc
        dataset/engine/ir/datasetops/shuffle_node.cc
        dataset/engine/ir/datasetops/skip_node.cc)

    set(MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES
        dataset/engine/ir/datasetops/source/album_node.cc
        dataset/engine/ir/datasetops/source/mnist_node.cc)

    set(MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SRC_FILES
        dataset/engine/ir/datasetops/source/samplers/distributed_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/samplers_ir.cc
        dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/skip_first_epoch_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/subset_random_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.cc
        dataset/engine/ir/datasetops/source/samplers/weighted_random_sampler_ir.cc)

    set(MINDDATA_ENGINE_DATASETOPS_SRC_FILES
        dataset/engine/datasetops/dataset_op.cc
        dataset/engine/datasetops/repeat_op.cc
        dataset/engine/datasetops/epoch_ctrl_op.cc
        dataset/engine/datasetops/data_queue_op.cc
        dataset/engine/datasetops/project_op.cc
        dataset/engine/datasetops/shuffle_op.cc
        dataset/engine/datasetops/skip_op.cc
        dataset/engine/datasetops/pipeline_op.cc
        dataset/engine/datasetops/batch_op.cc)

    set(MINDDATA_ENGINE_DATASETOPS_MAP_OP_SRC_FILES
        dataset/engine/datasetops/map_op/map_op.cc
        dataset/engine/datasetops/map_op/cpu_map_job.cc)

    set(MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES
        dataset/engine/datasetops/source/album_op.cc
        dataset/engine/datasetops/source/mnist_op.cc
        dataset/engine/datasetops/source/mappable_leaf_op.cc
        dataset/engine/datasetops/source/io_block.cc)

    set(MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES
        dataset/engine/datasetops/source/sampler/sampler.cc
        dataset/engine/datasetops/source/sampler/subset_sampler.cc
        dataset/engine/datasetops/source/sampler/distributed_sampler.cc
        dataset/engine/datasetops/source/sampler/pk_sampler.cc
        dataset/engine/datasetops/source/sampler/random_sampler.cc
        dataset/engine/datasetops/source/sampler/sequential_sampler.cc
        dataset/engine/datasetops/source/sampler/skip_first_epoch_sampler.cc
        dataset/engine/datasetops/source/sampler/subset_random_sampler.cc
        dataset/engine/datasetops/source/sampler/weighted_random_sampler.cc)

    set(MINDDATA_ENGINE_OPT_PRE_SRC_FILES
        dataset/engine/opt/pre/add_skip_pass.cc
        dataset/engine/opt/pre/cache_validation_pass.cc
        dataset/engine/opt/pre/debug_mode_pass.cc
        dataset/engine/opt/pre/deep_copy_pass.cc
        dataset/engine/opt/pre/epoch_ctrl_pass.cc
        dataset/engine/opt/pre/getter_pass.cc
        dataset/engine/opt/pre/input_validation_pass.cc
        dataset/engine/opt/pre/insert_map_pass.cc
        dataset/engine/opt/pre/node_removal_pass.cc
        dataset/engine/opt/pre/skip_pushdown_pass.cc)

    set(MINDDATA_ENGINE_OPT_POST_SRC_FILES
        dataset/engine/opt/post/auto_worker_pass.cc)

    set(MINDDATA_ENGINE_OPT_SRC_FILES
        dataset/engine/opt/pass.cc)

    set(MINDDATA_ENGINE_PERF_SRC_FILES
        dataset/engine/perf/auto_tune.cc
        dataset/engine/perf/connector_size.cc
        dataset/engine/perf/dataset_iterator_tracing.cc
        dataset/engine/perf/device_queue_tracing.cc
        dataset/engine/perf/info_collector.cc
        dataset/engine/perf/monitor.cc
        dataset/engine/perf/profiling.cc)

    set(MINDDATA_CALLBACK_SRC_FILES
        dataset/callback/callback_manager.cc)

    set(MINDDATA_FULL_SRC
        ${LITE_SRC_FILES}
        ${MINDDATA_API_SRC_FILES}
        ${MINDDATA_UTIL_SRC_FILES}
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_KERNELS_IR_SRC_FILES}
        ${MINDDATA_KERNELS_IR_DATA_SRC_FILES}
        ${MINDDATA_KERNELS_IR_VISION_SRC_FILES}
        ${MINDDATA_ENGINE_SRC_FILES}
        ${MINDDATA_ENGINE_CONSUMERS_SRC_FILES}
        ${MINDDATA_ENGINE_IR_DATASETOPS_SRC_FILES}
        ${MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SRC_FILES}
        ${MINDDATA_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_MAP_OP_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SRC_FILES}
        ${MINDDATA_ENGINE_DATASETOPS_SOURCE_SAMPLER_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_PRE_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_POST_SRC_FILES}
        ${MINDDATA_ENGINE_OPT_SRC_FILES}
        ${MINDDATA_ENGINE_PERF_SRC_FILES}
        ${MINDDATA_CALLBACK_SRC_FILES}
        ${CORE_DIR}/utils/status.cc
        )

        if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
            set(MINDDATA_FULL_SRC
                ${MINDDATA_FULL_SRC}
                dataset/kernels/image/pad_op.cc
                dataset/kernels/ir/vision/pad_ir.cc
                dataset/kernels/image/swap_red_blue_op.cc
                dataset/kernels/ir/vision/swap_red_blue_ir.cc
                dataset/kernels/image/rescale_op.cc
                dataset/kernels/ir/vision/rescale_ir.cc
                dataset/core/cv_tensor.cc
                dataset/kernels/image/resize_cubic_op.cc)

            if(MSLITE_ENABLE_ACL)
                add_definitions(-DENABLE_DVPP)
                set(MINDDATA_FULL_SRC
                    ${MINDDATA_FULL_SRC}
                    dataset/core/device_tensor.cc
                    dataset/kernels/ir/vision/ascend_vision_ir.cc)
                add_subdirectory(dataset/kernels/image/dvpp kernels-dvpp-image)
            endif()
        endif()

    add_library(minddata-lite-obj OBJECT
            ${MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES}
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log.cc
            ${CORE_DIR}/utils/ms_utils.cc
            ${MINDDATA_FULL_SRC}
            )
    add_dependencies(minddata-lite-obj fbs_src)
    set(minddata_lite_submodules
        $<TARGET_OBJECTS:minddata-lite-obj>
    )
    if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
        add_dependencies(minddata-lite-obj kernels-dvpp-image)
        set(minddata_lite_submodules
            ${minddata_lite_submodules}
            $<TARGET_OBJECTS:kernels-dvpp-image>
        )
    endif()

    add_library(minddata-lite SHARED ${minddata_lite_submodules})
    add_library(minddata-lite_static STATIC ${minddata_lite_submodules})
    set_target_properties(minddata-lite_static PROPERTIES OUTPUT_NAME "minddata-lite")

    if(TARGET_AOS_ARM)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        target_link_libraries(minddata-lite
                securec
                mindspore::jpeg_turbo
                mindspore::turbojpeg
                mindspore::json
                pthread
                )
        target_link_libraries(minddata-lite_static
                securec
                mindspore::jpeg_turbo
                mindspore::turbojpeg
                mindspore::json
                pthread
                )
    else()
        find_package(Threads REQUIRED)
        target_link_libraries(minddata-lite
            securec
            mindspore::jpeg_turbo
            mindspore::turbojpeg
            mindspore::json
            Threads::Threads
            )
        target_link_libraries(minddata-lite_static
            securec
            mindspore::jpeg_turbo
            mindspore::turbojpeg
            mindspore::json
            Threads::Threads
            )
    endif()

    if((MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE) AND MSLITE_ENABLE_ACL)
        target_link_libraries(minddata-lite
            mindspore_core
            mindspore_ops
            mindspore::opencv_core
            mindspore::opencv_imgcodecs
            mindspore::opencv_imgproc
            )
        target_link_libraries(minddata-lite_static
            mindspore_core
            mindspore_ops
            mindspore::opencv_core
            mindspore::opencv_imgcodecs
            mindspore::opencv_imgproc
            )
    endif()

    # ref: https://github.com/android/ndk/issues/1202
    if(ANDROID_NDK)
        if(PLATFORM_ARM32)
            file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
            if(LIBCLANG_RT_LIB STREQUAL "")
                MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
            endif()
            target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
            target_link_libraries(minddata-lite_static ${LIBCLANG_RT_LIB})
        endif()

        if(PLATFORM_ARM32 OR PLATFORM_ARM64)
            target_link_libraries(minddata-lite log)
            target_link_libraries(minddata-lite_static log)
        endif()
    endif()
elseif(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite")
    include_directories(./)
    include_directories(dataset)
    include_directories("${TOP_DIR}/mindspore-lite")

    if(MSLITE_ENABLE_ACL)
        include_directories(${CCSRC_DIR})
    endif()

    set(MINDDATA_API_SRC_FILES
        dataset/api/execute.cc
        dataset/api/transforms.cc)

    set(MINDDATA_UTIL_SRC_FILES
        dataset/util/memory_pool.cc
        dataset/util/path.cc
        dataset/util/status.cc)

    set(MINDDATA_CORE_SRC_FILES
        dataset/core/ascend_resource.cc
        dataset/core/config_manager.cc
        dataset/core/data_type.cc
        dataset/core/de_tensor.cc
        dataset/core/device_resource.cc
        dataset/core/device_tensor.cc
        dataset/core/global_context.cc
        dataset/core/tensor.cc
        dataset/core/tensor_helpers.cc
        dataset/core/tensor_row.cc
        dataset/core/tensor_shape.cc
        dataset/core/types.cc)

    set(MINDDATA_KERNELS_SRC_FILES
        dataset/kernels/c_func_op.cc
        dataset/kernels/tensor_op.cc)

    set(MINDDATA_KERNELS_IMAGE_SRC_FILES
        dataset/kernels/image/adjust_brightness_op.cc
        dataset/kernels/image/adjust_contrast_op.cc
        dataset/kernels/image/adjust_gamma_op.cc
        dataset/kernels/image/adjust_hue_op.cc
        dataset/kernels/image/adjust_saturation_op.cc
        dataset/kernels/image/auto_augment_op.cc
        dataset/kernels/image/convert_color_op.cc
        dataset/kernels/image/crop_op.cc
        dataset/kernels/image/decode_op.cc
        dataset/kernels/image/decode_video_op.cc
        dataset/kernels/image/erase_op.cc
        dataset/kernels/image/exif_utils.cc
        dataset/kernels/image/gaussian_blur_op.cc
        dataset/kernels/image/horizontal_flip_op.cc
        dataset/kernels/image/lite_image_utils.cc
        dataset/kernels/image/math_utils.cc
        dataset/kernels/image/normalize_op.cc
        dataset/kernels/image/normalize_pad_op.cc
        dataset/kernels/image/pad_to_size_op.cc
        dataset/kernels/image/perspective_op.cc
        dataset/kernels/image/rand_augment_op.cc
        dataset/kernels/image/random_adjust_sharpness_op.cc
        dataset/kernels/image/random_auto_contrast_op.cc
        dataset/kernels/image/random_equalize_op.cc
        dataset/kernels/image/random_invert_op.cc
        dataset/kernels/image/random_lighting_op.cc
        dataset/kernels/image/resize_cubic_op.cc
        dataset/kernels/image/resize_op.cc
        dataset/kernels/image/resized_crop_op.cc
        dataset/kernels/image/rgb_to_bgr_op.cc
        dataset/kernels/image/rotate_op.cc
        dataset/kernels/image/slice_patches_op.cc
        dataset/kernels/image/to_tensor_op.cc
        dataset/kernels/image/trivial_augment_wide_op.cc
        dataset/kernels/image/vertical_flip_op.cc
        dataset/kernels/image/video_utils.cc)

    set(MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES
        dataset/kernels/image/lite_cv/canny.cc
        dataset/kernels/image/lite_cv/gaussian_blur.cc
        dataset/kernels/image/lite_cv/image_process.cc
        dataset/kernels/image/lite_cv/lite_mat.cc
        dataset/kernels/image/lite_cv/warp_affine.cc)

    set(MINDDATA_KERNELS_DATA_SRC_FILES
        dataset/kernels/data/compose_op.cc
        dataset/kernels/data/concatenate_op.cc
        dataset/kernels/data/data_utils.cc
        dataset/kernels/data/duplicate_op.cc
        dataset/kernels/data/fill_op.cc
        dataset/kernels/data/mask_op.cc
        dataset/kernels/data/one_hot_op.cc
        dataset/kernels/data/pad_end_op.cc
        dataset/kernels/data/parse_example_op.cc
        dataset/kernels/data/random_apply_op.cc
        dataset/kernels/data/random_choice_op.cc
        dataset/kernels/data/slice_op.cc
        dataset/kernels/data/to_float16_op.cc
        dataset/kernels/data/type_cast_op.cc
        dataset/kernels/data/unique_op.cc)

    set(MINDDATA_KERNELS_IR_SRC_FILES
        dataset/kernels/ir/validators.cc)

    set(MINDDATA_KERNELS_IR_DATA_SRC_FILES
        dataset/kernels/ir/data/transforms_ir.cc)

    set(MINDDATA_KERNELS_IR_VISION_SRC_FILES
        dataset/kernels/ir/vision/affine_ir.cc
        dataset/kernels/ir/vision/auto_contrast_ir.cc
        dataset/kernels/ir/vision/bounding_box_augment_ir.cc
        dataset/kernels/ir/vision/center_crop_ir.cc
        dataset/kernels/ir/vision/crop_ir.cc
        dataset/kernels/ir/vision/cutmix_batch_ir.cc
        dataset/kernels/ir/vision/cutout_ir.cc
        dataset/kernels/ir/vision/decode_ir.cc
        dataset/kernels/ir/vision/equalize_ir.cc
        dataset/kernels/ir/vision/gaussian_blur_ir.cc
        dataset/kernels/ir/vision/hwc_to_chw_ir.cc
        dataset/kernels/ir/vision/invert_ir.cc
        dataset/kernels/ir/vision/mixup_batch_ir.cc
        dataset/kernels/ir/vision/normalize_ir.cc
        dataset/kernels/ir/vision/normalize_pad_ir.cc
        dataset/kernels/ir/vision/pad_ir.cc
        dataset/kernels/ir/vision/random_affine_ir.cc
        dataset/kernels/ir/vision/random_color_adjust_ir.cc
        dataset/kernels/ir/vision/random_color_ir.cc
        dataset/kernels/ir/vision/random_crop_decode_resize_ir.cc
        dataset/kernels/ir/vision/random_crop_ir.cc
        dataset/kernels/ir/vision/random_crop_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_horizontal_flip_ir.cc
        dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_posterize_ir.cc
        dataset/kernels/ir/vision/random_resized_crop_ir.cc
        dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_resize_ir.cc
        dataset/kernels/ir/vision/random_resize_with_bbox_ir.cc
        dataset/kernels/ir/vision/random_rotation_ir.cc
        dataset/kernels/ir/vision/random_select_subpolicy_ir.cc
        dataset/kernels/ir/vision/random_sharpness_ir.cc
        dataset/kernels/ir/vision/random_solarize_ir.cc
        dataset/kernels/ir/vision/random_vertical_flip_ir.cc
        dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.cc
        dataset/kernels/ir/vision/rescale_ir.cc
        dataset/kernels/ir/vision/resize_ir.cc
        dataset/kernels/ir/vision/resize_preserve_ar_ir.cc
        dataset/kernels/ir/vision/resize_with_bbox_ir.cc
        dataset/kernels/ir/vision/rgba_to_bgr_ir.cc
        dataset/kernels/ir/vision/rgba_to_rgb_ir.cc
        dataset/kernels/ir/vision/rgb_to_gray_ir.cc
        dataset/kernels/ir/vision/rotate_ir.cc
        dataset/kernels/ir/vision/swap_red_blue_ir.cc
        dataset/kernels/ir/vision/uniform_aug_ir.cc)

    add_library(minddata-lite SHARED
        ${MINDDATA_API_SRC_FILES}
        ${MINDDATA_UTIL_SRC_FILES}
        ${MINDDATA_CORE_SRC_FILES}
        ${MINDDATA_KERNELS_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_SRC_FILES}
        ${MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES}
        ${MINDDATA_KERNELS_DATA_SRC_FILES}
        ${MINDDATA_KERNELS_IR_SRC_FILES}
        ${MINDDATA_KERNELS_IR_DATA_SRC_FILES}
        ${MINDDATA_KERNELS_IR_VISION_SRC_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/common/log.cc
        ${CORE_DIR}/utils/ms_utils.cc)

    target_link_libraries(minddata-lite
        securec
        jpeg-turbo
        jpeg
        mindspore::json)

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()
elseif(MSLITE_MINDDATA_IMPLEMENT STREQUAL "lite_cv")
    include_directories(./)
    include_directories(dataset)
    include_directories(dataset/kernels/image)
    include_directories("${TOP_DIR}/mindspore-lite")

    if(MSLITE_ENABLE_ACL)
        include_directories(${CCSRC_DIR})
    endif()

    set(MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES
        dataset/kernels/image/lite_cv/canny.cc
        dataset/kernels/image/lite_cv/gaussian_blur.cc
        dataset/kernels/image/lite_cv/image_process.cc
        dataset/kernels/image/lite_cv/lite_mat.cc
        dataset/kernels/image/lite_cv/warp_affine.cc)

    add_library(minddata-lite SHARED
        ${MINDDATA_KERNELS_IMAGE_LITE_CV_SRC_FILES})

    # ref: https://github.com/android/ndk/issues/1202
    if(PLATFORM_ARM32)
        file(GLOB_RECURSE LIBCLANG_RT_LIB $ENV{ANDROID_NDK}/libclang_rt.builtins-arm-android.a)
        if(LIBCLANG_RT_LIB STREQUAL "")
            MESSAGE(FATAL_ERROR "Cannot find libclang_rt.builtins-arm-androi2d.a in $ENV{ANDROID_NDK}")
        endif()
        target_link_libraries(minddata-lite ${LIBCLANG_RT_LIB})
    endif()
endif()

if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
    target_link_libraries(minddata-lite ${PYTHON_LIBRARIES})
endif()
