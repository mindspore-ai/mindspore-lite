# find python library
if(MODE_ASCEND_ACL)
    get_filename_component(PYTHON_LIB_REALPATH ${PYTHON_LIBRARIES} ABSOLUTE)
    get_filename_component(PYTHON_LIB_DIR ${PYTHON_LIB_REALPATH} PATH)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import distutils.sysconfig as sysconfig; \
                  print(sysconfig.get_config_var('PY3LIBRARY'))"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE PY3LIBRARY)
    set(PY3_LIBG ${PYTHON_LIB_DIR}/${PY3LIBRARY})
    string(STRIP "${PY3_LIBG}" PY3_LIBG)
    message("Python3 general library = " ${PY3_LIBG})
endif()

if(ENABLE_D OR ENABLE_ACL)
    # build 910 and 310 code into one distro, files needed for 310 mode
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)

    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "acl/*.cc"
        "acl/model_converter_utils/*.cc"
        "acl/acl_vm/*.cc"
        )
endif()

if(ENABLE_D)
    # for 910 mode
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend/*.cc")
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE API_MS_INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}
        "gpu/*.cc")
endif()

set(MSLIB_INFER_SRC
    ${LITE_SRC_DIR}/litert/cxx_api/types.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cell.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_data.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model_group.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model/model_group_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/predict_task_queue.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_worker.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_pool.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_parallel_runner.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/model_parallel_runner_impl.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/model_pool/resource_manager.cc
    ${API_MS_INFER_SRC}
    ${API_ACL_SRC}
    ${API_OPS_SRC}
    )
add_library(mindspore_infer_shared_lib_obj OBJECT ${MSLIB_INFER_SRC})
add_dependencies(mindspore_infer_shared_lib_obj fbs_inner_src)
