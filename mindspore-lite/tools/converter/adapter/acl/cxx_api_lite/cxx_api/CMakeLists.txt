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

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
elseif(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_CUSTOM_PATH}/latest")
else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/latest")
endif()
set(ASCEND_VERSION_FILE "${ASCEND_HOME_PATH}/compiler/version.info")
if(NOT EXISTS ${ASCEND_VERSION_FILE})
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    set(ASCEND_VERSION_FILE "${ASCEND_HOME_PATH}/compiler/version.info")
endif()
if(MSLITE_ENABLE_ACL AND EXISTS ${ASCEND_VERSION_FILE})
    file(STRINGS ${ASCEND_VERSION_FILE} CANN_VERSION_STRING LIMIT_INPUT 20)
    string(REGEX MATCH "([0-9]+\\.[0-9]+)" EXTRACTED_VERSION "${CANN_VERSION_STRING}")
    set(VERSION_FLOW "${EXTRACTED_VERSION}")
    message("cann version:" ${VERSION_FLOW})
    if(VERSION_FLOW GREATER_EQUAL 7.5)
        message("define enable bundle")
        add_compile_definitions(ENABLE_BUNDLE)
    endif()
endif()

if(WIN32)
    # define this for msvc:dllexport or dllimport
    add_compile_definitions(BUILDING_DLL)
endif()

# build mindspore_shared_lib
include_directories(${TOP_DIR}/mindspore/ccsrc)
include_directories(${TOP_DIR}/mindspore/ccsrc/minddata/dataset)

if(ENABLE_D OR ENABLE_ACL)
    # build 910 and 310 code into one distro, files needed for 310 mode
    add_compile_definitions(ENABLE_ACL)
    include_directories(${CMAKE_BINARY_DIR}/proto/ge)
endif()

if(MODE_ASCEND_ACL OR BUILD_LITE)
    add_compile_definitions(MODE_ASCEND_ACL)
    file(GLOB_RECURSE API_ACL_SRC ${CMAKE_CURRENT_SOURCE_DIR}
            "model/acl/*.cc"
            "model/model_converter_utils/*.cc"
            "graph/acl/*.cc"
            )
endif()

set(MSLIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/types.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/cell.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/factory.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/any_utils.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_data.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc
              ${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc
              ${API_ACL_SRC}
              )
if(ENABLE_D OR ENABLE_GPU)
    list(APPEND MSLIB_SRC ${CMAKE_CURRENT_SOURCE_DIR}/akg_kernel_register.cc)
endif()

if(NOT BUILD_LITE)
    list(APPEND MSLIB_SRC_DEPEND
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/strategy_checkpoint/strategy_checkpoint_info.cc)
    list(APPEND MSLIB_SRC_DEPEND
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/group_manager.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/device_manager.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/device_matrix.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/array.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/map.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/arrangement.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/shape_util.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/frontend/parallel/tensor_layout/tensor_layout.cc)
    list(APPEND MSLIB_SRC_DEPEND ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc)
endif()

if((MODE_ASCEND_ACL) AND NOT BUILD_LITE)
    list(APPEND MSLIB_SRC
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/backend_manager/backend_jit_config.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/optimizer/pattern_engine.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/optimizer/helper.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/optimizer/node_pass.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/optimizer/visitor.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/operator/ops_backend_infer_function.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/common/kernel_build_info.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/kernel/kernel_info.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/ascend/optimizer/mindir/reduce_axis_update.cc"
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/ascend/optimizer/ge_optimization.cc")
    # MODE_ASCEND_ACL don't need reg akg kernel
    list(REMOVE_ITEM MSLIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/akg_kernel_register.cc")
endif()

if(NOT ENABLE_TESTCASES AND NOT BUILD_LITE)
    # users of shared_lib cannot find symbols in indirect dependency
    set(MSLIB_SRC ${MSLIB_SRC} ${CMAKE_SOURCE_DIR}/mindspore/core/utils/status.cc)
endif()

if(BUILD_LITE)
    list(APPEND MSLIB_SRC
        "${TOP_DIR}/mindspore/mindspore/ccsrc/utils/config_manager.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/model/aoe/auto_tune_process.cc")
    file(GLOB_RECURSE ACL_REMOVE_SRC ${CMAKE_CURRENT_SOURCE_DIR}
            "model/acl/acl_vm/*.cc"
            )
    list(REMOVE_ITEM MSLIB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/akg_kernel_register.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/acl/acl_model_multi.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/acl/acl_model.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/types.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/model/model_impl.cc"
            ${ACL_REMOVE_SRC})
endif()

add_library(common_shared_lib_obj OBJECT ${MSLIB_SRC})
if(BUILD_LITE)
    add_library(mslite_shared_lib SHARED $<TARGET_OBJECTS:common_shared_lib_obj> ${MSLIB_SRC_DEPEND})
else()
    add_library(mindspore_shared_lib SHARED $<TARGET_OBJECTS:common_shared_lib_obj> ${MSLIB_SRC_DEPEND})
endif()
if(BUILD_LITE)
    target_link_libraries(mslite_shared_lib PRIVATE mindspore_graph_ir)
    add_dependencies(mslite_shared_lib mindspore_graph_ir)
elseif(MODE_ASCEND_ACL)
    target_link_libraries(mindspore_shared_lib PRIVATE mindspore_graph_ir
            _mindspore_backend_graph_compiler_obj _mindspore_debug_obj mindspore_backend_static)
    if(MS_BUILD_GRPC)
        target_link_libraries(mindspore_shared_lib PRIVATE mindspore::grpc++)
    endif()
    set_target_properties(mindspore_shared_lib PROPERTIES OUTPUT_NAME mindspore)
else()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        # wheel package and ut
        add_library(api_lib SHARED ${CMAKE_CURRENT_SOURCE_DIR}/types.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/context.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cell.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_data.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/model/model.cc
            ${CMAKE_SOURCE_DIR}/mindspore/core/utils/status.cc
            )
        target_link_libraries(api_lib PRIVATE mindspore_shared_lib -Wl,--no-as-needed ${PYTHON_LIBRARIES})
        set_target_properties(api_lib PROPERTIES OUTPUT_NAME mindspore)
        set_target_properties(api_lib PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
endif()

if(ENABLE_D OR ENABLE_GPU)
    if(BUILD_LITE)
        target_link_libraries(mslite_shared_lib PRIVATE -Wl,--as-needed mindspore::securec
                mindspore_ms_backend mindspore_core mindspore_ops mindspore_common mindspore_ops_kernel_common
                mindspore_backend_common proto_input mindspore::protobuf mindspore_profiler mindspore_backend_manager)
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE -Wl,--as-needed mindspore::securec
                mindspore_ms_backend mindspore_core mindspore_ops mindspore_common mindspore_ops_kernel_common
                mindspore_backend_common mindspore_profiler proto_input mindspore::protobuf mindspore_backend_manager)
    endif()
else()
    if(BUILD_LITE)
        target_link_libraries(mslite_shared_lib PRIVATE mindspore_core mindspore_ops mindspore::securec)
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE ${PY3_LIBG} mindspore::securec
                mindspore_ms_backend mindspore mindspore_core mindspore_ops mindspore_common mindspore_ops_kernel_common
                mindspore_backend_common mindspore_profiler proto_input mindspore::protobuf mindspore_backend_manager)
    endif()
endif()

if(ENABLE_CPU)
    if(BUILD_LITE)
        target_link_libraries(mslite_shared_lib PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
    endif()
endif()

if(USE_GLOG)
    if(BUILD_LITE)
        target_link_libraries(mslite_shared_lib PRIVATE mindspore::glog)
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE mindspore::glog)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    if(BUILD_LITE)
        target_link_options(mslite_shared_lib PRIVATE -Wl,-init,common_log_init)
    else()
        target_link_options(mindspore_shared_lib PRIVATE -Wl,-init,common_log_init)
    endif()
endif()

if(MODE_ASCEND_ACL OR MSLITE_ENABLE_ACL)
    # 310 mode
    add_compile_definitions(ENABLE_DVPP_INTERFACE)
    find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_cblas libacl_cblas.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_runtime libruntime.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(ge_compiler libge_compiler.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libplatform libplatform.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libcompress libcompress.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libopskernel libopskernel.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicore_utils libaicore_utils.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(libaicpu_engine_common libaicpu_engine_common.so ${ASCEND_CANN_RUNTIME_PATH}
            ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

    if(BUILD_LITE)
        target_link_libraries(mslite_shared_lib PRIVATE -Wl,--no-as-needed graph ${ge_compiler}
                ${acl_retr} ${acl_cblas} ${acl_dvpp} ${acl_runtime} ${libplatform} ${libcompress} ${libopskernel}
                ${libaicore_utils} ${libaicpu_engine_common} ${acl} ${GE_RUNNER})
    else()
        target_link_libraries(mindspore_shared_lib PRIVATE -Wl,--no-as-needed graph ${ge_compiler}
                ${acl_retr} ${acl_cblas} ${acl_dvpp} ${acl_runtime} ${libplatform} ${libcompress} ${libopskernel}
                ${libaicore_utils} ${libaicpu_engine_common} ${acl} ${GE_RUNNER})
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
# duplicate, should be remove after backend decoupling is done
    set(MINDSPORE_SHARED_LIB_RPATH $ORIGIN)
    if(BUILD_LITE)
        set_target_properties(mslite_shared_lib PROPERTIES INSTALL_RPATH ${MINDSPORE_SHARED_LIB_RPATH})
    else()
        set_target_properties(mindspore_shared_lib PROPERTIES INSTALL_RPATH ${MINDSPORE_SHARED_LIB_RPATH})
    endif()
endif()
