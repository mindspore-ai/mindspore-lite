file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_definitions(-DENABLE_DVPP_INTERFACE)

set(DVPP_UTILS_SRC
    # Ascend310
    MDAclProcess.cc
    DvppCommon.cc
    ErrorCode.cpp
    ResourceManager.cc
    AclLiteUtils.cc
    VdecHelper.cc
    dvpp_video.cc
    # plugin
    acl_plugin.cc
    )
if(NOT MSLITE_ENABLE_ACL)
    set(DVPP_UTILS_SRC
    ${DVPP_UTILS_SRC}
    acl_env_guard.cc # in lite, src code has acl_env_guard.cc impl
    )
endif()

add_library(dvpp_utils SHARED ${DVPP_UTILS_SRC})
enable_target_when_only_build_plugins(dvpp_utils)

if(NOT MSLITE_ENABLE_ACL)
# if(MSLITE_ENABLE_ACL)
#     find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
#     find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
#     find_library(nnopbase libnnopbase.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
#     find_library(acl_dvpp_op libacl_dvpp_op.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
#     find_library(acl_dvpp_mpi libacl_dvpp_mpi.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
#     # find acl_env_guard in ascend_kernel_plugin
#     target_link_libraries(dvpp_utils PRIVATE ascend_kernel_plugin minddata-lite ${acl} ${acl_dvpp}
#         mindspore_core ${nnopbase} ${acl_dvpp_op} ${acl_dvpp_mpi})
# else()
    find_library(acl_dvpp libacl_dvpp.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl libascendcl.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(nnopbase libnnopbase.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_dvpp_op libacl_dvpp_op.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(acl_dvpp_mpi libacl_dvpp_mpi.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(dvpp_utils PRIVATE _c_dataengine ${acl} ${acl_dvpp}
        mindspore_core ${nnopbase} ${acl_dvpp_op} ${acl_dvpp_mpi})
endif()
add_dependencies(dvpp_utils _mindspore_ascend_symbol_obj)
target_link_libraries(dvpp_utils PRIVATE $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>)

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
    add_dependencies(dvpp_utils fbs_src)
endif()
