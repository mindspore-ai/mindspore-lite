set(BUILD_INC_910A_DIR ${ASCEND_PATH}/latest/include)
add_compile_options(-fstack-protector-all -D_FORTIFY_SOURCE=2 -O2)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)
opbuild(OPS_SRC ${ops_srcs} OUT_DIR ${ASCEND_AUTOGEN_PATH} INC_DIR ${BUILD_INC_910A_DIR}
        SEC_INC ${securec_INC})

add_library(cust_ascendc_op_proto SHARED ${ops_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
target_compile_definitions(cust_ascendc_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_ascendc_op_proto PRIVATE -fvisibility=hidden)
target_link_directories(cust_ascendc_op_proto PRIVATE
                        ${CMAKE_COMPILE_COMPILER_LIBRARY}
                        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
)
target_link_libraries(
  cust_ascendc_op_proto
  PRIVATE intf_pub
          exe_graph
          register
          tiling_api
          -Wl,--whole-archive
          rt2_registry
          -Wl,--no-whole-archive)
set_target_properties(cust_ascendc_op_proto PROPERTIES OUTPUT_NAME cust_opsproto_rt2.0)

add_library(cust_optiling SHARED ${ops_srcs})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE -fvisibility=hidden)
target_link_directories(cust_optiling PRIVATE
                        ${CMAKE_COMPILE_COMPILER_LIBRARY}
                        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
)
target_link_libraries(
  cust_optiling
  PRIVATE intf_pub
          exe_graph
          register
          tiling_api
          -Wl,--whole-archive
          rt2_registry
          -Wl,--no-whole-archive)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME cust_opmaster_rt2.0)

file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
add_library(cust_opapi SHARED ${aclnn_src})
target_link_directories(cust_opapi PRIVATE
                        ${CMAKE_COMPILE_COMPILER_LIBRARY}
                        ${CMAKE_COMPILE_RUNTIME_LIBRARY}
)
target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase)

add_custom_target(
  optiling_compat ALL
  COMMAND
    ln -sf
    lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
    ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so)

install(
  TARGETS cust_ascendc_op_proto
  LIBRARY
    DESTINATION
      ${PKG_PATH}/packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
)
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION ${PKG_PATH}/packages/vendors/${vendor_name}/op_proto/inc)
install(
  TARGETS cust_optiling
  LIBRARY
    DESTINATION
      ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
  DESTINATION ${PKG_PATH}/packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
install(TARGETS cust_opapi
        LIBRARY DESTINATION ${PKG_PATH}/packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION ${PKG_PATH}/packages/vendors/${vendor_name}/op_api/include)
