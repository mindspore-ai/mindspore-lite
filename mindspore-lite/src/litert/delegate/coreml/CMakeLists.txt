include(${TOP_DIR}/cmake/utils.cmake)
__download_pkg(coreml https://gitee.com/mirrors/coremltools/repository/archive/4.1
        be3f3fa4e5b215a585b7ba98b841215c1217b1c56cf3d598550b88d552e1e036)
__download_pkg(mnn https://gitee.com/mirrors/mnn/repository/archive/1.2.0
        613bd5cbce2b50227a4fb847de6ffafb98d8660c34f4f6819aa73f8a152581b8)
message("coreml package path:${coreml_SOURCE_DIR}")
message("mnn package path:${mnn_SOURCE_DIR}")
file(COPY ${coreml_SOURCE_DIR}/mlmodel/format/DataStructures.proto DESTINATION ${TOP_DIR}/third_party/proto/coreml)
file(COPY ${coreml_SOURCE_DIR}/mlmodel/format/FeatureTypes.proto DESTINATION ${TOP_DIR}/third_party/proto/coreml)
file(COPY ${mnn_SOURCE_DIR}/source/backend/coreml/mlmodel/proto/Model.proto
        DESTINATION ${TOP_DIR}/third_party/proto/coreml)
file(COPY ${coreml_SOURCE_DIR}/mlmodel/format/NeuralNetwork.proto DESTINATION ${TOP_DIR}/third_party/proto/coreml)
file(COPY ${coreml_SOURCE_DIR}/mlmodel/format/Parameters.proto DESTINATION ${TOP_DIR}/third_party/proto/coreml)
file(GLOB PROTO_FILE "" ${TOP_DIR}/third_party/proto/coreml/*.proto)
ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
add_library(coreml_proto_mid OBJECT ${PROTO_SRCS})
include_directories(${CMAKE_BINARY_DIR}/proto)

file(GLOB_RECURSE COREML_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/coreml_graph.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/pass/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/../delegate_utils.cc
        )
add_library(coreml_kernel_mid OBJECT ${COREML_RUNTIME_SRC})
add_dependencies(coreml_kernel_mid fbs_src)
target_link_libraries(coreml_kernel_mid coreml_proto_mid)
