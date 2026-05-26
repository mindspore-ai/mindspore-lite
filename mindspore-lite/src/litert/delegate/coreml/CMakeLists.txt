include(${TOP_DIR}/cmake/utils.cmake)
__download_pkg(coreml https://gitee.com/mirrors/coremltools/repository/archive/9.0
        6d914166b770709519056086017ebdcc1185c6e2ca05a228ffaa12b5ed972adb)
__download_pkg(mnn https://gitee.com/mirrors/mnn/repository/archive/3.3.0
        3f7d697e1a4db16992bec53a33561a5c8828b204f34ab4231e9dd15071b0cf81)
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
