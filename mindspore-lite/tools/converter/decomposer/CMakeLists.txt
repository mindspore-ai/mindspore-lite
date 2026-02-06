file(GLOB DECOMPOSER
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )
set_property(SOURCE ${DECOMPOSER} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_LITE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=maybe-uninitialized")

if(NOT MSLITE_SIMPLEST_CLOUD_INFERENCE)
    add_library(decomposer_mid OBJECT ${DECOMPOSER})

    add_dependencies(decomposer_mid fbs_src)
    add_dependencies(decomposer_mid fbs_inner_src)
endif()