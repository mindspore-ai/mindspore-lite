cmake_minimum_required(VERSION 3.12)
project(MindSpore_Lite_Python_API)

# set(CMAKE_VERBOSE_MAKEFILE on)
set(PYBIND11_CPP_STANDARD -std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function -Wl,-rpath,$ORIGIN/")

find_package(Python3 COMPONENTS Interpreter Development)

if(Python3_FOUND)
  find_package(Python3 COMPONENTS NumPy Development)

  if(Python3_NumPy_FOUND)
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${Python3_NumPy_INCLUDE_DIRS})
    include_directories(${TOP_DIR}/mindspore)
    include_directories(${TOP_DIR}/mindspore/mindspore/core/include)
    include_directories(${TOP_DIR}/mindspore/mindspore/core/mindrt)
    include_directories(${TOP_DIR}/mindspore/mindspore/core/mindrt/include)
    include_directories(${TOP_DIR}/mindspore/mindspore/ops/kernel/cpu/)

    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
      add_compile_definitions(MSLITE_ENABLE_CLOUD_INFERENCE)
      add_compile_definitions(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE)
    endif()

    file(GLOB_RECURSE PY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    if(NOT MSLITE_ENABLE_CONVERTER)
      file(GLOB_RECURSE CONVERTER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "converter_pybind.cc")
      list(REMOVE_ITEM PY_SRC_LIST ${CONVERTER_SRC_FILES})
    endif()
    if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
      file(GLOB_RECURSE LITE_INFER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lite_infer_pybind.cc")
      list(REMOVE_ITEM PY_SRC_LIST ${LITE_INFER_SRC_FILES})
    endif()
    pybind11_add_module(_c_lite_wrapper NO_EXTRAS ${PY_SRC_LIST})

    if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
      target_link_libraries(_c_lite_wrapper PRIVATE -Wl,--whole-archive mindspore-extendrt
              -Wl,--no-whole-archive -Wl,-z,relro,-z,now,-z,noexecstack -fstack-protector-all -s)
    endif()
    if(MSLITE_ENABLE_CONVERTER OR MSLITE_ENABLE_RUNTIME_GLOG)
      target_link_libraries(_c_lite_wrapper PRIVATE -Wl,--no-as-needed mindspore::glog
              -Wl,-z,relro,-z,now,-z,noexecstack -fstack-protector-all -s)
    endif()
    if(MSLITE_ENABLE_RUNTIME_GLOG AND NOT MSLITE_ENABLE_CONVERTER)
      target_link_options(_c_lite_wrapper PRIVATE -Wl,-init,mindspore_log_init)
    endif()
    if(MSLITE_ENABLE_CONVERTER)
      target_link_libraries(_c_lite_wrapper PRIVATE
              -Wl,-no-as-need
              mindspore_core
              mindspore_ops
              mslite_converter_plugin
              mindspore::opencv_core
              mindspore::opencv_imgcodecs
              mindspore::opencv_imgproc
              mindspore_converter
              )
    else()
      if(MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
        target_link_libraries(_c_lite_wrapper PRIVATE
                -Wl,-no-as-need
                mindspore_infer_shared_lib_obj
                )
      endif()
      if(NOT (MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE))
        target_link_libraries(_c_lite_wrapper PRIVATE -Wl,--whole-archive mindspore-lite_static
                -Wl,--no-whole-archive -Wl,-z,relro,-z,now,-z,noexecstack -fstack-protector-all -s)
      endif()
    endif()
  else()
    message(WARNING "NumPy not found, so Python API will not be compiled.")
  endif()
else()
  message(WARNING "Python3 not found, so Python API will not be compiled.")
endif()
