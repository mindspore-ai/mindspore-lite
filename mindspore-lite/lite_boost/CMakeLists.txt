cmake_minimum_required(VERSION 3.12)
project(MindSpore-Lite-Boost)

# ---------- Configurable variables with defaults ----------
if(NOT DEFINED Python3_EXECUTABLE)
  execute_process(
    COMMAND which python3
    OUTPUT_VARIABLE _python3_path
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Python3_EXECUTABLE "${_python3_path}" CACHE FILEPATH "Python3 executable")
endif()

if(NOT DEFINED PYTORCH_INSTALL_PATH)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE _site_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(PYTORCH_INSTALL_PATH "${_site_packages}/torch" CACHE PATH "PyTorch install path")
endif()

if(NOT DEFINED PYTORCH_NPU_INSTALL_PATH)
  if(NOT DEFINED _site_packages)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
      OUTPUT_VARIABLE _site_packages
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  set(PYTORCH_NPU_INSTALL_PATH "${_site_packages}/torch_npu" CACHE PATH "torch_npu install path")
endif()

if(NOT DEFINED ASCEND_PATH)
  if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
    set(ASCEND_PATH "$ENV{ASCEND_CUSTOM_PATH}" CACHE PATH "Ascend CANN root path")
  elseif(NOT "$ENV{ASCEND_TOOLKIT_HOME}" STREQUAL "")
    set(ASCEND_PATH "$ENV{ASCEND_TOOLKIT_HOME}" CACHE PATH "Ascend CANN root path")
  elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend CANN root path")
  else()
    message(FATAL_ERROR
      "ASCEND_PATH not set. "
      "Set ASCEND_CUSTOM_PATH, ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH.")
  endif()
endif()

set(CXX_STANDARD "17" CACHE STRING "C++ standard version")
set(ENABLE_GLIBCXX ON CACHE BOOL "Enable CXX11 ABI (ON=ABI 1, OFF=ABI 0)")

# ---------- Compiler flags ----------
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fpie -Wl,--build-id=none")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-common")

set(CMAKE_CXX_FLAGS "-fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack -fPIE -pie ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-fabi-version=11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-Wl,-Bsymbolic -rdynamic -Wl,--no-undefined ${CMAKE_CXX_FLAGS}")

set(LD_FLAGS_GLOBAL "-shared -rdynamic -ldl -Wl,-z,relro \
    -Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} -fexceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS_GLOBAL} \
    -pie -fPIE")

# ---------- Include & link directories ----------
link_directories(${PYTORCH_INSTALL_PATH}/lib)
link_directories(${PYTORCH_NPU_INSTALL_PATH}/lib)
link_directories(${ASCEND_PATH}/lib64)

set(ACL_INCLUDE_DIRS ${ASCEND_PATH}/include)
include_directories(
  ${ACL_INCLUDE_DIRS}
  ${ACL_INCLUDE_DIRS}/aclnn
  )

include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include/third_party/hccl/inc)
include_directories(${PYTORCH_NPU_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed)
include_directories(${PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ops)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('include'))"
  OUTPUT_VARIABLE Python3_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE Python3_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))"
  OUTPUT_VARIABLE Python3_LIBRARY_NAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(Python3_LIBRARIES ${Python3_LIBRARY_DIRS}/${Python3_LIBRARY_NAME})
if(NOT EXISTS "${Python3_LIBRARIES}")
    message(STATUS "Python library ${Python3_LIBRARIES} not found, trying alternate suffix...")
    string(REGEX MATCH "(\\.[^.]+)$" PYLIB_NAME_EXT "${Python3_LIBRARY_NAME}")
    string(REGEX REPLACE "(\\.[^.]+)$" "" PYLIB_NAME_WE "${Python3_LIBRARY_NAME}")
    if(PYLIB_NAME_EXT STREQUAL ".so")
        set(Python3_LIBRARIES "${Python3_LIBRARY_DIRS}/${PYLIB_NAME_WE}.a")
    elseif(PYLIB_NAME_EXT STREQUAL ".a")
        set(Python3_LIBRARIES "${Python3_LIBRARY_DIRS}/${PYLIB_NAME_WE}.so")
    else()
        message(FATAL_ERROR "Unknown Python library extension: ${Python3_LIBRARY_NAME}")
    endif()

    if(NOT EXISTS "${Python3_LIBRARIES}")
        message(FATAL_ERROR "Python library not found after checking .so and .a: ${Python3_LIBRARIES}")
    else()
        message(STATUS "Found alternate Python library: ${Python3_LIBRARIES}")
    endif()
else()
    message(STATUS "Found Python library: ${Python3_LIBRARIES}")
endif()
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# ---------- Target ----------
add_library(lite_boost_ops SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/plugin/rain_fusion_attention.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/plugin/recurrent_gated_delta_rule.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/plugin/pytorch_npu_helper.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops/register_ops.cc
)

target_compile_features(lite_boost_ops PRIVATE cxx_std_${CXX_STANDARD})
if(ENABLE_GLIBCXX)
  target_compile_definitions(lite_boost_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
else()
  target_compile_definitions(lite_boost_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_link_libraries(lite_boost_ops PRIVATE
  c10
  torch
  torch_cpu
  torch_npu
  ascendcl
  ${Python3_LIBRARIES}
  ${ASCEND_PATH}/lib64/libascendcl.so
  ${ASCEND_PATH}/lib64/libnnopbase.so
  ${ASCEND_PATH}/lib64/libopapi.so
  )
