/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LITE_BOOST_OPS_PLUGIN_YTORCH_NPU_HELPER_H_
#define LITE_BOOST_OPS_PLUGIN_YTORCH_NPU_HELPER_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "ATen/Tensor.h"
#include "c10/util/Exception.h"
#include "torch/extension.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"

#define NPU_NAME_SPACE at_npu::native
using AclOpExecutor = struct aclOpExecutor;
using AclTensor = struct aclTensor;
using AclScalar = struct aclScalar;
using AclIntArray = struct aclIntArray;
using AclFloatArray = struct aclFloatArray;
using AclBoolArray = struct aclBoolArray;
using AclTensorList = struct aclTensorList;

template <typename T = void>
using FunctionPtr = T *;
constexpr int K_HASH_BUF_SIZE = 8192;
constexpr int K_HASH_BUF_MAX_SIZE = K_HASH_BUF_SIZE + 1024;
constexpr int64_t ACL_TENSOR_MAX_DIM_FOR_FORMAT = 5;
constexpr int64_t DIM_NUM_3D = 3;
constexpr int64_t DIM_NUM_4D = 4;
constexpr int64_t DIM_NUM_5D = 5;
extern thread_local char g_hashBuf[K_HASH_BUF_SIZE];
extern thread_local int g_hashOffset;

extern const std::vector<std::string> g_customLibPath;
extern const std::vector<std::string> g_defaultCustomLibPath;
inline const char *GetOpApiLibName(void) { return "libopapi.so"; }
void *GetFuncFromDefaultLib(const std::string &apiName);
void *FindFuncInCustomLibPath(const char *apiName, const std::string &libPath);
void *FindFuncInDefaultLibPath(const char *apiName, const std::string &libPath);
void *AllocateWorkspace(uint64_t workspaceSize, at::Tensor &workspaceTensor);
void *GetOpApiFuncAddr(const char *apiName);

template <std::string_view const &ApiName>
inline std::string GetWorkspaceSizeApiName() {
  constexpr std::string_view suffix = "GetWorkspaceSize";
  std::string result(ApiName);
  result += suffix;
  return result;
}

constexpr aclDataType K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
  ACL_UINT8,      ACL_INT8,         ACL_INT16,        ACL_INT32,        ACL_INT64,
  ACL_FLOAT16,    ACL_FLOAT,        ACL_DOUBLE,       ACL_DT_UNDEFINED, ACL_COMPLEX64,
  ACL_COMPLEX128, ACL_BOOL,         ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED,
  ACL_BF16,       ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED, ACL_DT_UNDEFINED};

template <typename T>
inline bool CheckDataPointer(const T *data) {
  if (data == nullptr) {
    TORCH_CHECK(false, "memcpy failed: source data is null pointer");
    return false;
  }
  return true;
}

inline bool CheckDataSize(size_t size) {
  if (size == 0) {
    TORCH_CHECK(false, "memcpy failed: copy size is 0 (no data to copy)");
    return false;
  }
  return true;
}

inline bool CheckBufferSpace(size_t size) {
  if (g_hashOffset + size > K_HASH_BUF_SIZE) {
    g_hashOffset = K_HASH_BUF_MAX_SIZE;
    TORCH_CHECK(false, "memcpy failed: buffer overflow");
    return false;
  }
  return true;
}

template <typename T>
inline bool ValidateMemcpyParams(const T *data, size_t size) {
  return CheckDataPointer(data) && CheckDataSize(size) && CheckBufferSpace(size);
}
AclTensor *ConvertType(const at::Tensor &atTensor);
AclScalar *ConvertType(const at::Scalar &atScalar);
AclIntArray *ConvertType(const at::IntArrayRef &atArray);

template <std::size_t N>
inline AclBoolArray *ConvertType(const std::array<bool, N> &value) {
  auto array = aclCreateBoolArray(value.data(), value.size());
  return array;
}

AclBoolArray *ConvertType(const at::ArrayRef<bool> &value);
AclTensorList *ConvertType(const at::TensorList &atTensorList);
AclTensor *ConvertType(const c10::optional<at::Tensor> &optTensor);
AclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &optArray);
AclScalar *ConvertType(const c10::optional<at::Scalar> &optScalar);

inline aclDataType ConvertType(const at::ScalarType scalarType) {
  return K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalarType)];
}

template <typename T>
T ConvertType(T value) {
  return value;
}

template <typename TargetFuncType, typename SourceType>
struct FunctionPointerConverter {
  static TargetFuncType Convert(SourceType ptr) {
    static_assert(sizeof(TargetFuncType) == sizeof(SourceType), "Function pointer size mismatch");
    static_assert(std::is_pointer_v<SourceType>, "SourceType must be a pointer type");
    static_assert(std::is_pointer_v<TargetFuncType>, "TargetFuncType must be a function pointer type");
    union {
      SourceType ptr;
      TargetFuncType func;
    } converter;
    converter.ptr = ptr;
    return converter.func;
  }
};

template <typename Tuple, size_t... I, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr, std::index_sequence<I...>) {
  using OpApiFunc = int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  auto func = FunctionPointerConverter<OpApiFunc, FuncPtrType *>::Convert(opApiAddr);
  return func;
}

template <typename Tuple, typename FuncPtrType>
auto ConvertToOpApiFunc(const Tuple &params, FuncPtrType *opApiAddr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return ConvertToOpApiFunc(params, opApiAddr, std::make_index_sequence<size>{});
}

void Release(AclTensor *p);
void Release(AclScalar *p);
void Release(AclIntArray *p);
void Release(AclBoolArray *p);
void Release(AclTensorList *p);

template <typename T>
void Release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple &t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &&...args) {
  return std::make_tuple(ConvertType(std::forward<Ts>(args))...);
}

template <typename Function, typename Tuple, size_t... I>
auto Call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto Call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return Call(f, t, std::make_index_sequence<size>{});
}

template <std::string_view const &op_name, typename GetWorkspaceSizeFuncType, typename... Args>
auto PrepareParamsAndCalcWorkspaceSize(uint64_t *workspace_size_addr, AclOpExecutor **executor_addr,
                                       GetWorkspaceSizeFuncType getWorkspaceSizeFuncAddr, Args &&...args) {
  auto converted_params = ConvertTypes(std::forward<Args>(args)..., workspace_size_addr, executor_addr);
  static auto get_workspace_size_func = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
  auto workspace_status = Call(get_workspace_size_func, converted_params);
  TORCH_CHECK(workspace_status == 0, "call ", op_name.data(), " failed, detail:", aclGetRecentErrMsg());
  return converted_params;
}

template <std::string_view const &op_name, typename... Args>
void EXEC_NPU_CMD(Args &&...args) {
  static const auto get_workspace_size_func_addr =
    GetOpApiFuncAddr((std::string(op_name) + "GetWorkspaceSize").c_str());
  static const auto op_api_func_addr = GetOpApiFuncAddr(op_name.data());
  static const auto init_memory_addr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
  static const auto uninit_memory_addr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
  static const auto release_memory_addr = GetOpApiFuncAddr("ReleaseHugeMem");

  TORCH_CHECK(get_workspace_size_func_addr != nullptr && op_api_func_addr != nullptr && init_memory_addr != nullptr &&
                uninit_memory_addr != nullptr && release_memory_addr != nullptr,
              "NPU op ", op_name.data(), " api not found in ", GetOpApiLibName(), ".");

  using InitHugeMemThreadLocal = int (*)(FunctionPtr<void>, bool);
  InitHugeMemThreadLocal init_mem_func = reinterpret_cast<InitHugeMemThreadLocal>(init_memory_addr);
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }

  uint64_t workspace_size = 0;
  AclOpExecutor *executor = nullptr;
  auto converted_params = PrepareParamsAndCalcWorkspaceSize<op_name>(
    &workspace_size, &executor, get_workspace_size_func_addr, std::forward<Args>(args)...);

  at::Tensor workspace_tensor;
  void *workspace_addr = AllocateWorkspace(workspace_size, workspace_tensor);
  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);

  auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {
    using OpApiFunc = int (*)(FunctionPtr<>, uint64_t, AclOpExecutor *, const aclrtStream);
    auto opApiFunc = FunctionPointerConverter<OpApiFunc, void *>::Convert(op_api_func_addr);
    auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
    TORCH_CHECK(api_ret == 0, "call ", op_name.data(), " failed, detail:", aclGetRecentErrMsg());
    ReleaseConvertTypes(converted_params);
    using ReleaseHugeMem = void (*)(FunctionPtr<void>, bool);
    ReleaseHugeMem release_mem_func = reinterpret_cast<ReleaseHugeMem>(release_memory_addr);
    if (release_mem_func) {
      release_mem_func(nullptr, false);
    }
    return api_ret;
  };

  at_npu::native::OpCommand cmd;
  cmd.Name(op_name.data());
  cmd.SetCustomHandler(acl_call);
  cmd.Run();

  using UnInitHugeMemThreadLocal = void (*)(FunctionPtr<void>, bool);
  UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(uninit_memory_addr);
  if (unInitMemFunc) {
    unInitMemFunc(nullptr, false);
  }
}
#endif  // LITE_BOOST_OPS_PLUGIN_YTORCH_NPU_HELPER_H_
