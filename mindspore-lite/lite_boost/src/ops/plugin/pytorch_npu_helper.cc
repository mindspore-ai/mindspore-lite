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

#include "plugin/pytorch_npu_helper.h"
#include <unistd.h>
#include <dlfcn.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace {
const char *GetCustOpApiLibName() { return "libcust_opapi.so"; }

c10::Scalar ConvertTensorToScalar(const at::Tensor &tensor) {
  const at::Tensor *acl_input = &tensor;
  switch (acl_input->scalar_type()) {
    case at::ScalarType::Double:
      return c10::Scalar(*(double *)acl_input->data_ptr());
    case at::ScalarType::Long:
      return c10::Scalar(*(int64_t *)acl_input->data_ptr());
    case at::ScalarType::Float:
      return c10::Scalar(*(float *)acl_input->data_ptr());
    case at::ScalarType::Int:
      return c10::Scalar(*(int *)acl_input->data_ptr());
    case at::ScalarType::Half:
      return c10::Scalar(*(c10::Half *)acl_input->data_ptr());
    case at::ScalarType::Bool:
      return c10::Scalar(*(int8_t *)acl_input->data_ptr());
    case at::ScalarType::ComplexDouble:
      return c10::Scalar(*(c10::complex<double> *)acl_input->data_ptr());
    case at::ScalarType::ComplexFloat:
      return c10::Scalar(*(c10::complex<float> *)acl_input->data_ptr());
    case at::ScalarType::BFloat16:
      return c10::Scalar(*(c10::BFloat16 *)acl_input->data_ptr());
    default:
      return c10::Scalar();
  }
}

at::Tensor CopyTensorHostToDevice(const at::Tensor &cpu_tensor) {
  at::Tensor cpu_pin_mem_tensor = cpu_tensor.pin_memory();
  int device_index = 0;
  return cpu_pin_mem_tensor.to(c10::Device(torch_npu::utils::get_npu_device_type(), device_index),
                               cpu_pin_mem_tensor.scalar_type(), true, true);
}

at::Tensor CopyScalarToDevice(const c10::Scalar &cpu_scalar, at::ScalarType scalar_data_type) {
  return CopyTensorHostToDevice(scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

std::string RealPath(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return "";
  }
  char real_path_buf[PATH_MAX] = {0};
  if (realpath(path.c_str(), real_path_buf) == nullptr) {
    return "";
  }
  return std::string(real_path_buf);
}

std::vector<std::string> SplitStr(std::string s, const std::string &del) {
  int end = s.find(del);
  std::vector<std::string> path_list;
  while (end != -1) {
    path_list.push_back(s.substr(0, end));
    s.erase(s.begin(), s.begin() + end + 1);
    end = s.find(del);
  }
  path_list.push_back(s);
  return path_list;
}

std::vector<std::string> ProcessPathList(const std::string &path_str) { return SplitStr(path_str, ":"); }

void AppendLibPathSuffix(std::vector<std::string> &path_list) {
  std::transform(path_list.begin(), path_list.end(), path_list.begin(),
                 [](const std::string &current_path) { return current_path + "/op_api/lib/"; });
}

std::vector<std::string> ProcessCustomLibPath(const char *ascend_custom_opp_path) {
  std::string ascend_custom_opp_path_str(ascend_custom_opp_path);
  auto custom_lib_path_list = ProcessPathList(ascend_custom_opp_path_str);
  if (custom_lib_path_list.empty()) {
    return std::vector<std::string>();
  }
  AppendLibPathSuffix(custom_lib_path_list);
  return custom_lib_path_list;
}

std::vector<std::string> GetCustomLibPath() {
  const char *ascend_custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  if (ascend_custom_opp_path == nullptr) {
    return std::vector<std::string>();
  }
  return ProcessCustomLibPath(ascend_custom_opp_path);
}

std::string GetVendorsConfigFilePath(const std::string &vendors_path) { return RealPath(vendors_path + "/config.ini"); }

bool IsFileExist(const std::string &path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  return (access(path.c_str(), F_OK) == 0) ? true : false;
}

bool ValidateVendorsConfigFile(const std::string &config_file) {
  if (config_file.empty() || !IsFileExist(config_file)) {
    return false;
  }
  return true;
}

std::string ReadLoadPriorityLine(const std::string &config_file) {
  constexpr std::string_view kLoadPriorityPrefix = "load_priority=";
  std::ifstream ifs(config_file);
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.compare(0, kLoadPriorityPrefix.size(), kLoadPriorityPrefix) == 0) {
      break;
    }
  }
  return line;
}

std::string ExtractLoadPriorityValue(const std::string &line) {
  constexpr std::string_view kLoadPriorityPrefix = "load_priority=";
  std::string result = line;
  if (result.compare(0, kLoadPriorityPrefix.size(), kLoadPriorityPrefix) == 0) {
    result.erase(0, kLoadPriorityPrefix.size());
  }
  return result;
}

std::vector<std::string> ProcessVendorsList(const std::string &vendors_path, const std::string &line) {
  auto default_vendors_list = SplitStr(line, ",");
  std::transform(default_vendors_list.begin(), default_vendors_list.end(), default_vendors_list.begin(),
                 [&vendors_path](const std::string &it) { return RealPath(vendors_path + "/" + it + "/op_api/lib/"); });
  return default_vendors_list;
}

std::vector<std::string> ParseVendorsConfig(const std::string &vendors_path) {
  std::string vendors_config_file = GetVendorsConfigFilePath(vendors_path);
  if (!ValidateVendorsConfigFile(vendors_config_file)) {
    return {};
  }
  std::string line = ReadLoadPriorityLine(vendors_config_file);
  std::string priority_value = ExtractLoadPriorityValue(line);
  return ProcessVendorsList(vendors_path, priority_value);
}

std::vector<std::string> GetDefaultCustomLibPath() {
  const char *ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
  if (ascend_opp_path == nullptr) {
    return std::vector<std::string>();
  }
  std::string vendors_path(ascend_opp_path);
  vendors_path = vendors_path + "/vendors";
  return ParseVendorsConfig(vendors_path);
}

std::string GetCustomOpApiLibPath(const std::string &lib_path) {
  return RealPath(lib_path + "/" + GetCustOpApiLibName());
}

std::string GetDefaultCustomOpApiLibPath(const std::string &lib_path) {
  return RealPath(lib_path + "/" + GetCustOpApiLibName());
}

class DlHandleRegistry {
 public:
  ~DlHandleRegistry() {
    for (const auto &it : handles_) {
      if (it.second != nullptr) {
        (void)dlclose(it.second);
      }
    }
  }

  void *GetOrOpen(const char *lib_name) {
    if (lib_name == nullptr) {
      return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = handles_.find(lib_name);
    if (iter != handles_.end()) {
      return iter->second;
    }
    void *handler = dlopen(lib_name, RTLD_LAZY);
    handles_.emplace(lib_name, handler);
    return handler;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<std::string, void *> handles_;
};

DlHandleRegistry &GetDlHandleRegistry() {
  static DlHandleRegistry registry;
  return registry;
}

void *GetOpApiLibHandler(const char *lib_name) { return GetDlHandleRegistry().GetOrOpen(lib_name); }

void *GetOpApiFuncAddrInLib(void *handler, const std::string &api_name) { return dlsym(handler, api_name.c_str()); }

void *LoadDefaultCustomOpApiHandler(const std::string &default_cust_op_api_lib) {
  if (default_cust_op_api_lib.empty()) {
    return nullptr;
  }
  return GetOpApiLibHandler(default_cust_op_api_lib.c_str());
}

void *LoadCustomOpApiHandler(const std::string &cust_op_api_lib) {
  if (cust_op_api_lib.empty()) {
    return nullptr;
  }
  return GetOpApiLibHandler(cust_op_api_lib.c_str());
}
}  // namespace

const std::vector<std::string> g_customLibPath = GetCustomLibPath();
const std::vector<std::string> g_defaultCustomLibPath = GetDefaultCustomLibPath();

void *GetFuncFromDefaultLib(const std::string &api_name) {
  static auto op_api_handler = GetOpApiLibHandler(GetOpApiLibName());
  if (op_api_handler == nullptr) {
    return nullptr;
  }
  void *func_addr = GetOpApiFuncAddrInLib(op_api_handler, api_name);
  if (func_addr == nullptr) {
    return nullptr;
  }
  return func_addr;
}

void *FindFuncInCustomLibPath(const char *api_name, const std::string &lib_path) {
  auto cust_op_api_lib = GetCustomOpApiLibPath(lib_path);
  auto cust_op_api_handler = LoadCustomOpApiHandler(cust_op_api_lib);
  if (cust_op_api_handler != nullptr) {
    auto func_addr = GetOpApiFuncAddrInLib(cust_op_api_handler, api_name);
    if (func_addr != nullptr) {
      return func_addr;
    }
  }
  return nullptr;
}

void *FindFuncInDefaultLibPath(const char *api_name, const std::string &lib_path) {
  auto default_cust_op_api_lib = GetDefaultCustomOpApiLibPath(lib_path);
  auto cust_op_api_handler = LoadDefaultCustomOpApiHandler(default_cust_op_api_lib);
  if (cust_op_api_handler != nullptr) {
    auto func_addr = GetOpApiFuncAddrInLib(cust_op_api_handler, api_name);
    if (func_addr != nullptr) {
      return func_addr;
    }
  }
  return nullptr;
}

void *AllocateWorkspace(uint64_t workspace_size, at::Tensor &workspace_tensor) {
  if (workspace_size == 0) {
    return nullptr;
  }
  at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
  workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(c10::kByte));
  return const_cast<void *>(workspace_tensor.storage().data());
}

void *GetOpApiFuncAddr(const char *api_name) {
  for (const auto &lib_path : g_customLibPath) {
    void *func_addr = FindFuncInCustomLibPath(api_name, lib_path);
    if (func_addr != nullptr) {
      return func_addr;
    }
  }
  for (const auto &lib_path : g_defaultCustomLibPath) {
    void *func_addr = FindFuncInDefaultLibPath(api_name, lib_path);
    if (func_addr != nullptr) {
      return func_addr;
    }
  }
  return GetFuncFromDefaultLib(api_name);
}

AclTensor *ConvertType(const at::Tensor &at_tensor) {
  if (!at_tensor.defined()) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_type = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(acl_type != ACL_DT_UNDEFINED, std::string(c10::toString(scalar_data_type)) + " has not been supported")
  c10::SmallVector<int64_t, ACL_TENSOR_MAX_DIM_FOR_FORMAT> storage_dims;
  auto item_size = at_tensor.itemsize();
  if (item_size == 0) {
    AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
    return nullptr;
  }
  if (acl_type != ACL_STRING) {
    storage_dims.push_back(at_tensor.storage().nbytes() / item_size);
  }

  const auto dim_num = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  switch (dim_num) {
    case DIM_NUM_3D:
      format = ACL_FORMAT_NCL;
      break;
    case DIM_NUM_4D:
      format = ACL_FORMAT_NCHW;
      break;
    case DIM_NUM_5D:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }

  if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    c10::Scalar exp_scalar = ConvertTensorToScalar(at_tensor);
    at::Tensor acl_input = CopyScalarToDevice(exp_scalar, scalar_data_type);
    return aclCreateTensor(acl_input.sizes().data(), acl_input.sizes().size(), acl_type, acl_input.strides().data(),
                           acl_input.storage_offset(), format, storage_dims.data(), storage_dims.size(),
                           const_cast<void *>(acl_input.storage().data()));
  }

  auto acl_tensor_obj =
    aclCreateTensor(at_tensor.sizes().data(), at_tensor.sizes().size(), acl_type, at_tensor.strides().data(),
                    at_tensor.storage_offset(), format, storage_dims.data(), storage_dims.size(),
                    const_cast<void *>(at_tensor.storage().data()));
  return acl_tensor_obj;
}

AclScalar *ConvertType(const at::Scalar &at_scalar) {
  at::ScalarType scalar_data_type = at_scalar.type();
  aclDataType acl_type = K_ATEN_SCALAR_TYPE_TO_ACL_DATATYPE_TABLE[static_cast<int64_t>(scalar_data_type)];
  TORCH_CHECK(acl_type != ACL_DT_UNDEFINED, std::string(c10::toString(scalar_data_type)) + " has not been supported")
  AclScalar *acl_scalar_obj = nullptr;
  switch (scalar_data_type) {
    case at::ScalarType::Double: {
      double value = at_scalar.toDouble();
      acl_scalar_obj = aclCreateScalar(&value, acl_type);
      break;
    }
    case at::ScalarType::Long: {
      int64_t value = at_scalar.toLong();
      acl_scalar_obj = aclCreateScalar(&value, acl_type);
      break;
    }
    case at::ScalarType::Bool: {
      bool value = at_scalar.toBool();
      acl_scalar_obj = aclCreateScalar(&value, acl_type);
      break;
    }
    case at::ScalarType::ComplexDouble: {
      auto value = at_scalar.toComplexDouble();
      acl_scalar_obj = aclCreateScalar(&value, acl_type);
      break;
    }
    default:
      acl_scalar_obj = nullptr;
      break;
  }
  return acl_scalar_obj;
}

AclIntArray *ConvertType(const at::IntArrayRef &at_array) {
  return aclCreateIntArray(at_array.data(), at_array.size());
}

AclBoolArray *ConvertType(const at::ArrayRef<bool> &value) { return aclCreateBoolArray(value.data(), value.size()); }

AclTensorList *ConvertType(const at::TensorList &at_tensor_list) {
  std::vector<const AclTensor *> tensor_list(at_tensor_list.size());
  for (size_t i = 0; i < at_tensor_list.size(); i++) {
    tensor_list[i] = ConvertType(at_tensor_list[i]);
  }
  return aclCreateTensorList(tensor_list.data(), tensor_list.size());
}

AclTensor *ConvertType(const c10::optional<at::Tensor> &opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return ConvertType(opt_tensor.value());
  }
  return nullptr;
}

AclIntArray *ConvertType(const c10::optional<at::IntArrayRef> &opt_array) {
  if (opt_array.has_value()) {
    return ConvertType(opt_array.value());
  }
  return nullptr;
}

AclScalar *ConvertType(const c10::optional<at::Scalar> &opt_scalar) {
  if (opt_scalar.has_value()) {
    return ConvertType(opt_scalar.value());
  }
  return nullptr;
}

void Release(AclTensor *p) { aclDestroyTensor(p); }

void Release(AclScalar *p) { aclDestroyScalar(p); }

void Release(AclIntArray *p) { aclDestroyIntArray(p); }

void Release(AclBoolArray *p) { aclDestroyBoolArray(p); }

void Release(AclTensorList *p) { aclDestroyTensorList(p); }

thread_local char g_hashBuf[K_HASH_BUF_SIZE];
thread_local int g_hashOffset = 0;
