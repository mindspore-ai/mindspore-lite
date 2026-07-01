/**
 * Copyright 2021-2026 Huawei Technologies Co., Ltd
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

#include "extendrt/delegate/ascend_acl/model_process.h"
#include <sys/time.h>
#include <utility>
#include <algorithm>
#include <map>
#include <thread>
#include <set>
#include <string>
#include <vector>
#include "common/common.h"
#include "common/log_adapter.h"
#include "src/common/utils.h"
#include "src/common/log_util.h"
#include "src/extendrt/delegate/ascend_acl/acl_shared_memory_manager.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_mdl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace {
constexpr size_t kGranularitySize = 2097152;  // The physical memory size must be 2M aligned
constexpr char kINFOLogLevel = '1';
constexpr char kDEBUGLogLevel = '0';

static TypeId TransToDataType(aclDataType data_type) {
  static const std::map<aclDataType, enum TypeId> data_type_map = {
    {ACL_FLOAT16, TypeId::kNumberTypeFloat16}, {ACL_FLOAT, TypeId::kNumberTypeFloat32},
    {ACL_BF16, TypeId::kNumberTypeBFloat16},   {ACL_DOUBLE, TypeId::kNumberTypeFloat64},
    {ACL_INT8, TypeId::kNumberTypeInt8},       {ACL_INT16, TypeId::kNumberTypeInt16},
    {ACL_INT32, TypeId::kNumberTypeInt32},     {ACL_INT64, TypeId::kNumberTypeInt64},
    {ACL_UINT8, TypeId::kNumberTypeUInt8},     {ACL_UINT16, TypeId::kNumberTypeUInt16},
    {ACL_UINT32, TypeId::kNumberTypeUInt32},   {ACL_UINT64, TypeId::kNumberTypeUInt64},
    {ACL_BOOL, TypeId::kNumberTypeBool},       {ACL_BF16, TypeId::kNumberTypeBFloat16},
  };
  auto it = data_type_map.find(data_type);
  if (it == data_type_map.end()) {
    MS_LOG(ERROR) << "ModelProcess TransToDataType ERROR" << data_type;
    return TypeId::kNumberTypeEnd;
  } else {
    return it->second;
  }
}

bool CheckModelExecuteV2Support() {
  return HAS_ASCEND_API(aclmdlExecuteV2) && HAS_ASCEND_API(aclmdlCreateExecConfigHandle) &&
         HAS_ASCEND_API(aclmdlDestroyExecConfigHandle) && HAS_ASCEND_API(aclmdlSetExecConfigOpt);
}
}  // namespace

ModelProcess::~ModelProcess() {
  if (dynamic_dims_ != nullptr) {
    delete[] dynamic_dims_;
    dynamic_dims_ = nullptr;
  }
  if (allocator_ != nullptr) {
    delete allocator_;
    allocator_ = nullptr;
  }
}

aclError ModelProcess::AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
  uint64_t start_time = 0;
  auto env = std::getenv("GLOG_v");
  auto is_debug = env != nullptr && (env[0] == kDEBUGLogLevel || env[0] == kINFOLogLevel);
  if (is_debug) {
    start_time = lite::GetTimeUs();
  }
  auto ret = CALL_ASCEND_API(aclrtMemcpy, dst, destMax, src, count, kind);
  if (is_debug) {
    auto end_time = lite::GetTimeUs();
    auto cost = end_time - start_time;
    if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
      MS_LOG(INFO) << "[D2H] Device to Host copy in " << cost << " us";
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
      MS_LOG(INFO) << "[H2D] Host to Device copy in " << cost << " us";
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
      MS_LOG(INFO) << "[D2D] Device to Device copy in " << cost << " us";
    }
  }
  return ret;
}

Status ModelProcess::InitDynamicShapeConfig() {
  dynamic_shape_options_.batch_size = GetDynamicBatch();
  dynamic_shape_options_.image_size = GetDynamicImage();
  dynamic_shape_options_.dynamic_dims = GetDynamicDims();
  auto status = CheckAndSetDynFlag();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Check and set dynamic flag failed";
  }
  return status;
}

Status ModelProcess::InitModelStream() {
  auto &stream_sync_timeout = options_->model_exec_config.stream_sync_timeout;
  if (stream_sync_timeout.Value() == lite::kModelExecStreamSyncTimeoutIgnoreValue) {
    return kSuccess;
  }
  if (!CheckModelExecuteV2Support()) {
    MS_LOG(WARNING) << "The current CANN version does not support specify stream_sync_timeout, please upgrade CANN.";
    return kSuccess;
  }
  auto acl_ret = CALL_ASCEND_API(aclrtCreateStream, &stream_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Create stream failed.";
    return Status(kLiteAclInitFailed, "Create stream failed.");
  }
  exec_config_handle_ = CALL_ASCEND_API(aclmdlCreateExecConfigHandle);
  acl_ret = CALL_ASCEND_API(aclmdlSetExecConfigOpt, exec_config_handle_, ACL_MDL_STREAM_SYNC_TIMEOUT,
                            &stream_sync_timeout.Value(), stream_sync_timeout.Size());
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set stream sync timeout failed.";
    return Status(kLiteAclInitFailed, "Set stream sync timeout failed.");
  }
  return kSuccess;
}

Status ModelProcess::PreInitModelResource() {
  model_desc_ = CALL_ASCEND_API(aclmdlCreateDesc);
  auto acl_ret = CALL_ASCEND_API(aclmdlGetDesc, model_desc_, infer_id_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Read model desc failed, ret = " << acl_ret;
    return Status(kLiteAclInitFailed, "Get model description information failed!");
  }
  auto status = InitDynamicShapeConfig();
  if (status != kSuccess) {
    return status;
  }
  status = InitInputsBuffer();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Create input buffer failed.";
    return status;
  }
  status = InitOutputsBuffer();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Create output buffer failed.";
    return status;
  }
  status = InitModelStream();
  if (status != kSuccess) {
    return status;
  }
  if (is_dynamic_input_) {
    data_input_num_ = input_infos_.size();
    return kSuccess;
  }
  data_input_num_ = input_infos_.size();
  if (IsDynamicShape() && data_input_num_ > 0) {
    data_input_num_ -= 1;
  }
  dynamic_shape_options_.input_format = GetInputFormat();
  dynamic_shape_options_.input_shapes = GetInputShape();
  return dyn_shape_proc_.Init(dynamic_shape_options_);
}

std::set<uint64_t> ModelProcess::GetDynamicBatch() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, std::set<uint64_t>(), "Model desc is nullptr.");
  aclmdlBatch dynamic_batch;
  if (CALL_ASCEND_API(aclmdlGetDynamicBatch, model_desc_, &dynamic_batch) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic batch.";
    return std::set<uint64_t>();
  }
  size_t batch_count = dynamic_batch.batchCount;
  if (batch_count > ACL_MAX_BATCH_NUM) {
    MS_LOG(ERROR) << "Real batch count " << batch_count << " is larger than max " << ACL_MAX_BATCH_NUM;
    return std::set<uint64_t>();
  }
  std::set<uint64_t> batch;
  for (size_t i = 0; i < dynamic_batch.batchCount; ++i) {
    batch.insert(dynamic_batch.batch[i]);
  }
  return batch;
}

std::pair<aclmdlIODims *, size_t> ModelProcess::GetDynamicDims() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, (std::make_pair(nullptr, 0)), "Model desc is nullptr.");
  size_t gear_conut = 0;
  auto ret = CALL_ASCEND_API(aclmdlGetInputDynamicGearCount, model_desc_, -1, &gear_conut);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclmdlGetInputDynamicGearCount failed.";
    return std::make_pair(nullptr, 0);
  }
  MS_LOG(INFO) << "gear_conut is: " << gear_conut;
  if (gear_conut == 0) {
    MS_LOG(INFO) << "gear_conut is zero";
    return std::make_pair(nullptr, 0);
  }
  dynamic_dims_ = new aclmdlIODims[gear_conut];
  if (dynamic_dims_ == nullptr) {
    MS_LOG(ERROR) << "new aclmldIODims failed.";
    return std::make_pair(nullptr, 0);
  }
  if (CALL_ASCEND_API(aclmdlGetInputDynamicDims, model_desc_, -1, dynamic_dims_, gear_conut) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclmdlGetInputDynamicDims failed.";
    delete[] dynamic_dims_;
    dynamic_dims_ = nullptr;
    return std::make_pair(nullptr, 0);
  }
  return std::make_pair(dynamic_dims_, gear_conut);
}

std::set<std::pair<uint64_t, uint64_t>> ModelProcess::GetDynamicImage() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, (std::set<std::pair<uint64_t, uint64_t>>()), "Model desc is nullptr.");
  aclmdlHW dynamic_hw;
  if (CALL_ASCEND_API(aclmdlGetDynamicHW, model_desc_, 0, &dynamic_hw) != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Failed to get dynamic hw.";
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  size_t hw_count = dynamic_hw.hwCount;
  if (hw_count > ACL_MAX_HW_NUM) {
    MS_LOG(ERROR) << "Real hw count " << hw_count << " is larger than max " << ACL_MAX_HW_NUM;
    return std::set<std::pair<uint64_t, uint64_t>>();
  }
  std::set<std::pair<uint64_t, uint64_t>> image;
  for (size_t i = 0; i < dynamic_hw.hwCount; ++i) {
    image.insert(std::pair<uint64_t, uint64_t>(dynamic_hw.hw[i][0], dynamic_hw.hw[i][1]));
  }
  return image;
}

std::vector<Format> ModelProcess::GetInputFormat() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, std::vector<Format>(), "Model desc is nullptr.");
  std::vector<Format> input_formats;
  static const std::map<aclFormat, enum Format> acl_format_map = {
    {ACL_FORMAT_NCHW, NCHW}, {ACL_FORMAT_NHWC, NHWC}, {ACL_FORMAT_ND, NCHW}};
  for (size_t i = 0; i < data_input_num_; ++i) {
    aclFormat format = CALL_ASCEND_API(aclmdlGetInputFormat, model_desc_, i);
    auto iter = acl_format_map.find(format);
    if (iter != acl_format_map.end()) {
      input_formats.emplace_back(iter->second);
    } else {
      MS_LOG(INFO) << "aclFormat " << format << " not found in map, please double check and add...using default format";
      input_formats.emplace_back(DEFAULT_FORMAT);
    }
    MS_LOG(DEBUG) << "Format of Input " << i << " is " << static_cast<int32_t>(format);
  }
  return input_formats;
}

const std::vector<TypeId> ModelProcess::GetOutputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    TypeId data_type = TransToDataType(output_infos_[i].data_type);
    if (data_type == TypeId::kNumberTypeEnd) {
      MS_LOG(ERROR) << "ModelProcess GetOutputDataType error, data_type:" << data_type;
      return {};
    }
    data_types.emplace_back(data_type);
  }
  return data_types;
}

const std::vector<ShapeVector> ModelProcess::GetOutputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    shapes.emplace_back(output_infos_[i].dims);
  }
  return shapes;
}

const std::vector<ShapeVector> ModelProcess::GetInputShape() {
  std::vector<ShapeVector> shapes;
  for (size_t i = 0; i < data_input_num_; ++i) {
    shapes.push_back(input_infos_[i].dims);
  }
  return shapes;
}

const std::vector<TypeId> ModelProcess::GetInputDataType() {
  std::vector<TypeId> data_types;
  for (size_t i = 0; i < data_input_num_; ++i) {
    TypeId data_type = TransToDataType(input_infos_[i].data_type);
    data_types.emplace_back(data_type);
  }
  return data_types;
}

Status ModelProcess::CheckAndSetDynFlag() {
  aclError ret;
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_desc_);
  for (size_t i = 0; i < input_size; ++i) {
    auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
    aclmdlIODims input_dims;
    ret = CALL_ASCEND_API(aclmdlGetInputDimsV2, model_desc_, i, &input_dims);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Get input dims failed";
      return Status(kLiteAclInitFailed, "Get input dims failed!");
    }
    for (size_t j = 0; j < input_dims.dimCount; ++j) {
      if (input_dims.dims[j] < 0) {
        if (buffer_size == 0) {
          is_dynamic_input_ = true;
          MS_LOG(INFO) << "The input of model is dynamic.";
          break;
        } else {
          if (!IsDynamicShape()) {
            is_dynamic_shape_range_ = true;
            MS_LOG(INFO) << "The input of model is dynamic shape range";
          }
        }
      }
    }
    if (is_dynamic_input_ || is_dynamic_shape_range_) {
      break;
    }
  }
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  for (size_t i = 0; i < output_size; ++i) {
    aclmdlIODims output_dims;
    ret = CALL_ASCEND_API(aclmdlGetOutputDims, model_desc_, i, &output_dims);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Get output dims failed";
      return Status(kLiteAclInitFailed, "Get output dims failed!");
    }
    for (size_t j = 0; j < output_dims.dimCount; ++j) {
      if (output_dims.dims[j] < 0) {
        is_dynamic_output_ = true;
        MS_LOG(INFO) << "The output of model is dynamic.";
        return kSuccess;
      }
    }
  }
  return kSuccess;
}

Status ModelProcess::InitSingleInput(size_t i) {
  aclError ret;
  aclmdlIODims dims;
  if (is_dynamic_output_) {
    ret = CALL_ASCEND_API(aclmdlGetInputDims, model_desc_, i, &dims);
  } else {
    ret = CALL_ASCEND_API(aclmdlGetInputDimsV2, model_desc_, i, &dims);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Get input shape failed, ret = " << ret;
    return Status(kLiteAclInitFailed, "Get input shape failed!");
  }
  auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
  void *data_mem_buffer = nullptr;
  auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
  if (data_type == aclDataType::ACL_DT_UNDEFINED) {
    MS_LOG(ERROR) << "ModelProcess InitInputsBuffer ERROR" << data_type;
    return Status(kLiteAclInitFailed, "Get model input data type is invalid.");
  }
  if (!is_dynamic_input_) {
    auto status = CreateDataBuffer(&data_mem_buffer, buffer_size, inputs_);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Add input data buffer failed, buffer size " << buffer_size;
      return status;
    }
    auto input_format = CALL_ASCEND_API(aclmdlGetInputFormat, model_desc_, i);
    auto *desc = CALL_ASCEND_API(aclCreateTensorDesc, data_type, dims.dimCount, dims.dims, input_format);
    CALL_ASCEND_API(aclmdlSetDatasetTensorDesc, inputs_, desc, i);
    (void)CALL_ASCEND_API(aclDestroyTensorDesc, desc);
  }
  auto input_name = CALL_ASCEND_API(aclmdlGetInputNameByIndex, model_desc_, i);
  MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
  std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
  input_infos_.emplace_back(
    AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, input_name});
  return kSuccess;
}

Status ModelProcess::InitInputsBuffer() {
  inputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (inputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return Status(kLiteNullptr, "inputs_ is nullptr, Create input dataset failed.");
  }
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_desc_);
  MS_LOG(INFO) << "input_size = " << input_size;
  for (size_t i = 0; i < input_size; ++i) {
    auto status = InitSingleInput(i);
    if (status != kSuccess) {
      return status;
    }
  }
  MS_LOG(INFO) << "Create model inputs success";
  return kSuccess;
}

Status ModelProcess::InitOutputsBuffer() {
  aclError ret;
  outputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create output dataset failed";
    return Status(kLiteNullptr, "Create output dataset failed!");
  }
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  MS_LOG(INFO) << "Output_size = " << output_size;
  for (size_t i = 0; i < output_size; ++i) {
    aclmdlIODims dims;
    ret = CALL_ASCEND_API(aclmdlGetOutputDims, model_desc_, i, &dims);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Get output shape failed";
      return Status(kLiteAclInitFailed, "Get output shape failed!");
    }
    bool is_dynamic_output = false;
    for (size_t dim_idx = 0; dim_idx < dims.dimCount; dim_idx++) {
      is_dynamic_output = (dims.dims[dim_idx] < 0) ? true : false;
    }
    size_t buffer_size = 0;
    if (!is_dynamic_output) {
      buffer_size = CALL_ASCEND_API(aclmdlGetOutputSizeByIndex, model_desc_, i);
    }
    void *data_mem_buffer = nullptr;
    auto status = CreateDataBuffer(&data_mem_buffer, buffer_size, outputs_);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "Add output data buffer failed, buffer size " << buffer_size;
      return status;
    }
    aclFormat format = CALL_ASCEND_API(aclmdlGetOutputFormat, model_desc_, i);
    MS_LOG(DEBUG) << "The output format of om is " << format;
    aclDataType data_type = CALL_ASCEND_API(aclmdlGetOutputDataType, model_desc_, i);
    if (data_type == aclDataType::ACL_DT_UNDEFINED) {
      MS_LOG(ERROR) << "ModelProcess InitOutputsBuffer ERROR" << data_type;
      return Status(kLiteAclInitFailed, "Get model output data type is invalid.");
    }
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    if (is_dynamic_output) {
      shape = std::vector<int64_t>({-1});
    }
    std::string output_name = CALL_ASCEND_API(aclmdlGetOutputNameByIndex, model_desc_, i);
    if (output_name.empty()) {
      MS_LOG(WARNING) << "Get name of output " << i << " failed.";
    }
    MS_LOG(INFO) << "Name of om output " << i << " is " << output_name << "Buffer size " << buffer_size;
    output_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, output_name});
  }
  MS_LOG(INFO) << "Create model output success.";
  return kSuccess;
}

Status ModelProcess::AllocDataBufferMem(void **data_mem_buffer, size_t buffer_size) {
  aclError ret;
  if (buffer_size == 0) {
    return kSuccess;
  }
  if (!is_run_on_device_) {
    ret = CALL_ASCEND_API(aclrtMalloc, data_mem_buffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
    std::string error_msg = "Malloc device buffer failed, buffer size " + std::to_string(buffer_size);
    MS_CHECK_TRUE_MSG(ret == ACL_SUCCESS, Status(kLiteDeviceDataError, "Malloc device buffer failed!"),
                      error_msg.c_str());
  } else {
    ret = CALL_ASCEND_API(aclrtMallocHost, data_mem_buffer, buffer_size);
    std::string error_msg = "Malloc host buffer failed, buffer size " + std::to_string(buffer_size);
    MS_CHECK_TRUE_MSG(ret == ACL_SUCCESS, Status(kLiteHostDataError, "Malloc host buffer failed!"), error_msg.c_str());
  }
  is_weight_input_from_external_device_mem_ = false;
  return kSuccess;
}

Status ModelProcess::CreateDataBuffer(void **data_mem_buffer, size_t buffer_size, aclmdlDataset *dataset,
                                      bool use_existing_mem) {
  MS_CHECK_TRUE_MSG(data_mem_buffer != nullptr, Status(kLiteNullptr, "Data mem buffer is nullptr!"),
                    "Data mem buffer is nullptr!");
  auto free_data_buffer = [this](void *buf) {
    (void)(is_run_on_device_ ? CALL_ASCEND_API(aclrtFreeHost, buf) : CALL_ASCEND_API(aclrtFree, buf));
  };
  if (!use_existing_mem) {
    auto status = AllocDataBufferMem(data_mem_buffer, buffer_size);
    if (status != kSuccess) {
      return status;
    }
  } else {
    if (*data_mem_buffer == nullptr) {
      MS_LOG(ERROR) << "Existing memory buffer is nullptr, buffer size " << buffer_size;
      return Status(kLiteAclInitFailed, "data_mem_buffer is nullptr");
    }
    MS_LOG(INFO) << "Use existing memory, skip malloc, buffer size: " << buffer_size;
    is_weight_input_from_external_device_mem_ = true;
  }
  auto data_buffer = CALL_ASCEND_API(aclCreateDataBuffer, *data_mem_buffer, buffer_size);
  if (data_buffer == nullptr) {
    MS_LOG(ERROR) << "Create Data Buffer failed";
    if (!use_existing_mem && *data_mem_buffer != nullptr) {
      free_data_buffer(*data_mem_buffer);
    }
    return Status(kLiteAclInitFailed, "Create Data Buffer failed");
  }
  auto ret = CALL_ASCEND_API(aclmdlAddDatasetBuffer, dataset, data_buffer);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "add data buffer failed, ret: " << ret;
    if (!use_existing_mem && *data_mem_buffer != nullptr) {
      free_data_buffer(*data_mem_buffer);
    }
    CALL_ASCEND_API(aclDestroyDataBuffer, data_buffer);
    return Status(kLiteAclInitFailed, "aclmdlAddDatasetBuffer failed.");
  }
  return kSuccess;
}

void ModelProcess::DestroyInputsBuffer() {
  for (const auto &item : input_infos_) {
    if (item.device_data != nullptr) {
      if (!is_run_on_device_) {
        CALL_ASCEND_API(aclrtFree, item.device_data);
      } else {
        CALL_ASCEND_API(aclrtFreeHost, item.device_data);
      }
    }
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
    }
  }
  input_infos_.clear();

  if (inputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < CALL_ASCEND_API(aclmdlGetDatasetNumBuffers, inputs_); i++) {
    auto dataBuffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
    CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer);
  }
  CALL_ASCEND_API(aclmdlDestroyDataset, inputs_);
  inputs_ = nullptr;
}

void ModelProcess::DestroyOutputsBuffer() {
  if (!is_dynamic_output_) {
    for (const auto &item : output_infos_) {
      if (item.device_data != nullptr) {
        if (!is_run_on_device_) {
          CALL_ASCEND_API(aclrtFree, item.device_data);
        } else {
          CALL_ASCEND_API(aclrtFreeHost, item.device_data);
        }
      }
    }
  }
  output_infos_.clear();

  if (outputs_ == nullptr) {
    return;
  }
  for (size_t i = 0; i < CALL_ASCEND_API(aclmdlGetDatasetNumBuffers, outputs_); i++) {
    auto dataBuffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer);
  }
  CALL_ASCEND_API(aclmdlDestroyDataset, outputs_);
  outputs_ = nullptr;
}

Status ModelProcess::CreateModelOutputs() {
  if (!is_dynamic_output_) {
    for (size_t i = 0; i < output_infos_.size(); ++i) {
      const auto &output_info = output_infos_[i];
      auto host_data = malloc(output_info.buffer_size);
      MS_CHECK_TRUE_MSG(host_data != nullptr, Status(kLiteNullptr, "host_data is nullptr."), "Malloc data failed.");
      auto output = MSTensor(output_info.name, static_cast<DataType>(TransToDataType(output_info.data_type)),
                             output_info.dims, host_data, output_info.buffer_size);
      free(host_data);
      host_data = nullptr;
      model_outputs_.push_back(output);
    }
  }
  return kSuccess;
}

bool ModelProcess::AllocAndMapPhysicalMemory(size_t alloc_size) {
  aclrtMemLocation location = {static_cast<uint32_t>(options_->device_id), ACL_MEM_LOCATION_TYPE_DEVICE};
  aclrtPhysicalMemProp prop = {ACL_MEM_HANDLE_TYPE_NONE, ACL_MEM_ALLOCATION_TYPE_PINNED, ACL_HBM_MEM_HUGE, location};
  auto ret = CALL_ASCEND_API(aclrtMallocPhysical, &shareable_phy_addr_, alloc_size, &prop, 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtMallocPhysical failed! ret:" << ret;
    return false;
  }
  ret = CALL_ASCEND_API(aclrtMemExportToShareableHandle, shareable_phy_addr_, ACL_MEM_HANDLE_TYPE_NONE, 0,
                        &sharable_handle_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtMemExportToShareableHandle failed! ret:" << ret;
    return false;
  }
  std::vector<int> pids = lite::ConvertStringToIntVector(options_->pids);
  MS_CHECK_TRUE_MSG(!pids.empty(), false, "pids is empty!");
  ret = CALL_ASCEND_API(aclrtMemSetPidToShareableHandle, sharable_handle_, pids.data(), pids.size());
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set pid to shareable_handle failed! ret:" << ret;
    return false;
  }
  size_t alignment = 0;
  ret = CALL_ASCEND_API(aclrtReserveMemAddress, &multiprocess_weight_ptr_, alloc_size, alignment, nullptr, 1);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtReserveMemAddress failed! ret:" << ret;
    return false;
  }
  size_t offset = 0;
  ret = CALL_ASCEND_API(aclrtMapMem, multiprocess_weight_ptr_, alloc_size, offset, shareable_phy_addr_, 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtmapmem failed! ret:" << ret;
    return false;
  }
  return true;
}

bool ModelProcess::QueryModelSizeFromMem(const void *om_data, size_t om_data_size, size_t *work_size,
                                         size_t *weight_size) {
  auto acl_ret = CALL_ASCEND_API(aclmdlQuerySizeFromMem, om_data, om_data_size, work_size, weight_size);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
    return false;
  }
  return true;
}

bool ModelProcess::AllocWorkMemory(size_t work_size) {
  if (work_size == 0) {
    work_ptr_ = nullptr;
  } else {
    auto acl_ret = CALL_ASCEND_API(aclrtMalloc, &(work_ptr_), work_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Call aclrtMalloc failed, err_code = " << acl_ret;
      return false;
    }
  }
  return true;
}

bool ModelProcess::LoadModelFromMemWithMem(const void *om_data, size_t om_data_size, size_t work_size,
                                           size_t weight_size) {
  options_->share_weightspace = true;
  auto start_time = lite::GetTimeUs();
  auto acl_ret = CALL_ASCEND_API(aclmdlLoadFromMemWithMem, om_data, om_data_size, &model_id_, work_ptr_, work_size,
                                 multiprocess_weight_ptr_, weight_size);
  auto end_time = lite::GetTimeUs();
  MS_LOG(INFO) << "[init time] call aclmdlLoadFromMemWithMem cost " << (end_time - start_time) << " us";
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMemWithMem failed, ret = " << acl_ret;
    return false;
  }
  infer_id_ = model_id_;
  return true;
}

bool ModelProcess::MainProcess(const void *om_data, size_t om_data_size) {
  size_t work_size = 0;
  size_t weight_size = 0;
  if (!QueryModelSizeFromMem(om_data, om_data_size, &work_size, &weight_size)) {
    return false;
  }
  size_t alloc_size = ((weight_size / kGranularitySize) + 1) * kGranularitySize;
  if (!AllocAndMapPhysicalMemory(alloc_size)) {
    return false;
  }
  if (!AllocWorkMemory(work_size)) {
    return false;
  }
  return LoadModelFromMemWithMem(om_data, om_data_size, work_size, weight_size);
}

bool ModelProcess::SubProcess(const void *om_data, size_t om_data_size) {
  size_t work_size = 0;
  size_t weight_size = 0;
  if (!QueryModelSizeFromMem(om_data, om_data_size, &work_size, &weight_size)) {
    return false;
  }
  size_t alloc_size = ((weight_size / kGranularitySize) + 1) * kGranularitySize;
  auto ret = CALL_ASCEND_API(aclrtMemImportFromShareableHandle, options_->sharable_handle, options_->device_id,
                             &shareable_phy_addr_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "acl rt mem import from shareable handle failed! ret:" << ret
                  << " please Keep the main process running, otherwise the shared memory will become invalid!";
    return lite::RET_ERROR;
  }
  size_t alignment = 0;
  CALL_ASCEND_API(aclrtReserveMemAddress, &(multiprocess_weight_ptr_), alloc_size, alignment, nullptr, ACL_HBM_MEM);
  size_t offset = 0;
  ret = CALL_ASCEND_API(aclrtMapMem, multiprocess_weight_ptr_, alloc_size, offset, shareable_phy_addr_, 0);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclrtmapmem failed! weight size:" << alloc_size;
    return lite::RET_ERROR;
  }
  if (!AllocWorkMemory(work_size)) {
    return false;
  }
  return LoadModelFromMemWithMem(om_data, om_data_size, work_size, weight_size);
}

Status ModelProcess::ShareMemProcess(const void *om_data, size_t om_data_size) {
  MS_CHECK_TRUE_MSG(options_->is_bundle_model == false,
                    Status(kLiteAclInitFailed, "Update weight model don't support mem share!"),
                    "Update weight model don't support mem share!");
  MS_CHECK_TRUE_MSG(om_data != nullptr, Status(kLiteNullptr, "om_data is nullptr!"), "om_data is nullptr!");
  MS_LOG(INFO) << "using sharing mem by model group.";
  size_t work_size = 0;
  size_t weight_size = 0;
  auto acl_ret = CALL_ASCEND_API(aclmdlQuerySizeFromMem, om_data, om_data_size, &work_size, &weight_size);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlQuerySizeFromMem failed, ret = " << acl_ret;
    return Status(kLiteAclInitFailed, "Call aclmdlQuerySizeFromMem failed.");
  }
  if (options_->share_workspace) {
    auto ptr = AclSharedMemoryManager::GetInstance().ShareWorkspaceProcess(work_size, weight_size, options_);
    weight_ptr_ = ptr.first;
    work_ptr_ = ptr.second;
    if (weight_ptr_ == nullptr && work_ptr_ == nullptr) {
      MS_LOG(ERROR) << "ShareWorkspaceProcess failed! work_size:" << work_size;
      return Status(kLiteAclInitFailed, "ShareWorkspaceProcess failed!");
    }
    is_sharing_workspace_ = true;
  } else if (options_->share_weightspace) {
    auto ptr = AclSharedMemoryManager::GetInstance().ShareWeightspaceProcess(work_size, options_);
    weight_ptr_ = ptr.first;
    work_ptr_ = ptr.second;
    if (weight_ptr_ == nullptr && work_ptr_ == nullptr) {
      MS_LOG(ERROR) << "ShareWeightspaceProcess failed! work_size:" << work_size;
      return Status(kLiteAclInitFailed, "ShareWeightspaceProcess failed!");
    }
  } else if (options_->share_weightspace_workspace) {
    auto ptr = AclSharedMemoryManager::GetInstance().ShareWorkspaceAndWeightspaceProcess(work_size, options_);
    weight_ptr_ = ptr.first;
    work_ptr_ = ptr.second;
    if (weight_ptr_ == nullptr && work_ptr_ == nullptr) {
      MS_LOG(ERROR) << "ShareWorkspaceAndWeightspaceProcess failed! work_size:" << work_size;
      return Status(kLiteAclInitFailed, "ShareWorkspaceAndWeightspaceProcess failed!");
    }
    is_sharing_workspace_ = true;
  } else {
    MS_LOG(ERROR) << "Please specify the sharing type!";
    return Status(kLiteParamInvalid, "Please specify the sharing type!");
  }
  auto start_time = lite::GetTimeUs();
  acl_ret = CALL_ASCEND_API(aclmdlLoadFromMemWithMem, om_data, om_data_size, &model_id_, work_ptr_, work_size,
                            weight_ptr_, weight_size);
  auto end_time = lite::GetTimeUs();
  auto cost = end_time - start_time;
  MS_LOG(INFO) << "[init time] call aclmdlLoadFromMemWithMem cost " << cost << " us";
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMemWithMem failed, ret = " << acl_ret;
    return Status(kLiteAclInitFailed, "Call aclmdlLoadFromMemWithMem failed!");
  }
  infer_id_ = model_id_;
  return kSuccess;
}

Status ModelProcess::LoadModelForUpdateWeight(const void *om_data, size_t om_data_size) {
  auto start_time = lite::GetTimeUs();
  auto acl_ret = CALL_ASCEND_API(aclmdlBundleLoadFromMem, om_data, om_data_size, &model_id_);
  auto end_time = lite::GetTimeUs();
  auto cost = end_time - start_time;
  MS_LOG(INFO) << "[init time] call aclmdlBundleLoadFromMem cost " << cost << " us";
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
    return kLiteError;
  }
  acl_ret = CALL_ASCEND_API(aclmdlBundleGetModelId, model_id_, 0, &infer_id_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlBundleGetModelId failed, model id: 0, ret = " << acl_ret << "!";
    return kLiteError;
  }
  acl_ret = CALL_ASCEND_API(aclmdlBundleGetModelId, model_id_, 1, &update_id_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlBundleGetModelId failed, model id: 1, ret = " << acl_ret << "!";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelProcess::LoadModelByMode(const void *om_data, size_t om_data_size) {
  MS_LOG(INFO) << "options->pids:" << options_->pids << ",options_->sharable_handle:" << options_->sharable_handle;
  if (options_->is_bundle_model) {
    auto status = LoadModelForUpdateWeight(om_data, om_data_size);
    if (status != kSuccess) {
      MS_LOG(ERROR) << "LoadModelForUpdateWeight failed.";
    }
    return status;
  }
  if (options_->pids != "") {
    if (!MainProcess(om_data, om_data_size)) {
      MS_LOG(ERROR) << "Main process failed!";
      return Status(kLiteError, "Main process failed!");
    }
    return kSuccess;
  }
  if (options_->sharable_handle != 0) {
    if (!SubProcess(om_data, om_data_size)) {
      MS_LOG(ERROR) << "Sub process failed!";
      return Status(kLiteError, "Sub process failed!");
    }
    return kSuccess;
  }
  auto start_time = lite::GetTimeUs();
  auto acl_ret = CALL_ASCEND_API(aclmdlLoadFromMem, om_data, om_data_size, &model_id_);
  auto end_time = lite::GetTimeUs();
  MS_LOG(INFO) << "[init time] call aclmdlLoadFromMem cost " << (end_time - start_time) << " us";
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Call aclmdlLoadFromMem failed, ret = " << acl_ret;
    return Status(kLiteAclInitFailed, "Call aclmdlLoadFromMem failed.");
  }
  infer_id_ = model_id_;
  return kSuccess;
}

Status ModelProcess::Load(const void *om_data, size_t om_data_size) {
  if (loaded_) {
    return kSuccess;
  }
  MS_LOG(INFO) << "multi_model_sharing_mem_prepare: " << options_->multi_model_sharing_mem_prepare
               << ", multi_model_sharing_mem: " << options_->multi_model_sharing_mem;
  if (options_->multi_model_sharing_mem_prepare) {
    MS_CHECK_TRUE_MSG(!options_->is_bundle_model, Status(kLiteError, "Update weight model don't support mem share!"),
                      "Update weight model don't support mem share!");
    auto status = AclSharedMemoryManager::GetInstance().PrepareMutiModelShare(om_data, om_data_size, options_);
    if (status == kSuccess) {
      MS_LOG(DEBUG) << "shared memory prepare success.";
    }
    return status;
  }
  Status status =
    options_->multi_model_sharing_mem ? ShareMemProcess(om_data, om_data_size) : LoadModelByMode(om_data, om_data_size);
  if (status != kSuccess) {
    return status;
  }
  status = PreInitModelResource();
  if (status != kSuccess) {
    (void)CALL_ASCEND_API(aclmdlUnload, model_id_);
    MS_LOG(ERROR) << "Pre init model resource failed.";
    return status;
  }
  loaded_ = true;
  status = CreateModelOutputs();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Cannot pre-allocate buffer for tensor in static shape.";
    return status;
  }
  MS_LOG(INFO) << "Load model success.";
  return kSuccess;
}

void ModelProcess::ReleaseModelResources() {
  DestroyInputsBuffer();
  DestroyOutputsBuffer();
  DestoryUpdateWeightBuffer();
  if (options_->share_workspace && weight_ptr_ != nullptr) {
    CALL_ASCEND_API(aclrtFree, weight_ptr_);
    weight_ptr_ = nullptr;
  }
  if (options_->share_weightspace && work_ptr_ != nullptr) {
    CALL_ASCEND_API(aclrtFree, work_ptr_);
    work_ptr_ = nullptr;
  }
  if (multiprocess_weight_ptr_ != nullptr) {
    CALL_ASCEND_API(aclrtUnmapMem, multiprocess_weight_ptr_);
    CALL_ASCEND_API(aclrtReleaseMemAddress, multiprocess_weight_ptr_);
    multiprocess_weight_ptr_ = nullptr;
  }
  if (shareable_phy_addr_ != nullptr) {
    CALL_ASCEND_API(aclrtFreePhysical, shareable_phy_addr_);
    shareable_phy_addr_ = nullptr;
  }
}

Status ModelProcess::UnLoad() {
  if (!loaded_) {
    MS_LOG(INFO) << "Model has not been loaded or has been unloaded";
    return kSuccess;
  }
  loaded_ = false;
  auto ret = options_->is_bundle_model ? CALL_ASCEND_API(aclmdlBundleUnload, model_id_)
                                       : CALL_ASCEND_API(aclmdlUnload, model_id_);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
    return Status(kLiteAclInitFailed, "Unload model failed!");
  }
  if (model_desc_ != nullptr) {
    ret = CALL_ASCEND_API(aclmdlDestroyDesc, model_desc_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Unload model failed, ret = " << ret;
      return Status(kLiteAclInitFailed, "Unload model failed!");
    }
    model_desc_ = nullptr;
  }
  ReleaseModelResources();
  if (stream_ != nullptr) {
    ret = CALL_ASCEND_API(aclrtDestroyStream, stream_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Destroy stream failed";
      return Status(kLiteAclInitFailed, "Destroy stream failed");
    }
    stream_ = nullptr;
  }
  if (exec_config_handle_ != nullptr) {
    ret = CALL_ASCEND_API(aclmdlDestroyExecConfigHandle, exec_config_handle_);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Destroy exec config handle failed";
      return Status(kLiteAclInitFailed, "Destroy exec config handle failed");
    }
    exec_config_handle_ = nullptr;
  }
  MS_LOG(INFO) << "End unload model " << model_id_;
  return kSuccess;
}

bool ModelProcess::IsDynamicShape() { return IsDynamicBatchSize() || IsDynamicImageSize() || IsDynamicDims(); }

bool ModelProcess::IsDynamicBatchSize() { return !dynamic_shape_options_.batch_size.empty(); }

bool ModelProcess::IsDynamicImageSize() { return !dynamic_shape_options_.image_size.empty(); }

bool ModelProcess::IsDynamicDims() { return dynamic_shape_options_.dynamic_dims.second != 0; }

Status ModelProcess::ResetInputSize(const std::vector<ShapeVector> &new_shapes) {
  for (size_t index = 0; index < new_shapes.size(); index++) {
    std::vector<int64_t> shape = new_shapes[index];
    size_t elem_count = 1;
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= shape[i];
    }
    input_infos_[index].dims = shape;
    auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, index);
    if (data_type == aclDataType::ACL_DT_UNDEFINED) {
      MS_LOG(ERROR) << "ModelProcess ResetInputSize ERROR" << data_type;
      return Status(kLiteAclInitFailed, "Get model input data type is invalid.");
    }
    auto new_buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
    if (!is_dynamic_input_) {
      input_infos_[index].buffer_size = new_buffer_size;
    } else if (new_buffer_size > input_infos_[index].buffer_size) {
      is_dynamic_resize_input_ = true;
      input_infos_[index].buffer_size = new_buffer_size;
    }
  }
  return kSuccess;
}

Status ModelProcess::ResetOutputSize() {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr, Status(kLiteUninitializedObj, "Model desc is nullptr."),
                    "Model desc is nullptr.");
  aclDataType data_type;
  aclError ret;
  size_t output_size = CALL_ASCEND_API(aclmdlGetNumOutputs, model_desc_);
  for (size_t index = 0; index < output_size; index++) {
    struct aclmdlIODims dims;
    ret = CALL_ASCEND_API(aclmdlGetCurOutputDims, model_desc_, index, &dims);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "get output dim error.";
      return Status(kLiteAclInitFailed, "get output dim error!");
    }
    std::vector<int64_t> shape(dims.dims, dims.dims + dims.dimCount);
    size_t elem_count = 1;
    for (size_t i = 0; i < dims.dimCount; i++) {
      if (dims.dims[i] < 0) {
        elem_count = 0;
        break;
      }
      elem_count *= dims.dims[i];
    }
    data_type = CALL_ASCEND_API(aclmdlGetOutputDataType, model_desc_, index);
    if (data_type == aclDataType::ACL_DT_UNDEFINED) {
      MS_LOG(ERROR) << "ModelProcess ResetOutputSize ERROR" << data_type;
      return Status(kLiteAclInitFailed, "Get model output data type is invalid.");
    }
    output_infos_[index].dims = shape;
    output_infos_[index].buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
  }
  return kSuccess;
}

Status ModelProcess::Resize(const std::vector<ShapeVector> &new_shapes) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return Status(kLiteUninitializedObj, "Model has not been loaded!");
  }
  auto input_shapes = GetInputShape();
  if (input_shapes.size() != new_shapes.size()) {
    MS_LOG(ERROR) << "Invalid new input size " << new_shapes.size() << ", expect input size " << input_shapes.size();
    return Status(kLiteInputParamInvalid, "Invalid new input size!");
  }
  bool input_shape_changed = false;
  for (size_t i = 0; i < new_shapes.size(); i++) {
    auto new_shape = new_shapes[i];
    auto has_negtive_shape = std::any_of(new_shape.begin(), new_shape.end(), [](auto dim) { return dim < 0; });
    if (has_negtive_shape) {
      MS_LOG(ERROR) << "New shape of input " << i << " cannot be dynamic, new shape: " << new_shape;
      return Status(kLiteInputParamInvalid, "shape is wrong!");
    }
    if (input_shapes[i] != new_shape) {
      input_shape_changed = true;
    }
  }
  if (!input_shape_changed) {
    return kSuccess;
  }
  if (is_dynamic_input_) {
    return ResizeDynamicInputShape(new_shapes);
  }
  if (is_dynamic_shape_range_) {
    return ResizeDynamicInputShapeRange(new_shapes);
  }
  if (!IsDynamicShape()) {
    MS_LOG(ERROR) << "Not support dynamic input";
    return Status(kLiteInputParamInvalid, "Not support dynamic input.");
  }
  auto status = ResizeDynamicBatchAndImageSize(new_shapes);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Resize dynamic batch and image size failed";
    return status;
  }

  return kSuccess;
}

Status ModelProcess::ResizeDynamicInputShape(const std::vector<ShapeVector> &new_shapes) {
  MS_LOG(INFO) << "Start to resize dynamic input shape";
  // If it is not the first time to resize input shape, the old addr need to be free
  ResetInputSize(new_shapes);
  FreeResourceInput(input_infos_);
  if (is_dynamic_resize_input_) {
    inputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
    if (inputs_ == nullptr) {
      MS_LOG(ERROR) << "Create input dataset failed";
      return Status(kLiteNullptr, "Create input dataset failed.");
    }
  }
  for (size_t i = 0; i < new_shapes.size(); ++i) {
    if (is_dynamic_resize_input_) {
      void *data_buf = nullptr;
      auto status = CreateDataBuffer(&data_buf, input_infos_[i].buffer_size, inputs_);
      if (status != kSuccess) {
        MS_LOG(ERROR) << "Add input data buffer failed";
        return status;
      }
      auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
      if (data_type == aclDataType::ACL_DT_UNDEFINED) {
        MS_LOG(ERROR) << "ModelProcess ResizeDynamicInputShape ERROR" << data_type;
        return Status(kLiteAclInitFailed, "Get model input data type is invalid.");
      }
      std::string input_name = CALL_ASCEND_API(aclmdlGetInputNameByIndex, model_desc_, i);
      if (input_name.empty()) {
        MS_LOG(ERROR) << "Get name of input " << i << " failed.";
        return Status(kLiteAclInitFailed, "aclmdlGetInputNameByIndex failed, input_name is empty.");
      }
      MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
      input_infos_[i].cur_device_data = data_buf;
      input_infos_[i].device_data = data_buf;
      input_infos_[i].data_type = data_type;
      input_infos_[i].name = input_name;
      auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
      input_infos_[i].dynamic_acl_data_buffer = data_buffer;
    }

    aclTensorDesc *input_desc =
      CALL_ASCEND_API(aclCreateTensorDesc, ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
    auto ret = CALL_ASCEND_API(aclmdlSetDatasetTensorDesc, inputs_, input_desc, i);
    input_infos_[i].dynamic_acl_tensor_desc = input_desc;
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Acl set dataset tensor desc failed";
      return Status(kLiteInputParamInvalid, "Acl set dataset tensor desc failed");
    }
  }
  is_dynamic_resize_input_ = false;
  MS_LOG(INFO) << "Resize dynamic input shape success";
  return kSuccess;
}

Status ModelProcess::ResizeDynamicInputShapeRange(const std::vector<ShapeVector> &new_shapes) {
  MS_LOG(INFO) << "Start to resize dynamic input shape range";
  for (size_t i = 0; i < new_shapes.size(); ++i) {
    std::vector<int64_t> shape = new_shapes[i];
    auto buffer_size = CALL_ASCEND_API(aclmdlGetInputSizeByIndex, model_desc_, i);
    auto data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_desc_, i);
    if (data_type == aclDataType::ACL_DT_UNDEFINED) {
      MS_LOG(ERROR) << "ModelProcess ResizeDynamicInputShapeRange ERROR" << data_type;
      return Status(kLiteAclInitFailed, "Get model input data type is invalid.");
    }
    size_t elem_count = 1;
    for (size_t j = 0; j < shape.size(); ++j) {
      if (shape[j] < 0) {
        MS_LOG(ERROR) << "The resize shape has the dim less than 0";
        return Status(kLiteInputParamInvalid, "The resize shape has the dim less than 0.");
      }
      elem_count *= shape[j];
    }
    auto new_buffer_size = elem_count * CALL_ASCEND_API(aclDataTypeSize, data_type);
    if (new_buffer_size > buffer_size) {
      MS_LOG(ERROR) << "The resize shape is over shape range";
      return Status(kLiteInputParamInvalid, "The resize shape is over shape range.");
    }
    input_infos_[i].dims = shape;
    aclTensorDesc *input_desc =
      CALL_ASCEND_API(aclCreateTensorDesc, ACL_FLOAT, new_shapes[i].size(), &new_shapes[i][0], ACL_FORMAT_NCHW);
    auto ret = CALL_ASCEND_API(aclmdlSetDatasetTensorDesc, inputs_, input_desc, i);
    (void)CALL_ASCEND_API(aclDestroyTensorDesc, input_desc);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Acl set dataset tensor desc failed";
      return Status(kLiteInputParamInvalid, "Acl set dataset tensor desc failed");
    }
  }
  MS_LOG(INFO) << "Resize dynamic input shape range success";
  return kSuccess;
}
Status ModelProcess::SetDynamicBatchConfig(size_t index) {
  int32_t batch_size = 0;
  auto status = dyn_shape_proc_.CheckAndGetBatchSize(cur_input_shapes_, &batch_size);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to check batch size";
    return status;
  }
  MS_LOG(INFO) << "Set Batch size(" << batch_size << ") of input " << index << ".";
  auto ret = CALL_ASCEND_API(aclmdlSetDynamicBatchSize, infer_id_, inputs_, index, batch_size);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set dynamic batch size failed, model_id is " << infer_id_;
    return Status(kLiteInputParamInvalid, "aclmdlSetDynamicBatchSize failed.");
  }
  return kSuccess;
}

Status ModelProcess::SetDynamicImageConfig(size_t index) {
  int32_t height = 0;
  int32_t width = 0;
  auto status = dyn_shape_proc_.CheckAndGetImageSize(cur_input_shapes_, &height, &width);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to check image size";
    return status;
  }
  MS_LOG(INFO) << "Set Image size(" << height << "," << width << ") of input " << index << ".";
  auto ret = CALL_ASCEND_API(aclmdlSetDynamicHWSize, infer_id_, inputs_, index, height, width);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Set Image size(H, W) of input " << index << " failed, model_id is " << infer_id_;
    return Status(kLiteInputParamInvalid, "aclmdlSetDynamicHWSize failed.");
  }
  return kSuccess;
}

Status ModelProcess::SetDynamicDimsConfig(size_t index) {
  aclmdlIODims dynamic_dims;
  auto status = dyn_shape_proc_.CheckAndGetDynamicDims(cur_input_shapes_, &dynamic_dims);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "CheckAndGetDynamicDims failed.";
    return status;
  }
  auto ret = CALL_ASCEND_API(aclmdlSetInputDynamicDims, infer_id_, inputs_, index, &dynamic_dims);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "aclmdlSetInputDynamicDims failed.";
    return Status(kLiteInputParamInvalid, "aclmdlSetInputDynamicDims failed.");
  }
  return kSuccess;
}

Status ModelProcess::ResizeDynamicBatchAndImageSize(const std::vector<ShapeVector> &new_shapes) {
  MS_CHECK_TRUE_MSG(model_desc_ != nullptr && inputs_ != nullptr, Status(kLiteNullptr, "Model desc is nullptr."),
                    "Model desc is nullptr.");
  size_t index;
  auto ret = CALL_ASCEND_API(aclmdlGetInputIndexByName, model_desc_, ACL_DYNAMIC_TENSOR_NAME, &index);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Get index of dynamic tensor failed";
    return Status(kLiteAclInitFailed, "aclmdlGetInputIndexByName failed.");
  }
  cur_input_shapes_ = new_shapes;
  Status status;
  if (IsDynamicBatchSize()) {
    status = SetDynamicBatchConfig(index);
  } else if (IsDynamicImageSize()) {
    status = SetDynamicImageConfig(index);
  } else if (IsDynamicDims()) {
    status = SetDynamicDimsConfig(index);
  } else {
    MS_LOG(ERROR) << "Not support dynamic input";
    return Status(kLiteInputParamInvalid, "Not support dynamic input.");
  }
  if (status != kSuccess) {
    return status;
  }
  status = ResetInputSize(new_shapes);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Reset input size failed";
    return status;
  }
  status = ResetOutputSize();
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Reset output size failed";
  }
  return status;
}

Status ModelProcess::CheckInputTensors(const std::vector<MSTensor> &input_tensors) {
  if (data_input_num_ != input_tensors.size()) {
    MS_LOG(ERROR) << "Expect input size to be " << data_input_num_ << ", but got " << input_tensors.size();
    return Status(kLiteInputParamInvalid, "The given inputs size != model inputs size expected.");
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto &tensor = input_tensors[i];
    auto &info = input_infos_[i];
    if (tensor.Shape() != info.dims) {
      MS_LOG(WARNING) << "Note: input " << i << " shape not match, required " << info.dims << ", given "
                      << tensor.Shape() << "."
                      << "Please check input shape has been modified by DVPP method.";
    }
    if (static_cast<enum TypeId>(tensor.DataType()) != TransToDataType(info.data_type)) {
      MS_LOG(ERROR) << "Note: input " << i << " data type not match, required "
                    << static_cast<int>(TransToDataType(info.data_type)) << ", given "
                    << static_cast<int>(tensor.DataType());
      return Status(kLiteInputParamInvalid, "Input data type is wrong.");
    }
    void *device_data_addr = static_cast<MSTensor>(tensor).GetDeviceData();
    auto host_data_addr = tensor.Data().get();
    if (device_data_addr != nullptr) {
      if (!is_dynamic_input_ && !is_dynamic_shape_range_ && tensor.DataSize() != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << tensor.DataSize();
        return Status(kLiteDataSizeError, "Input data size is wrong.");
      }
    } else if (host_data_addr != nullptr) {
      if (!is_dynamic_input_ && !is_dynamic_shape_range_ && tensor.DataSize() != info.buffer_size) {
        MS_LOG(ERROR) << "Input " << i << " data size not match, required size " << info.buffer_size << ", given count "
                      << tensor.DataSize();
        return Status(kLiteDataSizeError, "Input data size is wrong.");
      }
    } else {
      MS_LOG(ERROR) << "Failed to get data from input " << i;
      return Status(kLiteInputParamInvalid, "Failed to get data from input.");
    }
  }
  return kSuccess;
}

void *ModelProcess::GetInputBuffer(size_t i, const MSTensor &input) {
  auto &info = input_infos_[i];
  auto device_data_addr = static_cast<MSTensor>(input).GetDeviceData();
  auto host_data_addr = const_cast<void *>(input.Data().get());
  if (device_data_addr != nullptr) {
    auto input_device_id = input.GetDeviceId();
    if (input_device_id == static_cast<int>(device_id_)) {
      return device_data_addr;
    }
    auto data_copy_size = input.DataSize();
    auto copy_result = allocator_->CopyDeviceDataToDevice(device_data_addr, info.device_data, data_copy_size,
                                                          info.buffer_size, input_device_id, device_id_);
    MS_CHECK_TRUE_MSG(copy_result == kSuccess, nullptr, "Copy input data from device to current device failed.");
    return info.device_data;
  }
  auto data = host_data_addr;
  auto size = input.DataSize();
  if (!is_run_on_device_) {
    auto ret = AclrtMemcpy(info.device_data, info.buffer_size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, src input size: " << size
                    << ", dst device buffer size: " << info.buffer_size;
      return nullptr;
    }
    return info.device_data;
  }
  return data;
}

Status ModelProcess::CheckAndInitInput(const std::vector<MSTensor> &inputs) {
  MS_CHECK_TRUE_MSG(allocator_ != nullptr, Status(kLiteNullptr, "allocator_ is nullptr!"), "allocator_ is nullptr!");
  auto status = CheckInputTensors(inputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Check input tensor failed.";
    return status;
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &info = input_infos_[i];
    auto input = inputs[i];
    auto *input_buffer = GetInputBuffer(i, input);
    if (input_buffer == nullptr) {
      return Status(kLiteDataSizeError, "input data size is wrong!");
    }
    auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, inputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of input " << i;
      return Status(kLiteDeviceDataError, "Failed to get dataset buffer");
    }
    auto ret = CALL_ASCEND_API(aclUpdateDataBuffer, data_buffer, input_buffer, info.buffer_size);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of input " << i << ", buffer size: " << info.buffer_size
                    << ", input shape: " << input.Shape();
      return Status(kLiteDeviceDataError, "Failed to update Data Buffer of input.");
    }
  }
  return kSuccess;
}

Status ModelProcess::CheckAndInitOutput(const std::vector<MSTensor> *outputs) {
  aclError ret;
  if (!outputs->empty() && outputs->size() != output_infos_.size()) {
    MS_LOG(ERROR) << "outputs size wrong.";
    return Status(kLiteOutputParamInvalid, "outputs size wrong.");
  }
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    void *output_device_buffer = nullptr;
    output_device_buffer = nullptr;  // in dynamic output shape, setting nullptr allows acl to alloc memory
    auto output_device_buffer_size = 0;
    if (outputs->size() > i) {
      auto &user_output = const_cast<std::vector<MSTensor> *>(outputs)->at(i);
      if (user_output.GetDeviceId() == device_id_ && user_output.GetDeviceData()) {
        output_device_buffer = user_output.GetDeviceData();
        output_device_buffer_size = user_output.DataSize();
      }
    }
    auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of output " << i;
      return Status(kLiteDeviceDataError, "Failed to get dataset buffer of output.");
    }
    ret = CALL_ASCEND_API(aclUpdateDataBuffer, data_buffer, output_device_buffer, output_device_buffer_size);
    if (ret != ACL_SUCCESS) {
      return Status(kLiteDeviceDataError, "Failed to update data buffer of output.");
    }
  }
  return kSuccess;
}

Status GetTensorDescDim(aclTensorDesc *tensor_info, size_t j, int64_t *dim) {
  MS_CHECK_TRUE_MSG(tensor_info != nullptr, kLiteError, "tensor_info is nullptr");
  MS_CHECK_TRUE_MSG(dim != nullptr, kLiteError, "dim is nullptr");

  if (HAS_ASCEND_API(aclGetTensorDescDimV2)) {
    auto ret = CALL_ASCEND_API(aclGetTensorDescDimV2, tensor_info, j, dim);
    MS_CHECK_TRUE_MSG(ret == ACL_SUCCESS, kLiteError, "Get tensor desc dim failed");
  } else if (HAS_ASCEND_API(aclGetTensorDescDim)) {
    *dim = CALL_ASCEND_API(aclGetTensorDescDim, tensor_info, j);
    // -1 means tensor desc or index is invalid. see aclGetTensorDescDim API doc for more details.
    if (*dim == -1) {
      MS_LOG(ERROR) << "Get tensor desc dim failed";
      return kLiteError;
    }
  } else {
    MS_LOG(ERROR) << "Cannot find aclGetTensorDescDimV2 or aclGetTensorDescDim API.";
    return kLiteError;
  }
  // 0 is a invalid dim
  if (*dim == 0) {
    MS_LOG(ERROR) << "dim is invalid value. got: 0";
    return kLiteError;
  }
  return kSuccess;
}

Status ModelProcess::ResetDynamicOutputTensor(const std::vector<MSTensor> *outputs) {
  dyn_out_sys_buf_addr_.clear();
  FreeResourceOutput(&output_infos_, outputs);
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto &output_info = output_infos_[i];

    // get actual output tensor info
    aclTensorDesc *tensor_info = CALL_ASCEND_API(aclmdlGetDatasetTensorDesc, outputs_, i);
    size_t output_desc_size = CALL_ASCEND_API(aclGetTensorDescSize, tensor_info);
    aclDataBuffer *data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    void *acl_device_data = CALL_ASCEND_API(aclGetDataBufferAddr, data_buffer);
    size_t dim_nums = CALL_ASCEND_API(aclGetTensorDescNumDims, tensor_info);
    ShapeVector shape;
    for (size_t j = 0; j < dim_nums; ++j) {
      int64_t shape_j;
      auto ret = GetTensorDescDim(tensor_info, j, &shape_j);
      if (ret != kSuccess) {
        MS_LOG(ERROR) << "Get tensor desc dim failed, output index: " << i << ", dim index: " << j;
        return ret;
      }
      shape.emplace_back(shape_j);
    }
    output_info.device_data = acl_device_data;
    output_info.cur_device_data = acl_device_data;
    output_info.buffer_size = output_desc_size;
    output_info.malloc_buffer_size = output_desc_size;
    output_info.dims = shape;
  }
  return kSuccess;
}

Status ModelProcess::ExecuteModel(uint32_t model_id, aclmdlDataset *input, aclmdlDataset *output) {
  aclError ret = ACL_SUCCESS;
  if (stream_ && exec_config_handle_) {
    ret = CALL_ASCEND_API(aclmdlExecuteV2, model_id, input, output, stream_, exec_config_handle_);
  } else {
    ret = CALL_ASCEND_API(aclmdlExecute, model_id, input, output);
  }
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Execute Model Failed, ret = " << ret << ", detail:" << CALL_ASCEND_API(aclGetRecentErrMsg);
    return kLiteError;
  }
  return kSuccess;
}

Status ModelProcess::ExecuteWithTiming() {
  uint64_t start_time = 0;
  auto env = std::getenv("GLOG_v");
  bool print_time = (env != nullptr && (env[0] == kINFOLogLevel || env[0] == kDEBUGLogLevel));
  if (print_time) {
    start_time = lite::GetTimeUs();
  }
  if (is_sharing_workspace_) {
    MS_LOG(DEBUG) << "Need to lock before aclmdlExecute.";
    AclSharedMemoryManager::GetInstance().Lock(options_->device_id);
  }
  auto model_ret = ExecuteModel(infer_id_, inputs_, outputs_);
  if (is_sharing_workspace_) {
    MS_LOG(DEBUG) << "Unlock after aclmdlExecute.";
    AclSharedMemoryManager::GetInstance().Unlock(options_->device_id);
  }
  if (print_time) {
    auto end_time = lite::GetTimeUs();
    MS_LOG(INFO) << "Model execute in " << end_time - start_time << " us";
  }
  return model_ret;
}

Status ModelProcess::ProcessPredictOutputs(const std::vector<MSTensor> *outputs) {
  if (is_dynamic_output_) {
    return ResetDynamicOutputTensor(outputs);
  }
  FreeResourceOutput(&output_infos_, outputs);
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    auto &output_info = output_infos_[i];
    auto *data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, outputs_, i);
    void *acl_device_data = CALL_ASCEND_API(aclGetDataBufferAddr, data_buffer);
    output_info.device_data = acl_device_data;
    output_info.cur_device_data = acl_device_data;
  }
  return kSuccess;
}

Status ModelProcess::PredictFromHost(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> *outputs) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded";
    return Status(kLiteUninitializedObj, "Model has not been loaded!");
  }
  auto status = CheckAndInitInput(inputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Check or init input failed";
    return status;
  }
  status = CheckAndInitOutput(outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Check output tensor failed";
    return status;
  }
  auto model_ret = ExecuteWithTiming();
  if (model_ret != kSuccess) {
    MS_LOG(ERROR) << "Execute Model Failed";
    return model_ret;
  }
  status = ProcessPredictOutputs(outputs);
  if (status != kSuccess) {
    return status;
  }
  status = GetOutputs(outputs);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Get outputs failed";
    return status;
  }
  FreeResourceOutput(&output_infos_, outputs);
  return kSuccess;
}

void *ModelProcess::GetWeightInputBuffer(size_t i, MSTensor &kernel_input) {
  auto &info = weight_input_infos_[i];
  auto device_data_addr = kernel_input.GetDeviceData();
  auto host_data_addr = kernel_input.Data().get();
  if (device_data_addr != nullptr) {
    auto input_device_id = kernel_input.GetDeviceId();
    if (input_device_id == static_cast<int>(device_id_)) {
      return device_data_addr;
    }
    auto data_copy_size = kernel_input.DataSize();
    auto copy_result = allocator_->CopyDeviceDataToDevice(device_data_addr, info.device_data, data_copy_size,
                                                          info.buffer_size, input_device_id, device_id_);
    MS_CHECK_TRUE_MSG(copy_result == kSuccess, nullptr, "Copy input data from device to current device failed!");
    return info.device_data;
  }
  auto data = host_data_addr;
  auto size = kernel_input.DataSize();
  if (size != info.buffer_size) {
    MS_LOG(ERROR) << "Buffer size: " << info.buffer_size << "!= input size: " << size
                  << ", current only support data type fp16!";
    return nullptr;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "Input data is null!";
    return nullptr;
  }
  if (!is_run_on_device_) {
    auto acl_ret = AclrtMemcpy(info.device_data, info.buffer_size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Acl memcpy input " << i << " data to device failed, src size: " << size
                    << ", dst buffer size: " << info.buffer_size << "!";
      return nullptr;
    }
    return info.device_data;
  }
  return const_cast<void *>(data);
}

bool ModelProcess::CreateWeightsInput(const std::vector<MSTensor> &kernel_inputs) {
  MS_CHECK_TRUE_MSG(weight_inputs_ != nullptr, false, "Weight inputs is nullptr!");
  MS_CHECK_TRUE_MSG(model_weight_desc_ != nullptr, false, "Weight desc is nullptr!");
  MS_CHECK_TRUE_MSG(allocator_ != nullptr, false, "allocator_ is nullptr!");
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_weight_desc_);
  if (input_size != kernel_inputs.size()) {
    MS_LOG(ERROR) << "variable weight num " << kernel_inputs.size() << "!="
                  << "variable node num " << input_size << "!";
    return false;
  }
  for (size_t i = 0; i < kernel_inputs.size(); ++i) {
    auto kernel_input = kernel_inputs.at(i);
    auto &info = weight_input_infos_[i];
    auto *input_buffer = GetWeightInputBuffer(i, kernel_input);
    if (input_buffer == nullptr) {
      return false;
    }
    auto data_buffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, weight_inputs_, i);
    if (data_buffer == nullptr) {
      MS_LOG(ERROR) << "Failed to get dataset buffer of input " << i;
      return false;
    }
    auto acl_ret = CALL_ASCEND_API(aclUpdateDataBuffer, data_buffer, input_buffer, info.buffer_size);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to update Data Buffer of input ";
      return false;
    }
  }
  return true;
}

void ModelProcess::DestoryUpdateWeightBuffer() {
  if (is_run_on_device_) {
    for (const auto &item : weight_input_infos_) {
      if (item.device_data != nullptr) {
        CALL_ASCEND_API(aclrtFreeHost, item.device_data);
      }
    }
  } else {
    for (const auto &item : weight_input_infos_) {
      if (item.device_data != nullptr && !is_weight_input_from_external_device_mem_) {
        CALL_ASCEND_API(aclrtFree, item.device_data);
      }
    }
  }
  weight_input_infos_.clear();
  if (weight_inputs_ != nullptr) {
    for (size_t i = 0; i < CALL_ASCEND_API(aclmdlGetDatasetNumBuffers, weight_inputs_); ++i) {
      aclDataBuffer *dataBuffer = CALL_ASCEND_API(aclmdlGetDatasetBuffer, weight_inputs_, i);
      (void)CALL_ASCEND_API(aclDestroyDataBuffer, dataBuffer);
    }
    (void)CALL_ASCEND_API(aclmdlDestroyDataset, weight_inputs_);
    weight_inputs_ = nullptr;
  }
  if (weight_outputs_ != nullptr) {
    MS_CHECK_TRUE_RET_VOID(weight_outputs_ != nullptr);
    (void)CALL_ASCEND_API(aclmdlDestroyDataset, weight_outputs_);
    weight_outputs_ = nullptr;
  }
  inited_weights_ = false;
  MS_LOG(INFO) << "Destroy weight input success.";
}

// for update weights
bool ModelProcess::InitUpdateWeightBuffer(const std::vector<MSTensor> &kernel_inputs) {
  weight_inputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (weight_inputs_ == nullptr) {
    MS_LOG(ERROR) << "Create input dataset failed";
    return false;
  }
  weight_outputs_ = CALL_ASCEND_API(aclmdlCreateDataset);
  if (weight_outputs_ == nullptr) {
    MS_LOG(ERROR) << "Create output dataset failed!";
    return false;
  }
  model_weight_desc_ = CALL_ASCEND_API(aclmdlCreateDesc);
  aclError acl_ret = CALL_ASCEND_API(aclmdlGetDesc, model_weight_desc_, update_id_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Read model desc failed, ret = " << acl_ret;
    return false;
  }
  size_t input_size = CALL_ASCEND_API(aclmdlGetNumInputs, model_weight_desc_);
  if (input_size != kernel_inputs.size()) {
    MS_LOG(ERROR) << "variable weight num " << kernel_inputs.size() << "!="
                  << "variable node num " << input_size << "!";
    return false;
  }
  bool all_device_tensor = std::all_of(kernel_inputs.begin(), kernel_inputs.end(), [](const MSTensor &tensor) {
    return (const_cast<MSTensor &>(tensor).GetDeviceData() != nullptr);
  });

  for (size_t i = 0; i < input_size; ++i) {
    auto kernel_input = kernel_inputs[i];
    auto shape = kernel_input.Shape();
    aclDataType data_type = CALL_ASCEND_API(aclmdlGetInputDataType, model_weight_desc_, i);
    if (data_type == aclDataType::ACL_DT_UNDEFINED) {
      MS_LOG(ERROR) << "ModelProcess InitUpdateWeightBuffer ERROR, data type is undefined";
      return false;
    }
    auto buffer_size = kernel_input.DataSize();
    void *data_mem_buffer = nullptr;
    if (all_device_tensor) {
      data_mem_buffer = kernel_input.GetDeviceData();
      MS_LOG(INFO) << "Use existing device memory for input " << i << ", buffer size: " << buffer_size;
      if (!CreateDataBuffer(&data_mem_buffer, buffer_size, weight_inputs_, true)) {
        MS_LOG(ERROR) << "Add input data buffer (use device mem) failed, buffer size " << buffer_size << "!";
        return false;
      }
    } else {
      MS_LOG(INFO) << "Malloc new device memory for host tensor input " << i << ", buffer size: " << buffer_size;
      if (!CreateDataBuffer(&data_mem_buffer, buffer_size, weight_inputs_)) {
        MS_LOG(ERROR) << "Add input data buffer (malloc device mem) failed, buffer size " << buffer_size << "!";
        return false;
      }
    }

    std::string input_name = CALL_ASCEND_API(aclmdlGetInputNameByIndex, model_weight_desc_, i);
    aclFormat input_format = CALL_ASCEND_API(aclmdlGetInputFormat, model_weight_desc_, i);
    aclTensorDesc *desc = CALL_ASCEND_API(aclCreateTensorDesc, data_type, shape.size(), shape.data(), input_format);
    acl_ret = CALL_ASCEND_API(aclmdlSetDatasetTensorDesc, weight_inputs_, desc, i);
    (void)CALL_ASCEND_API(aclDestroyTensorDesc, desc);
    if (acl_ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "aclmdlSetDatasetTensorDesc failed, ret = " << acl_ret << "!";
      return false;
    }
    if (input_name.empty()) {
      MS_LOG(WARNING) << "Get name of input " << i << " failed!";
    }
    MS_LOG(INFO) << "Name of input " << i << " is " << input_name;
    weight_input_infos_.emplace_back(
      AclTensorInfo{data_mem_buffer, data_mem_buffer, buffer_size, buffer_size, data_type, shape, input_name});
  }
  inited_weights_ = true;
  MS_LOG(INFO) << "Create model inputs success, all device tensor: " << (all_device_tensor ? "yes" : "no");
  return true;
}

bool ModelProcess::UpdateWeights(const std::vector<MSTensor> &kernel_weights) {
  if (!loaded_) {
    MS_LOG(ERROR) << "Model has not been loaded!";
    return false;
  }
  bool all_device_tensor = std::all_of(kernel_weights.begin(), kernel_weights.end(), [](const MSTensor &tensor) {
    return (const_cast<MSTensor &>(tensor).GetDeviceData() != nullptr);
  });
  if (!inited_weights_ || (inited_weights_ && is_weight_input_from_external_device_mem_ && !all_device_tensor)) {
    if (!InitUpdateWeightBuffer(kernel_weights)) {
      DestoryUpdateWeightBuffer();
      MS_LOG(ERROR) << "Init weight input buffer failed!";
      return false;
    }
  }
  aclError acl_ret;
  bool ret = CreateWeightsInput(kernel_weights);
  if (!ret) {
    MS_LOG(ERROR) << "create Weights input failed!";
    return false;
  }
  acl_ret = CALL_ASCEND_API(aclmdlExecute, update_id_, weight_inputs_, weight_outputs_);
  if (acl_ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Run update weights failed! ret:" << acl_ret << "!";
    return false;
  }
  return true;
}

void ModelProcess::FreeResourceInput(std::vector<AclTensorInfo> acl_tensor_info) {
  for (const auto &item : acl_tensor_info) {
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
    }
    if (is_dynamic_resize_input_) {
      if (item.device_data != nullptr) {
        if (!is_run_on_device_) {
          CALL_ASCEND_API(aclrtFree, item.device_data);
        } else {
          CALL_ASCEND_API(aclrtFreeHost, item.device_data);
        }
      }
      if (item.dynamic_acl_data_buffer != nullptr) {
        CALL_ASCEND_API(aclDestroyDataBuffer, item.dynamic_acl_data_buffer);
      }
    }
  }
  if (is_dynamic_resize_input_) {
    CALL_ASCEND_API(aclmdlDestroyDataset, inputs_);
    inputs_ = nullptr;
  }
}

void ModelProcess::FreeResourceOutput(std::vector<AclTensorInfo> *acl_tensor_info,
                                      const std::vector<MSTensor> *outputs) {
  for (size_t i = 0; i < acl_tensor_info->size(); i++) {
    void *user_device_data = nullptr;
    if (outputs->size() > i) {
      auto &user_output = const_cast<std::vector<MSTensor> *>(outputs)->at(i);
      if (user_output.GetDeviceId() == device_id_ && user_output.GetDeviceData() != nullptr) {
        user_device_data = user_output.GetDeviceData();
      }
    }
    auto &item = (*acl_tensor_info)[i];
    if (item.device_data != nullptr && user_device_data != item.device_data) {
      MS_LOG(DEBUG) << "freeing device buffer at addr: " << item.device_data;
      if (!is_run_on_device_) {
        CALL_ASCEND_API(aclrtFree, item.device_data);
      } else {
        CALL_ASCEND_API(aclrtFreeHost, item.device_data);
      }
    }
    item.device_data = nullptr;
    if (item.dynamic_acl_data_buffer != nullptr) {
      CALL_ASCEND_API(aclDestroyDataBuffer, item.dynamic_acl_data_buffer);
      item.dynamic_acl_data_buffer = nullptr;
    }
    if (item.dynamic_acl_tensor_desc != nullptr) {
      CALL_ASCEND_API(aclDestroyTensorDesc, item.dynamic_acl_tensor_desc);
      item.dynamic_acl_tensor_desc = nullptr;
    }
  }
}

MSTensor ModelProcess::GetOutputWithZeroCopy(const std::vector<MSTensor> *outputs, size_t index) {
  auto &user_output = const_cast<std::vector<MSTensor> *>(outputs)->at(index);
  auto &output_info = output_infos_[index];
  if (user_output.GetDeviceData()) {
    if (user_output.GetDeviceId() != device_id_) {
      auto ret = allocator_->CopyDeviceDataToDevice(output_info.cur_device_data, user_output.GetDeviceData(),
                                                    user_output.DataSize(), output_info.buffer_size, device_id_,
                                                    user_output.GetDeviceId());
      MS_CHECK_TRUE_MSG(ret == kSuccess, MSTensor(nullptr), "Copy output data from device to current device failed!");
    }
  } else if (user_output.Data() != nullptr) {
    MS_CHECK_TRUE_MSG(user_output.DataSize() >= output_info.buffer_size, MSTensor(nullptr),
                      "Memcpy output failed, user buffer size is less than output size.");
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
    auto ret = AclrtMemcpy(user_output.MutableData(), user_output.DataSize(), output_info.cur_device_data,
                           output_info.buffer_size, kind);
    if (ret != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Memcpy output " << index << " from device to host failed, memory size "
                    << output_info.buffer_size << ", ret: " << ret;
      return MSTensor(nullptr);
    }
  } else {
    user_output = CreateOutputTensor(index);
    if (user_output.impl() == nullptr) {
      MS_LOG(ERROR) << "CreateOutputTensor failed!";
      return MSTensor(nullptr);
    }
  }
  user_output.SetShape(output_info.dims);
  return outputs->at(index);
}

MSTensor ModelProcess::CreateOutputTensor(size_t index) {
  MS_CHECK_TRUE_MSG(output_infos_.size() > index, MSTensor(nullptr), "index should less than size of output_infos_!");
  aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
  auto &output_info = output_infos_[index];
  auto output =
    MSTensor(output_info.name, static_cast<DataType>(TransToDataType(output_info.data_type)), {}, nullptr, 0);
  output.SetShape(output_info.dims);
  auto ret = AclrtMemcpy(output.MutableData(), output_info.buffer_size, output_info.cur_device_data,
                         output_info.buffer_size, kind);
  if (ret != ACL_SUCCESS) {
    MS_LOG(ERROR) << "Memcpy output " << index << " from device to host failed, memory size " << output_info.buffer_size
                  << ", ret: " << ret;
    return MSTensor(nullptr);
  }
  return output;
}

Status ModelProcess::GetOutputs(const std::vector<MSTensor> *outputs) {
  std::vector<MSTensor> new_outputs;
  for (size_t i = 0; i < output_infos_.size(); ++i) {
    if (!outputs->empty()) {
      auto tensor = GetOutputWithZeroCopy(outputs, i);
      MS_CHECK_TRUE_MSG(tensor.impl() != nullptr, kLiteError, "tensor impl is nullptr!");
      new_outputs.push_back(tensor);
      continue;
    }
    auto output = CreateOutputTensor(i);
    if (output.impl() == nullptr) {
      MS_LOG(ERROR) << "CreateOutputTensor failed!";
      return kLiteError;
    }
    new_outputs.push_back(output);
  }
  const_cast<std::vector<MSTensor> *>(outputs)->clear();
  const_cast<std::vector<MSTensor> *>(outputs)->insert(outputs->end(), new_outputs.begin(), new_outputs.end());
  return kSuccess;
}
}  // namespace mindspore
