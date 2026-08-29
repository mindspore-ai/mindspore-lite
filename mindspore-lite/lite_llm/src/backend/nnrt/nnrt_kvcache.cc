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

#include "backend/nnrt/nnrt_kvcache.h"

#include <cstring>

#include "backend/nnrt/nnrt_log.h"
#include "backend/nnrt/nnrt_wrapper.h"

namespace mslite {
namespace backend {
namespace nnrt {

namespace {
// Create one ION tensor of the fixed KV shape. Returns nullptr on failure.
NN_Tensor *CreateKvTensor(size_t device_id, OH_NNExecutor *executor, size_t input_index, int kv_heads, int max_len,
                          int head_dim) {
  const auto &api = NNRTWrapper::GetApi();
  NN_TensorDesc *desc = api.Executor_CreateInputTensorDesc(executor, input_index);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "CreateInputTensorDesc failed for KV input " << input_index;
    return nullptr;
  }
  int32_t shape[4] = {1, kv_heads, max_len, head_dim};
  if (api.TensorDesc_SetShape(desc, shape, 4) != 0 || api.TensorDesc_SetDataType(desc, kOhNnFloat16) != 0) {
    MS_LOG(ERROR) << "KV TensorDesc SetShape/SetDataType failed";
    api.TensorDesc_Destroy(&desc);
    return nullptr;
  }
  NN_Tensor *tensor = api.Tensor_Create(device_id, desc);
  api.TensorDesc_Destroy(&desc);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "OH_NNTensor_Create failed for KV input " << input_index;
    return nullptr;
  }
  return tensor;
}
}  // namespace

bool KVCacheManager::Alloc(int num_layers, int kv_heads, int max_len, int head_dim, size_t device_id,
                           OH_NNExecutor *executor) {
  if (num_layers <= 0 || kv_heads <= 0 || max_len <= 0 || head_dim <= 0 || executor == nullptr) {
    MS_LOG(ERROR) << "KVCacheManager::Alloc invalid params";
    return false;
  }
  Free();

  byte_count_per_layer_ = static_cast<size_t>(1) * kv_heads * max_len * head_dim * sizeof(uint16_t);
  key_tensors_.resize(num_layers, nullptr);
  value_tensors_.resize(num_layers, nullptr);

  // KV input index layout: after 7 fixed inputs, key/value caches are interleaved
  // (key_cache_0, value_cache_0, key_cache_1, value_cache_1, ...).
  const size_t kFixedInputs = 7;
  for (int i = 0; i < num_layers; ++i) {
    size_t key_idx = kFixedInputs + static_cast<size_t>(2 * i);
    size_t val_idx = kFixedInputs + static_cast<size_t>(2 * i) + 1;
    key_tensors_[i] = CreateKvTensor(device_id, executor, key_idx, kv_heads, max_len, head_dim);
    value_tensors_[i] = CreateKvTensor(device_id, executor, val_idx, kv_heads, max_len, head_dim);
    if (key_tensors_[i] == nullptr || value_tensors_[i] == nullptr) {
      MS_LOG(ERROR) << "KV tensor creation failed at layer " << i;
      Free();
      return false;
    }
  }
  Reset();
  return true;
}

void KVCacheManager::Free() {
  const auto &api = NNRTWrapper::GetApi();
  for (auto *t : key_tensors_) {
    if (t != nullptr && api.Tensor_Destroy != nullptr) {
      api.Tensor_Destroy(&t);
    }
  }
  for (auto *t : value_tensors_) {
    if (t != nullptr && api.Tensor_Destroy != nullptr) {
      api.Tensor_Destroy(&t);
    }
  }
  key_tensors_.clear();
  value_tensors_.clear();
  byte_count_per_layer_ = 0;
}

void KVCacheManager::Reset() {
  const auto &api = NNRTWrapper::GetApi();
  for (auto *t : key_tensors_) {
    if (t != nullptr && api.Tensor_GetDataBuffer != nullptr) {
      std::memset(api.Tensor_GetDataBuffer(t), 0, byte_count_per_layer_);
    }
  }
  for (auto *t : value_tensors_) {
    if (t != nullptr && api.Tensor_GetDataBuffer != nullptr) {
      std::memset(api.Tensor_GetDataBuffer(t), 0, byte_count_per_layer_);
    }
  }
}

NN_Tensor *KVCacheManager::GetKeyTensor(size_t layer) {
  return (layer < key_tensors_.size()) ? key_tensors_[layer] : nullptr;
}
NN_Tensor *KVCacheManager::GetValueTensor(size_t layer) {
  return (layer < value_tensors_.size()) ? value_tensors_[layer] : nullptr;
}

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite
