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

#ifndef MSLLM_NNRT_KVCACHE_H
#define MSLLM_NNRT_KVCACHE_H

#include <cstddef>
#include <vector>

struct NN_Tensor;
struct OH_NNExecutor;

namespace mslite {
namespace backend {
namespace nnrt {

/// @brief Owns per-layer key/value KV cache as ION-backed NN_Tensor objects.
/// Tensors are reused as both input and output (omc model updates in place),
/// so there is no Read*/Write* copy API.
class KVCacheManager {
 public:
  KVCacheManager() = default;
  ~KVCacheManager() { Free(); }

  /// @brief Create num_layers key + value tensors, shape [1, kv_heads, max_len, head_dim] fp16,
  ///        zero-initialized via memset on their ION buffers.
  bool Alloc(int num_layers, int kv_heads, int max_len, int head_dim, size_t device_id, OH_NNExecutor *executor);

  void Free();
  void Reset();  // memset each tensor's GetDataBuffer to 0

  size_t NumLayers() const { return key_tensors_.size(); }
  NN_Tensor *GetKeyTensor(size_t layer);
  NN_Tensor *GetValueTensor(size_t layer);

 private:
  size_t byte_count_per_layer_{0};
  std::vector<NN_Tensor *> key_tensors_;
  std::vector<NN_Tensor *> value_tensors_;
};

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_KVCACHE_H
