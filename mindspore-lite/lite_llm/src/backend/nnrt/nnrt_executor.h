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

#ifndef MSLLM_NNRT_EXECUTOR_H
#define MSLLM_NNRT_EXECUTOR_H

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "backend/common/backend.h"
#include "backend/nnrt/nnrt_config.h"
#include "backend/nnrt/nnrt_embedding.h"
#include "backend/nnrt/nnrt_kvcache.h"

struct OH_NNCompilation;
struct OH_NNExecutor;
struct NN_TensorDesc;
struct NN_Tensor;

namespace mslite {
namespace backend {
namespace nnrt {

class NnrtExecutor {
 public:
  NnrtExecutor() = default;
  ~NnrtExecutor();

  bool Build(const NnrtConfig &config);
  // Run prefill or decode and expose a read-only view of the ION-backed logits
  // tensor. The view remains valid until the next Executor_RunSync call.
  bool Forward(const std::vector<int> &input_ids, bool is_prefill, mslite_llm::BackendOutput *output);
  bool Reset();

 private:
  bool InitConfig(const NnrtConfig &config);
  bool BuildModel();
  bool ConstructCompilation();
  bool MapOfflineModelFile(const std::string &path);
  void ReclaimOfflineModelPages() const;
  bool LoadExternalWeights();
  // Read an asset either from the single-file package reader (entry name) or
  // from a filesystem path, into a raw byte buffer.
  bool ReadAsset(const std::string &path_or_entry, std::vector<uint8_t> *out) const;
  // Fail-fast contract check: compare the .omc's actual I/O against the layout
  // implied by config.num_layers and the shared embedding inputs (interleaved
  // KV). Inputs are checked by name (the device preserves them); outputs by
  // count only, because the Kirin DDK renames them (enum-shape artifact) —
  // actual output names are logged for forensics.
  bool ValidateModelContract();
  // Read the logits width from the model output desc 0 into model_vocab_
  // (fallback: config vocab_size_); sampling uses the smaller vocabulary.
  void ReadModelVocab();
  bool LoadCpuBuffers(const NnrtConfig &config);
  // Copy an fp16 bin of count elements into dst; a missing path is not an error.
  bool LoadFp16Bin(const std::string &path, const char *what, std::vector<uint16_t> *dst, size_t count) const;
  bool CreateTensors();  // prefill/decode input groups + logits output
  NN_Tensor *CreateInputTensor(size_t index, const int32_t *shape, size_t dim_count, int32_t dtype,
                               size_t expected_bytes = 0);
  // Record the byte capacity of a created tensor for the WriteTensor overflow
  // guard. Degrades to no guard (with a warning) when the NNRT cannot report
  // the byte size.
  void RecordTensorCapacity(NN_Tensor *tensor, const NN_TensorDesc *desc, size_t index);
  // Copy data into a tensor's buffer. Returns false on invalid arguments, when
  // the write exceeds the recorded capacity, or when the data buffer is
  // unavailable.
  bool WriteTensor(NN_Tensor *tensor, const void *data, size_t size);
  // Publish the final ION-backed logits buffer as a step-scoped read-only view.
  bool ReadLogits(mslite_llm::BackendOutput *output) const;

  bool Prefill(const std::vector<int> &input_ids, mslite_llm::BackendOutput *output);
  bool Decode(const std::vector<int> &input_ids, mslite_llm::BackendOutput *output);

  // model info
  int64_t vocab_size_{0};   // tokenizer/sampling vocab (NnrtConfig.vocab_size, cropped)
  int64_t model_vocab_{0};  // logits width reported by the .omc (>= vocab_size_)
  int64_t head_dim_{0};
  int64_t max_length_{0};
  int64_t chunk_size_{0};
  int64_t eos_id_{-1};
  // Absolute position of the next token to decode. Reset at prefill, then
  // advanced one per decode step (the .omc KV scatter/rope/mask offsets are
  // absolute).
  int64_t history_{0};
  int hidden_size_{0};
  int num_key_value_heads_{0};
  int num_layers_{0};
  size_t device_id_{0};
  size_t non_kv_inputs_{0};  // six dynamic inputs plus the shared embedding inputs
  NnrtEmbedding embedding_;

  // NNRT handles
  OH_NNCompilation *nn_compilation_{nullptr};
  OH_NNExecutor *nn_executor_{nullptr};

  // CPU-side lookup buffers
  std::vector<uint16_t> sin_buffer_;             // [max_len, head_dim]
  std::vector<uint16_t> cos_buffer_;             // [max_len, head_dim]
  std::vector<uint16_t> attention_mask_buffer_;  // [max_len, max_len]

  // ION tensors — created once in Build, reused every step.
  // input index order: 0 valid_seq_len, 1 lmhead_idx, 2 rope_cos, 3 rope_sin,
  //                    4 input_embeds, 5 attn_mask, 6 embedding_weight, optional 7 embedding_scale, then K/V
  std::vector<NN_Tensor *> prefill_inputs_;  // 7 or 8 non-KV (chunk_size shape)
  std::vector<NN_Tensor *> decode_inputs_;   // 7 or 8 non-KV (1 shape)
  NN_Tensor *logits_tensor_{nullptr};        // output [1,1,1,model_vocab] fp32

  // Assembled I/O arrays (prefill_inputs + KV / decode_inputs + KV, etc.)
  std::vector<NN_Tensor *> prefill_in_;
  std::vector<NN_Tensor *> prefill_out_;
  std::vector<NN_Tensor *> decode_in_;
  std::vector<NN_Tensor *> decode_out_;

  // Byte capacity of each created ION tensor, used by WriteTensor to prevent
  // overwrites.
  std::unordered_map<NN_Tensor *, size_t> tensor_byte_sizes_;

  KVCacheManager kv_cache_manager_;
  std::string omc_path_;
  void *omc_mapping_{nullptr};
  size_t omc_mapping_size_{0};

  std::string om_weight_dir_;
  std::string external_weight_entry_ = "SubGraph_0.weight";
  bool has_external_weights_{false};
  std::string package_root_;
  std::string temp_weight_dir_;

  // Single-file .msl container support. package_reader_ owns the reader so its
  // .msl mmap stays alive for the executor's lifetime (the .omc is handed to
  // NNRT via the offline-model buffer API, which may reference the region).
  std::shared_ptr<MslPackageReader> package_reader_;
  bool single_file_{false};
};

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_EXECUTOR_H
