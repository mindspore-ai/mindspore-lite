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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "backend/nnrt/nnrt_kvcache.h"
#include "backend/nnrt/nnrt_config.h"

struct OH_NNCompilation;
struct OH_NNExecutor;
struct NN_TensorDesc;
struct NN_Tensor;

namespace mslite {
class MslPackageReader;
namespace backend {
namespace nnrt {

class NnrtExecutor {
 public:
  NnrtExecutor() = default;
  ~NnrtExecutor();

  bool Build(const NnrtConfig &config);
  /// Run prefill or decode. The executor never samples: when logits_out is
  /// non-null the host logits vector is returned for Pipeline::Sampler (the
  /// caller owns argmax / top-k / temperature). When null, the legacy ArgMax
  /// token is written to output_ids for callers that only need a token.
  bool Forward(const std::vector<int> &input_ids, int *output_ids, bool is_prefill,
               std::vector<float> *logits_out = nullptr);
  bool Reset();

 private:
  bool BuildModel();
  // Read an asset either from the single-file package reader (entry name) or
  // from a filesystem path, into a raw byte buffer.
  bool ReadAsset(const std::string &path_or_entry, std::vector<uint8_t> *out) const;
  // Fail-fast contract check: compare the .omc's actual I/O against the layout implied by
  // config.num_layers (7 non-KV + 2*num_layers inputs, interleaved KV). Inputs are checked
  // by name (the device preserves them); outputs by count only, because the Kirin DDK
  // renames them (enum-shape artifact) — actual output names are logged for forensics.
  bool ValidateModelContract();
  // Read the logits width from the model output desc 0 into model_vocab_ (fallback:
  // config vocab_size_). Fails if the tokenizer vocab exceeds the model vocab.
  bool ReadModelVocab();
  bool LoadCpuBuffers(const NnrtConfig &config);
  // Parse the W4A16 embedding bin layout (file rows / packed-per-row / scales-per-row /
  // group size) and validate it against vocab_size_. Does NOT build an fp16 table:
  // embedding rows are dequantized on demand by EmbeddingRow.
  bool DequantizeEmbeddingTable(int group_size);
  // Write token tid's fp16 embedding row into dst. W4A16 (embedding_quant) rows are
  // dequantized from the packed bin on demand; the fp16 path copies from the table.
  bool EmbeddingRow(int tid, uint16_t *dst);
  bool CreateTensors();  // prefill/decode input groups + logits output
  NN_Tensor *CreateInputTensor(size_t index, const int32_t *shape, size_t dim_count, int32_t dtype);
  // Create the idx6 embedding_weight tensor. The omc desc is only used to recover the
  // graph CAPACITY (byte size); the shape passed to Tensor_Create is always the original
  // ONNX rank — 1-dim INT8 [capacity] — because the device enum-shape gear matcher
  // rejects the NNRT-padded [capacity,1,1,1] desc at RunSync (proven on kirin9020,
  // 2026-07-25). fallback_capacity (the embedding bin size) is used when the NNRT cannot
  // report the byte size.
  NN_Tensor *CreateInputTensorFromOmc(size_t index, int32_t fallback_capacity, int32_t fallback_dtype);
  // Fit the idx6 embedding_weight write to the graph's INT8 capacity. The .omc declares a
  // fixed capacity (graph_rows * planar row bytes: K/2 int4-packed + K/group_size fp16
  // scales per row). When the bin holds MORE rows than the graph (full-vocab external
  // table vs cropped graph), the scale plane is not a file prefix: repack the first
  // graph_rows of each plane into a contiguous capacity-sized buffer. Fails when the
  // capacity is not a whole number of planar rows or the bin has fewer rows than the
  // graph. The CPU-side EmbeddingRow dequantizes from the FULL file, so token ids
  // can exceed the graph rows.
  bool PrepareEmbeddingWeightWrite(NN_Tensor *tensor, std::vector<uint8_t> *repacked, const void **data,
                                   size_t *size) const;
  // Record the byte capacity of a created tensor for the WriteTensor overflow guard.
  // Degrades to no guard (with a warning) when the NNRT cannot report the byte size.
  void RecordTensorCapacity(NN_Tensor *tensor, const NN_TensorDesc *desc, size_t index);
  // Copy data into a tensor's buffer. Returns false on invalid arguments, when the
  // write exceeds the recorded capacity, or when the data buffer is unavailable.
  bool WriteTensor(NN_Tensor *tensor, const void *data, size_t size);

  bool Prefill(const std::vector<int> &input_ids, int *output_ids, std::vector<float> *logits_out = nullptr);
  bool Decode(const std::vector<int> &input_ids, int *output_ids, std::vector<float> *logits_out = nullptr);

  // model info
  int64_t vocab_size_{0};   // tokenizer/sampling vocab (NnrtConfig.vocab_size, cropped)
  int64_t model_vocab_{0};  // logits width reported by the .omc (>= vocab_size_)
  int64_t head_dim_{0};
  int64_t max_length_{0};
  int64_t chunk_size_{0};
  int64_t eos_id_{-1};
  // Absolute position of the next token to decode. Reset at prefill, then advanced
  // one per decode step (the .omc KV scatter/rope/mask offsets are absolute).
  int64_t history_{0};
  int hidden_size_{0};
  int num_key_value_heads_{0};
  int num_layers_{0};
  size_t device_id_{0};
  bool embedding_quant_{false};  // NnrtConfig.embedding_quant: idx6 bin is W4A16 int4-packed
  int scale_gp_size_{32};        // NnrtConfig.scale_gp_size: W4A16 quantization group size

  // NNRT handles
  OH_NNCompilation *nn_compilation_{nullptr};
  OH_NNExecutor *nn_executor_{nullptr};

  // CPU-side lookup buffers
  // fp16 embedding table — ONLY for the non-quant (fp16) path, where the fp16 bin IS
  // the single source of truth. The W4A16 path keeps only the packed bin (73MB) and
  // dequantizes rows on demand via EmbeddingRow, so no 272MB fp16 copy is resident.
  uint16_t *embedding_table_{nullptr};
  size_t embedding_table_elems_{0};
  size_t embed_file_rows_ = 0;
  size_t embed_packed_per_row_ = 0;
  size_t embed_scales_per_row_ = 0;
  int embed_group_size_ = 32;
  std::vector<uint8_t> embedding_weight_buffer_;  // raw int4-packed bytes for idx6
  std::vector<uint16_t> sin_buffer_;              // [max_len, head_dim]
  std::vector<uint16_t> cos_buffer_;              // [max_len, head_dim]
  std::vector<uint16_t> attention_mask_buffer_;   // [max_len, max_len]

  // ION tensors — created once in Build, reused every step.
  // input index order: 0 valid_seq_len, 1 lmhead_idx, 2 rope_cos, 3 rope_sin,
  //                    4 input_embeds, 5 attn_mask, 6 embedding_weight, [7..] interleaved K/V caches
  std::vector<NN_Tensor *> prefill_inputs_;  // 7 non-KV (chunk_size shape)
  std::vector<NN_Tensor *> decode_inputs_;   // 7 non-KV (1 shape)
  NN_Tensor *logits_tensor_{nullptr};        // output [1,1,1,model_vocab] fp32

  // Assembled I/O arrays (prefill_inputs + KV / decode_inputs + KV, etc.)
  std::vector<NN_Tensor *> prefill_in_;
  std::vector<NN_Tensor *> prefill_out_;
  std::vector<NN_Tensor *> decode_in_;
  std::vector<NN_Tensor *> decode_out_;

  // Byte capacity of each created ION tensor, used by WriteTensor to prevent overwrites.
  std::unordered_map<NN_Tensor *, size_t> tensor_byte_sizes_;

  KVCacheManager kv_cache_manager_;
  std::string omc_path_;

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
