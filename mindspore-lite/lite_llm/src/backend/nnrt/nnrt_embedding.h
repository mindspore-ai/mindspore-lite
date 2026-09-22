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

#ifndef MSLLM_NNRT_EMBEDDING_H
#define MSLLM_NNRT_EMBEDDING_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "manifest/model_manifest.h"

struct NN_Tensor;
namespace mslite_llm {
class MslPackageReader;
}

namespace mslite {
namespace backend {
namespace nnrt {
using ::mslite_llm::EmbeddingFormat;

struct NnrtConfig;

// Owns the upload source until shared tensors are ready, then borrows their
// CPU-accessible buffers for row lookup. The caller owns the returned tensors
// and must keep them alive while using this object. No quantization details are
// needed by the executor; callbacks provide only generic asset/tensor I/O.
class NnrtEmbedding {
 public:
  using ReadAsset = std::function<bool(const std::string &, std::vector<uint8_t> *)>;
  using CreateTensor = std::function<NN_Tensor *(size_t, const std::vector<int32_t> &, int32_t, size_t)>;
  using UploadTensor = std::function<const uint8_t *(NN_Tensor *, const void *, size_t)>;

  NnrtEmbedding() = default;
  NnrtEmbedding(const NnrtEmbedding &) = delete;
  NnrtEmbedding &operator=(const NnrtEmbedding &) = delete;
  bool Configure(const NnrtConfig &config);
  bool Load(const ReadAsset &read_asset);
  // Writes each created tensor into inputs immediately, including on failure,
  // so the caller can release partially created inputs through its normal cleanup.
  bool CreateInputs(const CreateTensor &create, const UploadTensor &upload, std::vector<NN_Tensor *> *inputs);
  bool Row(int token, uint16_t *output) const;
  size_t InputCount() const { return input_count_; }

 private:
  bool InitStorage();
  void ReleaseUploadSource();
  std::vector<int32_t> InputShape(size_t input) const;

  int vocab_size_{0};
  int hidden_size_{0};
  bool quantized_{false};
  EmbeddingFormat format_{EmbeddingFormat::kUnknown};
  size_t input_count_{0};
  size_t weight_bytes_{0};
  size_t scale_bytes_{0};
  std::string path_;
  std::shared_ptr<mslite_llm::MslPackageReader> package_reader_;
  bool single_file_{false};
  std::vector<uint8_t> upload_buffer_;
  const uint8_t *weights_{nullptr};
  const uint8_t *scales_{nullptr};
};

// IEEE fp16 <-> fp32 bit-level conversions (fp16 values are passed as raw
// uint16 bits).
uint16_t Fp32ToFp16Bits(float value);
float Fp16BitsToFp32(uint16_t bits);

// Compact Q4_0 phase4 NZF: packed cells followed by per-row g32 fp16 scales.
// Cells are at most 64x1024, with no padding. Rows must be a multiple of 16,
// hidden a multiple of 32; total storage is rows * hidden / 32 * 18 bytes.
bool Q4NzfEmbeddingSize(int rows, int hidden, size_t *packed_bytes, size_t *total_bytes);

struct Q4EmbeddingShape {
  int rows = 0;
  int hidden = 0;
};

// Decode only the requested logical row directly into hidden fp16 output
// values. Adjacent K lanes occupy low/high signed two's-complement nibbles.
// Each 128-byte 16x16 tile interleaves four 32-byte phases: byte j of the
// adjacent-byte layout is at 4 * (j % 32) + j / 32. Scale bits are unchanged.
bool DequantizeEmbeddingRow(const uint8_t *blob, size_t blob_size, Q4EmbeddingShape shape, int row, uint16_t *out_fp16);

struct S16S4EmbeddingView {
  const uint8_t *weights{nullptr};
  size_t weight_bytes{0};
  const uint8_t *scales{nullptr};
  size_t scale_bytes{0};
  int hidden{0};
  int vocab{0};
};

bool ValidateS16S4Embedding(const S16S4EmbeddingView &view);
bool DequantizeS16S4EmbeddingRow(const S16S4EmbeddingView &view, int token, uint16_t *output);

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_EMBEDDING_H
