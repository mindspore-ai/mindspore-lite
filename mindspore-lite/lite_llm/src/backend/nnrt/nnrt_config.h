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
#ifndef MSLLM_NNRT_CONFIG_H
#define MSLLM_NNRT_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mslite_llm {
class MslPackageReader;
}  // namespace mslite_llm

namespace mslite {
namespace backend {
namespace nnrt {
using ::mslite_llm::MslPackageReader;

/// @brief Configuration subset extracted from ModelManifest + BackendConfig
/// that the NNRT executor needs.  Mirrors the PR 1059 NnrtConfig fields
/// actually consumed by the NNRT graph/executor path.
struct NnrtConfig {
  // ── Model architecture (from manifest.architecture) ──────────────────
  int32_t num_layers = 0;
  int32_t hidden_size = 0;
  int32_t num_key_value_heads = 0;
  int32_t head_dim = 0;
  int32_t vocab_size = 0;  // tokenizer vocab (<= model_vocab)
  int32_t max_position_embeddings = 0;
  int32_t max_length = 0;

  // ── Graph paths (from ModelResources) ────────────────────────────────
  std::string prefill_path;  // .omc prefill graph
  std::string decode_path;   // .omc decode graph

  // ── Asset paths (from ModelResources) ────────────────────────────────
  std::string embedding_path;
  std::string embedding_fp16_path;
  std::string rope_sin_path;
  std::string rope_cos_path;
  std::string attention_mask_path;

  // ── NNRT-specific ────────────────────────────────────────────────────
  size_t device_id = 0;
  int32_t chunk_size = 0;        // prefill chunk size (0 = full prompt)
  bool embedding_quant = false;  // W4A16 int4-packed embedding
  int32_t scale_gp_size = 32;    // W4A16 quant group size
  int32_t eos_id = -1;

  // ── Single-file .msl container ───────────────────────────────────────
  // Owns the reader so its .msl mmap stays alive for the executor's lifetime
  // (the .omc is handed to NNRT via the offline-model buffer API, which may
  // reference the mapped region). When single_file is true every path field
  // above is an entry name resolved via package_reader rather than a
  // filesystem path.
  std::shared_ptr<MslPackageReader> package_reader;
  bool single_file = false;
};

}  // namespace nnrt
}  // namespace backend
}  // namespace mslite

#endif  // MSLLM_NNRT_CONFIG_H
