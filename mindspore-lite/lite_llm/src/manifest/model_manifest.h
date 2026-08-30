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
#ifndef MSLLM_MODEL_MANIFEST_H
#define MSLLM_MODEL_MANIFEST_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../llm_types_internal.h"

namespace mslite_llm {

class MslPackageReader;

struct LiteRtDecodeVariant {
  int32_t past_len = -1;
  std::string path;
};

struct LiteRtManifest {
  bool present = false;
  MSLlmDType precision = MSLLM_DTYPE_FLOAT32;

  bool has_prefill = false;
  std::string prefill_path;
  int32_t prefill_seq_len = -1;

  bool has_decode = false;
  std::string decode_path;
  bool decode_dynamic_past_len = false;
  int32_t decode_past_len = -1;
  int32_t decode_max_past_len = -1;
  std::vector<LiteRtDecodeVariant> decode_variants;
};

struct ModelArchitecture {
  int32_t num_layers = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t num_heads = 0;
  int32_t num_kv_heads = 0;
  int32_t head_dim = 0;
  int32_t vocab_size = 0;
  int32_t max_position_embeddings = 0;
  float rope_theta = 10000.0f;
  float norm_eps = 1e-6f;
  int32_t tie_word_embeddings = 0;
  bool present = false;

  bool IsComplete() const;
};

struct GenerationPolicy {
  bool present = false;
  std::vector<int32_t> stop_token_ids;
  std::vector<int32_t> suppress_token_ids;
};

struct ModelManifestAssets {
  bool present = false;
  std::string tokenizer;
  std::string embedding;
  std::string embedding_fp16;
  std::string rope_sin;
  std::string rope_cos;
  std::string attention_mask;
};

/// NNRT (Kirin NPU) runtime parameters carried by the manifest ``npu`` section.
/// These are fixed at export time (max_length/chunk_size are encoded in the
/// .omc gear shapes) and consumed by NNRTBackend::BuildNnrtConfig.
struct NpuConfig {
  bool present = false;
  int32_t max_length = 0;        // max sequence length (must be a multiple of chunk_size)
  int32_t chunk_size = 0;        // prefill chunk size (prefill gear seq length)
  bool embedding_quant = false;  // W4A8/W4A16 int4-packed embedding weight
  int32_t scale_gp_size = 32;    // embedding quant group size
};

struct ModelManifest {
  std::string model_name;
  std::string version;
  std::string format_version;
  MSLlmDType dtype = MSLLM_DTYPE_FLOAT32;
  ModelArchitecture architecture;
  GenerationPolicy generation;
  LiteRtManifest litert;
  ModelManifestAssets assets;
  NpuConfig npu;
};

bool ParseDTypeName(const std::string &raw, MSLlmDType *out);
MSLlmStatus LoadModelManifest(const std::string &manifest_path, ModelManifest *manifest,
                              std::string *error_message = nullptr);

/// Parse a manifest from an in-memory JSON string (used by the single-file .msl reader).
MSLlmStatus ParseManifest(const std::string &content, ModelManifest *manifest, std::string *error_message = nullptr);

/// Build a manifest from the KV metadata of a single-file .msl (v1).
/// Unknown KV keys are skipped; missing optional keys keep defaults.
MSLlmStatus BuildModelManifestFromKv(const MslPackageReader &reader, ModelManifest *manifest,
                                     std::string *error_message = nullptr);

// ─── Resource Descriptor and Loader ─────────────────────────────────────────

struct ModelResources {
  std::string package_root;
  // Single-file .msl reader (nullptr when loaded from a directory package).
  std::shared_ptr<MslPackageReader> package_reader;
  bool single_file = false;
  // LiteRT graph paths (relative to package_root)
  std::string prefill_path;
  std::string decode_path;
  std::vector<LiteRtDecodeVariant> decode_variants;
  // Assets (relative to package_root)
  std::string tokenizer_path;
  std::string embedding_path;
  std::string embedding_fp16_path;
  std::string rope_sin_path;
  std::string rope_cos_path;
  std::string attention_mask_path;
};

/// Validate that \p candidate is a relative path strictly inside \p package_root
/// (no absolute paths, no ".." traversal, no colon-based drive escapes).
bool IsPackageRelativePath(const std::string &path);

/// Resolve \p candidate against \p package_root and verify it is a canonical
/// child of the package root.  Returns the resolved path on success.
bool ResolvePackagePath(const std::string &package_root, const std::string &candidate, std::string *resolved);

/// Load and validate all model resources from a directory .msl package.
/// Reads manifest.json and resolves every declared path against the package
/// root.  For NPU backends (require_npu_assets=true) the loader enforces that
/// prefill, decode, graph_io and tokenizer paths are present and valid.
MSLlmStatus LoadModelResources(const std::string &package_root, ModelResources *resources, MSLlmBackendType backend,
                               std::string *error_message = nullptr);

/// Load resources from a single-file .msl (mspacker) container.  Reads
/// manifest.json from the entry table and resolves every declared asset by
/// entry name; ``resources->package_reader`` is set for offset-based loading.
MSLlmStatus LoadModelResourcesFromSingleFile(const std::string &msl_path, ModelResources *resources,
                                             MSLlmBackendType backend, std::string *error_message = nullptr);

}  // namespace mslite_llm

#endif
