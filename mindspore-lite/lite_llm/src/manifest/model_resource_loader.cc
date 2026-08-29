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
#include <sys/stat.h>

#include <memory>
#include <string>
#include <vector>

#include "manifest/model_manifest.h"
#include "manifest/msl_package_reader.h"
namespace mslite_llm {

namespace {

/// Resolve a manifest-declared path relative to package_root.
/// Returns false when the asset is missing or fails path checks.
bool ResolveAsset(const std::string &package_root, const std::string &field_name, const std::string &candidate,
                  bool required, std::string *out, std::string *error_message) {
  if (candidate.empty()) {
    if (required) {
      if (error_message != nullptr) {
        *error_message = "missing required asset: " + field_name;
      }
      return false;
    }
    out->clear();
    return true;
  }
  std::string resolved;
  if (!ResolvePackagePath(package_root, candidate, &resolved)) {
    if (error_message != nullptr) {
      *error_message = "asset \"" + field_name + "\" path is invalid or outside package: " + candidate;
    }
    return false;
  }
  *out = std::move(resolved);
  return true;
}

bool IsRegularFile(const std::string &path) {
  struct stat st {};
  return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

/// Resolve a manifest-declared entry name against the single-file entry table.
/// The resolved value is the entry name itself (consumers read via MslPackageReader).
bool ResolveEntry(const MslPackageReader &reader, const std::string &field_name, const std::string &candidate,
                  bool required, std::string *out, std::string *error_message) {
  if (candidate.empty()) {
    if (required) {
      if (error_message != nullptr) {
        *error_message = "missing required asset: " + field_name;
      }
      return false;
    }
    out->clear();
    return true;
  }
  if (!IsPackageRelativePath(candidate)) {
    if (error_message != nullptr) {
      *error_message = "asset \"" + field_name + "\" path is invalid: " + candidate;
    }
    return false;
  }
  if (reader.Lookup(candidate) == nullptr) {
    if (error_message != nullptr) {
      *error_message = "asset \"" + field_name + "\" not found in .msl: " + candidate;
    }
    return false;
  }
  *out = candidate;
  return true;
}

}  // namespace

namespace {

/// Validate the NPU-specific manifest contract shared by directory and single-file modes.
MSLlmStatus ValidateNpuManifest(const ModelManifest &manifest, std::string *error_message) {
  if (!manifest.litert.has_prefill || manifest.litert.prefill_path.empty()) {
    if (error_message != nullptr) {
      *error_message = "NPU backend requires litert.prefill graph (.omc)";
    }
    return MSLLM_ERROR_MODEL_LOAD;
  }
  if (!manifest.npu.present) {
    if (error_message != nullptr) {
      *error_message = "NPU backend requires manifest.npu (max_length/chunk_size)";
    }
    return MSLLM_ERROR_MODEL_LOAD;
  }
  return MSLLM_SUCCESS;
}

}  // namespace

MSLlmStatus LoadModelResourcesFromSingleFile(const std::string &msl_path, ModelResources *resources,
                                             MSLlmBackendType backend, std::string *error_message) {
  if (resources == nullptr) {
    if (error_message != nullptr) {
      *error_message = "resources output is null";
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }

  auto reader = std::make_shared<MslPackageReader>();
  if (!reader->Open(msl_path, error_message)) {
    return MSLLM_ERROR_IO;
  }

  ModelManifest manifest;
  if (auto status = BuildModelManifestFromKv(*reader, &manifest, error_message); status != MSLLM_SUCCESS) {
    return status;
  }

  resources->package_root = msl_path;
  resources->package_reader = std::move(reader);
  resources->single_file = true;

  const bool is_nnrt = (backend == MSLLM_BACKEND_NNRT);
  if (is_nnrt) {
    if (auto status = ValidateNpuManifest(manifest, error_message); status != MSLLM_SUCCESS) {
      return status;
    }
  }

  // ── LiteRT graphs ────────────────────────────────────────────────────
  if (manifest.litert.has_prefill && !manifest.litert.prefill_path.empty()) {
    if (!ResolveEntry(*resources->package_reader, "prefill", manifest.litert.prefill_path, true,
                      &resources->prefill_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
  }
  if (manifest.litert.has_decode && !manifest.litert.decode_path.empty()) {
    if (!ResolveEntry(*resources->package_reader, "decode", manifest.litert.decode_path, true, &resources->decode_path,
                      error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
  }
  for (const auto &variant : manifest.litert.decode_variants) {
    std::string resolved;
    if (!ResolveEntry(*resources->package_reader, "decode_variant", variant.path, true, &resolved, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    LiteRtDecodeVariant resolved_variant = variant;
    resolved_variant.path = resolved;
    resources->decode_variants.push_back(std::move(resolved_variant));
  }

  // ── Assets ───────────────────────────────────────────────────────────
  if (manifest.assets.present) {
    if (!ResolveEntry(*resources->package_reader, "tokenizer", manifest.assets.tokenizer, is_nnrt,
                      &resources->tokenizer_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    const bool nnrt_assets_required = is_nnrt;
    if (!ResolveEntry(*resources->package_reader, "embedding", manifest.assets.embedding, nnrt_assets_required,
                      &resources->embedding_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    ResolveEntry(*resources->package_reader, "embedding_fp16", manifest.assets.embedding_fp16, false,
                 &resources->embedding_fp16_path, nullptr);
    if (!ResolveEntry(*resources->package_reader, "rope_sin", manifest.assets.rope_sin, nnrt_assets_required,
                      &resources->rope_sin_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    if (!ResolveEntry(*resources->package_reader, "rope_cos", manifest.assets.rope_cos, nnrt_assets_required,
                      &resources->rope_cos_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    if (!ResolveEntry(*resources->package_reader, "attention_mask", manifest.assets.attention_mask,
                      nnrt_assets_required, &resources->attention_mask_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
  } else if (is_nnrt) {
    if (error_message != nullptr) {
      *error_message = "NPU backend requires assets in manifest";
    }
    return MSLLM_ERROR_MODEL_LOAD;
  }

  return MSLLM_SUCCESS;
}

MSLlmStatus LoadModelResources(const std::string &package_root, ModelResources *resources, MSLlmBackendType backend,
                               std::string *error_message) {
  if (resources == nullptr) {
    if (error_message != nullptr) {
      *error_message = "resources output is null";
    }
    return MSLLM_ERROR_INVALID_ARGS;
  }

  // Single-file .msl container (mspacker): resolve via the entry table.
  if (IsRegularFile(package_root)) {
    return LoadModelResourcesFromSingleFile(package_root, resources, backend, error_message);
  }

  // Canonicalise the package root itself.
  std::string canonical_root;
  if (!ResolvePackagePath(package_root, ".", &canonical_root)) {
    // If "." doesn't resolve (e.g. the directory doesn't exist yet in tests),
    // use the provided root as-is after basic sanitisation.
    canonical_root = package_root;
  }

  // Load manifest.json
  const std::string manifest_path = canonical_root + "/manifest.json";
  ModelManifest manifest;
  if (auto status = LoadModelManifest(manifest_path, &manifest, error_message); status != MSLLM_SUCCESS) {
    return status;
  }

  resources->package_root = canonical_root;

  // ── LiteRT graphs ────────────────────────────────────────────────────
  const bool is_nnrt = (backend == MSLLM_BACKEND_NNRT);

  if (is_nnrt) {
    if (auto status = ValidateNpuManifest(manifest, error_message); status != MSLLM_SUCCESS) {
      return status;
    }
  }

  if (manifest.litert.has_prefill && !manifest.litert.prefill_path.empty()) {
    if (!ResolvePackagePath(canonical_root, manifest.litert.prefill_path, &resources->prefill_path)) {
      if (error_message != nullptr) {
        *error_message = "prefill graph path is invalid: " + manifest.litert.prefill_path;
      }
      return MSLLM_ERROR_MODEL_LOAD;
    }
  }
  if (manifest.litert.has_decode && !manifest.litert.decode_path.empty()) {
    if (!ResolvePackagePath(canonical_root, manifest.litert.decode_path, &resources->decode_path)) {
      if (error_message != nullptr) {
        *error_message = "decode graph path is invalid: " + manifest.litert.decode_path;
      }
      return MSLLM_ERROR_MODEL_LOAD;
    }
  }
  for (const auto &variant : manifest.litert.decode_variants) {
    std::string resolved;
    if (!ResolvePackagePath(canonical_root, variant.path, &resolved)) {
      if (error_message != nullptr) {
        *error_message = "decode variant path is invalid: " + variant.path;
      }
      return MSLLM_ERROR_MODEL_LOAD;
    }
    LiteRtDecodeVariant resolved_variant = variant;
    resolved_variant.path = resolved;
    resources->decode_variants.push_back(std::move(resolved_variant));
  }

  // ── Assets ───────────────────────────────────────────────────────────
  if (manifest.assets.present) {
    if (!ResolveAsset(canonical_root, "tokenizer", manifest.assets.tokenizer, is_nnrt, &resources->tokenizer_path,
                      error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    // NPU (NNRT) runtime reads these bins for the CPU-side embedding lookup,
    // RoPE constants and the causal mask; missing any of them fails at Build.
    const bool nnrt_assets_required = is_nnrt;
    if (!ResolveAsset(canonical_root, "embedding", manifest.assets.embedding, nnrt_assets_required,
                      &resources->embedding_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    ResolveAsset(canonical_root, "embedding_fp16", manifest.assets.embedding_fp16, false,
                 &resources->embedding_fp16_path, nullptr);
    if (!ResolveAsset(canonical_root, "rope_sin", manifest.assets.rope_sin, nnrt_assets_required,
                      &resources->rope_sin_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    if (!ResolveAsset(canonical_root, "rope_cos", manifest.assets.rope_cos, nnrt_assets_required,
                      &resources->rope_cos_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
    if (!ResolveAsset(canonical_root, "attention_mask", manifest.assets.attention_mask, nnrt_assets_required,
                      &resources->attention_mask_path, error_message)) {
      return MSLLM_ERROR_MODEL_LOAD;
    }
  } else if (is_nnrt) {
    if (error_message != nullptr) {
      *error_message = "NPU backend requires assets in manifest";
    }
    return MSLLM_ERROR_MODEL_LOAD;
  }

  return MSLLM_SUCCESS;
}

}  // namespace mslite_llm
