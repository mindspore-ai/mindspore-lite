// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0.
#pragma once

#include <filesystem>  // NOLINT(build/c++17): lite_llm requires C++17.
#include <memory>
#include <string>

#include "manifest/msl_package_reader.h"

namespace mslite_llm {
// Register an immutable package index. The registry holds only weak references;
// ModelResources owns the reader until all backend operations have finished.
void RegisterArtifactPackage(const std::string &root, const std::shared_ptr<MslPackageReader> &reader);
bool IsArtifactPackagePath(const std::string &path);
bool ArtifactExists(const std::filesystem::path &path);
// Returns an owned descriptor and the resource range within a registered .msl.
// Standalone filesystem artifacts are unsupported. Callers must close the descriptor.
int OpenArtifact(const std::string &path, size_t *size, size_t *offset, std::string *error);
bool ReadArtifact(const std::filesystem::path &path, void *destination, size_t bytes, std::string *error);
bool ReadArtifactBytes(const std::string &path, std::vector<uint8_t> *bytes, std::string *error);
std::string ArtifactResourceName(const std::string &path);
// NNRT's external-weight API accepts only a filesystem directory.
class ArtifactWeightDirectory {
 public:
  ArtifactWeightDirectory() = default;
  ArtifactWeightDirectory(const ArtifactWeightDirectory &) = delete;
  ArtifactWeightDirectory &operator=(const ArtifactWeightDirectory &) = delete;
  ~ArtifactWeightDirectory();
  bool Materialize(const std::string &path, std::string *directory, std::string *error);

 private:
  std::string directory_;
};
}  // namespace mslite_llm
