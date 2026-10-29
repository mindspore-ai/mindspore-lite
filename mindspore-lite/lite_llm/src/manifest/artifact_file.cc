// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0.
#include "manifest/artifact_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace mslite_llm {
namespace {
std::mutex package_mutex;
std::unordered_map<std::string, std::vector<std::weak_ptr<MslPackageReader>>> packages;

std::shared_ptr<MslPackageReader> FindPackage(const std::string &path, std::string *relative) {
  std::lock_guard<std::mutex> lock(package_mutex);
  for (size_t end = path.rfind('/'); end != std::string::npos; end = path.rfind('/', end - 1)) {
    auto found = packages.find(path.substr(0, end));
    if (found != packages.end()) {
      for (auto it = found->second.rbegin(); it != found->second.rend(); ++it) {
        auto reader = it->lock();
        if (reader) {
          *relative = path.substr(end + 1);
          return reader;
        }
      }
    }
    if (end == 0) break;
  }
  return nullptr;
}

bool ReadRange(int fd, size_t offset, void *destination, size_t bytes) {
  auto *output = static_cast<uint8_t *>(destination);
  size_t done = 0;
  while (done < bytes) {
    const auto count = pread(fd, output + done, bytes - done, static_cast<off_t>(offset + done));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    done += static_cast<size_t>(count);
  }
  return true;
}

std::string WeightTemporaryRoot(const std::string &path) {
  const char *configured = std::getenv("TMPDIR");
  if (configured && configured[0] != '\0') return configured;
  std::string relative;
  if (FindPackage(path, &relative)) {
    const auto package = path.substr(0, path.size() - relative.size() - 1);
    auto parent = std::filesystem::path(package).parent_path().string();
    if (parent.empty()) parent = ".";
    if (access(parent.c_str(), W_OK) == 0) return parent;
  }
  return access("/tmp", W_OK) == 0 ? "/tmp" : "/data/local/tmp";
}
}  // namespace

std::string ArtifactResourceName(const std::string &path) {
  if (path.size() < 64) return path;
  const uint64_t hash =
    std::accumulate(path.begin(), path.end(), uint64_t{14695981039346656037ULL},
                    [](uint64_t value, unsigned char byte) { return (value ^ byte) * 1099511628211ULL; });
  std::ostringstream name;
  name << "r/" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return name.str();
}

void RegisterArtifactPackage(const std::string &root, const std::shared_ptr<MslPackageReader> &reader) {
  std::lock_guard<std::mutex> lock(package_mutex);
  for (auto it = packages.begin(); it != packages.end();) {
    auto &readers = it->second;
    readers.erase(std::remove_if(readers.begin(), readers.end(), [](const auto &item) { return item.expired(); }),
                  readers.end());
    if (readers.empty())
      it = packages.erase(it);
    else
      ++it;
  }
  packages[root].push_back(reader);
}

bool IsArtifactPackagePath(const std::string &path) {
  std::string relative;
  return FindPackage(path, &relative) != nullptr;
}

bool ArtifactExists(const std::filesystem::path &path) {
  std::string relative;
  auto reader = FindPackage(path.string(), &relative);
  return reader && reader->Lookup(ArtifactResourceName(relative)) != nullptr;
}

int OpenArtifact(const std::string &path, size_t *size, size_t *offset, std::string *error) {
  std::string relative;
  auto reader = FindPackage(path, &relative);
  if (reader) {
    const auto *entry = reader->Lookup(ArtifactResourceName(relative));
    if (entry && entry->offset <= std::numeric_limits<size_t>::max() &&
        entry->size <= std::numeric_limits<size_t>::max() - entry->offset &&
        entry->offset <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) &&
        entry->size <= static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - entry->offset) {
      *size = static_cast<size_t>(entry->size);
      *offset = static_cast<size_t>(entry->offset);
      const int fd = fcntl(reader->Fd(), F_DUPFD_CLOEXEC, 0);
      if (fd >= 0) return fd;
    }
  }
  if (error) *error = "cannot open artifact: " + path;
  return -1;
}

bool ReadArtifact(const std::filesystem::path &path, void *destination, size_t bytes, std::string *error) {
  size_t size = 0, offset = 0;
  const int fd = OpenArtifact(path.string(), &size, &offset, error);
  if (fd < 0) return false;
  const bool ok = size == bytes && (bytes == 0 || destination != nullptr) && ReadRange(fd, offset, destination, bytes);
  close(fd);
  if (!ok && error) *error = "artifact size mismatch or read failed: " + path.string();
  return ok;
}

bool ReadArtifactBytes(const std::string &path, std::vector<uint8_t> *bytes, std::string *error) {
  size_t size = 0, offset = 0;
  const int fd = OpenArtifact(path, &size, &offset, error);
  if (fd < 0) return false;
  bytes->resize(size);
  const bool ok = ReadRange(fd, offset, bytes->data(), size);
  close(fd);
  if (!ok && error) *error = "artifact read failed: " + path;
  return ok;
}

ArtifactWeightDirectory::~ArtifactWeightDirectory() {
  if (!directory_.empty()) {
    unlink((directory_ + "/SubGraph_0.weight").c_str());
    rmdir(directory_.c_str());
  }
}

bool ArtifactWeightDirectory::Materialize(const std::string &path, std::string *directory, std::string *error) {
  if (!directory_.empty()) {
    if (error) *error = "external-weight directory is already materialized";
    return false;
  }
  size_t size = 0, offset = 0;
  const int source = OpenArtifact(path, &size, &offset, error);
  if (source < 0) return false;
  std::string pattern = WeightTemporaryRoot(path) + "/msl-weights-XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const char *created = mkdtemp(name.data());
  if (!created) {
    close(source);
    if (error) *error = "cannot create NNRT external-weight directory";
    return false;
  }
  directory_ = created;
  const int output = open((directory_ + "/SubGraph_0.weight").c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  bool ok = output >= 0;
  std::array<uint8_t, 65536> buffer{};
  for (size_t done = 0; ok && done < size;) {
    const size_t count = std::min(buffer.size(), size - done);
    ok = ReadRange(source, offset + done, buffer.data(), count);
    size_t written = 0;
    while (ok && written < count) {
      const auto result = write(output, buffer.data() + written, count - written);
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0)
        ok = false;
      else
        written += static_cast<size_t>(result);
    }
    done += count;
  }
  close(source);
  if (output >= 0 && close(output) != 0) ok = false;
  if (!ok && error) *error = "cannot materialize NNRT external weights: " + path;
  if (ok) *directory = directory_;
  return ok;
}
}  // namespace mslite_llm
