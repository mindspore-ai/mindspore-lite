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

#include "tools/mindir_exporter/shared_weight_store.h"
#include <sys/stat.h>
#include <fstream>
#include <sstream>
#include <vector>
#include "ir/tensor.h"
#include "src/common/file_utils.h"
#include "src/common/log_adapter.h"
#include "src/common/string_utils.h"
#include "utils/file_utils.h"

namespace mindspore::lite {
namespace {
bool IsSystemLittleEndian() {
  int check = 0x01;
  auto address = reinterpret_cast<char *>(&check);
  return *address == 0x01;
}
}  // namespace

SharedWeightStore::SharedWeightStore(const std::string &output_root) : root_path_(output_root) {}

SharedWeightStore::~SharedWeightStore() = default;

std::string SharedWeightStore::RelativeRootPath(const std::string &mindir_dir) const {
  // Canonicalize both paths, otherwise a relative output_root would not line up with the mindir dir.
  auto base = RealPath(mindir_dir.c_str());
  auto target = RealPath(root_path_.c_str());
  if (base.empty() || target.empty()) {
    // The output tree does not exist yet because the results are only written after the whole batch
    // converted. Both paths are derived from the same output root, so comparing them as given is
    // still correct.
    base = mindir_dir;
    target = root_path_;
  }
  if (base.empty() || target.empty()) {
    return ".";
  }
  auto split_path = [](const std::string &path) {
    std::vector<std::string> parts;
    std::istringstream stream(path);
    std::string part;
    while (std::getline(stream, part, '/')) {
      if (!part.empty()) {
        parts.push_back(part);
      }
    }
    return parts;
  };
  auto base_parts = split_path(base);
  auto target_parts = split_path(target);
  size_t common = 0;
  while (common < base_parts.size() && common < target_parts.size() && base_parts[common] == target_parts[common]) {
    ++common;
  }
  std::string relative;
  for (size_t i = common; i < base_parts.size(); ++i) {
    relative += "../";
  }
  for (size_t i = common; i < target_parts.size(); ++i) {
    relative += target_parts[i];
    if (i + 1 < target_parts.size()) {
      relative += "/";
    }
  }
  // An empty relative path is not a usable location value, use the current directory instead.
  if (relative.empty()) {
    return ".";
  }
  // Drop the trailing separator of the "../" chain, so the value reads as ".." / "../..".
  return relative.back() == '/' ? relative.substr(0, relative.size() - 1) : relative;
}

int SharedWeightStore::OrAllocate(const std::string &param_name, const tensor::TensorPtr &data, int64_t *offset,
                                  int64_t *length) {
  if (offset == nullptr || length == nullptr) {
    MS_LOG(ERROR) << "Output argument is nullptr.";
    return RET_NULL_PTR;
  }
  if (data == nullptr) {
    MS_LOG(ERROR) << "Tensor data of parameter " << param_name << " is nullptr.";
    return RET_NULL_PTR;
  }
  const auto *raw = static_cast<const char *>(data->data_c());
  int64_t data_length = static_cast<int64_t>(data->DataNBytes());
  *length = data_length;

  // Content-addressed dedup: parameter names are renamed to per-model unique ids by the trace
  // label, so they cannot match across models. Dedup by (data_type, shape, content hash) —
  // byte-identical tensors are safe to share regardless of their names.
  std::ostringstream key;
  key << static_cast<int>(data->data_type());
  for (const auto &dim : data->shape()) {
    key << "|" << dim;
  }
  key << "|" << StringHash64(raw, static_cast<size_t>(data_length));
  auto entry_iter = dedup_map_.find(key.str());
  if (entry_iter != dedup_map_.end()) {
    *offset = entry_iter->second;
    return RET_OK;
  }

  int64_t append_size = (data_length % kDataOffset == 0) ? 0 : (kDataOffset - data_length % kDataOffset);
  dedup_map_[key.str()] = cursor_;
  entries_.push_back({data, cursor_, data_length});
  *offset = cursor_;
  cursor_ += data_length + append_size;
  return RET_OK;
}

int SharedWeightStore::Save(const std::string &file_path) {
  auto sep = file_path.rfind('/');
  auto weight_dir = sep == std::string::npos ? std::string(".") : file_path.substr(0, sep);
  auto real_dir = FileUtils::CreateNotExistDirs(weight_dir, true);
  if (!real_dir.has_value()) {
    MS_LOG(ERROR) << "Create dir " << weight_dir << " failed.";
    return RET_ERROR;
  }
  ChangeFileMode(file_path, S_IWUSR);
  std::ofstream file(file_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!file.is_open()) {
    MS_LOG(ERROR) << "Open " << file_path << " failed.";
    return RET_ERROR;
  }
  char front_info[kDataOffset] = {0};
  front_info[0] = IsSystemLittleEndian() ? 1 : 0;
  file.write(front_info, kDataOffset);
  for (const auto &entry : entries_) {
    file.seekp(entry.offset);
    if (entry.length > 0) {
      file.write(static_cast<const char *>(entry.data->data_c()), entry.length);
    }
  }
  file.flush();
  if (!file.good()) {
    MS_LOG(ERROR) << "Write " << entries_.size() << " tensors to " << file_path << " failed.";
    file.close();
    return RET_ERROR;
  }
  file.close();
  return RET_OK;
}
}  // namespace mindspore::lite
