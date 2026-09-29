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

#ifndef MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_SHARED_WEIGHT_STORE_H_
#define MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_SHARED_WEIGHT_STORE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mindspore {
namespace tensor {
class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
}  // namespace tensor

namespace lite {
/**
 * SharedWeightStore collects the weight data of multiple models into ONE external data file
 * ("<output_root>/variables/data_0") and deduplicates identical tensors by (data_type, shape,
 * content hash). Every mindir graph converted in the same batch records the returned
 * (location, offset, length) in its TensorProto.external_data, so the runtime can load all the
 * graphs of the batch from this single weight file. The file layout is compatible with the
 * existing external-data loading: a 64-byte header whose first byte flags little-endian, and
 * tensor data 64-byte aligned afterwards.
 *
 * Nothing is written while the batch is being converted: the store keeps the tensors alive and
 * assigns their offsets in memory. Save() writes the file once every model converted, so a failed
 * batch leaves no partial weight file behind.
 */
class SharedWeightStore {
 public:
  explicit SharedWeightStore(const std::string &output_root);
  ~SharedWeightStore();

  /// Look up the tensor in the dedup table; on miss, reserve a 64-byte aligned region for it.
  int OrAllocate(const std::string &param_name, const tensor::TensorPtr &data, int64_t *offset, int64_t *length);

  /// The output root (the directory holding kSharedWeightLocation) relative to mindir_dir, e.g.
  /// "../.." for a mindir stored in a sub directory of the output root. It prefixes
  /// external_data.location, which is resolved relative to the directory of the mindir itself.
  std::string RelativeRootPath(const std::string &mindir_dir) const;

  /// Write every buffered tensor to `file_path`, creating its parent directory.
  int Save(const std::string &file_path);

  static constexpr char kSharedWeightLocation[] = "variables/data_0";

 private:
  struct WeightEntry {
    tensor::TensorPtr data;
    int64_t offset = 0;
    int64_t length = 0;
  };
  static constexpr int64_t kDataOffset = 64;

  std::string root_path_;
  int64_t cursor_ = kDataOffset;
  std::vector<WeightEntry> entries_;
  // content-addressed key: "<data_type>|<dims...>|<content hash>" -> offset in the weight file
  std::unordered_map<std::string, int64_t> dedup_map_{};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_MINDIR_EXPORTER_SHARED_WEIGHT_STORE_H_
