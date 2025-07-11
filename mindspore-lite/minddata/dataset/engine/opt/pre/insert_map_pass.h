/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_

#include <memory>

#include "mindspore-lite/minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {
class InsertMapPass : public IRNodePass {
 public:
  /// \brief Constructor
  InsertMapPass() = default;

  /// \brief Destructor
  ~InsertMapPass() override = default;

#ifndef ENABLE_ANDROID
  /// \brief Insert map node to parse the protobuf for TFRecord.
  /// \param[in] node The TFRecordNode being visited.
  /// \param[in, out] modified Indicator if the node was changed at all.
  /// \return The status code.
  Status Visit(std::shared_ptr<TFRecordNode> node, bool *const modified) override;
#endif
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_
