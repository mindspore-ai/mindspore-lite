/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef DELEGATE_MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_
#define DELEGATE_MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_

#include <vector>
#include <memory>
#include <set>
#include <utility>
#include <string>
#include "include/api/types.h"
#include "include/errorcode.h"

#include "extendrt/delegate/ascend_acl/model_process.h"
#include "extendrt/delegate/ascend_acl/acl_env_guard.h"
#include "extendrt/delegate/ascend_acl/acl_model_options.h"
#include "extendrt/delegate/ascend_acl/profiling.h"
#include "mindspore/core/include/mindapi/base/type_id.h"
namespace mindspore {

class ModelInfer {
 public:
  explicit ModelInfer(const std::shared_ptr<AclModelOptions> &options);
  ~ModelInfer() = default;

  Status Init();
  Status Finalize(bool process_ends = false);
  Status Load(const void *om_data, size_t om_data_size);
  Status Inference(const std::vector<mindspore::MSTensor> &inputs, std::vector<mindspore::MSTensor> *outputs);
  bool UpdateWeights(const std::vector<MSTensor> &inputs);
  std::vector<Format> GetInputFormat();
  const std::vector<std::vector<int64_t>> GetOutputShape();
  const std::vector<std::vector<int64_t>> GetInputShape();
  const std::vector<TypeId> GetInputDataType();
  const std::vector<TypeId> GetOutputDataType();
  uint64_t GetSharableHandle() { return sharable_handle_; }

  Status Resize(const std::vector<std::vector<int64_t>> &new_shapes);

 private:
  bool init_flag_;
  std::string device_type_;
  aclrtContext context_;
  aclrtStream stream_;
  std::shared_ptr<AclModelOptions> options_;
  ModelProcess model_process_;
  Profiling profiling_;
  std::shared_ptr<AclEnvGuard> acl_env_;
  uint64_t sharable_handle_ = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_MODEL_MODEL_INFER_H_
