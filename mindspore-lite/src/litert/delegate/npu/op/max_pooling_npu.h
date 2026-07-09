/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_MAX_POOLING_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_MAX_POOLING_NPU_H_
#include <vector>
#include <string>
#include "src/litert/delegate/npu/op/pooling_npu.h"

namespace mindspore::lite {
/**
 * @brief Max Pooling NPU operation class
 * Inherits from PoolingNPUOp template base class with NPUPoolingType::MAX
 * SetPoolingParam() and Init() common logic is extracted to the base class
 */
class MaxPoolingNPUOp : public PoolingNPUOp<NPUPoolingType::MAX> {
 public:
  MaxPoolingNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                  const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : PoolingNPUOp<NPUPoolingType::MAX>(primitive, in_tensors, out_tensors, name) {}

  ~MaxPoolingNPUOp() override = default;  // Base class handles destructor

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  // Init(), SetNPUInputs(), GetNPUOp() are implemented by base class

 protected:
  const void *GetPoolingPrimitive(const schema::Primitive *primitive) const override {
    return primitive->value_as_MaxPoolFusion();
  }

  schema::ActivationType GetActivationType(const void *pooling_prim) const override {
    auto *prim = static_cast<const schema::MaxPoolFusion *>(pooling_prim);
    return prim->activation_type();
  }
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_MAX_POOLING_NPU_H_
