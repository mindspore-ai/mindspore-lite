/**
 * Copyright 2022-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MAX_POOLING_COREML_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MAX_POOLING_COREML_H_

#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include "src/litert/delegate/coreml/op/pooling_coreml.h"

namespace mindspore::lite {
/**
 * @brief Max Pooling CoreML operation class
 * Inherits from PoolingCoreMLOp template base class with PoolingType::MAX
 * BuildLayer() common logic is extracted to the base class
 */
class MaxPoolingCoreMLOp : public PoolingCoreMLOp<PoolingType::MAX> {
 public:
  MaxPoolingCoreMLOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : PoolingCoreMLOp<PoolingType::MAX>(primitive, in_tensors, out_tensors, name) {}

  int InitParams() override;

  // BuildLayer() is implemented by base class

 protected:
  const void *GetPoolingPrimitive() const override { return pooling_prim_; }

 private:
  const schema::MaxPoolFusion *pooling_prim_ = nullptr;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_COREML_OP_MAX_POOLING_COREML_H_
