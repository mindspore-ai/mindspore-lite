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

#include "src/litert/delegate/coreml/op/max_pooling_coreml.h"
namespace mindspore::lite {
int MaxPoolingCoreMLOp::InitParams() {
  pooling_prim_ = op_primitive_->value_as_MaxPoolFusion();
  if (pooling_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

// BuildLayer() is implemented by PoolingCoreMLOp base class
}  // namespace mindspore::lite
