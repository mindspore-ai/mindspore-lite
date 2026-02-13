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

#define USE_DEPRECATED_API
#include "tools/optimizer/const_fold/constant_tag_node_infer.h"
#include <vector>
#include "ir/anf.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "src/tensor.h"
#include "tools/optimizer/const_fold/constant_tag_utils.h"

namespace mindspore {
namespace opt {

int ConstantTagNodeInfer::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs,
                                               converter::FmkType fmk_type, bool train_flag, bool copy_data) {
  return ConstantTagUtils::GetCNodeInputTensors(cnode, inputs, fmk_type, train_flag, copy_data);
}

}  // namespace opt
}  // namespace mindspore
