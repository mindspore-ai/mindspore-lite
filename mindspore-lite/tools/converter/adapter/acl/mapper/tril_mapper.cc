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

#include "tools/converter/adapter/acl/mapper/tril_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "tools/converter/adapter/acl/common/utils.h"
#include "infer/tensor_copy.h"
#include "src/common/log_util.h"
#include "mindspore/ops/op_def/op_name.h"
namespace mindspore {
namespace lite {
namespace {
const size_t kNumInputSize = 3;
const size_t kNumInputIndex2 = 2;
const size_t kNumCnodeInputIndex = 1;
}  // namespace
STATUS TrilMapper::Mapper(const CNodePtr &cnode) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_ERROR, "cnode is nullptr.");
  if (cnode->size() != kNumInputSize) {
    MS_LOG(ERROR) << "cnode input size is " << cnode->size() << ", not equal " << kNumInputSize;
    return RET_ERROR;
  }
  auto status = opt::AdjustInputToCnode(cnode, kNumCnodeInputIndex);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "AdjustInputToCnode failed.";
    return RET_ERROR;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "GetValueNodeAndPrimFromCnode failed.";
    return RET_ERROR;
  }
  if (value_node == nullptr || src_prim == nullptr) {
    MS_LOG(ERROR) << "value_node or src_prim is nullptr.";
    return RET_ERROR;
  }
  auto input_node = cnode->input(kNumInputIndex2);
  if (input_node == nullptr) {
    MS_LOG(ERROR) << "The second input is nullptr.";
    return RET_ERROR;
  }
  if (!utils::isa<ParameterPtr>(input_node)) {
    MS_LOG(ERROR) << "The input_node is not a parameter node. But the Tril ops only accepts the second input as a "
                     "parameter node. Please change the second input to a constant input.";
    return RET_ERROR;
  }
  auto diagonal_param = input_node->cast<ParameterPtr>();
  if (diagonal_param == nullptr) {
    MS_LOG(ERROR) << "The input_node does not cast to ParameterPtr.";
    return RET_ERROR;
  }
  auto diagonal_node = diagonal_param->default_param();
  if (diagonal_node == nullptr) {
    MS_LOG(ERROR) << "diagonal_node is nullptr.";
    return RET_ERROR;
  }
  auto diagonal_value = std::dynamic_pointer_cast<tensor::Tensor>(diagonal_node);
  if (diagonal_value == nullptr) {
    MS_LOG(ERROR) << "diagonal_value is nullptr.";
    return RET_ERROR;
  }
  auto diagonal_data = reinterpret_cast<int32_t *>(diagonal_value->data_c());
  if (diagonal_data == nullptr) {
    MS_LOG(ERROR) << "diagonal_data is nullptr.";
    return RET_ERROR;
  }
  if (diagonal_value->ElementsNum() != 1) {
    MS_LOG(ERROR) << "diagonal_value elements num is " << diagonal_value->ElementsNum();
    return RET_ERROR;
  }
  auto diagonal = diagonal_data[0];
  MS_LOG(INFO) << "diagonal: " << diagonal;
  cnode->set_inputs({cnode->input(0), cnode->input(1)});
  auto dst_prim = std::make_shared<acl::Tril>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make tril op failed.";
    return RET_ERROR;
  }
  dst_prim->SetAttrs(src_prim->attrs());
  dst_prim->AddAttr("diagonal", MakeValue(static_cast<int64_t>(diagonal)));
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}
REGISTER_PRIMITIVE_MAPPER(kNameTril, TrilMapper)
}  // namespace lite
}  // namespace mindspore
