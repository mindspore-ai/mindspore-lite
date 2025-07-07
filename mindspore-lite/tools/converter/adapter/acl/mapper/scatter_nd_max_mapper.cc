/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/scatter_nd_max_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"

namespace mindspore {
namespace lite {
namespace {
constexpr size_t kNumInputSize = 4;
constexpr size_t kNumCnodeInputIndex1 = 1;  // var
constexpr size_t kNumCnodeInputIndex3 = 3;  // updates
}  // namespace
STATUS ScatterNdMaxMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  if (cnode->size() != kNumInputSize) {
    MS_LOG(ERROR) << "cnode input size is " << cnode->size() << ", not equal kNumInputSize!";
    return RET_ERROR;
  }
  auto status = opt::AdjustInputToCnode(cnode, kNumCnodeInputIndex1);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "AdjustInputToCnode failed!";
    return RET_ERROR;
  }
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "GetValueNodeAndPrimFromCnode failed!";
    return RET_ERROR;
  }
  if (value_node == nullptr || src_prim == nullptr) {
    MS_LOG(ERROR) << "value_node or src_prim is nullptr!";
    return RET_ERROR;
  }
  auto dst_prim = std::make_shared<acl::ScatterNdMax>();
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "make ScatterNdMax failed!";
    return RET_ERROR;
  }
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameScatterNdMax, ScatterNdMaxMapper)
}  // namespace lite
}  // namespace mindspore
