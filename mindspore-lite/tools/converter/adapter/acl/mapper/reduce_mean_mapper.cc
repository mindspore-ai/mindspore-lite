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

#include "tools/converter/adapter/acl/mapper/reduce_mean_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"

namespace mindspore {
namespace lite {
STATUS ReduceMeanMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
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
  src_prim->AddAttr("noop_with_empty_axes", MakeValue<bool>(false));
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameReduceMean, ReduceMeanMapper)
REGISTER_PRIMITIVE_MAPPER(kNameReduceMax, ReduceMeanMapper)
REGISTER_PRIMITIVE_MAPPER(kNameReduceMin, ReduceMeanMapper)
}  // namespace lite
}  // namespace mindspore
