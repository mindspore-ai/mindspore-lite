/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/mapper/transformation_ops_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace lite {
STATUS SpaceToBatchNDMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<acl::SpaceToBatchTF>();
  CHECK_NULL_RETURN(dst_prim);
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "SpaceToBatchNDMapper mapper failed.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS BatchToSpaceMapper::Mapper(const CNodePtr &cnode) {
  auto dst_prim = std::make_shared<acl::BatchToSpaceTF>();
  CHECK_NULL_RETURN(dst_prim);
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "BatchToSpaceMapper mapper failed.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameSpaceToBatchND, SpaceToBatchNDMapper)
REGISTER_PRIMITIVE_MAPPER(kNameBatchToSpace, BatchToSpaceMapper)
}  // namespace lite
}  // namespace mindspore
