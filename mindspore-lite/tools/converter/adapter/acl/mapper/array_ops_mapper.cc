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

#include "tools/converter/adapter/acl/mapper/array_ops_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_t.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace lite {
using mindspore::ops::kNameDynamicShape;
using mindspore::ops::kNameTensorShape;

DynamicShapeMapper::DynamicShapeMapper() : PrimitiveMapper(kNameDynamicShape) {}

STATUS DynamicShapeMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  ops::Shape shape_op;
  auto dst_prim = shape_op.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "Dynamic shape Mapper mapper failed.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

TensorShapeMapper::TensorShapeMapper() : PrimitiveMapper(kNameTensorShape) {}

STATUS TensorShapeMapper::Mapper(const CNodePtr &cnode) {
  ops::Shape shape_op;
  auto dst_prim = shape_op.GetPrim();
  if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
    MS_LOG(ERROR) << "Tensor shape Mapper mapper failed.";
    return RET_ERROR;
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameDynamicShape, DynamicShapeMapper)
REGISTER_PRIMITIVE_MAPPER(kNameTensorShape, TensorShapeMapper)
}  // namespace lite
}  // namespace mindspore
