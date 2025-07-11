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

#include "tools/converter/adapter/acl/mapper/stridedslice_mapper.h"
#include <memory>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "ops_utils/op_utils.h"
#include "nnacl/op_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_s.h"
namespace mindspore {
namespace lite {
using mindspore::ops::kNameStridedSlice;

StridedSliceMapper::StridedSliceMapper() : PrimitiveMapper(kNameStridedSlice) {}
STATUS StridedSliceMapper::Mapper(const CNodePtr &cnode) {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int64_t fmk_type = attr_val != nullptr ? GetValue<int64_t>(attr_val) : converter::kFmkTypeTf;
  if (static_cast<converter::FmkType>(fmk_type) == converter::kFmkTypeOnnx) {
    auto dst_prim = std::make_shared<acl::StridedSliceV2>();
    MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
    dst_prim->SetAttrs(src_prim->attrs());
    value_node->set_value(dst_prim);
  }
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameStridedSlice, StridedSliceMapper)
}  // namespace lite
}  // namespace mindspore
