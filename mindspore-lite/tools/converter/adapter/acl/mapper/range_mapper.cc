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

#include "tools/converter/adapter/acl/mapper/range_mapper.h"
#include <string>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "src/common/log_util.h"
#include "infer/range_v2.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_r.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kNameInputNums = 4;
}

using mindspore::ops::kNameRange;

RangeMapper::RangeMapper() : PrimitiveMapper(kNameRange) {}

STATUS RangeMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  int input_num = cnode->size();
  if (input_num == kNameInputNums) {
    ops::RangeV2 rangev2;
    auto dst_prim = rangev2.GetPrim();
    if (MoveAttrMap(cnode, dst_prim) != RET_OK) {
      MS_LOG(ERROR) << "RangeV2 mapper failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameRange, RangeMapper)
}  // namespace lite
}  // namespace mindspore
