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

#include "mapper/lessorequal_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/fetch_content.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "op/cmp_operator.h"

namespace mindspore {
namespace dpico {
namespace {
const size_t kOfflineArgSize2 = 2;
}  // namespace
STATUS LessOrEqualMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                              const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto lessequal_operator = std::make_unique<mapper::CmpOperator>();
  if (SetCommonAttr(cnode, lessequal_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  lessequal_operator->SetOpType(mapper::OpType::CMP);
  lessequal_operator->SetCompType(mapper::CompType::COND_LE);

  if (PushOfflineArgs(cnode, lessequal_operator.get(), kOfflineArgSize2) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(lessequal_operator));
  return RET_OK;
}
REG_MAPPER(LessEqual, LessOrEqualMapper)
}  // namespace dpico
}  // namespace mindspore
