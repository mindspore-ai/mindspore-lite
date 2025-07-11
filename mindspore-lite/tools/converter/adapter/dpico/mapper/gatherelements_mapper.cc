/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "mapper/gatherelements_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "op/gather_elements_operator.h"

namespace mindspore {
namespace dpico {
STATUS GatherElementsMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                                 const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto gather_elements_prim = api::utils::cast<api::SharedPtr<ops::GatherD>>(prim);
  MS_ASSERT(gather_elements_prim != nullptr);

  auto gather_elements_operator = std::make_unique<mapper::GatherElementsOperator>();
  if (SetCommonAttr(cnode, gather_elements_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  gather_elements_operator->SetOpType(mapper::OpType::GATHER_ELEMENTS);
  if (gather_elements_prim->GetAttr("dims") != nullptr) {
    gather_elements_operator->SetGatherElementsAxis(api::GetValue<int32_t>(gather_elements_prim->GetAttr("dims")));
  }

  base_operators->push_back(std::move(gather_elements_operator));
  return RET_OK;
}
REG_MAPPER(GatherD, GatherElementsMapper)
}  // namespace dpico
}  // namespace mindspore
