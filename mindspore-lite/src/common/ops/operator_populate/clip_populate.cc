/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/clip_parameter.h"
#include "infer/clip.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
using mindspore::ops::kNameClip;
using mindspore::schema::PrimitiveType_Clip;
namespace mindspore {
namespace lite {
OpParameter *PopulateClipOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ClipParameter *>(PopulateOpParameter<ClipParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ClipParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::Clip *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "base_operator cast to Clip failed";
    free(param);
    return nullptr;
  }

  param->min_val_ = op->get_min();
  param->max_val_ = op->get_max();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameClip, PrimitiveType_Clip, PopulateClipOpParameter)
}  // namespace lite
}  // namespace mindspore
