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

#include "tools/converter/adapter/acl/mapper/avgpool_fusion_mapper.h"
#include <memory>
#include <vector>
#include "tools/converter/adapter/acl/mapper/primitive_mapper_register.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "include/registry/converter_context.h"
#include "src/common/log_util.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore {
namespace lite {
using mindspore::ops::kNameAvgPool;
constexpr const char *kDivisorOverride = "divisor_override";
constexpr const char *kExclusive = "exclusive";
constexpr int kSizeHW = 20;
constexpr int kSizeHWMul = 255;
enum class AvgPoolType {
  INVALID = 0,
  AVGPOOL_2D = 1,
  AVGPOOL_3D = 2,
};
namespace {
AvgPoolType JudgeAvgPoolType(const std::vector<int> &kernel_size) {
  int size_mul = kIndex1;
  for (auto dim : kernel_size) {
    size_mul *= dim;
    if ((dim > kSizeHW) || (size_mul > kSizeHWMul)) {
      return AvgPoolType::INVALID;
    }
  }
  if (kernel_size.size() == kDim2) {
    return AvgPoolType::AVGPOOL_2D;
  } else if (kernel_size.size() == kDim3) {
    return AvgPoolType::AVGPOOL_3D;
  }
  return AvgPoolType::INVALID;
}
}  // namespace

STATUS AvgPoolFusionMapper::Mapper(const CNodePtr &cnode) {
  CHECK_NULL_RETURN(cnode);
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get value node and primitive from cnode failed.";
    return lite::RET_ERROR;
  }

  auto attr_val = src_prim->GetAttr(ops::kFmkType);
  int fmk_type = attr_val != nullptr ? GetValue<int>(attr_val) : converter::kFmkTypeTf;
  PrimitivePtr dst_prim = nullptr;
  bool is_3d = false;
  CreateTargetPrim(src_prim, fmk_type, &dst_prim, &is_3d);
  CHECK_NULL_RETURN(dst_prim);
  dst_prim->SetAttrs(src_prim->attrs());
  if (!dst_prim->HasAttr(kDivisorOverride)) {
    // default value of divisor_override is 0
    dst_prim->AddAttr(kDivisorOverride, 0);
  }
  if (src_prim->HasAttr(ops::kCountIncludePad)) {
    bool exclusive = !GetValue<bool>(src_prim->GetAttr(ops::kCountIncludePad));
    dst_prim->AddAttr(kExclusive, MakeValue(exclusive));
  } else {
    // default value of exclusive is true
    dst_prim->AddAttr(kExclusive, MakeValue(true));
  }
  if (AdjustPoolAttr(fmk_type, kNameAvgPoolFusion, dst_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust pool attr failed.";
    return lite::RET_ERROR;
  }
  if (src_prim->HasAttr(ops::kFormat)) {
    // the attr format has been changed to data_format because of dynamic(defined in gen_lite_ops.h)
    dst_prim->AddAttr(kAttrDataFormat, src_prim->GetAttr(ops::kFormat));
  }
  if (is_3d) {
    dst_prim->AddAttr(kAttrFormat, MakeValue("NCDHW"));
    if (src_prim->HasAttr(ops::kPad)) {
      dst_prim->AddAttr(ops::kPadList, src_prim->GetAttr(ops::kPad));
    }
  }

  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

void AvgPoolFusionMapper::CreateTargetPrim(const PrimitivePtr &src_prim, int fmk_type, PrimitivePtr *dst_prim,
                                           bool *is_3d) {
  if (dst_prim == nullptr) {
    MS_LOG(ERROR) << "Target prim is nullptr.";
    return;
  }
  ops::AvgPool dst_node;
  *dst_prim = dst_node.GetPrim();
  if (fmk_type == converter::kFmkTypeCaffe) {
    *dst_prim = std::make_shared<acl::Pooling>();
  } else if (fmk_type == converter::kFmkTypeOnnx) {
    ValuePtr val_ptr = src_prim->GetAttr(ops::kKernelSize);
    if (val_ptr == nullptr) {
      *dst_prim = std::make_shared<acl::GlobalAveragePool>();
    } else {
      auto kernel_size = opt::CastToInt(val_ptr);
      MS_CHECK_TRUE_RET_VOID(kernel_size.size() == kDim2 || kernel_size.size() == kDim3);
      if (JudgeAvgPoolType(kernel_size) == AvgPoolType::AVGPOOL_2D) {
        *dst_prim = std::make_shared<acl::AvgPoolV2>();
      } else if (JudgeAvgPoolType(kernel_size) == AvgPoolType::AVGPOOL_3D) {
        *dst_prim = std::make_shared<acl::AvgPool3D>();
        *is_3d = true;
      }
    }
  }
}

REGISTER_PRIMITIVE_MAPPER(kNameAvgPool, AvgPoolFusionMapper)
REGISTER_PRIMITIVE_MAPPER(kNameAvgPoolFusion, AvgPoolFusionMapper)
}  // namespace lite
}  // namespace mindspore
