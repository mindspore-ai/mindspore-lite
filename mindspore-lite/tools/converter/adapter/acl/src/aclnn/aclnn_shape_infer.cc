/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "tools/converter/adapter/acl/src/aclnn/aclnn_shape_infer.h"
#include "tools/converter/adapter/acl/src/aclnn/aclnn_utils.h"
#include "tools/converter/optimizer_manager.h"
#include "common/common.h"

namespace mindspore {
namespace opt {

AclnnShapeInfer::AclnnShapeInfer(const std::shared_ptr<ConverterPara> &param,
                                 const lite::acl::AclModelOptionCfg &user_options_cfg)
    : param_(param), user_options_cfg_(user_options_cfg) {}

STATUS AclnnShapeInfer::InferStaticShape(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr");
  return lite::RunOptimizerPass(func_graph, {"ConstantTag"}) ? lite::RET_OK : lite::RET_ERROR;
}

STATUS AclnnShapeInfer::InferAllDynamicShape(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr");
  OrderShapes shapes;
  auto &build_options = user_options_cfg_.build_options_map;
  auto ret = AclnnUtils::ParseInputShape(build_options[lite::kInputShapeKey], &shapes);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Parse input_shape failed.");

  std::vector<std::vector<int64_t>> dim_groups;
  ret = AclnnUtils::ParseDynamicDims(build_options[lite::kDynamicDimsSearchKey], &dim_groups);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Parse ge.dynamicDims failed.");

  ret = AclnnUtils::CheckDims(shapes, dim_groups);
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "input_shape or ge.dynamicDims is invalid");
  for (const auto &group : dim_groups) {
    OrderShapes applied_shapes;
    ret = AclnnUtils::ApplyDims(shapes, group, &applied_shapes);
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Apply dims failed.");

    ret = AclnnUtils::ApplyInputShape(func_graph, {applied_shapes.begin(), applied_shapes.end()});
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Apply input shape failed.");

    ret = InferStaticShape(func_graph);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Infer static shape for " << group << " failed.";
      return lite::RET_ERROR;
    }
  }

  // apply origin dynamic shape
  ret = AclnnUtils::ApplyInputShape(func_graph, {shapes.begin(), shapes.end()});
  MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Apply origin dynamic shape failed.");

  return lite::RET_OK;
}

STATUS AclnnShapeInfer::InferShape(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr");
  auto &build_options = user_options_cfg_.build_options_map;
  auto dynamic_find = build_options.find(lite::kDynamicDimsSearchKey) != build_options.end();
  auto shape_find = build_options.find(lite::kInputShapeKey) != build_options.end();
  if (dynamic_find && shape_find) {
    auto ret = InferAllDynamicShape(func_graph);
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Infer all dynamic shape failed.");
  } else if (dynamic_find) {
    MS_LOG(ERROR) << "find " << lite::kDynamicDimsSearchKey << ". but cannot find " << lite::kInputShapeKey;
    return lite::RET_ERROR;
  } else {
    auto ret = InferStaticShape(func_graph);
    MS_CHECK_TRUE_MSG(ret == lite::RET_OK, ret, "Infer static shape failed.");
  }
  return lite::RET_OK;
}

}  // namespace opt
}  // namespace mindspore
