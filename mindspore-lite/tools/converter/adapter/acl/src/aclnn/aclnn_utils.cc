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

#include "tools/converter/adapter/acl/src/aclnn/aclnn_utils.h"
#include "tools/common/string_util.h"
#include "common/common.h"
#include "common/utils.h"

namespace mindspore {
namespace opt {

STATUS AclnnUtils::ParseInputShape(const std::string &shape_str, OrderShapes *shapes) {
  MS_CHECK_TRUE_MSG(shapes != nullptr, lite::RET_NULL_PTR, "shapes is nullptr");
  auto shape_split = lite::StrSplit(shape_str, ";");
  for (auto &pair_str : shape_split) {
    auto pair_split = lite::StrSplit(pair_str, ":");
    MS_CHECK_TRUE_MSG(pair_split.size() == 2, lite::RET_ERROR, "parse input_shape failed.");
    auto input_name = pair_split.front();
    auto dims_str = lite::StrSplit(pair_split.back(), ",");
    std::vector<int64_t> dims;
    for (auto &dim_str : dims_str) {
      int64_t dim;
      MS_CHECK_TRUE_MSG(lite::ConvertStrToInt(dim_str, &dim), lite::RET_ERROR, "parse dim failed.");
      dims.push_back(dim);
    }
    shapes->emplace_back(input_name, dims);
  }
  return lite::RET_OK;
}

STATUS AclnnUtils::ParseDynamicDims(const std::string dims_str, std::vector<std::vector<int64_t>> *dim_groups) {
  MS_CHECK_TRUE_MSG(dim_groups != nullptr, lite::RET_NULL_PTR, "dims is nullptr");
  auto groups = lite::StrSplit(dims_str, ";");
  for (const auto &group : groups) {
    if (group.empty()) {
      continue;
    }

    auto dims = lite::StrSplit(group, ",");
    MS_CHECK_TRUE_MSG(!dims.empty(), lite::RET_ERROR, "parse dynamic failed. group is empty.");
    std::vector<int64_t> dims_num;
    for (const auto &dim_str : dims) {
      int64_t dim;
      MS_CHECK_TRUE_MSG(lite::ConvertStrToInt(dim_str, &dim), lite::RET_ERROR, "parse dim failed.");
      dims_num.push_back(dim);
    }
    dim_groups->push_back(std::move(dims_num));
  }
  return lite::RET_OK;
}

STATUS AclnnUtils::CheckDims(const OrderShapes &dyn_shapes, const std::vector<std::vector<int64_t>> &dim_groups) {
  size_t dyn_dim_count =
    std::accumulate(dyn_shapes.begin(), dyn_shapes.end(), 0, [](auto init, const auto &shape_pair) {
      return init + std::count_if(shape_pair.second.begin(), shape_pair.second.end(), [](auto i) { return i == -1; });
    });

  for (const auto &group : dim_groups) {
    MS_CHECK_TRUE_MSG(dyn_dim_count == group.size(), lite::RET_ERROR, "Dynamic shape config mismatch.");
  }
  return lite::RET_OK;
}

STATUS AclnnUtils::ApplyDims(const OrderShapes &dyn_shapes, const std::vector<int64_t> &dim_group,
                             OrderShapes *shapes) {
  MS_CHECK_TRUE_MSG(shapes != nullptr, lite::RET_NULL_PTR, "dims is nullptr");
  auto it = dim_group.begin();
  auto end = dim_group.end();

  for (const auto &item : dyn_shapes) {
    std::vector<int64_t> shape = item.second;
    for (auto &dim : shape) {
      if (dim == -1) {
        if (it >= end) {
          MS_LOG(ERROR) << "Apply dims failed. dims in group " << dim_group << " is not enough.";
          return lite::RET_ERROR;
        }
        dim = *(it++);
      }
    }
    shapes->emplace_back(item.first, shape);
  }
  if (it != dim_group.end()) {
    MS_LOG(ERROR) << "Apply dim group: " << dim_group << " failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AclnnUtils::ApplyInputShape(const FuncGraphPtr &func_graph,
                                   const std::map<std::string, std::vector<int64_t>> &shapes) {
  MS_CHECK_TRUE_MSG(func_graph != nullptr, lite::RET_NULL_PTR, "func_graph is nullptr.");
  auto inputs = func_graph->get_inputs();
  for (const auto &input : inputs) {
    MS_CHECK_TRUE_MSG(input != nullptr, lite::RET_NULL_PTR, "input is nullptr");
    auto input_name = input->fullname_with_scope();
    auto find = shapes.find(input_name);
    if (find == shapes.end()) {
      MS_LOG(ERROR) << "cannot find shape for input: " << input_name;
      return lite::RET_ERROR;
    }

    auto abstract = input->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
    if (!abstract->isa<abstract::AbstractTensor>()) {
      MS_LOG(ERROR) << "cannot apply shape for " << input_name << ", is's not a tensor.";
      return lite::RET_ERROR;
    }
    auto abstract_tensor = abstract->cast<abstract::AbstractTensorPtr>();
    MS_CHECK_TRUE_MSG(abstract_tensor != nullptr, lite::RET_NULL_PTR, "abstract_tensor is nullptr");
    auto shape_ptr = abstract_tensor->shape();
    MS_CHECK_TRUE_MSG(shape_ptr != nullptr, lite::RET_NULL_PTR, "shape_ptr is nullptr");
    shape_ptr->set_shape(find->second);
  }
  return lite::RET_OK;
}

std::vector<int64_t> AclnnUtils::GetDynDimPosInGlobalGroup(const OrderShapes &dyn_shapes, int64_t index) {
  std::vector<int64_t> ret;
  size_t count =
    std::accumulate(dyn_shapes.begin(), dyn_shapes.begin() + index, 0, [](auto init, const auto &shape_pair) {
      return init + std::count_if(shape_pair.second.begin(), shape_pair.second.end(), [](auto i) { return i == -1; });
    });

  for (auto i : dyn_shapes[index].second) {
    if (i == -1) {
      ret.push_back(count++);
    }
  }
  return ret;
}

}  // namespace opt
}  // namespace mindspore
