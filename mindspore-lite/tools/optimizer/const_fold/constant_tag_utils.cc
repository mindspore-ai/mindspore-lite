/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/const_fold/constant_tag_utils.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "tools/converter/ms_depend/helper.h"
#include "ir/anf.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/common/node_util.h"
#include "tools/common/tensor_util.h"
#include "src/common/context_util.h"
#include "src/common/ops/populate/populate_register.h"
#include "src/executor/kernel_exec.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/inner_context.h"
#include "src/tensor.h"
#include "src/tensorlist.h"
#include "src/common/ops/anf_utils.h"
#include "src/litert/infer_manager.h"
#include "tools/optimizer/graph/lite_tensor_extractor.h"
#include "tools/optimizer/common/helper.h"
#include "tools/lite_exporter/fetch_content.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_w.h"

using mindspore::lite::Tensor;

namespace mindspore {
namespace opt {
namespace {
constexpr int kElementShapeIndex = 1;
constexpr int kElementNumOffset = 2;
constexpr int kBasicInfoMinSize = 3;

bool CheckTensorListIsValid(const std::vector<uint8_t> &tensorlist_data) {
  if (tensorlist_data.empty()) {
    return true;
  }
  auto basic_data_size = tensorlist_data.size() / sizeof(int);
  auto *data = reinterpret_cast<const int *>(tensorlist_data.data());
  if (basic_data_size < static_cast<size_t>(kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length illegal, which should be at least 3, now is " << basic_data_size;
    return false;
  }
  if (data[kElementShapeIndex] < 0 || INT_ADD_OVERFLOW(data[kElementShapeIndex], kBasicInfoMinSize)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  if (static_cast<size_t>((data[kElementShapeIndex] + kBasicInfoMinSize)) > basic_data_size) {
    MS_LOG(ERROR) << "tensorlist data length illegal. current tensorlist data length should be at least "
                  << (data[kElementShapeIndex] + kBasicInfoMinSize) << ", but now is " << basic_data_size;
    return false;
  }
  auto element_num = data[data[kElementShapeIndex] + kElementNumOffset];
  if (element_num > 0 && INT_ADD_OVERFLOW(element_num, 1)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  auto shape_once = data[kElementShapeIndex] + 1;
  auto shape_group_num = element_num < 0 ? 1 : element_num + 1;
  if (INT_MUL_OVERFLOW(shape_once, shape_group_num)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT mul overflow.";
    return false;
  }
  auto shape_info_size = shape_once * shape_group_num;
  if (INT_ADD_OVERFLOW(shape_info_size, kElementNumOffset)) {
    MS_LOG(ERROR) << "tensorlist data length is too big, INT add overflow.";
    return false;
  }
  size_t real_data_size = static_cast<size_t>(shape_info_size + kElementNumOffset);
  if (real_data_size != basic_data_size) {
    MS_LOG(ERROR) << "current tensorlist data length should be " << real_data_size << ", but now is "
                  << basic_data_size;
    return false;
  }
  return true;
}

TensorPtr GetCNodeTensorListVarInput(const lite::DataInfo &data_info) {
  auto tensor_list = std::make_shared<lite::TensorList>(data_info.shape_, std::vector<int>{});
  if (tensor_list == nullptr) {
    MS_LOG(ERROR) << "new a lite tensor list failed";
    return nullptr;
  }
  if (data_info.data_.empty()) {
    return tensor_list;
  }
  if (!CheckTensorListIsValid(data_info.data_)) {
    MS_LOG(ERROR) << "tensor list is invalid.";
    return nullptr;
  }
  auto status = tensor_list->Decode(reinterpret_cast<const int *>(data_info.data_.data()), data_info.data_.size());
  if (status != RET_OK) {
    MS_LOG(ERROR) << "decode tensor list failed.";
    return nullptr;
  }
  return tensor_list;
}
}  // namespace

int ConstantTagUtils::GetTensorDataNBytes(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor->device_address());
  if (tensor->device_address()->data() != nullptr) {
    return static_cast<int>(tensor->device_address()->data()->nbytes());
  } else {
    return tensor->DataNBytes();
  }
}

TensorPtr ConstantTagUtils::CreateTensorFromData(const lite::DataInfo &data_info, const bool &has_inferred,
                                                 const mindspore::Format &format) {
  if (data_info.data_type_ == static_cast<int>(kObjectTypeTensorType)) {
    auto tensor = GetCNodeTensorListVarInput(data_info);
    MS_CHECK_TRUE_MSG(tensor != nullptr, nullptr, "tensor is nullptr.");
    tensor->set_format((Format)(format));
    if (!has_inferred) {
      tensor->set_shape({-1});
    }
    return tensor;
  } else {
    auto tensor = std::make_shared<lite::Tensor>(TypeId(data_info.data_type_), data_info.shape_);
    MS_CHECK_TRUE_MSG(tensor != nullptr, nullptr, "tensor is nullptr.");
    tensor->set_format((Format)(format));
    if (!has_inferred) {
      tensor->set_shape({-1});
    }
    if (!data_info.data_.empty()) {
      tensor->MallocData();
      if (memcpy_s(tensor->MutableData(), tensor->Size(), data_info.data_.data(), data_info.data_.size()) != EOK) {
        MS_LOG(ERROR) << "memcpy data failed.";
        return nullptr;
      }
    }
    return tensor;
  }
}

int ConstantTagUtils::FetchDataFromCNodeAttr(const CNodePtr &cnode, const AbstractBasePtr &abstract,
                                             lite::DataInfo *data_info) {
  MS_CHECK_TRUE_MSG(data_info != nullptr, lite::RET_NULL_PTR, "data_info is nullptr");
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr");
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");
  if (lite::FetchDataFromAbstract(abstract, data_info) != lite::RET_OK) {
    MS_LOG(ERROR) << "FetchDataFromAbstract failed. cnode: " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto value = cnode->GetAttr(lite::kNameCNodeValueAttr);
  if (value != nullptr) {
    MS_CHECK_TRUE_MSG(value != nullptr, lite::RET_NULL_PTR, "value is nullptr");
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_CHECK_TRUE_MSG(tensor != nullptr, lite::RET_NULL_PTR, "tensor is nullptr");
    data_info->data_type_ = tensor->data_type_c();
    auto tensor_value_nbytes = GetTensorDataNBytes(tensor);
    data_info->data_.resize(tensor_value_nbytes);
    if (memcpy_s(data_info->data_.data(), tensor_value_nbytes, tensor->data_c(), tensor_value_nbytes) != EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

int ConstantTagUtils::GetCNodeVarInput(const CNodePtr &cnode, const size_t &index,
                                       std::vector<TensorPtr> *var_ms_inputs) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(var_ms_inputs != nullptr, lite::RET_NULL_PTR, "var_ms_inputs is nullptr.");
  if (!utils::isa<CNodePtr>(cnode->input(index))) {
    MS_LOG(ERROR) << "The " << index << "th input for " << cnode->fullname_with_scope() << "should be cnode.";
    return lite::RET_ERROR;
  }

  bool has_inferred{false};
  auto ret = DetermineCertainVarInputHasInferred(cnode, index, &has_inferred);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "determine infer flag failed.");
  Format format{mindspore::NHWC};
  ret = opt::DetermineCertainVarInputFormat(cnode, index, &format);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "determine format failed.");

  auto abstract = opt::GetCNodeInputAbstract(cnode, index);
  MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr.");
  if (utils::isa<abstract::AbstractTensor>(abstract)) {
    lite::DataInfo data_info;
    if (FetchDataFromCNodeAttr(cnode->input(index)->cast<CNodePtr>(), abstract, &data_info) != RET_OK) {
      MS_LOG(ERROR) << "FetchDataFromAbstract failed.";
      return lite::RET_ERROR;
    }
    auto tensor = CreateTensorFromData(data_info, has_inferred, format);
    MS_CHECK_TRUE_MSG(tensor != nullptr, lite::RET_NULL_PTR, "CreateTensorFromData failed.");
    var_ms_inputs->emplace_back(tensor);
  } else if (utils::isa<abstract::AbstractTuple>(abstract)) {
    auto tuple = std::reinterpret_pointer_cast<abstract::AbstractTuple>(abstract);
    MS_CHECK_TRUE_MSG(tuple != nullptr, lite::RET_NULL_PTR, "tuple is nullptr.");
    for (const auto &element : tuple->elements()) {
      lite::DataInfo data_info;
      if (lite::FetchDataFromAbstract(element, &data_info) != RET_OK) {
        MS_LOG(ERROR) << "FetchDataFromAbstract failed.";
        return RET_ERROR;
      }
      auto tensor = CreateTensorFromData(data_info, has_inferred, format);
      MS_CHECK_TRUE_MSG(tensor != nullptr, lite::RET_NULL_PTR, "CreateTensorFromData failed.");
      var_ms_inputs->emplace_back(tensor);
    }
  } else {
    MS_LOG(ERROR) << "abstract should be a AbstractTensor or AbstractTuple.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int ConstantTagUtils::GetCNodeInputTensors(const CNodePtr &cnode, std::vector<TensorPtr> *inputs,
                                           converter::FmkType fmk_type, bool train_flag, bool copy_data) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, lite::RET_NULL_PTR, "cnode is nullptr!");
  MS_CHECK_TRUE_MSG(inputs != nullptr, lite::RET_NULL_PTR, "inputs is nullptr!");
  auto origin_inputs = cnode->inputs();
  if (lite::RemoveIfDepend(cnode) != RET_OK) {
    MS_LOG(ERROR) << "remove depend failed.";
    return RET_ERROR;
  }
  if (lite::RemoveIfMakeTuple(cnode)) {
    MS_LOG(ERROR) << "remove makeTuple failed.";
    return RET_ERROR;
  }
  RemoveIfMonad(cnode);

  for (size_t i = 1; i < cnode->size(); ++i) {
    if (utils::isa<CNodePtr>(cnode->input(i))) {
      std::vector<TensorPtr> var_inputs;
      if (GetCNodeVarInput(cnode, i, &var_inputs) != RET_OK) {
        MS_LOG(ERROR) << "get var inputs failed.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
      inputs->insert(inputs->end(), var_inputs.begin(), var_inputs.end());
    } else {
      std::vector<TensorPtr> const_inputs;
      if (LiteTensorExtractor::GetCNodeConstInput(cnode, i, fmk_type, train_flag, copy_data, &const_inputs) != RET_OK) {
        MS_LOG(ERROR) << "get const inputs failed.";
        cnode->set_inputs(origin_inputs);
        return RET_ERROR;
      }
      inputs->insert(inputs->end(), const_inputs.begin(), const_inputs.end());
    }
  }
  cnode->set_inputs(origin_inputs);
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
