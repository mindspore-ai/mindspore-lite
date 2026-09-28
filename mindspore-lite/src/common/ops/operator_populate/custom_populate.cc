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
#include <limits>
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl_c/custom_parameter.h"
#include "nnacl_c/split_parameter.h"
#include "infer/custom.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
using mindspore::ops::kNameCustom;
using mindspore::schema::PrimitiveType_Custom;
namespace mindspore {
namespace lite {
namespace {
constexpr char kSplitParamAttrKey[] = "0";
constexpr char kSplitSizesAttrKey[] = "1";
constexpr size_t kSplitReduceConcatAttrNum = 2;

bool GetDataFromOp(void *dst, size_t len, const ops::Custom *custom_op, std::string index) {
  if (custom_op == nullptr) {
    return false;
  }
  const auto attrs = custom_op->get_attr();
  auto iter = attrs.find(index);
  if (iter == attrs.end()) {
    return false;
  }
  const auto &data = iter->second;
  auto data_size = data.size();
  // Only the SplitReduceConcatFusion populate calls this; a partial payload
  // would leave the tail of the caller's buffer uninitialized, so require an
  // exact-size match.
  if (len != data_size) {
    return false;
  }
  std::vector<uint8_t> buf(data_size, 0);
  for (size_t i = 0; i < data_size; ++i) {
    buf[i] = static_cast<char>(data[i]);
  }
  (void)memcpy(dst, buf.data(), data_size);
  return true;
}

void DestroySplitReduceConcatOpParam(OpParameter *parameter) {
  MS_CHECK_PTR_IF_NULL(parameter);
  auto param = reinterpret_cast<SplitParameter *>(parameter);
  if (param->split_sizes_ != nullptr) {
    free(param->split_sizes_);
    param->split_sizes_ = nullptr;
  }
}

OpParameter *PopulateSplitReduceConcatFusionParam(const ops::Custom *op) {
  if (op == nullptr || op->get_attr().size() < kSplitReduceConcatAttrNum) {
    return nullptr;
  }
  SplitParameter *param = static_cast<SplitParameter *>(malloc(sizeof(SplitParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SplitParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SplitParameter));
  if (!GetDataFromOp(param, sizeof(SplitParameter), op, kSplitParamAttrKey)) {
    MS_LOG(ERROR) << "Get SplitParameter value From prim fail.";
    free(param);
    param = nullptr;
    return nullptr;
  }

  // The OpParameter header (including the destroy_func_ function pointer) must
  // never be taken from the copied bytes.
  param->op_parameter_.destroy_func_ = DestroySplitReduceConcatOpParam;
  param->op_parameter_.is_train_session_ = false;
  param->op_parameter_.is_zero_shape_ = false;
  param->op_parameter_.name_[0] = '\0';
  param->op_parameter_.thread_num_ = 0;
  param->op_parameter_.quant_type_ = 0;
  if (param->num_split_ <= 0 || param->num_split_ > std::numeric_limits<int>::max() / static_cast<int>(sizeof(int))) {
    MS_LOG(ERROR) << "The value of param->num_split_ is not correct";
    free(param);
    param = nullptr;
    return nullptr;
  }

  auto split_sizes_size = static_cast<size_t>(param->num_split_) * sizeof(int);
  // Matches MAX_MALLOC_SIZE (src/litert/inner_allocator.h) without a layering
  // violation into litert from src/common/ops.
  constexpr size_t kMaxMallocSize = static_cast<size_t>(2000) * 1024 * 1024;
  if (split_sizes_size >= kMaxMallocSize) {
    MS_LOG(ERROR) << "split_sizes_size(" << split_sizes_size << ") is too large";
    free(param);
    param = nullptr;
    return nullptr;
  }
  param->split_sizes_ = reinterpret_cast<int *>(malloc(split_sizes_size));
  if (param->split_sizes_ == nullptr) {
    MS_LOG(ERROR) << "malloc split_sizes_ failed.";
    free(param);
    param = nullptr;
    return nullptr;
  }
  if (!GetDataFromOp(param->split_sizes_, split_sizes_size, op, kSplitSizesAttrKey)) {
    MS_LOG(ERROR) << "Get split value From prim fail.";
    DestroySplitReduceConcatOpParam(reinterpret_cast<OpParameter *>(param));
    free(param);
    param = nullptr;
    return nullptr;
  }

  param->op_parameter_.type_ = PrimType_Inner_SplitReduceConcatFusion;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

OpParameter *PopulateCustomOpParameter(const BaseOperatorPtr &base_operator) {
  if (base_operator == nullptr) {
    MS_LOG(ERROR) << "base_operator is nullptr";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Custom *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not NLLLoss.";
    return nullptr;
  }

  auto type = op->get_type();
  if (type == "ShapeFusion") {
    auto param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc ShapeParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_ShapeFusion;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "GraphKernel") {
    auto param = static_cast<CustomParameter *>(malloc(sizeof(CustomParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc CustomParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(CustomParameter));
    param->op_parameter_.type_ = PrimType_Inner_GraphKernel;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "SplitReduceConcatFusion") {
    return PopulateSplitReduceConcatFusionParam(op);
  } else if (type == "EncoderLayer") {
    std::cout << "EncoderLayer populate" << std::endl;
    auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc EncoderLayer failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(OpParameter));
    param->type_ = PrimType_Inner_EncoderLayer;
    return reinterpret_cast<OpParameter *>(param);
  } else if (type == "ACL") {
    auto param = static_cast<CustomParameter *>(malloc(sizeof(CustomParameter)));
    if (param == nullptr) {
      MS_LOG(ERROR) << "malloc CustomParameter failed.";
      return nullptr;
    }
    memset(param, 0, sizeof(CustomParameter));
    // NNACL does not need to solve ACl Custom op, add this is just return a parameter object
    param->op_parameter_.type_ = PrimType_Inner_AclCustomOp;
    return reinterpret_cast<OpParameter *>(param);
  } else {
    MS_LOG(ERROR) << "Unsupported custom type: " << type;
  }
  return nullptr;
}

REG_OPERATOR_POPULATE(kNameCustom, PrimitiveType_Custom, PopulateCustomOpParameter)
}  // namespace lite
}  // namespace mindspore
