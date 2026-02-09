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

#include "src/common/ops/primitive/reduce_external.h"
#include "utils/check_convert_utils.h"
#include "ir/kernel_tensor_value.h"
#include "primitive/op_name.h"

namespace mindspore::ops::lite {
void ReduceFuncCheckAxisInferImpl(const PrimitivePtr &prim, std::vector<int64_t> *axis, const size_t dim) {
  MS_EXCEPTION_IF_NULL(axis);
  int64_t dim_ = static_cast<int64_t>(dim);
  for (size_t i = 0; i < axis->size(); i++) {
    if (dim == 0) {
      if ((axis->at(i) != -1 && axis->at(i) != 0)) {
        MS_EXCEPTION(ValueError) << "For '" << prim->name()
                                 << "', 'axis' must be in [-1, 0]. But got 'axis' = " << axis->at(i) << ".";
      }
      axis->at(i) = 0;
      continue;
    }
    if (axis->at(i) < -dim_ || axis->at(i) >= dim_) {
      MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', 'axis' must be in [" << -dim_ << ", " << dim_
                               << "). But got 'axis' = " << axis->at(i) << ".";
    }
    if (axis->at(i) >= -dim_ && axis->at(i) < 0) {
      axis->at(i) += dim_;
    }
  }
}

ShapeVector ReduceFuncCalShapeInferImpl(const PrimitivePtr &, const ShapeVector &x_shape,
                                        const std::vector<int64_t> &axis, bool keep_dims_value) {
  ShapeVector out_shape;
  ShapeVector axis_value;
  (void)axis_value.insert(axis_value.end(), axis.begin(), axis.end());
  (void)out_shape.insert(out_shape.end(), x_shape.begin(), x_shape.end());
  std::sort(axis_value.begin(), axis_value.end());
  auto last = std::unique(axis_value.begin(), axis_value.end());
  axis_value.erase(last, axis_value.end());
  if (keep_dims_value) {
    if (x_shape.size() == 0) {
      return {};
    }
    for (auto i : axis_value) {
      out_shape.at(LongToSize(i)) = 1;
    }
    if (axis_value.empty()) {
      for (size_t i = 0; i < out_shape.size(); i++) {
        out_shape.at(i) = 1;
      }
    }
    return out_shape;
  }
  if (axis.size() == 0 || x_shape.size() == 0) {
    return {};
  }
  std::vector<int64_t>::reverse_iterator it_re;
  for (it_re = axis_value.rbegin(); it_re != axis_value.rend(); ++it_re) {
    (void)out_shape.erase(out_shape.begin() + *it_re);
  }
  return out_shape;
}

ShapeVector ReduceFuncCalShapeAxisDyn(const ShapeVector &x_shape, bool keep_dims) {
  ShapeVector out_shape;
  constexpr int dynamic_rank_value = -2;
  if (!keep_dims) {
    out_shape.push_back(dynamic_rank_value);
  } else {
    (void)out_shape.insert(out_shape.end(), x_shape.size(), -1LL);
  }
  return out_shape;
}

bool CheckAndGetAxisValue(const std::vector<abstract::AbstractBasePtr> &input_args, std::vector<int64_t> *axis_value,
                          int64_t *axis_shape_v, const PrimitivePtr &primitive) {
  MS_EXCEPTION_IF_NULL(axis_value);
  MS_EXCEPTION_IF_NULL(axis_shape_v);
  bool is_dynamic = false;
  const std::string &op_name = primitive->name();
  if (input_args.size() == 1) {
    CheckAndGetAxisValueFromAttr(primitive, axis_value, axis_shape_v);
    return false;
  }
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto input_value = input_args[kInputIndex1]->GetValue();
  if (input_value->isa<KernelTensorValue>()) {
    auto value_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]);
    auto value_array = value_opt.value();
    *axis_value = value_array.ToVector();
    return !value_opt.has_value();
  }
  if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
    is_dynamic = CheckAndGetAxisValueFromScalar(input_value, op_name, axis_value, axis_shape_v);
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractSequence>()) {
    is_dynamic =
      CheckAndGetAxisValueFromSequence(input_args[kInputIndex1], input_value, op_name, axis_value, axis_shape_v);
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    is_dynamic = CheckAndGetAxisValueFromTensor(input_args, input_value, op_name, axis_value, axis_shape_v);
  } else {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', the second input type should be tensor or scalar, but got invalid abstract type:"
                             << input_args[kInputIndex1]->type_name() << ".";
  }
  return is_dynamic;
}

void CheckAndGetAxisValueFromAttr(const PrimitivePtr &primitive, std::vector<int64_t> *axis_value, int64_t *) {
  auto op_name = primitive->name();
  auto axis_ptr = primitive->GetAttr("axis");
  MS_EXCEPTION_IF_NULL(axis_ptr);
  if (axis_ptr->isa<tensor::Tensor>()) {
    *axis_value = CheckAndConvertUtils::CheckTensorIntValue("axis", axis_ptr, op_name);
  } else {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", axis_ptr, op_name);
  }
}

bool CheckAndGetAxisValueFromScalar(const ValuePtr &input_value, const std::string &op_name,
                                    std::vector<int64_t> *axis_value, int64_t *axis_shape_v) {
  *axis_shape_v = 1;
  bool is_dynamic = false;
  if (IsValueKnown(input_value)) {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", input_value, op_name);
  } else {
    is_dynamic = true;
  }
  return is_dynamic;
}

bool CheckAndGetAxisValueFromSequence(const abstract::AbstractBasePtr &abs, const ValuePtr &input_value,
                                      const std::string &op_name, std::vector<int64_t> *axis_value,
                                      int64_t *axis_shape_v) {
  bool is_dynamic = false;
  if (IsValueKnown(input_value)) {
    *axis_value = CheckAndConvertUtils::CheckIntOrTupleInt("axis", input_value, op_name);
    if (axis_value->empty()) {
      *axis_shape_v = 0;
    }
  } else {
    is_dynamic = true;
    auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    *axis_shape_v = seq_abs->dynamic_len() ? -1 : SizeToLong(seq_abs->size());
  }

  return is_dynamic;
}

bool CheckAndGetAxisValueFromTensor(const std::vector<abstract::AbstractBasePtr> &input_args,
                                    const ValuePtr &input_value, const std::string &op_name,
                                    std::vector<int64_t> *axis_value, int64_t *axis_shape_v) {
  bool is_dynamic = false;
  (void)CheckAndConvertUtils::CheckTensorTypeValid("axis", input_args[kInputIndex1]->GetType(), {kInt32, kInt64},
                                                   op_name);
  if (input_value->isa<tensor::Tensor>()) {
    *axis_value = CheckAndConvertUtils::CheckTensorIntValue("axis", input_value, op_name);
    if (axis_value->empty()) {
      *axis_shape_v = 0;
    }
  } else {
    is_dynamic = true;
    auto axis_shape = CheckAndConvertUtils::GetTensorInputShape(op_name, input_args, 1);
    if (axis_shape->shape().size() > 1) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the axis's shape length should be 1, but got '"
                               << axis_shape->shape().size() << "'.";
    } else if (axis_shape->shape().size() == 0) {
      *axis_shape_v = 1;
    } else {
      *axis_shape_v = axis_shape->shape()[0];
    }
  }
  return is_dynamic;
}
}  // namespace mindspore::ops::lite
