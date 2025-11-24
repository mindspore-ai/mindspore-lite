/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/pnna/pnna_utils.h"
#include <vector>
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "nnacl_c/op_base.h"

namespace mindspore {
namespace lite {
pnna::DataType ConvertToPnnaDataType(DataType type_id) {
  pnna::DataType data_type = pnna::DataType::UNKNOWN;
  switch (type_id) {
    case DataType::kNumberTypeBool:
      data_type = pnna::DataType::BOOL8;
      break;
    case DataType::kNumberTypeInt8:
      data_type = pnna::DataType::INT8;
      break;
    case DataType::kNumberTypeInt16:
      data_type = pnna::DataType::INT16;
      break;
    case DataType::kNumberTypeInt32:
      data_type = pnna::DataType::INT32;
      break;
    case DataType::kNumberTypeUInt8:
      data_type = pnna::DataType::UINT8;
      break;
    case DataType::kNumberTypeUInt16:
      data_type = pnna::DataType::UINT16;
      break;
    case DataType::kNumberTypeUInt32:
      data_type = pnna::DataType::UINT32;
      break;
    case DataType::kNumberTypeFloat16:
      data_type = pnna::DataType::FLOAT16;
      break;
    case DataType::kNumberTypeFloat32:
      data_type = pnna::DataType::FLOAT32;
      break;
    default:
      MS_LOG(ERROR) << "Failed to convert the MindSpore Lite operand DataType code(" << static_cast<int>(type_id)
                    << ") to pnna::DataType !";
      break;
  }
  return data_type;
}

pnna::DataLayout ConvertToPnnaDataLayout(Format input_layout) {
  pnna::DataLayout output_layout = pnna::DataLayout::ANY;
  switch (input_layout) {
    case Format::NCHW:
      output_layout = pnna::DataLayout::WHCN;
      break;
    case Format::NHWC:
      output_layout = pnna::DataLayout::CWHN;
      break;
    default:
      MS_LOG(ERROR) << "Failed to convert the MindSpore Lite operand layout code(" << static_cast<int>(input_layout)
                    << ") to pnna::DataLayout !";
      break;
  }
  return output_layout;
}

pnna::ShapeType ConvertToPnnaShapeType(const std::vector<int64_t> &input_shape) {
  pnna::ShapeType output_shape;
  for (int i = input_shape.size() - 1; i >= 0; i--) {
    output_shape.push_back(input_shape[i]);
  }
  return output_shape;
}

std::shared_ptr<pnna::Tensor> CreatePnnaTensor(pnna::Graph *graph, pnna::ShapeType shape, pnna::DataType data_type,
                                               const void *buffer, pnna::DataLayout data_layout,
                                               pnna::TensorAttribute tensor_attr, const float *quant_scale,
                                               uint32_t quant_scale_count, const int32_t *quant_zero_point,
                                               uint32_t quant_channel_dim) {
  MS_ASSERT(graph != nullptr);
  pnna::TensorSpec tensor_spec;
  tensor_spec.SetDataType(data_type);
  tensor_spec.SetShape(shape);
  if (quant_scale) {
    MS_CHECK_GT(quant_scale_count, 0, nullptr);
    MS_CHECK_TRUE_MSG(data_type == pnna::DataType::INT8 || data_type == pnna::DataType::INT32, nullptr,
                      "Only INT8/INT32 supported for quantization.");
    pnna::Quantization tensor_quantization;
    if (quant_scale_count > 1) {
      MS_CHECK_GE(quant_channel_dim, 0, nullptr);
      tensor_quantization.SetType(pnna::QuantType::SYMMETRIC_PER_CHANNEL);
      tensor_quantization.SetChannelDim(quant_channel_dim);
    } else {
      tensor_quantization.SetType(pnna::QuantType::ASYMMETRIC);
    }
    std::vector<float> scales(quant_scale, quant_scale + quant_scale_count);
    std::vector<int32_t> zero_points;
    if (quant_zero_point) {
      zero_points.assign(quant_zero_point, quant_zero_point + quant_scale_count);
    } else {
      zero_points.assign(quant_scale_count, 0);
    }
    tensor_quantization.SetScales(std::move(scales));
    tensor_quantization.SetZeroPoints(std::move(zero_points));
    tensor_spec.SetQuantization(tensor_quantization);
  }
  tensor_spec.SetAttribute(tensor_attr);
  MS_ASSERT((!buffer && tensor_attr != pnna::TensorAttribute::CONSTANT) ||
            (buffer && tensor_attr == pnna::TensorAttribute::CONSTANT));
  auto tensor = buffer ? graph->CreateTensor(tensor_spec, buffer) : graph->CreateTensor(tensor_spec);
  MS_ASSERT(tensor);
  return tensor;
}

std::shared_ptr<pnna::Tensor> CreatePnnaTensor(pnna::Graph *graph, MSTensor *tensor,
                                               pnna::TensorAttribute tensor_attr) {
  auto shape = ConvertToPnnaShapeType(tensor->Shape());
  if (tensor->Shape().size() == 0) {
    shape.push_back(1);
  }
  auto data_type = ConvertToPnnaDataType(tensor->DataType());
  auto data_layout = ConvertToPnnaDataLayout(tensor->format());
  auto quant_params = tensor->QuantParams();
  std::vector<float> quant_scale;
  std::vector<int32_t> quant_zero_point;
  uint32_t quant_scale_count = 0;
  int32_t quant_channel_dim = -1;
  if (!quant_params.empty() && (data_type == pnna::DataType::INT8 || data_type == pnna::DataType::INT32)) {
    quant_scale_count = quant_params.size();
    for (auto &quant_param : quant_params) {
      quant_scale.emplace_back(quant_param.scale);
      quant_zero_point.emplace_back(quant_param.zero_point);
      // Only support for Conv2d
      if (shape.size() == Num4) {
        quant_channel_dim = 3;
      } else {
        quant_channel_dim = 0;
      }
    }
  }
  return CreatePnnaTensor(graph, shape, data_type, tensor->Data().get(), data_layout, tensor_attr, quant_scale.data(),
                          quant_scale_count, quant_zero_point.data(), quant_channel_dim);
}

std::vector<uint32_t> ConvertToPnnaPerm(const int32_t *data, size_t count) {
  std::vector<uint32_t> output;
  for (int i = count - 1; i >= 0; i--) {
    output.push_back(count - 1 - data[i]);
  }
  return output;
}

int32_t ConvertToPnnaAxis(int32_t axis, size_t count) { return count - 1 - (axis < 0 ? count + axis : axis); }
}  // namespace lite
}  // namespace mindspore
