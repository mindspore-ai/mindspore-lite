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

#include "src/train/optimizer/fusion/conv_batchnorm_fusion_pass.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include "schema/inner/model_generated.h"
#include "include/api/data_type.h"
#include "include/securec.h"
#include "src/common/utils.h"
#include "tools/common/meta_graph_utils.h"
#include "src/train/optimizer/common/fusion_utils.h"

namespace mindspore {
namespace lite {
namespace {
constexpr std::string_view ConvName = "CONV";
constexpr std::string_view BnName = "BN";

constexpr size_t kFusedBnInputSize = 5;  // input/scale/offset/mean/variance

constexpr int kDataTypeFloat32 = static_cast<int>(mindspore::DataType::kNumberTypeFloat32);

inline bool IsFloat32Tensor(const schema::TensorT &t) { return t.dataType == kDataTypeFloat32; }

// Tensor data is raw bytes; callers validate dataType before using this float view.
// cppcheck-suppress invalidPointerCast
inline const float *TensorAsFloat(const schema::TensorT &t) { return reinterpret_cast<const float *>(t.data.data()); }

inline size_t FloatCountFromDataSize(const schema::TensorT &t) { return t.data.size() / sizeof(float); }

inline float InvStd(float var, float eps) { return 1.0F / std::sqrt(var + eps); }

bool IsTensorConsumed(const schema::MetaGraphT *graph, uint32_t tensor_idx) {
  if (std::find(graph->outputIndex.begin(), graph->outputIndex.end(), tensor_idx) != graph->outputIndex.end()) {
    return true;
  }
  for (const auto &node : graph->nodes) {
    if (node == nullptr) {
      continue;
    }
    if (std::find(node->inputIndex.begin(), node->inputIndex.end(), tensor_idx) != node->inputIndex.end()) {
      return true;
    }
  }
  return false;
}

// A tensor may be rewritten in place only if this node is its sole consumer and it is not a graph
// output. Otherwise other nodes would observe the folded values.
bool IsTensorConsumedOnlyBy(const schema::MetaGraphT *graph, uint32_t tensor_idx, const schema::CNodeT *owner) {
  if (std::find(graph->outputIndex.begin(), graph->outputIndex.end(), tensor_idx) != graph->outputIndex.end() ||
      std::find(graph->inputIndex.begin(), graph->inputIndex.end(), tensor_idx) != graph->inputIndex.end()) {
    return false;
  }
  for (const auto &node : graph->nodes) {
    if (node == nullptr || node.get() == owner) {
      continue;
    }
    if (std::find(node->inputIndex.begin(), node->inputIndex.end(), tensor_idx) != node->inputIndex.end()) {
      return false;
    }
  }
  return true;
}

// Conv2D weight layout: [C_out, C_in/group, kH, kW]; scale broadcast dim is C_out (axis 0).
// Compute the scaled weight into a temporary buffer; the caller writes it back only after every
// check has passed, so a later failure cannot leave the graph half-updated.
STATUS ComputeScaledWeight(const schema::TensorT &weight_tensor, const float *scale_norm, size_t channel,
                           std::vector<float> *scaled_weight) {
  MS_CHECK_TRUE_RET(scaled_weight != nullptr, RET_NULL_PTR);
  if (weight_tensor.dims.empty()) {
    MS_LOG(ERROR) << "Conv weight dims empty.";
    return RET_ERROR;
  }
  if (weight_tensor.dims[0] != static_cast<int32_t>(channel)) {
    MS_LOG(DEBUG) << "Conv weight out-channel " << weight_tensor.dims[0] << " != BN channel " << channel
                  << ", skip folding.";
    return RET_NO_CHANGE;
  }
  const size_t total = FloatCountFromDataSize(weight_tensor);
  if (channel == 0 || total == 0 || total % channel != 0) {
    MS_LOG(ERROR) << "Conv weight element count " << total << " not divisible by channel " << channel;
    return RET_ERROR;
  }
  const size_t per_channel = total / channel;
  const float *weight = TensorAsFloat(weight_tensor);
  scaled_weight->resize(total);
  for (size_t c = 0; c < channel; c++) {
    float s = scale_norm[c];
    for (size_t i = 0; i < per_channel; i++) {
      (*scaled_weight)[c * per_channel + i] = weight[c * per_channel + i] * s;
    }
  }
  return RET_OK;
}

// Fetch a BN parameter tensor by input slot; nullptr when the tensor index is out of range.
schema::TensorT *GetBnParamTensor(const schema::MetaGraphT *graph, const schema::CNodeT *bn_node, int bn_input_idx) {
  uint32_t idx = bn_node->inputIndex.at(bn_input_idx);
  if (idx >= graph->allTensors.size()) {
    return nullptr;
  }
  return graph->allTensors.at(idx).get();
}

struct BnFoldParams {
  size_t channel = 0;
  const float *scale = nullptr;
  const float *mean = nullptr;
  const float *var = nullptr;
  const float *offset = nullptr;
};

// Validate BN's scale/offset/mean/var inputs and expose their float32 data.
// RET_NO_CHANGE means "leave this BN un-fused"; RET_ERROR means a malformed graph.
STATUS GetBnFoldParams(const schema::MetaGraphT *graph, const schema::CNodeT *bn_node, BnFoldParams *params) {
  // BN input order: 0=input, 1=scale, 2=offset, 3=mean, 4=variance
  auto *scale_t = GetBnParamTensor(graph, bn_node, opt::kInputIndexOne);
  auto *offset_t = GetBnParamTensor(graph, bn_node, opt::kInputIndexTwo);
  auto *mean_t = GetBnParamTensor(graph, bn_node, opt::kInputIndexThree);
  auto *var_t = GetBnParamTensor(graph, bn_node, opt::kInputIndexFour);
  if (scale_t == nullptr) {
    MS_LOG(ERROR) << "BN scale tensor is nullptr.";
    return RET_NULL_PTR;
  }
  if (offset_t == nullptr || mean_t == nullptr || var_t == nullptr) {
    MS_LOG(ERROR) << "BN param tensor is nullptr.";
    return RET_NULL_PTR;
  }
  if (!IsFloat32Tensor(*scale_t) || !IsFloat32Tensor(*offset_t) || !IsFloat32Tensor(*mean_t) ||
      !IsFloat32Tensor(*var_t)) {
    MS_LOG(DEBUG) << "BN params not float32, skip.";
    return RET_NO_CHANGE;
  }
  // Guard against untrained BN (params not yet populated). Folding empty tensors would produce
  // invalid weight/bias; leave such BNs intact for the runtime kernel to handle.
  if (scale_t->data.empty() || offset_t->data.empty() || mean_t->data.empty() || var_t->data.empty()) {
    MS_LOG(DEBUG) << "BN param tensors have no data (untrained?), skip folding.";
    return RET_NO_CHANGE;
  }
  size_t channel = FloatCountFromDataSize(*scale_t);
  if (FloatCountFromDataSize(*offset_t) != channel || FloatCountFromDataSize(*mean_t) != channel ||
      FloatCountFromDataSize(*var_t) != channel) {
    MS_LOG(ERROR) << "BN param tensor sizes mismatch.";
    return RET_ERROR;
  }
  if (channel == 0) {
    MS_LOG(ERROR) << "BN channel count is zero.";
    return RET_ERROR;
  }
  params->channel = channel;
  params->scale = TensorAsFloat(*scale_t);
  params->offset = TensorAsFloat(*offset_t);
  params->mean = TensorAsFloat(*mean_t);
  params->var = TensorAsFloat(*var_t);
  return RET_OK;
}

struct ConvBnFoldContext {
  BnFoldParams bn_params;
  schema::TensorT *weight_tensor = nullptr;
  schema::TensorT *bias_tensor = nullptr;
  float eps = 0.0F;
};

// Conv-side preconditions: op type, quantization, fused activation, input count and output wiring.
STATUS CheckConvFoldable(schema::MetaGraphT *graph, const schema::CNodeT &conv_node) {
  if (conv_node.quantType != schema::QuantType_QUANT_NONE) {
    MS_LOG(DEBUG) << "Conv is quantized, skip BN folding.";
    return RET_NO_CHANGE;
  }
  MS_CHECK_TRUE_RET(conv_node.primitive != nullptr, RET_NULL_PTR);
  auto conv_value = conv_node.primitive->value.AsConv2DFusion();
  MS_CHECK_TRUE_RET(conv_value != nullptr, RET_NULL_PTR);
  // Conv already applies an activation before BN; folding BN into the weights would move the BN
  // before that activation and change semantics.
  if (conv_value->activation_type != schema::ActivationType_NO_ACTIVATION) {
    MS_LOG(DEBUG) << "Conv has a fused activation, skip BN folding.";
    return RET_NO_CHANGE;
  }
  const auto &conv_inputs = conv_node.inputIndex;
  if (conv_inputs.size() != opt::kInputSizeTwo && conv_inputs.size() != opt::kInputSizeThree) {
    MS_LOG(DEBUG) << "Conv input count not 2 or 3, skip.";
    return RET_NO_CHANGE;
  }
  // ApplyFold hands BN's output to conv, so conv's own output must not be a model output:
  // re-wiring would leave that graph output without a producer.
  if (conv_node.outputIndex.size() != opt::kOutputSizeOne ||
      opt::IsTensorGraphOutput(graph, conv_node.outputIndex.at(opt::kOutputIndexZero))) {
    MS_LOG(DEBUG) << "Conv output is shared with the graph output, skip BN folding.";
    return RET_NO_CHANGE;
  }
  return RET_OK;
}

// BN-side preconditions: input count, params, and unconsumed auxiliary outputs.
STATUS CheckBnFoldable(const schema::MetaGraphT *graph, const schema::CNodeT &bn_node, ConvBnFoldContext *ctx) {
  if (bn_node.inputIndex.size() != kFusedBnInputSize) {
    MS_LOG(DEBUG) << "BN does not have 5 inputs (input/scale/offset/mean/var), skip.";
    return RET_NO_CHANGE;
  }
  MS_CHECK_TRUE_RET(bn_node.primitive != nullptr, RET_NULL_PTR);
  auto bn_value = bn_node.primitive->value.AsFusedBatchNorm();
  MS_CHECK_TRUE_RET(bn_value != nullptr, RET_NULL_PTR);
  // Note: we intentionally do NOT skip on bn_value->mode != 0. During on-device training the BN
  // carries mode=1 (training); when exporting an inference model the BN's INPUT mean/var tensors
  // already hold the final momentum-updated running stats (FusedBatchNormFp32MeanVar writes them
  // in place), so folding is mathematically valid regardless of mode. Gating on mode would skip
  // every BN in a trained model and defeat the pass. We instead validate that the scale/offset/mean/var
  // tensors carry data.
  ctx->eps = bn_value->epsilon;
  auto params_ret = GetBnFoldParams(graph, &bn_node, &ctx->bn_params);
  if (params_ret != RET_OK) {
    return params_ret;
  }
  // A non-positive or NaN variance + epsilon would write inf/NaN into the conv weight/bias via
  // 1/sqrt(var + eps); leave the BN un-fused instead of corrupting the graph.
  for (size_t c = 0; c < ctx->bn_params.channel; c++) {
    const float divisor = ctx->bn_params.var[c] + ctx->eps;
    if (!std::isfinite(divisor) || divisor <= 0.0F) {
      MS_LOG(DEBUG) << "BN variance + epsilon is not positive (channel " << c << "), skip folding.";
      return RET_NO_CHANGE;
    }
  }
  MS_CHECK_TRUE_RET(!bn_node.outputIndex.empty(), RET_ERROR);
  // BN's auxiliary outputs (out[1..]) carry running stats. Folding removes their producer, so
  // refuse to fold if any of them still has a consumer (including graph outputs).
  for (size_t i = 1; i < bn_node.outputIndex.size(); ++i) {
    if (IsTensorConsumed(graph, bn_node.outputIndex.at(i))) {
      MS_LOG(DEBUG) << "BN auxiliary output has consumers, skip folding.";
      return RET_NO_CHANGE;
    }
  }
  return RET_OK;
}

// Validate conv weight/bias: float32 ValueNodes that this conv alone consumes.
STATUS CheckConvTensorsFoldable(const schema::MetaGraphT *graph, const schema::CNodeT &conv_node, size_t channel,
                                ConvBnFoldContext *ctx) {
  const auto &conv_inputs = conv_node.inputIndex;
  uint32_t weight_idx = conv_inputs.at(opt::kInputIndexOne);
  MS_CHECK_TRUE_RET(weight_idx < graph->allTensors.size(), RET_ERROR);
  auto &weight_tensor = graph->allTensors.at(weight_idx);
  MS_CHECK_TRUE_RET(weight_tensor != nullptr, RET_NULL_PTR);
  if (!IsFloat32Tensor(*weight_tensor) || weight_tensor->nodeType != NodeType_ValueNode ||
      weight_tensor->data.empty()) {
    MS_LOG(DEBUG) << "Conv weight is not a non-empty float32 ValueNode, skip.";
    return RET_NO_CHANGE;
  }
  if (!IsTensorConsumedOnlyBy(graph, weight_idx, &conv_node)) {
    MS_LOG(DEBUG) << "Conv weight is shared or a graph output, skip.";
    return RET_NO_CHANGE;
  }
  ctx->weight_tensor = weight_tensor.get();

  if (conv_inputs.size() != opt::kInputSizeThree) {
    return RET_OK;
  }
  uint32_t bias_idx = conv_inputs.at(opt::kInputIndexTwo);
  MS_CHECK_TRUE_RET(bias_idx < graph->allTensors.size(), RET_ERROR);
  auto *bias_tensor = graph->allTensors.at(bias_idx).get();
  MS_CHECK_TRUE_RET(bias_tensor != nullptr, RET_NULL_PTR);
  if (!IsFloat32Tensor(*bias_tensor) || bias_tensor->nodeType != NodeType_ValueNode || bias_tensor->data.empty() ||
      FloatCountFromDataSize(*bias_tensor) != channel) {
    MS_LOG(DEBUG) << "Conv bias is not a non-empty float32 ValueNode of size " << channel << ", skip.";
    return RET_NO_CHANGE;
  }
  if (!IsTensorConsumedOnlyBy(graph, bias_idx, &conv_node)) {
    MS_LOG(DEBUG) << "Conv bias is shared or a graph output, skip.";
    return RET_NO_CHANGE;
  }
  ctx->bias_tensor = bias_tensor;
  return RET_OK;
}

// Compute the folded parameters into temporary buffers; the graph is not touched.
STATUS ComputeFoldedParams(const ConvBnFoldContext &ctx, std::vector<float> *new_weight, std::vector<float> *new_bias) {
  const BnFoldParams &bn = ctx.bn_params;
  // scale_norm[c] = scale[c] / sqrt(var[c] + eps)
  std::vector<float> scale_norm(bn.channel);
  for (size_t c = 0; c < bn.channel; c++) {
    scale_norm[c] = bn.scale[c] * InvStd(bn.var[c], ctx.eps);
  }
  auto wret = ComputeScaledWeight(*ctx.weight_tensor, scale_norm.data(), bn.channel, new_weight);
  if (wret != RET_OK) {
    return wret;
  }
  // b'[c] = conv_bias[c] * scale_norm[c] + (offset[c] - mean[c] * scale_norm[c])
  new_bias->resize(bn.channel);
  for (size_t c = 0; c < bn.channel; c++) {
    (*new_bias)[c] = bn.offset[c] - bn.mean[c] * scale_norm[c];
  }
  if (ctx.bias_tensor != nullptr) {
    const float *bias = TensorAsFloat(*ctx.bias_tensor);
    for (size_t c = 0; c < bn.channel; c++) {
      (*new_bias)[c] += bias[c] * scale_norm[c];
    }
  }
  return RET_OK;
}

// Commit the folded parameters and re-wire Conv/BN. Runs only after all checks pass.
STATUS ApplyFold(schema::MetaGraphT *graph, schema::CNodeT *conv_node, schema::CNodeT *bn_node,
                 const ConvBnFoldContext &ctx, const std::vector<float> &new_weight,
                 const std::vector<float> &new_bias) {
  const size_t channel = ctx.bn_params.channel;
  auto *weight = const_cast<float *>(TensorAsFloat(*ctx.weight_tensor));
  auto res = memcpy_s(weight, new_weight.size() * sizeof(float), new_weight.data(), new_weight.size() * sizeof(float));
  MS_CHECK_TRUE_MSG(res == EOK, RET_ERROR, "memcpy_s failed");
  if (ctx.bias_tensor != nullptr) {
    auto *bias = const_cast<float *>(TensorAsFloat(*ctx.bias_tensor));
    res = memcpy_s(bias, channel * sizeof(float), new_bias.data(), channel * sizeof(float));
    MS_CHECK_TRUE_MSG(res == EOK, RET_ERROR, "memcpy_s failed");
  } else {
    auto new_bias_tensor = std::make_unique<schema::TensorT>();
    MS_CHECK_TRUE_RET(new_bias_tensor != nullptr, RET_NULL_PTR);
    new_bias_tensor->dataType = kDataTypeFloat32;
    new_bias_tensor->dims = {static_cast<int32_t>(channel)};
    new_bias_tensor->data.resize(channel * sizeof(float));
    res = memcpy_s(new_bias_tensor->data.data(), channel * sizeof(float), new_bias.data(), channel * sizeof(float));
    MS_CHECK_TRUE_MSG(res == EOK, RET_ERROR, "memcpy_s failed");
    new_bias_tensor->nodeType = static_cast<int>(NodeType_ValueNode);
    conv_node->inputIndex.push_back(static_cast<uint32_t>(graph->allTensors.size()));
    graph->allTensors.push_back(std::move(new_bias_tensor));
  }

  // Re-wire: Conv takes over BN's first output (the normalized output, out_[0]); BN's auxiliary
  // outputs (scale/offset/mean/var at out_[1..]) carry running stats for the backward pass and have
  // no consumers in the inference graph. BN becomes isolated (cleaned by IsolatedNodeRemovePass).
  uint32_t bn_norm_output = bn_node->outputIndex.at(opt::kInputIndexZero);
  conv_node->outputIndex = {bn_norm_output};
  bn_node->inputIndex.clear();
  bn_node->outputIndex.clear();
  return RET_OK;
}
}  // namespace
STATUS ConvBatchNormFusionPass::DefinePattern() {
  auto conv_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(conv_op != nullptr, RET_NULL_PTR);
  conv_op->id = ConvName;
  conv_op->types = {schema::PrimitiveType_Conv2DFusion};
  auto bn_op = std::make_shared<PatternOp>();
  MS_CHECK_TRUE_RET(bn_op != nullptr, RET_NULL_PTR);
  bn_op->id = BnName;
  bn_op->types = {schema::PrimitiveType_FusedBatchNorm};
  bn_op->left = conv_op;
  auto fusion_pattern = std::make_unique<FusionPattern>("ConvBatchNormFusion");
  MS_CHECK_TRUE_MSG(fusion_pattern != nullptr, RET_NULL_PTR, "new fusion_pattern failed");
  fusion_pattern->AddPatternOp(conv_op);
  fusion_pattern->AddPatternOp(bn_op);
  fusion_pattern->Finish();
  this->patterns.emplace_back(fusion_pattern.release());
  return RET_OK;
}

STATUS ConvBatchNormFusionPass::DoFusion(MetaGraphT *graph, const std::string &pattern_name,
                                         const std::unordered_map<std::string, std::shared_ptr<Path>> &matched_path) {
  MS_CHECK_TRUE_RET(graph != nullptr, RET_NULL_PTR);
  if (matched_path.size() != opt::kMatchPathLenTwo) {
    MS_LOG(ERROR) << "Conv-BN-Fusion should have two NodeIndex in matchedPair";
    return RET_PARAM_INVALID;
  }

  size_t conv_index = 0;
  auto ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(ConvName), &conv_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get conv_index");
  auto &conv_node = graph->nodes.at(conv_index);
  MS_CHECK_TRUE_MSG(conv_node != nullptr, RET_NULL_PTR, "conv_node is nullptr");
  size_t bn_index = 0;
  ret = opt::GetMatchNodeIndex(graph, matched_path, std::string(BnName), &bn_index);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "cannot get bn_index");
  auto &bn_node = graph->nodes.at(bn_index);
  MS_CHECK_TRUE_MSG(bn_node != nullptr, RET_NULL_PTR, "bn_node is nullptr");

  ret = CheckConvFoldable(graph, *conv_node);
  if (ret != RET_OK) {
    return ret;
  }
  ConvBnFoldContext ctx;
  ret = CheckBnFoldable(graph, *bn_node, &ctx);
  if (ret != RET_OK) {
    return ret;
  }
  // --- Validation only from here to the write-back: any failure must leave the graph untouched. ---
  ret = CheckConvTensorsFoldable(graph, *conv_node, ctx.bn_params.channel, &ctx);
  if (ret != RET_OK) {
    return ret;
  }

  // --- Compute the folded parameters into temporary buffers, then write back once. ---
  std::vector<float> new_weight;
  std::vector<float> new_bias;
  ret = ComputeFoldedParams(ctx, &new_weight, &new_bias);
  if (ret != RET_OK) {
    return ret;
  }
  return ApplyFold(graph, conv_node.get(), bn_node.get(), ctx, new_weight, new_bias);
}

ConvBatchNormFusionPass::~ConvBatchNormFusionPass() = default;
}  // namespace lite
}  // namespace mindspore
