/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

// #define USE_DEPRECATED_API
#include "tools/optimizer/graph/add_variable_node_pass.h"
#include <memory>
#include <vector>
#include <set>
#include <algorithm>
#include <map>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "src/common/log_util.h"
#include "src/common/common.h"
#include "tools/common/tensor_util.h"
#include "tools/common/func_graph_utils.h"
#include "tools/optimizer/fusion/matmul_allreduce_fusion.h"
#include "op_def/auto_generate/gen_lite_ops.h"
#include "tools/common/parse_config_utils.h"
#include "op_def/conv_pool_ops.h"
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kInputSize3 = 3;
constexpr size_t kInputIndex2 = 2;
constexpr size_t kConstantMatmulWeightShapeSize = 2;
constexpr size_t kConstantConvWeightShapeSize = 4;
constexpr size_t kWeightInitLen = 1;
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr float kInitZero = 0.0;
constexpr float kInitOne = 1.0;
constexpr size_t kInitBatchSize = 1;
constexpr size_t kMaxConfigLen = 1e6;
constexpr uint16_t kFloatOne = 15360;
}  // namespace

template <typename T>
ParameterPtr InsertVariableNodePass::BuildFloat16ZeroVecNDParameterNode(const FuncGraphPtr &anf_graph,
                                                                        ShapeVector weight_shape,
                                                                        const std::string &node_name, T value,
                                                                        TypeId dtype) {
  if (dtype != kNumberTypeFloat16 && dtype != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Only Support kNumberTypeFloat16 and kNumberTypeFloat32! Current dtype:" << dtype << "!";
    return nullptr;
  }
  if (std::find_if(weight_shape.begin(), weight_shape.end(), [](int64_t num) { return num <= 0; }) !=
      weight_shape.end()) {
    MS_LOG(ERROR) << "Weight shape has zero or negative value!"
                  << "node name:" << node_name << ", weight shape:" << weight_shape << "!";
    return nullptr;
  }
  MS_CHECK_TRUE_RET(anf_graph != nullptr, nullptr);
  auto param_node = anf_graph->add_parameter();
  MS_CHECK_TRUE_RET(param_node != nullptr, nullptr);
  param_node->set_name(node_name);
  int weight_length = kWeightInitLen;
  for (auto dim : weight_shape) {
    MS_CHECK_INT_MUL_NOT_OVERFLOW(weight_length, dim, nullptr);
    weight_length *= dim;
  }

  std::vector<T> data_1d(weight_length, value);
  auto size = data_1d.size() * sizeof(T);
  auto tensor_info = lite::CreateTensorInfo(data_1d.data(), size, weight_shape, dtype);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed! weight_shape:" << weight_shape << "!";
    return nullptr;
  }
  auto status = lite::InitParameterFromTensorInfo(param_node, tensor_info);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "init parameter from tensor info failed!";
    return nullptr;
  }
  return param_node;
}

lite::STATUS FetchWeightShape(AnfNodePtr weight, ShapeVector *weight_shape, const CNodePtr &cnode, bool is_matmul) {
  if (!utils::isa<ParameterPtr>(weight)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  auto weight_param = weight->cast<ParameterPtr>();
  MS_CHECK_TRUE_RET(weight_param != nullptr, false);
  auto value = weight_param->default_param();
  MS_CHECK_TRUE_RET(value != nullptr, false);
  auto weight_tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
  MS_CHECK_TRUE_RET(weight_tensor != nullptr, false);
  *weight_shape = weight_tensor->shape();
  if ((is_matmul && weight_shape->size() != kConstantMatmulWeightShapeSize) ||
      (!is_matmul && weight_shape->size() != kConstantConvWeightShapeSize)) {
    MS_LOG(ERROR) << "now only support 2 dims matmul constant weight, or 4 dims conv constant weight!"
                  << "weight shape size:" << weight_shape->size() << ", node name:" << cnode->fullname_with_scope()
                  << "!";
    return RET_ERROR;
  }
  return RET_OK;
}

lite::STATUS CreateBMMNode(AnfNodePtrList &&bmm_inputs, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const std::string &suffix, AnfNodePtr *bmm_param) {
  auto bmm = std::make_shared<ops::BatchMatMul>();
  auto bmm_prim_c = bmm->GetPrim();
  auto bmm_cnode = func_graph->NewCNode(bmm_prim_c, bmm_inputs);
  if (bmm_cnode == nullptr) {
    MS_LOG(ERROR) << "new bmm node failed!";
    return RET_ERROR;
  }
  bmm_cnode->set_fullname_with_scope(node->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(bmm_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_OK;
  }
  *bmm_param = bmm_cnode->cast<AnfNodePtr>();
  if (node->abstract() != nullptr) {
    bmm_cnode->set_abstract(node->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateMulNode(AnfNodePtrList &&mul_inputs, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const std::string &suffix, AnfNodePtr *mul_param) {
  auto mul = std::make_shared<ops::Mul>();
  auto mul_prim_c = mul->GetPrim();
  auto mul_cnode = func_graph->NewCNode(mul_prim_c, mul_inputs);
  if (mul_cnode == nullptr) {
    MS_LOG(ERROR) << "new alpha mul node failed!";
    return false;
  }
  mul_cnode->set_fullname_with_scope(node->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(mul_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  *mul_param = mul_cnode->cast<AnfNodePtr>();
  if (node->abstract() != nullptr) {
    mul_cnode->set_abstract(node->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateReduceSumNode(AnfNodePtrList &&reduce_sum_inputs, const FuncGraphPtr &func_graph,
                                 const AnfNodePtr &node, const std::string &suffix, AnfNodePtr *reduce_sum_param) {
  auto reduce_sum = std::make_shared<ops::ReduceSum>();
  auto reduce_sum_prim_c = reduce_sum->GetPrim();
  auto reduce_sum_cnode = func_graph->NewCNode(reduce_sum_prim_c, reduce_sum_inputs);
  if (reduce_sum_cnode == nullptr) {
    MS_LOG(ERROR) << "new reduce sum node failed!";
    return RET_ERROR;
  }
  reduce_sum_cnode->set_fullname_with_scope(node->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(reduce_sum_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  *reduce_sum_param = reduce_sum_cnode->cast<AnfNodePtr>();
  if (node->abstract() != nullptr) {
    reduce_sum_cnode->set_abstract(node->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateTransposeNode(AnfNodePtrList &&transpose_inputs, const FuncGraphPtr &func_graph,
                                 const AnfNodePtr &node, const std::string &suffix, AnfNodePtr *transpose_param) {
  auto transpose = std::make_shared<ops::Transpose>();
  auto transpose_prim_c = transpose->GetPrim();
  auto transpose_cnode = func_graph->NewCNode(transpose_prim_c, transpose_inputs);
  if (transpose_cnode == nullptr) {
    MS_LOG(ERROR) << "new reduce sum node failed!";
    return false;
  }
  transpose_cnode->set_fullname_with_scope(node->fullname_with_scope() + suffix);
  if (!utils::isa<AnfNodePtr>(transpose_cnode)) {
    MS_LOG(ERROR) << "matmul weight is not constant, can not update weight!";
    return RET_ERROR;
  }
  *transpose_param = transpose_cnode->cast<AnfNodePtr>();
  if (node->abstract() != nullptr) {
    transpose_cnode->set_abstract(node->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS CreateAddNode(AnfNodePtrList &&add_inputs, const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                           const std::string &suffix, CNodePtr *add_cnode) {
  auto add = std::make_shared<ops::Add>();
  auto add_prim_c = add->GetPrim();
  (*add_cnode) = func_graph->NewCNode(add_prim_c, add_inputs);
  if (*add_cnode == nullptr) {
    MS_LOG(ERROR) << "new add node failed!";
    return RET_ERROR;
  }
  (*add_cnode)->set_fullname_with_scope(node->fullname_with_scope() + suffix);
  if (node->abstract() != nullptr) {
    (*add_cnode)->set_abstract(node->abstract()->Clone());
  }
  return RET_OK;
}

lite::STATUS FetchNodeNameMap(const CNodePtr &cnode, std::unordered_map<std::string, std::string> *node_name_map,
                              const bool &has_alpha) {
  auto node_name = cnode->fullname_with_scope();
  size_t last_slash_pos = node_name.find_last_of('/');
  MS_CHECK_TRUE_RET(last_slash_pos != std::string::npos, RET_ERROR);
  auto search_key = node_name.substr(0, last_slash_pos);
  (*node_name_map)[search_key + "variable_up"] = cnode->fullname_with_scope() + "_lora_up_const";
  (*node_name_map)[search_key + "variable_down"] = cnode->fullname_with_scope() + "_lora_down_const";
  if (has_alpha) {
    (*node_name_map)[search_key + "variable_alpha"] = cnode->fullname_with_scope() + "_lora_alpha_const";
  }
  return RET_OK;
}

lite::STATUS InsertVariableNodePass::InsertVariableNodeForMatmul(
  const AnfNodePtr &node, const CNodePtr &cnode, const FuncGraphPtr &func_graph, const std::vector<int> &up_shape,
  std::unordered_map<std::string, std::string> *node_name_map, bool has_alpha, int max_weight_batch) {
  MS_CHECK_TRUE_RET(cnode->inputs().size() >= kInputSize3 && up_shape.size() == kConstantMatmulWeightShapeSize,
                    RET_ERROR);
  auto weight = cnode->input(kInputIndex2);
  MS_CHECK_TRUE_RET(weight != nullptr, RET_ERROR);
  ShapeVector weight_shape;
  auto ret = FetchWeightShape(weight, &weight_shape, cnode, true);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch wieght shape failed! ret:" << ret << "!";
    return ret;
  }
  auto low_rank = MIN(up_shape[kIndex0], up_shape[kIndex1]);
  auto value_node = cnode->input(kIndex0)->cast<ValueNodePtr>();
  MS_CHECK_TRUE_RET(value_node != nullptr, RET_ERROR);
  auto src_prim = GetValueNode<PrimitivePtr>(value_node);
  MS_CHECK_TRUE_RET(src_prim != nullptr, RET_ERROR);
  int64_t up_high_rank = weight_shape[kIndex1];
  int64_t down_high_rank = weight_shape[kIndex0];
  bool is_gemm = false;
  if (src_prim->GetAttr(mindspore::ops::kTransposeA) != nullptr ||
      src_prim->GetAttr(mindspore::ops::kTransposeB) != nullptr) {
    up_high_rank = weight_shape[kIndex0];
    down_high_rank = weight_shape[kIndex1];
    is_gemm = true;
  }
  ShapeVector lora_up_shape = {max_weight_batch, up_high_rank, low_rank};
  ShapeVector lora_down_shape = {max_weight_batch, low_rank, down_high_rank};
  ShapeVector lora_add_shape = {max_weight_batch, kInitBatchSize, kInitBatchSize};
  ShapeVector lora_alpha_shape = {max_weight_batch, kInitBatchSize, kInitBatchSize};
  AnfNodePtr lora_up_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, lora_up_shape, cnode->fullname_with_scope() + "_lora_up", 0.0, kNumberTypeFloat16);
  AnfNodePtr lora_down_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, lora_down_shape, cnode->fullname_with_scope() + "_lora_down", 0.0, kNumberTypeFloat16);
  AnfNodePtr add_weights_param_node = BuildFloat16ZeroVecNDParameterNode<float>(
    func_graph, lora_add_shape, cnode->fullname_with_scope() + "_lora_add_weights", kInitOne, kNumberTypeFloat32);
  AnfNodePtr alpha_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, lora_alpha_shape, cnode->fullname_with_scope() + "_lora_alpha", kFloatOne, kNumberTypeFloat16);
  AnfNodePtr axes_param_node =
    opt::BuildIntValueParameterNode(func_graph, kInitZero, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  MS_CHECK_TRUE_RET(lora_up_param_node != nullptr && lora_down_param_node != nullptr &&
                      add_weights_param_node != nullptr && alpha_param_node != nullptr && axes_param_node != nullptr,
                    RET_ERROR);
  if (FetchNodeNameMap(cnode, node_name_map, has_alpha) != RET_OK) {
    MS_LOG(ERROR) << "FetchNodeNameMap failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto bmm_inputs = {lora_up_param_node, lora_down_param_node};
  AnfNodePtr bmm_param = nullptr;
  if (CreateBMMNode(bmm_inputs, func_graph, node, "_lora_bmm", &bmm_param) != RET_OK) {
    MS_LOG(ERROR) << "Create BMM node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto mul_alpha_inputs = {bmm_param, alpha_param_node};
  AnfNodePtr alpha_mul_param = nullptr;
  if (CreateMulNode(mul_alpha_inputs, func_graph, node, "_lora_alpha_mul", &alpha_mul_param) != RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto mul_add_weights_inputs = {alpha_mul_param, add_weights_param_node};
  AnfNodePtr mul_add_weights_param = nullptr;
  if (CreateMulNode(mul_add_weights_inputs, func_graph, node, "_lora_add_weights_mul", &mul_add_weights_param) !=
      RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto reduce_sum_inputs = {mul_add_weights_param, axes_param_node};
  AnfNodePtr reduce_sum_param = nullptr;
  if (CreateReduceSumNode(reduce_sum_inputs, func_graph, node, "_lora_reduce_sum", &reduce_sum_param) != RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  std::vector<int> perm = {kIndex1, kIndex0};
  if (is_gemm) {
    perm = {kIndex0, kIndex1};
  }
  AnfNodePtr perm_param_node =
    opt::BuildIntVecParameterNode(func_graph, perm, cnode->fullname_with_scope() + "_trans_perm");
  auto transpose_inputs = {reduce_sum_param, perm_param_node};
  AnfNodePtr transpose_param = nullptr;
  if (CreateTransposeNode(transpose_inputs, func_graph, node, "_lora_transpose", &transpose_param) != RET_OK) {
    MS_LOG(ERROR) << "Create transpose node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto add_inputs = {transpose_param, weight};
  CNodePtr add_cnode = nullptr;
  if (CreateAddNode(add_inputs, func_graph, node, "_lora_add", &add_cnode) != RET_OK) {
    MS_LOG(ERROR) << "Create Add node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto manager = Manage(func_graph);
  (void)manager->Replace(weight, add_cnode);
  return RET_OK;
}

lite::STATUS InsertVariableNodePass::InsertVariableNodeForConv(
  const AnfNodePtr &node, const CNodePtr &cnode, const FuncGraphPtr &func_graph, const std::vector<int> &up_shape,
  std::unordered_map<std::string, std::string> *node_name_map, bool has_alpha, int max_weight_batch) {
  MS_CHECK_TRUE_RET(cnode->inputs().size() >= kInputSize3, RET_ERROR);
  MS_CHECK_TRUE_RET(up_shape.size() == kConstantConvWeightShapeSize, RET_ERROR);
  auto weight = cnode->input(kInputIndex2);
  MS_CHECK_TRUE_RET(weight != nullptr, RET_ERROR);
  ShapeVector weight_shape;
  auto ret = FetchWeightShape(weight, &weight_shape, cnode, false);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch wieght shape failed! ret:" << ret << "!";
    return ret;
  }
  int kernel_size_down = weight_shape[kIndex2] / up_shape[kIndex2];
  ShapeVector lora_up_shape = {max_weight_batch, up_shape[kIndex0], up_shape[kIndex1], up_shape[kIndex2],
                               up_shape[kIndex3]};
  ShapeVector lora_down_shape = {max_weight_batch, up_shape[kIndex1], weight_shape[kIndex1], kernel_size_down,
                                 kernel_size_down};
  ShapeVector add_weights_shape = {max_weight_batch, kIndex1, kIndex1, kIndex1, kIndex1};
  ShapeVector alpha_weights_shape = {max_weight_batch, kIndex1, kIndex1, kIndex1, kIndex1};
  AnfNodePtr lora_up_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, lora_up_shape, cnode->fullname_with_scope() + "_lora_up", 0.0, kNumberTypeFloat16);
  AnfNodePtr lora_down_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, lora_down_shape, cnode->fullname_with_scope() + "_lora_down", 0.0, kNumberTypeFloat16);
  AnfNodePtr add_weights_param_node = BuildFloat16ZeroVecNDParameterNode<float>(
    func_graph, add_weights_shape, cnode->fullname_with_scope() + "_lora_add", kInitOne, kNumberTypeFloat32);
  AnfNodePtr alpha_param_node = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, alpha_weights_shape, cnode->fullname_with_scope() + "_lora_alpha", kFloatOne, kNumberTypeFloat16);
  ret = FetchNodeNameMap(cnode, node_name_map, has_alpha);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FetchNodeNameMap failed! ret:" << ret << "!";
    return ret;
  }
  AnfNodePtr axes_param_node =
    opt::BuildIntValueParameterNode(func_graph, kInitZero, cnode->fullname_with_scope() + "_reduce_sum_axes", true);
  std::vector<int> perm = {kIndex3, kIndex2, kIndex1, kIndex0};
  AnfNodePtr perm_param_node =
    opt::BuildIntVecParameterNode(func_graph, perm, cnode->fullname_with_scope() + "_trans_perm");
  std::vector<int> perm_reverse = {kIndex0, kIndex4, kIndex3, kIndex2, kIndex1};
  AnfNodePtr perm_reverse_param_node =
    opt::BuildIntVecParameterNode(func_graph, perm_reverse, cnode->fullname_with_scope() + "_trans_reverse_perm");
  MS_CHECK_TRUE_RET(lora_up_param_node != nullptr && lora_down_param_node != nullptr &&
                      add_weights_param_node != nullptr && alpha_param_node != nullptr && perm_param_node != nullptr &&
                      perm_reverse_param_node != nullptr,
                    RET_ERROR);
  // transpose up
  auto transpose_up_inputs = {lora_up_param_node, perm_reverse_param_node};
  AnfNodePtr transpose_up_param = nullptr;
  if (CreateTransposeNode(transpose_up_inputs, func_graph, node, "_lora_up_transpose", &transpose_up_param) != RET_OK) {
    MS_LOG(ERROR) << "Create transpose node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  // transpose down
  auto transpose_down_inputs = {lora_down_param_node, perm_reverse_param_node};
  AnfNodePtr transpose_down_param = nullptr;
  if (CreateTransposeNode(transpose_down_inputs, func_graph, node, "_lora_down_transpose", &transpose_down_param) !=
      RET_OK) {
    MS_LOG(ERROR) << "Create transpose node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  // bmm
  auto bmm_inputs = {transpose_down_param, transpose_up_param};
  AnfNodePtr bmm_param = nullptr;
  if (CreateBMMNode(bmm_inputs, func_graph, node, "_lora_bmm", &bmm_param) != RET_OK) {
    MS_LOG(ERROR) << "Create bmm node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto mul_alpha_inputs = {bmm_param, alpha_param_node};
  AnfNodePtr alpha_mul_param = nullptr;
  if (CreateMulNode(mul_alpha_inputs, func_graph, node, "_lora_alpha_mul", &alpha_mul_param) != RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto mul_add_weights_inputs = {alpha_mul_param, add_weights_param_node};
  AnfNodePtr mul_add_weights_param = nullptr;
  if (CreateMulNode(mul_add_weights_inputs, func_graph, node, "_lora_add_weights_mul", &mul_add_weights_param) !=
      RET_OK) {
    MS_LOG(ERROR) << "Create mul node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto reduce_sum_inputs = {mul_add_weights_param, axes_param_node};
  AnfNodePtr reduce_sum_param = nullptr;
  if (CreateReduceSumNode(reduce_sum_inputs, func_graph, node, "_lora_reduce_sum", &reduce_sum_param) != RET_OK) {
    MS_LOG(ERROR) << "Create reducesum node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  // transpose
  auto transpose_inputs = {reduce_sum_param, perm_param_node};
  AnfNodePtr transpose_param = nullptr;
  if (CreateTransposeNode(transpose_inputs, func_graph, node, "_lora_transpose", &transpose_param) != RET_OK) {
    MS_LOG(ERROR) << "Create transpose node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  // add
  auto add_inputs = {transpose_param, weight};
  CNodePtr add_cnode = nullptr;
  if (CreateAddNode(add_inputs, func_graph, node, "_lora_add", &add_cnode) != RET_OK) {
    MS_LOG(ERROR) << "Create add node failed! ret:" << ret << "!";
    return RET_ERROR;
  }
  auto manager = Manage(func_graph);
  (void)manager->Replace(weight, add_cnode);
  return RET_OK;
}

STATUS InsertVariableNodePass::ParseShapeStr(std::string shape_str, std::vector<int> *shape) {
  int shape_len = shape_str.size();
  if (shape_len <= 2) {
    MS_LOG(ERROR) << "size of shape_str:" << shape_len << " <= 2! It must larger than 2!";
    return RET_ERROR;
  }
  std::string shape_nums = shape_str.substr(1, shape_len - 2);
  std::stringstream ss(shape_nums);
  std::string token;
  while (std::getline(ss, token, ',')) {
    shape->push_back(std::stoi(token));
  }
  if (shape->size() != kConstantConvWeightShapeSize && shape->size() != kConstantMatmulWeightShapeSize) {
    MS_LOG(ERROR) << "Weight shape is " << shape->size() << ", it should be 2 or 4!";
    return RET_FAILED;
  }
  return RET_OK;
}

lite::STATUS InsertVariableNodePass::ParseInsertNode(std::string file_path,
                                                     std::map<std::string, std::vector<int>> *variable_nodes,
                                                     std::unordered_map<std::string, std::string> *node_name_map,
                                                     std::vector<std::string> *node_name_list, bool *has_alpha) {
  MS_CHECK_TRUE_RET(variable_nodes != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(node_name_map != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(node_name_list != nullptr, lite::RET_NULL_PTR);
  MS_CHECK_TRUE_RET(has_alpha != nullptr, lite::RET_NULL_PTR);
  std::ifstream file;
  auto ret = lite::ReadFileToIfstream(file_path, &file);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "read file to ifstream failed!";
    return ret;
  }
  size_t config_len = 0;
  std::string line;
  while (std::getline(file, line)) {
    config_len++;
    if (config_len >= kMaxConfigLen) {
      MS_LOG(ERROR) << "Support max config len is " << kMaxConfigLen << ", current len:" << config_len << "!";
      return RET_ERROR;
    }
    auto pos_colon = line.find(':');
    if (pos_colon == std::string::npos) {
      MS_LOG(ERROR) << "Parse variable weight file error!";
      file.close();
      return RET_FAILED;
    }
    auto variable_para_name = line.substr(0, pos_colon);
    if (variable_para_name.find("alpha") != std::string::npos && (*has_alpha) != true) {
      (*has_alpha) = true;
    }
    auto pos_semicolon = line.find(';');
    if (pos_semicolon == std::string::npos) {
      MS_LOG(ERROR) << "Parse variable weight file error!";
      file.close();
      return RET_FAILED;
    }
    auto weight_shape_str = line.substr(pos_colon + 1, pos_semicolon - pos_colon - 1);
    auto node_name = line.substr(pos_semicolon + 1);
    std::string record_name = "";
    if (variable_para_name.find(".up.") != std::string::npos ||
        variable_para_name.find("lora_up") != std::string::npos) {
      record_name = node_name + "variable_up";
    } else if (variable_para_name.find(".down.") != std::string::npos ||
               variable_para_name.find("lora_down") != std::string::npos) {
      record_name = node_name + "variable_down";
    } else if (variable_para_name.find("alpha") != std::string::npos) {
      record_name = node_name + "variable_alpha";
    } else {
      MS_LOG(ERROR) << "Only support up weight, down weight and alpha!";
      return RET_ERROR;
    }
    if (node_name_map->find(record_name) == node_name_map->end()) {
      (*node_name_map)[record_name] = "";
      (*node_name_list).push_back(record_name);
    }
    // Only Upsape is recorded, so that you can easily check node name
    if (variable_para_name.find(".up.") == std::string::npos &&
        variable_para_name.find("lora_up") == std::string::npos) {
      continue;
    }
    std::vector<int> shape;
    ret = ParseShapeStr(weight_shape_str, &shape);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ParseShapeStr error! ret:" << ret << "!";
      file.close();
      return ret;
    }
    variable_nodes->insert({node_name, shape});
  }
  file.close();
  return RET_OK;
}

lite::STATUS InsertVariableNodePass::CheckOnlyReplace(CNodePtr cnode, const std::vector<int> &para_shape,
                                                      const bool &is_matmul, bool *compare_res) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr!");
  MS_CHECK_TRUE_MSG(compare_res != nullptr, RET_ERROR, "compare_res is nullptr!");
  auto weight = cnode->input(kInputIndex2);
  MS_CHECK_TRUE_RET(weight != nullptr, RET_ERROR);
  ShapeVector weight_shape;
  auto ret = FetchWeightShape(weight, &weight_shape, cnode, is_matmul);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch wieght shape failed! ret:" << ret << "!";
    return ret;
  }
  if (weight_shape.size() != para_shape.size()) {
    *compare_res = false;
    return RET_OK;
  }
  *compare_res = std::equal(weight_shape.begin(), weight_shape.end(), para_shape.begin());
  return RET_OK;
}

lite::STATUS InsertVariableNodePass::RecordVariableName(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                                        const string &search_key, bool is_matmul,
                                                        std::unordered_map<std::string, std::string> *node_name_map) {
  MS_CHECK_TRUE_RET(node_name_map != nullptr, RET_ERROR);
  MS_CHECK_TRUE_RET(cnode != nullptr, RET_ERROR);
  if (cnode->inputs().size() < kInputSize3) {
    MS_LOG(ERROR) << "Weight size must greater than 3, current size:" << cnode->inputs().size() << "!";
    return RET_ERROR;
  }
  auto weight = cnode->input(kInputIndex2);
  MS_CHECK_TRUE_RET(weight != nullptr, RET_ERROR);
  ShapeVector weight_shape;
  auto ret = FetchWeightShape(weight, &weight_shape, cnode, is_matmul);
  if (ret != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch wieght shape failed! ret:" << ret << "!";
    return ret;
  }
  AnfNodePtr fp16_weight = BuildFloat16ZeroVecNDParameterNode<uint16_t>(
    func_graph, weight_shape, weight->fullname_with_scope(), 0.0, kNumberTypeFloat16);
  MS_CHECK_TRUE_MSG(fp16_weight != nullptr, RET_ERROR, "fp16_weight is nullptr!");
  (*node_name_map)[search_key + "variable_up"] = weight->fullname_with_scope() + "_const";
  auto manager = Manage(func_graph);
  MS_CHECK_TRUE_RET(manager != nullptr, RET_ERROR);
  (void)manager->Replace(weight, fp16_weight);
  return RET_OK;
}

void InsertVariableNodePass::InitWeightParam(const std::shared_ptr<ConverterPara> &param,
                                             std::string *variable_weights_file, int32_t *max_weight_batch) {
  if (param->config_infos.find(lite::kAscendContextSection) != param->config_infos.end()) {
    auto ascend_context = param->config_infos.at(lite::kAscendContextSection);
    if (ascend_context.find(lite::kVariableWeightsFile) != ascend_context.end()) {
      *variable_weights_file = ascend_context.at(lite::kVariableWeightsFile);
    }
    if (ascend_context.find(lite::kMaxWeightBatch) != ascend_context.end()) {
      *max_weight_batch = std::stoi(ascend_context.at(lite::kMaxWeightBatch));
    }
  }
}

lite::STATUS InsertVariableNodePass::BuildVariableNode(const std::shared_ptr<ConverterPara> &param,
                                                       FuncGraphPtr func_graph, std::vector<std::string> *const_names) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, RET_ERROR);
  std::string variable_weights_file = "";
  int32_t max_weight_batch = 1;
  InitWeightParam(param, &variable_weights_file, &max_weight_batch);
  MS_CHECK_TRUE_RET(variable_weights_file != "", RET_OK);
  bool has_alpha = false;
  std::map<std::string, std::vector<int>> variable_nodes;
  std::unordered_map<std::string, std::string> node_name_map;
  std::vector<std::string> node_name_list;
  auto ret = ParseInsertNode(variable_weights_file, &variable_nodes, &node_name_map, &node_name_list, &has_alpha);
  MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "ParseInsertNode failed!");
  uint32_t matched_num = 0;
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    auto node_name = node->fullname_with_scope();
    size_t last_slash_pos = node_name.find_last_of('/');
    std::string search_key = "";
    if (last_slash_pos != std::string::npos) {
      search_key = node_name.substr(0, last_slash_pos);
    } else {
      MS_LOG(INFO) << "Find last slash failed! Cnode name:" << node->fullname_with_scope() << "!";
      continue;
    }
    if (variable_nodes.find(search_key) == variable_nodes.end() || !utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = utils::cast<CNodePtr>(node);
    MS_CHECK_TRUE_RET(cnode != nullptr, false);
    if (mindspore::opt::CheckPrimitiveType(node, mindspore::prim::kPrimMatMulV2) ||
        mindspore::opt::CheckPrimitiveType(node, mindspore::prim::kPrimMatMulFusion) ||
        mindspore::opt::CheckPrimitiveType(node, mindspore::prim::kPrimBatchMatMul)) {
      bool replace_origin = false;
      ret = CheckOnlyReplace(cnode, variable_nodes.at(search_key), true, &replace_origin);
      MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "CheckOnlyReplace failed!");
      if (replace_origin) {
        ret = RecordVariableName(func_graph, cnode, search_key, true, &node_name_map);
      } else {
        ret = InsertVariableNodeForMatmul(node, cnode, func_graph, variable_nodes.at(search_key), &node_name_map,
                                          has_alpha, max_weight_batch);
      }
    } else if (mindspore::opt::CheckPrimitiveType(node, mindspore::prim::kPrimConv2D) ||
               mindspore::opt::CheckPrimitiveType(node, mindspore::prim::kPrimConv2DFusion)) {
      bool replace_origin = false;
      ret = CheckOnlyReplace(cnode, variable_nodes.at(search_key), false, &replace_origin);
      MS_CHECK_TRUE_MSG(ret == RET_OK, ret, "CheckOnlyReplace failed!");
      if (replace_origin) {
        ret = RecordVariableName(func_graph, cnode, search_key, false, &node_name_map);
      } else {
        ret = InsertVariableNodeForConv(node, cnode, func_graph, variable_nodes.at(search_key), &node_name_map,
                                        has_alpha, max_weight_batch);
      }
    } else {
      continue;
    }
    matched_num++;
  }
  if (matched_num != variable_nodes.size()) {
    MS_LOG(ERROR) << "matched num:" << matched_num << " != all node num:" << variable_nodes.size() << "!";
    return RET_ERROR;
  }
  for (auto s : node_name_list) {
    if (node_name_map.find(s) == node_name_map.end()) {
      continue;
    }
    const_names->push_back(node_name_map[s]);
  }
  return RET_OK;
}

bool InsertVariableNodePass::Run(const FuncGraphPtr &graph) {
  if (BuildVariableNode(param_, graph, &(param_->const_names)) != RET_OK) {
    return false;
  }
  if (param_->const_names.size() > 0) {
    graph->set_attr(lite::kBundleModel, MakeValue("True"));
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
