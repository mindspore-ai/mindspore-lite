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

#ifndef MINDSPORE_LITE_SRC_LITERT_DELEGATE_FUSION_PASS_UTILS_H_
#define MINDSPORE_LITE_SRC_LITERT_DELEGATE_FUSION_PASS_UTILS_H_

#include <vector>
#include <set>
#include <functional>
#include "include/api/types.h"
#include "schema/inner/model_generated.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace delegate {

// Fallback empty type for C++14 compatibility (replaces EmptyType)
struct EmptyType {};

// Common NHWC/NCHW index constants (unified across delegates)
// Use prefixed names to avoid macro conflicts with op_base.h
constexpr int kDelegateNHWC_N = 0;
constexpr int kDelegateNHWC_H = 1;
constexpr int kDelegateNHWC_W = 2;
constexpr int kDelegateNHWC_C = 3;

// Common shape size
constexpr int kCommShapeSize = 4;

/**
 * @brief Template function to update pre-tensors for fusion passes
 *
 * This function handles the common logic for updating input tensors
 * before fusion, supporting both CoreMLOp and NPUOp types.
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @param cur_op Current operation to update
 * @param shape_size Size of shape tensor (usually 4)
 * @return int RET_OK on success, error code otherwise
 */
template <typename T>
int UpdatePreTensorsForFusion(T *cur_op, int shape_size = kCommShapeSize) {
  auto in_tensors_vec = cur_op->inputs();
  for (auto in_op : cur_op->in_ops()) {
    if (in_op->inputs().empty() || in_op->outputs().empty()) {
      MS_LOG(ERROR) << "in_tensors or out_tensors of input op is empty.";
      return RET_ERROR;
    }
    mindspore::MSTensor cur_tensor;
    auto in_tensor = in_op->inputs()[0];
    auto out_tensor = in_op->outputs()[0];
    if (!in_op->in_ops().empty()) {
      auto pre_op = in_op->in_ops()[0];
      for (size_t i = 0; i < pre_op->outputs().size(); i++) {
        if (pre_op->outputs()[i] == in_tensor) {
          cur_tensor = pre_op->outputs()[i];
          break;
        }
      }
    } else {
      // graph input
      cur_tensor = in_tensor;
    }

    for (size_t i = 0; i < in_tensors_vec.size(); i++) {
      if (in_tensors_vec[i] == out_tensor) {
        in_tensors_vec[i] = cur_tensor;
      }
    }
  }
  cur_op->set_inputs(in_tensors_vec);
  return RET_OK;
}

/**
 * @brief Template function to update post-tensors for fusion passes
 *
 * This function handles the common logic for updating output tensors
 * after fusion, supporting both CoreMLOp and NPUOp types.
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @param cur_op Current operation to update
 * @param shape_size Size of shape tensor (usually 4)
 * @param nhwc_n Index for N in NHWC format (usually 0)
 * @param nhwc_c Index for C in NHWC format (usually 3)
 * @param nhwc_h Index for H in NHWC format (usually 1)
 * @param nhwc_w Index for W in NHWC format (usually 2)
 * @return int RET_OK on success, error code otherwise
 */
template <typename T>
int UpdatePostTensorsForFusion(T *cur_op, int shape_size = kCommShapeSize, int nhwc_n = kDelegateNHWC_N,
                               int nhwc_c = kDelegateNHWC_C, int nhwc_h = kDelegateNHWC_H,
                               int nhwc_w = kDelegateNHWC_W) {
  mindspore::MSTensor new_post_input;
  for (auto out_op : cur_op->out_ops()) {
    auto in_tensor = out_op->inputs()[0];
    auto out_tensor = out_op->outputs()[0];
    auto nhwc_shape = in_tensor.Shape();
    if (in_tensor.format() == Format::NHWC) {
      MS_CHECK_TRUE_MSG(nhwc_shape.size() == shape_size, RET_ERROR, "Invalid transpose dim size!");
      in_tensor.SetShape({nhwc_shape[nhwc_n], nhwc_shape[nhwc_c], nhwc_shape[nhwc_h], nhwc_shape[nhwc_w]});
      in_tensor.SetFormat(Format::NCHW);
    }
    // out_op is a graph output op
    if (out_op->out_ops().empty()) {
      auto out_tensors_vec = cur_op->outputs();
      for (size_t i = 0; i < out_tensors_vec.size(); i++) {
        if (out_tensors_vec[i] == in_tensor) {
          out_tensors_vec[i] = out_op->outputs()[0];
        }
      }
      cur_op->set_outputs(out_tensors_vec);
      // exist other out_ops using the same tensor as the current out_op, note that the other out_op has likely been
      // updated, which mean it may be not a Transpose op anymore.
      for (auto other_out_op : cur_op->out_ops()) {
        auto other_in_tensors_vec = other_out_op->inputs();
        for (size_t i = 0; i < other_in_tensors_vec.size(); i++) {
          if (other_in_tensors_vec[i] == in_tensor) {
            other_in_tensors_vec[i] = out_op->outputs()[0];
          }
        }
        other_out_op->set_inputs(other_in_tensors_vec);
      }
    }
    // out_op is not a graph out op
    for (auto post_op : out_op->out_ops()) {
      auto in_tensors_vec = post_op->inputs();
      for (size_t i = 0; i < in_tensors_vec.size(); i++) {
        if (in_tensors_vec[i] == out_tensor) {
          in_tensors_vec[i] = in_tensor;
        }
      }
      post_op->set_inputs(in_tensors_vec);
    }
  }
  return RET_OK;
}

/**
 * @brief Template function to update pre-ops for fusion passes
 *
 * This function handles the common logic for updating input operations
 * before fusion, supporting both CoreMLOp and NPUOp types.
 * NPU version includes deduplication check via std::set.
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam EnableDedup Whether to enable deduplication check (true for NPU, false for CoreML)
 * @param cur_op Current operation to update
 * @param all_ops Pointer to all ops vector (needed for RemoveAndFreeOp)
 * @param remove_and_free_op Function to remove and free an op
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, bool EnableDedup = false>
int UpdatePreOps(T *cur_op, std::vector<T *> *all_ops, std::function<void(T *)> remove_and_free_op) {
  auto cur_in_ops = cur_op->in_ops();
  std::conditional_t<EnableDedup, std::set<T *>, EmptyType> has_visited;

  for (auto in_op : cur_op->in_ops()) {
    if constexpr (EnableDedup) {
      if (has_visited.find(in_op) != has_visited.end()) {
        continue;
      }
    }
    // graph in op
    if (in_op->in_ops().empty()) {
      cur_in_ops.erase(find(cur_in_ops.begin(), cur_in_ops.end(), in_op));
    } else {
      auto pre_op = in_op->in_ops()[0];
      auto pre_out_ops = pre_op->out_ops();
      for (size_t i = 0; i < pre_out_ops.size(); i++) {
        if (pre_out_ops[i] == in_op) {
          pre_out_ops[i] = cur_op;
          if constexpr (!EnableDedup) break;
        }
      }
      pre_op->set_out_ops(pre_out_ops);

      for (size_t i = 0; i < cur_in_ops.size(); i++) {
        if (cur_in_ops[i] == in_op) {
          cur_in_ops[i] = pre_op;
          if constexpr (!EnableDedup) break;
        }
      }
    }
    if (remove_and_free_op) {
      remove_and_free_op(in_op);
    }
    if constexpr (EnableDedup) {
      (void)has_visited.insert(in_op);
    }
  }
  cur_op->set_in_ops(cur_in_ops);
  return RET_OK;
}

/**
 * @brief Template function to update post-ops for fusion passes
 *
 * This function handles the common logic for updating output operations
 * after fusion, supporting both CoreMLOp and NPUOp types.
 * NPU version includes deduplication check via std::set.
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam EnableDedup Whether to enable deduplication check (true for NPU, false for CoreML)
 * @param cur_op Current operation to update
 * @param all_ops Pointer to all ops vector (needed for RemoveAndFreeOp)
 * @param remove_and_free_op Function to remove and free an op
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, bool EnableDedup = false>
int UpdatePostOps(T *cur_op, std::vector<T *> *all_ops, std::function<void(T *)> remove_and_free_op) {
  auto cur_out_ops = cur_op->out_ops();
  std::conditional_t<EnableDedup, std::set<T *>, EmptyType> has_visited;

  for (auto out_op : cur_op->out_ops()) {
    if constexpr (EnableDedup) {
      if (has_visited.find(out_op) != has_visited.end()) {
        continue;
      }
    }
    // graph out op
    if (out_op->out_ops().empty()) {
      cur_out_ops.erase(find(cur_out_ops.begin(), cur_out_ops.end(), out_op));
    } else {
      auto post_op = out_op->out_ops()[0];
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == out_op) {
          post_in_ops[i] = cur_op;
          if constexpr (!EnableDedup) break;
        }
      }
      post_op->set_in_ops(post_in_ops);

      for (size_t i = 0; i < cur_out_ops.size(); i++) {
        if (cur_out_ops[i] == out_op) {
          cur_out_ops[i] = post_op;
          if constexpr (!EnableDedup) break;
        }
      }
    }
    if (remove_and_free_op) {
      remove_and_free_op(out_op);
    }
    if constexpr (EnableDedup) {
      (void)has_visited.insert(out_op);
    }
  }
  cur_op->set_out_ops(cur_out_ops);
  return RET_OK;
}

/**
 * @brief Template function for complete update operation
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam EnableDedup Whether to enable deduplication check
 * @param cur_op Current operation to update
 * @param all_ops Pointer to all ops vector
 * @param remove_and_free_op Function to remove and free an op
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, bool EnableDedup = false>
int UpdateOp(T *cur_op, std::vector<T *> *all_ops, std::function<void(T *)> remove_and_free_op) {
  if (cur_op == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return RET_ERROR;
  }
  auto ret = UpdatePreTensorsForFusion(cur_op, kCommShapeSize);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePreTensors failed.";
    return RET_ERROR;
  }
  ret = UpdatePostTensorsForFusion(cur_op, kCommShapeSize);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePostTensors failed.";
    return RET_ERROR;
  }
  ret = UpdatePreOps<T, EnableDedup>(cur_op, all_ops, remove_and_free_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePreOps failed.";
    return RET_ERROR;
  }
  ret = UpdatePostOps<T, EnableDedup>(cur_op, all_ops, remove_and_free_op);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UpdatePostOps failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

/**
 * @brief Template function to update out ops of pre op during format fusion
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @param cur_op Current operation
 * @param found_graph_out_tensor Whether graph output tensor was found
 * @param graph_out_tensor Graph output tensor
 * @param pre_insert_ops Pre-insert operations
 */
template <typename T>
void UpdateOutOpsOfPreOp(T *cur_op, bool found_graph_out_tensor, const mindspore::MSTensor &graph_out_tensor,
                         const std::vector<T *> &pre_insert_ops) {
  if (cur_op == nullptr) {
    MS_LOG(ERROR) << "cur_op is nullptr.";
    return;
  }
  auto is_graph_input = cur_op->in_ops().empty();
  auto cur_op_in_tensor = cur_op->inputs()[0];
  if (!is_graph_input) {
    auto pre_op = cur_op->in_ops()[0];
    auto pre_out_ops = pre_op->out_ops();
    size_t cur_op_index = 0;
    for (size_t index = 0; index < pre_out_ops.size(); index++) {
      if (pre_out_ops[index] == cur_op) {
        pre_out_ops.erase(pre_out_ops.begin() + index);
        cur_op_index = index;
        index--;
      } else if (found_graph_out_tensor) {
        // only in this case, the output of pre_op is specified to 2nd trans op's output
        auto tensors_vec = pre_out_ops[index]->inputs();
        for (size_t i = 0; i < tensors_vec.size(); i++) {
          if (tensors_vec[i] == cur_op_in_tensor) {
            tensors_vec[i] = graph_out_tensor;
            break;
          }
        }
        pre_out_ops[index]->set_inputs(tensors_vec);
      }
    }
    pre_out_ops.insert(pre_out_ops.begin() + cur_op_index, pre_insert_ops.begin(), pre_insert_ops.end());
    pre_op->set_out_ops(pre_out_ops);
  }
}

/**
 * @brief Template function for FormatFusion operation
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam GraphType Graph type (CoreMLGraph or NPUGraph)
 * @param cur_op Current operation
 * @param subgraph Pointer to subgraph
 * @param all_ops Pointer to all ops vector
 * @param all_tensors Pointer to all tensors vector
 * @param name Pass name
 * @param remove_and_free_op Function to remove and free an op
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, typename GraphType>
int FormatFusion(T *cur_op, GraphType *subgraph, std::vector<T *> *all_ops, const std::string &name,
                 std::function<void(T *)> remove_and_free_op) {
  CHECK_NULL_RETURN(cur_op);
  auto is_graph_input = cur_op->in_ops().empty();
  auto cur_op_in_tensor = cur_op->inputs()[0];
  std::vector<T *> pre_insert_ops;
  T *pre_op = nullptr;
  if (!is_graph_input) {
    pre_op = cur_op->in_ops()[0];
  }
  mindspore::MSTensor graph_out_tensor;
  bool found_graph_out_tensor = false;
  auto graph_outputs = subgraph->outputs();
  // if the output of second trans op(s) is graph output, find it out and use it as the pre-op's output.
  for (const auto &sec_op : cur_op->out_ops()) {
    if (std::find(graph_outputs.begin(), graph_outputs.end(), sec_op->outputs()[0]) != graph_outputs.end()) {
      graph_out_tensor = sec_op->outputs()[0];
      if (!is_graph_input) {
        found_graph_out_tensor = true;
        // cur_op is the first trans op, it's input op num and input tensor num must be 1
        pre_op->set_outputs({graph_out_tensor});
        // in fp16 mode, tensor data type fp16 need to be changed back.
        auto tensor = pre_op->outputs()[0];
        if (tensor.DataType() == DataType::kNumberTypeFloat16) {
          tensor.SetDataType(DataType::kNumberTypeFloat32);
        }
        break;
      } else {
        MS_LOG(WARNING) << "Existing graph output equivalent to graph input, which is unsupported now.";
        return RET_OK;
      }
    }
  }
  for (const auto &trans_op : cur_op->out_ops()) {
    for (const auto &post_op : trans_op->out_ops()) {
      // update tensor
      auto tensors_vec = post_op->inputs();
      for (size_t i = 0; i < tensors_vec.size(); i++) {
        if (tensors_vec[i] == trans_op->outputs()[0]) {
          tensors_vec[i] = found_graph_out_tensor ? graph_out_tensor : cur_op_in_tensor;
          break;
        }
      }
      post_op->set_inputs(tensors_vec);

      // update op
      auto post_in_ops = post_op->in_ops();
      for (size_t i = 0; i < post_in_ops.size(); i++) {
        if (post_in_ops[i] == trans_op) {
          if (is_graph_input) {
            post_in_ops.erase(post_in_ops.begin() + i);
          } else {
            post_in_ops[i] = pre_op;
          }
          break;
        }
      }
      post_op->set_in_ops(post_in_ops);
      pre_insert_ops.push_back(post_op);
    }
    if (remove_and_free_op) {
      remove_and_free_op(trans_op);
    }
  }
  UpdateOutOpsOfPreOp(cur_op, found_graph_out_tensor, graph_out_tensor, pre_insert_ops);
  if (remove_and_free_op) {
    remove_and_free_op(cur_op);
  }
  return RET_OK;
}

// ============================================================================
// Transform Pass Template Functions
// ============================================================================

/**
 * @brief Strategy interface for Transform Pass utilities
 *
 * This interface defines the required utilities for Transform Pass operations.
 * Each delegate (CoreML, NPU) should provide an implementation of this interface.
 */
template <typename T>
struct TransformPassUtils {
  // Check if op is nchw2nhwc transpose
  static bool IsNchw2Nhwc(T *op);

  // Check if op is nhwc2nchw transpose
  static bool IsNhwc2Nchw(T *op);

  // Get non-constant inputs from op
  static std::vector<mindspore::MSTensor> GetNonConstInputs(T *op);

  // Get input op from tensor
  static T *OpInputFromOp(T *op, const mindspore::MSTensor &tensor);

  // Create nhwc2nchw op
  static T *CreateNhwc2NchwOp(const std::vector<mindspore::MSTensor> &inputs,
                              const std::vector<mindspore::MSTensor> &outputs, const std::string &name);

  // Create nchw2nhwc op
  static T *CreateNchw2NhwcOp(const std::vector<mindspore::MSTensor> &inputs,
                              const std::vector<mindspore::MSTensor> &outputs, const std::string &name);

  // Update op connections
  static void UpdateOp(T *op, const std::vector<T *> &in_ops, const std::vector<T *> &out_ops,
                       const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs);

  // Update nh2nc trans node pre op
  static void UpdateNH2NCTransNodePreOp(T *cur_op, T *nh2nc_op, T *post_op);

  // Update nc2nh trans node post op
  static void UpdateNC2NHTransNodePostOp(T *cur_op, T *nc2nh_op, T *post_op,
                                         const mindspore::MSTensor &trans_in_tensor);
};

/**
 * @brief Template function to get insert state for transform pass
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam Utils Utils type with static methods for pass utilities
 * @param op Current operation
 * @param subgraph_outputs Subgraph outputs
 * @param format_depend_nodes Set of format-dependent node types
 * @return lite::InsertState Insert state enum
 */
template <typename T, typename Utils>
lite::InsertState GetInsertState(T *op, const std::vector<mindspore::MSTensor> &subgraph_outputs,
                                 const std::set<mindspore::schema::PrimitiveType> &format_depend_nodes) {
  // filter out irrelevant op
  if (format_depend_nodes.find(op->type()) != format_depend_nodes.end()) {
    return lite::InsertState::InsertNone;
  }

  std::vector<mindspore::MSTensor> inputs = Utils::GetNonConstInputs(op);
  size_t in_out_tensor_num =
    inputs.size() + std::max(std::max(op->out_ops().size(), static_cast<size_t>(1)), op->outputs().size());
  size_t transpose_input_num = 0;
  size_t transpose_output_num = 0;
  size_t graph_input_num = 0;
  size_t graph_output_num = 0;
  bool need_pre_insert = false;
  bool need_post_insert = false;

  // count number of input tensor from nc2nh and output tensor to nh2nc
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto in_op = Utils::OpInputFromOp(op, inputs.at(i));
    if (Utils::IsNchw2Nhwc(in_op)) {
      transpose_input_num++;
    } else {
      need_pre_insert = true;
    }
    if (in_op == nullptr) {
      graph_input_num++;
    }
  }

  for (auto output : op->outputs()) {
    if (std::find(subgraph_outputs.begin(), subgraph_outputs.end(), output) != subgraph_outputs.end()) {
      graph_output_num++;
      need_post_insert = true;
    }
  }

  for (const auto out_op : op->out_ops()) {
    for (auto out_op_input : out_op->inputs()) {
      if (std::find(subgraph_outputs.begin(), subgraph_outputs.end(), out_op_input) != subgraph_outputs.end()) {
        in_out_tensor_num++;
      }
    }
    if (Utils::IsNhwc2Nchw(out_op)) {
      transpose_output_num++;
    } else {
      need_post_insert = true;
    }
  }

  // won't insert any thing if num of transpose tensor is smaller than half of total op inputs and op outputs
  size_t transpose_tensor_num = transpose_input_num + transpose_output_num;
  size_t connected_in_out_tensor_num = in_out_tensor_num - graph_output_num - graph_input_num;
  if (transpose_tensor_num == 0 || transpose_tensor_num * lite::REPEAT_TIMES2 < connected_in_out_tensor_num ||
      transpose_tensor_num == in_out_tensor_num) {
    return lite::InsertState::InsertNone;
  }

  lite::InsertState ret =
    (need_pre_insert && need_post_insert)
      ? lite::InsertState::BothInsert
      : (need_pre_insert ? lite::InsertState::PreInsert
                         : (need_post_insert ? lite::InsertState::PostInsert : lite::InsertState::InsertNone));

  return ret;
}

/**
 * @brief Template function to insert transform node
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam Utils Utils type with static methods for pass utilities
 * @param op Current operation
 * @param post_op Post operation
 * @param trans_in_tensor Input tensor for transform
 * @param trans_ops Output vector for transform ops
 * @param all_tensors All tensors vector
 * @param name Pass name
 * @param total Counter for naming
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, typename Utils>
int InsertTransNode(T *op, T *post_op, const mindspore::MSTensor &trans_in_tensor, std::vector<T *> *trans_ops,
                    std::vector<mindspore::MSTensor *> *all_tensors, const std::string &name, int &total) {
  if (op == nullptr && post_op == nullptr) {
    MS_LOG(ERROR) << "op and post_op are both nullptr.";
    return RET_ERROR;
  }
  std::string op_name;
  std::vector<T *> in_ops;
  std::vector<T *> out_ops;
  if (op != nullptr) {
    op_name = op->name() + "_post";
    in_ops.emplace_back(op);
  }
  if (post_op != nullptr) {
    op_name = post_op->name() + "_pre";
    out_ops.emplace_back(post_op);
  }

  auto nhwc_shape = trans_in_tensor.Shape();
  std::vector<int64_t> nchw_shape = {nhwc_shape[kDelegateNHWC_N], nhwc_shape[kDelegateNHWC_C],
                                     nhwc_shape[kDelegateNHWC_H], nhwc_shape[kDelegateNHWC_W]};

  auto nh2nc_name = op_name + "_nh2nc_" + std::to_string(total++);
  auto nh2nc_tensor =
    mindspore::MSTensor::CreateTensor(nh2nc_name + "/output0", trans_in_tensor.DataType(), nchw_shape, nullptr, 0);
  if (nh2nc_tensor == nullptr) {
    MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc op.";
    return RET_ERROR;
  }
  nh2nc_tensor->SetFormat(Format::NCHW);
  std::vector<mindspore::MSTensor> nh2nc_tensors = {*nh2nc_tensor};
  all_tensors->push_back(nh2nc_tensor);

  auto nc2nh_name = op_name + "_nc2nh_" + std::to_string(total++);
  auto nc2nh_tensor =
    mindspore::MSTensor::CreateTensor(nc2nh_name + "/output0", trans_in_tensor.DataType(), nhwc_shape, nullptr, 0);
  if (nc2nh_tensor == nullptr) {
    MS_LOG(ERROR) << "New nhwc tensor failed when inserting nhwc2nchw op.";
    return RET_ERROR;
  }
  nc2nh_tensor->SetFormat(Format::NHWC);
  std::vector<mindspore::MSTensor> nc2nh_tensors = {*nc2nh_tensor};
  all_tensors->push_back(nc2nh_tensor);

  auto *nh2nc_op = Utils::CreateNhwc2NchwOp({trans_in_tensor}, nh2nc_tensors, nh2nc_name);
  if (nh2nc_op == nullptr) {
    MS_LOG(ERROR) << "nh2nc_op is nullptr.";
    return RET_ERROR;
  }
  trans_ops->push_back(nh2nc_op);

  auto *nc2nh_op = Utils::CreateNchw2NhwcOp(nh2nc_tensors, nc2nh_tensors, nc2nh_name);
  if (nc2nh_op == nullptr) {
    MS_LOG(ERROR) << "nc2nh_op is nullptr.";
    return RET_ERROR;
  }
  trans_ops->push_back(nc2nh_op);

  Utils::UpdateOp(nh2nc_op, in_ops, {nc2nh_op}, {trans_in_tensor}, nh2nc_tensors);
  Utils::UpdateOp(nc2nh_op, {nh2nc_op}, out_ops, {nh2nc_tensors[0]}, nc2nh_tensors);

  if (op != nullptr) {
    Utils::UpdateNH2NCTransNodePreOp(op, nh2nc_op, post_op);
  }
  if (post_op != nullptr) {
    Utils::UpdateNC2NHTransNodePostOp(op, nc2nh_op, post_op, trans_in_tensor);
  } else {
    // post_op nullptr mean output, we remain graph output tensor name unchanged
    auto graph_output_name = trans_in_tensor.Name();
    nc2nh_tensor->SetTensorName(graph_output_name + "_after_" + name);
  }
  return RET_OK;
}

/**
 * @brief Template function to insert pre nodes for transform pass
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam Utils Utils type with static methods for pass utilities
 * @param op Current operation
 * @param trans_ops Output vector for transform ops
 * @param subgraph_outputs Subgraph outputs
 * @param all_tensors All tensors vector
 * @param name Pass name
 * @param total Counter for naming
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, typename Utils>
int InsertPreNodes(T *op, std::vector<T *> *trans_ops, const std::vector<mindspore::MSTensor> &subgraph_outputs,
                   std::vector<mindspore::MSTensor *> *all_tensors, const std::string &name, int &total) {
  int ret = RET_OK;
  auto inputs = Utils::GetNonConstInputs(op);
  for (auto tensor : inputs) {
    if (tensor.Shape().size() < kCommShapeSize) {
      continue;
    }
    // the input tensor can only come from a single op
    auto pre_op = Utils::OpInputFromOp(op, tensor);
    if (Utils::IsNchw2Nhwc(pre_op)) {
      continue;
    }
    // if this tensor is input of graph, pre_op is nullptr.
    ret = InsertTransNode<T, Utils>(pre_op, op, tensor, trans_ops, all_tensors, name, total);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
      return ret;
    }
  }
  return ret;
}

/**
 * @brief Template function to insert post nodes for transform pass
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam Utils Utils type with static methods for pass utilities
 * @param op Current operation
 * @param trans_ops Output vector for transform ops
 * @param subgraph_outputs Subgraph outputs
 * @param all_tensors All tensors vector
 * @param name Pass name
 * @param total Counter for naming
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, typename Utils>
int InsertPostNodes(T *op, std::vector<T *> *trans_ops, const std::vector<mindspore::MSTensor> &subgraph_outputs,
                    std::vector<mindspore::MSTensor *> *all_tensors, const std::string &name, int &total) {
  int ret = RET_OK;
  for (size_t idx = 0; idx < op->outputs().size(); idx++) {
    auto out_tensor = op->outputs().at(idx);
    if (out_tensor.Shape().size() < kCommShapeSize) {
      continue;
    }
    if (std::find(subgraph_outputs.begin(), subgraph_outputs.end(), out_tensor) != subgraph_outputs.end()) {
      // the case that op's out tensor is graph output
      ret = InsertTransNode<T, Utils>(op, nullptr, op->outputs().at(idx), trans_ops, all_tensors, name, total);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
        return RET_ERROR;
      }
      // use origin output as the last trans op's output in order to avoid the lost of the output tensor
      auto last_trans = trans_ops->back();
      auto trans_output = last_trans->outputs();
      auto cur_outputs = op->outputs();
      cur_outputs[idx] = last_trans->outputs()[0];
      trans_output[0] = op->outputs()[idx];
      last_trans->set_outputs(trans_output);
      op->set_outputs(cur_outputs);
    }

    // besides of being as graph outputs, the output tensors also can connected with multiple ops.
    for (auto post_op : op->out_ops()) {
      auto post_op_input = post_op->inputs();
      auto it = std::find(post_op_input.begin(), post_op_input.end(), out_tensor);
      if (it == post_op_input.end()) {
        continue;
      }
      auto related_idx = it - post_op_input.begin();
      post_op_input[related_idx] = op->outputs().at(idx);
      post_op->set_inputs(post_op_input);

      if (Utils::IsNhwc2Nchw(post_op)) {
        continue;
      }
      // the case that op's out tensor is one of post_op's input tensor
      ret = InsertTransNode<T, Utils>(op, post_op, op->outputs().at(idx), trans_ops, all_tensors, name, total);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
        return RET_ERROR;
      }
    }
  }
  return ret;
}

/**
 * @brief Template function for Transform Pass Run method
 *
 * @tparam T Op type (CoreMLOp or NPUOp)
 * @tparam GraphType Graph type (CoreMLGraph or NPUGraph)
 * @tparam Utils Utils type with static methods for pass utilities
 * @param subgraph Pointer to subgraph
 * @param format_depend_nodes Set of format-dependent node types
 * @param name Pass name
 * @return int RET_OK on success, error code otherwise
 */
template <typename T, typename GraphType, typename Utils>
int TransformPassRun(GraphType *subgraph, const std::set<mindspore::schema::PrimitiveType> &format_depend_nodes,
                     const std::string &name) {
  auto all_ops = subgraph->GetOps();
  auto all_tensors = subgraph->GetInsertTensors();
  std::vector<T *> insert_ops;
  int total = 0;

  for (int j = 0; j < lite::REPEAT_TIMES2; ++j) {
    for (size_t i = 0; i < all_ops->size(); i++) {
      auto op = (*all_ops)[i];
      auto insert_state = GetInsertState<T, Utils>(op, subgraph->outputs(), format_depend_nodes);
      insert_ops.clear();

      // If the every output op is nhwc2nchw, insert
      // modify loop index add post_ops.size() to the next op in the origin vector
      switch (insert_state) {
        case lite::InsertState::PreInsert: {
          auto ret = InsertPreNodes<T, Utils>(op, &insert_ops, subgraph->outputs(), all_tensors, name, total);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops->insert(all_ops->begin() + i, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case lite::InsertState::PostInsert: {
          auto ret = InsertPostNodes<T, Utils>(op, &insert_ops, subgraph->outputs(), all_tensors, name, total);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops->insert(all_ops->begin() + i + 1, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        case lite::InsertState::BothInsert: {
          auto ret = InsertPreNodes<T, Utils>(op, &insert_ops, subgraph->outputs(), all_tensors, name, total);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op before op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops->insert(all_ops->begin() + i, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();

          insert_ops.clear();
          ret = InsertPostNodes<T, Utils>(op, &insert_ops, subgraph->outputs(), all_tensors, name, total);
          if (ret != RET_OK) {
            MS_LOG(ERROR) << "Insert nhwc2nchw op and nchw2nhwc op after op " << op->name() << " failed.";
            return RET_ERROR;
          }
          all_ops->insert(all_ops->begin() + i + 1, insert_ops.begin(), insert_ops.end());
          i += insert_ops.size();
          break;
        }
        default:
          MS_LOG(DEBUG) << "Insert Nothing on op " << op->name();
      }
    }
  }
  return RET_OK;
}

}  // namespace delegate
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_LITERT_DELEGATE_FUSION_PASS_UTILS_H_
