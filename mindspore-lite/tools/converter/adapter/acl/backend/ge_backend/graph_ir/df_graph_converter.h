/**
 * modified from
 * https://gitcode.com/mindspore/mindspore/blob/master/mindspore/ccsrc/backend/ge_backend/graph_ir/convert.h
 *
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_DF_GRAPH_CONVERT_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_DF_GRAPH_CONVERT_H_

#include <cstdlib>
#include <memory>
#include <map>
#include <set>
#include <unordered_set>
#include <vector>
#include <string>
#include <utility>
#include <stack>
#include <fstream>
#include <sstream>

#include "utils/config_manager.h"
#include "primitive/structure_ops.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "utils/phase.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "tools/converter/adapter/acl/backend/ge_backend/graph_ir/df_graph_manager.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_desc.h"
#include "graph/operator_reg.h"
#include "ge/ge_api.h"
#include "include/backend/visible.h"
#include "include/utils/utils.h"

namespace mindspore::lite::backend::ge_backend {
enum Status : int { SUCCESS = 0, FAILED, INVALID_ARGUMENT, ALREADY_EXISTS, NOT_FOUND };

using OpAdapterPtr = device::ascend::OpAdapterPtr;
using OutHandler = device::ascend::OutHandler;
using ParamIndexMap = std::map<std::size_t, std::size_t>;
using MeTensor = mindspore::tensor::Tensor;
using MeTensorPtr = std::shared_ptr<MeTensor>;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using GeTensorDesc = ::ge::TensorDesc;
using AnfGraph = FuncGraph;
using AnfGraphPtr = std::shared_ptr<FuncGraph>;
using Operator = ::ge::Operator;
using OperatorPtr = std::shared_ptr<::ge::Operator>;
using DfGraph = ::ge::Graph;
using DfGraphPtr = std::shared_ptr<DfGraph>;
using TensorOrderMap = std::map<std::string, std::shared_ptr<tensor::Tensor>>;
enum class GraphType { kNormal, kCond, kBody, kAfter, kBranch };
enum class RefModeFlag {
  kRefModeNone,
  kRefModeVariable,  // Only Variables will be treated as RefData
  kRefModeAll,       // All Parameter including Variables and Constants will be treated as RefData
  kRefModeEnv        // depend on REF_MODE, default value is on, ref mode type will be kRefModeAll
};
constexpr char kGraphFlagHasGetNext[] = "graph_has_getnext";
constexpr auto kNoNeedAllocDeviceAddress = "no_need_alloc_device_address";

using SetDynRefDataFunc = std::function<ShapeVector(const AnfNodePtr &, const ShapeVector &)>;

class BACKEND_EXPORT DfGraphConvertor {
 public:
  explicit DfGraphConvertor(const AnfGraphPtr &anf_graph, const std::string &phase_prefix,
                            RefModeFlag ref_mode_type = RefModeFlag::kRefModeEnv,
                            const std::vector<std::string> &extra_variables_names = {},
                            SetDynRefDataFunc dyn_ref_data_func = nullptr, bool offline_convert = false)
      : anf_graph_(anf_graph), extra_variables_names_(extra_variables_names), offline_convert_(offline_convert) {
    MS_EXCEPTION_IF_NULL(anf_graph);
    if (ref_mode_type == RefModeFlag::kRefModeEnv) {
      ref_mode_type_ = IsTwoPhaseInfer() ? RefModeFlag::kRefModeVariable : RefModeFlag::kRefModeAll;
    } else {
      ref_mode_type_ = ref_mode_type;
    }

    if (anf_graph->has_flag("broadcast_flag")) {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::DISTRIBUTION);
    } else {
      ConfigManager::GetInstance().set_parallel_strategy(ParallelStrategy::ONE_DEVICE);
    }
    df_graph_ = std::make_shared<DfGraph>(anf_graph_->ToString());

    std::string graph_name = anf_graph_->ToString();
    graph_manager_ = Manage(anf_graph_, true);
    MS_EXCEPTION_IF_NULL(graph_manager_);
    MS_LOG(INFO) << "Create DfGraphConvertor with graph: " << graph_name << ", dynamic input: " << dynamic_shape_inputs_
                 << ", ref_mod_type_: " << ref_mode_type_;
  }

  ~DfGraphConvertor() {}

  static void RegisterAdapter(const std::string &name, OpAdapterPtr adpt);
  static void RegisterAdapter(const std::string &name, OpAdapterPtr train_adpt, OpAdapterPtr infer_adpt);

  void DrawComputeGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << compute_sout_.str();
    fout.close();
  }

  void DrawInitGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << init_sout_.str();
    fout.close();
  }

  void DrawSaveCheckpointGraph(const std::string &name) {
    std::ofstream fout(name);
    if (!fout.is_open()) {
      MS_LOG(ERROR) << "Open file '" << name << "' failed!";
      return;
    }
    fout << checkpoint_sout_.str();
    fout.close();
  }

  DfGraphConvertor &ConvertAllNode();
  DfGraphConvertor &BuildGraph(const std::string &name);
  DfGraphConvertor &InitParam(const TensorOrderMap &tensors);
  void InitParamWithData(const TensorOrderMap &tensors);
  bool NodeInputKeepUpdate(const FuncGraphManagerPtr &manager, const AnfNodePtr &node);
  OutHandler GetNormalOpInput(const AnfNodePtr &node, const AnfNodePtr &pred);
  void DrawOpInput(const AnfNodePtr &node, const AnfNodePtr &pred, size_t i);
  void SetOpInput(const OpAdapterPtr &adpt, const CNodePtr &node);
  void SetOpAttrToInput(const OpAdapterPtr &adpt, const CNodePtr &node);
  void SetupParamInitSubGraph(const TensorOrderMap &tensors, const std::vector<::ge::Operator> *init_input,
                              bool is_sink_size_repeat);
  void DrawParamInitSubGraph(const std::string &name, const AnfNodePtr &it);

  DfGraphPtr GetComputeGraph();
  DfGraphPtr GetInitGraph();
  std::vector<std::string> GetInitDataNames() const { return init_data_names_; }
  std::vector<std::string> GetRefDataNames() const { return ref_data_names_; }
  int ErrCode() const { return static_cast<int>(error_); }

  bool dynamic_shape_inputs() const { return dynamic_shape_inputs_; }
  std::vector<ShapeVector> input_shapes() { return input_shapes_; }

 private:
  OperatorPtr Convert(AnfNodePtr node);
  OperatorPtr ConvertCNode(CNodePtr node);
  OperatorPtr ConvertParameter(AnfNodePtr node);
  void SetNodeAbstract(const CNodePtr &node) const;
  Status TryConvertValueNodeToMultiConst(const ValueNodePtr node);
  OperatorPtr ConvertValueNode(ValueNodePtr node);
  void SaveParamFormat(CNodePtr node);

  // Helper functions for SaveParamFormat
  std::string ExtractFormatFromOpDef(const PrimitivePtr &prim, const CNodePtr &node);
  std::string ExtractFormatFromAttr(const PrimitivePtr &prim);

  void ConvertTopK(const CNodePtr &node);
  AnfNodePtr CreateCast(const AnfNodePtr &input, const TypePtr &dst_type) const;
  void ConvertReshape(const CNodePtr &node);
  void ConvertHcomFusionId(const CNodePtr &node);
  void ConvertHcclNode(const CNodePtr &node);
  void ConvertAlltoAllVGE(const CNodePtr &node);
  void ConvertUniformReal(const CNodePtr &node);
  void AddCommAttrForHcclNode(const CNodePtr &node, const OperatorPtr &converted_op) const;
  void ConvertOCRRecPreHandle(const CNodePtr &node);
  void ConvertConv2D(const CNodePtr &node);
  void ConvertDynamicStitch(const CNodePtr &node);
  void ConvertParallelGroupToHcom(const CNodePtr &node);
  void ConvertParallelGroupIdToHcom(const CNodePtr &node);
  std::vector<int64_t> CastToInt(const ValuePtr &value) const;
  void TransDataType(const FuncGraphPtr &anf_graph) const;
  void TransInputDataType(const CNodePtr &node, const std::string &node_name) const;
  void TransAttrDataType(const CNodePtr &node, const std::string &node_name) const;
  bool CheckCNode(const std::string &name, const CNodePtr node);
  void SetNodeInput(AnfNodePtr node);
  void UpdateOpDesc(AnfNodePtr node);
  void DrawCNode(const CNodePtr node, const OpAdapterPtr adpt);
  void UpdateDataOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const;
  void UpdateConstOpDesc(const AnfNodePtr &it, const OperatorPtr &op) const;
  void AddGraphConstInput(const OperatorPtr &op);
  AnfNodePtr ParseLoadInput(const CNodePtr &cnode) const;
  void SetGraphInputs(std::vector<Operator> *inputs);

  // Helper functions for SetGraphInputs
  OperatorPtr FindGetNextInput(const std::vector<PrimitivePtr> &input_prims);
  void ProcessParameterInput(const AnfNodePtr &it, int *index, std::vector<Operator> *inputs);
  void ProcessVarInput(const AnfNodePtr &it, const std::string &name, std::vector<Operator> *inputs);
  void CollectParameterShape(const ParameterPtr &param);
  void TransformConstOp(const CNodePtr &node, const AnfNodePtr &pred);
  void ProcessInputData(std::vector<Operator> *init_input,
                        std::unordered_set<std::string> *infer_need_update_parameter_names, const OperatorPtr &param_op,
                        const string &name, const std::shared_ptr<GeTensorDesc> &desc);
  AnfNodePtr GetRealInputNode(const CNodePtr &node, const AnfNodePtr &input);

  bool IsDataInput(const AnfNodePtr &node, const AnfNodePtr &input, size_t input_index);
  void SetMakeTupleInput(const OpAdapterPtr &adpt, const CNodePtr &make_tuple_node);
  void SetDynamicInputHandleByMultiInput(const OpAdapterPtr &adpt, const CNodePtr &node,
                                         const CNodePtr &from_node_input);
  void SetNodeControlInput(const AnfNodePtr &node, const AnfNodePtr &input);
  void SetGraphOutputs(bool is_main_graph = false);
  std::vector<OutHandler> GetInputHandles(const AnfNodePtr &node, const AnfNodePtr &input);
  void FillEmptyInputsWithNoInputOp(std::vector<Operator> *);
  bool IsDynamicInputBeforeNormalInput(const OpAdapterPtr &adpt, int *ge_input_size,
                                       mindspore::HashMap<int, int> *ge_input_to_ms_input);
  void SetDynamicInputBeforeNormalInput(const OpAdapterPtr &adpt, const CNodePtr &node,
                                        const std::vector<AnfNodePtr> &inputs, const int &ge_input_size,
                                        const mindspore::HashMap<int, int> &ge_input_to_ms_input,
                                        std::vector<int64_t> *dyn_input_sizes);

  // Identity Optimization
  void IdentityOptimization();
  std::string GetGNodeName(const ::ge::GNode &node) const;
  std::string GetGNodeType(const ::ge::GNode &node) const;
  bool IsIdentityRedundant(const ::ge::GNode &node) const;
  bool IsIdentityInUpdateGraph(const ::ge::GNode &node) const;
  void RemoveIdentity(::ge::GNode identity_node);
  void NoOpOptimization();
  bool IsNoOpRedundant(const ::ge::GNode &node) const;
  void RemoveNoOp(::ge::GNode noop);
  void BuildInitDataGraph(const std::string &name);
  bool IsConstantOp(const OperatorPtr &op) const;

  std::ostringstream compute_sout_;
  std::ostringstream init_sout_;
  std::ostringstream checkpoint_sout_;
  std::ostringstream restore_checkpoint_sout_;
  mindspore::HashMap<AnfNode *, std::string> op_draw_name_;
  std::map<std::string, std::string> param_format_;

  std::shared_ptr<AnfGraph> anf_graph_{nullptr};
  FuncGraphManagerPtr graph_manager_{nullptr};
  RefModeFlag ref_mode_type_ = RefModeFlag::kRefModeNone;
  std::vector<std::string> extra_variables_names_;
  std::vector<std::string> ref_data_names_;
  std::set<std::string> unsupported_ops_names_;

  std::shared_ptr<DfGraph> df_graph_{nullptr};
  std::shared_ptr<DfGraph> init_graph_{nullptr};
  mindspore::HashMap<AnfNode *, OperatorPtr> op_cache_;
  /* record "getnext"<->"out_handler" mapping */
  mindspore::HashMap<AnfNode *, OutHandler> out_handle_cache_;
  /* record "value tuple"<->"out_handler vector" mapping */
  mindspore::HashMap<AnfNode *, std::shared_ptr<std::vector<OutHandler>>> tuple_out_handle_cache_;
  mindspore::HashMap<std::string, AnfNodePtr> params_;
  mindspore::HashMap<std::string, OperatorPtr> vars_;
  std::vector<std::pair<::ge::Operator, std::string>> graph_outputs_;
  std::vector<OperatorPtr> graph_const_inputs_;
  std::vector<OperatorPtr> init_ops_;
  std::vector<std::string> init_data_names_;
  ShapeArray input_shapes_;
  Status error_ = SUCCESS;
  bool dynamic_shape_inputs_ = false;
  mindspore::HashMap<OperatorPtr, std::shared_ptr<tensor::Tensor>> const_op_to_value_;
  bool offline_convert_ = false;
};
}  // namespace mindspore::lite::backend::ge_backend

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_ACL_BACKEND_GE_BACKEND_GRAPH_IR_DF_GRAPH_CONVERT_H_
