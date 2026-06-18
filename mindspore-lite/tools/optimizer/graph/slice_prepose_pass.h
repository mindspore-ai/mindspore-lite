/**
 * Copyright 2020-2026 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "tools/converter/ms_depend/pass.h"
#include "include/errorcode.h"
#include "ir/manager.h"
#include "include/registry/converter_context.h"
#include "mindspore/ops/op_def/lite_ops.h"

using mindspore::converter::FmkType;
namespace mindspore::opt {
using lite::RET_ERROR;
using lite::RET_OK;
using lite::STATUS;
using TransactionPtr = std::shared_ptr<mindspore::FuncGraphTransaction>;
using NodeUsedListPtr = std::shared_ptr<std::vector<std::pair<AnfNodePtr, int>>>;

struct SliceReshapeInfo {
  const std::vector<int64_t> &shape_in;
  const std::vector<int64_t> &shape_out;
  int64_t abnormal_axe_in = 0;
  int64_t abnormal_index_out = 0;
  bool slice_at_front = false;
};

struct AbnormalSliceParams {
  int64_t count_sliced_axe_in = 0;
  int64_t count_sliced_axe_front = 0;
  int64_t count_sliced_axe_rear = 0;
  int64_t count_sliced_abnormal_axe = 0;
  int64_t outer_size_in = 1;
  int64_t outer_size_out = 1;
  int64_t abnormal_axe_out = 0;
  bool slice_at_front = false;
};

class SlicePreposePass : public Pass {
 public:
  SlicePreposePass() : Pass("slice_prepose_pass") {}
  ~SlicePreposePass() override = default;
  bool Run(const FuncGraphPtr &graph) override;
  void SetFmkType(FmkType fmkType) { this->fmk_type = fmkType; }

 private:
  static void ClearCNodeAbstractValue(const CNodePtr &cnode);
  static STATUS SwapSliceWithPreceed(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                     const CNodePtr &preceed_cnode, int index, const TransactionPtr &tr = nullptr);
  static ValueNodePtr CreateSliceValueNode(const std::vector<int64_t> &axes);
  static ValueNodePtr CopySliceValueNode(const CNodePtr &slice_cnode);
  static CNodePtr InsertSlice(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &inputs,
                              const CNodePtr &preceed_cnode, int index, const TransactionPtr &tr);
  static STATUS VerifySliceAttrs(const CNodePtr &slice_cnode, int dim = -1);
  static STATUS ApplySliceOnNonBroadcastAxes(const std::vector<int64_t> &origin_axes,
                                             const std::vector<int> &origin_begin, const std::vector<int> &origin_size,
                                             const std::vector<int64_t> &ref_shape, std::vector<int> *begin,
                                             std::vector<int> *size);
  static STATUS SliceParamDeBroadcast(const CNodePtr &slice_cnode, const std::vector<int64_t> &ref_shape,
                                      std::vector<int64_t> *axes, std::vector<int> *begin, std::vector<int> *size);
  static CNodePtr CreateReshapeCNode(const FuncGraphPtr &graph, const std::vector<int64_t> &shape,
                                     const AbstractBasePtr &abstract, const CNodePtr &preceed_cnode);
  static bool SiblingsAreSameSlice(const NodeUsedListPtr &output_node_list, const std::vector<int64_t> &ref_shape = {});
  static int64_t GetReshapeAbnormalAxeIn(const std::vector<int64_t> &shape_in, const std::vector<int64_t> &shape_out,
                                         std::vector<int64_t> *mapped_axe);
  static int64_t GetReshapeAbnormalIndexOut(const CNodePtr &slice_cnode, const std::vector<int64_t> &mapped_axe,
                                            const std::vector<int64_t> &shape_out, std::vector<int64_t> *shape_out_copy,
                                            bool *is_normal_mode, bool *support_abnormal_mode);
  static bool UpdateReshapeShapeParam(const FuncGraphPtr &graph, const CNodePtr &reshape_cnode,
                                      const std::vector<int64_t> &shape_out_copy);
  static bool PreposeWithNormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                       const CNodePtr &reshape_cnode, const std::vector<int64_t> &shape_in,
                                       const std::vector<int64_t> &shape_out_copy,
                                       const std::vector<int64_t> &mapped_axe);
  static CNodePtr CreateSlice1ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                const CNodePtr &matmul_cnode, int64_t count_sliced_axe_in,
                                                const SliceReshapeInfo &info);
  static CNodePtr CreateSlice2ForReshapePrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                                const CNodePtr &new_reshape1_cnode,
                                                const std::vector<int64_t> &new_shape1, int64_t count_sliced2,
                                                const SliceReshapeInfo &info);
  static int64_t CalcPartialProduct(const std::vector<int64_t> &shape, int start, int end);
  static bool CalcAbnormalSliceParams(const CNodePtr &slice_cnode, const SliceReshapeInfo &info,
                                      struct AbnormalSliceParams *params);
  static bool PreposeWithAbnormalReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const CNodePtr &matmul_cnode, const SliceReshapeInfo &info);
  static bool GetArithmeticInputInfo(const CNodePtr &arithmetic_cnode, std::vector<AnfNodePtr> *inputs,
                                     std::vector<std::vector<int64_t>> *shapes, std::vector<bool> *is_default_params);
  static bool IsSoftmaxAxisNotSliced(const CNodePtr &slice_cnode, const std::vector<int64_t> &softmax_axis,
                                     const std::vector<int64_t> &shape);
  static bool ValidateReshapeShapes(const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode,
                                    std::vector<int64_t> *shape_in, std::vector<int64_t> *shape_out);
  static CNodePtr GetMatmulBeforeReshape(const FuncGraphPtr &graph, const CNodePtr &reshape_cnode);
  static bool InsertMatmulSlice(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &matmul_cnode,
                                int input_index, const std::vector<int64_t> &axes, std::vector<int> *begin,
                                std::vector<int> *size, int skip_dim, const TransactionPtr &tr);
  static void DetermineMatmulPreposeDirections(const std::vector<int64_t> &axes, const std::vector<int> &begin,
                                               const std::vector<int> &size, const std::vector<int64_t> &matmul_shape,
                                               int dims, bool *prepose_to_left, bool *prepose_to_right);
  static bool MapFcOutputAxesToInput(const std::vector<int64_t> &shape_in, const std::vector<int64_t> &shape_out,
                                     std::vector<int64_t> *mapped_axe);
  static bool ValidateFcSliceAxes(const CNodePtr &slice_cnode, const std::vector<int64_t> &shape_out);
  static bool BuildFcSliceParams(const CNodePtr &slice_cnode, const std::vector<int64_t> &shape_in,
                                 const std::vector<int64_t> &shape_out, std::vector<int64_t> *new_axes,
                                 std::vector<int> *new_begin, std::vector<int> *new_size);
  static int ProcessArithmeticInput(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                    const CNodePtr &arithmetic_cnode, size_t input_index,
                                    const std::vector<AnfNodePtr> &inputs,
                                    const std::vector<std::vector<int64_t>> &shapes, const TransactionPtr &tr);
  static bool InsertArithmeticDebroadcastSlice(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                               const CNodePtr &arithmetic_cnode, size_t input_index,
                                               const std::vector<int64_t> &new_axes, const std::vector<int> &new_begin,
                                               const std::vector<int> &new_size, const TransactionPtr &tr);
  static void MergeSliceAxesParams(const std::vector<int64_t> &axes_slice1, const std::vector<int> &begin_slice1,
                                   const std::vector<int> &size_slice1, const std::vector<int64_t> &axes_slice2,
                                   const std::vector<int> &begin_slice2, const std::vector<int> &size_slice2,
                                   int64_t axe_max, std::vector<int> *begin_new, std::vector<int> *size_new);
  static bool UpdateMergedSliceParams(const FuncGraphPtr &graph, const CNodePtr &slice2_cnode,
                                      const std::vector<int64_t> &axes_new, const std::vector<int> &begin_new,
                                      const std::vector<int> &size_new);
  static bool ProcessSliceNode(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, bool *oom_fatal);

  static bool DoPrepose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &preceed_cnode);
  static bool RemapSliceAxesByPerm(const std::vector<int> &perm, const std::vector<int64_t> &old_axes,
                                   const std::vector<int> &old_begin, const std::vector<int> &old_size,
                                   std::vector<int> *slice_begin, std::vector<int> *slice_size);
  static AnfNodePtr BuildSliceParamNodes(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                         const std::vector<int> &slice_begin, const std::vector<int> &slice_size);

  static bool PreposeWithSoftmax(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &softmax_cnode);
  static bool PreposeWithReshape(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &reshape_cnode);
  static bool PreposeWithMatmul(const FuncGraphPtr &graph, const CNodePtr &slice_cnode, const CNodePtr &matmul_cnode);
  static bool PreposeWithFullConnection(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                        const CNodePtr &fc_cnode);
  static bool PreposeWithTranspose(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                   const CNodePtr &transpose_cnode);
  static bool PreposeWithArithmetic(const FuncGraphPtr &graph, const CNodePtr &slice_cnode,
                                    const CNodePtr &arithmetic_cnode);
  static bool MergeSequentialSlice(const FuncGraphPtr &graph, const CNodePtr &slice1_cnode,
                                   const CNodePtr &slice2_cnode);
  static bool MergeParallelSlice(const FuncGraphPtr &graph, const NodeUsedListPtr &slices);

 private:
  FmkType fmk_type = converter::kFmkTypeOnnx;
};
}  // namespace mindspore::opt

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SLICE_PREPOSE_PASS_H_
