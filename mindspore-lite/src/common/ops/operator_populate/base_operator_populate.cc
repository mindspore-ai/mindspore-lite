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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl_c/where_parameter.h"
#include "nnacl_c/sparse_to_dense_parameter.h"
#include "nnacl_c/transpose_parameter.h"
#include "nnacl_c/triu_tril_parameter.h"
#include "nnacl_c/fp32/unique_fp32.h"
#include "nnacl_c/scatter_nd_parameter.h"
#include "nnacl_c/op_base.h"
#include "nnacl_c/gather_parameter.h"
#include "nnacl_c/gather_nd_parameter.h"
#include "nnacl_c/reshape_parameter.h"
#include "mindspore/ops/op_def/auto_generate/gen_lite_ops.h"
#include "infer/adam.h"
#include "infer/assert.h"
#include "infer/where.h"
#include "infer/unique.h"
#include "infer/ops_func_impl/triu.h"
#include "infer/tril.h"
#include "infer/sparse_to_dense.h"
#include "infer/sparse_segment_sum.h"
#include "infer/sparse_reshape.h"
#include "infer/sparse_fill_empty_rows.h"
#include "infer/size.h"
#include "infer/scatter_nd_update.h"
#include "infer/tensor_scatter_add.h"
#include "infer/ragged_range.h"
#include "infer/ops_func_impl/isfinite.h"
#include "infer/invert_permutation.h"
#include "infer/ops_func_impl/gather.h"
#include "infer/fill.h"
#include "infer/switch.h"
#include "infer/tensor_array_read.h"
#include "infer/tensor_array_write.h"
#include "infer/custom_extract_features.h"
#include "infer/custom_normalize.h"
#include "infer/hashtable_lookup.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_n.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_w.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace lite {
REG_OP_BASE_POPULATE(Adam)
REG_OP_BASE_POPULATE(Assert)
REG_OP_BASE_POPULATE(Assign)
REG_OP_BASE_POPULATE(AssignAdd)
REG_OP_BASE_POPULATE(Cast)
REG_OP_BASE_POPULATE(Erf)
REG_OP_BASE_POPULATE(Fill)
REG_OP_BASE_POPULATE(LinSpace)
REG_OP_BASE_POPULATE(IsFinite)
REG_OP_BASE_POPULATE(InvertPermutation)
REG_OP_BASE_POPULATE(UnsortedSegmentSum)
REG_OP_BASE_POPULATE(SparseSegmentSum)
REG_OP_BASE_POPULATE(SparseReshape)
REG_OP_BASE_POPULATE(SparseFillEmptyRows)
REG_OP_BASE_POPULATE(Size)
REG_OP_BASE_POPULATE(Shape)
REG_OP_BASE_POPULATE(Select)
REG_OP_BASE_POPULATE(ExpandDims)
REG_OP_BASE_POPULATE(Rank)
REG_OP_BASE_POPULATE(OnesLike)
REG_OP_BASE_POPULATE(NonZero)
REG_OP_BASE_POPULATE(Switch)
REG_OP_BASE_POPULATE(TensorArrayRead)
REG_OP_BASE_POPULATE(TensorArrayWrite)
REG_OP_BASE_POPULATE(CustomExtractFeatures)
REG_OP_BASE_POPULATE(CustomNormalize)
REG_OP_BASE_POPULATE(HashtableLookup)
REG_OP_BASE_POPULATE(RaggedRange)

REG_OP_DEFAULT_POPULATE(SparseToDense)
REG_OP_DEFAULT_POPULATE(Transpose)
REG_OP_DEFAULT_POPULATE(Tril)
REG_OP_DEFAULT_POPULATE(Triu)
REG_OP_DEFAULT_POPULATE(Where)
REG_OP_DEFAULT_POPULATE(Unique)
REG_OP_DEFAULT_POPULATE(Reshape)

using mindspore::ops::kNameGather;
using mindspore::ops::kNameGatherD;
using mindspore::ops::kNameGatherNd;
using mindspore::ops::kNameScatterNd;
using mindspore::ops::kNameScatterNdUpdate;
using mindspore::ops::kNameTensorScatterAdd;
using mindspore::schema::PrimitiveType_Gather;
using mindspore::schema::PrimitiveType_GatherD;
using mindspore::schema::PrimitiveType_GatherNd;
using mindspore::schema::PrimitiveType_ScatterNd;
using mindspore::schema::PrimitiveType_ScatterNdUpdate;
using mindspore::schema::PrimitiveType_TensorScatterAdd;
REG_OPERATOR_POPULATE(kNameGather, PrimitiveType_Gather, PopulateOpParameter<GatherParameter>)
REG_OPERATOR_POPULATE(kNameGatherD, PrimitiveType_GatherD, PopulateOpParameter<GatherParameter>)
REG_OPERATOR_POPULATE(kNameGatherNd, PrimitiveType_GatherNd, PopulateOpParameter<GatherNdParameter>)
REG_OPERATOR_POPULATE(kNameScatterNd, PrimitiveType_ScatterNd, PopulateOpParameter<ScatterNDParameter>)
REG_OPERATOR_POPULATE(kNameScatterNdUpdate, PrimitiveType_ScatterNdUpdate, PopulateOpParameter<ScatterNDParameter>)
REG_OPERATOR_POPULATE(kNameTensorScatterAdd, PrimitiveType_TensorScatterAdd, PopulateOpParameter<ScatterNDParameter>)
}  // namespace lite
}  // namespace mindspore
