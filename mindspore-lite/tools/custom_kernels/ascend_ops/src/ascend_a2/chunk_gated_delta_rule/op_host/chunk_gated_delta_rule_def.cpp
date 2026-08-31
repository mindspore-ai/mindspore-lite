/** * copy from https://gitcode.com/cann/ops-transformer/tree/master/attention/chunk_gated_delta_rule
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

/*!
 * \file chunk_gated_delta_rule_def.cpp
 * \brief Op prototype (op-def) for ChunkGatedDeltaRule — interface matches the open-source
 *        ops-transformer spec (TND layout, same input/output names, order, dtypes). Trimmed to
 *        the ascend910b (910B3) AICore config only. The 7 flexible tensors (query/key/value/beta/
 *        initial_state/out/final_state) accept BOTH BF16 and FP16 via DataTypeList/FormatList
 *        (the upstream multi-dtype API); `g` is FLOAT (optional), `actual_seq_lengths` INT32.
 *        Note: DataTypeList + DynamicFormatFlag(true) makes opbuild emit one kernel binary per
 *        dtype combination (2^7 = 128); tiling rejects mixed combos at runtime, so only the
 *        all-BF16 and all-FP16 binaries are ever selected. (Earlier note that 2-element lists
 *        tripped "Inconsistent format/data type size" was from using the single-dtype `DataType`
 *        method with a multi-element list; the correct `DataTypeList` API builds cleanly.)
 *        FP32 state remains ascend950-only (tiling rejects it on 910b).
 *        NO UnknownShapeFormat is declared: with DataTypeList the per-tensor dtype/format lists
 *        expand to 128 (2^7 cross-product) but UnknownShapeFormat is recorded as a literal (it
 *        does NOT expand), so any finite list mismatches dtype len and trips the converter's
 *        InitUnknownFormatAndDtype ("Inconsistent format size [N] and data type size [128]").
 *        Omitting it makes the converter take the "unknownshape_format not found -> use format
 *        list" fallback (SUCCESS) — the same pattern as built-in DynamicRNNV2 on 910b.
 *
 * Interface (TND layout):
 *   query            BF16  (T, Nk, Dk)
 *   key              BF16  (T, Nk, Dk)
 *   value            BF16  (T, Nv, Dv)
 *   beta             BF16  (T, Nv)
 *   initial_state    BF16  (B, Nv, Dv, Dk)
 *   actual_seq_lengths INT32 (B,)
 *   g (optional)     FLOAT (T, Nv)
 *   out              BF16  (T, Nv, Dv)
 *   final_state      BF16  (B, Nv, Dv, Dk)
 *   scale_value      FLOAT attr, default 1.0
 */

#include "register/op_def_registry.h"

namespace ops {
class ChunkGatedDeltaRule : public OpDef {
 public:
  explicit ChunkGatedDeltaRule(const char *name) : OpDef(name) {
    this->Input("query")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("key")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("value")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("beta")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("initial_state")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("actual_seq_lengths").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
    this->Input("g").ParamType(OPTIONAL).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

    this->Output("out")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
      .AutoContiguous();
    this->Output("final_state")
      .ParamType(REQUIRED)
      .DataTypeList({ge::DT_BF16, ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(false)
      .ExtendCfgInfo("softsync.flag", "true");
    // 910B3 target only (ascend910b compute unit). Upstream also registers ascend910_93
    // and ascend950; both dropped here per "910B3 only".
    this->AICore().AddConfig("ascend910b", aicoreConfig);
  }
};
OP_ADD(ChunkGatedDeltaRule);  // register operator info library
}  // namespace ops
