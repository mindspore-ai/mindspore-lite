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
 * \file chunk_gated_delta_rule_tiling.h
 * \brief Host tiling for ChunkGatedDeltaRule (ascend910b / arch22).
 *
 * Ported from ops-transformer. The upstream ChunkGatedDeltaRuleTiling inherits
 * Ops::Transformer::OpTiling::TilingBaseClass (a framework class from the ops-transformer
 * common module that is NOT part of the CANN toolkit). This repo builds against the bare CANN
 * "customize" harness, so the tiling is de-frameworked into a plain class:
 *   - the 7 virtual steps (IsCapable/GetPlatformInfo/.../PostTiling) become plain methods,
 *     with their bodies kept verbatim from upstream (same shape/dtype validation, workspace
 *     sizing, matmul tiling, tiling-key logic);
 *   - the TilingBaseClass::DoTiling() orchestrator that calls them in order is inlined as
 *     ChunkGatedDeltaRuleTiling::DoTiling();
 *   - context_ / workspaceSize_ / tilingKey_ (formerly inherited) become members;
 *   - registration uses the standalone IMPL_OP_OPTILING(...).Tiling(func, sizeof(struct))
 *     instead of REGISTER_OPS_TILING_TEMPLATE + the TilingRegistry.
 * All tiling *logic* is identical to the open-source op.
 */

#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__

#include <tiling/tiling_api.h>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "../op_kernel/chunk_gated_delta_rule_tiling_data.h"

namespace optiling {

struct ChunkGatedDeltaRuleCompileInfo {
  uint64_t aivNum{0UL};
  uint64_t aicNum{0UL};
};

class ChunkGatedDeltaRuleTiling {
 public:
  explicit ChunkGatedDeltaRuleTiling(gert::TilingContext *context) : context_(context) { InitCompileInfo(); }
  ~ChunkGatedDeltaRuleTiling() = default;

  // Inlined from Ops::Transformer::OpTiling::TilingBaseClass::DoTiling (the upstream
  // framework orchestrator). Runs the steps in order and sets the tiling key.
  // NOTE: upstream's IsCapable() step is dropped — this repo targets ascend910b only, so
  // platform-capability negotiation is unnecessary (IsCapable always returned true).
  ge::graphStatus DoTiling() {
    auto ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = DoLibApiTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = GetWorkspaceSize();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
      return ret;
    }
    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
  }

 protected:
  // 1. Get platform info such as CoreNum, UB/L1/L0C resource sizes (compileInfo_ cached in ctor)
  ge::graphStatus GetPlatformInfo();

  // 2. Get INPUT/OUTPUT/ATTR info
  ge::graphStatus GetShapeAttrsInfo();

  // 3. Compute data tiling TilingData
  ge::graphStatus DoOpTiling();

  // 4. Compute high-level API TilingData
  ge::graphStatus DoLibApiTiling();

  // 5. Compute TilingKey
  uint64_t GetTilingKey() const;

  // 6. Compute workspace size
  ge::graphStatus GetWorkspaceSize();

  // 7. Save tiling data
  ge::graphStatus PostTiling();

 protected:
  void InitCompileInfo();

  ge::graphStatus CheckContext();
  ge::graphStatus AnalyzeDtype();
  // AnalyzeDtype split into three dtype-category checks (same order, functionally equivalent).
  ge::graphStatus CheckLowDtype();
  ge::graphStatus CheckStateDtype();
  ge::graphStatus CheckAuxDtype();
  ge::graphStatus AnalyzeShapes();
  ge::graphStatus CheckDerivedDimConstraints();
  ge::graphStatus GetScale();
  ge::graphStatus GetStrides();
  ge::graphStatus GetOptionalInput();
  ge::graphStatus AnalyzeFormat();
  ge::graphStatus DoMatmulTiling();
  ge::graphStatus SetScheduleConfig();
  ge::graphStatus CheckExpectedShapes(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                      const gert::Shape &valueShape, const gert::Shape &betaShape,
                                      const gert::Shape &stateShape, const gert::Shape &actualSeqLengthsShape,
                                      const gert::Shape &outShape, const gert::Shape &finalStateShape,
                                      const gert::Shape *gShape);
  bool CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc);
  bool CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name);

  gert::TilingContext *context_ = nullptr;
  uint64_t workspaceSize_{0UL};
  uint64_t tilingKey_{0UL};

  ChunkGatedDeltaRuleCompileInfo compileInfo_;
  ChunkGatedDeltaRuleTilingData tilingData_;
  platform_ascendc::SocVersion socVersion_;
};

}  // namespace optiling
#endif  // __OP_HOST_CHUNK_GATED_DELTA_RULE_TILING_H__
