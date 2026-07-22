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
 * \file chunk_gated_delta_rule_tiling.cpp
 * \brief Host tiling implementation for ChunkGatedDeltaRule (ascend910b / arch22).
 *
 * Ported from ops-transformer. Every tiling-decision method body (shape/dtype/format
 * validation, derived-dim constraints, workspace sizing, matmul tiling, tiling-key, strides,
 * schedule config) is kept verbatim from the upstream op. Only the scaffolding differs from
 * upstream (see chunk_gated_delta_rule_tiling.h): the Ops::Transformer framework base class
 * and REGISTER_OPS_TILING_TEMPLATE/TilingRegistry are replaced by a standalone
 * IMPL_OP_OPTILING(...).Tiling(func, sizeof(struct)) registration, and the upstream error
 * macros (OP_CHECK_IF / OP_CHECK_NULL_WITH_CONTEXT / OP_LOGE / OPS_REPORT_CUBE_INNER_ERR from
 * err/ops_err.h — not shipped in this CANN install) are replaced by plain null-checks +
 * GRAPH_FAILED returns. PostTiling writes the struct via the proven GetTilingData<> + struct
 * copy (no memcpy_s/securec dependency).
 */

#include <string>

#include "chunk_gated_delta_rule_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_gated_delta_rule_tiling_key.h"

namespace optiling {

const size_t QUERY_INDEX = 0;
const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t BETA_INDEX = 3;
const size_t STATE_INDEX = 4;
const size_t ACTUAL_SEQ_LENGTHS_INDEX = 5;
const size_t G_INDEX = 6;

const size_t OUTPUT_OUT_IDX = 0;
const size_t OUTPUT_FINAL_STATE_IDX = 1;

const size_t QKV_DIM_NUM = 3;
const size_t BETA_DIM_NUM = 2;
const size_t STATE_DIM_NUM = 4;
const size_t ACTUAL_SEQ_LENGTHS_DIM_NUM = 1;
const size_t G_DIM_NUM = 2;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

// Fixed system workspace size (16 MB)
constexpr int64_t SYS_WORKSPACE_SIZE = 16777216;

// Matmul tiling related constants
constexpr uint32_t MATMUL_BASE_M = 128;
constexpr uint32_t MATMUL_BASE_K = 128;
constexpr uint32_t MATMUL_BASE_N = 128;

constexpr uint32_t STAGE_ONE_TWO = 2;
constexpr uint32_t STAGE_ONE_THREE = 3;
constexpr uint32_t STAGE_ONE_PARA_NUM = 4;
constexpr uint32_t MASK_NUM = 4;
constexpr int64_t P_NUM = 2;

// Initialize compile info, read platform resources, and cache core counts into tilingData_
void ChunkGatedDeltaRuleTiling::InitCompileInfo() {
  auto platformInfoPtr = context_->GetPlatformInfo();
  if (platformInfoPtr == nullptr) {
    return;
  }
  const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
  compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
  compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
  socVersion_ = ascendcPlatform.GetSocVersion();

  if (compileInfo_.aivNum == 0 || compileInfo_.aicNum == 0) {
    return;
  }
  tilingData_.aiCoreNum = compileInfo_.aicNum;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

// Get input/output info and run context, dtype, shape, attr, optional-input and format checks in order
ge::graphStatus ChunkGatedDeltaRuleTiling::GetShapeAttrsInfo() {
  if (CheckContext() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (GetOptionalInput() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (GetScale() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (AnalyzeDtype() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (AnalyzeShapes() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (GetStrides() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (AnalyzeFormat() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoOpTiling() {
  int64_t c = 64;     // chunk size is 64
  int64_t p = P_NUM;  // max chunks processed per core in a chunk group
  tilingData_.chunkSize = c;
  tilingData_.maxGroupLength = p * tilingData_.aiCoreNum * tilingData_.chunkSize;
  tilingData_.stageOneParaNum = STAGE_ONE_PARA_NUM;  // stage1 parallelism

  tilingData_.interWorkspaceSz = 0;
  int64_t sizeHigh = ge::GetSizeByDataType(ge::DT_FLOAT);
  int64_t sizeLow = ge::GetSizeByDataType(tilingData_.isFp16 ? ge::DT_FLOAT16 : ge::DT_BF16);
  int64_t nv = tilingData_.nv;
  int64_t dv = tilingData_.dv;
  int64_t dk = tilingData_.dk;
  int64_t s = tilingData_.maxGroupLength;
  tilingData_.interWorkspaceSz += sizeHigh * nv * s;      // gCumExp (FP32)
  tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;  // kCumDecay (BF16)
  if (tilingData_.stateIsFp32) {
    tilingData_.interWorkspaceSz += sizeHigh * nv * s * dv;  // vInner (FP32)
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;   // vInnerBf16 (BF16, for stage3)
  } else {
    tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;  // vInner (BF16)
  }
  tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;  // qPrime (BF16)
  tilingData_.interWorkspaceSz += sizeLow * nv * s * dv;  // attnInter (BF16, arch22 compat)
  tilingData_.interWorkspaceSz += sizeLow * nv * s * dk;  // kg (BF16)
  tilingData_.interWorkspaceSz += sizeLow * nv * s * c;   // qkt (BF16)
  if (tilingData_.stateIsFp32) {
    tilingData_.interWorkspaceSz += sizeLow * nv * dv * dk;  // stateBf16Wk (BF16, arch35)
  } else if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
    // arch22: kernel unconditionally advances offset
    tilingData_.interWorkspaceSz += sizeHigh * tilingData_.b * nv * dv * dk;  // highState_
  }
  tilingData_.interWorkspaceSz += sizeHigh * c * c * tilingData_.aiCoreNum * MASK_NUM;  // mask (FP32)

  // stage1 temporary workspace
  tilingData_.stageWorkspaceSz =
    sizeLow * c * (STAGE_ONE_TWO * c + STAGE_ONE_THREE * dk + dv) * tilingData_.stageOneParaNum;
  tilingData_.stageWorkspaceSz *= tilingData_.aiCoreNum;

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoMatmulTiling() {
  // Adaptive Cube tile base (P0-2 perf opt, see PERFORMANCE.md). The op's matmuls are
  // ~chunkSize(64) x Dk x Dv; for small head_dim (Dk,Dv <= 64) they are 64^3-class. A fixed
  // 128 base makes the Cube run a 128-tile for a 64^3 matmul (pads M/N/K 2x -> wastes Cube
  // cycles + L0 bandwidth). Use a 64 base when BOTH Dk,Dv <= 64 (Qwen3.5-0.8B-class small
  // models, both bf16 & fp16); keep 128 otherwise so larger head_dim is unchanged (no
  // regression). 64 is a valid Cube L0 tile size; dtype-agnostic (base applies to bf16 & fp16).
  constexpr uint32_t SMALL_BASE = 64;
  uint32_t baseDim = (tilingData_.dk <= SMALL_BASE && tilingData_.dv <= SMALL_BASE)
                       ? SMALL_BASE
                       : MATMUL_BASE_M;  // MATMUL_BASE_M == _K == _N == 128
  uint32_t baseM = baseDim;
  uint32_t baseK = baseDim;
  uint32_t baseN = baseDim;

  // ========== MT_FP32: low(BF16|FP16) -> low -> low (arch22 Cube GEMM) ==========
  matmul_tiling::MultiCoreMatmulTiling mm_;
  const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
  uint64_t ubSize;
  uint64_t l1Size;
  uint64_t l0CSize;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
  mm_.SetBufferSpace(l1Size, l0CSize, ubSize);
  auto lowMmDtype = tilingData_.isFp16 ? matmul_tiling::DataType::DT_FLOAT16 : matmul_tiling::DataType::DT_BFLOAT16;
  mm_.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, lowMmDtype, true);
  mm_.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, lowMmDtype, true);
  mm_.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, lowMmDtype);
  mm_.SetBias(false);
  mm_.SetDim(1);
  mm_.SetShape(baseM, baseN, baseK);
  mm_.SetOrgShape(baseM, baseN, baseK);
  mm_.SetFixSplit(baseM, baseN, baseK);
  if (mm_.GetTiling(tilingData_.matmulTilingFp32) == -1) {
    return ge::GRAPH_FAILED;
  }
  tilingData_.matmulTilingFp32.dbL0C = 1;
  tilingData_.matmulTilingFp32.stepKa = 1;
  tilingData_.matmulTilingFp32.stepKb = 1;
  tilingData_.matmulTilingFp32.depthA1 = 1;
  tilingData_.matmulTilingFp32.depthB1 = 1;
  tilingData_.matmulTilingFp32.stepM = 1;
  tilingData_.matmulTilingFp32.stepN = 1;

  // ========== MT_FP32C: BF16 -> BF16 -> FP32 (for FP32 state path; ascend950 only) ==========
  if (tilingData_.stateIsFp32) {
    matmul_tiling::MultiCoreMatmulTiling mmFp32C;
    mmFp32C.SetBufferSpace(l1Size, l0CSize, ubSize);
    mmFp32C.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                     true);
    mmFp32C.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                     true);
    mmFp32C.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mmFp32C.SetBias(false);
    mmFp32C.SetDim(1);
    mmFp32C.SetShape(baseM, baseN, baseK);
    mmFp32C.SetOrgShape(baseM, baseN, baseK);
    mmFp32C.SetFixSplit(baseM, baseN, baseK);
    if (mmFp32C.GetTiling(tilingData_.matmulTilingFp32C) == -1) {
      return ge::GRAPH_FAILED;
    }
    tilingData_.matmulTilingFp32C.dbL0C = 1;
    tilingData_.matmulTilingFp32C.stepKa = 1;
    tilingData_.matmulTilingFp32C.stepKb = 1;
    tilingData_.matmulTilingFp32C.depthA1 = 1;
    tilingData_.matmulTilingFp32C.depthB1 = 1;
    tilingData_.matmulTilingFp32C.stepM = 1;
    tilingData_.matmulTilingFp32C.stepN = 1;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::DoLibApiTiling() {
  // bf16 and fp16 share tiling key 0 (they use the SAME tiling struct -- only the isFp16
  // flag differs; the device entry dispatches CGDR<half,float> vs CGDR<bfloat16_t,float> on
  // tilingData.isFp16, NOT on the tiling key). Introducing a separate fp16 tiling key (e.g.
  // 2) is NOT registered in CANN's autogen tiling_struct_expr_map and raises KeyError at
  // convert (gen_static_shape_v2). So keep the original 0/1 scheme; fp16 reuses key 0.
  tilingKey_ = tilingData_.stateIsFp32 ? TILING_KEY_CGDR_FP32_STATE : TILING_KEY_CGDR_BF16_STATE;

  // Run matmul tiling
  if (DoMatmulTiling() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Return tiling key
uint64_t ChunkGatedDeltaRuleTiling::GetTilingKey() const { return tilingKey_; }

// Compute workspace size
ge::graphStatus ChunkGatedDeltaRuleTiling::GetWorkspaceSize() {
  workspaceSize_ = SYS_WORKSPACE_SIZE;
  workspaceSize_ += tilingData_.interWorkspaceSz;
  workspaceSize_ += tilingData_.stageWorkspaceSz;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::SetScheduleConfig() {
  constexpr uint32_t batchMode = 1U;
  auto ret = context_->SetScheduleMode(batchMode);
  if (ret != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Write back tilingData and workspace info
ge::graphStatus ChunkGatedDeltaRuleTiling::PostTiling() {
  context_->SetBlockDim(tilingData_.aiCoreNum);

  auto td = context_->GetTilingData<ChunkGatedDeltaRuleTilingData>();
  if (td == nullptr) {
    return ge::GRAPH_FAILED;
  }
  *td = tilingData_;  // struct copy into the raw tiling buffer (size fixed by registration sizeof)

  size_t *workspaces = context_->GetWorkspaceSizes(1);  // set workspace
  if (workspaces == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspaces[0] = workspaceSize_;

  if (SetScheduleConfig() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Validate the presence of required inputs, optional inputs and outputs in the context
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckContext() {
  if (context_->GetInputShape(QUERY_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(QUERY_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(KEY_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(KEY_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(VALUE_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(VALUE_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(BETA_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(BETA_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(STATE_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(STATE_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetOutputShape(OUTPUT_OUT_IDX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetOutputDesc(OUTPUT_OUT_IDX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  if (context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX) == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
  auto gTensor = context_->GetOptionalInputTensor(G_INDEX);
  auto gShape = context_->GetOptionalInputShape(G_INDEX);
  bool hasDesc = (gDesc != nullptr);
  bool hasTensor = (gTensor != nullptr);
  bool hasShape = (gShape != nullptr);
  if ((hasDesc != hasTensor) || (hasDesc != hasShape)) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Validate input/output dtype: q/k/v/beta/out must all be bf16 or all fp16; state/final_state on 910b
// follows that low dtype (fp32 state is ascend950-only); actual_seq_lengths is int32, optional g is float.
// Split by dtype category into CheckLowDtype/CheckStateDtype/CheckAuxDtype, in the same order as the
// original implementation, functionally equivalent.
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeDtype() {
  if (CheckLowDtype() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (CheckStateDtype() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (CheckAuxDtype() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Validate low dtype: q/k/v/beta/out must all be BF16 or all FP16, and cache isFp16 (used by state check).
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckLowDtype() {
  auto queryDtype = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
  auto keyDtype = context_->GetInputDesc(KEY_INDEX)->GetDataType();
  auto valueDtype = context_->GetInputDesc(VALUE_INDEX)->GetDataType();
  auto betaDtype = context_->GetInputDesc(BETA_INDEX)->GetDataType();
  auto outDtype = context_->GetOutputDesc(OUTPUT_OUT_IDX)->GetDataType();
  // q/k/v/beta/out must all be BF16 or all FP16 (low dtype).
  auto isLowDtype = [](ge::DataType d) { return d == ge::DT_BF16 || d == ge::DT_FLOAT16; };
  if (!isLowDtype(queryDtype) || !isLowDtype(keyDtype) || !isLowDtype(valueDtype) || !isLowDtype(betaDtype) ||
      !isLowDtype(outDtype)) {
    return ge::GRAPH_FAILED;
  }
  if (queryDtype != keyDtype || queryDtype != valueDtype || queryDtype != betaDtype || queryDtype != outDtype) {
    return ge::GRAPH_FAILED;  // reject mixed low-dtype combinations (dead branch produced by DataTypeList)
  }
  tilingData_.isFp16 = (queryDtype == ge::DT_FLOAT16) ? 1 : 0;
  return ge::GRAPH_SUCCESS;
}

// Validate state dtype: state/final_state must match; FP32 state is ascend950-only, follows low dtype on 910b.
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckStateDtype() {
  auto stateDtype = context_->GetInputDesc(STATE_INDEX)->GetDataType();
  auto finalStateDtype = context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX)->GetDataType();
  if (stateDtype != ge::DT_BF16 && stateDtype != ge::DT_FLOAT && stateDtype != ge::DT_FLOAT16) {
    return ge::GRAPH_FAILED;
  }
  // FP32 state is supported only on ascend950; on 910b state must follow the low dtype.
  if (stateDtype == ge::DT_FLOAT && socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
    return ge::GRAPH_FAILED;
  }
  if (socVersion_ != platform_ascendc::SocVersion::ASCEND950) {
    auto expectedStateDtype = tilingData_.isFp16 ? ge::DT_FLOAT16 : ge::DT_BF16;
    if (stateDtype != expectedStateDtype) {
      return ge::GRAPH_FAILED;
    }
  }
  if (finalStateDtype != ge::DT_BF16 && finalStateDtype != ge::DT_FLOAT && finalStateDtype != ge::DT_FLOAT16) {
    return ge::GRAPH_FAILED;
  }
  if (stateDtype != finalStateDtype) {
    return ge::GRAPH_FAILED;
  }
  tilingData_.stateIsFp32 = (stateDtype == ge::DT_FLOAT) ? 1 : 0;
  return ge::GRAPH_SUCCESS;
}

// Validate auxiliary input dtype: actual_seq_lengths is INT32, optional g is FLOAT.
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckAuxDtype() {
  auto actualSeqLengthsDtype = context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX)->GetDataType();
  if (actualSeqLengthsDtype != ge::DT_INT32) {
    return ge::GRAPH_FAILED;
  }

  if (tilingData_.hasGamma != 0) {
    auto gammaDtype = context_->GetOptionalInputDesc(G_INDEX)->GetDataType();
    if (gammaDtype != ge::DT_FLOAT) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

// Validate whether the number of shape dimensions matches the expectation
bool ChunkGatedDeltaRuleTiling::CheckDim(const gert::Shape &shape, const size_t dim, const std::string &dimDesc) {
  (void)dimDesc;
  if (shape.GetDimNum() != dim) {
    return false;
  }
  return true;
}

// Uniformly validate that all input/output shapes match the expected shapes
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckExpectedShapes(
  const gert::Shape &queryShape, const gert::Shape &keyShape, const gert::Shape &valueShape,
  const gert::Shape &betaShape, const gert::Shape &stateShape, const gert::Shape &actualSeqLengthsShape,
  const gert::Shape &outShape, const gert::Shape &finalStateShape, const gert::Shape *gShape) {
  const gert::Shape expectQueryShape = gert::Shape({tilingData_.t, tilingData_.nk, tilingData_.dk});
  const gert::Shape expectKeyShape = gert::Shape({tilingData_.t, tilingData_.nk, tilingData_.dk});
  const gert::Shape expectValueShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
  const gert::Shape expectBetaShape = gert::Shape({tilingData_.t, tilingData_.nv});
  const gert::Shape expectStateShape = gert::Shape({tilingData_.b, tilingData_.nv, tilingData_.dv, tilingData_.dk});
  const gert::Shape expectActualSeqLengthsShape = gert::Shape({tilingData_.b});
  const gert::Shape expectOutShape = gert::Shape({tilingData_.t, tilingData_.nv, tilingData_.dv});
  const gert::Shape expectFinalStateShape =
    gert::Shape({tilingData_.b, tilingData_.nv, tilingData_.dv, tilingData_.dk});

  if (queryShape != expectQueryShape) {
    return ge::GRAPH_FAILED;
  }
  if (keyShape != expectKeyShape) {
    return ge::GRAPH_FAILED;
  }
  if (valueShape != expectValueShape) {
    return ge::GRAPH_FAILED;
  }
  if (betaShape != expectBetaShape) {
    return ge::GRAPH_FAILED;
  }
  if (stateShape != expectStateShape) {
    return ge::GRAPH_FAILED;
  }
  if (actualSeqLengthsShape != expectActualSeqLengthsShape) {
    return ge::GRAPH_FAILED;
  }
  if (outShape != expectOutShape) {
    return ge::GRAPH_FAILED;
  }
  if (finalStateShape != expectFinalStateShape) {
    return ge::GRAPH_FAILED;
  }

  if (gShape != nullptr) {
    const gert::Shape expectGShape = gert::Shape({tilingData_.t, tilingData_.nv});
    if (*gShape != expectGShape) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

// Validate the dimension constraints derived from shapes
ge::graphStatus ChunkGatedDeltaRuleTiling::CheckDerivedDimConstraints() {
  if (tilingData_.t <= 0 || tilingData_.b <= 0 || tilingData_.nk <= 0 || tilingData_.dk <= 0 || tilingData_.nv <= 0 ||
      tilingData_.dv <= 0) {
    return ge::GRAPH_FAILED;
  }
  if (tilingData_.nk > 64 || tilingData_.nv > 64) {
    return ge::GRAPH_FAILED;
  }
  if (tilingData_.dv > 128 || tilingData_.dk > 128) {
    return ge::GRAPH_FAILED;
  }
  if (tilingData_.nv % tilingData_.nk != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Uniformly validate input/output shape/rank constraints, and parse tilingData dimensions from anchor shapes
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeShapes() {
  const auto &queryShape = context_->GetInputShape(QUERY_INDEX)->GetOriginShape();
  const auto &keyShape = context_->GetInputShape(KEY_INDEX)->GetOriginShape();
  const auto &valueShape = context_->GetInputShape(VALUE_INDEX)->GetOriginShape();
  const auto &betaShape = context_->GetInputShape(BETA_INDEX)->GetOriginShape();
  const auto &stateShape = context_->GetInputShape(STATE_INDEX)->GetOriginShape();
  const auto &actualSeqLengthsShape = context_->GetInputShape(ACTUAL_SEQ_LENGTHS_INDEX)->GetOriginShape();
  const auto &outShape = context_->GetOutputShape(OUTPUT_OUT_IDX)->GetOriginShape();
  const auto &finalStateShape = context_->GetOutputShape(OUTPUT_FINAL_STATE_IDX)->GetOriginShape();
  const gert::Shape *gShape = nullptr;

  // Validate anchor ranks first to make subsequent GetDim safe
  if (!CheckDim(queryShape, QKV_DIM_NUM, "query") || !CheckDim(valueShape, QKV_DIM_NUM, "value") ||
      !CheckDim(stateShape, STATE_DIM_NUM, "state")) {
    return ge::GRAPH_FAILED;
  }

  // Parse common parameters from anchor shapes
  tilingData_.t = queryShape.GetDim(DIM_0);
  tilingData_.nk = queryShape.GetDim(DIM_1);
  tilingData_.dk = queryShape.GetDim(DIM_2);
  tilingData_.nv = valueShape.GetDim(DIM_1);
  tilingData_.dv = valueShape.GetDim(DIM_2);
  tilingData_.b = stateShape.GetDim(DIM_0);

  if (CheckDerivedDimConstraints() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (tilingData_.hasGamma != 0) {
    gShape = &context_->GetOptionalInputShape(G_INDEX)->GetOriginShape();
  }

  if (CheckExpectedShapes(queryShape, keyShape, valueShape, betaShape, stateShape, actualSeqLengthsShape, outShape,
                          finalStateShape, gShape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// Format validation at the tiling stage is based on the primary format.
// GetPrimaryFormat() can absorb some derived formats, but layouts like NCL/NCHW are not folded back to ND.
// Therefore only the currently unsupported FRACTAL_NZ is rejected here, to avoid falsely blocking other
// valid ND-derived layouts.
bool ChunkGatedDeltaRuleTiling::CheckFormat(const gert::CompileTimeTensorDesc *desc, const std::string &name) {
  (void)name;
  auto primaryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
  if (primaryFormat == ge::FORMAT_FRACTAL_NZ) {
    return false;
  }
  return true;
}

// Validate input/output format; optional gamma must also be validated when present
ge::graphStatus ChunkGatedDeltaRuleTiling::AnalyzeFormat() {
  if (!CheckFormat(context_->GetInputDesc(QUERY_INDEX), "query") ||
      !CheckFormat(context_->GetInputDesc(KEY_INDEX), "key") ||
      !CheckFormat(context_->GetInputDesc(VALUE_INDEX), "value") ||
      !CheckFormat(context_->GetInputDesc(BETA_INDEX), "beta") ||
      !CheckFormat(context_->GetInputDesc(STATE_INDEX), "state") ||
      !CheckFormat(context_->GetInputDesc(ACTUAL_SEQ_LENGTHS_INDEX), "actual_seq_lengths") ||
      !CheckFormat(context_->GetOutputDesc(OUTPUT_OUT_IDX), "out") ||
      !CheckFormat(context_->GetOutputDesc(OUTPUT_FINAL_STATE_IDX), "final_state")) {
    return ge::GRAPH_FAILED;
  }
  if (tilingData_.hasGamma != 0) {
    if (!CheckFormat(context_->GetOptionalInputDesc(G_INDEX), "gamma")) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetScale() {
  auto attrs = context_->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto scalePtr = attrs->GetAttrPointer<float>(0);
  if (scalePtr == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tilingData_.scale = *scalePtr;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleTiling::GetStrides() {
  auto strideState = context_->GetInputStride(STATE_INDEX);
  if (strideState != nullptr && strideState->GetDimNum() == STATE_DIM_NUM) {
    tilingData_.stateStride0 = strideState->GetStride(0);
    tilingData_.stateStride1 = strideState->GetStride(1);
  } else {
    tilingData_.stateStride1 = static_cast<uint64_t>(tilingData_.dk) * static_cast<uint64_t>(tilingData_.dv);
    tilingData_.stateStride0 = static_cast<uint64_t>(tilingData_.nv) * tilingData_.stateStride1;
  }
  return ge::GRAPH_SUCCESS;
}

// Determines whether gamma exists and writes the status to tilingData_.hasGamma (0 or 1)
ge::graphStatus ChunkGatedDeltaRuleTiling::GetOptionalInput() {
  auto gDesc = context_->GetOptionalInputDesc(G_INDEX);
  auto gTensor = context_->GetOptionalInputTensor(G_INDEX);
  auto gShape = context_->GetOptionalInputShape(G_INDEX);
  tilingData_.hasGamma = (gDesc != nullptr && gTensor != nullptr && gShape != nullptr) ? 1 : 0;
  return ge::GRAPH_SUCCESS;
}

// Tiling dispatch entry: construct a standalone tiling object and run DoTiling.
static ge::graphStatus ChunkGatedDeltaRuleTilingFunc(gert::TilingContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ChunkGatedDeltaRuleTiling t(context);
  return t.DoTiling();
}

// Registration: standalone IMPL_OP_OPTILING (replaces upstream REGISTER_OPS_TILING_TEMPLATE + TilingRegistry).
// The second parameter is the tiling-data size, by which the framework reserves the buffer for GetTilingData<>.
IMPL_OP_OPTILING(ChunkGatedDeltaRule).Tiling(ChunkGatedDeltaRuleTilingFunc, sizeof(ChunkGatedDeltaRuleTilingData));

}  // namespace optiling
