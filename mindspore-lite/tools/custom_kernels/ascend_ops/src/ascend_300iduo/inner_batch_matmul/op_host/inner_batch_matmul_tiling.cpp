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
 * ============================================================================
 */
// InnerBatchMatMul: tiling of the hand-written Cube kernel (carries its own tiling data so that
// it stays decoupled from the bmm channel).
#include <cstdint>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "ref_shape_rules.h"                 // NOLINT(build/include_subdir)
#include "inner_batch_matmul_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace ge;                      // NOLINT(build/namespaces)
using namespace inner_batch_matmul_ref;  // NOLINT(build/namespaces)

// The host shape rules and the host/device shared tiling data must agree on the batch rank bound:
// the kernel iterates the shared arrays with batchDimNum == kIbmmMaxBatchDim while the shape rules
// validate against IBMM_MAX_BATCH_DIM.
static_assert(IBMM_MAX_BATCH_DIM == InnerBatchMatMul::kIbmmMaxBatchDim,
              "batch rank bound mismatch between host rules and tiling data");

namespace {
// Attribute / input positions of the op def.
constexpr size_t kAttrAdjX1Idx = 0;
constexpr size_t kAttrAdjX2Idx = 1;
constexpr size_t kAttrBaseMIdx = 2;
constexpr size_t kAttrBaseNIdx = 3;
constexpr size_t kInputBiasIdx = 2;

constexpr int64_t kBaseM = 128;
constexpr int64_t kBaseN = 128;
constexpr int64_t kFractalDim = 16;                           // fractal edge length of FRACTAL_NZ
constexpr int64_t kFractalElems = kFractalDim * kFractalDim;  // elements per fractal
// Largest tile edge along M/N accepted by the tiling (L0A/L0B and the UB staging cap both at 256).
constexpr int64_t kTileDimMax = 256;
// Elements of one L0A/L0B buffer (64KB in half): the capacity one tile of A and B must fit into.
constexpr int64_t kL0MaxElems = 32768;
// K span of one L1 ping-pong block: 128 for narrow N tiles, 64 for wide ones so that A and B still
// fit twice into L0.
constexpr int64_t kKL1Narrow = 128;
constexpr int64_t kKL1Wide = 64;
// Upper M bound of the dedicated small-M path.
constexpr int64_t kSmallMaxM = 64;
// Small-M tile split: a tile count that divides the cores (or fills them many times over) is preferred.
constexpr int64_t kSmallMTileBalance = 32;
// Enabling threshold of the 2x2 output blocking: beyond this K the whole-K B no longer fits in L1
// and MTE2 becomes the bottleneck, so halving the A/B re-reads from GM pays off the most; small-K
// shapes (compute dominated) keep a single tile.
constexpr int64_t kLargeKFor2x2 = 1536;
// 2x2 output blocking: one super tile bundles four base tiles; it is enabled only when the per-core
// work grows by at most 6/5 and all cores stay busy.
constexpr int64_t kTileMul2x2 = 2;
constexpr int64_t kTile2QuadCount = 4;
constexpr int64_t kTile2SlackNum = 5;
constexpr int64_t kTile2SlackDen = 6;
// L1 budget of a whole-K resident B, and the per-wave A resident set bound that keeps A in L2.
constexpr uint64_t kL1BResidentBytes = 768ULL << 10;
constexpr uint64_t kL2ABytes = 12ULL << 20;
// Bias modes shared with the device (host/device contract of InnerBatchMatMulParams::biasMode).
constexpr uint32_t kNoBias = 0;
constexpr uint32_t kRowBias = 1;
constexpr uint32_t kFullBias = 2;

// The shape reaches the device as uint32 and every GM<->L1/L0/UB copy uses 16-bit strides, so the
// bounds below are mandatory on the host side (the kernel cannot detect a wrap):
//   A srcStride = (M/16 - m1) * 16  <= UINT16_MAX  ->  M <= 65536
//   B srcStride = (K/16 - kf) * 16  <= UINT16_MAX  ->  K <= 65536
//   y dstStride = (N - n1*16) / 16  <= UINT16_MAX  ->  N <= 1048560 (and N/16 <= UINT16_MAX)
constexpr int64_t kMaxDimM = 65536;
constexpr int64_t kMaxDimK = 65536;
constexpr int64_t kMaxDimN = 65535 * 16;

// Tile shape chosen for the current problem.
struct TilePlan {
  int64_t tileM = kBaseM;
  int64_t tileN = kBaseN;
  int64_t tileK = 0;
  int64_t tileMul = 1;
  int64_t smallM = 0;
};

// Right-aligned 6-dim batch dims of x1 / x2 / y after broadcast expansion (the source of the strides).
struct BatchSplit {
  int64_t a[IBMM_MAX_BATCH_DIM] = {0};
  int64_t b[IBMM_MAX_BATCH_DIM] = {0};
  int64_t c[IBMM_MAX_BATCH_DIM] = {0};
};

// Raw static dims of x1 / x2 as read from the shapes (filled by ParseDims).
struct RawDims {
  int64_t x1[kMaxRank] = {0};
  int64_t x2[kMaxRank] = {0};
};

// Parses the static shapes and validates the supported range. Also fills the aligned raw dimension
// arrays used by the batch-broadcast step.
static bool ParseDims(const gert::RuntimeAttrs *attrs, const gert::StorageShape *x1Shape,
                      const gert::StorageShape *x2Shape, MatMulDims &dims, RawDims &raw) {
  // adj_x1/adj_x2 are INT in the op def (matching what ONNX can express); any non-zero value fails explicitly.
  const int64_t *adjX1 = attrs->GetAttrPointer<int64_t>(kAttrAdjX1Idx);
  const int64_t *adjX2 = attrs->GetAttrPointer<int64_t>(kAttrAdjX2Idx);
  if ((adjX1 != nullptr && *adjX1 != 0) || (adjX2 != nullptr && *adjX2 != 0)) {
    return false;
  }
  size_t x1Rank = 0;
  size_t x2Rank = 0;
  if (!GetStaticDims(x1Shape->GetOriginShape(), kMaxRank, raw.x1, x1Rank) ||
      !GetStaticDims(x2Shape->GetOriginShape(), kMaxRank, raw.x2, x2Rank)) {
    return false;
  }
  if (!GetMatMulDims(raw.x1, x1Rank, raw.x2, x2Rank, dims) ||
      !BroadcastBatchDims(raw.x1, dims.x1BatchRank, raw.x2, dims.x2BatchRank, dims)) {
    return false;
  }
  if (dims.m % kFractalDim != 0 || dims.n % kFractalDim != 0 || dims.k % kFractalDim != 0) {
    return false;
  }
  if (dims.m > kMaxDimM || dims.k > kMaxDimK || dims.n > kMaxDimN) {
    return false;
  }
  return true;
}

// Right-aligns the batch dims of x1/x2/output and multiplies out the broadcast batch count.
static bool AlignBatchDimsAll(const RawDims &raw, const MatMulDims &dims, BatchSplit &split, uint64_t &cBatchAll) {
  if (!AlignBatchDims(raw.x1, dims.x1BatchRank, dims.outBatch, dims.outBatchRank, split.a) ||
      !AlignBatchDims(raw.x2, dims.x2BatchRank, dims.outBatch, dims.outBatchRank, split.b) ||
      !AlignBatchDims(dims.outBatch, dims.outBatchRank, dims.outBatch, dims.outBatchRank, split.c)) {
    return false;
  }
  cBatchAll = 1;
  for (size_t i = 0; i < IBMM_MAX_BATCH_DIM; ++i) {
    cBatchAll *= static_cast<uint64_t>(split.c[i]);
  }
  // batchCount is narrowed to uint32 on the device and multiplied by mTiles*nTiles there.
  return cBatchAll <= static_cast<uint64_t>(UINT32_MAX);
}

// Optional bias: full size [batch..., M, N] (same shape as the output, NZ input) or row broadcast
// [batch..., 1, N]; batch dims are broadcast right-aligned.
static bool BuildBiasTiling(const gert::TilingContext *context, const MatMulDims &dims, InnerBatchMatMulParams &cp) {
  cp.biasMode = kNoBias;
  for (int32_t i = 0; i < IBMM_MAX_BATCH_DIM; ++i) {
    cp.biasBatchStride[i] = 0;
  }
  const gert::StorageShape *biasShapePtr = context->GetOptionalInputShape(kInputBiasIdx);
  if (biasShapePtr == nullptr || biasShapePtr->GetOriginShape().GetDimNum() == 0) {
    return true;  // no bias: the zero-initialised strides above are the contract
  }
  int64_t biasDims[kMaxRank] = {0};
  size_t biasRank = 0;
  if (!GetStaticDims(biasShapePtr->GetOriginShape(), kMaxRank, biasDims, biasRank) || biasRank < kMinRank) {
    return false;
  }
  if (biasDims[biasRank - 1] != dims.n) {
    return false;
  }
  if (biasRank - kMatDims > dims.outBatchRank) {
    return false;  // more bias batch dims than the output: not broadcastable
  }
  const bool rowBias = (biasDims[biasRank - kMatDims] == 1);
  if (!rowBias && biasDims[biasRank - kMatDims] != dims.m) {
    return false;  // only full-size or row-broadcast bias is supported
  }
  int64_t biasBatch[IBMM_MAX_BATCH_DIM] = {0};
  // The second-to-last dim is the broadcast row (row bias) or M (full bias), never a batch dim, so
  // a bias always has rank-2 batch dims.
  if (!AlignBatchDims(biasDims, biasRank - kMatDims, dims.outBatch, dims.outBatchRank, biasBatch)) {
    return false;
  }
  // Elements per batch in NZ storage (the last two dims are padded to 16):
  //   full [..,M,N] -> (N/16)*(M/16)*256 = M*N; row [..,1,N] -> (N/16)*1*256 = 16*N.
  uint64_t biasAcc = rowBias ? static_cast<uint64_t>((dims.n / kFractalDim) * kFractalElems)
                             : static_cast<uint64_t>(dims.m) * static_cast<uint64_t>(dims.n);
  for (int32_t i = IBMM_MAX_BATCH_DIM - 1; i >= 0; --i) {
    cp.biasBatchStride[i] = biasBatch[i] > 1 ? biasAcc : 0;
    biasAcc *= static_cast<uint64_t>(biasBatch[i]);
  }
  cp.biasMode = rowBias ? kRowBias : kFullBias;
  return true;
}

// Initial M/N tile shape (auto selection and the base_m/base_n overrides, then the capacity clamps).
static void SelectTileShape(const gert::RuntimeAttrs *attrs, const MatMulDims &dims, int32_t coreNum, TilePlan &plan) {
  // Base tile: when K is small enough for the whole-K B to stay in L1, a 256-wide N tile halves the
  // tile count per core; large K uses 128x128. The N tile is 256 only if the tile count divides the
  // core count (balanced), otherwise it falls back to 128.
  {
    const int64_t mTilesTmp = (dims.m + plan.tileM - 1) / plan.tileM;
    const int64_t nTiles256 = (dims.n + kTileDimMax - 1) / kTileDimMax;
    const int64_t tiles256 = mTilesTmp * nTiles256;
    const bool bFits = static_cast<int64_t>(dims.k) * kTileDimMax * static_cast<int64_t>(sizeof(uint16_t)) <=
                       static_cast<int64_t>(kL1BResidentBytes);
    if (bFits && (coreNum <= 0 || tiles256 % coreNum == 0 || tiles256 < coreNum)) {
      plan.tileN = kTileDimMax;
    }
  }
  if (plan.tileN * dims.k * static_cast<int64_t>(sizeof(uint16_t)) > static_cast<int64_t>(kL1BResidentBytes)) {
    plan.tileN = kBaseN;
  }
  // The guard must cover the largest index read below (base_n): with exactly three attributes
  // present (adj_x1, adj_x2, base_m) reading index 3 would step past the attribute array.
  if (attrs->GetAttrNum() > kAttrBaseNIdx) {
    const int64_t *bm = attrs->GetAttrPointer<int64_t>(kAttrBaseMIdx);
    const int64_t *bn = attrs->GetAttrPointer<int64_t>(kAttrBaseNIdx);
    if (bm != nullptr && *bm > 0) {
      plan.tileM = *bm;
    }
    if (bn != nullptr && *bn > 0) {
      plan.tileN = *bn;
    }
  }
  plan.tileM = plan.tileM < kFractalDim ? kFractalDim : (plan.tileM / kFractalDim) * kFractalDim;
  plan.tileN = plan.tileN < kFractalDim ? kFractalDim : (plan.tileN / kFractalDim) * kFractalDim;
  plan.tileM = plan.tileM > kTileDimMax ? kTileDimMax : plan.tileM;
  plan.tileN = plan.tileN > kTileDimMax ? kTileDimMax : plan.tileN;
  while (plan.tileM * plan.tileN > kL0MaxElems) {
    plan.tileN -= kFractalDim;
  }
  // must match the kernel kL1 (ping-pong needs <= 32KB per block)
  plan.tileK = plan.tileN <= kBaseN ? kKL1Narrow : kKL1Wide;
  // A is double buffered in L0A (64KB), so one tile of A must satisfy 2 * tileM * kL1 * 2B <= 64KB,
  // i.e. tileM * kL1 <= 16384 elements. An explicit base_m above that (256 with kL1 = 128) is
  // clamped, otherwise the kernel Init cannot allocate the two L0A buffers.
  constexpr int64_t kL0aElemsPerTile = 16384;
  if (plan.tileM * plan.tileK > kL0aElemsPerTile) {
    plan.tileM = kL0aElemsPerTile / plan.tileK;
  }
}

// Dedicated small-M path (M <= 64): mTiles is always 1, so resident B cannot help and the per
// tile / per chunk fixed costs dominate. baseM = M, and baseN prefers 256 (halves the tile count)
// as long as the cores stay balanced.
static void SelectSmallM(const gert::RuntimeAttrs *attrs, const MatMulDims &dims, int32_t coreNum, TilePlan &plan) {
  if (dims.m > kSmallMaxM) {
    return;
  }
  plan.smallM = 1;
  plan.tileM = dims.m;
  const int64_t nT256 = (dims.n + kTileDimMax - 1) / kTileDimMax;
  const int64_t nT128 = (dims.n + kBaseN - 1) / kBaseN;
  int64_t smallTileN;  // assigned by exactly one of the branches below
  if (dims.n >= kTileDimMax && (nT256 % coreNum == 0 || nT256 >= kSmallMTileBalance * coreNum)) {
    smallTileN = kTileDimMax;
  } else if (dims.n >= kBaseN && (nT128 % coreNum == 0 || nT128 >= kSmallMTileBalance * coreNum)) {
    smallTileN = kBaseN;
  } else {
    smallTileN = dims.n < kBaseN ? dims.n : kBaseN;
  }
  // An explicit base_n attribute wins; otherwise use the rule above.
  bool bnForced = false;
  if (attrs->GetAttrNum() > kAttrBaseNIdx) {
    const int64_t *bn = attrs->GetAttrPointer<int64_t>(kAttrBaseNIdx);
    bnForced = (bn != nullptr && *bn > 0);
  }
  if (!bnForced) {
    plan.tileN = smallTileN;
  }
  plan.tileN = plan.tileN < kFractalDim ? kFractalDim : (plan.tileN / kFractalDim) * kFractalDim;
  plan.tileN = plan.tileN > kTileDimMax ? kTileDimMax : plan.tileN;
}

// 2x2 output blocking: a super tile of 2*baseM x 2*baseN with four L0C accumulators, so A and B are
// each reused twice. Enabled for large K (B cannot stay in L1, MTE2 dominated) only when the
// per-core tile count does not grow.
static void SelectTileMul(const MatMulDims &dims, int32_t coreNum, TilePlan &plan) {
  const int64_t superM = plan.tileM * kTileMul2x2;
  const int64_t superN = plan.tileN * kTileMul2x2;
  const int64_t mT2 = (dims.m + superM - 1) / superM;
  const int64_t nT2 = (dims.n + superN - 1) / superN;
  const int64_t tot2 = mT2 * nT2;
  const int64_t mT1 = (dims.m + plan.tileM - 1) / plan.tileM;
  const int64_t nT1 = (dims.n + plan.tileN - 1) / plan.tileN;
  const int64_t tot1 = mT1 * nT1;
  const int64_t per2 = (tot2 + coreNum - 1) / coreNum;
  const int64_t per1 = (tot1 + coreNum - 1) / coreNum;
  // The per-core work (in baseM x baseN units) stays within 1.2x of the single-tile schedule and all
  // cores stay busy.
  if (plan.smallM == 0 && plan.tileM == kBaseM && plan.tileN == kBaseN && dims.k > kLargeKFor2x2 && dims.m >= superM &&
      dims.n >= superN && tot2 >= coreNum && per2 * kTile2QuadCount * kTile2SlackNum <= per1 * kTile2SlackDen) {
    plan.tileMul = kTileMul2x2;
  }
}

// Tile counts, core split and the small-M stride assignment.
static bool FillTileGeometry(InnerBatchMatMulParams &cp, int64_t smallM) {
  cp.tileMul = static_cast<uint32_t>(smallM == 0 ? cp.tileMul : 1);
  cp.mTiles = (cp.m + cp.baseM * cp.tileMul - 1) / (cp.baseM * cp.tileMul);
  cp.nTiles = (cp.n + cp.baseN * cp.tileMul - 1) / (cp.baseN * cp.tileMul);
  // Compute the tile count in 64 bits first: a 32-bit product would wrap and silently leave a
  // suffix of the output untouched.
  const uint64_t totalTiles64 = static_cast<uint64_t>(cp.batchCount) * cp.mTiles * cp.nTiles;
  if (totalTiles64 > static_cast<uint64_t>(UINT32_MAX)) {
    return false;
  }
  cp.totalTiles = static_cast<uint32_t>(totalTiles64);
  cp.usedCoreNum = cp.totalTiles < cp.coreNum ? cp.totalTiles : cp.coreNum;
  if (cp.usedCoreNum == 0) {
    cp.usedCoreNum = 1;
  }
  cp.tilePerCore = (cp.totalTiles + cp.usedCoreNum - 1) / cp.usedCoreNum;
  cp.usedCoreNum = (cp.totalTiles + cp.tilePerCore - 1) / cp.tilePerCore;
  if (smallM != 0 && cp.totalTiles >= cp.coreNum) {
    // Small-M tiles share no data (mTiles=1, B not resident), so assignment uses the cross-core stride
    // to even the split: all cores are used and per-core tile counts differ by at most 1.
    cp.strideAssign = 1U;
    cp.usedCoreNum = (cp.totalTiles < cp.coreNum) ? cp.totalTiles : cp.coreNum;
    cp.tilePerCore = (cp.totalTiles + cp.usedCoreNum - 1) / cp.usedCoreNum;
  }
  return true;
}

// Per-batch element strides of x1/x2/y (a broadcast dim gets stride 0).
static void FillBatchStrides(const MatMulDims &dims, const BatchSplit &split, InnerBatchMatMulParams &cp) {
  uint64_t aAcc = static_cast<uint64_t>(dims.m) * static_cast<uint64_t>(dims.k);
  uint64_t bAcc = static_cast<uint64_t>(dims.k) * static_cast<uint64_t>(dims.n);
  uint64_t cAcc = static_cast<uint64_t>(dims.m) * static_cast<uint64_t>(dims.n);
  for (int32_t i = IBMM_MAX_BATCH_DIM - 1; i >= 0; --i) {
    cp.outBatchDim[i] = static_cast<uint32_t>(split.c[i]);
    cp.x1BatchStride[i] = split.a[i] > 1 ? aAcc : 0;
    cp.x2BatchStride[i] = split.b[i] > 1 ? bAcc : 0;
    cp.cBatchStride[i] = split.c[i] > 1 ? cAcc : 0;
    aAcc *= static_cast<uint64_t>(split.a[i]);
    bAcc *= static_cast<uint64_t>(split.b[i]);
    cAcc *= static_cast<uint64_t>(split.c[i]);
  }
}

// Cross-core stride assignment and the A-stationary phased schedule.
static void SelectSchedules(InnerBatchMatMulParams &cp, int64_t smallM) {
  // When the whole-K B does not fit in L1, switch to cross-core stride assignment: the cores work on
  // the same B column at the same time (shared through L2), cutting the B DRAM re-reads from mTiles to ~1.
  if (smallM == 0) {  // the small-M strideAssign was decided separately (all cores + even split)
    cp.strideAssign =
      (static_cast<uint64_t>(cp.k) * cp.baseN * cp.tileMul * sizeof(uint16_t) > kL1BResidentBytes) ? 1U : 0U;
  }
  // A-stationary phased schedule (grpSched=1): core c finishes all ni of mi = c before moving to
  // mi = c + usedCoreNum. The stride loop varies mi fastest, so the same-wave A resident set spans many
  // A tiles and A is re-read from DRAM whenever ni changes; after phasing each core pins one A tile and
  // the A DRAM traffic drops from mTiles*nTiles*A_tile to mTiles*A_tile.
  // The batch dim is the outermost loop (the kernel's grpSched branch runs the whole phase sequence per batch).
  cp.grpSched = 0;
  if (smallM == 0) {
    const uint32_t tileMAll = cp.baseM * cp.tileMul;
    const uint64_t aBytes = static_cast<uint64_t>(cp.usedCoreNum) * tileMAll * cp.k * sizeof(uint16_t);
    if (cp.tileMul > 1U && cp.strideAssign != 0U && cp.mTiles % cp.usedCoreNum == 0U && cp.nTiles >= 2U &&
        aBytes <= kL2ABytes) {
      cp.grpSched = 1U;
    }
  }
}

static uint32_t InnerBatchMatMulTilingFunc(gert::TilingContext *context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  auto x1ShapePtr = context->GetInputShape(0);
  auto x2ShapePtr = context->GetInputShape(1);
  auto attrs = context->GetAttrs();
  auto *tiling = context->GetTilingData<InnerBatchMatMulTilingData>();
  if (x1ShapePtr == nullptr || x2ShapePtr == nullptr || attrs == nullptr || tiling == nullptr) {
    return ge::GRAPH_FAILED;
  }
  MatMulDims dims;
  RawDims raw;
  if (!ParseDims(attrs, x1ShapePtr, x2ShapePtr, dims, raw)) {
    return ge::GRAPH_FAILED;
  }
  BatchSplit split;
  uint64_t cBatchAll = 1;
  if (!AlignBatchDimsAll(raw, dims, split, cBatchAll)) {
    return ge::GRAPH_FAILED;
  }
  auto &cp = tiling->cube;
  if (!BuildBiasTiling(context, dims, cp)) {
    return ge::GRAPH_FAILED;
  }

  platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
  int32_t coreNum = ascendcPlatform.GetCoreNumAic();
  if (coreNum <= 0) {
    coreNum = 1;
  }

  cp.m = static_cast<uint32_t>(dims.m);
  cp.n = static_cast<uint32_t>(dims.n);
  cp.k = static_cast<uint32_t>(dims.k);
  cp.batchCount = static_cast<uint32_t>(cBatchAll);
  cp.coreNum = static_cast<uint32_t>(coreNum);

  TilePlan plan;
  SelectTileShape(attrs, dims, coreNum, plan);
  SelectSmallM(attrs, dims, coreNum, plan);
  SelectTileMul(dims, coreNum, plan);
  cp.tileMul = static_cast<uint32_t>(plan.tileMul);
  cp.baseM = static_cast<uint32_t>(plan.tileM);
  cp.baseN = static_cast<uint32_t>(plan.tileN);
  cp.smallM = static_cast<uint32_t>(plan.smallM);
  if (!FillTileGeometry(cp, plan.smallM)) {
    return ge::GRAPH_FAILED;
  }
  FillBatchStrides(dims, split, cp);
  SelectSchedules(cp, plan.smallM);

  cp.batchDimNum = InnerBatchMatMul::kIbmmMaxBatchDim;
  tiling->useCube = 1;
  context->SetBlockDim(cp.usedCoreNum);
  size_t *workspace = context->GetWorkspaceSizes(1);
  if (workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  workspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}  // namespace

// Single operator: bias is OPTIONAL, so the same tiling / kernel serves both cases.
// Without bias GetOptionalInputShape(2) is empty -> biasMode=0 and the kernel bias pointer is nullptr.
IMPL_OP(InnerBatchMatMul).Tiling(InnerBatchMatMulTilingFunc, sizeof(InnerBatchMatMulTilingData));
