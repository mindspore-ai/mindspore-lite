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

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "fused_rms_norm_nz_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input indices of the FusedRmsNormNZ operator.
constexpr uint32_t kXInput = 0;
constexpr uint32_t kGammaInput = 1;
// dav_m200 has no DataCopyPad: N must keep every whole-row copy, every NZ
// 16-element segment and every row base 32B aligned, hence a multiple of
// 32 (e.g. N=2048 qualifies).
constexpr uint32_t kAlignCols = 32;
constexpr uint32_t kMinCols = 32;
constexpr uint32_t kRowBlockCandidates[] = {16, 12, 8, 7, 6, 4, 3, 2, 1};
// Plain count-form vector ops (Cast/Mul/Add with int32 count) run over the
// whole [R, n] tile in one instruction each: counter-mode masks have no
// repeat cap, but the largest flat counts verified on this chip are 8192
// (fused_qk_norm_rope deploy) and 12288 (gelu v3 chunk), so R*n is capped
// at 16384.
constexpr uint32_t kMaxFlatCount = 16384;
// UB budget: the 200KB conservative cap minus the margin that keeps every
// candidate inside the ~196KB empirically proven envelope (swiglu v3's
// measured ceiling). Must match the kernel's ClampRowBlock formula so a
// stale v1 MindIR clamps to a value the kernel actually allocated for.
constexpr uint32_t kUbBudgetBytes = 200 * 1024;
constexpr uint32_t kUbMarginBytes = 4 * 1024;
constexpr uint32_t kUbSlackBytes = 512;
// DataCopyParams blockCount and strides are uint16 (strides in 32B blocks):
// the NZ per-row segment stride m1t*16-1 and the whole-row scatter block
// count n/16 must both fit.
constexpr uint32_t kMaxCopyParam = 65535;
constexpr uint32_t kMinBlockDim = 1;
constexpr float kDefaultEpsilon = 1e-6;
// Whole-row residency tile bytes per row-block candidate, mirroring the
// kernel's InitBuffer (coefficients are bytes per unit of the dimension in
// the name).
constexpr uint32_t kWholeRowBytesPerElem = 12;  // 4 flat tiles: fp16 x/y + fp32 work/tmp (2+4+4+2)
constexpr uint32_t kSlotTilesBytesPerRow = 96;  // three [R, 8] fp32 slot tiles
constexpr uint32_t kGammaBytesPerElem = 6;      // fp16 gamma staging + fp32 resident copy (2+4)
// NZ storage trailing dimensions for a logical [P..., M, N] fp16 tensor:
// [P..., n1, m1t, 16, 16] (16x16 fractal row-major, C0 bytes per block).
constexpr uint32_t kNzInnerDim = 16;  // fractal inner size (fp16 elements per 32B block)
constexpr uint32_t kNzTailDims = 4;   // trailing dims that describe the NZ layout
constexpr uint32_t kNzM1tOffset = 3;  // m1t (row-fractal count) sits at dimNum-3
constexpr uint32_t kNzTailOffset = 2;
uint32_t CeilDiv(uint32_t val, uint32_t d) { return (val + d - 1) / d; }

bool GetPlatformInfo(const TilingContext *context, uint32_t &aivNum) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    return false;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  aivNum = platform.GetCoreNumAiv();
  if (aivNum == 0) {
    aivNum = platform.GetCoreNum();
  }
  return aivNum > 0;
}

// Read the optional epsilon attr (idx 0, the only attr), falling back to the
// OpDef default (same pattern as op B / ChunkGatedDeltaRule).
void ParseEpsilon(const TilingContext *context, float &epsilon) {
  epsilon = kDefaultEpsilon;
  auto attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetAttrNum() == 0) {
    return;
  }
  const float *epsPtr = attrs->GetAttrPointer<float>(0);
  if (epsPtr != nullptr) {
    epsilon = *epsPtr;
  }
}

// Exact mirror of the kernel's InitBuffer footprint for one rowBlock
// candidate, in bytes (whole-row flat tiles + slot tiles + gamma tables).
uint32_t BlockBytes(uint32_t n, uint32_t rowBlock) {
  return rowBlock * (kWholeRowBytesPerElem * n + kSlotTilesBytesPerRow) + kGammaBytesPerElem * n + kUbSlackBytes;
}

// Flatten the logical [P..., M, N] x shape and detect the storage layout.
// FRACTAL_NZ storage of a logical fp16 tensor is [P..., n1, m1t, 16, 16]:
// column fractals outermost (n1 = N/16), then row blocks (m1t = M_pad/16),
// then the 16x16 fractal row-major [m0, n0] (CANN include
// pto/common/pto_tile.hpp: TileShape2D<Layout::NZ> =
// Shape<1, cols/C0, rows/16, 16, C0> with C0 = 32B/sizeof(T) = 16 for
// fp16). The gamma length disambiguates NZ from ND: an ND row-norm operand
// always ends in dim == N, while NZ ends in [.., n1, m1t, 16, 16] with
// n1*16 == N. Delegated out of FusedRmsNormNZTilingFunc so the tiling
// function's cyclomatic complexity stays bounded.
bool ResolveRmsLayout(const gert::StorageShape *xShape, uint32_t n, uint32_t *mOut, uint32_t *isNzOut,
                      uint32_t *m1tOut) {
  auto xStorage = xShape->GetStorageShape();
  size_t dimNum = xStorage.GetDimNum();
  if (dimNum == 0) {
    return false;
  }
  if (dimNum >= kNzTailDims && xStorage.GetDim(dimNum - 1) == kNzInnerDim &&
      xStorage.GetDim(dimNum - kNzTailOffset) == kNzInnerDim &&
      static_cast<uint64_t>(xStorage.GetDim(dimNum - kNzTailDims)) * kNzInnerDim == n) {
    uint32_t m1t = static_cast<uint32_t>(xStorage.GetDim(dimNum - kNzM1tOffset));
    uint64_t rowsPerPrefix = static_cast<uint64_t>(m1t) * kNzInnerDim;
    uint64_t prefix = 1;
    for (size_t d = 0; d + kNzTailDims < dimNum; ++d) {
      prefix *= static_cast<uint64_t>(xStorage.GetDim(d));
    }
    uint64_t rows = prefix * rowsPerPrefix;
    if (rows == 0 || rows > 0xFFFFFFFFULL || m1t == 0) {
      return false;
    }
    // uint16 stride guard: NZ row segments sit m1t*512B apart, i.e. the
    // DataCopyParams stride m1t*16-1 (32B blocks).
    if (m1t * kNzInnerDim - 1 > kMaxCopyParam) {
      return false;
    }
    *mOut = static_cast<uint32_t>(rows);
    *isNzOut = 1;
    *m1tOut = m1t;
    return true;
  }
  if (static_cast<uint64_t>(xStorage.GetDim(dimNum - 1)) != n) {
    return false;
  }
  uint64_t rows = 1;
  for (size_t d = 0; d + 1 < dimNum; ++d) {
    rows *= static_cast<uint64_t>(xStorage.GetDim(d));
  }
  if (rows == 0 || rows > 0xFFFFFFFFULL) {
    return false;
  }
  *mOut = static_cast<uint32_t>(rows);
  *isNzOut = 0;
  *m1tOut = 0;
  return true;
}

uint32_t FusedRmsNormNZTilingFunc(TilingContext *context) {
  auto xShape = context->GetInputShape(kXInput);
  auto gammaShape = context->GetInputShape(kGammaInput);
  if (xShape == nullptr || gammaShape == nullptr) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  if (!GetPlatformInfo(context, aivNum)) {
    return GRAPH_FAILED;
  }

  // gamma is the row-length vector: [N].
  auto gammaStorage = gammaShape->GetStorageShape();
  if (gammaStorage.GetDimNum() != 1) {
    return GRAPH_FAILED;
  }
  uint64_t nU = static_cast<uint64_t>(gammaStorage.GetDim(0));
  if (nU < kMinCols || nU % kAlignCols != 0 || nU > 0xFFFFFFFFULL) {
    return GRAPH_FAILED;
  }
  uint32_t n = static_cast<uint32_t>(nU);

  // Resolve the flattened row count m and the layout (NZ vs ND); the detail
  // lives in ResolveRmsLayout to keep this function's cyclomatic complexity
  // bounded.
  uint32_t m = 0;
  uint32_t isNz = 0;
  uint32_t m1t = 0;
  if (!ResolveRmsLayout(xShape, n, &m, &isNz, &m1t)) {
    return GRAPH_FAILED;
  }
  // uint16 blockCount guard: the whole-row NZ scatter is n/16 blocks.
  if (n / kNzInnerDim > kMaxCopyParam) {
    return GRAPH_FAILED;
  }

  float epsilon = kDefaultEpsilon;
  ParseEpsilon(context, epsilon);

  auto td = context->GetTilingData<FusedRmsNormNZTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }

  // Largest row block that fits the proven UB envelope with whole-row
  // residency and keeps the flat pass-1 ops inside the verified
  // counter-mode envelope. Whole-row residency bounds N: even R=1 needs
  // 18*N + 96 + 512 <= 200KB - 4KB, i.e. N <= 11116 (the largest accepted
  // multiple of 32 is 11104; v1 chunked around this and no practical N
  // size comes close) -- larger N fails here.
  uint32_t rowBlock = 0;
  for (uint32_t r = 0; r < sizeof(kRowBlockCandidates) / sizeof(kRowBlockCandidates[0]); ++r) {
    uint32_t cand = kRowBlockCandidates[r];
    if (BlockBytes(n, cand) <= kUbBudgetBytes - kUbMarginBytes && cand * n <= kMaxFlatCount) {
      rowBlock = cand;
      break;
    }
  }
  if (rowBlock == 0) {
    return GRAPH_FAILED;
  }

  // Row blocks are striped over cores; never launch more cores than there
  // are blocks. The kernel derives its block range as
  // [blockIdx*blocks/coreNum, (blockIdx+1)*blocks/coreNum), rows
  // [start*R, min(end*R, m)). Pad rows of the NZ tail (m1t*16 > logical M)
  // are garbage-in-garbage-out within the allocation, like the gelu op.
  uint32_t blocks = CeilDiv(m, rowBlock);
  uint32_t blockDim = (blocks < aivNum) ? blocks : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->m = m;
  td->n = n;
  // colChunk retired by v2 whole-row residency; kept in the struct for
  // byte-compatibility with v1 MindIR tiling blocks (the kernel ignores
  // it) and written as n for readability.
  td->colChunk = n;
  td->rowBlock = rowBlock;
  td->isNz = isNz;
  td->epsilon = epsilon;
  td->m1t = m1t;

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: x streams straight from the sub-layer GEMM.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedRmsNormNZ).Tiling(FusedRmsNormNZTilingFunc, sizeof(FusedRmsNormNZTilingData));
