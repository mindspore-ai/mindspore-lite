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
#include "fused_add_rms_norm_smooth_quant_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {
// Input indices of the FusedAddRmsNormSmoothQuant operator.
constexpr uint32_t kX1Input = 0;
constexpr uint32_t kX2Input = 1;
constexpr uint32_t kGammaInput = 2;
constexpr uint32_t kSmoothInput = 3;
// Shape layout: [M..., H], leading dims flattened into M by the kernel.
constexpr uint32_t kDimN = 1;
// Rows are striped over cores at 8-row-group granularity so every core's
// token_scale DataCopy is a whole 32B block (fp32 x 8) at a 32B-aligned GM
// offset; the 32B-padded allocation absorbs the ragged last group.
constexpr uint32_t kScaleGroupRows = 8;
// dav_m200 has no DataCopyPad: H must keep every per-chunk copy 32B aligned
// for fp16 in/out and int8 out, hence a multiple of 32.
constexpr uint32_t kAlignCols = 32;
constexpr uint32_t kMinCols = 32;
constexpr uint32_t kMinColChunk = 32;
// DataCopyParams blockCount/strides are uint16 (strides in 32B blocks): the
// NZ per-row gather/scatter uses blockCount n/16 (fp16) and n/32 (int8), and
// the shared segment stride m1t*16-1 (a 16x16 fp16 fractal and a 16x32 int8
// fractal are both 512B, so fp16 and int8 segments sit the same m1t*512B
// apart).
constexpr uint32_t kMaxCopyParam = 65535;
// NZ fractal geometry (fp16): the fractal is 16 rows x 16 cols (C0 =
// 32B/sizeof(half) = 16); the int8 C0 is 32. kNz*Offset are the positions
// of the NZ storage suffix [n1, m1t, 16, 16] counted from the last dim.
constexpr uint32_t kNzFractalRows = 16;
constexpr uint32_t kNzHalfBlockCols = 16;
constexpr uint32_t kNzInt8BlockCols = 32;
constexpr size_t kNzN0Offset = 1;
constexpr size_t kNzM0Offset = 2;
constexpr size_t kNzM1tOffset = 3;
constexpr size_t kNzN1Offset = 4;
// The NZ branch only supports 2D origins [M, H] (see DetectLayout).
constexpr size_t kNzOriginRank = 2;
// v3 row-block execution: the kernel runs its whole V section as one
// in-order stream with barriers only at the scalar readbacks, so the
// per-block cost is dominated by vector op issue, which scales with chunks
// per row C (every extra chunk re-runs the sum tree and exposes an in-block
// MTE2 drain that nothing overlaps). rowBlock must still divide the 8-row
// scale group so block boundaries never straddle a group.
constexpr uint32_t kRowBlockCandidates[] = {8, 4, 2, 1};
// UB budget. Fixed per core: fp16 row-group tiles rRow + actRow (4B/col/row)
// and int8 y tile (1B/col/row) scale with rowBlock; the absmax/pass3 scratch
// tmpH is sized pow2ceil(H) fp16; the fused norm constant gsF32 (4B/col,
// gamma/smooth are staged inside the x tiles at Init, not separately); the
// pow2ceil(chunk) fp32 scratch (shared by pass1 squares and pass2 act math);
// per-chunk x1/x2 fp16 tiles (4B/col/row) and three 32B-strided per-row
// scalar slot tiles (sum/max/invRms, 96B/row); the 32B scale stage.
constexpr uint32_t kGsBytesPerCol = 4;
constexpr uint32_t kScalarStageBytes = 256;
constexpr uint32_t kUbBudgetBytes = 200 * 1024;
constexpr uint32_t kMinBlockDim = 1;
constexpr float kDefaultEpsilon = 1e-6;
// UB footprint composites for BlockBytes below.
constexpr uint32_t kXChunkTilesBytesPerCol = 4;  // x1 + x2 fp16 chunk tiles
constexpr uint32_t kRowTilesBytesPerCol = 5;     // rRow + actRow fp16 + y int8
constexpr uint32_t kFp32ScratchBytesPerCol = 4;  // fp32 scratch (pass1 squares + pass2 act)
constexpr uint32_t kTmpHalfBytesPerCol = 2;      // tmpH absmax/pass3 fp16 scratch
constexpr uint32_t kSlotTilesBytes = 3 * 32;     // 32B-strided scalar slot tiles (sum/max/invRms)
// Explicit-sync model for SyncsPerRow below.
constexpr uint32_t kSyncsPerChunk = 2;
constexpr uint32_t kFixedSyncsPerBlock = 4;
// The colChunk search halves the candidate each step (largest first).
constexpr uint32_t kColChunkHalving = 2;

uint32_t CeilDiv(uint32_t val, uint32_t d) { return (val + d - 1) / d; }
uint32_t AlignDown32(uint32_t v) { return v / kAlignCols * kAlignCols; }

// pow2 ceiling of v (v >= 1); mirrors the kernel's Pow2Ceil.
uint32_t Pow2Ceil(uint32_t v) {
  uint32_t p = 32;
  while (p < v) {
    p <<= 1;
  }
  return p;
}

bool GetPlatformInfo(TilingContext *context, uint32_t &aivNum) {
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

bool HasSmoothScale(TilingContext *context) {
  // Same optional-input detection as ChunkGatedDeltaRule's gamma input.
  return context->GetOptionalInputDesc(kSmoothInput) != nullptr &&
         context->GetOptionalInputTensor(kSmoothInput) != nullptr &&
         context->GetOptionalInputShape(kSmoothInput) != nullptr;
}

// Read the optional epsilon attr (idx 0, the only attr), falling back to the
// OpDef default (same pattern as ChunkGatedDeltaRule's scale_value).
void ParseEpsilon(TilingContext *context, float &epsilon) {
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

// Layout detection. FRACTAL_NZ storage of a logical [P..., M, H] fp16
// tensor is [P..., n1, m1t, 16, 16] (column fractals outermost, the 16x16
// fractal row-major [m0, n0]; CANN include pto/common/pto_tile.hpp:
// TileShape2D<Layout::NZ> = Shape<1, cols/C0, rows/16, 16, C0> with
// C0 = 32B/sizeof(T) = 16 for fp16). The gamma length disambiguates: an ND
// row-major operand ends in dim == H, while NZ ends in [.., n1, m1t, 16,
// 16] with n1*16 == H. m always comes from the ORIGIN (logical) shape:
// the NZ storage pads rows up to m1t*16 per prefix, and token_scale is a
// logical [M] ND output that must never be written past the logical end.
// Derives isNz / m1t / m; returns false on any malformed shape.
bool DetectLayout(TilingContext *context, uint64_t nU, uint32_t &isNz, uint32_t &m1t, uint32_t &m) {
  auto x1Shape = context->GetInputShape(kX1Input);
  auto storage = x1Shape->GetStorageShape();
  auto origin = x1Shape->GetOriginShape();
  size_t dimNum = storage.GetDimNum();
  if (dimNum == 0) {
    return false;
  }
  if (dimNum >= kNzN1Offset && storage.GetDim(dimNum - kNzN0Offset) == kNzHalfBlockCols &&
      storage.GetDim(dimNum - kNzM0Offset) == kNzFractalRows &&
      static_cast<uint64_t>(storage.GetDim(dimNum - kNzN1Offset)) * kNzHalfBlockCols == nU) {
    isNz = 1;
    m1t = static_cast<uint32_t>(storage.GetDim(dimNum - kNzM1tOffset));
    if (m1t == 0 || m1t * kNzFractalRows - 1 > kMaxCopyParam) {
      return false;
    }
    // Logical rows from the ORIGIN shape. NZ addressing (NzRowBase) maps a
    // flattened logical row to (row >> 4) inside a single prefix, which is
    // only correct for 2D origins [M, H] (P = 1); with 3D+ origins the
    // flattened row would cross per-prefix pad boundaries and misaddress.
    // Every site edge in the W8A8 graph (and the bench, through GE-inserted
    // TransData) is 2D, so guard rather than generalize. Pad rows
    // (m1t*16 > M) are never processed; storage must still cover all M rows.
    if (origin.GetDimNum() != kNzOriginRank) {
      return false;
    }
    uint64_t rows = static_cast<uint64_t>(origin.GetDim(0));
    if (rows == 0 || rows > 0xFFFFFFFFULL || rows > static_cast<uint64_t>(m1t) * kNzFractalRows) {
      return false;
    }
    m = static_cast<uint32_t>(rows);
  } else {
    // ND row-major: storage == origin, last dim is H.
    if (static_cast<uint64_t>(storage.GetDim(dimNum - 1)) != nU) {
      return false;
    }
    uint64_t rows = 1;
    for (size_t d = 0; d + 1 < dimNum; ++d) {
      rows *= static_cast<uint64_t>(storage.GetDim(d));
    }
    if (rows == 0 || rows > 0xFFFFFFFFULL) {
      return false;
    }
    m = static_cast<uint32_t>(rows);
  }
  return true;
}

// Column contract: H must be a multiple of 32 (32B-aligned copies) and the
// NZ per-row gather/scatter blockCounts (n/16 fp16, n/32 int8) must fit
// the uint16 DataCopyParams fields.
bool ColumnsValid(uint32_t n) {
  return n >= kMinCols && n % kAlignCols == 0 && n / kNzHalfBlockCols <= kMaxCopyParam &&
         n / kNzInt8BlockCols <= kMaxCopyParam;
}

// Graph contract: x2 mirrors x1 (full storage-shape equality, layout
// included).
bool X2MirrorsX1(TilingContext *context) {
  const auto &x1Storage = context->GetInputShape(kX1Input)->GetStorageShape();
  const auto &x2Storage = context->GetInputShape(kX2Input)->GetStorageShape();
  if (x2Storage.GetDimNum() != x1Storage.GetDimNum()) {
    return false;
  }
  for (size_t d = 0; d < x1Storage.GetDimNum(); ++d) {
    if (x2Storage.GetDim(d) != x1Storage.GetDim(d)) {
      return false;
    }
  }
  return true;
}

// The optional smooth is an [H] vector.
bool SmoothShapeValid(TilingContext *context, uint32_t n) {
  auto smoothShape = context->GetOptionalInputShape(kSmoothInput);
  if (smoothShape == nullptr) {
    return false;
  }
  // Match the gamma contract: a per-channel vector, not any tensor whose
  // first dim happens to equal n.
  const auto &smoothStorage = smoothShape->GetStorageShape();
  return smoothStorage.GetDimNum() == 1 && smoothStorage.GetDim(0) == static_cast<int64_t>(n);
}

// Exact mirror of the kernel's InitBuffer footprint for one (R, colChunk)
// candidate, in bytes.
uint32_t BlockBytes(uint32_t n, uint32_t rowBlock, uint32_t colChunk) {
  return rowBlock * (kXChunkTilesBytesPerCol * colChunk + kSlotTilesBytes)  // x1+x2 chunk tiles + 3 slot tiles
         + rowBlock * kRowTilesBytesPerCol * n                              // rRow + actRow fp16 + y int8
         + kFp32ScratchBytesPerCol * Pow2Ceil(colChunk)                     // fp32 scratch (pass1 squares + pass2 act)
         + kTmpHalfBytesPerCol * Pow2Ceil(n)                                // tmpH absmax/pass3 scratch
         + kGsBytesPerCol * n                                               // fused norm constant gs
         + kScalarStageBytes;
}

// Explicit syncs per block ~= 2C + 4 (C = chunks per row); amortized per row
// = that / rowBlock. v3 ranks candidates lexicographically by (chunks,
// syncs): the fewest chunks wins first (tree op count and the exposed
// in-block MTE2 drain both scale with C), syncs only break ties. Returned
// scaled by 8 (rowBlock is a power of two dividing 8) to stay in integers.
uint32_t SyncsPerRow(uint32_t n, uint32_t rowBlock, uint32_t colChunk) {
  uint32_t chunks = CeilDiv(n, colChunk);
  return (kSyncsPerChunk * chunks + kFixedSyncsPerBlock) * kScaleGroupRows / rowBlock;
}

// Pick (rowBlock, colChunk): fewest chunks, then fewest amortized syncs
// (see SyncsPerRow for the ranking), among the candidates that fit the UB
// budget. Returns false when even rowBlock=1 with the smallest chunk
// overflows UB.
bool SelectTile(uint32_t n, uint32_t &bestRowBlock, uint32_t &bestColChunk) {
  uint32_t bestChunks = 0xFFFFFFFF;
  uint32_t bestSyncs = 0xFFFFFFFF;
  for (uint32_t r = 0; r < sizeof(kRowBlockCandidates) / sizeof(kRowBlockCandidates[0]); ++r) {
    uint32_t rowBlock = kRowBlockCandidates[r];
    for (uint32_t colChunk = n; colChunk >= kMinColChunk; colChunk = AlignDown32(colChunk / kColChunkHalving)) {
      if (BlockBytes(n, rowBlock, colChunk) > kUbBudgetBytes) {
        continue;
      }
      // First hit for this rowBlock has the largest colChunk (fewest chunks
      // for this rowBlock); smaller chunks only pay more, so stop scanning
      // once one fits.
      uint32_t chunks = CeilDiv(n, colChunk);
      uint32_t syncs = SyncsPerRow(n, rowBlock, colChunk);
      if (chunks < bestChunks || (chunks == bestChunks && syncs < bestSyncs)) {
        bestChunks = chunks;
        bestSyncs = syncs;
        bestRowBlock = rowBlock;
        bestColChunk = colChunk;
      }
      break;
    }
  }
  return bestRowBlock != 0;
}

uint32_t FusedAddRmsNormSmoothQuantTilingFunc(TilingContext *context) {
  auto x1Shape = context->GetInputShape(kX1Input);
  auto x2Shape = context->GetInputShape(kX2Input);
  auto gammaShape = context->GetInputShape(kGammaInput);
  if (x1Shape == nullptr || x2Shape == nullptr || gammaShape == nullptr) {
    return GRAPH_FAILED;
  }
  uint32_t aivNum = 0;
  if (!GetPlatformInfo(context, aivNum)) {
    return GRAPH_FAILED;
  }

  // gamma is the row-length vector: [H]. It also disambiguates the x1
  // storage layout (ND vs FRACTAL_NZ, the rms_norm_nz pattern).
  auto gammaStorage = gammaShape->GetStorageShape();
  if (gammaStorage.GetDimNum() != 1) {
    return GRAPH_FAILED;
  }
  uint64_t nU = static_cast<uint64_t>(gammaStorage.GetDim(0));

  uint32_t isNz = 0;
  uint32_t m1t = 0;
  uint32_t m = 0;
  if (!DetectLayout(context, nU, isNz, m1t, m)) {
    return GRAPH_FAILED;
  }
  uint32_t n = static_cast<uint32_t>(nU);
  if (!ColumnsValid(n)) {
    return GRAPH_FAILED;
  }
  if (!X2MirrorsX1(context)) {
    return GRAPH_FAILED;
  }

  bool hasSmooth = HasSmoothScale(context);
  if (hasSmooth && !SmoothShapeValid(context, n)) {
    return GRAPH_FAILED;
  }

  float epsilon = kDefaultEpsilon;
  ParseEpsilon(context, epsilon);

  auto td = context->GetTilingData<FusedAddRmsNormSmoothQuantTilingData>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }

  // Pick (rowBlock, colChunk): fewest chunks, then fewest amortized syncs,
  // among the candidates that fit the UB budget.
  uint32_t bestRowBlock = 0;
  uint32_t bestColChunk = 0;
  if (!SelectTile(n, bestRowBlock, bestColChunk)) {
    // Even rowBlock=1 with the smallest chunk overflows UB.
    return GRAPH_FAILED;
  }

  // Row groups (8 rows) are striped over cores; never launch more cores than
  // there are groups. The kernel derives its group range as
  // [blockIdx*G/coreNum, (blockIdx+1)*G/coreNum).
  uint32_t groups = CeilDiv(m, kScaleGroupRows);
  uint32_t blockDim = (groups < aivNum) ? groups : aivNum;
  if (blockDim == 0) {
    blockDim = kMinBlockDim;
  }
  td->coreNum = blockDim;
  td->m = m;
  td->n = n;
  td->colChunk = bestColChunk;
  td->rowBlock = bestRowBlock;
  td->hasSmooth = hasSmooth ? 1 : 0;
  td->epsilon = epsilon;
  td->nzInfo = (isNz != 0 ? 1u : 0u) | (m1t << 1);

  context->SetBlockDim(blockDim);
  context->SetTilingKey(0);

  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // No device workspace: x1/x2 stream straight from the sub-layer outputs.
    ws[0] = 0;
  }
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(FusedAddRmsNormSmoothQuant)
  .Tiling(FusedAddRmsNormSmoothQuantTilingFunc, sizeof(FusedAddRmsNormSmoothQuantTilingData));
