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

// InnerQuantBatchMatmulV3 host tiling (Atlas 300I Duo / Ascend310P3).
//
// Faithful reimplementation of the CANN built-in QuantBatchMatmulV3 310P
// per-token tiling chain -- pp_matmul TilingFunc cost search -> SetBaseOp
// (with the quant-v3 n-spread branch) -> Swizzle -> End -- with ONE
// correction in End, the k0 L1 budget:
//
//   built-in:  k0Max = 131072 / (m0 + n0)
//              (budgets the a-ping and b-ping L1 buffers as ONE shared pool)
//   this op:   k0Max = 131072 / max(m0, n0)
//              (the kernel allocates FOUR separate 131072-int8 L1 buffers
//               a-ping/a-pong/b-ping/b-pong; each k-chunk's a-tile
//               m_round*k_round and b-tile n_round*k_round must each fit
//               ONE buffer, so the per-buffer budget is 131072/max(m0,n0))
//
// On 310P int8 the search caps m0/n0 at 256 (axes0Max = 512/1/2), so
// k0Max >= 512 and the built-in's 256-granularity flooring applies
// unchanged. For the W8A8 suffix_1536 GEMM tiles (m0,n0)=(256,128) or
// (128,256) this doubles k0 256 -> 512 and halves the k-loop round trips
// (kLoop 16 -> 8 at K=4096, 48 -> 24 at K=12288). Bit-exactness is
// preserved by construction: int32 L0C accumulation is exact and
// associative, so re-chunking k only regroups exact integer adds; the
// epilogue math chain is untouched (kernel vendored verbatim).
//
// The four target shapes (batch=1, m=1536) all divide their tiles exactly
// (OP_LOGD at convert):
//   q/o_proj   m=1536 k=4096  n=4096  -> (256,128) k0=512 kLoop=8  Nz/4
//   k/v_proj   m=1536 k=4096  n=1024  -> (128,256) k0=512 kLoop=8  Zn/4
//   gate/up    m=1536 k=4096  n=12288 -> (256,128) k0=512 kLoop=8  Nz/4
//   down_proj  m=1536 k=12288 n=4096  -> (128,256) k0=512 kLoop=24 Nz/2
// (tile choice reproduced by the search below; listed for reference. All
// four pick max(m0,n0)=256, hence k0=512 on every target shape.)

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "inner_quant_batch_matmul_v3_utils.h"   // NOLINT(build/include_subdir)
#include "inner_quant_batch_matmul_v3_tiling.h"  // NOLINT(build/include_subdir)

using namespace ge;    // NOLINT(build/namespaces)
using namespace gert;  // NOLINT(build/namespaces)

namespace {

// Parses up to maxCount comma-separated unsigned values (same acceptance as
// the "%lu,%lu,..." scan previously used); returns how many were parsed.
int ParseUint64List(const char *text, uint64_t *out, int maxCount) {
  const char *cursor = text;
  int count = 0;
  while (count < maxCount) {
    while (*cursor == ' ' || *cursor == '\t') {
      ++cursor;
    }
    if (*cursor < '0' || *cursor > '9') {
      break;
    }
    char *end = nullptr;
    const uint64_t value = std::strtoull(cursor, &end, 10);
    if (end == cursor) {
      break;
    }
    out[count] = value;
    ++count;
    cursor = end;
    if (count < maxCount) {
      if (*cursor != ',') {
        break;
      }
      ++cursor;
    }
  }
  return count;
}

// Input indices (mirror of the built-in entry and the op def port order).
constexpr uint32_t kX1 = 0;
constexpr uint32_t kX2 = 1;
constexpr uint32_t kScale = 2;
constexpr uint32_t kOffset = 3;
constexpr uint32_t kBias = 4;
constexpr uint32_t kPertokenScale = 5;

// Matmul rank layout: [..., M, K] x [..., N, K] -> [..., M, N].
constexpr size_t kMatmulRankTail = 2;
// int8 FRACTAL_NZ requires the k axis to be a multiple of 32 elements.
constexpr uint64_t kInt8NzBlock = 32;
// IQBMM_FORCE_TILING: minimum (m0,n0,k0) and full (.., swizzleDirect, swizzleCount) field counts.
constexpr int kForceTilingMinFields = 3;
constexpr int kForceTilingFullFields = 5;
// bias accepts [n] or [b, 1, n].
constexpr size_t kBiasBatchRank = 3;
// Non-zero workspace floor (see WriteTilingData).
constexpr size_t kMinWorkspaceBytes = 32;
// N-spread heuristic trigger: coreLoop % coreNum < 3/4 * coreNum.
constexpr uint64_t kNSpreadNumerator = 3;
constexpr uint64_t kNSpreadDenominator = 4;
// Attr indices follow the def declaration order: quant_dtype (the built-in's
// "dtype" attr renamed -- a def attr literally named "dtype" breaks CANN 9.2
// op_build's ops-info generation, see the def comment), transpose_x1,
// transpose_x2, group_size.
constexpr uint32_t kAttrDtype = 0;
constexpr uint32_t kAttrTransX1 = 1;
constexpr uint32_t kAttrTransX2 = 2;
constexpr uint32_t kAttrGroupSize = 3;
constexpr uint32_t kDimN = 1;

// Constants mirrored from pp_matmul_common_tiling.h and the arch20 kernel
// (cann_ops/ops-nn/matmul/transpose_batch_mat_mul/op_host/op_tiling/
// pp_matmul_common_tiling.h; .../quant_batch_matmul_v3/op_kernel/arch20/
// quant_batch_matmul_v3_pertoken_arch20.h).
constexpr uint64_t kBlockSize16 = 16;          // BLOCK_SIZE
constexpr uint64_t kBaseBlockStep = 2;         // BASE_BLOCK_STEP
constexpr uint64_t kAxesAlignSize = 512;       // AXES_ALIGN_SIZE
constexpr uint64_t kFp32Size = 4;              // FP32_SIZE
constexpr uint64_t kL1PpBuffLenInt8 = 131072;  // L1_PINGPONG_BUFFER_LEN_INT8 (per L1 buffer)
constexpr uint64_t kUbLimitPertoken = 131072;  // UB_LIMIT_SIZE_PERTOKEN_ARCH20
constexpr uint64_t kAlign256 = 256;            // k0Init / k0 flooring granularity
constexpr uint64_t kL2BwCoef = 5;              // HardwareInfo l2/hbm bw = 5/1 (defaults)
constexpr uint64_t kInt8Size = 1;              // sizeInDtype for int8

// CeilDiv/RoundUp with the built-in's overflow guard semantics.
uint64_t CeilDiv(uint64_t dividend, uint64_t divisor) {
  if (divisor == 0 || dividend + divisor - 1 < dividend) {
    return dividend;
  }
  return (dividend + divisor - 1) / divisor;
}
uint64_t RoundUp(uint64_t num, uint64_t rnd) {
  if (rnd == 0 || num + rnd - 1 < num) {
    return num;
  }
  return (num + rnd - 1) / rnd * rnd;
}

// Scratch mirror of PpMatmulDefaultTilingData (only the fields this tiling
// chain computes; compress/splitk/shuffle fields are constant-off in this
// contract and are not carried).
struct TilingCalc {
  uint64_t batchSize = 0;
  uint64_t m = 0;
  uint64_t k = 0;
  uint64_t n = 0;
  uint64_t m0 = 0;
  uint64_t k0 = 0;
  uint64_t n0 = 0;
  uint64_t mLoop = 1;
  uint64_t kLoop = 1;
  uint64_t nLoop = 1;
  uint64_t coreLoop = 1;
  uint64_t blockDim = 1;
  uint64_t swizzleDirect = 0;
  uint64_t swizzleCount = 1;
};

// Built-in SetBaseOp (pp_matmul_default.cpp:66-103) for the 310P path:
// isNpuArch2002=true, so the !arch2002 m-spread branch never runs; the
// quant-v3 n-spread branch is kept verbatim with its guards fixed by this
// op's contract (isQuantBatchMatmulV3=true, tilingK=0, transB=true).
void SetBaseOp(TilingCalc &t, uint64_t coreNum, uint64_t l0cSize, uint64_t mBase, uint64_t nBase,
               bool allowNSpread = true) {
  t.m0 = mBase;
  t.n0 = nBase;
  t.mLoop = CeilDiv(t.m, t.m0);
  t.nLoop = CeilDiv(t.n, t.n0);
  t.coreLoop = t.batchSize * t.mLoop * t.nLoop;
  // quant-v3 branch: single m-row-block GEMMs spread n-tiles across cores
  // when most cores would otherwise idle (coreLoop % coreNum < 3/4 * coreNum).
  // The forced-tiling path passes allowNSpread=false: a forced tile
  // must not be second-guessed by the spread heuristic.
  if (allowNSpread && t.mLoop == 1 && t.coreLoop % coreNum < coreNum / kNSpreadDenominator * kNSpreadNumerator) {
    uint64_t x = CeilDiv(t.n, coreNum);
    uint64_t y = CeilDiv(x, kAlign256);
    nBase = RoundUp(CeilDiv(x, y), kBlockSize16);
    if (mBase * nBase * kFp32Size < l0cSize) {
      t.n0 = nBase;
      t.nLoop = CeilDiv(t.n, t.n0);
      t.coreLoop = t.batchSize * t.nLoop;
    }
  }
  t.blockDim = std::min(t.coreLoop, coreNum);
}

// Built-in CostFunc (pp_matmul_common_tiling.h:139-161) with this op's
// fixed MatMulInfo: isInt8 (sizeInDtype=1), bwCoef = l2BandWidth/hbmBandWidth
// = 5/1 (the HardwareInfo defaults -- GetHardwareInfo never overrides them).
float CostFunc(uint64_t coreNum, uint64_t l2Size, const TilingCalc &s, uint64_t m0, uint64_t n0) {
  float aCoef = 1.0f;
  float bCoef = 1.0f;
  const float bwCoef = static_cast<float>(kL2BwCoef);
  uint64_t mLoop = CeilDiv(s.m, m0);
  uint64_t nLoop = CeilDiv(s.n, n0);
  if (mLoop == 0 || nLoop == 0) {
    return 1.0f;
  }
  uint64_t coreNeed = s.batchSize * mLoop * nLoop;
  uint64_t blockDim = std::min(coreNeed, coreNum);
  uint64_t mOnce = blockDim < nLoop ? m0 : blockDim / nLoop * m0;
  uint64_t nOnce = blockDim < nLoop ? coreNum * n0 : s.n;
  if (mOnce * s.k * kInt8Size > l2Size) {
    aCoef = bwCoef;
  }
  if (nOnce * s.k * kInt8Size > l2Size) {
    bCoef = bwCoef;
  }
  return 1.0f / (aCoef * static_cast<float>(n0)) + 1.0f / (bCoef * static_cast<float>(m0));
}

// Built-in Swizzle (pp_matmul_common_tiling.h:208-243), verbatim: float
// arithmetic, <= on the Nz branch / < on the Zn branch, and swizzleDirect
// keeps the LAST inspected i's direction even when its cost was not better
// (quirk included -- the kernel dispatch must see the same value the
// built-in would have produced).
void Swizzle(TilingCalc &t) {
  uint64_t swizzleDirect = 0;
  uint64_t swizzleCount = 1;
  float m0 = static_cast<float>(t.m0);
  float n0 = static_cast<float>(t.n0);
  float m = static_cast<float>(t.m);
  float k = static_cast<float>(t.k);
  float n = static_cast<float>(t.n);
  float mincost = m * k + k * n;
  for (uint32_t i = 1; i <= t.blockDim; ++i) {
    int32_t c = static_cast<int32_t>((t.blockDim + i - 1) / i);
    float cost;
    if (i * n0 + m < m0 * c + n) {  // B0 + A < A0 + B
      swizzleDirect = 1;            // Nz
      cost = n0 * i + m0 * c;
      if (cost <= mincost) {
        mincost = cost;
        swizzleCount = i;
      }
    } else {
      swizzleDirect = 0;  // Zn
      cost = m0 * i + n0 * c;
      if (cost < mincost) {
        mincost = cost;
        swizzleCount = i;
      }
    }
  }
  t.swizzleDirect = swizzleDirect;
  t.swizzleCount = swizzleCount;
}

// Built-in End (pp_matmul_default.cpp:105-138) for isNpuArch2002 (310P),
// with the L1 budget CORRECTED -- see the file header. Everything else is
// verbatim: k0Init=256, the 256/16 flooring, kLoop.
void EndK0(TilingCalc &t) {
  uint64_t k0Max = kL1PpBuffLenInt8 / std::max(t.m0, t.n0);
  t.k0 = k0Max < kAlign256 ? k0Max / kBlockSize16 * kBlockSize16 : k0Max / kAlign256 * kAlign256;
  t.kLoop = CeilDiv(t.k, t.k0);
}

// Built-in TilingFunc search (pp_matmul_common_tiling.h:164-206),
// specialized to this op's fixed MatMulInfo: isInt8, isCompress=false,
// tilingN=1, isPertokenArch20=true. PRI_FLAG = (m >= n): priAxes0 walks the
// larger side. The candidate grid is the x2 ladder from 16 with L0C and
// pertoken-UB caps; the first candidate (16,16) always clears costMin=1, so
// the search always adopts a tile.
void SearchTile(TilingCalc &t, uint64_t coreNum, uint64_t l2Size, uint64_t l0cSize) {
  const bool priFlag = t.m >= t.n;
  uint64_t roundBase = static_cast<uint64_t>(
    pow(2, ceil(log(static_cast<double>(CeilDiv(priFlag ? t.n : t.m, kBlockSize16))))) * kBlockSize16);
  uint64_t priAxes = RoundUp(priFlag ? t.m : t.n, kBlockSize16);
  uint64_t axes = RoundUp(priFlag ? t.n : t.m, roundBase);
  uint64_t axes0Max = kAxesAlignSize / kInt8Size;  // 512 for int8
  axes0Max /= 2;                                   // int8 on 310P/910: halved to 256
  const uint64_t n0TilingLimit = kAxesAlignSize;   // GetN0TilingLimit, non-compress 310P
  float costMin = 1.0f;
  for (uint64_t priAxes0 = kBlockSize16; priAxes0 <= priAxes && priAxes0 <= axes0Max; priAxes0 *= kBaseBlockStep) {
    for (uint64_t axes0 = kBlockSize16; axes0 <= axes && axes0 <= axes0Max; axes0 *= kBaseBlockStep) {
      uint64_t basicBlockSize = priAxes0 * axes0 * kFp32Size;
      if (basicBlockSize > l0cSize) {
        continue;
      }
      // IsExceedTilingLimit (int8, 310P pertoken): the 512 n0 limit is
      // vacuous under axes0Max=256; the pertoken UB cap is the binding one.
      if ((priFlag && axes0 > n0TilingLimit) || (!priFlag && priAxes0 > n0TilingLimit) ||
          basicBlockSize > kUbLimitPertoken) {
        continue;
      }
      uint64_t m0 = priFlag ? priAxes0 : axes0;
      uint64_t n0 = priFlag ? axes0 : priAxes0;
      float cost = CostFunc(coreNum, l2Size, t, m0, n0);
      if (cost >= costMin) {
        continue;
      }
      costMin = cost;
      SetBaseOp(t, coreNum, l0cSize, m0, n0);
    }
  }
}

// Env-forced tiling -- the on-device override (per-shape tiling exploration
// without rebuilds): IQBMM_FORCE_TILING="m0,n0,k0[,swizzleDirect,swizzleCount]"
// is read at convert time, validated against the kernel-legal caps
// (axes0<=256 int8, m0/n0 16-multiples, L0C budget, L1 per-buffer k0 budget)
// and applied verbatim. Wide tiles (m0*n0*4 in (128KB, 256KB], e.g. (256,256)
// for the AI=128 lever) are legal ONLY here and via the shape-keyed override
// table: the kernel's column-pass epilogue drains them in
// <=128-column passes sized to the 128KB work region, while the search keeps
// the built-in's stricter kUbLimitPertoken parity so default tilings stay
// bit-compatible with the built-in's picks. Rationale: the built-in cost
// model is a coarse L2-span heuristic; steady-state efficiency
// spans 39.5-46 TMAC/s across the four deployment sites, so deployed tilings
// are chosen per shape. Invalid values fail the convert LOUDLY -- an invalid
// override must never silently fall back to the search tiling.
enum class ForcedTiling { kNone, kApplied, kInvalid };

// Shared validate+apply for the two override channels (the env override
// above and the shape-keyed table in the tiling func): checks the tile
// against the kernel-legal caps (axes0<=256 int8, m0/n0 16-multiples, L0C
// budget, L1 per-buffer k0 budget, swizzle range vs the recomputed
// blockDim) and rebuilds every derived field -- mLoop/nLoop/coreLoop/
// blockDim via SetBaseOp with the n-spread branch off, swizzle either
// explicit or derived, then k0/kLoop. `desc` names the source in errors
// (the env string or the table entry). Loud failure by design: an illegal
// override must never silently measure the search's tiling instead.
bool ApplyForcedTile(TilingCalc &t, uint64_t coreNum, uint64_t l0cSize, uint64_t m0, uint64_t n0, uint64_t k0,
                     bool hasSwizzle, uint64_t swD, uint64_t swC, const char *desc, const char *opName) {
  const uint64_t axes0Cap = kAxesAlignSize / kInt8Size / 2;  // 256: int8 310P axes0Max (search parity)
  if (m0 < kBlockSize16 || n0 < kBlockSize16 || m0 % kBlockSize16 != 0 || n0 % kBlockSize16 != 0) {
    OP_LOGE(opName, "forced tiling '%s': m0/n0 must be multiples of 16 (>=16)", desc);
    return false;
  }
  if (m0 > axes0Cap || n0 > axes0Cap) {
    OP_LOGE(opName, "forced tiling '%s': m0/n0 cap %lu (int8 310P axes0Max)", desc, axes0Cap);
    return false;
  }
  if (m0 * n0 * kFp32Size > l0cSize) {
    OP_LOGE(opName, "forced tiling '%s': tile %lux%lu exceeds the L0C budget %lu", desc, m0, n0,
            static_cast<uint64_t>(l0cSize));
    return false;
  }
  if (k0 == 0 || k0 % kInt8NzBlock != 0) {
    OP_LOGE(opName, "forced tiling '%s': k0 must be a positive multiple of 32 (int8 NZ block)", desc);
    return false;
  }
  const uint64_t k0Max = kL1PpBuffLenInt8 / std::max(m0, n0);
  if (k0 > k0Max) {
    OP_LOGE(opName, "forced tiling '%s': k0 %lu exceeds the L1 per-buffer budget %lu", desc, k0, k0Max);
    return false;
  }
  // A forced tile must not be second-guessed by the N-spread heuristic.
  SetBaseOp(t, coreNum, l0cSize, m0, n0, false);
  if (hasSwizzle) {
    if (swD > 1 || swC < 1 || swC > t.blockDim) {
      OP_LOGE(opName, "forced tiling '%s': swizzle %lu/%lu out of range (direct<=1, 1<=count<=blockDim=%lu)", desc, swD,
              swC, t.blockDim);
      return false;
    }
    t.swizzleDirect = swD;
    t.swizzleCount = swC;
  } else {
    Swizzle(t);
  }
  t.k0 = k0;
  t.kLoop = CeilDiv(t.k, t.k0);
  return true;
}

ForcedTiling TryEnvForcedTiling(TilingCalc &t, uint64_t coreNum, uint64_t l0cSize, const char *opName) {
  const char *env = getenv("IQBMM_FORCE_TILING");
  if (env == nullptr || env[0] == '\0') {
    return ForcedTiling::kNone;
  }
  uint64_t m0 = 0;
  uint64_t n0 = 0;
  uint64_t k0 = 0;
  uint64_t swD = 0;
  uint64_t swC = 0;
  uint64_t fields[kForceTilingFullFields] = {0, 0, 0, 0, 0};
  const int parsed = ParseUint64List(env, fields, static_cast<int>(kForceTilingFullFields));
  m0 = fields[0];
  n0 = fields[1];
  k0 = fields[2];
  swD = fields[3];
  swC = fields[4];
  if (parsed < kForceTilingMinFields) {
    OP_LOGE(opName, "IQBMM_FORCE_TILING '%s' malformed (want m0,n0,k0[,swizzleDirect,swizzleCount])", env);
    return ForcedTiling::kInvalid;
  }
  if (!ApplyForcedTile(t, coreNum, l0cSize, m0, n0, k0, parsed >= kForceTilingFullFields, swD, swC, env, opName)) {
    return ForcedTiling::kInvalid;
  }
  OP_LOGD(opName, "InnerQBMMV3 tiling FORCED (env): m0=%lu n0=%lu k0=%lu kLoop=%lu swizzle=%lu/%lu blockDim=%lu", m0,
          n0, k0, t.kLoop, t.swizzleDirect, t.swizzleCount, t.blockDim);
  return ForcedTiling::kApplied;
}

// Contract validation + tiling for the fixed 310P NZ per-token combo.
// Validation failures mirror the built-in's error classes and add explicit
// guards where the built-in would read OOB silently (scale/pertoken/bias
// element counts).

static bool GetPlatformAndMem(TilingContext *context, const char *opName, uint64_t &coreNum, uint64_t &l2Size,
                              uint64_t &l0cSize, size_t &sysWs) {
  auto platformInfo = context->GetPlatformInfo();
  if (platformInfo == nullptr) {
    OP_LOGE(opName, "platform info is nullptr");
    return GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(platformInfo);
  if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND310P) {
    OP_LOGE(opName, "InnerQuantBatchMatmulV3 targets Ascend310P only (vendored dav-m200 kernel)");
    return GRAPH_FAILED;
  }
  coreNum = static_cast<uint64_t>(platform.GetCoreNumAic());
  l2Size = 0;
  l0cSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
  if (coreNum == 0 || l0cSize == 0) {
    OP_LOGE(opName, "invalid platform: aic core num or L0_C size is zero");
    return GRAPH_FAILED;
  }

  sysWs = platform.GetLibApiWorkSpaceSize();
  return true;
}

static bool ParseMatmulShapes(TilingContext *context, const char *opName, uint64_t &batch, uint64_t &m, uint64_t &k,
                              uint64_t &n) {
  // Shapes: x1 origin [.., M, K], x2 origin [.., N, K] (transpose_x2=true),
  // leading batch dims flattened (built-in GetShapeAttrsInfo semantics).
  auto x1Shape = context->GetInputShape(kX1);
  auto x2Shape = context->GetInputShape(kX2);
  if (x1Shape == nullptr || x2Shape == nullptr) {
    OP_LOGE(opName, "input shape is nullptr");
    return GRAPH_FAILED;
  }
  const auto &aOrigin = x1Shape->GetOriginShape();
  const auto &bOrigin = x2Shape->GetOriginShape();
  size_t aDims = aOrigin.GetDimNum();
  size_t bDims = bOrigin.GetDimNum();
  if (aDims < kMatmulRankTail || bDims < kMatmulRankTail) {
    OP_LOGE(opName, "x1 and x2 must have at least 2 dims");
    return GRAPH_FAILED;
  }
  size_t batchDims = aDims - 2;
  if (bDims - kMatmulRankTail != batchDims) {
    OP_LOGE(opName, "x1 and x2 must have the same batch dim count");
    return GRAPH_FAILED;
  }
  batch = 1;
  for (size_t d = 0; d < batchDims; ++d) {
    if (aOrigin.GetDim(d) != bOrigin.GetDim(d) || aOrigin.GetDim(d) <= 0) {
      OP_LOGE(opName, "x1/x2 batch dims must agree and be positive");
      return GRAPH_FAILED;
    }
    batch *= static_cast<uint64_t>(aOrigin.GetDim(d));
  }
  int64_t mI = aOrigin.GetDim(aDims - 2);
  int64_t kI = aOrigin.GetDim(aDims - 1);
  int64_t nI = bOrigin.GetDim(bDims - 2);
  int64_t kBI = bOrigin.GetDim(bDims - 1);
  if (mI <= 0 || kI <= 0 || nI <= 0) {
    OP_LOGE(opName, "static shapes required (m/k/n must be positive)");
    return GRAPH_FAILED;
  }
  if (kBI != kI) {
    OP_LOGE(opName, "x2 last dim (%ld) must equal x1 k (%ld) for transpose_x2=true", kBI, kI);
    return GRAPH_FAILED;
  }
  if (kI % static_cast<int64_t>(kInt8NzBlock) != 0) {
    OP_LOGE(opName, "k (%ld) must be a multiple of 32 (int8 NZ storage)", kI);
    return GRAPH_FAILED;
  }
  m = static_cast<uint64_t>(mI);
  k = static_cast<uint64_t>(kI);
  n = static_cast<uint64_t>(nI);
  return true;
}

static bool CheckTensorContract(TilingContext *context, const char *opName) {
  // Dtypes/storage: the def binds the contract; re-check for explicit
  // errors. scale must be fp32 (arch20 path guard in the built-in dispatch).
  auto x1Desc = context->GetInputDesc(kX1);
  auto x2Desc = context->GetInputDesc(kX2);
  auto scaleDesc = context->GetInputDesc(kScale);
  if (x1Desc == nullptr || x2Desc == nullptr || scaleDesc == nullptr) {
    OP_LOGE(opName, "input desc is nullptr");
    return GRAPH_FAILED;
  }
  if (x1Desc->GetDataType() != ge::DT_INT8 || x2Desc->GetDataType() != ge::DT_INT8) {
    OP_LOGE(opName, "x1/x2 must be int8");
    return GRAPH_FAILED;
  }
  if (scaleDesc->GetDataType() != ge::DT_FLOAT) {
    OP_LOGE(opName, "scale must be float32 (per-token arch20 path guard)");
    return GRAPH_FAILED;
  }
  if (x1Desc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ || x2Desc->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGE(opName, "x1/x2 storage must be FRACTAL_NZ");
    return GRAPH_FAILED;
  }

  // Attrs (def order: quant_dtype, transpose_x1, transpose_x2, group_size).
  auto attrs = context->GetAttrs();
  if (attrs == nullptr) {
    OP_LOGE(opName, "attrs is nullptr");
    return GRAPH_FAILED;
  }
  int64_t dtypeAttr = *(attrs->GetAttrPointer<int64_t>(kAttrDtype));
  if (dtypeAttr != static_cast<int64_t>(ge::DT_FLOAT16)) {
    OP_LOGE(opName, "quant_dtype attr (%ld) must be float16 (y is fixed fp16 by the def)", dtypeAttr);
    return GRAPH_FAILED;
  }
  // Int-encoded transposes (def declares Int 0/1; the CANN ONNX plugin
  // convention: i == 1 -> true). Only the NZ per-token combo is vendored.
  int64_t transA = *(attrs->GetAttrPointer<int64_t>(kAttrTransX1));
  int64_t transB = *(attrs->GetAttrPointer<int64_t>(kAttrTransX2));
  if (transA != 0 || transB != 1) {
    OP_LOGE(opName, "only transpose_x1=0, transpose_x2=1 is supported (NZ per-token combo), got %ld/%ld", transA,
            transB);
    return GRAPH_FAILED;
  }
  int64_t groupSize = *(attrs->GetAttrPointer<int64_t>(kAttrGroupSize));
  if (groupSize != 0) {
    OP_LOGE(opName, "group_size must be 0 (no group quant on this path)");
    return GRAPH_FAILED;
  }

  // pertoken_scale is required (the vendored epilogue always multiplies it).

  return true;
}

static bool ResolveScaleBiasContract(TilingContext *context, const char *opName, uint64_t batch, uint64_t m, uint64_t n,
                                     bool &withBias, bool &biasWithBatch, uint32_t &biasDtype) {
  auto pertokenDesc = context->GetOptionalInputDesc(kPertokenScale);
  if (pertokenDesc == nullptr) {
    OP_LOGE(opName, "pertoken_scale is required (per-token variant; use the built-in op otherwise)");
    return GRAPH_FAILED;
  }
  if (pertokenDesc->GetDataType() != ge::DT_FLOAT) {
    OP_LOGE(opName, "pertoken_scale must be float32");
    return GRAPH_FAILED;
  }
  uint64_t pertokenNumel = 1;
  const auto &ptOrigin = context->GetOptionalInputShape(kPertokenScale)->GetOriginShape();
  for (size_t d = 0; d < ptOrigin.GetDimNum(); ++d) {
    pertokenNumel *= static_cast<uint64_t>(ptOrigin.GetDim(d));
  }
  if (pertokenNumel != m) {
    OP_LOGE(opName, "pertoken_scale numel (%lu) must equal m (%lu)", pertokenNumel, m);
    return GRAPH_FAILED;
  }
  uint64_t scaleNumel = 1;
  const auto &scOrigin = context->GetInputShape(kScale)->GetOriginShape();
  for (size_t d = 0; d < scOrigin.GetDimNum(); ++d) {
    scaleNumel *= static_cast<uint64_t>(scOrigin.GetDim(d));
  }
  if (scaleNumel != n) {
    OP_LOGE(opName, "scale numel (%lu) must equal n (%lu)", scaleNumel, n);
    return GRAPH_FAILED;
  }

  // Bias/offset resolution, mirroring the built-in GetShapeAttrsInfo
  // (quant_batch_matmul_v3_tiling_arch20.cpp:90-110): bias is honored only
  // when offset is absent (the per-token formula set drops both terms);
  // int32 and fp32 bias share one static bin, dispatched by biasDtype.
  withBias = false;
  biasWithBatch = false;
  biasDtype = static_cast<uint32_t>(ge::DT_INT32);
  bool hasOffset = context->GetOptionalInputDesc(kOffset) != nullptr;
  if (hasOffset) {
    OP_LOGW(opName, "offset is present: ignored by the per-token path, bias dropped (built-in combo semantics)");
  }
  auto biasDesc = context->GetOptionalInputDesc(kBias);
  if (biasDesc != nullptr && !hasOffset) {
    auto biasDt = biasDesc->GetDataType();
    if (biasDt != ge::DT_INT32) {
      // float bias is a built-in-only luxury: a two-value DataType list in
      // the def kills CANN 9.2 op_build's ops-info generation for the whole
      // op (see the def comment), so this def admits int32 only.
      OP_LOGE(opName, "bias dtype must be int32 (arch20 pertoken mode; def restricts to int32)");
      return GRAPH_FAILED;
    }
    withBias = true;
    biasDtype = static_cast<uint32_t>(biasDt);
    const auto &biasOrigin = context->GetOptionalInputShape(kBias)->GetOriginShape();
    if (biasOrigin.GetDimNum() == 1) {
      biasWithBatch = false;
    } else if (biasOrigin.GetDimNum() == kBiasBatchRank) {
      biasWithBatch = true;
    } else {
      OP_LOGW(opName, "bias only supports [n] or [b, 1, n]");
    }
    uint64_t biasNumel = 1;
    for (size_t d = 0; d < biasOrigin.GetDimNum(); ++d) {
      biasNumel *= static_cast<uint64_t>(biasOrigin.GetDim(d));
    }
    uint64_t biasNeed = biasWithBatch ? batch * n : n;
    if (biasNumel != biasNeed) {
      OP_LOGE(opName, "bias numel (%lu) must equal %lu", biasNumel, biasNeed);
      return GRAPH_FAILED;
    }
  }
  return true;
}

// Empirical shape-keyed tiling overrides: per-tier winners that the built-in
// cost search cannot express -- wide tiles (m0*n0*4 in (128KB, 256KB], legal only via
// the override channels: the kernel's column-pass epilogue drains them in
// <=128-column passes sized to the 128KB work region, while the search keeps the
// built-in's stricter kUbLimitPertoken parity) and swizzle pairings the derived
// Swizzle() misses. Entry: {b, m, k, n, m0, n0, k0, swD, swC}; a 0 tile field
// keeps the search pick for that field, and swC=0 derives the swizzle (swD is
// then ignored -- 0 is a legal direction value, so the count is the sentinel).
// Nonzero tile entries route through ApplyForcedTile, the same validation +
// derived-field recompute as the env channel, so a bad entry fails the convert
// loudly instead of silently corrupting the kernel contract. Bit-identity holds
// by construction for every entry kind: int32 L0C accumulation is exact, k0
// re-chunking only regroups integer adds, swizzle is pure work-item scheduling.
struct TilingOverrideEntry {
  uint64_t b;
  uint64_t m;
  uint64_t k;
  uint64_t n;
  uint64_t m0;
  uint64_t n0;
  uint64_t k0;
  uint64_t swD;
  uint64_t swC;
};

static const TilingOverrideEntry kTilingOverrides[] = {
  // Wide tile (m0,n0,k0) = (256,256,512) per tier; tiers are keyed by
  // (batch, m, k, n) and sites without an entry keep the search pick.
  // The swizzle pairing follows the B-matrix L2 span (Nz for the narrow B,
  // Zn for the wide ones); swC=0 entries derive the direction.
  // seqlen 3072: wide tile on all four sites; swizzle follows the B-matrix L2 span (S1 Nz/4, S2 Zn/2, S3/S4 Zn/4).
  {1, 1536, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 1536, 4096, 1024, 256, 256, 512, 0, 2},
  {1, 1536, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 1536, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 2560: wide tile on S1/S3/S4; S2 keeps the search tile (forced shape leaves a 2.5/core imbalance).
  {1, 1280, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 1280, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 1280, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 2304: S2/S3 keep the search tile (per-tier L2 sensitivity); S1/S4 take the wide tile.
  {1, 1152, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 1152, 4096, 1024, 256, 256, 512, 0, 2},
  {1, 1152, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 2048: wide tile on all four sites.
  {1, 1024, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 1024, 4096, 1024, 256, 256, 512, 0, 2},
  {1, 1024, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 1024, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 1792: wide tile on all four sites; S2 pairs Zn/3.
  {1, 896, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 896, 4096, 1024, 256, 256, 512, 0, 3},
  {1, 896, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 896, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 1536: S2 keeps the search tile (forced 1.5/core leaves the second wave half idle); S1/S3/S4 take the wide
  // tile.
  {1, 768, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 768, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 768, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 1280: S2 keeps the search tile (forced 1.5/core vs base 3.0/core); S1/S3/S4 take the wide tile.
  {1, 640, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 640, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 640, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 1024: S2 pairs Zn/3; wide tile on all four sites.
  {1, 512, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 512, 4096, 1024, 256, 256, 512, 0, 3},
  {1, 512, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 512, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 768: S2/S3 keep the search tile; S1/S4 take the wide tile.
  {1, 384, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 384, 4096, 1024, 256, 256, 512, 0, 4},
  {1, 384, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 384, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 512: only S3/S4 take the wide tile (mLoop=1 domain; forced S1/S2 shapes collapse blockDim).
  {1, 256, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 256, 12288, 4096, 256, 256, 512, 0, 4},
  // seqlen 3584: wide tile on S1/S3/S4; S2 keeps the search tile (3.5/core imbalance).
  {1, 1792, 4096, 4096, 256, 256, 512, 1, 4},
  {1, 1792, 4096, 12288, 256, 256, 512, 0, 4},
  {1, 1792, 12288, 4096, 256, 256, 512, 0, 4},
};

static bool ApplyShapeKeyedOverride(TilingCalc &t, uint64_t coreNum, uint64_t l0cSize, const char *opName) {
  for (const auto &ov : kTilingOverrides) {
    if (t.batchSize != ov.b || t.m != ov.m || t.k != ov.k || t.n != ov.n) {
      continue;
    }
    const bool tileEntry = ov.m0 != 0 || ov.n0 != 0 || ov.k0 != 0;
    const bool swEntry = ov.swC != 0;
    if (tileEntry) {
      if (ov.m0 == 0 || ov.n0 == 0 || ov.k0 == 0) {
        OP_LOGE(opName, "shape-keyed override [b=%lu,m=%lu,k=%lu,n=%lu]: partial tile %lu/%lu/%lu (all-or-none)",
                static_cast<uint64_t>(t.batchSize), static_cast<uint64_t>(t.m), static_cast<uint64_t>(t.k),
                static_cast<uint64_t>(t.n), static_cast<uint64_t>(ov.m0), static_cast<uint64_t>(ov.n0),
                static_cast<uint64_t>(ov.k0));
        return GRAPH_FAILED;
      }
      const std::string desc =
        "shape-keyed " + std::to_string(ov.m0) + "x" + std::to_string(ov.n0) + "x" + std::to_string(ov.k0);
      if (!ApplyForcedTile(t, coreNum, l0cSize, ov.m0, ov.n0, ov.k0, swEntry, ov.swD, ov.swC, desc.c_str(), opName)) {
        return GRAPH_FAILED;
      }
    } else if (swEntry) {
      if (ov.swD > 1 || ov.swC > t.blockDim) {
        OP_LOGE(opName,
                "shape-keyed override [b=%lu,m=%lu,k=%lu,n=%lu]: swizzle %lu/%lu out of range (direct<=1, "
                "count<=blockDim=%lu)",
                static_cast<uint64_t>(t.batchSize), static_cast<uint64_t>(t.m), static_cast<uint64_t>(t.k),
                static_cast<uint64_t>(t.n), static_cast<uint64_t>(ov.swD), static_cast<uint64_t>(ov.swC),
                static_cast<uint64_t>(t.blockDim));
        return GRAPH_FAILED;
      }
      t.swizzleDirect = ov.swD;
      t.swizzleCount = ov.swC;
    }
    OP_LOGD(opName,
            "InnerQBMMV3 tiling OVERRIDE (shape-keyed): tile %lux%lux%lu swizzle %lu/%lu for "
            "[b=%lu,m=%lu,k=%lu,n=%lu]",
            static_cast<uint64_t>(ov.m0), static_cast<uint64_t>(ov.n0), static_cast<uint64_t>(ov.k0),
            static_cast<uint64_t>(t.swizzleDirect), static_cast<uint64_t>(t.swizzleCount),
            static_cast<uint64_t>(t.batchSize), static_cast<uint64_t>(t.m), static_cast<uint64_t>(t.k),
            static_cast<uint64_t>(t.n));
    break;
  }
  return true;
}

static uint32_t WriteTilingData(TilingContext *context, const TilingCalc &t, uint32_t biasDtype, bool withBias,
                                bool biasWithBatch, size_t sysWs, const char *opName) {
  auto td = context->GetTilingData<QuantMatmulPertokenTilingDataArch20>();
  if (td == nullptr) {
    return GRAPH_FAILED;
  }
  td->batchSize = static_cast<uint32_t>(t.batchSize);
  td->m = static_cast<uint32_t>(t.m);
  td->k = static_cast<uint32_t>(t.k);
  td->n = static_cast<uint32_t>(t.n);
  td->m0 = static_cast<uint32_t>(t.m0);
  td->k0 = static_cast<uint32_t>(t.k0);
  td->n0 = static_cast<uint32_t>(t.n0);
  td->mLoop = static_cast<uint32_t>(t.mLoop);
  td->kLoop = static_cast<uint32_t>(t.kLoop);
  td->nLoop = static_cast<uint32_t>(t.nLoop);
  td->coreLoop = static_cast<uint32_t>(t.coreLoop);
  td->blockDim = static_cast<uint32_t>(t.blockDim);
  td->swizzleDirect = static_cast<uint32_t>(t.swizzleDirect);
  td->swizzleCount = static_cast<uint32_t>(t.swizzleCount);
  td->biasDtype = biasDtype;
  td->withBias = withBias;
  td->biasWithBatch = biasWithBatch;
  OP_LOGD(opName,
          "InnerQBMMV3 tiling: b=%u m=%u k=%u n=%u m0=%u k0=%u n0=%u mLoop=%u kLoop=%u nLoop=%u coreLoop=%u "
          "blockDim=%u swizzle=%u/%u bias=%d/%u",
          td->batchSize, td->m, td->k, td->n, td->m0, td->k0, td->n0, td->mLoop, td->kLoop, td->nLoop, td->coreLoop,
          td->blockDim, td->swizzleDirect, td->swizzleCount, static_cast<int>(td->withBias), td->biasDtype);
  context->SetBlockDim(td->blockDim);
  context->SetTilingKey(0);
  size_t *ws = context->GetWorkspaceSizes(1);
  if (ws != nullptr) {
    // Mirror the built-in PostTiling (workspace = GetLibApiWorkSpaceSize);
    // the 32 floor guards the CANN 9.2 GE quirk where a zero-size ws0 gets
    // dropped from the opio and the first tensor parks at the feature-map
    // base (~9x DMA degradation).
    ws[0] = sysWs != 0 ? sysWs : kMinWorkspaceBytes;
  }
  return GRAPH_SUCCESS;

  return GRAPH_SUCCESS;
}

uint32_t InnerQuantBatchMatmulV3TilingFunc(TilingContext *context) {
  if (context == nullptr) {
    return GRAPH_FAILED;
  }
  const char *opName = context->GetNodeName();
  uint64_t coreNum = 0;
  uint64_t l2Size = 0;
  uint64_t l0cSize = 0;
  size_t sysWs = 0;
  if (!GetPlatformAndMem(context, opName, coreNum, l2Size, l0cSize, sysWs)) {
    return GRAPH_FAILED;
  }
  uint64_t batch = 0;
  uint64_t m = 0;
  uint64_t k = 0;
  uint64_t n = 0;
  if (!ParseMatmulShapes(context, opName, batch, m, k, n)) {
    return GRAPH_FAILED;
  }
  if (!CheckTensorContract(context, opName)) {
    return GRAPH_FAILED;
  }
  bool withBias = false;
  bool biasWithBatch = false;
  uint32_t biasDtype = static_cast<uint32_t>(ge::DT_INT32);
  if (!ResolveScaleBiasContract(context, opName, batch, m, n, withBias, biasWithBatch, biasDtype)) {
    return GRAPH_FAILED;
  }
  // Tiling chain: env-forced (override knob) -> [search -> swizzle -> End] -> struct fill.
  TilingCalc t;
  t.batchSize = batch;
  t.m = m;
  t.k = k;
  t.n = n;
  const auto forced = TryEnvForcedTiling(t, coreNum, l0cSize, opName);
  if (forced == ForcedTiling::kInvalid) {
    return GRAPH_FAILED;
  }
  if (forced == ForcedTiling::kNone) {
    SearchTile(t, coreNum, l2Size, l0cSize);
    Swizzle(t);
    EndK0(t);
    if (!ApplyShapeKeyedOverride(t, coreNum, l0cSize, opName)) {
      return GRAPH_FAILED;
    }
  }

  if (t.m0 == 0 || t.n0 == 0 || t.k0 == 0 || t.blockDim == 0) {
    OP_LOGE(opName, "tiling search produced an empty tile");
    return GRAPH_FAILED;
  }

  return WriteTilingData(context, t, biasDtype, withBias, biasWithBatch, sysWs, opName);
}
}  // namespace

IMPL_OP_OPTILING(InnerQuantBatchMatmulV3)
  .Tiling(InnerQuantBatchMatmulV3TilingFunc, sizeof(QuantMatmulPertokenTilingDataArch20));
