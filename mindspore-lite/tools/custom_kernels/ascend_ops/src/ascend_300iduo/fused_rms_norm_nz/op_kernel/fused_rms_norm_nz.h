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

#ifndef FUSED_RMS_NORM_NZ_H
#define FUSED_RMS_NORM_NZ_H

#include "kernel_operator.h"                // NOLINT(build/include_subdir)
#include "fused_rms_norm_nz_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedRmsNormNZ {

// ---------------------------------------------------------------------------
// FusedRmsNormNZ v3: single-kernel RMSNorm for Atlas 300I Duo (dav_m200).
// Pure-vector kernel (AICore task, vector pipe only).
//
// Math (numerics deliberately aligned op-for-op with the chain):
//   ms[m]    = mean_j(x[m,j]^2) + epsilon          (epsilon attr, 1e-6)
//   y[m, j]  = x[m, j] * rsqrt(ms[m]) * gamma[j]
// Statistics and the normalize multiply run in fp32 inside UB (the fp32
// red line: fp16 statistics are rejected); squares are
// taken over f32(x) -- exactly the fp16 values, matching the chain's Square
// on Cast(x). The only fp16 rounding is the output cast, the same
// single-rounding semantics the chain has. v1 measured max 1 ULP vs the
// fp32 reference; v2 associates the row sum differently (one whole-row
// tree instead of per-chunk trees added up), which perturbs ms by at most
// ~1 fp32 rounding, so the gate is <= 2 ULP.
//
// Layout: x/y are FRACTAL_NZ at the deployment site (GEMM epilogue -> norm
// -> GEMM, all NZ, no TransData) or ND in the single-op test chain. NZ
// storage of a logical [P..., M, N] fp16 tensor is [P..., n1, m1t, 16, 16]:
// column fractals outermost, the 16x16 fractal row-major [m0, n0], so one
// row's data is N/16 segments of 16 fp16 (32B each) sitting m1t*512B apart.
// The kernel gathers each row with ONE strided DataCopyParams copy into a
// row-contiguous UB slot, runs flat row-major vector passes, and scatters
// back with the inverse params. DataCopyParams blockLen and strides are in
// 32-BYTE units (the chunk_gated_delta_rule convention: blockLen =
// elements*bytes/32, stride = end-of-burst to start-of-next gap in 32B
// blocks), so one NZ segment of 16 fp16 = blockLen 1 and the inter-segment
// gap m1t*512B-32B = m1t*16-1. ND storage is flat rows and uses one
// contiguous copy per block for both directions.
//
// dav_m200 constraints:
//   - No DataCopyPad: N is a multiple of 32 elements, so every copy (whole
//     row, whole block) stays 32B aligned.
//   - DataCopyParams blockCount/strides are uint16: m1t*16-1 <= 65535 and
//     n/16 <= 65535 are guarded in the tiling.
//   - No cross-core reduce: rows are disjoint.
//   - Plain count-form vector ops (Cast/Mul/Add with int32 count) are
//     counter-mode masked with no repeat cap; the largest flat counts
//     verified on this chip are 8192 (fused_qk_norm_rope deploy) and 12288
//     (gelu v3 chunk), so R*n is capped at 16384.
//
// Pad rows of the NZ tail (m1t*16 > logical M) are garbage-in-garbage-out
// within the allocation, never read downstream (same stance as the gelu
// op).
// ---------------------------------------------------------------------------
constexpr uint32_t kMaxRowBlock = 16;                            // R upper bound
constexpr uint32_t kTreeLandF32 = 8;                             // flat tree landing elems/row
constexpr uint32_t kSlotStrideF32 = 8;                           // 32B slot stride (fp32)
constexpr uint32_t kF32PerBlk = 8;                               // fp32 elements in one 32B block
constexpr int32_t kVcgSrcRepStride = 8;                          // production vcgadd tuple: 8 x 32B = one repeat width
constexpr uint32_t kNzInnerDim = 16;                             // NZ 16x16 fractal inner size (fp16 per 32B block)
constexpr uint32_t kNzFractalElems = kNzInnerDim * kNzInnerDim;  // elements in one 16x16 fractal
constexpr uint32_t kNewtonRefinements = 2;                       // rsqrt Newton refinement passes
// Whole-row residency tile bytes per row-block candidate (mirror of the
// tiling-side budget): 4 flat tiles (fp16 x/y + fp32 work/tmp) at 12 B/elem,
// three [R, 8] fp32 slot tiles at 96 B/row, gamma staging + resident copy
// at 6 B/elem.
constexpr uint32_t kWholeRowBytesPerElem = 12;
constexpr uint32_t kSlotTilesBytesPerRow = 96;
constexpr uint32_t kGammaBytesPerElem = 6;
// 200KB cap minus the margin that keeps every candidate inside the ~196KB
// empirically proven envelope (swiglu v3's measured ceiling); must match
// the tiling-side budget so the Init clamp never disagrees with selection.
constexpr uint32_t kUbBudgetBytes = 200 * 1024;
constexpr uint32_t kUbMarginBytes = 4 * 1024;
constexpr uint32_t kUbSlackBytes = 512;    // TBuf 32B-alignment slack
constexpr uint32_t kMaxFlatCount = 16384;  // verified counter-mode ceiling

class FusedRmsNormNZ {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, const FusedRmsNormNZTilingData *__restrict tiling,
                              TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    m_ = tiling->m;
    n_ = tiling->n;
    isNz_ = tiling->isNz;
    epsilon_ = tiling->epsilon;
    m1t_ = tiling->m1t;
    // v1 MindIR guard (swiglu v3 lesson): stale v1 tiling bakes rowBlock
    // up to 16 on chunked staging; clamp to what whole-row residency
    // allows, recomputed from n with the tiling's own formula.
    rowBlock_ = ClampRowBlock(tiling->rowBlock);
    if (rowBlock_ == 0) {
      // Tiling block that can never fit the UB envelope (n == 0 or no R
      // fits): keep myRows_ = 0 so Process() exits without touching any
      // LocalTensor; never InitBuffer an over-budget block.
      myRows_ = 0;
      return;
    }
    // Row blocks striped over cores: integer block ranges, so every core's
    // row range starts at a multiple of R and blocks are never shared.
    uint32_t blocks = (m_ + rowBlock_ - 1) / rowBlock_;
    uint32_t bStart = blockIdx_ * blocks / coreNum_;
    uint32_t bEnd = (blockIdx_ + 1) * blocks / coreNum_;
    rowStart_ = bStart * rowBlock_;
    uint32_t rowEnd = bEnd * rowBlock_;
    if (rowEnd > m_) {
      rowEnd = m_;
    }
    myRows_ = rowEnd - rowStart_;

    // NZ addressing constants: n1 column fractals, rowsPerPrefix rows per
    // prefix dim (see the layout note above).
    n1_ = n_ / kNzInnerDim;
    rowsPerPrefix_ = m1t_ * kNzInnerDim;

    // v3 reduce plan (see the header note): whole-tile vcgadd ladder while
    // the per-row live count is a whole number of 8-element blocks; the
    // landing count feeds the per-row lane folds. n % 8 != 0 (never legal
    // from the tiling) keeps v2's flat-tree path with redLevels_ = 0.
    redLand_ = n_;
    redLevels_ = 0;
    if ((n_ & (kF32PerBlk - 1)) == 0) {
      uint32_t perRow = n_;
      while (perRow >= kF32PerBlk && (perRow & (kF32PerBlk - 1)) == 0) {
        perRow /= kF32PerBlk;
        ++redLevels_;
      }
      redLand_ = perRow;
    }

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(gamma));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));

    // Whole-row flat tiles: fp16 in/out + fp32 work/squares (12*R*n), the
    // three 32B-strided fp32 slot tiles (96*R) and the gamma tables (6*n).
    pipe->InitBuffer(xBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(workF32Buf_, rowBlock_ * n_ * sizeof(float));
    pipe->InitBuffer(tmpF32Buf_, rowBlock_ * n_ * sizeof(float));
    pipe->InitBuffer(yRowBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(sumRowBuf_, rowBlock_ * kSlotStrideF32 * sizeof(float));
    pipe->InitBuffer(invRowBuf_, rowBlock_ * kSlotStrideF32 * sizeof(float));
    pipe->InitBuffer(newtonF32Buf_, rowBlock_ * kSlotStrideF32 * sizeof(float));
    pipe->InitBuffer(gammaBuf_, n_ * sizeof(half));
    pipe->InitBuffer(gsF32Buf_, n_ * sizeof(float));

    xTileH_ = xBuf_.Get<half>();
    workF32_ = workF32Buf_.Get<float>();
    tmpF32_ = tmpF32Buf_.Get<float>();
    yRowH_ = yRowBuf_.Get<half>();
    sumRowF32_ = sumRowBuf_.Get<float>();
    invRowF32_ = invRowBuf_.Get<float>();
    newtonF32_ = newtonF32Buf_.Get<float>();
    gammaH_ = gammaBuf_.Get<half>();
    gsF32_ = gsF32Buf_.Get<float>();
    LoadGamma();
  }

  __aicore__ inline void Process() {
    if (myRows_ == 0) {
      return;
    }
    // First block's tile is loaded here; every later iteration prefetches
    // its own tile inside the previous block.
    uint32_t firstRows = (myRows_ < rowBlock_) ? myRows_ : rowBlock_;
    LoadBlockTile(rowStart_, firstRows);
    for (uint32_t m0 = rowStart_; m0 < rowStart_ + myRows_; m0 += rowBlock_) {
      uint32_t rowsInBlock = rowStart_ + myRows_ - m0;
      if (rowsInBlock > rowBlock_) {
        rowsInBlock = rowBlock_;
      }
      ProcessBlock(m0, rowsInBlock);
    }
    // Drain the last block's delayed MTE3_V (the in-loop wait only covers
    // blocks that follow one).
    if (mte3Armed_) {
      WaitFlag<HardEvent::MTE3_V>(0);
    }
  }

 private:
  // Largest row block the whole-row footprint allows for this n: the same
  // budget formula the v2 tiling enumerates candidates against (12*R*n +
  // 96*R + 6*n + 512 <= 200KB - 4KB margin) plus the flat-count ceiling
  // R*n <= 16384, clamped into [1, kMaxRowBlock]. A v1 MindIR whose baked
  // rowBlock exceeds this is clamped down (correct, slower blocking); n so
  // large that even R=1 overflows (n > 11116) is refused by the v2 tiling
  // and unsupported from stale MindIRs -- see the header note. Returns 0 for
  // a tiling block that can never fit (n == 0, or no R fits); Init then
  // skips the block instead of allocating past the UB envelope.
  __aicore__ inline uint32_t ClampRowBlock(uint32_t rowBlock) const {
    if (n_ == 0) {
      return 0;
    }
    uint32_t perRowBytes = kWholeRowBytesPerElem * n_ + kSlotTilesBytesPerRow;
    uint32_t fixedBytes = kGammaBytesPerElem * n_ + kUbSlackBytes;
    uint32_t budget = kUbBudgetBytes - kUbMarginBytes;
    uint32_t ubCap = (budget > fixedBytes) ? (budget - fixedBytes) / perRowBytes : 0;
    uint32_t flatCap = kMaxFlatCount / n_;
    // ubCap == 0 means even R=1 does not fit the envelope: refuse it rather
    // than raising cap back to 1 and InitBuffering over the budget.
    uint32_t cap = (ubCap < flatCap) ? ubCap : flatCap;
    if (cap == 0) {
      return 0;
    }
    if (cap > kMaxRowBlock) {
      cap = kMaxRowBlock;
    }
    if (rowBlock > cap) {
      return cap;
    }
    if (rowBlock < 1) {
      return 1;
    }
    return rowBlock;
  }

  // GM element offset of the FIRST column segment of flattened row `row`:
  // fractal index (prefix p, column block 0, row block m1) * 256 + m0*16.
  __aicore__ inline uint64_t NzRowBase(uint32_t row) const {
    uint32_t p = row / rowsPerPrefix_;
    uint32_t m = row - p * rowsPerPrefix_;
    return (static_cast<uint64_t>(p) * n1_ * m1t_ + (m / kNzInnerDim)) * kNzFractalElems +
           (m % kNzInnerDim) * kNzInnerDim;
  }

  // Batched copy-in of the whole rows [rowStart, rowStart+rows): the rows
  // land contiguously striped in the [R, n] staging tile. rows is the
  // block's live row count (ragged tail blocks and prefetched blocks load
  // only their live rows -- never past the core's row range). NZ: one
  // strided gather per row (16-elem segments, m1t*512B apart); ND: one
  // flat contiguous copy of the whole [rows, n] block.
  __aicore__ inline void LoadBlockTile(uint32_t rowStart, uint32_t rows) {
    if (isNz_ != 0) {
      // blockLen 1 = one 32B block = one 16-fp16 NZ segment (32B-unit
      // convention, see header); srcStride m1t*16-1 = the m1t*512B-32B gap
      // from this segment's end to the next column block's segment start.
      DataCopyParams params(static_cast<uint16_t>(n_ / kNzInnerDim), 1, static_cast<uint16_t>(m1t_ * kNzInnerDim - 1),
                            0);
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(xTileH_[r * n_], xGm_[NzRowBase(rowStart + r)], params);
      }
    } else {
      DataCopy(xTileH_, xGm_[(uint64_t)rowStart * n_], rows * n_);
    }
    PipeBarrier<PIPE_MTE2>();
  }

  // Batched MTE3 out for the whole block. NZ: one strided scatter per row
  // (row-contiguous staging out to the m1t*512B-apart segments); ND: one
  // contiguous copy of the whole [rows, n] block.
  __aicore__ inline void StoreBlockTile(uint32_t rowStart, uint32_t rows) {
    if (isNz_ != 0) {
      // Inverse of the gather: blockLen 1 (32B segment), dstStride m1t*16-1.
      DataCopyParams params(static_cast<uint16_t>(n_ / kNzInnerDim), 1, 0,
                            static_cast<uint16_t>(m1t_ * kNzInnerDim - 1));
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(yGm_[NzRowBase(rowStart + r)], yRowH_[r * n_], params);
      }
    } else {
      DataCopy(yGm_[(uint64_t)rowStart * n_], yRowH_, rows * n_);
    }
  }

  // One-time per core: load gamma, cast to fp32 and keep it resident for
  // every row (gs = gamma exactly; there is no smooth to fuse).
  __aicore__ inline void LoadGamma() {
    DataCopy(gammaH_, gammaGm_, n_);
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Cast(gsF32_, gammaH_, RoundMode::CAST_NONE, n_);
    PipeBarrier<PIPE_V>();
  }

  // ---------------------------------------------------------------------------
  // One row block (R rows starting at m0):
  //   pass 1 (flat)  : f32(x) Cast + squares Mul over rows*n lanes, the
  //                    native vcgadd ladder + per-row lane folds and slot
  //                    accumulate (v2's garbage-tail tree only on the
  //                    never-legal n % 8 != 0); the next block's MTE2
  //                    prefetch is issued right after the Cast (the x
  //                    tile's only reader) and drains under everything
  //                    below.
  //   scalar phase   : per-row invRms (vector chain on the sum slots, one
  //                    V_S for the whole block).
  //   pass 2 (rows)  : y = x * invRms * gamma from the resident fp32 work
  //                    tile into the y staging tile, after draining the
  //                    PREVIOUS block's delayed MTE3_V (yRow conflict).
  //   batched MTE3   : ND contiguous block / NZ per-row scatter; the
  //                    MTE3_V drain is delayed to the next block's pass 2.
  // ---------------------------------------------------------------------------
  __aicore__ inline void ProcessBlock(uint32_t m0, uint32_t rowsInBlock) {
    // Reset the per-row sum slots (the lane-fold accumulate writes lane 0).
    // Zero intra-V barrier from here on (gelu v5 / fused_qk_norm_rope v3 / BSH v2 law): the
    // slot tile's only consumer is the accumulate, far down the same
    // in-order V pipe.
    Duplicate(sumRowF32_, 0.0f, rowsInBlock * kSlotStrideF32);
    Pass1Squares(m0, rowsInBlock);
    ReduceRows(rowsInBlock);
    float invRms[kMaxRowBlock];
    InvRmsChain(rowsInBlock, invRms);
    Pass2Normalize(m0, rowsInBlock, invRms);
    // The prefetched next-block tile (if any) is waited for at the top of
    // the next ProcessBlock via the shared MTE2_V flag.
  }

  // Pass 1a: f32(x) for the whole block. The x tile's ONLY reader is this
  // Cast, so the next block's prefetch arms and issues right here, draining
  // under the squares, the tree, the chain, pass 2 and the store (v1 had to
  // wait out its whole chunk loop before it could arm the prefetch).
  __aicore__ inline void Pass1Squares(uint32_t m0, uint32_t rowsInBlock) {
    uint32_t elems = rowsInBlock * n_;
    // Wait the prefetched x tile (the first block's was loaded in Process).
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    Cast(workF32_, xTileH_, RoundMode::CAST_NONE, elems);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    uint32_t nextM0 = m0 + rowBlock_;
    if (nextM0 < rowStart_ + myRows_) {
      uint32_t nextRows = rowStart_ + myRows_ - nextM0;
      if (nextRows > rowBlock_) {
        nextRows = rowBlock_;
      }
      LoadBlockTile(nextM0, nextRows);
    }
    // Sum of squares over the fp16 values in fp32 (the chain's Square on
    // Cast(x) sees exactly these values).
    Mul(tmpF32_, workF32_, workF32_, elems);
  }

  // Pass 1b: reduce each row's squares to one scalar in its sum slot.
  __aicore__ inline void ReduceRows(uint32_t rowsInBlock) {
    uint32_t elems = rowsInBlock * n_;
    // v3 reduce: the native block-sum ladder (see the header note). Level
    // k is ONE whole-tile in-place BlockReduceSum (vcgadd) with a counter
    // mask spanning rows*live_k fp32 elements: every 32B block of 8 sums
    // to one slot packed at the prefix, row r's partials landing at
    // [r*live_{k+1}, +live_{k+1}). The ladder advances while live_k is a
    // whole number of blocks (>= 8 and % 8 == 0) so blocks never straddle
    // rows. Stride tuple and mask-state barrier discipline are the
    // production arch-2002 softmax's verbatim: PipeBarrier only where the
    // mask registers change (a pending V instruction reads them at
    // execution time), SetMaskNorm + ResetMask closing the bracket for
    // the count-form ops below.
    uint32_t laneStride;  // row base stride of the per-row partials below
    uint32_t live;        // per-row partial count entering the folds
    if (redLevels_ > 0) {
      uint32_t lv = n_;
      PipeBarrier<PIPE_V>();
      SetMaskCount();
      for (uint32_t level = 0; level < redLevels_; ++level) {
        SetVectorMask<float, MaskMode::COUNTER>(0, static_cast<int32_t>(rowsInBlock * lv));
        BlockReduceSum<float, false>(tmpF32_, tmpF32_, 1, MASK_PLACEHOLDER, 1, 1, kVcgSrcRepStride);
        lv /= kF32PerBlk;
        PipeBarrier<PIPE_V>();
      }
      SetMaskNorm();
      ResetMask();
      laneStride = redLand_;
      live = redLand_;
    } else {
      // v2 flat-tree fallback for n % 8 != 0 (the tiling never produces
      // it): garbage-tail halving folds over the whole tile, landing
      // [5, 8] partials per row.
      uint32_t span = n_;
      while (span > kTreeLandF32) {
        uint32_t h = span >> 1;
        Add(tmpF32_, tmpF32_, tmpF32_[h], elems - h);
        span -= h;
      }
      laneStride = n_;
      live = span;
    }
    // Per-row lane folds down to one scalar (v2 sequence, generalized to
    // the packed row base): odd live counts fold t[0] += t[live-1] first.
    // Count-form Adds run as in-order RAW chains on the V pipe -- no
    // barriers between steps or rows.
    while (live > 1) {
      if ((live & 1U) == 0) {
        live >>= 1;
        for (uint32_t r = 0; r < rowsInBlock; ++r) {
          LocalTensor<float> t = tmpF32_[r * laneStride];
          Add(t, t, t[live], live);
        }
      } else {
        for (uint32_t r = 0; r < rowsInBlock; ++r) {
          LocalTensor<float> t = tmpF32_[r * laneStride];
          Add(t, t, t[live - 1], 1);
        }
        live -= 1;
      }
    }
    // Add is lanewise, it cannot reduce: the scalar at t[0] is the row sum
    // -- accumulate it into the row's sum slot (lane 0 of its 32B stride).
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      Add(sumRowF32_[r * kSlotStrideF32], sumRowF32_[r * kSlotStrideF32], tmpF32_[r * laneStride], 1);
    }
  }

  // Scalar phase: invRms for the whole block as ONE batched vector chain
  // over the R*kSlotStrideF32 contiguous lanes -- slot lanes hold real
  // sums, pad lanes were zeroed and sit on the v=epsilon fixed point
  // (y=1/sqrt(eps) is a Newton fixed point too), never read, never trap.
  // The dav_m200 vector Sqrt/Reciprocal instructions are low precision
  // (measured max ~2.8e-3 relative over 2048 rows, ~2^-8.5), so the seed
  // gets two Newton refinements of rsqrt,
  //   y <- y * (3 - v*y^2) / 2,   err' ~= 1.5 * err^2,
  // landing at fp32-rounding-limited accuracy (~2^-23, < 0.001 fp16 ULP at
  // the output). v stays resident in the sum tile (mean + eps in place);
  // the seed y lands in the invRms tile; scratch is the Newton tile. This
  // is also cheaper than a per-row count=1 loop: 15 ops per block total.
  __aicore__ inline void InvRmsChain(uint32_t rowsInBlock, float *invRms) {
    float invN = 1.0f / static_cast<float>(static_cast<int32_t>(n_));
    uint32_t slotLanes = rowsInBlock * kSlotStrideF32;
    // Same-pipe RAW chain, no barriers between steps (v3: 15 -> 0).
    Muls(sumRowF32_, sumRowF32_, invN, slotLanes);
    Adds(sumRowF32_, sumRowF32_, epsilon_, slotLanes);
    Sqrt(invRowF32_, sumRowF32_, slotLanes);
    Reciprocal(invRowF32_, invRowF32_, slotLanes);
    for (uint32_t it = 0; it < kNewtonRefinements; ++it) {
      Mul(newtonF32_, invRowF32_, invRowF32_, slotLanes);
      Mul(newtonF32_, newtonF32_, sumRowF32_, slotLanes);
      Muls(newtonF32_, newtonF32_, -1.0f, slotLanes);
      Adds(newtonF32_, newtonF32_, 3.0f, slotLanes);
      Mul(newtonF32_, newtonF32_, invRowF32_, slotLanes);
      Muls(invRowF32_, newtonF32_, 0.5f, slotLanes);
    }
    event_t evSum = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evSum);
    WaitFlag<HardEvent::V_S>(evSum);
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      invRms[r] = static_cast<float>(invRowF32_.GetValue(r * kSlotStrideF32));
    }
  }

  // Pass 2 + batched MTE3 out for the whole row block.
  __aicore__ inline void Pass2Normalize(uint32_t m0, uint32_t rowsInBlock, const float *invRms) {
    // Delayed MTE3_V (gelu v2 pattern): the previous block's store only
    // conflicts with yRow, which nothing below touched, so it has been
    // draining under the tree and the chain. The first block has no armed
    // store yet.
    if (mte3Armed_) {
      WaitFlag<HardEvent::MTE3_V>(0);
    }
    // Per row from the resident fp32 work tile (v1's fp16 residency
    // round-trip f16->f32->f16 is exact identity, so the values are the
    // same), in the chain's multiply order (x*invRms first, then *gamma),
    // single fp16 rounding. Per-row Muls/Mul/Cast is an in-order RAW chain
    // on disjoint work rows -- no barriers (v3: 2R -> 0).
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      Muls(workF32_[r * n_], workF32_[r * n_], invRms[r], n_);
      Mul(workF32_[r * n_], workF32_[r * n_], gsF32_, n_);
      Cast(yRowH_[r * n_], workF32_[r * n_], RoundMode::CAST_NONE, n_);
    }
    // The drain wait after the store is DELAYED to the next block's pass 2
    // (the store conflicts with nothing else).
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    StoreBlockTile(m0, rowsInBlock);
    SetFlag<HardEvent::MTE3_V>(0);
    mte3Armed_ = true;
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint32_t m_ = 0;
  uint32_t n_ = 0;
  uint32_t rowBlock_ = 1;
  uint32_t isNz_ = 0;
  float epsilon_ = 1e-6;
  uint32_t m1t_ = 0;
  uint32_t n1_ = 0;
  uint32_t rowsPerPrefix_ = 0;
  uint32_t redLevels_ = 0;  // whole-tile vcgadd ladder levels (0 = v2 tree)
  uint32_t redLand_ = 0;    // per-row partials the ladder lands for the folds
  uint32_t rowStart_ = 0;
  uint32_t myRows_ = 0;
  bool mte3Armed_ = false;

  GlobalTensor<half> xGm_;
  GlobalTensor<half> gammaGm_;
  GlobalTensor<half> yGm_;

  TBuf<TPosition::VECCALC> xBuf_;
  TBuf<TPosition::VECCALC> workF32Buf_;
  TBuf<TPosition::VECCALC> tmpF32Buf_;
  TBuf<TPosition::VECCALC> yRowBuf_;
  TBuf<TPosition::VECCALC> sumRowBuf_;
  TBuf<TPosition::VECCALC> invRowBuf_;
  TBuf<TPosition::VECCALC> newtonF32Buf_;
  TBuf<TPosition::VECCALC> gammaBuf_;
  TBuf<TPosition::VECCALC> gsF32Buf_;

  LocalTensor<half> xTileH_;
  LocalTensor<float> workF32_;
  LocalTensor<float> tmpF32_;
  LocalTensor<half> yRowH_;
  LocalTensor<float> sumRowF32_;
  LocalTensor<float> invRowF32_;
  LocalTensor<float> newtonF32_;
  LocalTensor<half> gammaH_;
  LocalTensor<float> gsF32_;
};

}  // namespace FusedRmsNormNZ

#endif  // FUSED_RMS_NORM_NZ_H
