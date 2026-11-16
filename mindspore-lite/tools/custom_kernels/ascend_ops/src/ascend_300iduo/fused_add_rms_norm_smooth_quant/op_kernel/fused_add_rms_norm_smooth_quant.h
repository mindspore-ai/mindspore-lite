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

#ifndef FUSED_ADD_RMS_NORM_SMOOTH_QUANT_H
#define FUSED_ADD_RMS_NORM_SMOOTH_QUANT_H

#include "kernel_operator.h"                              // NOLINT(build/include_subdir)
#include "fused_add_rms_norm_smooth_quant_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedAddRmsNormSmoothQuant {

// ---------------------------------------------------------------------------
// FusedAddRmsNormSmoothQuant: fused residual-add + RMSNorm + per-channel
// smooth + per-token dynamic quant for Atlas 300I Duo (dav_m200) W8A8 LLM
// graphs. Pure-vector kernel (AICore task, vector pipe only, no Cube).
//
//
// Math (numerics deliberately aligned op-for-op with the chain):
//   r[m, j] = fp16(x1[m, j] + x2[m, j])
//     single-rounding fp16 vector Add of the exact fp32 sum -- bit-identical
//     to te_add's fp16 output (empirically gated by the bench's
//     res_out-vs-chain bit-identity check; the v2 fp32 add + round path is
//     the documented fallback if the hardware adder ever diverges).
//     r feeds res_out AND (as the rounded fp16 value) the rms accumulation,
//     exactly what te_rmsnorm sees downstream of the add.
//   rms[m]  = rsqrt(mean_j(r^2) + epsilon)          (epsilon attr, 1e-6)
//   n[m, j] = r[m, j] * rms[m] * (gamma[j] * smooth[j])   (gs pre-fused once
//     per core in fp16, resident in UB as fp32 -- same value the chain
//     produces as rmsnorm_out * smooth with one extra fp16 rounding)
//   absmax[m] = max_j |n[m, j]|          (fp16 abs + max halving tree)
//   token_scale[m] = absmax[m] / 127
//   y[m, j] = int8_round(n[m, j] * (127 / absmax[m]))  (fp16 Muls + CAST_ROUND)
//
// Layout notes for dav_m200:
//   - DataCopyPad is unavailable: H must be a multiple of 32 elements so every
//     per-chunk fp16/int8 copy and every row base stay 32B aligned.
//   - No cross-core reduce: rows are disjoint.
//
// FRACTAL_NZ storage (isNz, decoded from the tiling's nzInfo): fp16 NZ of a
// logical [M, H] is [H/16, m1t, 16, 16] -- column fractals outermost, the
// 16x16 fractal row-major [m0, n0] (CANN include pto/common/pto_tile.hpp,
// TileShape2D<Layout::NZ>); a row is H/16 segments of 16 elems (32B),
// m1t*512B apart on the same lattice. int8 NZ is [H/32, m1t, 16, 32]
// (C0 = 32B/sizeof(T)): a row is H/32 segments of 32 elems on the SAME
// m1t*512B lattice (both fractals are 512B, so the segment stride
// m1t*16-1 in 32B units is shared). Gather/scatter use the rms-v3-verified
// per-row strided DataCopyParams form (blockLen = 1; K2's MTE3 copy-loss
// matrix only ever hit the blockLen >= 2 form). UB compute is untouched by
// the layout -- x tiles land row-major striped exactly as in ND mode, so
// every vector op is bit-identical. Row blocks are R-aligned with R | 8, so
// a block never crosses a 16-row fractal boundary. m is the LOGICAL row
// count (origin shape): NZ pad rows (m1t*16 > M) are never processed and
// the [M] token_scale output never writes past the logical end. MTE3
// scatters are issued in gated sub-bursts of <= 4 descriptors per queue
// drain (the K2 >4-descriptor undrained-burst loss law; the ND path's 3
// descriptors per block are already inside the safe bound).
//
// v3 execution model (issue-count driven; supersedes v2's sync-count model):
//   v2 kept one PipeBarrier<PIPE_V> after every vector op (~320 per 8-row
//   block at [M,2048]) and processed the invRms chain one op per row. msprof
//   on the sibling kernels (K2 v2 / FusedQKNormRoPEBSH) established that the
//   V pipe executes instructions in order, so a barrier between dependent
//   same-pipe ops is pure scalar issue overhead; BSH's rope section chains
//   six dependent ops (with scratch reuse) under ONE trailing barrier.
//   v3 applies that discipline here:
//     - zero intra-V barriers: a block's whole V section runs as one in-order
//       instruction stream, with ONE PipeBarrier at each V->scalar readback
//       (invRms, absmax) and the standard MTE arm/drain round trips. ~3
//       barriers per block instead of ~320.
//     - r = fp16 Add directly on the x tiles (1 op replaces v2's
//       Cast+Cast+Add+Cast fp32 path); the fp32 squares view comes from one
//       Cast of the rounded fp16 row.
//     - the 4-op invRms chain runs batched over all rows of the block as
//       count = rows*8 contiguous fp32 (the 32B-strided slots are one
//       contiguous block; garbage lanes hold sum=0 -> 0*invN -> +eps ->
//       sqrt(eps) -> 1/sqrt(eps), all finite, and are never read back).
//     - UB diet: the fp32 element scratch lives once (pow2ceil(chunk) float,
//       shared by pass1 squares and pass2 act math via in-order reuse);
//       gamma/smooth stage inside the x tiles at Init, so no dedicated
//       staging buffers. At H=4096 this fits rowBlock=4 with colChunk=n
//       (one chunk per row): the tree runs once per row instead of twice
//       (v2's C=2 tiling doubled the tree op count and exposed the
//       in-block chunk-1 MTE2 drain -- the root of v2's H=4096 regression).
//
// Zero padding for the halving trees: trees fold count -> count/2 with
// contiguous views, so any 32-multiple chunk length works -- the tail
// [len, pow2ceil(len)) of the scratch buffers is zeroed once at Init and
// only ever read (folds write [0, len/2) strictly inside the data region),
// so zeros persist across every row and block.
//
// Tile model: 8-row groups striped over cores (keeps every token_scale
// DataCopy a whole 32B block at a 32B-aligned GM offset; the fp32 [M] output
// allocation is 32B padded, which absorbs the ragged last group). Within a
// core, rows advance in blocks of rowBlock R (R divides 8, so block
// boundaries never cross a group boundary); each block's x tile is
// prefetched at the end of the previous block's chunk loop and drains under
// the pass2/absmax/pass3/MTE3 tail of that block.
// ---------------------------------------------------------------------------
constexpr uint32_t kScaleGroupRows = 8;  // fp32 x 8 = one 32B GM block
constexpr uint32_t kMaxRowBlock = 8;     // R upper bound (divides 8)
constexpr uint32_t kTreeLandF32 = 8;     // sum tree landing elems/row
constexpr uint32_t kTreeLandF16 = 8;     // max tree landing elems/row
constexpr uint32_t kSlotStrideF32 = 8;   // 32B slot stride (fp32)
constexpr uint32_t kSlotStrideF16 = 16;  // 32B slot stride (fp16)
constexpr float kQuantMaxValue = 127.0f;
constexpr float kQuantEpsilon = 1e-12;  // DynamicQuant 310P epsilon
constexpr uint32_t kMte3Gate = 4;       // K2-safe descriptors per MTE3 drain
// NZ fractal geometry (see the NZ paragraph above): the fp16 fractal is
// 16 rows x 16 cols (C0 = 32B / sizeof(half)), the int8 fractal 16 rows x
// 32 cols; both are 512B, so a row's same-lattice 32B segments sit
// m1t * kNzFractalBlocks32B blocks apart (the shared gather/scatter stride
// in 32B units).
constexpr uint32_t kNzFractalRows = 16;       // fractal rows (the row >> 4 lattice)
constexpr uint32_t kNzF16FractalCols = 16;    // fp16 elems per 32B segment
constexpr uint32_t kNzI8FractalCols = 32;     // int8 elems per 32B segment
constexpr uint32_t kNzF16FractalElems = 256;  // 16 x 16 fp16 fractal
constexpr uint32_t kNzI8FractalElems = 512;   // 16 x 32 int8 fractal
constexpr uint32_t kNzFractalBlocks32B = 16;  // 512B fractal / 32B block
// Scalar-fold widths of the landed sum-tree partials (8 -> 4 -> 2 -> 1).
constexpr uint32_t kTreeFold4 = kTreeLandF32 / 2;
constexpr uint32_t kTreeFold2 = kTreeLandF32 / 4;

class FusedAddRmsNormSmoothQuant {
 public:
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth, GM_ADDR resOut, GM_ADDR y,
                              GM_ADDR tokenScale, const FusedAddRmsNormSmoothQuantTilingData *__restrict tiling,
                              TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    m_ = tiling->m;
    n_ = tiling->n;
    colChunk_ = tiling->colChunk;
    rowBlock_ = tiling->rowBlock;
    // Host emits R in {8,4,2,1}; clamp defensively so that invRms/absmax
    // (kMaxRowBlock entries) and scaleStage_ (kScaleGroupRows entries) can
    // never be indexed past their bounds by a malformed tiling.
    if (rowBlock_ == 0 || rowBlock_ > kMaxRowBlock || (kMaxRowBlock % rowBlock_) != 0) {
      rowBlock_ = 1;
    }
    hasSmooth_ = tiling->hasSmooth;
    epsilon_ = tiling->epsilon;
    // NZ layout (bit-packed nzInfo, see the tiling header): bit 0 isNz,
    // bits 1..12 m1t = 16-row storage blocks per column-fractal lattice.
    isNz_ = tiling->nzInfo & 1u;
    m1t_ = tiling->nzInfo >> 1;
    // 8-row-group striping: group boundaries are integers, so every core's
    // row range starts at a multiple of 8 and no group is shared. R divides 8,
    // so the per-core row blocks never straddle a group boundary either.
    uint32_t groups = (m_ + kScaleGroupRows - 1) / kScaleGroupRows;
    uint32_t gStart = blockIdx_ * groups / coreNum_;
    uint32_t gEnd = (blockIdx_ + 1) * groups / coreNum_;
    rowStart_ = gStart * kScaleGroupRows;
    uint32_t rowEnd = gEnd * kScaleGroupRows;
    if (rowEnd > m_) {
      rowEnd = m_;
    }
    myRows_ = rowEnd - rowStart_;
    groupRowBase_ = rowStart_;

    x1Gm_.SetGlobalBuffer((__gm__ half *)x1);
    x2Gm_.SetGlobalBuffer((__gm__ half *)x2);
    gammaGm_.SetGlobalBuffer((__gm__ half *)gamma);
    resGm_.SetGlobalBuffer((__gm__ half *)resOut);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
    scaleGm_.SetGlobalBuffer((__gm__ float *)tokenScale);
    if (hasSmooth_ != 0) {
      smoothGm_.SetGlobalBuffer((__gm__ half *)smooth);
    }

    // pow2ceil of a 32-multiple column count; trees fold [len -> len/2) so
    // any 32-multiple works, the scratch just needs a pow2 ceiling of space.
    pow2Chunk_ = Pow2Ceil(colChunk_);
    pow2N_ = Pow2Ceil(n_);

    pipe->InitBuffer(x1Buf_, rowBlock_ * colChunk_ * sizeof(half));
    pipe->InitBuffer(x2Buf_, rowBlock_ * colChunk_ * sizeof(half));
    pipe->InitBuffer(tmpF32Buf_, pow2Chunk_ * sizeof(float));
    pipe->InitBuffer(rRowBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(actRowBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(outI8Buf_, rowBlock_ * n_ * sizeof(int8_t));
    pipe->InitBuffer(tmpHBuf_, pow2N_ * sizeof(half));
    pipe->InitBuffer(sumRowBuf_, rowBlock_ * kSlotStrideF32 * sizeof(float));
    pipe->InitBuffer(maxRowBuf_, rowBlock_ * kSlotStrideF16 * sizeof(half));
    pipe->InitBuffer(invRowBuf_, rowBlock_ * kSlotStrideF32 * sizeof(float));
    pipe->InitBuffer(scaleStageBuf_, kScaleGroupRows * sizeof(float));
    pipe->InitBuffer(gsF32Buf_, n_ * sizeof(float));

    x1H_ = x1Buf_.Get<half>();
    x2H_ = x2Buf_.Get<half>();
    tmpF32_ = tmpF32Buf_.Get<float>();
    rRowH_ = rRowBuf_.Get<half>();
    actRowH_ = actRowBuf_.Get<half>();
    outI8_ = outI8Buf_.Get<int8_t>();
    tmpH_ = tmpHBuf_.Get<half>();
    sumRowF32_ = sumRowBuf_.Get<float>();
    maxRowH_ = maxRowBuf_.Get<half>();
    invRowF32_ = invRowBuf_.Get<float>();
    scaleStage_ = scaleStageBuf_.Get<float>();
    gsF32_ = gsF32Buf_.Get<float>();
    FuseGammaSmooth();
    InitTreePad();
  }

  __aicore__ inline void Process() {
    if (myRows_ == 0) {
      return;
    }
    // First block's chunk-0 tile is loaded here; every later iteration
    // prefetches its own first chunk inside the previous block.
    uint32_t firstRows = (myRows_ < rowBlock_) ? myRows_ : rowBlock_;
    LoadXTile(rowStart_, 0, firstRows);
    for (uint32_t m0 = rowStart_; m0 < rowStart_ + myRows_; m0 += rowBlock_) {
      uint32_t rowsInBlock = rowStart_ + myRows_ - m0;
      if (rowsInBlock > rowBlock_) {
        rowsInBlock = rowBlock_;
      }
      ProcessBlock(m0, rowsInBlock);
    }
  }

 private:
  // pow2 ceiling of v (v >= 1, 32-multiple in practice).
  __aicore__ inline uint32_t Pow2Ceil(uint32_t v) {
    uint32_t p = 32;
    while (p < v) {
      p <<= 1;
    }
    return p;
  }

  // Zero the tree tail regions once: folds only ever write [0, len/2), so
  // the zero tail of each scratch buffer is read-only for the whole kernel
  // lifetime and needs no re-zeroing per row / block. The ragged-chunk tail
  // inside [len, colChunk_) is NOT covered here (full chunks overwrite it
  // with real squares) and is re-zeroed per row in Pass1Chunk.
  __aicore__ inline void InitTreePad() {
    if (pow2Chunk_ > colChunk_) {
      Duplicate(tmpF32_[colChunk_], 0.0f, pow2Chunk_ - colChunk_);
      PipeBarrier<PIPE_V>();
    }
    if (pow2N_ > n_) {
      Duplicate(tmpH_[n_], static_cast<half>(0), pow2N_ - n_);
      PipeBarrier<PIPE_V>();
    }
  }

  // One-time per core: gs[j] = gamma[j] * smooth[j] staged INSIDE the x
  // tiles (they are free until the first LoadXTile), cast to fp32 and kept
  // resident for every row -- no dedicated gamma/smooth UB. Chunked so any
  // colChunk <= n works; with the preferred colChunk == n tiling it is a
  // single iteration. Ends with the V_MTE2 arm so Process can overwrite the
  // x tiles right away.
  __aicore__ inline void FuseGammaSmooth() {
    for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
      uint32_t len = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
      if (c0 != 0) {
        // The previous chunk's Cast was the last x-tile reader: hand the
        // tiles back to MTE2 before overwriting them.
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
      }
      DataCopy(x1H_, gammaGm_[c0], len);
      if (hasSmooth_ != 0) {
        DataCopy(x2H_, smoothGm_[c0], len);
      }
      PipeBarrier<PIPE_MTE2>();
      SetFlag<HardEvent::MTE2_V>(0);
      WaitFlag<HardEvent::MTE2_V>(0);
      if (hasSmooth_ != 0) {
        Mul(x1H_, x1H_, x2H_, len);
      }
      Cast(gsF32_[c0], x1H_, RoundMode::CAST_NONE, len);
    }
    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
  }

  // GM element offset of the first fp16 column segment of `row` at column
  // fractal jc (see the header NZ paragraph): (jc * m1t + (row >> 4)) * 256
  // + (row & 15) * 16.
  __aicore__ inline uint64_t NzRowBase(uint32_t row, uint32_t jc) const {
    return (static_cast<uint64_t>(jc) * m1t_ + row / kNzFractalRows) * kNzF16FractalElems +
           (row % kNzFractalRows) * kNzF16FractalCols;
  }

  // Same for the int8 y output ([H/32, m1t, 16, 32], 512B fractals):
  // (jc * m1t + (row >> 4)) * 512 + (row & 15) * 32.
  __aicore__ inline uint64_t NzRowBaseI8(uint32_t row, uint32_t jc) const {
    return (static_cast<uint64_t>(jc) * m1t_ + row / kNzFractalRows) * kNzI8FractalElems +
           (row % kNzFractalRows) * kNzI8FractalCols;
  }

  // Batched copy-in of one column chunk for rows [m0, m0+rows): the rows'
  // chunk tiles land contiguously striped in the [R, colChunk] staging tiles.
  // rows is the block's live row count (ragged tail blocks and prefetched
  // blocks load only their live rows -- never past the core's row range).
  // NZ: one strided gather per row (rms-v3 form: chunkLen/16 segments of
  // 32B, srcStride m1t*16-1); ND: one flat copy per row.
  __aicore__ inline void LoadXTile(uint32_t m0, uint32_t c0, uint32_t rows) {
    uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
    if (isNz_ != 0) {
      DataCopyParams seg(static_cast<uint16_t>(chunkLen / kNzF16FractalCols), 1,
                         static_cast<uint16_t>(m1t_ * kNzFractalBlocks32B - 1), 0);
      uint32_t jc = c0 / kNzF16FractalCols;
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(x1H_[r * colChunk_], x1Gm_[NzRowBase(m0 + r, jc)], seg);
      }
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(x2H_[r * colChunk_], x2Gm_[NzRowBase(m0 + r, jc)], seg);
      }
    } else {
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(x1H_[r * colChunk_], x1Gm_[(uint64_t)(m0 + r) * n_ + c0], chunkLen);
      }
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(x2H_[r * colChunk_], x2Gm_[(uint64_t)(m0 + r) * n_ + c0], chunkLen);
      }
    }
    PipeBarrier<PIPE_MTE2>();
  }

  // Pass 1 on one (row, chunk): r = fp16(x1 + x2) straight into the
  // row-group resident rRow tile (single-rounding fp16 add of the exact
  // fp32 sum, bit-identical to te_add), fp32 view of the ROUNDED fp16 r,
  // squares, then a halving tree down to kTreeLandF32 partials accumulated
  // into the row's sum slot. The whole sequence is one in-order V stream --
  // no intra-section barriers.
  __aicore__ inline void Pass1Chunk(uint32_t r, uint32_t c0, uint32_t chunkLen) {
    Add(rRowH_[r * n_ + c0], x1H_[r * colChunk_], x2H_[r * colChunk_], chunkLen);
    Cast(tmpF32_, rRowH_[r * n_ + c0], RoundMode::CAST_NONE, chunkLen);
    Mul(tmpF32_, tmpF32_, tmpF32_, chunkLen);
    // Halving tree: fold [0, pow2ceil(len)) -> [0, 8). The tail beyond a
    // RAGGED chunk (len < colChunk_) holds stale squares from the full
    // chunks of previous rows, so it is re-zeroed here; the tail beyond a
    // full chunk ([colChunk_, pow2Chunk_)) was zeroed once at Init and is
    // never written again (folds write strictly inside [0, len)).
    uint32_t span = Pow2Ceil(chunkLen);
    if (chunkLen != colChunk_ && span > chunkLen) {
      Duplicate(tmpF32_[chunkLen], 0.0f, span - chunkLen);
    }
    while (span > kTreeLandF32) {
      span >>= 1;
      Add(tmpF32_, tmpF32_, tmpF32_[span], span);
    }
    // Add is lanewise, it cannot reduce: fold the 8 landed partials down to
    // one scalar (4 -> 2 -> 1) before the 1-element accumulate into the row's
    // sum slot, otherwise the chain below would only ever see lane 0
    // (~1/8 of the true sum, invRms off by ~sqrt(8)).
    Add(tmpF32_, tmpF32_, tmpF32_[kTreeFold4], kTreeFold4);
    Add(tmpF32_, tmpF32_, tmpF32_[kTreeFold2], kTreeFold2);
    Add(tmpF32_, tmpF32_, tmpF32_[1], 1);
    Add(sumRowF32_[r * kSlotStrideF32], sumRowF32_[r * kSlotStrideF32], tmpF32_, 1);
  }

  // Pass 2 on one (row, chunk): act = rRow * invRms * gs into the actRow
  // tile. Reuses tmpF32_ as the fp32 scratch (pass1's tree is done with it;
  // the V pipe is in order).
  __aicore__ inline void Pass2Chunk(uint32_t r, uint32_t c0, uint32_t chunkLen, float invRms) {
    Cast(tmpF32_, rRowH_[r * n_ + c0], RoundMode::CAST_NONE, chunkLen);
    Muls(tmpF32_, tmpF32_, invRms, chunkLen);
    Mul(tmpF32_, tmpF32_, gsF32_[c0], chunkLen);
    Cast(actRowH_[r * n_ + c0], tmpF32_, RoundMode::CAST_NONE, chunkLen);
  }

  // Whole-row absmax: |act| into tmpH_ (zero tail pre-set at Init), max
  // halving tree down to kTreeLandF16 partials copied into the row's max
  // slot; the scalar side combines them after the block's single V_S.
  __aicore__ inline void AbsMaxRow(uint32_t r) {
    Abs(tmpH_, actRowH_[r * n_], n_);
    uint32_t span = pow2N_;
    while (span > kTreeLandF16) {
      span >>= 1;
      Max(tmpH_, tmpH_, tmpH_[span], span);
    }
    Adds(maxRowH_[r * kSlotStrideF16], tmpH_, static_cast<half>(0), kTreeLandF16);
  }

  // Pass 3 on one (row, chunk): y = int8(act * 127/absmax) - fp16 Muls +
  // CAST_ROUND (the 310P DynamicQuant quant tail). tmpH_ doubles as the
  // fp16 staging (absmax trees are done by then).
  __aicore__ inline void Pass3Chunk(uint32_t r, uint32_t c0, uint32_t chunkLen, float quantMul) {
    Muls(tmpH_, actRowH_[r * n_ + c0], static_cast<half>(quantMul), chunkLen);
    Cast(outI8_[r * n_ + c0], tmpH_, RoundMode::CAST_ROUND, chunkLen);
  }

  // token_scale[row] = absmax / 127, staged per 8-row group; the flush
  // happens in the block's MTE3 batch whenever the group completes.
  __aicore__ inline void StageTokenScale(uint32_t row, float absmax) {
    float scaleOut = absmax / kQuantMaxValue;
    uint32_t slot = row - groupRowBase_;
    Duplicate(scaleStage_[slot], scaleOut, 1);
  }

  // MTE3 sub-burst drain: enqueue the completion marker and wait it. Used to
  // gate NZ per-row scatters into sub-bursts of <= kMte3Gate descriptors
  // (the K2 MTE3 copy-loss law: > ~4 descriptors in an undrained burst lose
  // copies non-deterministically; the rms-v3-verified blockLen=1 strided
  // form at 7 descriptors passed, but gating stays strictly inside the law).
  // 1:1 set/wait pairs -- an unpaired second SetFlag would leave a stale
  // marker that satisfies the NEXT WaitFlag immediately.
  __aicore__ inline void DrainMte3() {
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
  }

  // ---------------------------------------------------------------------------
  // One row block (R rows starting at m0) -- the whole V section is a single
  // in-order instruction stream with barriers only at the scalar readbacks:
  //   chunk loop      : batched MTE2 x-tile -> pass1 (fp16 add + squares +
  //                     sum tree), per row, no barriers
  //   invRms phase    : 4-op chain batched over all rows' contiguous slots
  //   -> V_S readback : invRms[r] to scalars
  //   pass2 rows      : act = r * invRms * gs (tmpF32_ scratch reuse)
  //   absmax rows     : |act| + max tree into per-row slots
  //   -> V_S readback : 8 partials per row combined in scalars, token_scale
  //                     staged, quantMul = 127/absmax
  //   pass3 rows      : int8 quant into the y tile
  //   batched MTE3    : contiguous res / y blocks (+ any completed 8-row
  //                     scale group), then prefetch the next block's first
  //                     chunk so the copy-in drains under the next block.
  //                     NZ: per-row strided scatters in gated <= 4-descriptor
  //                     sub-bursts (DrainMte3) -- the store cannot be
  //                     deferred like rms v3's (this op's MTE3 reads rRow,
  //                     which the next block's pass1 rewrites immediately),
  //                     so the scatter runs exposed behind the block tail.
  // ---------------------------------------------------------------------------
  __aicore__ inline void ProcessBlock(uint32_t m0, uint32_t rowsInBlock) {
    // Reset the per-row sum slots (accumulate across chunks).
    Duplicate(sumRowF32_, 0.0f, rowsInBlock * kSlotStrideF32);

    for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
      uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
      if (c0 == 0) {
        // First chunk was prefetched (or loaded in Process); wait it.
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
      } else {
        LoadXTile(m0, c0, rowsInBlock);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
      }
      for (uint32_t r = 0; r < rowsInBlock; ++r) {
        Pass1Chunk(r, c0, chunkLen);
      }
      // x tiles are free: hand them back and prefetch the next block's
      // first chunk right away (drains under pass 2/3 and the MTE3 batch).
      // For in-block chunks the next LoadXTile reuses the same staging.
      SetFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::V_MTE2>(0);
      uint32_t nextM0 = m0 + rowBlock_;
      if (c0 + colChunk_ >= n_ && nextM0 < rowStart_ + myRows_) {
        uint32_t nextRows = rowStart_ + myRows_ - nextM0;
        if (nextRows > rowBlock_) {
          nextRows = rowBlock_;
        }
        LoadXTile(nextM0, 0, nextRows);
      }
    }

    // invRms for all rows in 4 batched ops: the 32B-strided per-row slots
    // are one contiguous [rows*8] fp32 block. Lanes 1..7 of every slot hold
    // sum 0 -> 0 -> eps -> sqrt(eps) -> 1/sqrt(eps): finite, never read.
    // The intermediate int32_t cast is required by the Ascend C compiler:
    // a direct float <- uint32_t cast is rejected ("cast between floating and
    // unsigned integer variable is not allowed in aicore function"). n_ is a
    // hidden size, far below INT32_MAX, so the narrowing is lossless.
    float invN = 1.0f / static_cast<float>(static_cast<int32_t>(n_));
    Muls(invRowF32_, sumRowF32_, invN, rowsInBlock * kSlotStrideF32);
    Adds(invRowF32_, invRowF32_, epsilon_, rowsInBlock * kSlotStrideF32);
    Sqrt(invRowF32_, invRowF32_, rowsInBlock * kSlotStrideF32);
    Reciprocal(invRowF32_, invRowF32_, rowsInBlock * kSlotStrideF32);
    PipeBarrier<PIPE_V>();
    event_t evSum = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evSum);
    WaitFlag<HardEvent::V_S>(evSum);
    float invRms[kMaxRowBlock];
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      invRms[r] = static_cast<float>(invRowF32_.GetValue(r * kSlotStrideF32));
    }

    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
        uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
        Pass2Chunk(r, c0, chunkLen, invRms[r]);
      }
    }
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      AbsMaxRow(r);
    }
    PipeBarrier<PIPE_V>();
    event_t evMax = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evMax);
    WaitFlag<HardEvent::V_S>(evMax);
    float absmax[kMaxRowBlock];
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      float mx = 0.0f;
      for (uint32_t k = 0; k < kTreeLandF16; ++k) {
        float v = static_cast<float>(maxRowH_.GetValue(r * kSlotStrideF16 + k));
        if (v > mx) {
          mx = v;
        }
      }
      absmax[r] = mx;
      StageTokenScale(m0 + r, mx);
    }

    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      float quantMul = (absmax[r] > kQuantEpsilon) ? (kQuantMaxValue / absmax[r]) : 0.0f;
      for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
        uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
        Pass3Chunk(r, c0, chunkLen, quantMul);
      }
    }

    // Batched MTE3 out for the whole row group. ND: one contiguous copy each
    // for res / y plus the scale flush -- 3 descriptors, inside the K2-safe
    // burst bound, one drain. NZ: per-row strided scatters (the rms-v3
    // inverse of the gather), gated into <= kMte3Gate descriptors per drain.
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    if (isNz_ != 0) {
      DataCopyParams segF16(static_cast<uint16_t>(n_ / kNzF16FractalCols), 1, 0,
                            static_cast<uint16_t>(m1t_ * kNzFractalBlocks32B - 1));
      DataCopyParams segI8(static_cast<uint16_t>(n_ / kNzI8FractalCols), 1, 0,
                           static_cast<uint16_t>(m1t_ * kNzFractalBlocks32B - 1));
      uint32_t queued = 0;
      for (uint32_t r = 0; r < rowsInBlock; ++r) {
        DataCopy(resGm_[NzRowBase(m0 + r, 0)], rRowH_[r * n_], segF16);
        if (++queued == kMte3Gate) {
          DrainMte3();
          queued = 0;
        }
      }
      for (uint32_t r = 0; r < rowsInBlock; ++r) {
        DataCopy(yGm_[NzRowBaseI8(m0 + r, 0)], outI8_[r * n_], segI8);
        if (++queued == kMte3Gate) {
          DrainMte3();
          queued = 0;
        }
      }
    } else {
      DataCopy(resGm_[(uint64_t)m0 * n_], rRowH_, rowsInBlock * n_);
      DataCopy(yGm_[(uint64_t)m0 * n_], outI8_, rowsInBlock * n_);
    }
    bool groupDone = false;
    uint32_t lastRow = m0 + rowsInBlock;
    if (lastRow - groupRowBase_ >= kScaleGroupRows || lastRow == rowStart_ + myRows_) {
      groupDone = true;
    }
    if (groupDone) {
      DataCopy(scaleGm_[groupRowBase_], scaleStage_, kScaleGroupRows);
    }
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
    if (groupDone) {
      groupRowBase_ += kScaleGroupRows;
    }
    // The prefetched next-block tile (if any) is waited for at the top of
    // the next ProcessBlock via the shared MTE2_V flag.
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint32_t m_ = 0;
  uint32_t n_ = 0;
  uint32_t colChunk_ = 0;
  uint32_t rowBlock_ = 1;
  uint32_t pow2Chunk_ = 32;
  uint32_t pow2N_ = 32;
  uint32_t hasSmooth_ = 0;
  float epsilon_ = 1e-6;
  uint32_t isNz_ = 0;  // 1 = x1/x2/res_out/y are FRACTAL_NZ storage
  uint32_t m1t_ = 0;   // NZ 16-row storage blocks (row >> 4 lattice)
  uint32_t rowStart_ = 0;
  uint32_t myRows_ = 0;
  uint32_t groupRowBase_ = 0;

  GlobalTensor<half> x1Gm_;
  GlobalTensor<half> x2Gm_;
  GlobalTensor<half> gammaGm_;
  GlobalTensor<half> smoothGm_;
  GlobalTensor<half> resGm_;
  GlobalTensor<int8_t> yGm_;
  GlobalTensor<float> scaleGm_;

  TBuf<TPosition::VECCALC> x1Buf_;
  TBuf<TPosition::VECCALC> x2Buf_;
  TBuf<TPosition::VECCALC> tmpF32Buf_;
  TBuf<TPosition::VECCALC> rRowBuf_;
  TBuf<TPosition::VECCALC> actRowBuf_;
  TBuf<TPosition::VECCALC> outI8Buf_;
  TBuf<TPosition::VECCALC> tmpHBuf_;
  TBuf<TPosition::VECCALC> sumRowBuf_;
  TBuf<TPosition::VECCALC> maxRowBuf_;
  TBuf<TPosition::VECCALC> invRowBuf_;
  TBuf<TPosition::VECCALC> scaleStageBuf_;
  TBuf<TPosition::VECCALC> gsF32Buf_;

  LocalTensor<half> x1H_;
  LocalTensor<half> x2H_;
  LocalTensor<float> tmpF32_;
  LocalTensor<half> rRowH_;
  LocalTensor<half> actRowH_;
  LocalTensor<int8_t> outI8_;
  LocalTensor<half> tmpH_;
  LocalTensor<float> sumRowF32_;
  LocalTensor<half> maxRowH_;
  LocalTensor<float> invRowF32_;
  LocalTensor<float> scaleStage_;
  LocalTensor<float> gsF32_;
};

}  // namespace FusedAddRmsNormSmoothQuant

#endif  // FUSED_ADD_RMS_NORM_SMOOTH_QUANT_H
