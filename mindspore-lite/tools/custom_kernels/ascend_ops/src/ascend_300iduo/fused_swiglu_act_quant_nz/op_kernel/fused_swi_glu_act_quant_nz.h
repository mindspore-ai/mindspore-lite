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

#ifndef FUSED_SWI_GLU_ACT_QUANT_NZ_H
#define FUSED_SWI_GLU_ACT_QUANT_NZ_H

#include "kernel_operator.h"                         // NOLINT(build/include_subdir)
#include "fused_swi_glu_act_quant_nz_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedSwiGLUActQuantNZ {

// ---------------------------------------------------------------------------
// FusedSwiGLUActQuantNZ: fused FFN activation + per-token dynamic quant for
// Atlas 300I Duo (dav_m200) W8A8 LLM graphs. Pure-vector kernel (AICore task,
// vector pipe only, no Cube).
//
// Replaces the unfused per-layer chain:
//   ConcatD(434us) + SwiGlu(600us) + Mul-smooth(190us) + DynamicQuant(373us)
//   + TransData(123us) ~= 1718us
//
// Math (v4a: exact-f32 exp head + fp16 activation tail, fp16 quant tail):
//   act[m, j] = silu(gate[m, j]) * up[m, j] (* smooth_scale[j])
//   silu(g) = g * sigmoid(g), sigmoid = 1/(1+exp(-g)); gate is the FIRST
//             branch (matching the CANN built-in SwiGlu / te_swiglu)
//   absmax[m] = max_j |act[m, j]|           (fp16 abs + max halving tree)
//   token_scale[m] = absmax[m] / 127        (dequant convention, DynamicQuant)
//   y[m, j] = int8_round(act[m, j] * (127 / absmax[m]))   (fp16 Muls + CAST_ROUND)
//
// v1 -> v2 -> v3 kept every arithmetic op, dtype and rounding step
// bit-for-bit -- only the EXECUTION MODEL changed. v4a is the first
// numerics change (fp16 activation tail; see the v4a section below); v4b
// keeps v4a's numerics bit-identical (execution-model only, like v1->v3).
// v1 processed one row at a time with ~10 pipe event round-trips per row
// plus ReduceMax's internal scalar syncs (msprof: scalar_ratio 0.96, task
// 787us @1536x6144). v2 batches
// rowBlock R rows per block: batched MTE2 tile loads (one 2D gapped gather per
// tensor, K1-v4 form), a manual halving Max tree (no Level-2 reduce syncs),
// ONE V_S per block, one contiguous [R, n] MTE3 y copy + per-group scale
// flush, next-block chunk-0 prefetch and a deferred MTE3_V drain (gelu
// ping-pong pattern). Measured: task 687us, vec 0.89, scalar 0.26 @1536x6144;
// 1835us, vec 0.80, scalar 0.21 @1768x12288 (v1/v2 outputs bit-identical).
//
// v3 closes the remaining vector-pipe idle gaps (all bit-identical):
//   - PING-PONG gate/up tile pairs when colChunk < n: the next chunk's MTE2
//     is issued BEFORE this chunk's chain, into the pair handed back one
//     chunk ago, so the copy-in drains under the chain instead of stalling
//     the vector pipe between chunks (v2 loaded in-block chunks only after
//     the previous chain's handback -- ~3 exposed load latencies per block
//     at the C=4 deployment shape).
//   - NON-POW2 max tree: the halving fold starts at span = n (odd spans fold
//     their dangling last partial into slot 0), dropping v2's zero pad to
//     pow2ceil(n). All values ever read by the landing are per-subset maxes
//     of this row's |act| (Abs covers [0, n), n >= 32), so the max is exact
//     and order-independent while the tree scratch shrinks to n halves --
//     exactly the UB the ping-pong pair needs to fit the 200KB budget.
//   - FOLD-TO-1 landing: three extra trivial folds land ONE max per row, so
//     the block's scalar readback is R GetValues instead of 8*R -- the
//     vector pipe idled behind those scalar UB reads between tree and
//     pass 3.
//   - RESIDENT FP32 smooth table (tiling-flagged, budget-permitted): the
//     per-(row, chunk) Cast of the fp16 smooth slice is replaced by one
//     whole-row cast at Init. f16 -> f32 is exact, so every chain operand
//     is bit-identical; drops one of the 11 chain passes at shapes whose
//     UB headroom allows it (not at n=12288, where the ping-pong pair
//     takes the bytes).
//
// v4a swaps the activation tail from the f32 pipe to the fp16 pipe (the
// gelu-v4h tail pattern; the chain was the dominant instruction stream --
// 63% of the issue count at the deployment shape):
//   - The ONLY f32 work left is the exp head: n16 = -g16 (exact fp16 sign
//     flip), Cast-up (f16->f32 exact), in-place Exp<f32> -- e32 is
//     BIT-IDENTICAL to v3's (v3 built -g32 via an exact f32 negation of
//     f32(g16); v4a's f32(f16(-g16)) is the same value), so the 310P
//     exp-floor signature carries over -- then Cast-down to e16.
//   - Everything after runs fp16: d16 = 1+e16, r16 = vrec(d16), s16 = g*r,
//     t16 = s*u, act16 = t*smooth (or act16 = s*u without smooth; the
//     final Mul writes actRow directly, no copy). vrec<half> is the ONE
//     native special-form instruction on dav_m200 (gelu v4c measured ~1
//     unit vs the ~7.6-unit emulated vexp/vdiv<half>).
//   - This ELIMINATES the fast-approximate Reciprocal<f32>: a device probe
//     (isolated v3 vendor, per-element diff vs an exact-arith simulator)
//     measured eps up to ~2.8e-3 relative on v3's d in [1, 2] -- the same
//     instruction family as K1 gotcha-13's Sqrt/Reciprocal. v4a's r16 is
//     exact 1/x rounded once to fp16.
//   - NOT bit-identical to v3: three fp16 roundings (e16, s16, t16 for the
//     smooth path) replace v3's single act16 rounding, and r16 replaces
//     fast-recip32. Bounded and one-sided-safe: simulator (device-calibrated
//     on v3) + margin analysis bound the act error well under the int8 flip
//     threshold (1.5/127 = 1.2e-2 at |y|=127) and the bench's scale_rel <
//     5e-3 gate; expect at most +-1 int8 flips on a small element fraction.
//   - The v3 resident fp32 smooth table is REMOVED: the fp16 tail reads
//     smoothH_ directly, so the per-chunk smooth Cast is gone at EVERY
//     shape (v3 still paid it at n=12288). The tiling slot stays 0 for
//     legacy-tiling layout compatibility.
//   - tmpF32_ is gone too: one f32 scratch (workF32_) stages the Cast-up
//     and the in-place Exp. Chain units drop from 22 (v3 per-chunk-cast
//     path), or 20 (resident-f32 path), to 14 with smooth or 13 without;
//     unit model: fp16 tensor 1, vrec 1, f32 tensor 2, fp16 scalar 2.
//     Freeing tmpF32_ + the resident fp32 table also lets the 200KB budget
//     take colChunk 6144 (was 3072) at n=12288 -- half the chunks, so half
//     the per-(row, chunk) chain issue cost at that shape too.
//
// v4b (single-chunk row fusion, bit-identical to v4a): at colChunk == n the
// block's [R, n] gate/up tiles and its actRow slice are each ONE contiguous
// range, so with 2+ live rows the whole block flows through the v4a chain
// as ONE wide beat set -- 9 beats at width rows*n instead of 9 beats per
// row. Element-wise vector ops are count-insensitive, so the output is
// bit-identical to the per-row chain; only the per-(row, chunk) issue +
// barrier-marker cost collapses to once per block (the deployment shape
// 1536x6144 is single-chunk: 9 beats per 2-row block vs 18). Costs: the f32
// exp scratch, the fp16 chain staging and a replicated smooth table
// (smooth2x_, R copies filled once at Init) all scale with R*cc; the budget
// still holds at every bench shape (196,704B at R=2, cc=6144, smooth).
//
// Zero padding: none needed anymore -- v3's tree never reads past n (the
// fold-to-1 tail lives inside [0, n)), and pass-3 staging writes [0,
// chunkLen) with chunkLen <= n.
//
// Layout notes for dav_m200:
//   - DataCopyPad is unavailable: N must be a multiple of 32 elements so every
//     per-chunk fp16/int8 copy and every row base stay 32B aligned.
//   - No cross-core reduce: rows are disjoint.
//
// FRACTAL_NZ storage (isNz from the tiling): fp16 NZ of a logical [M, N] is
// [N/16, m1t, 16, 16] -- column fractals outermost, the 16x16 fractal
// row-major [m0, n0] (CANN include pto/common/pto_tile.hpp, TileShape2D
// <Layout::NZ>) -- a row is N/16 segments of 16 elems (32B), m1t*512B apart.
// int8 NZ is [N/32, m1t, 16, 32] (C0 = 32B/sizeof(T)): a row is N/32
// segments of 32 elems on the SAME m1t*512B lattice (both fractals are 512B,
// so the segment stride m1t*16-1 in 32B units is shared). Gather/scatter use
// the rms-v3-verified per-row strided DataCopyParams form (blockLen = 1;
// K2's MTE3 copy-loss matrix only ever hit the blockLen >= 2 form). UB
// compute is untouched by the layout -- gate/up tiles land row-major striped
// exactly as in ND mode, so every chain beat is bit-identical. Row blocks
// are R-aligned with R | 8, so a block never crosses a 16-row fractal
// boundary. m is the LOGICAL row count (origin shape): NZ pad rows
// (m1t*16 > M) are never processed and the [M] token_scale output never
// writes past the logical end. The NZ y scatter queues 2R+1 descriptors
// (R rows + the scale flush); the K2 MTE3 copy-loss law caps undrained
// bursts at ~4, so sub-bursts of <= kMte3Gate descriptors take an immediate
// queue drain while the FINAL sub-burst keeps the deferred drain (the
// deployment shapes pick R <= 2 -> one sub-burst of <= 3, all deferred).
//
// Tile model: 8-row groups striped over cores (keeps every token_scale
// DataCopy a whole 32B block at a 32B-aligned GM offset; the fp32 [M] output
// allocation is 32B padded, which absorbs the ragged last group). Within a
// core, rows advance in blocks of rowBlock R (R divides 8, so block
// boundaries never cross a group boundary). UB buffers are partitioned by
// pipe: gate/up A (+B when ping-pong) = MTE2 in / V read (tiles, handed back
// per chunk); actRowH_ = V-only row-resident; outI8_/scaleStage_ = V write /
// MTE3 read; workF32_/tmpH_/maxRowH_ = V-only scratch. tmpH_ is triple-use
// (fp16 chain staging during the chunk loop -- R*cc halves on single-chunk
// shapes for v4b's fused chain, cc otherwise -- then absmax tree scratch,
// then pass-3 fp16 staging; sequential V-pipe phases, and the trees are
// consumed by the block's single V_S before pass 3 starts). workF32_ is
// sized R*cc on single-chunk shapes (the fused Cast-up/Exp beat set), cc
// otherwise. smooth2x_ (single-chunk smooth shapes only) = MTE2 in at Init
// / V read by the fused chain's final Mul.
// ---------------------------------------------------------------------------
constexpr uint32_t kScaleGroupRows = 8;  // fp32 x 8 = one 32B GM block
constexpr uint32_t kMaxRowBlock = 8;     // R upper bound (divides 8)
constexpr uint32_t kSlotStrideF16 = 16;  // 32B slot stride (fp16)
constexpr uint32_t kAlignBytes = 32;     // DataCopy alignment unit
constexpr float kQuantMaxValue = 127.0f;
constexpr float kQuantEpsilon = 1e-12;  // DynamicQuant 310P epsilon
constexpr float kNegOne = -1.0f;
constexpr float kOne = 1.0f;
constexpr uint32_t kMte3Gate = 4;  // K2-safe descriptors per MTE3 drain
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
// v4b single-chunk fusion floor: blocks with fewer live rows fall back to
// the per-row chain (a lone ragged row gains nothing from wide beats).
constexpr uint32_t kFusedChainMinRows = 2;

class FusedSwiGLUActQuantNZ {
 public:
  __aicore__ inline void Init(GM_ADDR gate, GM_ADDR up, GM_ADDR smooth, GM_ADDR y, GM_ADDR tokenScale,
                              const FusedSwiGLUActQuantNZTilingData *__restrict tiling, TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    m_ = tiling->m;
    n_ = tiling->n;
    colChunk_ = tiling->colChunk;
    rowBlock_ = tiling->rowBlock;
    hasSmooth_ = tiling->hasSmooth;
    // NZ layout (ex-reserved fields, see the tiling header): legacy pre-NZ
    // tilings bake 0s, which decode as ND.
    isNz_ = tiling->isNz;
    m1t_ = tiling->m1t;
    // Legacy-tiling guard: v1 tilings baked reserved fields as 0, and a zero
    // rowBlock would make the block loop never advance. Clamp to the values the
    // host can emit (R in {8,4,2,1}, 0 < colChunk <= n) so a malformed tiling
    // cannot index past absmax[kMaxRowBlock] / scaleStage_[kScaleGroupRows].
    if (rowBlock_ == 0 || rowBlock_ > kMaxRowBlock || (kMaxRowBlock % rowBlock_) != 0) {
      rowBlock_ = 1;
    }
    if (colChunk_ == 0 || colChunk_ > n_) {
      colChunk_ = n_;
    }
    pingpong_ = (colChunk_ < n_);
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

    gateGm_.SetGlobalBuffer((__gm__ half *)gate);
    upGm_.SetGlobalBuffer((__gm__ half *)up);
    yGm_.SetGlobalBuffer((__gm__ int8_t *)y);
    scaleGm_.SetGlobalBuffer((__gm__ float *)tokenScale);
    if (hasSmooth_ != 0) {
      smoothGm_.SetGlobalBuffer((__gm__ half *)smooth);
    }

    // tmpH_ is triple-use: fp16 chain staging (R*cc halves on single-chunk
    // shapes -- v4b's fused chain stages the whole [R, cc] tile at once; cc
    // otherwise), then absmax tree scratch (n halves), then pass-3 staging
    // (cc halves). workF32_ mirrors the chain width (R*cc / cc).
    uint32_t stagingHalves = (colChunk_ == n_) ? (rowBlock_ * colChunk_) : ((n_ > colChunk_) ? n_ : colChunk_);
    uint32_t workF32Elems = (colChunk_ == n_) ? (rowBlock_ * colChunk_) : colChunk_;

    pipe->InitBuffer(gateABuf_, rowBlock_ * colChunk_ * sizeof(half));
    pipe->InitBuffer(upABuf_, rowBlock_ * colChunk_ * sizeof(half));
    if (pingpong_) {
      pipe->InitBuffer(gateBBuf_, rowBlock_ * colChunk_ * sizeof(half));
      pipe->InitBuffer(upBBuf_, rowBlock_ * colChunk_ * sizeof(half));
    }
    pipe->InitBuffer(workF32Buf_, workF32Elems * sizeof(float));
    pipe->InitBuffer(actRowBuf_, rowBlock_ * n_ * sizeof(half));
    pipe->InitBuffer(outI8Buf_, rowBlock_ * n_ * sizeof(int8_t));
    pipe->InitBuffer(tmpHBuf_, stagingHalves * sizeof(half));
    pipe->InitBuffer(maxRowBuf_, rowBlock_ * kSlotStrideF16 * sizeof(half));
    pipe->InitBuffer(scaleStageBuf_, kScaleGroupRows * sizeof(float));
    if (hasSmooth_ != 0) {
      pipe->InitBuffer(smoothBuf_, n_ * sizeof(half));
      if (colChunk_ == n_) {
        pipe->InitBuffer(smooth2xBuf_, rowBlock_ * n_ * sizeof(half));
      }
    }

    gateAH_ = gateABuf_.Get<half>();
    upAH_ = upABuf_.Get<half>();
    if (pingpong_) {
      gateBH_ = gateBBuf_.Get<half>();
      upBH_ = upBBuf_.Get<half>();
    }
    workF32_ = workF32Buf_.Get<float>();
    actRowH_ = actRowBuf_.Get<half>();
    outI8_ = outI8Buf_.Get<int8_t>();
    tmpH_ = tmpHBuf_.Get<half>();
    maxRowH_ = maxRowBuf_.Get<half>();
    scaleStage_ = scaleStageBuf_.Get<float>();
    if (hasSmooth_ != 0) {
      smoothH_ = smoothBuf_.Get<half>();
      if (colChunk_ == n_) {
        smooth2xH_ = smooth2xBuf_.Get<half>();
      }
    }
    LoadSmoothTable();
  }

  __aicore__ inline void Process() {
    if (myRows_ == 0) {
      return;
    }
    // First block's chunk-0 tile is loaded here; every later chunk's tile is
    // issued inside the previous chunk's step (ping-pong) or by the previous
    // block's tail (single pair).
    uint32_t firstRows = (myRows_ < rowBlock_) ? myRows_ : rowBlock_;
    LoadTile(rowStart_, 0, firstRows, gateAH_, upAH_);
    curBuf_ = 0;
    for (uint32_t m0 = rowStart_; m0 < rowStart_ + myRows_; m0 += rowBlock_) {
      uint32_t rowsInBlock = rowStart_ + myRows_ - m0;
      if (rowsInBlock > rowBlock_) {
        rowsInBlock = rowBlock_;
      }
      ProcessBlock(m0, rowsInBlock);
    }
    // Drain the last block's MTE3 (its MTE3_V set has no in-block consumer;
    // the channel must stay 1:1 set/wait, and the copies must land before
    // the kernel's outputs are read).
    if (mte3InFlight_) {
      WaitFlag<HardEvent::MTE3_V>(0);
    }
  }

 private:
  // One-time per core (smooth path): the smooth vector is shared by every
  // row this core processes -- copy it once and keep it resident in UB.
  // v4a's fp16 tail consumes smoothH_ directly (no per-chunk or resident
  // fp32 cast at all). Single-chunk shapes also fill smooth2x_ (the row
  // replicated R times) so v4b's fused final Mul reads one contiguous
  // [R, n] operand; all copies drain under the single MTE2_V sync below.
  __aicore__ inline void LoadSmoothTable() {
    if (hasSmooth_ == 0) {
      return;
    }
    DataCopy(smoothH_, smoothGm_, n_);
    if (colChunk_ == n_) {
      for (uint32_t r = 0; r < rowBlock_; ++r) {
        DataCopy(smooth2xH_[r * n_], smoothGm_, n_);
      }
    }
    PipeBarrier<PIPE_MTE2>();
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
  }

  // GM element offset of the first fp16 column segment of `row` at column
  // fractal jc (see the header NZ paragraph): (jc * m1t + (row >> 4)) * 256
  // + (row & 15) * 16.
  __aicore__ inline uint64_t NzRowBase(uint32_t row, uint32_t jc) const {
    return (static_cast<uint64_t>(jc) * m1t_ + row / kNzFractalRows) * kNzF16FractalElems +
           (row % kNzFractalRows) * kNzF16FractalCols;
  }

  // Same for the int8 y output ([N/32, m1t, 16, 32], 512B fractals):
  // (jc * m1t + (row >> 4)) * 512 + (row & 15) * 32.
  __aicore__ inline uint64_t NzRowBaseI8(uint32_t row, uint32_t jc) const {
    return (static_cast<uint64_t>(jc) * m1t_ + row / kNzFractalRows) * kNzI8FractalElems +
           (row % kNzFractalRows) * kNzI8FractalCols;
  }

  // Batched copy-in of one column chunk for rows [m0, m0+rows) into the
  // given tile pair: the rows' chunk tiles land contiguously striped in the
  // [rows, colChunk] staging tiles. rows is the block's live row count
  // (ragged tail blocks and prefetched blocks load only their live rows --
  // never past the core's row range). NO PipeBarrier: the consumer's
  // SetFlag/WaitFlag<MTE2_V> (enqueued after the copies on the MTE2 queue)
  // is the sync point -- the K1 no-barrier pattern.
  __aicore__ inline void LoadTile(uint32_t m0, uint32_t c0, uint32_t rows, LocalTensor<half> gateDst,
                                  LocalTensor<half> upDst) {
    uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
    if (isNz_ != 0) {
      // NZ gather, one strided copy per row per tensor (the rms-v3 form:
      // chunkLen/16 segments of 32B, srcStride m1t*16-1) -- the ND 2D gapped
      // form cannot express the fractal layout. Covers both the single-
      // chunk and ping-pong paths uniformly.
      DataCopyParams seg(static_cast<uint16_t>(chunkLen / kNzF16FractalCols), 1,
                         static_cast<uint16_t>(m1t_ * kNzFractalBlocks32B - 1), 0);
      uint32_t jc = c0 / kNzF16FractalCols;
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(gateDst[r * colChunk_], gateGm_[NzRowBase(m0 + r, jc)], seg);
      }
      for (uint32_t r = 0; r < rows; ++r) {
        DataCopy(upDst[r * colChunk_], upGm_[NzRowBase(m0 + r, jc)], seg);
      }
      return;
    }
    if (colChunk_ == n_) {
      // Single chunk per row: the [rows, n] tile is one contiguous GM
      // range per tensor (row-major), one plain 1D copy each.
      DataCopy(gateDst, gateGm_[(uint64_t)m0 * n_], rows * n_);
      DataCopy(upDst, upGm_[(uint64_t)m0 * n_], rows * n_);
      return;
    }
    // 2D gapped gather, ONE descriptor per tensor: `rows` blocks of
    // chunkLen fp16, source gap between row starts n - chunkLen elements,
    // destination gap colChunk - chunkLen so each row keeps the colChunk
    // stride of the [rows, colChunk] UB tile. blockLen/gaps are 32B-unit
    // uint16 fields (tiling keeps n, chunkLen, colChunk 32-multiples).
    // Byte-identical to per-row 1D copies (K1 v4 verified form).
    DataCopyParams params(static_cast<uint16_t>(rows), static_cast<uint16_t>(chunkLen * sizeof(half) / kAlignBytes),
                          static_cast<uint16_t>((n_ - chunkLen) * sizeof(half) / kAlignBytes),
                          static_cast<uint16_t>((colChunk_ - chunkLen) * sizeof(half) / kAlignBytes));
    DataCopy(gateDst, gateGm_[(uint64_t)m0 * n_ + c0], params);
    DataCopy(upDst, upGm_[(uint64_t)m0 * n_ + c0], params);
  }

  // Pass 1 on one (row, chunk): act = silu(gate) * up (* smooth) into the
  // row-resident actRow tile (fp16). v4a fp16 tail: the f32 stage is just
  // the exact negation Cast-up + in-place Exp -- e32 is bit-identical to
  // v3's (the 310P exp-floor signature carries over) -- then the
  // reciprocal + product chain runs on the fp16 pipe with native vrec<half>
  // (gelu v4h tail, measured native ~1 unit). tmpH_ is the fp16 staging
  // (n16 -> e16 -> d16 -> r16 -> s16 -> t16, each in-place beat's sources
  // consumed by the previous barrier-ordered beat); the final Mul writes
  // actRow directly.
  __aicore__ inline void ChainChunk(uint32_t r, uint32_t c0, uint32_t chunkLen, LocalTensor<half> gateSrc,
                                    LocalTensor<half> upSrc) {
    Muls(tmpH_, gateSrc[r * colChunk_], static_cast<half>(kNegOne), chunkLen);  // n = -g (exact)
    PipeBarrier<PIPE_V>();
    Cast(workF32_, tmpH_, RoundMode::CAST_NONE, chunkLen);  // n -> f32 (exact)
    PipeBarrier<PIPE_V>();
    Exp(workF32_, workF32_, chunkLen);  // e32 (== v3, floored)
    PipeBarrier<PIPE_V>();
    Cast(tmpH_, workF32_, RoundMode::CAST_NONE, chunkLen);  // e16
    PipeBarrier<PIPE_V>();
    Adds(tmpH_, tmpH_, static_cast<half>(kOne), chunkLen);  // d = 1 + e16
    PipeBarrier<PIPE_V>();
    Reciprocal(tmpH_, tmpH_, chunkLen);  // r = vrec(d16)
    PipeBarrier<PIPE_V>();
    Mul(tmpH_, tmpH_, gateSrc[r * colChunk_], chunkLen);  // s = g * r
    PipeBarrier<PIPE_V>();
    if (hasSmooth_ != 0) {
      Mul(tmpH_, tmpH_, upSrc[r * colChunk_], chunkLen);  // t = s * u
      PipeBarrier<PIPE_V>();
      Mul(actRowH_[r * n_ + c0], tmpH_, smoothH_[c0], chunkLen);  // act = t * smooth
    } else {
      Mul(actRowH_[r * n_ + c0], tmpH_, upSrc[r * colChunk_], chunkLen);  // act = s * u
    }
    PipeBarrier<PIPE_V>();
  }

  // v4b fused chain (single-chunk shapes, 2+ live rows): the block's whole
  // [rows, n] gate/up tiles are contiguous, so ALL rows run through the
  // v4a chain as ONE wide beat set (9 beats at width rows*n instead of 9
  // beats per row). Element-wise beats are count-insensitive, so every
  // element sees bit-identical arithmetic to ChainChunk -- only the issue +
  // barrier-marker cost is paid once per block. actRow's [rows, n] block
  // slice and the R-replicated smooth2x_ table line up as plain contiguous
  // operands. Ragged blocks with a single live row take ChainChunk instead.
  __aicore__ inline void ChainBlockFused(uint32_t rows, LocalTensor<half> gateSrc, LocalTensor<half> upSrc) {
    uint32_t len = rows * n_;
    Muls(tmpH_, gateSrc, static_cast<half>(kNegOne), len);  // n = -g (exact)
    PipeBarrier<PIPE_V>();
    Cast(workF32_, tmpH_, RoundMode::CAST_NONE, len);  // n -> f32 (exact)
    PipeBarrier<PIPE_V>();
    Exp(workF32_, workF32_, len);  // e32 (== v3, floored)
    PipeBarrier<PIPE_V>();
    Cast(tmpH_, workF32_, RoundMode::CAST_NONE, len);  // e16
    PipeBarrier<PIPE_V>();
    Adds(tmpH_, tmpH_, static_cast<half>(kOne), len);  // d = 1 + e16
    PipeBarrier<PIPE_V>();
    Reciprocal(tmpH_, tmpH_, len);  // r = vrec(d16)
    PipeBarrier<PIPE_V>();
    Mul(tmpH_, tmpH_, gateSrc, len);  // s = g * r
    PipeBarrier<PIPE_V>();
    if (hasSmooth_ != 0) {
      Mul(tmpH_, tmpH_, upSrc, len);  // t = s * u
      PipeBarrier<PIPE_V>();
      Mul(actRowH_, tmpH_, smooth2xH_, len);  // act = t * smooth
    } else {
      Mul(actRowH_, tmpH_, upSrc, len);  // act = s * u
    }
    PipeBarrier<PIPE_V>();
  }

  // Whole-row absmax: |act| into tmpH_, in-place halving Max fold from
  // span = n down to ONE value (v1's ReduceMax and v2's 8-slot tree compute
  // the same exact max -- fp16 Abs/Max are exact and order-independent, and
  // every fold operand is a per-subset max of this row's |act|). Odd spans
  // fold their dangling last partial into slot 0 first, so no pow2 zero pad
  // is needed. The single result lands in the row's maxRow slot (tmpH_ is
  // reused by the next row's tree and by pass-3 staging).
  __aicore__ inline void AbsMaxRow(uint32_t r) {
    Abs(tmpH_, actRowH_[r * n_], n_);
    PipeBarrier<PIPE_V>();
    uint32_t span = n_;
    while (span > 1) {
      if ((span & 1) != 0) {
        Max(tmpH_, tmpH_, tmpH_[span - 1], 1);
        PipeBarrier<PIPE_V>();
        --span;
      }
      span >>= 1;
      Max(tmpH_, tmpH_, tmpH_[span], span);
      PipeBarrier<PIPE_V>();
    }
    Adds(maxRowH_[r * kSlotStrideF16], tmpH_, static_cast<half>(0), 1);
    PipeBarrier<PIPE_V>();
  }

  // token_scale[row] = absmax / 127, staged per 8-row group; the flush
  // happens in the block's MTE3 batch whenever the group completes. A ragged
  // final group (< kScaleGroupRows rows) is written straight to GM with 4B
  // scalar stores: flushing it as a whole 32B block would write past [M].
  __aicore__ inline void StageTokenScale(uint32_t row, float absmax) {
    float scaleOut = absmax / kQuantMaxValue;
    if (m_ - groupRowBase_ < kScaleGroupRows) {
      scaleGm_.SetValue(row, scaleOut);
      return;
    }
    uint32_t slot = row - groupRowBase_;
    Duplicate(scaleStage_[slot], scaleOut, 1);
    PipeBarrier<PIPE_V>();
  }

  // Pass 3 on one (row, chunk): y = int8(act * 127/absmax) - fp16 Muls +
  // CAST_ROUND (the 310P DynamicQuant quant tail, v1 verbatim). tmpH_
  // doubles as the fp16 staging (the absmax trees are done by then);
  // gate/up tiles may already hold prefetched data, so they must NOT be
  // used as staging.
  __aicore__ inline void Pass3Chunk(uint32_t r, uint32_t c0, uint32_t chunkLen, float quantMul) {
    Muls(tmpH_, actRowH_[r * n_ + c0], static_cast<half>(quantMul), chunkLen);
    PipeBarrier<PIPE_V>();
    Cast(outI8_[r * n_ + c0], tmpH_, RoundMode::CAST_ROUND, chunkLen);
    PipeBarrier<PIPE_V>();
  }

  // Prefetch the next block's chunk-0 tile into the given pair (if this
  // core has another block); it drains under the tree/quant/MTE3 tail of
  // the current block.
  __aicore__ inline void PrefetchNextBlock(uint32_t m0, LocalTensor<half> gateDst, LocalTensor<half> upDst) {
    uint32_t nextM0 = m0 + rowBlock_;
    if (nextM0 < rowStart_ + myRows_) {
      uint32_t nextRows = rowStart_ + myRows_ - nextM0;
      if (nextRows > rowBlock_) {
        nextRows = rowBlock_;
      }
      LoadTile(nextM0, 0, nextRows, gateDst, upDst);
    }
  }

  // ---------------------------------------------------------------------------
  // One row block (R rows starting at m0):
  //   chunk loop      : per-chunk MTE2 wait -> NEXT tile issue (ping-pong)
  //                     -> per-row activation chain -> tile handback
  //   absmax rows     : |act| + fold-to-1 max tree, ONE V_S per block
  //   scale + pass 3  : token_scale staging + int8 quant into the y tile
  //   batched MTE3    : contiguous y block (+ any completed 8-row scale
  //                     group)
  // ---------------------------------------------------------------------------
  __aicore__ inline void ProcessBlock(uint32_t m0, uint32_t rowsInBlock) {
    if (pingpong_) {
      // PING-PONG (colChunk < n): the next chunk's tile is issued BEFORE
      // this chunk's chain, into the pair handed back one chunk ago, so the
      // MTE2 drain overlaps the chain on the vector pipe. The scalar-side
      // V_MTE2 handback wait at the end of chunk k orders the reload of
      // pair(k) at chunk k+1's pre-chain step.
      for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
        uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
        LocalTensor<half> gateCur = (curBuf_ == 0) ? gateAH_ : gateBH_;
        LocalTensor<half> upCur = (curBuf_ == 0) ? upAH_ : upBH_;
        LocalTensor<half> gateNext = (curBuf_ == 0) ? gateBH_ : gateAH_;
        LocalTensor<half> upNext = (curBuf_ == 0) ? upBH_ : upAH_;
        // This chunk's tile was issued one chunk ago (or by the prologue).
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        uint32_t nextC0 = c0 + colChunk_;
        if (nextC0 < n_) {
          LoadTile(m0, nextC0, rowsInBlock, gateNext, upNext);
        } else {
          // Last chunk of the block: prefetch the next block's chunk 0
          // (drains under the tree/quant/MTE3 tail).
          PrefetchNextBlock(m0, gateNext, upNext);
        }
        for (uint32_t r = 0; r < rowsInBlock; ++r) {
          ChainChunk(r, c0, chunkLen, gateCur, upCur);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        curBuf_ ^= 1;
      }
    } else {
      // SINGLE PAIR (colChunk == n, one chunk per block): chunk 0 was
      // prefetched by the previous block's tail (or the prologue). The next
      // block's prefetch reuses the SAME pair, so it must follow the
      // handback (no second pair to overlap with). v4b: with 2+ live rows
      // the whole [rows, n] tile takes the fused chain (bit-identical,
      // once-per-block issue cost); a lone ragged row takes ChainChunk.
      for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
        uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        if (rowsInBlock >= kFusedChainMinRows) {
          ChainBlockFused(rowsInBlock, gateAH_, upAH_);
        } else {
          ChainChunk(0, c0, chunkLen, gateAH_, upAH_);
        }
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        if (c0 + colChunk_ >= n_) {
          PrefetchNextBlock(m0, gateAH_, upAH_);
        }
      }
    }

    AbsMaxQuantBlock(m0, rowsInBlock);
    Mte3Block(m0, rowsInBlock);
  }

  // Per-row absmax trees, then ONE scalar round-trip for the block
  // (fold-to-1: a single GetValue per row), then the previous block's MTE3
  // drain wait, token_scale staging and the pass-3 int8 quant.
  __aicore__ inline void AbsMaxQuantBlock(uint32_t m0, uint32_t rowsInBlock) {
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      AbsMaxRow(r);
    }
    event_t evMax = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evMax);
    WaitFlag<HardEvent::V_S>(evMax);
    float absmax[kMaxRowBlock];
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      absmax[r] = static_cast<float>(maxRowH_.GetValue(r * kSlotStrideF16));
    }

    // The previous block's MTE3 (y + scale copies) may still be draining:
    // outI8_ and scaleStage_ must not be rewritten until it lands. Their
    // next writers (pass-3 Casts / scale Duplicates) are the first V ops
    // below, so the whole chunk loop + tree phase overlapped the drain.
    if (mte3InFlight_) {
      WaitFlag<HardEvent::MTE3_V>(0);
      mte3InFlight_ = false;
    }

    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      StageTokenScale(m0 + r, absmax[r]);
    }
    for (uint32_t r = 0; r < rowsInBlock; ++r) {
      float quantMul = (absmax[r] > kQuantEpsilon) ? (kQuantMaxValue / absmax[r]) : 0.0f;
      for (uint32_t c0 = 0; c0 < n_; c0 += colChunk_) {
        uint32_t chunkLen = (c0 + colChunk_ > n_) ? (n_ - c0) : colChunk_;
        Pass3Chunk(r, c0, chunkLen, quantMul);
      }
    }
  }

  // Batched MTE3 out. ND: the whole [rows, n] y block is ONE contiguous
  // copy, plus the current 8-row scale group whenever it completes (R <= 8
  // -> at most one group per block; the ragged final group is written with
  // 4B scalar stores instead) -- at most 2 copies per gate (310P MTE3
  // copy-loss law: <= 4 descriptors per burst, all 1D, never at queue
  // head -- the first MTE3 work follows the first block's compute).
  // NZ: per-row strided scatters (the rms-v3 inverse of the gather); with
  // R rows plus the scale flush the descriptor count exceeds 4 at R >= 4
  // shapes, so sub-bursts of <= kMte3Gate take an immediate queue drain
  // and only the FINAL sub-burst keeps the deferred drain. Deployment
  // shapes (R <= 2) stay in one all-deferred sub-burst of <= 3.
  __aicore__ inline void Mte3Block(uint32_t m0, uint32_t rowsInBlock) {
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    bool groupDone = false;
    uint32_t lastRow = m0 + rowsInBlock;
    if (lastRow - groupRowBase_ >= kScaleGroupRows || lastRow == rowStart_ + myRows_) {
      groupDone = true;
    }
    if (isNz_ != 0) {
      DataCopyParams segI8(static_cast<uint16_t>(n_ / kNzI8FractalCols), 1, 0,
                           static_cast<uint16_t>(m1t_ * kNzFractalBlocks32B - 1));
      // Total descriptors this gate: one y scatter per row + the scale
      // flush; drain between sub-bursts, defer the last.
      uint32_t total = rowsInBlock + (groupDone ? 1 : 0);
      uint32_t queued = 0;
      for (uint32_t r = 0; r < rowsInBlock; ++r) {
        DataCopy(yGm_[NzRowBaseI8(m0 + r, 0)], outI8_[r * n_], segI8);
        ++queued;
        if (queued == kMte3Gate && r + 1 < total) {
          SetFlag<HardEvent::MTE3_V>(0);
          WaitFlag<HardEvent::MTE3_V>(0);
          queued = 0;
        }
      }
      if (groupDone && (m_ - groupRowBase_ >= kScaleGroupRows)) {
        DataCopy(scaleGm_[groupRowBase_], scaleStage_, kScaleGroupRows);
      }
    } else {
      DataCopy(yGm_[(uint64_t)m0 * n_], outI8_, rowsInBlock * n_);
      if (groupDone && (m_ - groupRowBase_ >= kScaleGroupRows)) {
        DataCopy(scaleGm_[groupRowBase_], scaleStage_, kScaleGroupRows);
      }
    }
    if (groupDone) {
      groupRowBase_ += kScaleGroupRows;
    }
    // Deferred drain: consumed before the next block's scale staging (or
    // the post-loop drain in Process). Keeps MTE3_V 1:1 set/wait.
    SetFlag<HardEvent::MTE3_V>(0);
    mte3InFlight_ = true;
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint32_t m_ = 0;
  uint32_t n_ = 0;
  uint32_t colChunk_ = 0;
  uint32_t rowBlock_ = 1;
  uint32_t hasSmooth_ = 0;
  uint32_t isNz_ = 0;  // 1 = gate/up/y are FRACTAL_NZ storage
  uint32_t m1t_ = 0;   // NZ 16-row storage blocks (row >> 4 lattice)
  uint32_t rowStart_ = 0;
  uint32_t myRows_ = 0;
  uint32_t groupRowBase_ = 0;
  uint32_t curBuf_ = 0;
  bool pingpong_ = false;
  bool mte3InFlight_ = false;

  GlobalTensor<half> gateGm_;
  GlobalTensor<half> upGm_;
  GlobalTensor<half> smoothGm_;
  GlobalTensor<int8_t> yGm_;
  GlobalTensor<float> scaleGm_;

  TBuf<TPosition::VECCALC> gateABuf_;
  TBuf<TPosition::VECCALC> upABuf_;
  TBuf<TPosition::VECCALC> gateBBuf_;
  TBuf<TPosition::VECCALC> upBBuf_;
  TBuf<TPosition::VECCALC> workF32Buf_;
  TBuf<TPosition::VECCALC> actRowBuf_;
  TBuf<TPosition::VECCALC> outI8Buf_;
  TBuf<TPosition::VECCALC> tmpHBuf_;
  TBuf<TPosition::VECCALC> maxRowBuf_;
  TBuf<TPosition::VECCALC> scaleStageBuf_;
  TBuf<TPosition::VECCALC> smoothBuf_;
  TBuf<TPosition::VECCALC> smooth2xBuf_;

  LocalTensor<half> gateAH_;
  LocalTensor<half> upAH_;
  LocalTensor<half> gateBH_;
  LocalTensor<half> upBH_;
  LocalTensor<float> workF32_;
  LocalTensor<half> actRowH_;
  LocalTensor<int8_t> outI8_;
  LocalTensor<half> tmpH_;
  LocalTensor<half> maxRowH_;
  LocalTensor<float> scaleStage_;
  LocalTensor<half> smoothH_;
  LocalTensor<half> smooth2xH_;
};

}  // namespace FusedSwiGLUActQuantNZ

#endif  // FUSED_SWI_GLU_ACT_QUANT_NZ_H
