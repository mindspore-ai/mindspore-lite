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

#ifndef FUSED_GELU_TANH_NZ_H
#define FUSED_GELU_TANH_NZ_H

#include "kernel_operator.h"                 // NOLINT(build/include_subdir)
#include "fused_gelu_tanh_nz_tiling_data.h"  // NOLINT(build/include_subdir)

using namespace AscendC;  // NOLINT(build/namespaces)

namespace FusedGeluTanhNZ {

// ---------------------------------------------------------------------------
// FusedGeluTanhNZ: single-kernel gelu_pytorch_tanh for Atlas 300I Duo
// (dav_m200). Pure-vector kernel (AICore task, vector pipe only, no Cube).
//
// Math (exp reformulation of gelu_pytorch_tanh, math-identical via
// tanh(y) = 2/(1+e^(-2y)) - 1; same recipe as CANN GeluV2(approximate="tanh")
// and gelu_quant_base.h::ComputeGeluTanh):
//   y = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3)))
// v4h: the v4c chain with the ONE scalar head op demoted to a plain tensor
// op against a hoisted constant tensor. The exponent argument is the
// Horner refold (v4a/v4c),
//   w = x * (A + B * x^2),  A = -sqrt(8/pi),  B = -sqrt(8/pi)*0.044715,
// with the B line computed as a TENSOR Mul against a per-core constant
// tensor (Duplicate-filled once) instead of the scalar-broadcast
// Muls<half>; the A line stays scalar Adds<half>.
// v5: the v4h chain VERBATIM (same 10 instructions, same issue order, so
// bit-identical arithmetic) with the SYNC model rebuilt -- zero intra-V
// PipeBarriers and the MTE2 prefetch repositioned to run under the vector
// chain (see the Process loop note). v4h kept one PipeBarrier<PIPE_V> per
// chain op (the conservative gelu_quant_base.h discipline): each stalls
// the scalar issue thread until the pipe drains, so the 10-op chain issued
// one instruction at a time and the loop's prefetch -- which follows the
// chain in issue order -- could never get under it; every chunk paid its
// ~0.12us MTE2 copy as a serial phase (the 19.9us in-window gap at
// 4096^2: window 628.8 - vec 608.9). The dav_m200 V pipe executes in
// order, so same-pipe RAW chains need no drain -- the repo-wide zero-
// barrier discipline proven bit-identical at scale in sibling kernels
// (same count-form vector APIs).
//
// Cost model (device-calibrated instruction UNIT model, 1 unit = one plain
// fp16 tensor instruction per chunk, ~0.21us at 4096^2/8 cores; fitted to
// every measured vec time v2 837 / v3 864 / v3b 675 / v4a ~675 flat /
// v4c 648 / v4h 609):
//   plain fp16 tensor op = 1; f32 op (tensor or scalar) = 2;
//   fp16 scalar ops (the Muls and Adds scalar forms) cost 2 units each:
//   they run at f32-like throughput, which is why v4a's scalar refold
//   benched exactly flat -- its head adds up to 4+2s units, v3b's 17+s,
//   and the two meet at s=2;
//   the fp16 reciprocal is native at cost 1 unit (v4c measured -4% vs v4a)
//   vexp<half>/vdiv<half> ~= 7.6 and vmadd<half> ~= 7 (emulated/compiled
//   out -- v3's +2.6% regression, and v4g's FMA-head attempt benched +7.5%
//   WORSE than v4c: only vrec<half> among the special forms is
//   native).
// Chain cost: v3b/v4a 19 units, v4c 18, v4h 15 modeled (head Mul + tensor
// Mul + scalar Adds + Mul = 5, tail Cast-up + Exp + Cast-down + Adds<half>
// + vrec<half> + Mul = 10). v4h MEASURED (msprof @4096^2): op task
// 743.4 -> 670.8us (-9.8% vs v3b), vec 675 -> 608.9us = 17.2 EFFECTIVE
// units -- the model's direction held in every A/B (v4c < v4a, v4h < v4c,
// v4g worst) but it over-predicts small swaps: a hoisted const tensor also
// adds a 2B/elem UB read, and issue/dependency serialization is a real
// floor. The follow-up v4j (both remaining scalars hoisted, 20B/elem
// forcing chunk 9728) benched +2% WORSE than v4h -- marginal returns go
// negative after the first hoist; the chain-restructuring avenue is
// closed (see the README v4 section). The one extra const tensor costs
// 2B/elem (16B total vs v3b's 14B), 12288-elem chunks = 196608B, i.e. the
// full 192KB dav_m200 UB budget (v1/v2 already ran 16B/elem there, and the
// tiling re-checks the footprint against the platform-reported UB size);
// chunk size is
// unchanged since v2, so the bench A/B against v3b/v4c isolates the chain.
// Precision: BIT-IDENTICAL to v4c -- the tensor Mul multiplies by the same
// fp16 value the scalar Muls cast to (the scalar is pre-cast to the tensor
// dtype in both cases), and everything else is v4c verbatim; v4c device
// benched 5/5 PASS with per-case max|d| 0.0156/0.0156/0.0234/0.0234/128,
// exactly the calibrated simulation's prediction (the binding max|d| sits
// in the Exp-floor region, x >= ~2.9, where E is floored to 2^-9 and the
// output is path independent: 2^-9 and 1+2^-9 are both fp16-exact).
// Extreme |x| (x^2 overflow fp16 -> t = inf -> B*t = -inf) lands on the
// same floored-E values as v1/v2/v3b (x>0: E floored to 2^-9; x<0:
// E = inf, R = 0, y = -0).
//
// Layout: the kernel is layout-agnostic. x and y share shape and format
// (FRACTAL_NZ at the deployment site, ND accepted), and for an elementwise
// map both storages are dense [totalElems] buffers with one-to-one flat
// positions, so the kernel simply streams flat chunks: GM -> UB (fp16),
// fp16/fp32 hybrid compute, UB (fp16) -> GM at the same flat offset. NZ fractals are
// 256 contiguous elements each and chunk boundaries are 256-multiples, so
// every DataCopy stays 32B aligned (dav_m200 has no DataCopyPad).
//
// Tile model: chunks striped over cores. x and y each stage in TWO ping-pong
// UB buffers so the three pipes overlap (v2; v1 ran them back to back):
// while V computes chunk i, MTE2 prefetches chunk i+1 and MTE3 drains
// chunk i-1's output (structural since v5: the prefetch is issued before
// the V_MTE3 wait so it queues under the chain; in v2-v4h the per-op
// chain barriers pushed its issue past the chain, and only the MTE3 copy
// hid, under the NEXT chain). Measured on v2 (msprof, 4096x4096, device 2):
// task 915.7us with vec 837us hidden under it (ratio-sum 1.27, copies
// overlapped)
// -- the remaining wall is the vector chain's UB traffic itself (v1/v2 move
// 100B/elem), so v3b cuts that traffic: 72B/elem by running the poly head in
// fp16 while keeping the fp32 tail native (the all-fp16 v3 measured +2.6%
// WORSE: vexp<half>/vdiv<half> are emulated on dav_m200, see the chain note).
// UB buffers are partitioned by pipe, following the fused_swiglu_act_quant_nz
// discipline. The xH and yH buffers each come in a ping-pong pair: MTE2 fills
// one xH buffer while the vector pipe reads the other, and the vector pipe
// writes one yH buffer while MTE3 drains the other, so the inbound copy of the
// next chunk overlaps the compute of the current one and the outbound copy of
// the previous result. The tmpH buffer is fp16 scratch for the poly result,
// then the fp16 reciprocal; the tmpF32 buffer is fp32 scratch for z, E, the
// denominator and the fp32 reciprocal, and MTE never touches either scratch
// buffer. The last buffer is constBB, the hoisted fp16 constant tensor,
// duplicated once per core in the Process prologue, read by every chunk's
// B-line tensor multiply, never touched by MTE and never rewritten.
// Do NOT stage the output in xH_ in place: the first bench run of that draft
// corrupted every chunk after the first per core -- WaitFlag<MTE3_V> only
// orders the vector pipe, and the next iteration's MTE2 copy-in is an
// MTE2-queue op that no V-side wait guards, so it overwrote xH_ mid-flight.
// Four event channels, each consumed strictly FIFO (1:1 set/wait), pair the
// pipes across iterations (chunk i lives in buffer i%2):
//   MTE2_V  copy-in(i) landed  -> V may compute chunk i
//   V_MTE3  compute(i) landed  -> MTE3 may read yH_[i%2]
//   MTE3_V  copy-out(i) drained-> V(i+2) may rewrite yH_[i%2]
//   V_MTE2  V(i) read xH_[i%2] -> MTE2(i+2) may rewrite xH_[i%2]
// Each MTE3 burst stays ONE 1D copy wrapped in its gate (the 310P MTE3
// copy-loss law: <=4 copies per four-flag gate, all 1D, never at queue
// head) and the first MTE3 work follows the first compute, so the
// queue-head position is not a store.
// ---------------------------------------------------------------------------

// Constants. kB is materialized ONCE per core as an fp16 constant TENSOR
// (Duplicate(half(kB)) in the Process prologue) so the head's B-line is a
// plain tensor Mul (1 unit) instead of a scalar-broadcast Muls<half>
// (modeled 2 units); the tensor value is the same fp16 rounding the
// scalar cast produced, so v4h is bit-identical to v4c. kA/kOne stay
// scalars: hoisting them too was tried as v4j (3 const tensors = 20B/elem,
// chunk 9728) and benched +2% WORSE -- the extra const reads and chunk
// boundaries eat the scalar-op savings, so after the first hoist the
// marginal return is negative (see the README v4 section). The fp16
// roundings of kA
// (rel err ~4.1e-5) and kB (~8.6e-4) perturb z by <= ~1e-3, which the
// exp-reform sensitivity damps (dy/y = E/(1+E)*dz) and the calibrated
// chain simulation covered before the device bench (see the header note).
constexpr float kA = -1.5957691216057308f;   // -sqrt(8/pi) = -2*sqrt(2/pi)
constexpr float kB = -0.07135481627260025f;  // kA * 0.044715 (beta folded in)
constexpr float kOne = 1.0f;
constexpr uint32_t kPingPongBuffers = 2;  // ping-pong staging buffers for x and y

class FusedGeluTanhNZ {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const FusedGeluTanhNZTilingData *__restrict tiling, TPipe *pipe) {
    blockIdx_ = GetBlockIdx();
    coreNum_ = tiling->coreNum;
    totalElems_ = tiling->totalElems;
    chunkElems_ = tiling->chunkElems;
    // Chunk striping: chunk boundaries are integers, so every core's chunk
    // range is disjoint and every chunk stays 256-element aligned. All of it
    // stays in the 64-bit domain: totalElems_ is the full storage product, so
    // the chunk count, the chunk range and every GM offset below are 64-bit
    // and cannot wrap at 2^32 elements.
    const uint64_t totalChunks = (totalElems_ + chunkElems_ - 1) / chunkElems_;
    chunkStart_ = static_cast<uint64_t>(blockIdx_) * totalChunks / coreNum_;
    chunkEnd_ = (static_cast<uint64_t>(blockIdx_) + 1) * totalChunks / coreNum_;

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));

    // Ping-pong x/y (2 buffers each) + fp16 and fp32 V scratch + the one
    // hoisted fp16 constant tensor: 16B per element (2*2 + 2*2 + 2 + 4 + 2).
    // 12288-elem chunks fill 196608B, exactly the 192KB dav_m200 UB budget
    // (v1/v2's fp32 work/tmp pair already ran 16B/elem = 192KB; v3b/v4c cut
    // it to 14B = 172KB, v4h spends 2B of that on the const tensor; the
    // tiling rejects any chunk whose footprint exceeds the platform UB).
    // Chunk size is unchanged since v2 so the bench A/B isolates the chain.
    pipe->InitBuffer(xBuf_[0], chunkElems_ * sizeof(half));
    pipe->InitBuffer(xBuf_[1], chunkElems_ * sizeof(half));
    pipe->InitBuffer(yBuf_[0], chunkElems_ * sizeof(half));
    pipe->InitBuffer(yBuf_[1], chunkElems_ * sizeof(half));
    pipe->InitBuffer(tmpHBuf_, chunkElems_ * sizeof(half));
    pipe->InitBuffer(tmpF32Buf_, chunkElems_ * sizeof(float));
    pipe->InitBuffer(constBBuf_, chunkElems_ * sizeof(half));

    xH_[0] = xBuf_[0].Get<half>();
    xH_[1] = xBuf_[1].Get<half>();
    yH_[0] = yBuf_[0].Get<half>();
    yH_[1] = yBuf_[1].Get<half>();
    tmpH_ = tmpHBuf_.Get<half>();
    tmpF32_ = tmpF32Buf_.Get<float>();
    constBB_ = constBBuf_.Get<half>();
  }

  __aicore__ inline void Process() {
    const uint64_t nChunks = chunkEnd_ - chunkStart_;
    if (nChunks == 0) {
      return;
    }
    // One-time per-core constant fill: the head's B lives as an fp16
    // constant TENSOR so the B-line is a plain tensor Mul (1 unit) instead
    // of a scalar-broadcast Muls<half> (measured 2 units -- see the header
    // unit-model note). Duplicate lowers to one vector_dup; the fill cost
    // amortizes to zero over the chunks. No barrier before the first
    // chunk's tensor Mul (v5): both are V-pipe ops and the V pipe executes
    // in order, so the in-issue-order RAW is already ordered.
    Duplicate(constBB_, static_cast<half>(kB), static_cast<int32_t>(chunkElems_));
    // Prologue: chunk 0 -> buffer 0. Buffers alternate per chunk; with two
    // buffers the in-loop prefetch of chunk i+1 overlaps the compute of
    // chunk i (the copy targets the buffer chunk i-1 released).
    CopyIn(chunkStart_, xH_[0]);
    for (uint64_t i = 0; i < nChunks; ++i) {
      const uint32_t b = static_cast<uint32_t>(i & 1u);
      const uint64_t chunk = chunkStart_ + i;
      // V: chunk i's data landed; yH_[b] drained (by chunk i-2's copy-out).
      WaitFlag<HardEvent::MTE2_V>(0);
      if (i >= kPingPongBuffers) {
        WaitFlag<HardEvent::MTE3_V>(0);
      }
      ComputeChain(xH_[b], yH_[b], LenOf(chunk));
      SetFlag<HardEvent::V_MTE3>(0);  // yH_[b] ready for MTE3
      SetFlag<HardEvent::V_MTE2>(0);  // xH_[b] free for the i+2 copy-in
      // MTE2: prefetch chunk i+1 into the other buffer, issued BEFORE the
      // V_MTE3 wait (v5). With zero intra-V barriers the chain issues in
      // ~an instant, so this copy starts while the V pipe is still chewing
      // chunk i and lands well inside the chain time -- by the time the
      // loop comes around, the loop-top WaitFlag<MTE2_V> is already
      // satisfied. v4h issued the prefetch AFTER the chain (and the V_MTE3
      // drain), so every chunk paid the copy as a serial phase. The guard
      // consumes the PREVIOUS iteration's V_MTE2 flag (V(i-1) finished
      // reading this buffer), which has long fired; this iteration's own
      // flag stays queued for the i+2 guard (FIFO per channel).
      if (i + 1 < nChunks) {
        if (i + 1 >= kPingPongBuffers) {
          WaitFlag<HardEvent::V_MTE2>(0);
        }
        CopyIn(chunkStart_ + i + 1, xH_[(i + 1) & 1u]);
      }
      // MTE3: one gated 1D copy-out (MTE3 copy-loss law compliant). The
      // wait consumes this iteration's own V_MTE3 flag (fires when the
      // chain completes); the copy then overlaps the next chunk's chain.
      WaitFlag<HardEvent::V_MTE3>(0);
      CopyOut(chunk, yH_[b]);
    }
    // Channel balance (1:1 set/wait): V_MTE2 and MTE3_V each end with up to
    // two unmatched sets -- the last two chunks have no successor compute /
    // copy-in to consume them. Drain them so no channel exits unbalanced.
    const uint32_t drains = (nChunks > kPingPongBuffers) ? kPingPongBuffers : static_cast<uint32_t>(nChunks);
    for (uint32_t k = 0; k < drains; ++k) {
      WaitFlag<HardEvent::V_MTE2>(0);
      WaitFlag<HardEvent::MTE3_V>(0);
    }
  }

 private:
  // Elements of chunk `chunk` (clamped for the ragged tail chunk).
  __aicore__ inline uint32_t LenOf(uint64_t chunk) const {
    uint64_t gmOfs = chunk * chunkElems_;
    uint32_t len = chunkElems_;
    if (gmOfs + len > totalElems_) {
      len = static_cast<uint32_t>(totalElems_ - gmOfs);
    }
    return len;
  }

  __aicore__ inline void CopyIn(uint64_t chunk, const LocalTensor<half> &dst) {
    // Zero-barrier load: the SetFlag is queued on the MTE2
    // pipe behind the DataCopy, so it fires exactly when the copy lands --
    // the consumer-point WaitFlag<MTE2_V> is the only sync needed. The
    // v4h PipeBarrier<MTE2> here only stalled the scalar issue thread
    // until the copy landed (prefetch turned inline load).
    DataCopy(dst, xGm_[chunk * chunkElems_], LenOf(chunk));
    SetFlag<HardEvent::MTE2_V>(0);
  }

  __aicore__ inline void CopyOut(uint64_t chunk, const LocalTensor<half> &src) {
    DataCopy(yGm_[chunk * chunkElems_], src, LenOf(chunk));
    SetFlag<HardEvent::MTE3_V>(0);
  }

  // The v4h compute chain, VERBATIM arithmetic (v5 keeps every
  // instruction and the issue order; only the sync between them changed):
  // v4c plus the B-line as a plain tensor Mul against the hoisted
  // constant tensor. 10 instructions. Unit costs (device calibrated, see
  // the header note): head Mul(1) + Mul-tensor(1) + Adds<half>(2,
  // measured) + Mul(1); tail Cast-up(2) + Exp(2, native vexp<f32>) +
  // Cast-down(2) + Adds<half>(2, measured) + Reciprocal<half>(1, measured
  // native in v4c) + final Mul(1) = 15 units vs v3b/v4a 19 and v4c 18.
  // Bit-identical to v4c/v4h: the tensor Mul multiplies by the same fp16
  // constant value the scalar Muls pre-cast to. The final Mul consumes xH
  // directly in fp16 -- no x->fp32 cast at all. In-place dst==src0
  // patterns follow the official recipe and v1..v4c's proven usage.
  // ZERO intra-V barriers (v5): every dependency in the chain is
  // same-pipe RAW and the dav_m200 V pipe executes in order, so issue
  // order alone orders the chain -- the repo-wide discipline proven
  // bit-identical at scale in sibling kernels (their chains
  // are the same count-form vector APIs, incl. Cast; Exp/Reciprocal are
  // native V-pipe forms here). CANN's gelu_quant_base.h keeps per-op
  // barriers as a portable conservative default; on this arch they only
  // cost issue throughput (see the header v5 note).
  __aicore__ inline void ComputeChain(const LocalTensor<half> &xH, const LocalTensor<half> &yH, uint32_t len) {
    //   w  = x*(A + B*x^2) = -sqrt(8/pi)*(x + beta*x^3)   (fp16; inf at
    //       |x|>255.99 (x^2 overflow) lands on the same floored-E values
    //       as v1/v2/v3b)
    //   R  = 1/(1+exp(w))             (fp32 Exp, then fp16 tail; E floored
    //       to 2^-9 -- 2^-9 and 1+2^-9 are both fp16-exact, so the floored
    //       region is bit-stable across the fp16 tail)
    //   y  = x * R                    (fp16)
    Mul(tmpH_, xH, xH, len);                           // t = x^2 (fp16)
    Mul(tmpH_, tmpH_, constBB_, len);                  // B*t (fp16 tensor Mul vs hoisted const)
    Adds(tmpH_, tmpH_, static_cast<half>(kA), len);    // v = A + B*t (fp16)
    Mul(tmpH_, xH, tmpH_, len);                        // w = x*v (fp16)
    Cast(tmpF32_, tmpH_, RoundMode::CAST_NONE, len);   // w -> fp32 (z)
    Exp(tmpF32_, tmpF32_, len);                        // E = e^z (fp32 vexp, native)
    Cast(tmpH_, tmpF32_, RoundMode::CAST_NONE, len);   // Eh = E -> fp16
    Adds(tmpH_, tmpH_, static_cast<half>(kOne), len);  // D = 1 + Eh (fp16)
    Reciprocal(tmpH_, tmpH_, len);                     // R = 1/D (fp16 vrec, native)
    Mul(yH, xH, tmpH_, len);                           // y = x*R (fp16)
  }

  uint32_t blockIdx_ = 0;
  uint32_t coreNum_ = 1;
  uint64_t totalElems_ = 0;
  uint32_t chunkElems_ = 0;
  uint64_t chunkStart_ = 0;
  uint64_t chunkEnd_ = 0;

  GlobalTensor<half> xGm_;
  GlobalTensor<half> yGm_;

  TBuf<TPosition::VECCALC> xBuf_[2];
  TBuf<TPosition::VECCALC> yBuf_[2];
  TBuf<TPosition::VECCALC> tmpHBuf_;
  TBuf<TPosition::VECCALC> tmpF32Buf_;
  TBuf<TPosition::VECCALC> constBBuf_;

  LocalTensor<half> xH_[2];
  LocalTensor<half> yH_[2];
  LocalTensor<half> tmpH_;
  LocalTensor<float> tmpF32_;
  LocalTensor<half> constBB_;
};

}  // namespace FusedGeluTanhNZ

#endif  // FUSED_GELU_TANH_NZ_H
