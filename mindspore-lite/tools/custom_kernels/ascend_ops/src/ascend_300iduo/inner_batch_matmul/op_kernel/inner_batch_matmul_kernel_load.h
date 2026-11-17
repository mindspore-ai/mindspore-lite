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
// GM -> L1 / L1 -> L0 loaders of the InnerBatchMatMul kernel.
// Included from inner_batch_matmul_kernel.h after the class declaration (not standalone).
#ifndef INNER_BATCH_MATMUL_KERNEL_LOAD_H_
#define INNER_BATCH_MATMUL_KERNEL_LOAD_H_

namespace InnerBatchMatMul {

// K chunk ci of A (kL1_ deep) -> the ping-pong L1 buffer.
// The L1 layout is the compact [k-frac][m1] of the current tile, so the L1->L0 k stride must
// use the real m1 (a tail tile has m1 < tileM_/16).
__aicore__ inline void InnerBatchMatMulKernel::LoadChunkA(uint32_t ci, uint64_t offA, uint32_t mLo, uint32_t m1) {
  const uint32_t k0 = ci * kL1_;
  const uint32_t kLenL1 = (p_->k - k0) < kL1_ ? (p_->k - k0) : kL1_;
  const uint32_t kf = (kLenL1 + kFractalDim - 1) / kFractalDim;
  LocalTensor<half> aL1 = L1A(ci);
  // The input is already NZ: [K1][M1][16][16] with [m][k] inside, so a strided copy is enough
  // and no ND->NZ conversion is needed.
  const uint32_t m1All = p_->m / kFractalDim;
  DataCopyParams pa;
  pa.blockCount = static_cast<uint16_t>(kf);
  pa.blockLen = static_cast<uint16_t>(m1 * kFractalDim);             // m1 fractals (32B units)
  pa.srcStride = static_cast<uint16_t>((m1All - m1) * kFractalDim);  // skip the other m-fractals
  pa.dstStride = 0;
  // Use the absolute k offset (ci*kL1_/16 k-fractals) instead of ci*kf: for a tail chunk kf
  // shrinks and ci*kf would point too low. kL1_ is always a multiple of 16, so ci*kL1_/16 ==
  // ci*(kL1_/16) and full chunks are byte-identical.
  const uint32_t k0Frac = ci * (kL1_ / kFractalDim);
  const uint64_t baseA =
    offA + (static_cast<uint64_t>(k0Frac) * m1All + static_cast<uint64_t>(mLo) / kFractalDim) * kFractalElems;
  DataCopy(aL1, aGm_[baseA], pa);
}

// K chunk ci of B -> L1 (non-resident path).
__aicore__ inline void InnerBatchMatMulKernel::LoadChunkB(uint32_t ci, uint64_t offB, uint32_t nLo, uint32_t nCols) {
  const uint32_t k0 = ci * kL1_;
  const uint32_t kLenL1 = (p_->k - k0) < kL1_ ? (p_->k - k0) : kL1_;
  LocalTensor<half> bL1 = L1B(ci);
  // B in NZ: take the kf fractals of the ci-th k segment from every n-fractal.
  const uint32_t kf = (kLenL1 + 15) / 16;
  DataCopyParams pb;
  pb.blockCount = static_cast<uint16_t>(nCols / kFractalDim);
  pb.blockLen = static_cast<uint16_t>(kf * kFractalDim);
  pb.srcStride = static_cast<uint16_t>((p_->k / kFractalDim - kf) * kFractalDim);
  pb.dstStride = 0;
  const uint32_t k0Frac = ci * (kL1_ / kFractalDim);
  const uint64_t baseB =
    offB + (static_cast<uint64_t>(nLo) / 16 * (p_->k / 16) + static_cast<uint64_t>(k0Frac)) * kFractalElems;
  DataCopy(bL1, bGm_[baseB], pb);
}

// Whole-K block of B -> resident L1.
__aicore__ inline void InnerBatchMatMulKernel::LoadBFull(uint64_t offB, uint32_t nLo, uint32_t nCols) {
  LocalTensor<half> bL1 = b1Buf_.Get<half>();
  // Resident B: a whole column is contiguous in NZ, so one large copy is enough.
  DataCopyParams pf;
  pf.blockCount = static_cast<uint16_t>(nCols / kFractalDim);
  pf.blockLen = static_cast<uint16_t>((p_->k / kFractalDim) * kFractalDim);
  pf.srcStride = 0;
  pf.dstStride = 0;
  DataCopy(bL1, bGm_[offB + static_cast<uint64_t>(nLo) / kFractalDim * (p_->k / kFractalDim) * kFractalElems], pf);
}

// L1 -> L0A[m][k] / L0B[k][n].
__aicore__ inline void InnerBatchMatMulKernel::LoadL0(uint32_t ci, uint32_t buf, uint32_t m1, uint32_t n1,
                                                      uint32_t kf) {
  LocalTensor<half> aL1 = L1A(ci);
  LocalTensor<half> a2 = L0A(buf);
  LocalTensor<half> b2 = L0B(buf);

  // The descriptors only depend on m1/n1/kf, so they are hoisted out of the loop, which then
  // only changes addresses.
  LoadData2DParams lda;  // A: L1[k][m] -> L0A[m][k] (dst stride kf)
  lda.startIndex = 0;
  lda.repeatTimes = static_cast<uint8_t>(m1);
  lda.srcStride = 1;
  lda.dstGap = static_cast<uint16_t>(kf - 1);
  lda.ifTranspose = false;
  for (uint32_t kk = 0; kk < kf; ++kk) {
    LoadData(a2[kk * kFractalElems], aL1[kk * m1 * kFractalElems], lda);
  }
  if (bResident_) {
    LocalTensor<half> bL1 = b1Buf_.Get<half>();
    LoadData2DParams ldb;  // resident B: n-fractals are kfAll_ apart
    ldb.startIndex = 0;
    ldb.repeatTimes = static_cast<uint8_t>(n1);
    ldb.srcStride = static_cast<uint16_t>(kfAll_);
    ldb.dstGap = 0;
    ldb.ifTranspose = true;
    // The resident L1 layout is [n-frac][k-frac] over the whole K, so chunk ci starts at the
    // absolute k-fractal ci*(kL1_/16); a tail chunk (smaller kf) must not shrink that base offset.
    const uint32_t kOffFrac = ci * (kL1_ / kFractalDim);
    for (uint32_t kk = 0; kk < kf; ++kk) {
      LoadData(b2[kk * n1 * kFractalElems], bL1[(kOffFrac + kk) * kFractalElems], ldb);
    }
  } else {
    LocalTensor<half> bL1 = L1B(ci);
    LoadData2DParams ldb;
    ldb.startIndex = 0;
    ldb.repeatTimes = static_cast<uint8_t>(n1);
    ldb.srcStride = static_cast<uint16_t>(kf);
    ldb.dstGap = 0;
    ldb.ifTranspose = true;
    for (uint32_t kk = 0; kk < kf; ++kk) {
      LoadData(b2[kk * n1 * kFractalElems], bL1[kk * kFractalElems], ldb);
    }
  }
}

// Small-M (M <= 64) path: L1 -> single-buffered L0A/L0B.
// The key difference is the A-side descriptor: with m1 == 1 both L1[k][m] and L0A[m][k]
// degenerate to "k contiguous", so one LoadData (repeatTimes=kf, srcStride=1, dstGap=0) moves
// the whole chunk of A and cuts the per-chunk A instruction count from kf to 1 (8 -> 1 at
// M=16/kL1=128, 16 -> 1 at kL1=256). B is the same as the generic path: L1[n][k] -> L0B[k][n],
// one LoadData per kk with repeatTimes=n1.
__aicore__ inline void InnerBatchMatMulKernel::LoadL0Small(uint32_t ci, uint32_t m1, uint32_t n1, uint32_t kf) {
  LocalTensor<half> aL1 = L1A(ci);
  LocalTensor<half> bL1 = L1B(ci);
  LocalTensor<half> a2 = L0A(0);
  LocalTensor<half> b2 = L0B(0);

  LoadData2DParams lda;
  lda.startIndex = 0;
  if (m1 == 1U) {
    lda.repeatTimes = static_cast<uint8_t>(kf);
    lda.srcStride = 1;
    lda.dstGap = 0;
    lda.ifTranspose = false;
    LoadData(a2, aL1, lda);
  } else {
    lda.repeatTimes = static_cast<uint8_t>(m1);
    lda.srcStride = 1;
    lda.dstGap = static_cast<uint16_t>(kf - 1);
    lda.ifTranspose = false;
    for (uint32_t kk = 0; kk < kf; ++kk) {
      LoadData(a2[kk * kFractalElems], aL1[kk * m1 * kFractalElems], lda);
    }
  }

  LoadData2DParams ldb;
  ldb.startIndex = 0;
  ldb.repeatTimes = static_cast<uint8_t>(n1);
  ldb.srcStride = static_cast<uint16_t>(kf);
  ldb.dstGap = 0;
  ldb.ifTranspose = true;
  for (uint32_t kk = 0; kk < kf; ++kk) {
    LoadData(b2[kk * n1 * kFractalElems], bL1[kk * kFractalElems], ldb);
  }
}

// 2x2 (tail-chunk fallback): L1 -> L0A[m][k] (tileM rows) / L0B[k][n] (full width).
// The descriptors are built outside the K loop (kf/m1/n1 are constant inside a tile), so the
// loop body only changes addresses.
__aicore__ inline void InnerBatchMatMulKernel::LoadL02(uint32_t ci, uint32_t buf, uint32_t m1, uint32_t n1,
                                                       uint32_t kf) {
  LocalTensor<half> aL1 = L1A(ci);
  LocalTensor<half> a2 = L0A(buf);
  LocalTensor<half> b2 = L0B(buf);
  LocalTensor<half> bL1 = bResident_ ? b1Buf_.Get<half>() : L1B(ci);
  // A in L1 is the compact [k-frac][m1] of this tile: the k stride is the real m1 (a tail tile
  // has m1 < tileM_/16).
  const uint32_t m1All = m1;
  // Absolute k-fractal start of the resident B (again: ci*kf would be too small for a tail chunk).
  const uint32_t kOffBase = bResident_ ? (ci * (kL1_ / kFractalDim)) : 0U;

  LoadData2DParams lda;
  lda.startIndex = 0;
  lda.repeatTimes = static_cast<uint8_t>(m1);
  lda.srcStride = 1;
  lda.dstGap = static_cast<uint16_t>(kf - 1);
  lda.ifTranspose = false;
  LoadData2DParams ldb;
  ldb.startIndex = 0;
  ldb.repeatTimes = static_cast<uint8_t>(n1);
  ldb.srcStride = static_cast<uint16_t>(bResident_ ? kfAll_ : kf);
  ldb.dstGap = 0;
  ldb.ifTranspose = true;

  for (uint32_t kk = 0; kk < kf; ++kk) {
    LoadData(a2[kk * kFractalElems], aL1[kk * m1All * kFractalElems], lda);
  }
  for (uint32_t kk = 0; kk < kf; ++kk) {
    LoadData(b2[kk * n1 * kFractalElems], bL1[(kOffBase + kk) * kFractalElems], ldb);
  }
}

}  // namespace InnerBatchMatMul

#endif  // INNER_BATCH_MATMUL_KERNEL_LOAD_H_
