/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <vector>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "include/api/model.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/matmul_parameter.h"

namespace mindspore::lite::dsp::test {

class TestDSP_MatMulFusion : public DSPCommonTest {};

static void FillFloat(float *data, int size, float base = 0.1f) {
  for (int i = 0; i < size; ++i) {
    data[i] = base * static_cast<float>((i % 10));
  }
}

// Large size tests (M=N=K=256) across dtypes
TEST_F(TestDSP_MatMulFusion, MatMulFusion_Fp32_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeFloat32, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeFloat32, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeFloat32, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeFloat32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  FillFloat(reinterpret_cast<float *>(t_A->MutableData()), M * K, 0.02f);
  FillFloat(reinterpret_cast<float *>(t_B->MutableData()), K * N, 0.03f);
  FillFloat(reinterpret_cast<float *>(t_bias->MutableData()), M * N, 0.005f);
  std::memset(t_out->MutableData(), 0, M * N * sizeof(float));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  auto A = reinterpret_cast<float *>(t_A->MutableData());
  auto B = reinterpret_cast<float *>(t_B->MutableData());
  auto bias = reinterpret_cast<float *>(t_bias->MutableData());
  auto C = reinterpret_cast<float *>(t_out->MutableData());
  std::vector<float> expect(M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      sum += bias[m * N + n];
      expect[m * N + n] = sum > 0.f ? sum : 0.f;
    }
  }
  ASSERT_EQ(0, CompareOutputData(C, expect.data(), M * N, 1e-3));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}

TEST_F(TestDSP_MatMulFusion, MatMulFusion_Int32_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeInt32, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeInt32, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeInt32, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeInt32, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<int32_t *>(t_A->MutableData());
  auto B = reinterpret_cast<int32_t *>(t_B->MutableData());
  auto bias = reinterpret_cast<int32_t *>(t_bias->MutableData());
  auto C = reinterpret_cast<int32_t *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A[i] = (i % 11) - 5;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = (i % 13) - 6;
  }
  for (int i = 0; i < M * N; ++i) {
    bias[i] = (i % 9) - 4;
  }
  std::memset(C, 0, M * N * sizeof(int32_t));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<int32_t> expect(M * N, 0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int64_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int64_t>(A[m * K + k]) * B[k * N + n];
      }
      sum += static_cast<int64_t>(bias[m * N + n]);
      expect[m * N + n] = static_cast<int32_t>(sum > 0 ? sum : 0);
    }
  }
  ASSERT_EQ(0, CompareOutputData(C, expect.data(), M * N, 0.f));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}

TEST_F(TestDSP_MatMulFusion, MatMulFusion_Int16_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeInt16, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeInt16, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeInt16, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeInt16, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<int16_t *>(t_A->MutableData());
  auto B = reinterpret_cast<int16_t *>(t_B->MutableData());
  auto bias = reinterpret_cast<int16_t *>(t_bias->MutableData());
  auto C = reinterpret_cast<int16_t *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A[i] = static_cast<int16_t>((i % 21) - 10);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = static_cast<int16_t>((i % 19) - 9);
  }
  for (int i = 0; i < M * N; ++i) {
    bias[i] = static_cast<int16_t>(i % 15);
  }
  std::memset(C, 0, M * N * sizeof(int16_t));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt16, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<int16_t> expect(M * N, 0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int64_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int64_t>(A[m * K + k]) * B[k * N + n];
      }
      sum += static_cast<int64_t>(bias[m * N + n]);
      sum = sum > 0 ? sum : 0;
      if (sum > std::numeric_limits<int16_t>::max()) sum = std::numeric_limits<int16_t>::max();
      expect[m * N + n] = static_cast<int16_t>(sum);
    }
  }
  ASSERT_EQ(0, CompareOutputData(C, expect.data(), M * N, 0.f));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}

TEST_F(TestDSP_MatMulFusion, MatMulFusion_Complex64_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeComplex64, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeComplex64, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeComplex64, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeComplex64, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<float *>(t_A->MutableData());
  auto B = reinterpret_cast<float *>(t_B->MutableData());
  auto bias = reinterpret_cast<float *>(t_bias->MutableData());
  auto C = reinterpret_cast<float *>(t_out->MutableData());  // complex64 stored as interleaved real,imag
  for (int i = 0; i < M * K; ++i) {
    A[2 * i] = 0.01f * (i % 17);
    A[2 * i + 1] = 0.02f * (i % 19);
  }
  for (int i = 0; i < K * N; ++i) {
    B[2 * i] = 0.03f * (i % 23);
    B[2 * i + 1] = 0.01f * (i % 29);
  }
  for (int i = 0; i < M * N; ++i) {
    bias[2 * i] = 0.002f * (i % 31);
    bias[2 * i + 1] = 0.001f * (i % 37);
  }
  std::memset(C, 0, M * N * 2 * sizeof(float));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex64, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<float> expect(2 * M * N, 0.f);
  std::vector<float> actual(2 * M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float real = 0.f;
      float imag = 0.f;
      for (int k = 0; k < K; ++k) {
        float ar = A[2 * (m * K + k)];
        float ai = A[2 * (m * K + k) + 1];
        float br = B[2 * (k * N + n)];
        float bi = B[2 * (k * N + n) + 1];
        real += ar * br - ai * bi;
        imag += ar * bi + ai * br;
      }
      real += bias[2 * (m * N + n)];
      imag += bias[2 * (m * N + n) + 1];
      if (real < 0.f) real = 0.f;
      expect[2 * (m * N + n)] = real;
      expect[2 * (m * N + n) + 1] = imag;
      actual[2 * (m * N + n)] = C[2 * (m * N + n)];
      actual[2 * (m * N + n) + 1] = C[2 * (m * N + n) + 1];
    }
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), expect.data(), 2 * M * N, 5e-2));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
}

#ifdef SUPPORT_FT04
TEST_F(TestDSP_MatMulFusion, MatMulFusion_Fp16_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeFloat16, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeFloat16, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeFloat16, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeFloat16, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A16 = reinterpret_cast<uint16_t *>(t_A->MutableData());
  auto B16 = reinterpret_cast<uint16_t *>(t_B->MutableData());
  auto bias16 = reinterpret_cast<uint16_t *>(t_bias->MutableData());
  auto C16 = reinterpret_cast<uint16_t *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A16[i] = Fp32ToFp16(0.01f * static_cast<float>(i % 13));
  }
  for (int i = 0; i < K * N; ++i) {
    B16[i] = Fp32ToFp16(0.02f * static_cast<float>(i % 17));
  }
  for (int i = 0; i < M * N; ++i) {
    bias16[i] = Fp32ToFp16(0.003f * static_cast<float>(i % 11));
  }
  std::memset(C16, 0, M * N * sizeof(uint16_t));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat16, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<float> expect_fp32(M * N, 0.f);
  std::vector<float> actual_fp32(M * N, 0.f);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.f;
      for (int k = 0; k < K; ++k) {
        float a = Fp16ToFp32(A16[m * K + k]);
        float b = Fp16ToFp32(B16[k * N + n]);
        sum += a * b;
      }
      sum += Fp16ToFp32(bias16[m * N + n]);
      expect_fp32[m * N + n] = sum > 0.f ? sum : 0.f;
      actual_fp32[m * N + n] = Fp16ToFp32(C16[m * N + n]);
    }
  }
  ASSERT_EQ(0, CompareOutputData(actual_fp32.data(), expect_fp32.data(), M * N, 5e-2));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}
#endif

#ifdef SUPPORT_FT78
TEST_F(TestDSP_MatMulFusion, MatMulFusion_Fp64_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeFloat64, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeFloat64, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeFloat64, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeFloat64, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<double *>(t_A->MutableData());
  auto B = reinterpret_cast<double *>(t_B->MutableData());
  auto bias = reinterpret_cast<double *>(t_bias->MutableData());
  auto C = reinterpret_cast<double *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A[i] = 0.015 * static_cast<double>(i % 13);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = 0.018 * static_cast<double>(i % 17);
  }
  for (int i = 0; i < M * N; ++i) {
    bias[i] = 0.004 * static_cast<double>(i % 19);
  }
  std::fill_n(C, M * N, 0.0);
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat64, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<double> expect(M * N, 0.0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      sum += bias[m * N + n];
      expect[m * N + n] = sum > 0.0 ? sum : 0.0;
    }
  }
  ASSERT_EQ(0, CompareOutputData(C, expect.data(), M * N, 1e-6));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}

TEST_F(TestDSP_MatMulFusion, MatMulFusion_Int8_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 32, K = 32, N = 32;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeInt8, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeInt8, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeInt8, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeInt8, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<int8_t *>(t_A->MutableData());
  auto B = reinterpret_cast<int8_t *>(t_B->MutableData());
  auto bias = reinterpret_cast<int8_t *>(t_bias->MutableData());
  auto C = reinterpret_cast<int8_t *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A[i] = static_cast<int8_t>((i % 7) - 3);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = static_cast<int8_t>((i % 9) - 4);
  }
  for (int i = 0; i < M * N; ++i) {
    bias[i] = static_cast<int8_t>(i % 5 - 2);
  }
  std::fill_n(C, M * N, static_cast<int8_t>(0));
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeInt8, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<int8_t> expect(M * N, static_cast<int8_t>(0));
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[k * N + n]);
      }
      sum += static_cast<int32_t>(bias[m * N + n]);
      sum = sum < 0 ? 0 : sum;
      if (sum > std::numeric_limits<int8_t>::max()) {
        sum = std::numeric_limits<int8_t>::max();
      }
      expect[m * N + n] = static_cast<int8_t>(sum);
    }
  }
  ASSERT_EQ(0, CompareOutputData(C, expect.data(), M * N, 0.0f));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}

TEST_F(TestDSP_MatMulFusion, MatMulFusion_Complex128_Large_BiasRelu) {
  InitDSPRuntime();
  const int M = 256, K = 256, N = 256;
  std::vector<int> a_shape = {M, K};
  std::vector<int> b_shape = {K, N};
  std::vector<int> out_shape = {M, N};
  std::vector<int> bias_shape = {M, N};
  auto t_A = new lite::Tensor(kNumberTypeComplex128, a_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_B = new lite::Tensor(kNumberTypeComplex128, b_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_bias = new lite::Tensor(kNumberTypeComplex128, bias_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeComplex128, out_shape, NHWC, lite::Category::CONST_TENSOR);
  t_A->MallocData(allocator_);
  t_B->MallocData(allocator_);
  t_bias->MallocData(allocator_);
  t_out->MallocData(allocator_);
  auto A = reinterpret_cast<double *>(t_A->MutableData());
  auto B = reinterpret_cast<double *>(t_B->MutableData());
  auto bias = reinterpret_cast<double *>(t_bias->MutableData());
  auto C = reinterpret_cast<double *>(t_out->MutableData());
  for (int i = 0; i < M * K; ++i) {
    A[2 * i] = 0.01f * (i % 17);
    A[2 * i + 1] = 0.02f * (i % 19);
  }
  for (int i = 0; i < K * N; ++i) {
    B[2 * i] = 0.03f * (i % 23);
    B[2 * i + 1] = 0.01f * (i % 29);
  }
  for (int i = 0; i < M * N; ++i) {
    bias[2 * i] = 0.002f * (i % 31);
    bias[2 * i + 1] = 0.001f * (i % 37);
  }
  std::memset(C, 0, M * N * 2 * sizeof(double));
  std::vector<lite::Tensor *> inputs_{t_A, t_B, t_bias};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new MatMulParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_MatMulFusion);
  param->act_type_ = ActType_Relu;
  param->has_bias_ = true;
  param->row_ = M;
  param->col_ = N;
  param->deep_ = K;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeComplex128, NHWC, schema::PrimitiveType_MatMulFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);
  std::vector<double> expect(2 * M * N, 0.0);
  std::vector<double> actual(2 * M * N, 0.0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double real = 0.0;
      double imag = 0.0;
      for (int k = 0; k < K; ++k) {
        double ar = A[2 * (m * K + k)];
        double ai = A[2 * (m * K + k) + 1];
        double br = B[2 * (k * N + n)];
        double bi = B[2 * (k * N + n) + 1];
        real += ar * br - ai * bi;
        imag += ar * bi + ai * br;
      }
      real += bias[2 * (m * N + n)];
      imag += bias[2 * (m * N + n) + 1];
      if (real < 0.0) {
        real = 0.0;
      }
      expect[2 * (m * N + n)] = real;
      expect[2 * (m * N + n) + 1] = imag;
      actual[2 * (m * N + n)] = C[2 * (m * N + n)];
      actual[2 * (m * N + n) + 1] = C[2 * (m * N + n) + 1];
    }
  }
  ASSERT_EQ(0, CompareOutputData(actual.data(), expect.data(), 2 * M * N, 1e-3));
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_A;
  delete t_B;
  delete t_bias;
  delete t_out;
}
#endif

}  // namespace mindspore::lite::dsp::test
