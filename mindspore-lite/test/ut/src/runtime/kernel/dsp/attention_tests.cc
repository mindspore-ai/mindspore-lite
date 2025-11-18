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
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "include/api/context.h"
#include "include/api/data_type.h"
#include "schema/inner/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/nnacl_c/attention_parameter.h"

namespace mindspore::lite::dsp::test {

#ifdef SUPPORT_FT78
class TestDSP_Attention : public DSPCommonTest {};

constexpr int kAttentionArgsCount = 8;

static void SoftmaxNormalize(std::vector<float> *row) {
  float max_val = *std::max_element(row->begin(), row->end());
  float sum = 0.f;
  for (auto &v : *row) {
    v = std::exp(v - max_val);
    sum += v;
  }
  if (sum == 0.f) {
    return;
  }
  for (auto &v : *row) {
    v /= sum;
  }
}

TEST_F(TestDSP_Attention, Attention_Fp32_FT78) {
  InitDSPRuntime();
  const int batch = 1;
  const int seq_len = 2;
  const int head_num = 32;
  const int head_dim = 32;
  std::vector<int> tensor_shape = {batch, seq_len, head_num, head_dim};
  auto t_Q = new lite::Tensor(kNumberTypeFloat32, tensor_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_K = new lite::Tensor(kNumberTypeFloat32, tensor_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_V = new lite::Tensor(kNumberTypeFloat32, tensor_shape, NHWC, lite::Category::CONST_TENSOR);
  auto t_out = new lite::Tensor(kNumberTypeFloat32, tensor_shape, NHWC, lite::Category::CONST_TENSOR);
  std::vector<int> args_shape = {kAttentionArgsCount};
  auto t_args = new lite::Tensor(kNumberTypeInt32, args_shape, NHWC, lite::Category::CONST_TENSOR);
  t_Q->MallocData(allocator_);
  t_K->MallocData(allocator_);
  t_V->MallocData(allocator_);
  t_out->MallocData(allocator_);
  t_args->MallocData(allocator_);
  auto Q = reinterpret_cast<float *>(t_Q->MutableData());
  auto K = reinterpret_cast<float *>(t_K->MutableData());
  auto V = reinterpret_cast<float *>(t_V->MutableData());
  auto out = reinterpret_cast<float *>(t_out->MutableData());
  auto args = reinterpret_cast<int32_t *>(t_args->MutableData());
  auto tensor_offset = [&](int b, int h, int l, int d) { return ((b * head_num + h) * seq_len + l) * head_dim + d; };

  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < head_num; ++h) {
      for (int l = 0; l < seq_len; ++l) {
        for (int d = 0; d < head_dim; ++d) {
          int index = tensor_offset(b, h, l, d);
          Q[index] = static_cast<float>((index % 7) - 3) * 0.1f;
          K[index] = static_cast<float>(((index + 1) % 5) - 2) * 0.2f;
          V[index] = static_cast<float>(((index + 2) % 11) - 5) * 0.15f;
        }
      }
    }
  }
  std::memset(out, 0, batch * seq_len * head_num * head_dim * sizeof(float));
  const int64_t per_head_logits = static_cast<int64_t>(seq_len) * seq_len;
  const int64_t scratch_elements = per_head_logits * head_num * batch;
  ASSERT_GT(scratch_elements, 0);
  const size_t scratch_bytes = static_cast<size_t>(scratch_elements) * sizeof(float);
  void *qk_buffer = allocator_->Malloc(scratch_bytes);
  ASSERT_NE(qk_buffer, nullptr);
  std::memset(qk_buffer, 0, scratch_bytes);
  void *softmax_buffer = allocator_->Malloc(scratch_bytes);
  ASSERT_NE(softmax_buffer, nullptr);
  std::memset(softmax_buffer, 0, scratch_bytes);
  auto pack_low32 = [](uint64_t addr) -> int32_t {
    return static_cast<int32_t>(static_cast<uint32_t>(addr & 0xFFFFFFFFULL));
  };
  auto pack_high32 = [](uint64_t addr) -> int32_t {
    return static_cast<int32_t>(static_cast<uint32_t>((addr >> 32) & 0xFFFFFFFFULL));
  };
  uint64_t qk_device_ptr = allocator_->GetDeviceMemPtr(qk_buffer);
  uint64_t softmax_device_ptr = allocator_->GetDeviceMemPtr(softmax_buffer);
  ASSERT_NE(qk_device_ptr, 0ULL);
  ASSERT_NE(softmax_device_ptr, 0ULL);
  args[0] = batch;
  args[1] = seq_len;
  args[2] = head_num;
  args[3] = head_dim;
  args[4] = pack_low32(qk_device_ptr);
  args[5] = pack_high32(qk_device_ptr);
  args[6] = pack_low32(softmax_device_ptr);
  args[7] = pack_high32(softmax_device_ptr);
  std::vector<lite::Tensor *> inputs_{t_Q, t_K, t_V, t_args};
  std::vector<lite::Tensor *> outputs_{t_out};
  auto ctx = new lite::InnerContext;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  auto *param = new AttentionParameter();
  param->op_parameter_.type_ = static_cast<int>(schema::PrimitiveType_Attention);
  param->head_num_ = head_num;
  param->head_size_ = head_dim;
  param->cross_ = false;
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Attention};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs_, outputs_, reinterpret_cast<OpParameter *>(param), ctx, key);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), lite::RET_OK);
  ASSERT_EQ(kernel->Run(), lite::RET_OK);

  std::vector<float> expect(batch * seq_len * head_num * head_dim, 0.f);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  for (int b = 0; b < batch; ++b) {
    for (int h = 0; h < head_num; ++h) {
      const int block_base = ((b * head_num + h) * seq_len) * head_dim;
      for (int q = 0; q < seq_len; ++q) {
        std::vector<float> logits(seq_len, 0.f);
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
          float sum = 0.f;
          for (int d = 0; d < head_dim; ++d) {
            int q_index = block_base + q * head_dim + d;
            int k_index = block_base + d * seq_len + k_idx;
            sum += Q[q_index] * K[k_index];
          }
          logits[k_idx] = sum * scale;
        }
        SoftmaxNormalize(&logits);
        for (int d = 0; d < head_dim; ++d) {
          float value = 0.f;
          for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            int v_index = block_base + k_idx * head_dim + d;
            value += logits[k_idx] * V[v_index];
          }
          int out_index = block_base + q * head_dim + d;
          expect[out_index] = value;
        }
      }
    }
  }

  ASSERT_EQ(0, CompareOutputData(out, expect.data(), expect.size(), 1e-3f));
  allocator_->Free(softmax_buffer);
  allocator_->Free(qk_buffer);
  UninitDSPRuntime();
  delete ctx;
  delete kernel;
  delete t_Q;
  delete t_K;
  delete t_V;
  delete t_out;
  delete t_args;
}
#endif

}  // namespace mindspore::lite::dsp::test
