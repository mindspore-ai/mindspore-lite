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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <vector>
#include "ut/src/runtime/kernel/dsp/dsp_test.h"
#include "src/litert/kernel/cpu/nnacl_c/matmul_parameter.h"
#include "src/litert/lite_session.h"

namespace mindspore::lite::dsp::test {
namespace {
kernel::KernelExec *CreateDspNode(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  schema::PrimitiveType primitive_type, TypeId data_type, OpParameter *parameter,
                                  const lite::InnerContext *context) {
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, data_type, NHWC, primitive_type};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  if (creator == nullptr) {
    free(parameter);
    return nullptr;
  }
  auto *lite_kernel = creator(inputs, outputs, parameter, context, key);
  if (lite_kernel == nullptr) {
    return nullptr;
  }
  std::shared_ptr<kernel::Kernel> shared_kernel(lite_kernel);
  auto *node = new (std::nothrow) kernel::KernelExec(shared_kernel);
  if (node == nullptr) {
    return nullptr;
  }
  node->set_desc(key);
  node->set_name(schema::EnumNamePrimitiveType(primitive_type));
  return node;
}

kernel::KernelExec *CreatePrepareOnlyNode(const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs) {
  auto *parameter = static_cast<OpParameter *>(calloc(1, sizeof(OpParameter)));
  if (parameter == nullptr) {
    return nullptr;
  }
  parameter->type_ = schema::PrimitiveType_AddFusion;
  auto lite_kernel = std::make_shared<kernel::LiteKernel>(parameter, inputs, outputs, nullptr);
  auto *node = new (std::nothrow) kernel::KernelExec(lite_kernel);
  if (node == nullptr) {
    return nullptr;
  }
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kDSP, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddFusion};
  node->set_desc(key);
  node->set_name("PrepareOnlyDSPNode");
  return node;
}

std::unique_ptr<kernel::DspSubGraph> CreateSubGraph(const std::vector<kernel::KernelExec *> &nodes,
                                                    const std::vector<lite::Tensor *> &inputs,
                                                    const std::vector<lite::Tensor *> &outputs,
                                                    const lite::InnerContext *context) {
  auto *subgraph_kernel = new (std::nothrow) kernel::LiteKernel(nullptr, inputs, outputs, context);
  if (subgraph_kernel == nullptr) {
    return nullptr;
  }
  auto subgraph = std::make_unique<kernel::DspSubGraph>(nodes, nodes, nodes, subgraph_kernel);
  subgraph->set_context(context);
  return subgraph;
}
}  // namespace

class TestDSP_SubGraphConstInput : public DSPCommonTest {
 public:
  void SetUp() override {
    InitDSPRuntime();
    ASSERT_NE(allocator_, nullptr);
  }

  void TearDown() override {
    allocator_.reset();
    UninitDSPRuntime();
  }
};

class DspConstInputTestSession : public lite::LiteSession {
 public:
  static void RetainDspConstInputs(const std::vector<kernel::KernelExec *> &kernels) { MarkSharedWeight(kernels); }
  static void ReleasePackedOpWeights(const std::vector<kernel::KernelExec *> &kernels) { FreePackOpWeight(kernels); }
};

TEST_F(TestDSP_SubGraphConstInput, AddHostConstAndContinuousExecute) {
  const std::vector<int> shape = {2, 2};
  lite::Tensor input(kNumberTypeFloat32, shape, NHWC, lite::Category::GRAPH_INPUT);
  lite::Tensor constant(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor output(kNumberTypeFloat32, shape, NHWC, lite::Category::GRAPH_OUTPUT);
  ASSERT_EQ(input.MallocData(allocator_), RET_OK);
  ASSERT_EQ(constant.MallocData(), RET_OK);
  std::fill_n(static_cast<float *>(input.MutableData()), input.ElementsNum(), 2.0f);
  std::fill_n(static_cast<float *>(constant.MutableData()), constant.ElementsNum(), 1.0f);
  void *old_const_data = constant.data();

  lite::InnerContext context;
  ASSERT_EQ(context.Init(), RET_OK);
  auto *parameter = static_cast<ArithmeticParameter *>(calloc(1, sizeof(ArithmeticParameter)));
  ASSERT_NE(parameter, nullptr);
  parameter->op_parameter_.type_ = schema::PrimitiveType_AddFusion;
  std::unique_ptr<kernel::KernelExec> node(CreateDspNode({&input, &constant}, {&output},
                                                         schema::PrimitiveType_AddFusion, kNumberTypeFloat32,
                                                         reinterpret_cast<OpParameter *>(parameter), &context));
  ASSERT_NE(node, nullptr);
  auto subgraph = CreateSubGraph({node.get()}, {&input}, {&output}, &context);
  ASSERT_NE(subgraph, nullptr);
  (void)node.release();

  ASSERT_EQ(subgraph->Prepare(), RET_OK);
  ASSERT_NE(constant.data(), old_const_data);
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(constant.data()));
  ASSERT_EQ(allocator_->RefCount(constant.data()), 1);
  void *uploaded_data = constant.data();
  ASSERT_EQ(subgraph->Execute(), RET_OK);
  std::array<float, 4> expected = {3.0f, 3.0f, 3.0f, 3.0f};
  ASSERT_EQ(CompareOutputData(static_cast<float *>(output.data()), expected.data(), expected.size()), RET_OK);

  ASSERT_EQ(subgraph->Execute(), RET_OK);
  ASSERT_EQ(constant.data(), uploaded_data);
  ASSERT_EQ(allocator_->RefCount(constant.data()), 1);
  ASSERT_EQ(CompareOutputData(static_cast<float *>(output.data()), expected.data(), expected.size()), RET_OK);
}

TEST_F(TestDSP_SubGraphConstInput, MulHostConst) {
  const std::vector<int> shape = {2, 2};
  lite::Tensor input(kNumberTypeFloat32, shape, NHWC, lite::Category::GRAPH_INPUT);
  lite::Tensor constant(kNumberTypeFloat32, shape, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor output(kNumberTypeFloat32, shape, NHWC, lite::Category::GRAPH_OUTPUT);
  ASSERT_EQ(input.MallocData(allocator_), RET_OK);
  std::array<float, 4> const_data = {3.0f, 3.0f, 3.0f, 3.0f};
  constant.set_data(const_data.data(), false);
  std::fill_n(static_cast<float *>(input.MutableData()), input.ElementsNum(), 2.0f);

  lite::InnerContext context;
  ASSERT_EQ(context.Init(), RET_OK);
  auto *parameter = static_cast<ArithmeticParameter *>(calloc(1, sizeof(ArithmeticParameter)));
  ASSERT_NE(parameter, nullptr);
  parameter->op_parameter_.type_ = schema::PrimitiveType_MulFusion;
  std::unique_ptr<kernel::KernelExec> node(CreateDspNode({&input, &constant}, {&output},
                                                         schema::PrimitiveType_MulFusion, kNumberTypeFloat32,
                                                         reinterpret_cast<OpParameter *>(parameter), &context));
  ASSERT_NE(node, nullptr);
  auto subgraph = CreateSubGraph({node.get()}, {&input}, {&output}, &context);
  ASSERT_NE(subgraph, nullptr);
  (void)node.release();

  ASSERT_EQ(subgraph->Prepare(), RET_OK);
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(constant.data()));
  ASSERT_EQ(subgraph->Execute(), RET_OK);
  std::array<float, 4> expected = {6.0f, 6.0f, 6.0f, 6.0f};
  ASSERT_EQ(CompareOutputData(static_cast<float *>(output.data()), expected.data(), expected.size()), RET_OK);
  ASSERT_EQ(const_data[0], 3.0f);
}

TEST_F(TestDSP_SubGraphConstInput, MatMulHostWeightAndBias) {
  lite::Tensor input(kNumberTypeFloat32, {2, 3}, NHWC, lite::Category::GRAPH_INPUT);
  lite::Tensor weight(kNumberTypeFloat32, {3, 2}, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor bias(kNumberTypeFloat32, {2, 2}, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor output(kNumberTypeFloat32, {2, 2}, NHWC, lite::Category::GRAPH_OUTPUT);
  ASSERT_EQ(input.MallocData(allocator_), RET_OK);
  std::array<float, 6> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::array<float, 6> weight_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::copy(input_data.begin(), input_data.end(), static_cast<float *>(input.MutableData()));
  weight.set_data(weight_data.data(), false);
  ASSERT_EQ(bias.MallocData(), RET_OK);
  std::fill_n(static_cast<float *>(bias.MutableData()), bias.ElementsNum(), 1.0f);

  lite::InnerContext context;
  ASSERT_EQ(context.Init(), RET_OK);
  auto *parameter = static_cast<MatMulParameter *>(calloc(1, sizeof(MatMulParameter)));
  ASSERT_NE(parameter, nullptr);
  parameter->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  parameter->has_bias_ = true;
  std::unique_ptr<kernel::KernelExec> node(CreateDspNode({&input, &weight, &bias}, {&output},
                                                         schema::PrimitiveType_MatMulFusion, kNumberTypeFloat32,
                                                         reinterpret_cast<OpParameter *>(parameter), &context));
  ASSERT_NE(node, nullptr);
  auto subgraph = CreateSubGraph({node.get()}, {&input}, {&output}, &context);
  ASSERT_NE(subgraph, nullptr);
  (void)node.release();

  ASSERT_EQ(subgraph->Prepare(), RET_OK);
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(weight.data()));
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(bias.data()));
  ASSERT_EQ(allocator_->RefCount(weight.data()), 1);
  ASSERT_EQ(allocator_->RefCount(bias.data()), 1);
  DspConstInputTestSession::RetainDspConstInputs({subgraph.get()});
  DspConstInputTestSession::ReleasePackedOpWeights({subgraph.get()});
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(weight.data()));
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(bias.data()));
  ASSERT_EQ(subgraph->Execute(), RET_OK);
  std::array<float, 4> expected = {23.0f, 29.0f, 50.0f, 65.0f};
  ASSERT_EQ(CompareOutputData(static_cast<float *>(output.data()), expected.data(), expected.size(), 1e-5), RET_OK);
  ASSERT_EQ(weight_data[0], 1.0f);
}

TEST_F(TestDSP_SubGraphConstInput, SharedConstUploadedOnlyOnceAndCpuReadable) {
  lite::Tensor graph_input(kNumberTypeFloat32, {1}, NHWC, lite::Category::GRAPH_INPUT);
  lite::Tensor constant(kNumberTypeFloat32, {4}, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor output0(kNumberTypeFloat32, {4}, NHWC, lite::Category::GRAPH_OUTPUT);
  lite::Tensor output1(kNumberTypeFloat32, {4}, NHWC, lite::Category::GRAPH_OUTPUT);
  std::array<float, 4> const_data = {1.0f, 2.0f, 3.0f, 4.0f};
  constant.set_data(const_data.data(), false);
  std::unique_ptr<kernel::KernelExec> node0(CreatePrepareOnlyNode({&graph_input, &constant}, {&output0}));
  ASSERT_NE(node0, nullptr);
  std::unique_ptr<kernel::KernelExec> node1(CreatePrepareOnlyNode({&graph_input, &constant}, {&output1}));
  ASSERT_NE(node1, nullptr);
  auto subgraph = CreateSubGraph({node0.get(), node1.get()}, {&graph_input}, {&output0, &output1}, nullptr);
  ASSERT_NE(subgraph, nullptr);
  (void)node0.release();
  (void)node1.release();

  ASSERT_EQ(subgraph->Prepare(), RET_OK);
  void *uploaded_data = constant.data();
  ASSERT_TRUE(allocator_->HasDeviceMemPtr(uploaded_data));
  ASSERT_EQ(allocator_->RefCount(uploaded_data), 1);
  ASSERT_EQ(subgraph->Prepare(), RET_OK);
  ASSERT_EQ(constant.data(), uploaded_data);
  ASSERT_EQ(allocator_->RefCount(uploaded_data), 1);
  ASSERT_TRUE(std::equal(const_data.begin(), const_data.end(), static_cast<float *>(constant.data())));

  std::array<float, 4> cpu_input_data = {10.0f, 10.0f, 10.0f, 10.0f};
  std::array<float, 4> cpu_output_data = {};
  lite::Tensor cpu_input(kNumberTypeFloat32, {4}, NHWC, lite::Category::VAR);
  lite::Tensor cpu_output(kNumberTypeFloat32, {4}, NHWC, lite::Category::VAR);
  cpu_input.set_data(cpu_input_data.data(), false);
  cpu_output.set_data(cpu_output_data.data(), false);
  auto *parameter = static_cast<ArithmeticParameter *>(calloc(1, sizeof(ArithmeticParameter)));
  ASSERT_NE(parameter, nullptr);
  parameter->op_parameter_.type_ = schema::PrimitiveType_AddFusion;
  lite::InnerContext context;
  ASSERT_EQ(context.Init(), RET_OK);
  kernel::KernelKey key = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_AddFusion};
  auto creator = KernelRegistry::GetInstance()->GetCreator(key);
  ASSERT_NE(creator, nullptr);
  std::unique_ptr<kernel::LiteKernel> cpu_kernel(
    creator({&cpu_input, &constant}, {&cpu_output}, reinterpret_cast<OpParameter *>(parameter), &context, key));
  ASSERT_NE(cpu_kernel, nullptr);
  ASSERT_EQ(cpu_kernel->Prepare(), RET_OK);
  ASSERT_EQ(cpu_kernel->Run(), RET_OK);
  std::array<float, 4> expected = {11.0f, 12.0f, 13.0f, 14.0f};
  ASSERT_EQ(CompareOutputData(cpu_output_data.data(), expected.data(), expected.size()), RET_OK);
}

TEST_F(TestDSP_SubGraphConstInput, UnpreparedDspCandidateDoesNotModifyHostConst) {
  lite::Tensor graph_input(kNumberTypeFloat32, {1}, NHWC, lite::Category::GRAPH_INPUT);
  lite::Tensor constant(kNumberTypeFloat32, {4}, NHWC, lite::Category::CONST_TENSOR);
  lite::Tensor output(kNumberTypeFloat32, {4}, NHWC, lite::Category::GRAPH_OUTPUT);
  std::array<float, 4> const_data = {1.0f, 2.0f, 3.0f, 4.0f};
  constant.set_data(const_data.data(), false);
  void *host_data = constant.data();
  std::unique_ptr<kernel::KernelExec> node(CreatePrepareOnlyNode({&graph_input, &constant}, {&output}));
  ASSERT_NE(node, nullptr);
  auto subgraph = CreateSubGraph({node.get()}, {&graph_input}, {&output}, nullptr);
  ASSERT_NE(subgraph, nullptr);
  (void)node.release();

  ASSERT_EQ(constant.data(), host_data);
  ASSERT_EQ(constant.allocator(), nullptr);
  ASSERT_FALSE(allocator_->HasDeviceMemPtr(constant.data()));
  ASSERT_TRUE(std::equal(const_data.begin(), const_data.end(), static_cast<float *>(constant.data())));
}
}  // namespace mindspore::lite::dsp::test
