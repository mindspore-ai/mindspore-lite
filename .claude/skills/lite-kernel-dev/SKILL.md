---
name: lite-kernel-dev
description: Operator and kernel development, NNACL, delegates, custom kernel registration. Use when adding operators, implementing NNACL kernels, writing delegate adapters (NPU/CoreML/Ascend), registering custom kernels, or modifying operator schema.
paths:
  - mindspore-lite/src/litert/kernel/**
  - mindspore-lite/src/litert/delegate/**
  - mindspore-lite/src/litert/kernel_registry.*
  - mindspore-lite/src/extendrt/delegate/**
  - mindspore-lite/schema/ops.fbs
  - mindspore-lite/providers/**
---

# MindSpore Lite Operator and Kernel Development

## Kernel Architecture

```
mindspore-lite/src/litert/kernel/
  cpu/
    base/              # Base kernel implementations (convolution_base, reduce_base, etc.)
    bolt/              # Bolt optimized kernels
    control/           # Control flow kernels (If, While)
    fp32/              # Float32 operator implementations
    fp16/              # Float16 operators (ARM v8.2)
    fp32_grad/         # Training gradient kernels (FP32)
    fp16_grad/         # Training gradient kernels (FP16)
    fp32_sparse/       # Sparse computation kernels
    int8/              # INT8 quantized operators
    nnacl/             # NNACL C++ kernel wrapper layer
    string/            # String type operators
  gpu/
    opencl/            # OpenCL GPU kernel backend
  opencl/              # OpenCL runtime infrastructure (allocator, wrapper)
  dsp/                 # DSP backend
```

Note: `LiteKernel` base class is in `mindspore-lite/src/litert/lite_kernel.cc/h`.

## Adding a New Operator

### Step 1: Schema Definition

Add operator in `mindspore-lite/schema/ops.fbs` (~1.3K lines):

```
// 1. Add to OpType enum
table MyCustomOp {
    attr1: int;
    attr2: float;
}
// 2. Register in union Op
union Op {
    ... existing operators
    MyCustomOp = XXX,  // Next available number
}
```

Generate code:

```bash
cd mindspore-lite/schema
flatc --cpp --gen-mutable --gen-object-api ops.fbs
flatc --cpp --gen-mutable --gen-object-api model.fbs
```

Also update: `schema/ops_types.fbs`, `schema/gpu_cache.fbs`.

### Step 2: NNACL C Implementation

Create under `mindspore-lite/src/litert/kernel/cpu/nnacl/`:

```c
// nnacl/my_custom_op.h
typedef struct MyCustomOpParameter {
  OpParameter op_parameter_;  // Must be first member
  int attr1;
  float attr2;
} MyCustomOpParameter;

int DoMyCustomOpFp32(void *input, void *output, MyCustomOpParameter *param, int size);
```

**NNACL naming**: `operator_datatype.c` (e.g., `matmul_fp32.c`, `conv_int8.c`). Assembly in `nnacl/assembly/`.

### Step 3: C++ Kernel Wrapper

```cpp
// kernel/cpu/fp32/my_custom_op_cpu_kernel.h
class MyCustomOpCPUKernel : public LiteKernel {
 public:
  int Prepare() override;
  int ReSize() override;
  int Run() override;
 private:
  MyCustomOpParameter *param_{nullptr};
};
```

### Step 4: Registration

```cpp
REGISTER_CUSTOM_KERNEL(CPU, VendorName, kNumberTypeFloat32, kMyCustomOp, CreatorFunc)
```

Or use `REGISTER_CLASS_CREATOR` macro in `kernel_registry.cc`.

### Step 5: Unit Tests

```cpp
// test/ut/src/runtime/kernel/arm/my_custom_op_test.cc
TEST(MyCustomOpTest, BasicTest) {
  // Setup tensor, params, context
  auto *kernel = new kernel::MyCustomOpCPUKernel(param, inputs, outputs, &ctx);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
  ASSERT_EQ(kernel->Run(), RET_OK);
}
```

Build with `MSLITE_ENABLE_TESTCASES=ON`.

## Delegate Mechanism

Delegate abstracts hardware acceleration, allowing subgraphs to execute on dedicated hardware.

### LiteRT Delegates

```
mindspore-lite/src/litert/delegate/
  npu/                # HiSilicon NPU delegate
  coreml/             # Apple CoreML delegate (.mm files)
  pnna/               # PNNA NPU delegate
  parameter_cache/    # Parameter cache (GPU/host/embedding)
```

### ExtendRT Delegates

```
mindspore-lite/src/extendrt/delegate/
  ascend_ge/          # Ascend Graph Engine delegate (full GE integration)
  ascend_acl/         # Ascend ACL direct kernel delegate
  graph_executor/     # Graph executor with LiteRT plugin
  plugin/             # Plugin dynamic loading
```

### Writing a Custom Delegate

```cpp
class MyDelegate : public Delegate {
 public:
  Status Init() override;
  Status Build(DelegateModel<schema::Primitive> *model) override;
  // Traverse graph -> identify supported ops -> group into subgraphs
  // -> Replace with hardware execution
};
```

Register via `DelegateRegistry` or `context->SetDelegate()`.

### Custom Kernel Registration (Public API)

```
include/registry/
  register_kernel.h         # REGISTER_KERNEL / REGISTER_CUSTOM_KERNEL macros
  register_kernel_interface.h  # REGISTER_KERNEL_INTERFACE macro
  kernel_interface.h         # KernelInterface base class
  pass_base.h               # Pass base class for optimizer extensions
  pass_registry.h           # REG_PASS / REG_SCHEDULED_PASS macros
```

Registration macros:

```cpp
REGISTER_KERNEL(arch, provider, data_type, op_type, creator)
REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)
REGISTER_KERNEL_INTERFACE(provider, op_type, creator)
```

## Provider Extensions

```
mindspore-lite/providers/
  dpico/            # DPICO NPU plugin
  nnie/             # NNIE hardware plugin
  nnie_proposal/    # NNIE Proposal plugin
  siteai/           # SiteAI plugin
```

Providers compile independently from main repository.

## Operator Fusion (Optimizer Pass)

```cpp
// In mindspore-lite/tools/optimizer/fusion/
class ConvBNFusionPass : public Pass {
 public:
  bool Run(const FuncGraphPtr &graph) override {
    // 1. Match Conv2D -> BatchNorm pattern
    // 2. Absorb BN params into conv weights
    // 3. Replace original nodes
    return changed;
  }
};
```

Fusion passes execute during converter phase, not at runtime.

## Schema Change Process

1. Edit `.fbs` files in `mindspore-lite/schema/`
2. Run `tools/schema_gen/` to generate C++ code
3. Backward compatibility: new fields use defaults, do not remove existing fields
4. Synchronize converter parser mapping

## Performance Optimization Points

- **Assembly optimization**: Critical ops (Conv, MatMul) have NEON/SVE implementations in NNACL
- **Memory layout**: NHWC (device-side default) vs NCHW
- **Quantization**: New ops should provide fp32, fp16, and int8 implementations
- **Parallel splitting**: Large ops (MatMul) support multi-threaded split
