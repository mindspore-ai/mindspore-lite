---
name: lite-device-side-infer
description: Device-side inference with LiteRT, NNACL and hardware delegates. Use for mobile/IoT inference, Android/iOS integration, NPU/GPU/CoreML delegates, Micro codegen for MCU, on-device training, or C/C++/Java/Python API usage with .ms models.
paths:
  - mindspore-lite/src/litert/**
  - mindspore-lite/java/**
  - mindspore-lite/tools/cropper/**
  - include/api/**
  - include/c_api/**
---

# MindSpore Lite Device-side Inference (LiteRT)

## Architecture Overview

```
                +-------------------------------------+
                |         LiteRT Runtime              |
                +-------------------------------------+
  .ms -->       |  LiteSession (session management)   |
                |  Scheduler (graph scheduling)       |
                |  Executor (kernel execution)        |
                |  MindrtExecutor (Actor parallel)    |
                +-------------------------------------+
                |  NNACL (high-perf operator library) |
                |    FP32 / FP16 / INT8               |
                |    ARM assembly optimized (NEON/SVE) |
                +-------------------------------------+
                |  Delegates:                         |
                |    HiSilicon NPU                     |
                |    Apple CoreML                      |
                |    PNNA                              |
                |    OpenCL GPU                        |
                +-------------------------------------+
                |  Micro (MCU code generation)        |
                |  On-device Training                 |
                +-------------------------------------+
```

### Key Directories

```
mindspore-lite/src/litert/
  c_api/                    # C API (embedded-friendly)
  cxx_api/
    model/                  # Model API
    tensor/                 # Tensor API
    train/                  # Device-side training API
    graph/                  # Graph API
    kernel_executor/        # Direct kernel execution
  kernel/
    cpu/                    # CPU kernels (fp32/fp16/int8/bolt/control/...)
    gpu/opencl/             # OpenCL GPU kernels
    opencl/                 # OpenCL runtime (allocator, wrapper)
    dsp/                    # DSP backend
  delegate/
    npu/                    # HiSilicon NPU delegate
    coreml/                 # Apple CoreML delegate (.mm)
    pnna/                   # PNNA NPU delegate
    parameter_cache/        # Parameter cache (GPU/host/embedding)
  pass/                     # Runtime optimization passes
  lite_session.cc           # Inference session
  scheduler.cc              # Kernel scheduling, subgraph partitioning
  executor.cc               # Kernel executor
  mindrt_executor.cc        # Actor model parallel execution
  weight_decoder.cc         # Quantized weight decoding
  pack_weight.cc            # NPU/DSP weight packing
```

## Model Conversion and Deployment

```bash
# MindIR -> .ms
./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model

# With quantization
./converter_lite --fmk=MINDIR --modelFile=model.mindir \
  --outputFile=model --quantType=WeightQuant --bitNum=8

# TF/ONNX -> .ms
./converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
```

### Library Cropper

```bash
# Remove unused operators to reduce so size
./cropper --modelFile=model.ms --outputFile=libmindspore-lite-cropped.so
```

## C++ API (Primary)

```cpp
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/mstensor.h"

auto context = std::make_shared<mindspore::Context>();
context->SetThreadNum(2);
context->SetThreadAffinity(0);  // 0=none, 1=big cores, 2=little cores

auto &devices = context->MutableDeviceInfo();
// CPU (required as fallback)
auto cpu = std::make_shared<mindspore::CPUDeviceInfo>();
cpu->SetEnableFP16(false);
devices.push_back(cpu);
// GPU (optional)
devices.push_back(std::make_shared<mindspore::GPUDeviceInfo>());
// NPU (optional, frequency: 1=low, 2=balanced, 3=high, 4=extreme)
auto npu = std::make_shared<mindspore::KirinNPUDeviceInfo>();
npu->SetFrequency(3);
devices.push_back(npu);

auto model = std::make_shared<mindspore::Model>();
model->Build(model_path, mindspore::ModelType::kMindIR, context);

auto inputs = model->GetInputs();
for (size_t i = 0; i < inputs.size(); i++)
    memcpy(inputs[i].MutableData(), data[i], inputs[i].DataSize());

std::vector<mindspore::MSTensor> outputs;
model->Predict(inputs, &outputs);
```

**Context configuration**: up to 3 devices ordered by priority (NPU > GPU > CPU fallback).

### Model::Build Signatures

```cpp
Build(path, ModelType, Context)            // From file path
Build(data, size, ModelType, Context)      // From buffer
Build(path, ModelType, Context, Key, ...)  // Encrypted model
```

### Dynamic Shape

```cpp
model->Resize(inputs, {{1, 3, 224, 224}});
```

### Callback Inference

```cpp
auto cb = [](const std::vector<MSTensor> &in, const std::vector<MSTensor> &out,
             const MSCallBackParam &param) -> bool {
  MS_LOG(INFO) << param.node_name_;
  return true;
};
model->Predict(inputs, &outputs, nullptr, cb);
```

## C API (Embedded/MCU)

```c
#include "include/c_api/model_c.h"
OH_AI_ContextHandle ctx = OH_AI_ContextCreate();
OH_AI_ContextSetThreadNum(ctx, 2);
OH_AI_DeviceInfoHandle cpu = OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);
OH_AI_ContextAddDeviceInfo(ctx, cpu);

OH_AI_ModelHandle model = OH_AI_ModelCreate();
OH_AI_ModelBuild(model, path, OH_AI_MODELTYPE_MINDIR, ctx);
OH_AI_TensorHandleArray inputs = OH_AI_ModelGetInputs(model);
OH_AI_TensorHandleArray outputs;
OH_AI_ModelPredict(model, inputs, &outputs);
OH_AI_ModelDestroy(&model);
```

## Java API (Android)

```java
Context context = new Context();
context.cpu = new Cpu(); context.cpu.threadNum = 2;
context.gpu = new Gpu();  // optional
context.npu = new Npu();  // optional

Model model = new Model();
model.compile(modelPath, ModelType.MT_MINDIR, context);
MSTensor input = model.getInputByTensorName("input");
input.setData(inputData);
Map<String, MSTensor> outputs = model.run();
model.free();
```

Integration: `mindspore-lite.aar` in `libs/` with Gradle flatDir.

## Python API

```python
import mindspore_lite as mslite
context = mslite.Context()
context.append_device_info(mslite.CPUDeviceInfo(fp16=False))
model = mslite.Model()
model.build(model_path, mslite.ModelType.MINDIR, context)
inputs = model.get_inputs()
outputs = model.predict(inputs)
```

## Heterogeneous Scheduling

```
Scheduler receives graph
  -> Traverse nodes
     -> NPU supports? -> NPU subgraph
     -> GPU supports? -> GPU subgraph
     -> CPU subgraph (NNACL)
  -> Merge adjacent same-type subgraphs
  -> Generate execution sequence
```

`src/litert/kernel_registry.cc` manages kernel registration, selects by architecture and data type.

## NNACL Kernel Library

```
mindspore-lite/src/litert/kernel/cpu/nnacl/
  # C++ wrappers calling pure C implementations
  # Flat file layout: nnacl_convolution.cc, nnacl_matmul.cc, etc.
```

Data type support: FP32 (standard), FP16 (ARM v8.2+), INT8 (quantized, fastest).

## On-device Training

```cpp
auto train_cfg = std::make_shared<mindspore::TrainCfg>();
model->Build(train_path, mindspore::ModelType::kMindIR, context, train_cfg);
model->SetTrainMode(true);
model->RunStep();    // Single training step
model->SetTrainMode(false);
model->Export(output_path);
```

Training gradient kernels: `kernel/cpu/fp32_grad/` and `fp16_grad/`.

## Micro Code Generation

```bash
./converter_lite --fmk=MINDIR --modelFile=model.mindir \
  --outputFile=model --configFile=micro_config.cfg
# Generates standalone C code, no runtime dependency
# Runs on Cortex-M / RISC-V MCUs
```

## Memory Optimization

| Mechanism | Description |
|------|------|
| `InnerAllocator` | Basic memory allocator |
| `RuntimeAllocator` | Runtime memory pool (max 3GB, max 2GB per allocation) |
| Memory pool sharing | `device_info2->SetAllocator(device_info1->GetAllocator())` |

## Execution Pipeline

```
LiteSession::Init()
  -> LiteModel loads .ms (FlatBuffers)
    -> Scheduler: subgraph partitioning by device
      -> KernelRegistry selects kernels
        -> Executor::Run() or MindrtExecutor (Actor parallel)
```

## Build Configuration

```bash
bash build.sh -I arm64 -j8                  # ARM64 (Android)
bash build.sh -I arm64 -e gpu -j8           # With OpenCL GPU
bash build.sh -I arm64 -e npu -j8           # With NPU
bash build.sh -I arm64 -e cpu -t on -j8     # With training
bash build.sh -I arm64 -T ios -j8           # iOS
bash build.sh -I cortex-m -j8               # Micro MCU
```
