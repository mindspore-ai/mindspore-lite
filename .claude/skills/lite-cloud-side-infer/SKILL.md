---
name: lite-cloud-side-infer
description: Cloud-side inference with ExtendRT and Ascend backends. Use for server-side inference, Ascend 310/910 deployment, ModelParallelRunner for concurrent serving, ModelGroup for weight sharing, distributed inference, or .mindir format loading.
paths:
  - mindspore-lite/src/extendrt/**
---

# MindSpore Lite Cloud-side Inference (ExtendRT)

## Architecture Overview

ExtendRT is the cloud-side inference runtime for servers and data centers.

```
              +-----------------------------------------+
              |       ExtendRT Cloud Runtime            |
              +-----------------------------------------+
  .mindir ->  |  Graph Scheduler                       |
              |  Graph Executor                         |
              |  MindIR Loader                          |
              +-----------------------------------------+
              |  Delegates:                             |
              |    Ascend GE (Graph Engine)             |
              |    Ascend ACL (direct kernel)           |
              |    CPU (fallback to LiteRT via plugin)  |
              +-----------------------------------------+
              |  ModelParallelRunner (concurrent infer) |
              |  ModelGroup (weight sharing)            |
              +-----------------------------------------+
```

### Key Directories

```
mindspore-lite/src/extendrt/
  cxx_api/              # Model API implementation
  delegate/
    ascend_ge/          # Ascend Graph Engine delegate (full GE integration)
    ascend_acl/         # Ascend ACL direct kernel delegate
    graph_executor/     # Graph executor with LiteRT plugin
    plugin/             # Plugin dynamic loading (LiteRT fallback)
  session/              # Session implementations (DelegateSession, factory)
  mindir_loader/        # MindIR model loading
  convert/              # Conversion utilities
  utils/                # Utility code
  mock/                 # Mock implementations for testing
```

**LiteRT fallback**: ExtendRT loads LiteRT via `delegate/plugin/litert_executor_plugin.cc` for CPU subgraph execution.

## Supported Hardware

| Hardware            | Device Type | Usage |
|---------------------|------------|-------|
| Atlas 300I Duo/800I | Atlas Inference Series | Inference |
| NVIDIA GPU          | Tesla/A100 | TensorRT (device-side runtime) |
| CPU                 | x86_64 / aarch64 | General Inference |

## Three Ascend Inference Backends

| Backend | Provider | Features |
|---------|----------|----------|
| **ACL** (default) | unset | Global/model-level options. Independent graphs. No weight sharing. Supports pre-built models. |
| **GE** | `"ge"` | Global/session/graph-level options. Weight sharing across models in same session. Variable support. No pre-built models. |
| **GE-v1** | `"ge-v1"` | Refactored GE for zero-copy inference (v2.8+). Device memory for input/output. Eliminates host-device round trips. |

## API Usage

### C++ Inference

```cpp
#include "include/api/model.h"
#include "include/api/context.h"

auto context = std::make_shared<mindspore::Context>();
auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
ascend->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend);

auto model = std::make_shared<mindspore::Model>();
model->Build(model_path, mindspore::kMindIR, context);

auto inputs = model->GetInputs();
// Fill inputs...
std::vector<mindspore::MSTensor> outputs;
model->Predict(inputs, &outputs);
model->Reset();  // Release compilation resources
```

### Precision Modes (Ascend)

```cpp
ascend->SetPrecisionMode("enforce_fp32");     // Force FP32
ascend->SetPrecisionMode("preferred_fp32");   // Prefer FP32, some FP16
ascend->SetPrecisionMode("enforce_fp16");     // Force FP16
ascend->SetPrecisionMode("preferred_optimal"); // Auto optimal
```

### Dynamic Shape (Ascend)

Config file:

```ini
[ascend_context]
input_shape=input_1:[-1,3,224,224]
dynamic_dims=[1~4],[8],[16]
```

`-1` marks dynamic dims. Ranges: `[start~end]`. Discrete: `[val]`. Larger range = longer compilation.

```cpp
model->LoadConfig(config_file);
model->Build(model_path, mindspore::kMindIR, context);
```

### Zero-copy (GE-v1)

```cpp
ascend->SetProvider("ge-v1");
// Create device-side tensors for pipeline without host-device round trips
auto input_tensor = MSTensor::CreateTensor("input", kFloat32, {1, 3, 224, 224},
                                            data, size, "ascend", 0);
```

## ModelParallelRunner (Concurrent Serving)

```cpp
#include "include/api/model_parallel_runner.h"

auto runner_config = std::make_shared<mindspore::RunnerConfig>();
auto context = std::make_shared<mindspore::Context>();
auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
ascend->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend);
runner_config->SetContext(context);
runner_config->SetWorkersNum(4);

auto runner = std::make_shared<mindspore::ModelParallelRunner>();
runner->Init(model_path, runner_config);

auto inputs = runner->GetInputs();
// Fill inputs per request...
std::vector<mindspore::MSTensor> outputs;
runner->Predict(inputs, &outputs);
```

Constraints:

- FP32 data inference not supported (use FP16 or quantization)
- CPU pinning: unbound or big cores only
- Workers x threads should not exceed machine cores

## ModelGroup (Weight Sharing)

```cpp
auto group = std::make_shared<mindspore::ModelGroup>(
    mindspore::ModelGroupFlag::kShareWeight);
// Or: kShareWorkspace (Ascend), kShareWeightAndWorkspace
group->AddModel({model_path1, model_path2});
group->CalMaxSizeOfWorkspace(mindspore::kMindIR, context);
```

## Distributed Inference

Multi-process for Atlas training series. Each process loads a sliced model.

```cpp
ascend->SetRankID(rank_id);
ascend->SetProvider("ge");  // Only GE supports distributed
model->LoadConfig(config_path);  // HCCL networking info
model->Build(model_path, mindspore::kMindIR, context);
```

Requires: `provider="ge"`, separate `device_id`/`rank_id` per process, HCCL config file.

## Configuration File Sections

| Section | Purpose |
|---------|---------|
| `[ascend_context]` | input_shape, dynamic_dims, timeout, precision_mode |
| `[common_context]` | compile_graph_parallel (on/off) |
| `[ge_global_options]` | GE global options |
| `[ge_session_options]` | ge.externalWeight (for weight sharing) |
| `[ge_graph_options]` | precision, inputShape, dynamicDims |

### ACL Inference Timeout

```ini
[ascend_context]
timeout=-1   # Wait indefinitely
timeout=50   # Limit to 50ms
```

### Multi-threaded Model Loading

```ini
[common_context]
compile_graph_parallel=on
```

## Build and Deploy

```bash
# Cloud-side CPU
bash build.sh -I x86_64 -e cpu -a x64 -j8

# Cloud-side Ascend
bash build.sh -I x86_64 -e ascend -a x64 -j8

# CMake options: MSLITE_ENABLE_CLOUD_INFERENCE=ON, MSLITE_ENABLE_ACL=ON
```

### Output Packages

```
mindspore-lite-{version}-linux-x64.tar.gz      # CPU cloud-side
mindspore-lite-{version}-linux-aarch64.tar.gz   # ARM cloud-side
```

### Sample Code Locations

| Feature | Path |
|---------|------|
| Basic C++ | `examples/cloud_infer/runtime_cpp` |
| Basic Python | `examples/cloud_infer/quick_start_python` |
| Parallel C++ | `examples/cloud_infer/quick_start_parallel_cpp` |
| Parallel Python | `examples/cloud_infer/quick_start_parallel_python` |
| Distributed | `examples/cloud_infer/ascend_ge_distributed_cpp` |
