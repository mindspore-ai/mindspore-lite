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
| Ascend | Atlas 300I Duo/800I A2/A3 | Inference |
| CPU                 | x86_64 / aarch64 | General Inference |

## Three Ascend Inference Backends

| Backend | Provider | Features |
|---------|----------|----------|
| **ACL** (default) | unset | Global/model-level options. Independent graphs. No weight sharing. Supports pre-built models. |
| **GE** | `"ge"` | Global/session/graph-level options. Weight sharing across models in same session. Variable support. No pre-built models. |
| **GE-v1** | `"ge-v1"` | Refactored GE for zero-copy inference (v2.8+). Device memory for input/output. Eliminates host-device round trips. |

## API Usage

### Model Converter

#### Ascend Model Conversion

```bash
converter_lite --fmk=ONNX --modelFile=/path/to/model --configFile=/path/to/config.ini --optimize=ascend_oriented --outputFile=/path/to/output
```

##### Dynamic Shape (Ascend)

Config file:

```ini
[acl_build_options]
input_shape="input1:-1,3,224,224;input2:1,3,-1,-1"
```

##### Dynamic Range Shape (Ascend)

Config file:

```ini
[acl_build_options]
input_format="ND"
input_shape="input1:-1,3,224,224;input2:1,3,-1,-1"
ge.dynamicDims="1,256,256;2,512,512"
```

##### Static Shape (Ascend)

Config file:

```ini
[acl_init_options]
input_shape="input1:1,3,224,224;input2:1,3,256,256"
```

#### Cpu Model Conversion

```bash
converter_lite --fmk=ONNX --modelFile=/path/to/model --outputFile=/path/to/output
```

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
model->Finalize();
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
| `[ascend_context]` | timeout, precision_mode |
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
export MSLITE_ENABLE_CLOUD_INFERENCE=on
bash build.sh -I x86_64 -j8

# Cloud-side Ascend
export MSLITE_ENABLE_CLOUD_INFERENCE=on
export MSLITE_ENABLE_ACL=on
bash build.sh -I arm64 -j8
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
