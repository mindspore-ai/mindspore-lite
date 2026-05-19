---
name: lite-converter
description: Model conversion pipeline, parser development, optimization passes and quantization. Use when converting models to .ms, writing parser code, implementing optimizer passes, or configuring quantization.
paths:
  - mindspore-lite/tools/converter/**
  - mindspore-lite/tools/optimizer/**
  - mindspore-lite/schema/**
  - mindspore-lite/tools/schema_gen/**
---

# MindSpore Lite Model Conversion and Optimization

## Conversion Pipeline

```
Input Model (MindIR/TF/Caffe/ONNX/TFLite/PyTorch)
  -> Parse (framework-specific Parser) -> Unified MindIR (ANF Graph)
  -> Import -> Internal graph representation
  -> Optimize (Constant Folding, Op Fusion, Format Transform, Parallel Split, Redundant elimination)
  -> Quantize (optional: Weight / Full / Mixed Precision)
  -> Export (.ms for LiteRT or .mindir for ExtendRT)
```

## Conversion Tool Usage

```bash
# MindIR -> .ms (device-side)
./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model

# TensorFlow / TFLite / Caffe / ONNX / PyTorch -> .ms
./converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
./converter_lite --fmk=CAFFE --modelFile=model.prototxt --weightFile=model.caffemodel --outputFile=model
./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
./converter_lite --fmk=PYTORCH --modelFile=model.pt --outputFile=model

# With quantization
./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model \
  --quantType=WeightQuant --bitNum=8

### Key Parameters

| Parameter | Description |
|------|------|
| `--fmk` | Input framework: MINDIR/TF/TFLITE/CAFFE/ONNX/PYTORCH |
| `--modelFile` | Input model file path |
| `--weightFile` | Caffe weight file (Caffe only) |
| `--outputFile` | Output file path (without extension) |
| `--quantType` | WeightQuant / FullQuant / NoQuant |
| `--bitNum` | Quantization bits: 1-8 (default 8) |
| `--optimize` | ascend_oriented / general / none |
| `--configFile` | Quantization or runtime config file |
| `--inputShape` | Dynamic shape input (e.g., `input1:1,3,224,224;input2:1,3,256,256`) |

## Parser Development

```
mindspore-lite/tools/converter/parser/
  onnx/             # onnx_model_parser.cc + per-operator parsers
  tf/               # tf_model_parser.cc + per-operator parsers
  tflite/           # tflite_model_parser.cc + per-operator parsers
  caffe/            # caffe_model_parser.cc + per-operator parsers
```

### Adding a New Framework Parser

1. Create directory under `tools/converter/parser/`
2. Implement `XxxModelParser` with `Parse()` method (original model -> ANF Graph)
3. Operator mapping: original operator -> MindSpore Primitive
4. Register FMK type in `tools/converter/converter_flags.cc`
5. Add unit tests

### Operator Mapping

Each Parser maps original operators to MindSpore internals:

- **Attribute conversion**: e.g., TF padding format to MindSpore format
- **Data layout**: NHWC <-> NCHW conversion
- **Missing operators**: simulate with composites (e.g., ReduceMean for GlobalAvgPool)

## Converter Directory Structure

```
mindspore-lite/tools/converter/
  adapter/          # Format adapters
  config_parser/    # Config file parsing
  converter_lite/   # CLI tool entry point
  cxx_api/          # Converter C++ API
  decomposer/       # Operator decomposition
  import/           # Model import (mindir_importer, primitive_adjust)
  legacy_optimizer/ # Legacy optimization passes
  micro/            # Micro code generation
  ops/              # Operator utilities
  parser/           # Framework-specific parsers
  preprocess/       # Data preprocessing pipeline
  quantizer/        # Quantization implementations
    calibrator.cc        # Calibration data processing
    full_quant_quantizer/  # Full quantization
    weight_quantizer/      # Weight-only quantization
    gptq_quantizer/        # GPTQ quantization
    fse_encoder/           # FSE encoding
    huffman_encode/        # Huffman encoding
  registry/         # Extension registration
  session/          # Conversion session
```

## Optimizer Pass Development

```
mindspore-lite/tools/optimizer/
  common/           # Common optimization utilities
  const_fold/       # Constant folding passes
  fiss on/          # Operator fission passes (note: directory is "fisson")
  format/           # Format transform passes
  fusion/           # Operator fusion passes (conv_bn, conv_activation, matmul_add, etc.)
  graph/            # Graph-level optimization
  parallel/         # Parallel split passes
```

### Writing an Optimization Pass

```cpp
class MyFusionPass : public Pass {
 public:
  bool Run(const FuncGraphPtr &graph) override {
    auto node_list = TopoSort(graph->get_return());
    for (auto &node : node_list) {
      if (!CheckPattern(node)) continue;
      DoFusion(node);
      changed = true;
    }
    return changed;
  }
};
```

Passes execute during converter phase in a fixed order chain.

## Quantization

### Post-training Quantization

1. Prepare calibration dataset (100-500 representative samples)
2. Write config file:

```ini
[common_quant_param]
quant_type=WEIGHT_QUANT
bit_num=8

[data_preprocess]
calibrate_path=/path/to/calibration/images/
calibrate_size=100

[input_format]
input_type=IMAGE
resize_height=224
resize_width=224
```

3. Run: `./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model_quant --configFile=quant_config.cfg`

### Quantization Types

| Type | Description | Use Case |
|------|------|---------|
| WeightQuant | Weight-only | Reduce model size, minimal accuracy loss |
| FullQuant | Weight + activation | Maximum compression, needs calibration data |
| Mixed Precision | Partial INT8 + FP32 | Balance accuracy and performance |

## Export Formats

| Format | Target Runtime | Serialization |
|------|------|----------|
| `.ms` | LiteRT (device-side) | FlatBuffers, zero-copy deserialization |

Schema files: `mindspore-lite/schema/ops.fbs` (~1.3K lines), `model.fbs`, `ops_types.fbs`

## Common Conversion Issues

1. **Unsupported operator**: Check `schema/ops.fbs` operator list
2. **Shape inference failure**: Use `--inputShape` to specify input shapes
3. **Quantization accuracy drop**: Try mixed precision or more calibration data
4. **Caffe missing BatchNorm params**: Ensure `.caffemodel` weight file path is correct
