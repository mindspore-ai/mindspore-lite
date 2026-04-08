# Qwen3-VL-2B ONNX Export and Inference

This directory provides complete tooling for exporting Qwen3-VL-2B to ONNX format and running end-to-end inference. The implementation splits the model into three optimized components for efficient deployment.

## Overview

Qwen3-VL-2B is a multimodal large language model that processes both images and text. This toolkit provides:

- **ONNX Export**: Split model into Vision, LLM Prefill, and LLM Decode components

- **ONNX Inference**: Complete end-to-end inference pipeline with all three models

- **MindSpore Lite Integration**: Optional conversion to `.mindir` format for Ascend deployment

## Architecture

The model is split into three ONNX components:

1. **Vision Tower** (`qwen3_vl_vision.onnx`): Extracts visual features from images
2. **LLM Prefill** (`qwen3_vl_llm_prefill.onnx`): Processes the full prompt (text + image tokens) in one pass
3. **LLM Decode** (`qwen3_vl_llm_decode.onnx`): Incremental token generation using KV cache

This separation avoids KV cache recomputation and enables efficient autoregressive generation.

## Prerequisites

### Python Environment

```bash
pip install -U "transformers>=4.50" torch onnx onnxscript pillow numpy
pip install -U onnxruntime
```

For GPU acceleration:

```bash
pip install -U onnxruntime-gpu
```

### Model Access

Ensure you have access to `Qwen/Qwen3-VL-2B-Instruct` on HuggingFace.

## Quick Start

### 1. Export to ONNX

Export all three ONNX models:

```bash
python export_qwen3_vl_onnx.py \
    --model-id Qwen/Qwen3-VL-2B-Instruct \
    --output-dir ./qwen3_vl_onnx \
    --device cpu \
    --vision-image-size 128
```

This generates:

- `qwen3_vl_vision.onnx` - Vision tower

- `qwen3_vl_llm_prefill.onnx` + `.data` - LLM prefill model

- `qwen3_vl_llm_decode.onnx` + `.data` - LLM decode model

### 2. Run ONNX Inference

Execute end-to-end inference:

```bash
python infer_qwen3_vl_onnx.py \
    --vision qwen3_vl_onnx/qwen3_vl_vision.onnx \
    --prefill qwen3_vl_onnx/qwen3_vl_llm_prefill.onnx \
    --decode qwen3_vl_onnx/qwen3_vl_llm_decode.onnx \
    --processor Qwen/Qwen3-VL-2B-Instruct \
    --image ./your_image.jpg \
    --prompt "Describe this image." \
    --max-new-tokens 128 \
    --device cpu
```

## Model I/O Specifications

### Vision Model

**Inputs:**

- `pixel_values`: `float16`, shape `(seq_len, 1536)`

- `grid_thw`: `int64`, shape `(1, 3)` - temporal, height, width grid dimensions

**Outputs:**

- `image_embeds`: `float16`, shape `(num_image_tokens, hidden_size)`

- `deepstack_embeds`: `float16`, shape `(num_deepstack, num_image_tokens, hidden_size)`

### LLM Prefill Model

**Inputs:**

- `input_ids`: `int64`, shape `(batch, seq_len)`

- `attention_mask`: `int64`, shape `(batch, seq_len)`

- `position_ids`: `int64`, shape `(4, batch, seq_len)`

- `image_embeds`: `float16`, shape `(num_image_tokens, hidden_size)`

- `deepstack_embeds`: `float16`, shape `(num_deepstack, num_image_tokens, hidden_size)`

**Outputs:**

- `logits`: `float16/float32`, shape `(batch, seq_len, vocab_size)`

- `present_key_values`: `float16`, shape `(2*num_layers, batch, num_kv_heads, seq_len, head_dim)`

### LLM Decode Model

**Inputs:**

- `input_ids`: `int64`, shape `(batch, 1)` - single token
- `attention_mask`: `int64`, shape `(batch, total_seq_len)`
- `position_ids`: `int64`, shape `(4, batch, 1)`
- `past_key_values`: `float16`, shape `(2*num_layers, batch, num_kv_heads, past_seq_len, head_dim)`

**Outputs:**

- `logits`: `float16/float32`, shape `(batch, 1, vocab_size)`
- `present_key_values`: `float16`, shape `(2*num_layers, batch, num_kv_heads, total_seq_len, head_dim)`

## MindSpore Lite Integration (Optional)

For deployment on Ascend devices, convert ONNX models to `.mindir` format:

```bash
# Convert Vision model
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_onnx/qwen3_vl_vision.onnx \
    --outputFile=./qwen3_vl_onnx/qwen3_vl_vision \
    --device=Ascend \
    --saveType=MINDIR

# Convert Prefill model
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_onnx/qwen3_vl_llm_prefill.onnx \
    --outputFile=./qwen3_vl_onnx/qwen3_vl_llm_prefill \
    --device=Ascend \
    --saveType=MINDIR

# Convert Decode model
converter_lite \
    --fmk=ONNX \
    --modelFile=./qwen3_vl_onnx/qwen3_vl_llm_decode.onnx \
    --outputFile=./qwen3_vl_onnx/qwen3_vl_llm_decode \
    --device=Ascend \
    --saveType=MINDIR
```

Then run inference with MindSpore Lite:

```bash
python infer_qwen3_vl_mslite.py \
    --vision-model ./qwen3_vl_onnx/qwen3_vl_vision.mindir \
    --prefill-model ./qwen3_vl_onnx/qwen3_vl_llm_prefill.mindir \
    --decode-model ./qwen3_vl_onnx/qwen3_vl_llm_decode.mindir \
    --image ./your_image.jpg \
    --prompt "Describe this image." \
    --device-id 0
```

## File Structure

```Shell
qwen3_vl_2b/
├── export_qwen3_vl_onnx.py          # ONNX export script (3 models)
├── infer_qwen3_vl_onnx.py           # Complete ONNX inference pipeline
├── infer_qwen3_vl_mslite.py         # MindSpore Lite inference
├── README.md                        # This file
└── qwen3_vl_onnx/                   # Exported models directory
    ├── qwen3_vl_vision.onnx
    ├── qwen3_vl_llm_prefill.onnx + .data
    └── qwen3_vl_llm_decode.onnx + .data
```

## Key Features

### Prefill/Decode Separation

The LLM is split into prefill and decode models to optimize inference:

- **Prefill**: Processes the entire prompt (including image tokens) in one forward pass
- **Decode**: Generates tokens incrementally using cached key-value pairs
- **Benefit**: Avoids recomputing attention for all previous tokens at each generation step

### Memory Management

The export script includes automatic memory management to handle large models:

- Clears CUDA cache between export phases
- Deletes unused model components
- Works on systems with 8GB RAM

### Dynamic Shape Support

Models support dynamic batch sizes and sequence lengths through ONNX dynamic axes.

## Troubleshooting

### Memory Issues During Export

If you encounter OOM errors during export:

- Use `--device cpu` for CPU export (slower but lower memory)
- Reduce `--vision-image-size` (default 128)
- Close other applications to free memory

### Image Embeds Length Mismatch

If inference fails with image embeds length mismatch:

- Ensure processor and model versions match
- Verify `--vision-image-size` matches the export configuration
- Check that `image_grid_thw` is consistent

## References

- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [MindSpore Lite Ascend Inference](https://www.mindspore.cn/lite/docs/zh-CN/master/use/ascend_info.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

## License

This toolkit follows the license of the Qwen3-VL model. Please refer to the [Qwen3-VL license](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) for details.
