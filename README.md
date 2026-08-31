[查看中文](./README_CN.md)

## What Is MindSpore Lite

MindSpore Lite provides lightweight AI inference acceleration capabilities for different hardware devices, enabling intelligent applications and providing end-to-end solutions for developers. It offers development friendly, efficient, and flexible deployment experiences for algorithm engineers and data scientists, helping the AI software and hardware application ecosystem thrive. In the future, MindSpore Lite will work with the MindSpore AI community to enrich the AI software and hardware application ecosystem.

## Example

MindSpore Lite achieves double the inference performance for AIGC, speech algorithms, and CV model inference, and has been deployed in Huawei's flagship smartphones for commercial use. As shown in the figure below, MindSpore Lite supports image style transfer and image segmentation for CV algorithms.

<img src="docs/img/mindir_infer_case_1.gif" alt="mindir infer case 1" width="240"/> <img src="docs/img/mindir_infer_case_2.gif" alt="mindir infer case 2" width="240"/> <img src="docs/img/mindir_infer_case_3.gif" alt="mindir infer case 3" style="height: 135px;" width="240"/>

<img src="docs/img/screenshot_001.png" alt="original image for image segmentation" width="180"/> <img src="docs/img/screenshot_002.png" alt="image segmentation rendering" width="180"/> <img src="docs/img/screenshot_003.png" alt="image style transfer original image" width="180"/> <img src="docs/img/screenshot_004.png" alt="Image style transfer rendering" width="180"/>

## Quick Start

1. Compile

    MindSpore Lite has multiple different hardware backends, including:

    - For service side devices, users can compile dynamic libraries and Python wheel packages by setting compilation options such as `MSLITE_ENABLE_CLOUD_INFERENCE` for inference of upgrade and CPU hardware. For detailed compilation tutorials, please refer to [the official website of MindSpore Lite](https://www.mindspore.cn/lite/cloud_docs/en/master/use/build.html).

    - For end and edge devices, different dynamic libraries can be compiled through different cross compilation toolchains. For detailed compilation tutorials, please refer to [the official website of MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/build.html).

2. Model conversion

    MindSpore Lite supports the conversion of models serialized from various AI frameworks such as MindSpore, ONNX, TF, etc. into MindSpore Lite format IR. In order to achieve more efficient model inference, MindSpore Lite supports the conversion of models into `.ms` format or `.mindir` format, where:

    - The `.mindir` model is used for inference on service side devices and is more compatible with the model structure exported by the MindSpore training framework. It is mainly suitable for Ascend cards and X86/Arm architecture CPU hardware. For detailed conversion methods, please refer to [the Conversion Tool Tutorial](https://www.mindspore.cn/lite/cloud_docs/en/master/mindir/converter_tool.html).

    - The `.ms` model is mainly used for inference of end and edge devices, and is mainly suitable for terminal hardware such as Kirin NPU and Arm architecture CPU. In order to better reduce the size of the model file, the `.ms` model is serialized and deserialized through protobuffer. For detailed instructions on how to use the conversion tool, please refer to [the Conversion Tool](https://www.mindspore.cn/lite/docs/en/master/converter/converter_tool.html)

3. Model inference

    MindSpore Lite provides three APIs: Python, C++, and Java, and complete usage cases for the corresponding APIs:

    - Python API Interface Use Case

        [`.mindir` Reasoning Case Based on Python Interface](https://www.mindspore.cn/lite/cloud_docs/en/master/mindir/runtime_python.html)

    - C/C++ Complete Use Cases

        [`.mindir` model based on C/C++ Interface Inference Use Case](https://www.mindspore.cn/lite/cloud_docs/en/master/mindir/runtime_cpp.html)

        [`.ms` Model Based on C/C++ Interface Reasoning Case](https://developer.huawei.com/consumer/en/doc/harmonyos-guides/mindspore-guidelines-based-native)

    - Complete Java Use Cases

        [`.mindir` model based on Java interface inference use case](https://www.mindspore.cn/lite/cloud_docs/en/master/mindir/runtime_java.html)

## Technical Solution

### MindSpore Lite Features

<img src="docs/img/MindSpore-Lite-architecture-en.png" alt="MindSpore Lite Architecture" width="800"/>

- MindSpore Lite exports `.mindir` models for cloud-side inference on Atlas 300I Duo, Atlas 800IA3 Ascend accelerators and X86/ARM CPUs, and `.ms` models for device-side inference on general-purpose CPUs and Kirin NPUs.

1. Device and Cloud one-stop inference deployment

    - Provide end-to-end workflow for model transformation optimization, deployment, and inference.

    - The unified IR realizes the device-cloud AI application integration.

2. Lightweight

    - Provides model compression, which could help to improve performance.

    - Provides the ultra-lightweight inference solution - MindSpore Lite Micro, to meet the deployment requirements in extreme environments such as smart watches and headphones.

3. High-performance

    - The built-in kernel computing library NNACL supports high-performance inference for dedicated chips such as CPU, NNRt, and Ascend, maximizing hardware computing power while minimizing inference latency and power consumption.

    - Use assembly instructions to improve performance of kernels.

4. Versatility

    - Supports deployment of multiple hardware such as server-side Ascend and CPU.

    - Supports HarmonyOS and Android mobile operating systems.

## Further Understanding of MindSpore Lite

If you wish to further learn and use MindSpore Lite, please refer to the following content:

### Supported models for cloud-side inference

| LLM & VLM | Generative (Image / Video) | Speech (ASR / TTS) | Retrieval (Embedding / Reranker / NER) | Vision Perception (Detection / Backbone / 3D) | Embodied AI (VLA) |
| :-------: | :------------------------: | :----------------: | :------------------------------------: | :--------------------------------------------: | :---------------: |
| [Qwen3.5-0.8B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_5_0.8b) / [2B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3.5_2b) / [4B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3.5_4b) &#9989; | [Wan2.1-T2V-1.3B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/wan2_1_t2v_1_3b) &#9989; | [FireRedASR-AED-L](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/fireredasr_aed_l) &#9989; | [Qwen3-Embedding-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_embedding_0_6b) / [4B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_embedding_4b) &#9989; | [yolov8](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/yolov8) / [yolov10x](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/yolov10/yolov10-X) &#9989; / YOLO26 / YOLOv13 | [openpi pi 0.5](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/open_pi_0.5) &#9989; |
| [Qwen3-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_0.6b) / [1.7B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_1.7b) / [4B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_4b) / [8B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_8b) &#9989; | FLUX.1-dev | [Qwen3-ASR-1.7B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_asr_1.7b) &#9989; | [Qwen3-VL-Embedding-2B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_embedding_2b) &#9989; | [Grounding-DINO-Base](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/grounding_dino_base) &#9989; | OpenVLA |
| [Qwen2.5-0.5B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_0.5b) / [3B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_3b) / [7B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_7b) &#9989; | Stable-Diffusion-3.5-Large | [CosyVoice2-0.5B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/cosyvoice2_0.5b) &#9989; | [Qwen3-Reranker-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_reranker_0.6b) / [4B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_reranker_4b) / [8B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_reranker_8b) &#9989; | [vit-base-patch16-224](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/vit_base_patch16_224) &#9989; | RDT-1B |
| [Qwen2-0.5B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2_0.5b) / [1.5B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2_1.5b) / [7B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2_7b) &#9989; | SDXL | [Qwen3-TTS-12Hz-1.7B-CustomVoice](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_tts_12hz_1.7b_customvoice) &#9989; | [Qwen3-VL-Reranker-2B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_reranker_2b) &#9989; | [convnext](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/convnext) &#9989; | SmolVLA |
| [Qwen2.5-Math-1.5B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_math_1.5b) / [1.5B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_math_1.5b_instruct) &#9989; | Kandinsky-3 / Kandinsky-5.0 (T2I/I2V/I2I) | SenseVoice-Small | [jina-reranker-v3](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/jina_reranker_v3) &#9989; | [CLIP-ViT-Base-Patch32](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/clip_vit_base_patch32) &#9989; | CogACT |
| [Qwen3-VL-2B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_2b_instruct) / [2B-Thinking](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_2b_thinking) / [4B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_4b_instruct) / [4B-Thinking](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_4b_thinking) / [8B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_8b_instruct) &#9989; | Qwen-Image | Whisper-v3-Turbo | [GLiNER-Large-v2.5](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/gliner_large-v2.5) &#9989; | [bevdet-r50](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/bevdet) &#9989; | GR00T-N1 |
| [Qwen2.5-VL-3B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen2.5_vl_3b_instruct) &#9989; | GLM-Image | Paraformer-large | BGE-M3 | [vggt](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/vggt) / [vggt-omega](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/vggt_omega) &#9989; | Octo |
| [GLM-OCR](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/glm_ocr) &#9989; | HiDream-I1 | WeNet | gte-Qwen2-7B-instruct | D-FINE | OpenVLA-OFT |
| [Qwen3Guard-Gen-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3guard_gen_0_6b) &#9989; | CogView4 | F5-TTS | Nemotron-3-Embed-8B | RT-DETR | HPT |
| [bert-base-chinese](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/bert_base_chinese) &#9989; | Sana | IndexTTS | stella | DINOv3 |  |
| GLM-4.6 | Kolors | Spark-TTS | nomic-embed | SigLIP-2 |  |
| DeepSeek-R1-Distill (1.5B/7B/14B) | Wan2.2 (A14B) | MaskGCT | BGE-VL | EVA-02 |  |
| Qwen3-Coder-30B-A3B | HunyuanVideo | GPT-SoVITS | jina-clip-v2 | StreamPETR |  |
| Qwen3.5-9B | LTX-Video |  | bge-reranker-v2.5-gemma2 | BEVFormer-v2 |  |
| Qwen3-14B | CogVideoX-5B |  | BCE-reranker | Far3D |  |
| InternVL3 (8B/14B) | Stable-Video-Diffusion |  | GoLLIE | Sparse4D-v3 |  |
| GLM-4.6V | DynamiCrafter |  |  | Metric3D |  |
| GOT-OCR2.0 |  |  |  | DUSt3R / MASt3R |  |
| ModernBERT-large |  |  |  |  |  |

### Supported operators for cloud-side inference

| Operator | Hardware Backend | Description |
|---|---|---|
| ChunkGatedDeltaRule ([Atlas 300I Duo](mindspore-lite/tools/custom_kernels/ascend_ops/src/ascend_300iduo/chunk_gated_delta_rule/README.md) / [Atlas 800I A2](mindspore-lite/tools/custom_kernels/ascend_ops/src/ascend_a2/chunk_gated_delta_rule/README.md)) | Atlas 300I Duo / Atlas 800I A2 | Gated Delta Rule (linear attention / linear RNN) forward operator, chunkwise with a fixed `chunkSize=64` |
| [InnerPromptFlashAttention](mindspore-lite/tools/custom_kernels/ascend_ops/src/ascend_300iduo/inner_prompt_flash_attention/README.md) | Atlas 300I Duo | Prompt Flash Attention forward fusion operator (FP16) for prefill, with `S1 != S2` and GQA/MQA support |
| [RecurrentGatedDeltaRule](mindspore-lite/tools/custom_kernels/ascend_ops/src/ascend_300iduo/recurrent_gated_delta_rule/README.md) | Atlas 300I Duo | Gated Delta Rule token-by-token recurrent forward operator for decode / multi-token prediction (MTP) |
| [QuantMatmulW4a8](mindspore-lite/tools/custom_kernels/ascend_ops/src/ascend_a2/quant_matmul_w4a8/README.md) | Atlas 800I A2 | W4A8 (INT4 weight / INT8 activation) quantized matrix multiplication operator for LLM Linear-layer acceleration |

### API and documentation

1. API documentation:

    - [C++ API documentation](https://www.mindspore.cn/lite/api/en/master/api_cpp/mindspore.html)

    - [Java API documentation](https://www.mindspore.cn/lite/api/en/master/api_java/class_list.html)

    - [Python API documentation](https://www.mindspore.cn/lite/api/en/master/mindspore_lite.html)

    - [HarmonyOS API Document](https://developer.huawei.com/consumer/en/doc/harmonyos-references/development-intro-api)

2. [MindSpore Lite Official Website Document](https://www.mindspore.cn/lite/docs/en/master/index.html)

### Key characteristic capability

- [Support Ascend hardware inference](https://www.mindspore.cn/lite/cloud_docs/en/master/mindir/runtime_python.html)

- [Supporting HarmonyOS](https://developer.huawei.com/consumer/cn/sdk/mindspore-lite-kit)

- [Quantification after Training](https://www.mindspore.cn/lite/docs/en/master/advanced/quantization.html)

- [Lightweight Micro inference deployment](https://www.mindspore.cn/lite/docs/en/master/advanced/micro.html#%20Model%20inference%20code%20generation)

- [Benchmark Debugging Tool](https://www.mindspore.cn/lite/docs/en/master/tools/benchmark.html)

> Additionally, MindSpore Lite cloud-side inference provides a standalone acceleration component, Lite Boost, to improve inference performance for cloud-side inference based on the PyTorch interface. For details, see the [Lite Boost](mindspore-lite/lite_boost/README.md) section.

## Communication and Feedback

- Welcome to [AtomGit Issues](https://atomgit.com/mindspore/mindspore-lite/issues): submit questions, reports, and suggestions;

- Welcome to [Community Forum](https://discuss.mindspore.cn/c/mindspore-lite/38): engage in technical and problem-solving exchanges;

- Welcome to [Sig](https://www.mindspore.cn/sig/MindSpore%20Lite): to manage and improve workflow, participate in discussions and exchanges;

## Surrounding communities

- [MindSpore](https://atomgit.com/mindspore/mindspore)

- [MindOne](https://atomgit.com/mindspore/mindone)

- [Mindyolo](https://atomgit.com/mindspore/mindyolo)

- [OpenHarmony](https://atomgit.com/openharmony/third_party_mindspore)

- [GraphEngine](https://gitcode.com/cann/ge)
