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

| Image/Video Generation Models | Vision-Language Models (VLM) | Large Language Models (LLM) | Audio Models (ASR/TTS) | Information Retrieval / Embeddings / CNN / Others |
| :---------------------------: | :--------------------------: | :-------------------------: | :-------------------: | :----------------------------------------------: |
|   Kandinsky-5.0-I2V-Lite-5s   | Qwen3-VL-4B-Thinking         | Qwen3.5-9B                  | Qwen3-ASR-1.7B        | [Qwen3-Reranker-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_reranker_0.6b) &#9989; |
| Kand0-T2V0-T2V-Lite-sft-10s   | Qwen3-VL-4B-Instruct         | Qwen3.5-4B                  | Qwen3-ASR-0.6B        | [Qwen3-VL-Embedding-2B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_embedding_2b) &#9989; |
|    Kandinsky-5.0-T2I-Lite     | Qwen3-VL-2B-Thinking         | Qwen3.5-2B                  | Qwen3-TTS-12Hz-1.7B-Base | [jina_reranker_v3](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/jina_reranker_v3) &#9989; |
|    Kandinsky-5.0-I2I-Lite     | [Qwen3-VL-2B-Instruct](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_vl_2b_instruct) &#9989; | Qwen3.5-0.8B                | Qwen3-TTS-12Hz-1.7B-CustomVoice | YoLo                                     |
|        Wan2.1-T2V-1.3B        | Qwen2.5-VL-3B-Instruct       | Qwen3-8B                    | Qwen3-TTS-12Hz-1.7B-VoiceDesign | ViT                                      |
|        Wan2.1-T2V-1.3B        | Qwen2.5-VL-3B-Instruct       | Qwen3-8B                    | Qwen3-TTS-12Hz-1.7B-VoiceDesign | [ViT-Base-Patch16-224](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/vit_base_patch16_224) &#9989; |
|        Wan2.1-T2V-14B         | Qwen2-VL-2B-Instruct         | Qwen3-4B                    | Qwen3-TTS-12Hz-0.6B-Base |                                          |
|      Wan2.1-I2V-14B-480P      | Qwen2-VL-2B                  | Qwen3-1.7B                  | Qwen3-TTS-12Hz-0.6B-CustomVoice |                                       |
|        Wan2.2-TI2V-5B         | InternVL3_5-4B-Flash         | [Qwen3-0.6B](https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/qwen3_0.6b) &#9989; | CosyVoice3-0.5B       |                                          |
|        Wan2.2-T2V-A14B        | InternVL3_5-2B-Flash         | Qwen2.5-7B                  | CosyVoice2-0.5B       |                                          |
|        Wan2.2-I2V-A14B        | InternVL3_5-1B-Flash         | Qwen2.5-3B                  | FireRedASR            |                                          |
|      Wan2.2-Animate-14B       | InternVL3-2B                 | Qwen2.5-1.5B                | WeNet                 |                                          |
|        Qwen-Image-Edit        | InternVL3-1B                 | Qwen2.5-0.5B                |                       |                                          |
|          Qwen-Image           | llava-v1.6                   | Qwen2-7B                    |                       |                                          |
|          FLUX.1-dev           | LLaVa                        | Qwen2-1.5B                  |                       |                                          |
|     stable-diffusion-v1-5     | BLIP                        | Qwen2-0.5B                  |                       |                                          |
|     stable-diffusion-2-1      | BLIP-2                      |                             |                       |                                          |
| stable-diffusion-xl-base-1.0  | CLIP                        |                             |                       |                                          |

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

## Communication and Feedback

- Welcome to [AtomGit Issues](https://atomgit.com/mindspore/mindspore-lite/issues): submit questions, reports, and suggestions;

- Welcome to [Community Forum](https://discuss.mindspore.cn/c/mindspore-lite/38): engage in technical and problem-solving exchanges;

- Welcome to [Sig](https://www.mindspore.cn/sig/MindSpore%20Lite): to manage and improve workflow, participate in discussions and exchanges;

## Surrounding communities

- [MindSpore](https://atomgit.com/mindspore/mindspore)

- [MindOne](https://github.com/mindspore-lab/mindone)

- [Mindyolo](https://github.com/mindspore-lab/mindyolo)

- [OpenHarmony](https://atomgit.com/openharmony/third_party_mindspore)
