# InternVL3.5-8B ONNX 模型导出与 MindSpore Lite 推理部署教程

本教程介绍如何将 [OpenGVLab/InternVL3_5-8B](https://www.modelscope.cn/models/OpenGVLab/InternVL3_5-8B) 视觉语言模型按结构拆分导出为 ONNX，转换为 MindSpore Lite MindIR，并在 **Ascend Atlas 300I Duo** 上完成图文推理与精度对齐。

InternVL3.5 = InternViT 视觉编码器 + pixel-shuffle + MLP 投影 + 自回归 LLM（Qwen3 系列）。固定 shape 部署拆分为三子模型（与 qwen2.5_vl 示例同一模式）：

| 子模型 | 输入 | 输出 |
| --- | --- | --- |
| `internvl_vision`（InternViT+mlp1） | pixel_values[1,3,448,448] | image_embeds[1,256,hidden] |
| `internvl_llm_prefill` | inputs_embeds[1,seq,hidden], position_ids[1,seq] | logits + present KV |
| `internvl_llm_decode` | inputs_embeds[1,1,hidden], position_ids[1,1], cache_position[1], past_keys/past_values | logits + new KV |

> 维度（hidden/num_layers/num_heads/head_dim）从 checkpoint config 读取，**同一套脚本适用于 2B/4B/8B 及 Flash 变体**，仅模型 id 与固定 shape 不同。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python | 3.11 |
| torch | 2.9.0+cpu |
| onnx / onnxruntime | 1.19.1 / 1.24.2 |
| transformers | 5.9.0（须 trust_remote_code） |
| mindspore-lite | 2.10.0 |
| CANN | 8.5.0 |

```bash
source /home/yf/env.sh
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 transformers mindspore-lite modelscope
```

## 2. 模型下载

```bash
python -c "from modelscope import snapshot_download as s; print(s('OpenGVLab/InternVL3_5-8B', cache_dir='/home/yf/modelscope_cache'))"
ln -sfn /home/yf/modelscope_cache/OpenGVLab/InternVL3_5-8B ./InternVL3_5-8B
```

## 3. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/internvl3_5_8b
python export_internvl3_5_8b_onnx.py \
  --model-id OpenGVLab/InternVL3_5-8B \
  --output-dir ./internvl3_5_8b_onnx \
  --image-size 448 --num-img-tokens 256 --max-text-len 64 --max-total-len 1024 \
  --dtype float32
```

LLM 注意力导出为 CANN `PromptFlashAttention` Custom 算子（monkeypatch `F.scaled_dot_product_attention`，BNSD，cheap 保形 forward）；legacy exporter、opset 17、float32、`do_constant_folding=False`。导出脚本会打印真实 hidden/num_layers，用于核对 config。

## 4. MindSpore Lite 转换

```bash
CONV=/home/yf/Target/mindspore-lite-2.9.0-linux-aarch64/tools/converter/converter/converter_lite
$CONV --fmk=ONNX --modelFile=./internvl3_5_8b_onnx/internvl_vision.onnx --outputFile=./internvl3_5_8b_onnx/internvl_vision --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_vision.config
$CONV --fmk=ONNX --modelFile=./internvl3_5_8b_onnx/internvl_llm_prefill.onnx --outputFile=./internvl3_5_8b_onnx/internvl_llm_prefill --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_llm_prefill.config
$CONV --fmk=ONNX --modelFile=./internvl3_5_8b_onnx/internvl_llm_decode.onnx --outputFile=./internvl3_5_8b_onnx/internvl_llm_decode --optimize=ascend_oriented --saveType=MINDIR --configFile=./configs/internvl_llm_decode.config
```

> prefill/decode 的 `input_shape`（hidden、num_layers、num_kv_heads、head_dim、max_total_len）须与导出时打印的真实值一致；如不一致请按实际值重写 config。

```log
CONVERT RESULT SUCCESS:0   （待运行后填入完整日志）
```

## 5. MindSpore Lite 推理

```bash
python infer_internvl3_5_8b_mslite.py \
  --mindir-dir ./internvl3_5_8b_onnx --model-dir ./InternVL3_5-8B \
  --image ./test.jpg --prompt "Describe this image in detail." --max-new-tokens 128
```

```log
（待运行后填入：输出文本 + vision/prefill/decode/端到端 耗时）
```

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| Vision 编码 (InternViT, dev1) | _待运行填入_ |
| LLM prefill | _待运行填入_ |
| LLM decode（总 / 平均步） | _待运行填入_ |
| **端到端** | **_待运行填入_** |
| **吞吐** | **_待运行填入_ tok/s** |

## 7. 精度对齐

```bash
python align_internvl3_5_8b.py --mindir-dir ./internvl3_5_8b_onnx --model-dir ./InternVL3_5-8B --image ./test.jpg --max-new-tokens 32
```

对比 HF InternVL3.5（CPU greedy generate）与 MSLite 的回答（exact match + token overlap）。

## 8. 常见问题

1. **InternVL 加载报错** —— 须 `trust_remote_code=True`；transformers 版本须支持 InternVL3.5。
2. **prefill/decode config shape 不匹配** —— hidden/num_layers 等须与导出时打印的真实值一致，按实际 checkpoint 重写 config。
3. **视觉 token 数** —— 取决于 image_size 与 pixel-shuffle 比例；`--num-img-tokens` 须与导出一致。
4. **8B 变体内存** —— fp16 约 16GB，单 44GB 芯片可容纳；如需更大 max_total-len 可减小或分芯。

## 9. 参考资源与许可证

- 上游：<https://github.com/OpenGVLab/InternVL>、ModelScope `OpenGVLab/InternVL3_5-8B`
- MindSpore Lite：<https://www.mindspore.cn/lite>
- 脚本遵循 MindSpore Lite 仓库许可证；上游模型/代码许可证以其仓库为准。
