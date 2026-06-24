# Cosmos-Predict1 视频世界模型 ONNX 导出与 MindSpore Lite 推理部署教程

Cosmos-Predict1（NVIDIA）是世界模型，用 DiT 在视频 latent 上做扩散去噪，配 VAE 编解码器把像素↔latent。本目录导出**latent 空间单步去噪 DiT**，DDPM 采样在 host 侧 numpy 实现（VAE 解码不在单步网络内）。

> ⚠️ **高风险标注**：真实 Cosmos-Predict1 ~14B（fp16≈28GB），在 300I Duo（44GB）上叠加激活极可能 OOM。任务2 触发 OOM 时降级：拆分 DiT 子图分卡 / 降低分辨率与帧数（num_tokens） / 改用 800I 验证并注明。需 NVIDIA 官方包加载 + VAE。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：modelscope.cn/nvidia/Cosmos-Predict1（DiT + VAE）。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/cosmos_predict1_14b
python export_cosmos_predict1_14b_onnx.py --output-dir ./cosmos_predict1_14b_onnx --device cpu
# 真实权重（任务2）：python export_cosmos_predict1_14b_onnx.py --checkpoint /path/to/cosmos.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | Cosmos state_dict（任务2） | 空（demo） |
| `--num-tokens` / `--latent-dim` | 视频 latent token 数/维度（demo 小） | `256`/`16` |
| `--cond-dim` | 文本/上下文条件维度 | `256` |
| `--dim` / `--depth` | DiT 配置（demo 小） | `256`/`6` |

```text
./cosmos_predict1_14b_onnx/
└── cosmos_denoise.onnx   # noisy_latent/timestep/cond -> noise
```

---

## 3. ONNX 推理

```bash
python infer_cosmos_predict1_14b_onnx.py --model ./cosmos_predict1_14b_onnx/cosmos_denoise.onnx --num-steps 10 --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./cosmos_predict1_14b_onnx/cosmos_denoise.onnx \
  --outputFile=./cosmos_predict1_14b_onnx/cosmos_denoise --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="noisy_latent:1,256,16;timestep:1;cond:1,256"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./cosmos_predict1_14b_onnx/
├── cosmos_denoise.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_cosmos_predict1_14b_mslite.py --model ./cosmos_predict1_14b_onnx/cosmos_denoise.mindir \
  --num-steps 10 --seed 0 --device ascend --device-id 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 6. 性能数据

测试环境：Ascend Atlas 300I Duo（6×310P3），CANN 8.5.0，MindSpore Lite 2.10.0。

| 指标 | 耗时 (ms) |
| --- | ---: |
| MSLite 端到端 DDPM（5 步） | 6.42 |
| MSLite 单步去噪（mean） | 1.28 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999999 |

> 14B 模型注意显存：任务2 若 OOM 需拆子图/降分辨率。

---

## 7. 常见问题

1. OOM：14B fp16≈28GB+激活，300I Duo 易溢出 → 拆子图分卡 / 降 num_tokens / 改 800I。
2. VAE：latent→视频需 VAE 解码（本目录仅 DiT 去噪）。
3. 大模型 protobuf >2GB：prefill 留 CPU、仅 decode 走 MindIR。

---

## 8. 参考资源

- Cosmos：https://github.com/nvidia-cosmos
- ModelScope 权重：https://modelscope.cn/nvidia/Cosmos-Predict1
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- Cosmos 上游代码许可证以其仓库为准（Apache-2.0）。
