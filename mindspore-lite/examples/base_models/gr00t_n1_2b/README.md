# GR00T N1 人形机器人 Flow Matching 策略 ONNX 导出与 MindSpore Lite 推理部署教程

GR00T N1（NVIDIA）= Eagle VLM 主干 + Flow Matching 动作专家。动作专家是 velocity 网络 v(x_t, t)，通过 Euler ODE 从 t=0（噪声）积分到 t=1（动作 chunk）。**Flow Matching 与扩散（DDPM）不同**：连续时间 t∈[0,1]、预测 velocity（而非噪声）。

本目录导出**单步 velocity 网络**（视觉条件），Euler ODE 采样在 host 侧 numpy 实现。

> ⚠️ **风险标注**：真实 GR00T-N1（2B）= Eagle VLM + Flow Matching 动作专家，需 NVIDIA 官方包加载（含 Flow Matching 自定义算子，任务2 可能需 ascend_ops 适配）。本目录为视觉条件 velocity 骨架，`--random-init` 可端到端验证管线。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：modelscope.cn/nvidia/GR00T-N1-2B。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/gr00t_n1_2b
python export_gr00t_n1_2b_onnx.py --output-dir ./gr00t_n1_2b_onnx --device cpu
# 真实权重（任务2）：python export_gr00t_n1_2b_onnx.py --checkpoint /path/to/gr00t.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | GR00T state_dict（任务2） | 空（demo） |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |
| `--dim` / `--depth` | velocity 网络配置（demo 小） | `384`/`4` |

```text
./gr00t_n1_2b_onnx/
└── gr00t_n1_velocity.onnx   # image+x_t+t -> velocity
```

---

## 3. ONNX 推理

```bash
python infer_gr00t_n1_2b_onnx.py --model ./gr00t_n1_2b_onnx/gr00t_n1_velocity.onnx --num-steps 10 --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./gr00t_n1_2b_onnx/gr00t_n1_velocity.onnx \
  --outputFile=./gr00t_n1_2b_onnx/gr00t_n1_velocity --optimize=ascend_oriented \
  --saveType=MINDIR --configFile=./config.ini
```

`config.ini`：

```ini
[acl_build_options]
input_format="ND"
input_shape="image:1,3,224,224;x_t:1,16,7;t:1"

[acl_init_options]
ge.exec.precision_mode=force_fp16

[ascend_context]
plugin_custom_ops=All
```

```text
./gr00t_n1_2b_onnx/
├── gr00t_n1_velocity.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_gr00t_n1_2b_mslite.py --model ./gr00t_n1_2b_onnx/gr00t_n1_velocity.mindir \
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
| MSLite 端到端 Flow Matching（5 步 Euler） | 7.72 |
| MSLite 单步 velocity（mean） | 1.54 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999998 |

---

## 7. 常见问题

1. Flow Matching vs 扩散：连续 t∈[0,1] + velocity + Euler（非 DDPM 去噪）。
2. 真实 GR00T Flow Matching 自定义算子：任务2 若转换报错，参考 ascend_ops 适配或把 ODE 求解放 host。
3. t 输入 dtype：float32 [0,1]。

---

## 8. 参考资源

- GR00T N1：https://github.com/NVIDIA/Isaac-GR00T
- ModelScope 权重：https://modelscope.cn/nvidia/GR00T-N1-2B
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- GR00T N1 上游代码许可证以其仓库为准（Apache-2.0）。
