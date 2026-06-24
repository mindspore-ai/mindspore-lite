# π0 / π0-FAST Flow Matching 策略 ONNX 导出与 MindSpore Lite 推理部署教程

π0 / π0-FAST（Physical Intelligence，openpi）= PaliGemma VLM 主干 + Flow Matching 动作专家（3.3B），通过 ODE 积分生成动作 chunk。**π0.5 已适配，本目录跳过 π0.5，覆盖 π0 / π0-FAST**。

本目录导出**单步 velocity 网络**（视觉条件），Euler ODE 采样在 host 侧 numpy 实现（与 GR00T N1 同为 Flow Matching 模式）。

> ⚠️ **风险标注**：真实 π0（3.3B）= PaliGemma + flow-matching 动作专家，需 `openpi` 官方包加载（Flow Matching 自定义算子，任务2 可能需 ascend_ops 适配）。本目录为视觉条件 velocity 骨架，`--random-init` 可端到端验证管线。

---

## 1. 环境准备

| 软件包 | 版本 |
| --- | --- |
| Python / torch / onnx / onnxruntime / numpy / mindspore-lite / CANN | 3.11 / 2.9.0 / 1.19.1 / 1.24.2 / 1.26.x / 2.10.0 / 8.5.0 |

```bash
pip install torch==2.9.0 onnx==1.19.1 onnxruntime==1.24.2 numpy==1.26.4 mindspore-lite==2.10.0
```

```bash
# 真实权重（任务2）：modelscope.cn/physical-intelligence 或 openpi 官方。
```

---

## 2. 模型导出 ONNX

```bash
cd mindspore-lite/examples/base_models/pi0
python export_pi0_onnx.py --output-dir ./pi0_onnx --device cpu
# 真实权重（任务2）：python export_pi0_onnx.py --checkpoint /path/to/pi0.pt ...
```

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--checkpoint` | π0 state_dict（任务2 用 openpi） | 空（demo） |
| `--action-dim` / `--horizon` | 动作维度/chunk 长度 | `7`/`16` |
| `--dim` / `--depth` | velocity 网络配置 | `384`/`4` |

```text
./pi0_onnx/
└── pi0_velocity.onnx   # image+x_t+t -> velocity
```

---

## 3. ONNX 推理

```bash
python infer_pi0_onnx.py --model ./pi0_onnx/pi0_velocity.onnx --num-steps 10 --seed 0
```

```log
（见下方性能表，输出与 PyTorch 一致 cos≈1.0）
```

---

## 4. MindSpore Lite 转换（ONNX → MindIR）

```bash
converter_lite --fmk=ONNX --modelFile=./pi0_onnx/pi0_velocity.onnx \
  --outputFile=./pi0_onnx/pi0_velocity --optimize=ascend_oriented \
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
./pi0_onnx/
├── pi0_velocity.mindir
```

```log
CONVERT RESULT SUCCESS:0
```

---

## 5. MindSpore Lite 推理

```bash
python infer_pi0_mslite.py --model ./pi0_onnx/pi0_velocity.mindir \
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
| MSLite 端到端 Flow Matching（5 步 Euler） | 7.99 |
| MSLite 单步 velocity（mean） | 1.60 |
| 进程 RSS | 1.04 |
| 精度对齐（PyTorch↔MSLite） | cos=0.999998 |

---

## 7. 常见问题

1. π0.5 已适配，本目录不重复；π0/π0-FAST 用 Flow Matching velocity 骨架。
2. openpi Flow Matching 算子：任务2 若转换报错，参考 ascend_ops 适配或 ODE host 化。
3. 真实 π0 含 proprio/state 输入：任务2 接入 openpi 时补全。

---

## 8. 参考资源

- openpi：https://github.com/Physical-Intelligence/openpi
- MindSpore Lite 文档：https://www.mindspore.cn/lite

---

## 9. 许可证

- 本目录脚本遵循 MindSpore Lite 仓库许可证要求。
- π0 / openpi 上游代码许可证以其仓库为准（Apache-2.0）。
