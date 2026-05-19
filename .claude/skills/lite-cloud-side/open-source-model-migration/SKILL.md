---
name: open-source-model-migration
description: 基于MindSpore Lite云侧推理实现开源模型从PyTorch→ONNX→MindIR→MindSpore Lite的端到端导出/验证/部署/性能评测。用户要求模型拆分导出、精度对齐、MindIR转换或部署工具链时调用。
---

# 开源模型迁移

本技能用于把“开源算法模型”做成可落地的端到端部署方案：按网络结构拆分导出 ONNX、生成模型元信息、ONNX Runtime CPU 推理脚本（CLI+API）、精度对齐验证（含报告）、ONNX→MindIR 转换与 MindSpore Lite Ascend 推理实现，并交付一键化工具链与测试/基准/容器化方案。

## 适用范围（何时调用）

当用户提出以下诉求之一时调用：

- “按算法结构/模块拆分导出 ONNX”
- “提供 ONNX Runtime 部署脚本（CPU）/动态 shape 支持/性能统计”
- “做 PyTorch vs ONNX 精度对齐验证并出报告”
- “ONNX 转 MindIR，MindSpore Lite 云侧Ascend后端部署代码”
- “提供一键式工具链、单测、benchmark 报告”

## 总体交付物清单（建议目录结构）

在目标模型目录（例如 `mindspore-lite/examples/base_models/<model_name>/`）下建议形成如下结构（按需裁剪）：

- `export_<model>_onnx.py`：按模块导出 ONNX
- `infer_<model>_onnx.py`：ONNX Runtime CPU 推理（CLI + 可 import 的 API）
- `infer_<model>_mslite.py`：MindSpore Lite Ascend 推理（CLI + 可 import 的 API）
- `README.md`：模型迁移说明文档，需要包含一下几个模块：

  - 环境准备
  - ONNX模型导出
  - ONNX模型推理
  - MindSpore Lite模型转换
  - MindSpore Lite Ascend 推理
  - 性能数据
  - 参考文档
  - 许可协议

## 阶段 1：模型结构分析与模块拆分

目标：识别可独立部署的模块组件，并定义模块间接口（张量语义、dtype、shape）。

建议拆分思路：

- **Encoder / Backbone / Head**：如分类、检测、reranker
- **Text Encoder / UNet(or DiT) / VAE**：扩散类 T2I/T2V/I2I
- **Vision Encoder / LLM Prefill / LLM Decode**：多模态自回归类（KV cache 拆分）
- **Encoder / Decoder**：ASR、Seq2Seq

产出：

- 模块列表与每个模块的 I/O 规格（含 dtype、dynamic axis）
- 固定/可变维度（batch、seq_len、H/W、frames 等）

## 阶段 2：按模块导出 ONNX（torch.onnx.export）

要求：

- 每个 ONNX 文件包含完整计算图与权重（不要外部权重依赖，除非模型过大且部署链路接受 `.onnx.data`）
- 显式指定 `opset_version`（优先 17/18，结合仓库与转换器能力）
- `dynamic_axes` 覆盖 batch 与核心动态维度
- 推理模式：`eval()` + `torch.no_grad()`

模板要点：

- 为每个模块写 wrapper `torch.nn.Module`，只暴露必要输入输出
- 统一输入名/输出名，便于后续部署脚本对齐
- 导出后可选做 ONNX 简单清理（例如移除无用节点、常量折叠），但避免引入不在仓库依赖里的额外工具

## 阶段 3：ONNX Runtime CPU 部署脚本（CLI + API）

要求：

- 动态 batch / 动态输入尺寸（只要模型声明支持）
- 完整 pipeline：加载→预处理→推理→后处理
- 性能监控：延迟、吞吐、内存占用
- 两种调用方式：
  - CLI：`python infer_xxx_onnx.py --model ... --input ...`
  - API：`from infer_xxx_onnx import Inferencer; Inferencer(...).infer(...)`

性能采集建议：

- 延迟：`time.perf_counter()`，区分 warmup 与 measurement
- 内存：
  - Linux：读取 `/proc/self/status` 里的 `VmRSS/VmHWM`
  - 或 `psutil`（仅当仓库已有该依赖；否则不要强依赖）

## 阶段 4：精度对齐验证（PyTorch vs ONNX）

要求（来自用户约束）：

- 统一测试数据集：不少于 1000 样本
- 对比指标：
  - 输出余弦相似度（cosine similarity）
  - 最大绝对误差（max abs error）
- 通过条件：
  - 所有输出张量误差 < 1e-5
  - 关键业务指标差异 < 0.1%
- 报告：
  - 汇总统计（均值/分位数）
  - 逐层输出对比（如可抓取中间层）

实现建议（不引入额外训练/标注成本）：

- 默认提供一个“可复现实验数据生成器”：随机但固定 seed，覆盖典型 shape 组合
- 若用户提供真实数据集路径，则优先用真实样本
- 逐层对比：为 PyTorch 增加 forward hook；ONNX 侧可通过额外导出“debug 模块”或在导出 wrapper 中暴露中间张量（按需）

## 阶段 5：ONNX → MindIR 转换

要求：

- 使用 MindSpore Lite 的转换工具（项目规则：构建/产物由 `build.sh` 产生，不要直接跑 CMake）
- 记录转换命令、参数与失败日志
- 兼容性问题：
  - 优先通过导出侧改图（替换不支持算子、冻结 shape、简化控制流）
  - 必要时才考虑自定义算子（需要用户确认目标设备与算子注册方式）

常用转换命令模式（以 `converter_lite` , 昇腾推理后端为例）：

```bash
converter_lite --fmk=ONNX --optimize=ascend_oriented --modelFile=xxx.onnx --outputFile=xxx --saveType=MINDIR
```

## 阶段 6：MindSpore Lite 云侧Ascend推理实现

目标：功能与 ONNX 版本保持一致。

要求：

- 支持同样的输入形式与动态维度（在模型支持范围内）
- 对齐预处理与后处理（避免“模型正确但 pipeline 不一致”造成的精度差异）
- 错误处理：
  - 模型加载失败、输入 shape 不合法、设备初始化失败等
  - 明确错误信息与返回码（Python 脚本可抛异常并打印定位信息）

## 阶段 7：一键化工具链、单测、基准

说明：是否生成这些交付物取决于用户是否明确要求，避免无必要新增文件。

建议：

- 一键脚本：串联 `export → ort_infer_sanity → accuracy_eval → convert → mslite_infer_sanity → benchmark`
- 单元测试：最少覆盖关键路径与异常路径
- 基准报告：输出 CSV/JSON（便于 CI 或可视化）

## 阶段 8：教程编写

教程中必须包含的模块，要求：

- 环境准备：提供详细的环境要求与安装步骤
- ONNX模型导出：提供脚本执行命令以及参数说明
- ONNX模型推理：提供脚本执行命令以及参数说明
- MindSpore Lite模型转换：提供详细的转换命令与参数说明
- MindSpore Lite Ascend 推理：提供脚本执行命令以及参数说明
- 性能数据：提供详细的性能指标（如延迟、吞吐、内存占用等）
- 参考文档：MindSpore Lite 官方文档以及对应的开源模型文档
- 许可协议：MindSpore Lite 项目的开源协议（如 Apache 2.0）

教程注意事项：
- 提供故障排除指南（如模型加载失败、输入 shape 不合法等）
- 考虑用户可能遇到的问题与解决方法
- 使用中文编写README.md文件，且教程中的路径使用相对路径，不要使用绝对路径

## 执行检查清单（交付前自检）

- ONNX 能被 onnxruntime 正常加载并推理
- 动态 shape 在至少 2 组不同 shape 下通过
- 精度报告满足阈值
- ONNX → MindIR 转换成功
- MindSpore Lite 推理输出与 ORT 在阈值内一致
- 脚本提供 CLI + 可 import API
- 关键路径有测试覆盖
