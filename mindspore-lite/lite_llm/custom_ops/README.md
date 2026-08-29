# Kirin AI 融合算子库（custom_ops）

面向 Kirin AI NPU 处理器的 **AscendC 自定义算子库**，附带完整的开发、构建与验证基础设施。

仓库承担两个职责：

1. **DDK 环境宿主** —— 解包后的 DDK 工具链与平台插件位于 `ddk_env/`。注意，如果用户环境提供了默认DDK_PATH环境，和用户确认是否直接复用！

## 算子能力总览

> ✅ = 已真机验证（Kirin 9020 实机，BinRunner）· ☐ = 尚未支持/验证（规划中）

| 序号 | 自定义算子 | 支持芯片 | 支撑模型 | 精度误差 | 性能优化情况 | 约束与限制备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `MsRmsNorm`<br>（FP16 RMSNorm，可选 gamma） | ✅ kirin9020<br>☐ kirin9030<br>☐ kirinx90 | ✅ Qwen2.5-0.5B<br>☐ Qwen3-0.6B | max_abs_diff 1.95e-3（atol 1e-2 / rtol 1e-3） | 物理核感知 + UB 自适应多行批处理 | K 为 16 倍数且 K≤8192 |

> **性能指标说明**（Kirin 9020 实机，表内时间均为 median）：`wall` 是关闭
> profiling 后一次同步模型调用的端到端耗时，包含 NNRT 下发、调度、Kernel 执行和
> 同步；`NPU` 是独立 profiling 运行中采集的算子执行时间，更接近 Kernel 本体。
> 二者来自独立运行，不能逐样本相减，长耗时 case 的 NPU median 也可能因批次波动
> 略高于 wall median。九算子标准 wall 协议为 `10` 次 warmup、`50×3` 次正式迭代，
> 共包含 5,400 个 wall 样本和 350 个 NPU 样本，详见
> [九算子标准基线](docs/performance/baselines/2026-08-17-36a9ddf-kirin9020/README.md)。
> RmsNorm、AddSoftmax 和当前 GroupMatmul 的 before/after 均采用各自相同的标准协议，
> 可以直接计算同算子提升；GroupMatmul 的最新对比以 upstream `d257165` 为 before，
> 详细协议和同设备状态复核见其性能报告。
> Qwen2.5-0.5B 已验证配置：hidden 896 / intermediate 4864 / heads 14 / kv 2 / head_dim 64（decode S=1，prefill S∈{64,128}，KV cache 1024）。
>

## 目录结构

```bash

├── ascendc_ops/              # 算子目录集合（当前仅 MsRmsNorm）
├── torch_custom/             # torch 对接层：每算子一个 ms_<op>.py（eager 参考实现 + ONNX symbolic，供 lite_llm export 使用）
├── workspace/                # 只读构建模板（cmake/、CMakeLists、presets）
├── templates/operator/       # 算子创作模板（经 scaffold 实例化）
├── scripts/                  # 工程脚手架：build_operator.sh、scaffold.py、lima/
├── tools/model_run_tool/     # 设备端通用 OMC runner（C++）
├── tests/                    # 统一 pytest 入口（host + device 两阶段）
└── build.py                  # 算子构建/安装入口（--ops ... --install）

```

## 环境准备

```bash

# 激活 Python venv 与 DDK 环境（任何 DDK 操作前都必须执行）
source .venv/bin/activate
source ./ddk_env/tools/tools_ascendc/set_ascendc_env.sh

```text

`set_ascendc_env.sh` 为官方激活脚本，设置 `DDK_PATH`、`TOOLCHAIN_HOME`、`HIAI_VERSION` 等关键变量。

- **首次安装 DDK**：将 DDK 包放入 `package/`，执行 `./.claude/skills/setting-up-kirin-ai-ddk/install_ddk.sh`
- **Ubuntu 22.04 注意**：omg/ccec 需要本地 `libtinfo.so.5` shim（见 setting-up-kirin-ai-ddk 技能的 Known Local Pitfall）
- **Apple Silicon**：x86_64 工具链在 Lima VM 中运行，见 `scripts/lima/README.md`

## 构建算子

```bash

./build.py                         # 聚合打包 ascendc_ops/ 下全部算子
./build.py --ops MsRmsNorm         # 只打包指定算子（推荐写法）
./build.py MsRmsNorm               # 兼容的单算子简写
./build.py --ops <op> --install "$DDK_PATH"  # 打包后安装到 DDK

```text

根目录只保留 `build.py` 这个公开构建入口。它把所选算子的 host、kernel、framework
及二进制实现配置汇总到只读 `workspace/` 模板的副本 `build/ms_ops_pack/`，再调用其中的
内部构建脚本生成一个聚合 `.run` 包。最终发布物落入 `output/ms_ops_pack/`；根目录不再
保留 Python 套壳形式的 `build.sh`。无参数或 `--all` 只选择 `ascendc_ops/` 下的正式
算子。清理正式构建树前会先在临时目录完成逐算子及聚合预检，以汇总缺文件、配置错误和跨算子冲突。

## 测试

构建与测试是**两个显式阶段**（ADR-0005）：

```bash

# Host 测试（无需设备）
python -m pytest tests/ -m "not device"

# 单算子设备测试（需 REMOTE_TARGET、MODEL_RUN_TOOLS_PATH）
python -m pytest tests/ -m device -k rms_norm

# BinRunner 沙箱传输（部分算子如 MsGroupMatmul 的 UT 强制要求 binapp）
# 注意：TCP 方式连接设备（REMOTE_TARGET=IP:port）时，hdc 目标选项必须是 -t；
#      默认 -s（serial）会让所有 hdc 命令（含文件拉取）报 Connect failed，
#      且错误被 capture_output 吞掉，表现为“设备端 SUCCESS 但结果文件缺失”。
DEVICE_TRANSPORT=binapp HDC_TARGET_OPTION=-t \
  python -m pytest tests/ -m device -k group_matmul

```text

设备用例的完整环境搭建、运行方法、用例闭环和已知限制（kernel 缓存、xdist 并行、耗时构成）
见 [`run-ascendc-op-device-ut`](.claude/skills/run-ascendc-op-device-ut/SKILL.md) skill。

优化前/后的标准化性能采集使用 `tests/perf/operator_baseline.py`；协议、数据格式和
比较方法见 [`docs/performance/operator-baseline.md`](docs/performance/operator-baseline.md)。
后续性能采集、回归判断和台账更新统一遵循项目内
[`benchmark-ascendc-operators`](.claude/skills/benchmark-ascendc-operators/SKILL.md) skill；
瓶颈定位、优化方法选择、受控实验和成功/失败尝试记录遵循
[`optimize-ascendc-operators`](.claude/skills/optimize-ascendc-operators/SKILL.md) skill。

## 算子开发

### 命名约定（强制）

- 每个算子自带 **ONNX 框架插件**（`framework/onnx_plugin/onnx_ms_<op>_plugin.cc`），OMG 侧同名直通
- 每个算子在 `torch_custom/` 有对应模块 `ms_<op>.py`（`torch.autograd.Function`：eager 参考实现 + ONNX symbolic，导出 `custom::Ms<Op>` 节点）

### 用 scaffold 创建新算子（自动满足全部约定）

```bash

python scripts/scaffold.py new MsMyOp   # 生成 ascendc_ops/MsMyOp/ + torch_custom/ms_my_op.py + 注册 __init__
python scripts/scaffold.py check        # 校验全部算子目录合规
python scripts/scaffold.py check --strict  # 额外真构建模板，防模板与 workspace 漂移

```

算子目录只允许携带**差异内容**：`operator.json`、`op_host/`、`op_kernel/` 必选；`DESIGN.md`、`framework/onnx_plugin/`、`onnx/`、`scripts/`、`gen_data.py`、`temp.json` 可选。构建基础设施（CMakeLists、cmake/、presets）属于 `workspace/`，禁止放入算子目录。

## torch_custom 对接层

所有算子统一走 **torch → ONNX 整图导出** 链路（不引入其他框架）：

- `torch_custom/ms_<op>.py` 只定义叶子 `Function`（eager + symbolic）；wrapper 模型与整图 `torch.export()` 由消费者（每算子的 `onnx/` 脚本、pytest 用例）自行构建
- 两种导入方式：`from torch_custom import MsRmsNorm` 或 `from torch_custom.ms_rms_norm import MsRmsNorm`

## 关键文档

- **术语表**：`docs/glossary.md`（目录语义的规范定义）
- **架构决策**：`docs/adr/`（ADR-0001 ~ ADR-0012，仓库形态的"为什么"）
- **NPU 架构**：`docs/reference/cce_reference/architecture.md`（GM/L1/L0/UB 存储层次、MTE 引擎、数据流）
- **构建管线**：`docs/development/build_process_analysis.md`
- **AscendC / CCE 参考**：`docs/reference/`
- **算子约定**：`tests/README.md`
