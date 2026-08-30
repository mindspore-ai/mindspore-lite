# Operator unit tests

This directory is the common pytest entry for operator host and device tests.
Operator implementation directories should contain kernels, host registration,
model conversion, but not reusable pytest infrastructure.

## Layout

| File | Purpose |
| --- | --- |
| `ms_rms_norm.py` (in `../torch_custom/`) | Torch adapter layer: `MsRmsNorm` eager + ONNX symbolic (one module per `Ms<Op>`) |
| `precision.py` | fp16/fp32 element-wise precision comparison (from asc_bench dissolution) |
| `base_test.py` | Standard HDC lifecycle, transfer, cleanup, and profiling |
| `binrunner_test.py` | BinRunner application-sandbox transport + app restart robustness |
| `ut/` | Per-operator device test cases: `test_<op>_op.py` |
| `perf/operator_baseline.py` | Precision-gated wall-clock + NPU profiling baseline collector |

## Run

```bash
python -m pip install -r tests/requirements-test.txt
python -m pytest -s -q -m "not device" tests/
```

The device test requires `REMOTE_TARGET` and `MODEL_RUN_TOOLS_PATH`:

```bash
python -m pytest -s -q -m device tests/ut/test_rms_norm_op.py
```

Set `DEVICE_TRANSPORT=binapp` before pytest collection to select
`BinRunnerTestCaseBasic`; otherwise `TestCaseBasic` uses HDC. New operator
device tests should inherit the selected transport base and keep only their
model generation, execution, and validation logic in `test_<op>_op.py`.

### 性能基线

性能采集不直接把设备 UT 的一次 stdout 当作结论。统一入口复用 UT 的确定性数据、
OMC 和 golden，先做精度门禁，再保存多批原始 wall 样本与独立 NPU profiling：

```bash
python tests/perf/operator_baseline.py \
  --warmup 10 --iterations 50 --repeats 3 --profiling
```

协议、正式 case 矩阵、快照位置和前后比较方法见
[`docs/performance/operator-baseline.md`](../docs/performance/operator-baseline.md)；后续性能相关测试应同步更新
[`benchmark-ascendc-operators`](../.claude/skills/benchmark-ascendc-operators/SKILL.md) skill。

### BinRunner 真机完整配方（本机实测，2026-08-14）

设备 127.0.0.1:15555（TCP 转发）+ BinRunner v1.1.2 沙箱 + model_run_tool。
三步前置 + 一套环境变量：

```bash
# ① 算子 install 到 DDK（否则 omg 报 libcustom_op.so: cannot open）
./build.py --ops MsRmsNorm

# ② BinRunner HAP 安装（v1.1.2 wheel 已含新设备 UDID 白名单）+ fport 转发
pip install binrunner==1.1.2        # 或用 third_party/BinRunner submodule 的 CLI（python -m binrunner，ADR-0015）
br setup --reinstall
br forward                           # 建立 hdc fport 8888（后台驻留，断了会 Connect refused）

# ③ DDK 环境 + 真机变量（HDC_TARGET_OPTION=-t 是关键，见下）
export HIAI_DDK=/home/zhugd/tool/hisi_ddk/ddk_2
source $HIAI_DDK/tools/tools_ascendc/set_ascendc_env.sh
export REMOTE_TARGET=127.0.0.1:15555
export MODEL_RUN_TOOLS_PATH=$PWD/tools/model_run_tool/build/model_run_tool
export DEVICE_TRANSPORT=binapp
export HDC_PATH=$HOME/tool/hmos-ndk/command-line-tools/sdk/default/openharmony/toolchains/hdc
export HDC_TARGET_OPTION=-t          # 当前 hdc 版本 -s 是 server、-t 才是设备
# 可选：OMG_PATH / SOC_VERSION（base_test 的 find_omg 默认从 DDK_PATH 推导）

/home/zhugd/.conda/envs/llm/bin/python -m pytest -s -q -m device tests/ut/test_rms_norm_op.py
```

> **坑（HDC_TARGET_OPTION）**：`base_test.py` 默认用 `-s` 指定设备，但当前 hdc
> 3.2.x 里 `-s` 是 hdc server 地址、`-t` 才是目标设备。用默认 `-s` 时 exec（br run）
> 正常、download 阶段所有 hdc 命令报 `Connect server failed`，表现成"设备跑完但
> 收不到反馈"卡在重试。必须 `export HDC_TARGET_OPTION=-t`。
>
> **坑（model_run_tool 路径）**：binapp transport 下 model_run_tool 的参数
> 必须用 `@/bin/...` 绝对路径（ArkTS 层展开为 filesDir）；相对路径会在 App 沙箱
> cwd 下找不到文件，NNRT 报 `modelSize is 0` / `Compilation_Build failed (rc=2)`。
> `test_<op>_op.py` 用 `self.remote_path()` 生成，天然正确，手动 br run 时注意。

`test_rms_norm_op.py` imports `MsRmsNorm` from `torch_custom/ms_rms_norm.py` (the
repo-level Torch adapter layer) and uses `torch.onnx.export(..., opset_version=18,
dynamo=False)` with `OperatorExportTypes.ONNX_FALLTHROUGH`. The legacy exporter is
selected explicitly so `MsRmsNorm.symbolic` emits one `custom::MsRmsNorm` node
rather than an expansion into standard ONNX ops. The custom domain remains at
opset 1.
