# MsRmsNorm for Kirin 9020

基于 `op_host/ms_rms_norm_tiling_data.h` 重写的 FP16 RMSNorm。算子对输入
最后一维执行归一化，支持任意非空 ND 前导维以及可选 gamma：

```text
y = fp16(fp32(x) * rsqrt(mean(fp32(x)^2, -1) + epsilon) * gamma)
```

gamma 缺省时不执行乘权重。

## 接口

| 方向 | 名称 | dtype | shape |
|---|---|---|---|
| Input 0 | x | fp16 | `[..., K]`，K 为 16 的倍数且 K≤8192 |
| Input 1 | w | fp16 | 可选 `[K]` |
| Output 0 | y | fp16 | 同 x |

`epsilon` 默认 `1e-6`。2048、4096、896、2560、8192 等 hidden size 都满足
当前对齐契约；shape 不再硬编码到某一个模型。

## 目录

```text
ascendc_ops/MsRmsNorm/
├── operator.json
├── op_host/
│   ├── ms_rms_norm.cpp
│   └── ms_rms_norm_tiling_data.h
├── op_kernel/
│   ├── config.json
│   ├── ms_rms_norm.cpp
│   └── ms_rms_norm_impl.cpp
├── framework/onnx_plugin/onnx_ms_rms_norm_plugin.cc
├── gen_data.py
├── temp.json
└── DESIGN.md
```

Torch eager 与 ONNX symbolic 位于仓库根目录
`torch_custom/ms_rms_norm.py`，测试位于 `tests/ut/test_rms_norm_op.py`。

## 构建与测试

```bash
source .venv/bin/activate
source ddk_env/tools/tools_ascendc/set_ascendc_env.sh
./build.py --ops MsRmsNorm

python -m pytest -q tests/ut/test_rms_norm_op.py -m "not device"
DEVICE_TRANSPORT=binapp python -m pytest -s -vv \
  tests/ut/test_rms_norm_op.py -m device
```

发布包位于 `output/ms_ops_pack/custom_opp_ubuntu_x86_64.run`。只有完成
run 包完整性检查、从发布来源生成 OMC 并在实机与 golden 对拍后，才可将该
版本标记为实机验证通过。
