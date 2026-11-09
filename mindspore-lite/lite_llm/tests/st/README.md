# Lite LLM ST 入口

`run_llm_st.sh` 检查门禁准备好的 DDK、自定义算子和 Python 依赖，随后执行当前节点平台对应的 ST 用例。

## 参数

| 参数 | 必填 / 默认值 | 填写内容 |
| --- | --- | --- |
| `--package FILE` | 必填 | 本次构建的 Lite LLM NNRT `.tar.gz` 发布包。 |
| `--model-dir DIR` | 必填 | ST 模型根目录。各用例按 `conftest.py` 中登记的相对路径读取自己的模型。 |
| `--device ID` | 必填 | 当前门禁节点分配的设备 SN。 |
| `--platform PLATFORM` | 必填 | 当前设备的平台标签，只能填写 `kirin9020` 或 `kirin9030`。 |
| `--br-port PORT` | 可选 / `8888` | 当前设备已有的 BinRunner 本地转发端口，范围为 `1`–`65535`。 |
| `--output-dir DIR` | 可选 / `./st_results` | 结果父目录；每次执行创建新的 `run_*` 子目录。路径仅使用英文字母、数字、`/`、`_`、`-`。 |
| `-h`、`--help` | 可选 | 显示调用格式。 |

模型目录当前至少包含：

```text
<model-dir>/
└── qwen2.5-0.5b-instruct-q4_0.gguf
```

后续增加 4B 等模型时，将模型放入同一根目录，在 `conftest.py` 的 `MODELS` 中登记相对路径和导出参数，并在测试函数上声明对应的 `model_id`。入口脚本无需增加模型参数。

每个 ST 必须声明运行平台。只支持一个平台时添加对应 marker；两个平台都支持时叠加两个：

```python
@pytest.mark.kirin9020
@pytest.mark.kirin9030
def test_full_chain(...):
    ...
```

9020 和 9030 节点分别调用一次脚本。脚本只收集与 `--platform` 相同的 marker；当前平台没有用例时正常退出。一个用例同时声明两个 marker 时，会由两个节点各执行一次。

## 职责

门禁在调用脚本前负责：

1. 激活固定的 Python 环境并安装导出依赖。
2. 配置 GCC C++ 头文件，安装并激活 DDK 6.1.1.0。
3. 安装版本匹配的算子 `.run` 和 `mslite_llm_ops` wheel。
4. 准备模型目录、BinRunner 和真机连接。

脚本负责：

1. 校验设备、平台、DDK、自定义算子和 Python 依赖。
2. 根据 `--platform` 选择 pytest marker，并配置对应的 DDK Python 和动态库路径。
3. 在当前节点分配的设备上执行用例，生成 `results.xml`。
4. 当前平台没有用例时正常退出；已选中的用例发生失败、异常、跳过或零执行时返回非零退出码。
5. 真机执行失败时，将脱敏后的 BinRunner、NNRT 和 CANN 诊断摘要打印到门禁控制台。

脚本不安装或修改 Python、GCC、DDK、算子及其动态库环境。

## 执行

```bash
# 查看设备 SN
br devices

# 9020 节点
bash /path/to/lite_llm/tests/st/run_llm_st.sh \
  --package /artifacts/mindspore-lite-llm-linux-x64-0.1.0.tar.gz \
  --model-dir /models/lite_llm \
  --device DEVICE_9020_SN \
  --platform kirin9020 \
  --br-port 8888 \
  --output-dir /artifacts/llm-st

# 9030 节点
bash /path/to/lite_llm/tests/st/run_llm_st.sh \
  --package /artifacts/mindspore-lite-llm-linux-x64-0.1.0.tar.gz \
  --model-dir /models/lite_llm \
  --device DEVICE_9030_SN \
  --platform kirin9030 \
  --br-port 8888 \
  --output-dir /artifacts/llm-st
```

门禁使用脚本退出码判断结果，并归档 `run_*/results.xml` 和 `run_*/pytest/`。每次调用都会创建唯一的 `run_*` 目录，两个节点不会覆盖结果。pytest 输出直接打印到门禁控制台；失败诊断不打印设备 SN、绝对路径、提示词或生成内容。
