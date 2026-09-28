# Lite LLM ST 入口

`run_llm_st.sh` 检查门禁准备好的 DDK、自定义算子和 Python 依赖，随后执行 `tests/st/` 下的全部用例。

## 参数

| 参数 | 必填 / 默认值 | 填写内容 |
| --- | --- | --- |
| `--package FILE` | 必填 | 本次构建的 Lite LLM NNRT `.tar.gz` 发布包。 |
| `--model-dir DIR` | 必填 | ST 模型根目录。各用例按 `conftest.py` 中登记的相对路径读取自己的模型。 |
| `--device ID` | 必填 | `br devices` 列出的 Kirin 9020 真机 SN，原样传给 BinRunner。 |
| `--br-port PORT` | 可选 / `8888` | 当前设备对应的本机 HDC 转发端口；多设备共用服务器时传入各自端口。 |
| `--output-dir DIR` | 可选 / `./st_results` | 结果父目录；每次执行创建新的 `run_*` 子目录。路径仅使用英文字母、数字、`/`、`_`、`-`。 |
| `-h`、`--help` | 可选 | 显示调用格式。 |

模型目录当前至少包含：

```text
<model-dir>/
└── qwen2.5-0.5b-instruct-q4_0.gguf
```

后续增加 4B 等模型时，将模型放入同一根目录，在 `conftest.py` 的 `MODELS` 中登记相对路径和导出参数，并在测试函数上声明对应的 `model_id`。入口脚本无需增加模型参数。

## 职责

门禁在调用脚本前负责：

1. 激活固定的 Python 环境并安装导出依赖。
2. 配置 GCC C++ 头文件，安装并激活 DDK 6.1.1.0；准备 DDK platform plugin 的动态库和 Python 路径。
3. 安装版本匹配的算子 `.run` 和 `mslite_llm_ops` wheel。
4. 准备模型目录、BinRunner 和真机连接。

脚本负责：

1. 校验参数、DDK、自定义算子和 Python 依赖。
2. 创建独立结果目录，并设置本次 pytest 的临时目录、设备 SN、缓存目录和 TeFusion 单进程编译策略。
3. 执行整个 `tests/st/`，生成 `results.xml`。
4. 将失败、异常、跳过或零用例作为非零退出码返回门禁。
5. 真机执行失败时，将脱敏后的 BinRunner、NNRT 和 CANN 诊断摘要打印到门禁控制台。

脚本不安装或修改 Python、GCC、DDK、算子及其动态库环境。

## 执行

```bash
# 查看设备 SN
br devices

# 替换本次发布包、模型根目录、设备 SN 和结果目录
bash /path/to/lite_llm/tests/st/run_llm_st.sh \
  --package /artifacts/mindspore-lite-llm-linux-x64-0.1.0.tar.gz \
  --model-dir /models/lite_llm \
  --device DEVICE_SN \
  --br-port DEVICE_LOCAL_PORT \
  --output-dir /artifacts/llm-st
```

门禁使用脚本退出码判断结果，并归档 `run_*/results.xml` 和 `run_*/pytest/`。pytest 输出直接打印到门禁控制台；失败诊断不打印设备 SN、绝对路径、提示词或生成内容。
