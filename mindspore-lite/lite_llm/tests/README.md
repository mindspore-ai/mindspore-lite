# Lite LLM unit tests

`run_llm_ut.sh` runs all Python tests under `tests/py/`, builds the host C++ test targets in an isolated result directory, and runs every CTest case registered from `tests/ut/`.

## Parameters

| Parameter | Required / default | Value |
| --- | --- | --- |
| `--output-dir DIR` | Optional / `./ut_results` | Result parent directory. Each invocation creates a new `run_*` directory. |
| `--jobs N` | Optional / `8` | Positive number of parallel C++ build jobs. |
| `-h`, `--help` | Optional | Print command usage. |

CI must prepare the Python dependencies, including the `mslite_llm_ops` wheel matching the selected release artifacts. It must also provide CMake, a host C/C++ compiler, and the repository's third-party dependency cache. The script does not install dependencies or modify the DDK.

```bash
bash /path/to/lite_llm/tests/run_llm_ut.sh \
  --output-dir /artifacts/llm-ut \
  --jobs 8
```

The exit code is written to `run_*/exit_code.txt`. Python results are written to `run_*/python-results.xml`; CTest failures and build output are printed to the CI console. The Python and C++ phases both run, and the script returns nonzero when either phase fails.
