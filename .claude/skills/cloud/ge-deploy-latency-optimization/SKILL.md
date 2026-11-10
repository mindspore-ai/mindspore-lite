---
name: ge-deploy-latency-optimization
description: MindSpore Lite 云侧 GE 在线推理（provider=ge、optimize=none MindIR）多卡部署时延优化：通过配置项与部署脚本编排，降低模型加载与首次推理编译的等待时间，并提供编译缓存的正确使用与限制说明。用户要优化 GE 在线推理的部署/加载/第一次推理时延，或需要配置编译缓存、排查编译竞态与缓存未命中时调用。
---

# GE Online Inference Deployment Latency Optimization

本技能说明 GE 在线推理（converter `--optimize=none` 产出的 MindIR + 运行时 `provider=ge`）多卡部署的时延优化方案：每项方案的原理、实现方式（配置文件样例或脚本样例）、使用约束与风险。所有方案均在 Atlas 300I Duo + Qwen3-8B TP=2 上实测通过。

## 适用范围（何时调用）

- 优化 GE 在线推理的**部署等待时间**（模型加载 build、第一次推理的编译窗口）
- 多卡（TP）场景的模型加载与预热编排
- 配置/使用 GE 编译缓存（图级缓存、算子级缓存），或排查缓存未命中
- 排查编译竞态（E40021 / EI0005 / EI0006）

**不适用**：端侧推理（LiteRT/Micro）、非 Ascend 后端。

## 可用性总表：什么时候可以、什么时候不可以、误用后果

| 方案 | 何时可以使用 | 何时不可以使用 | 误用后果 |
|---|---|---|---|
| 并行编译池（方案 1） | 任何时候；必须在第一次算子编译发生前设置 | 无 | 值 >32 或 <1 时被静默回退为 8；rank 数×N > CPU 核数×80%：机器负载过高、整体变慢甚至卡死 |
| 双 rank 并行 build（方案 2） | 两 rank 均为 GE provider 的独立 Model 实例 | — | 无（build 阶段不涉及集合通信） |
| 双图并发首推编译（方案 3） | config 已设置 `ge.AllowMultiGraphParallelCompile=1`；两 rank 预热在 barrier 内同步开始 | 未设置该开关；与 `ge.exec.variable_acc=True` 同时设置 | 未加开关强行并发：编译结果不可信（变量优化共享状态竞争）；已加开关但不同步：HCCL EI0005/EI0006 报错；与 variable_acc 同设：GE 直接拒绝 |
| 图级编译缓存（方案 4） | 同 MindIR + 同 config + 同 SoC；CANN 版本/架构与 mslite 构建版本需人工保持一致（**均不在缓存键中**）；缓存目录已预创建 | 任一条件变化时指望命中（会自动 miss 回退在线编译，功能安全但无加速）；重装自定义算子 vendor 后指望缓存自动失效（**不会**——算子代码不进指纹）；升级 CANN 后指望缓存自动失效（**不会**——CANN 安装信息不进指纹） | 重装 vendor 后未清缓存：**静默误用旧算子二进制**（精度/行为错误且无报错）；升级 CANN 后未清缓存：可能复用旧版本编译的产物；cache 目录放 /tmp：容量/清理风险 |
| 构建期预热+制品化（方案 5） | 构建机与生产机：同 SoC、同 config（cache_dir 路径可不同）；CANN 版本/架构与 mslite 构建版本需人工保持一致（均不在缓存键中） | 跨 SoC 分发；跨 CANN 版本分发（不会自动 miss，需先清理缓存） | 指纹不一致自动 miss（安全），但分发无效、白占磁盘；跨 CANN 版本分发且未清缓存 → 可能复用旧产物；分发不完整（缺 .rdcpkt/.idx 附属文件）→ 命中失败 |
| 算子级编译缓存（方案 6） | `ge.opDebugLevel` 未设置或为 0；`mode=enable`；cache 目录在持久文件系统 | `ge.opDebugLevel≠0`（算子级缓存被强制禁用）；`mode=force` 且存在其他实例共用该目录 | `opDebugLevel≠0` + 期望命中 → 静默不命中（白配）；`force` 共用目录 → **删除他人缓存**；cache 目录磁盘满 → 编译失败 |
| 串行预热编排（方案 7） | 任何时候（推荐默认形态） | — | 两 rank 预热顺序/档位不一致：HCCL EI0005；单 rank 独自预热：HCCL 死锁（等对端） |
| ONNX 原名 inputShape（方案 8） | 新版本 mslite **仅接受模型输入名（ONNX 原名）**；输入数与顺序与 ONNX 一致 | ① 旧版本 mslite → 必须用图参数名（U 名，每次导出会变化）；② **config 写 U 名（U<数字>）在本版本直接报错**：`Invalid input name 'U292' ... Use the original ONNX input names instead`（**没有位置回退/兼容**；U 名每次导出变化，属脆弱写法） | config 输入个数与图输入个数不一致 → build 失败 `FuncGraph input size != input size`（报错明确，易排查） |

**通用红线（任何方案都不允许）**：
- `ge.tiling_schedule_optimize=1`：实测编译窗口劣化 +143%，禁止设置；
- `ge.op_bank_path` / `ge.op_bank_update`：在该链路不生效，设置无意义；
- `ge.jit_compile=0`：算子落劣质实现，稳态劣化约 3×；
- **重装/更新自定义算子实现（vendor 写入 opp/vendors）后不清缓存**：图级与算子级缓存的命中 key 均不含算子代码内容，必须同时清空 `ge.graph_compiler_cache_dir` 与 `ge.op_compiler_cache_dir` 两个目录，否则静默复用旧算子二进制（精度/行为错误且无任何报错）；
- 预热的输出 buffer 的 dtype/shape 必须符合"物理产出"规则（见方案 7 的 4 条规则），错误会直接 E19999 或 size mismatch。

## GE 流程部署优化方案

### 1. 并行编译池扩容（TE_PARALLEL_COMPILER）

**原理**：GE 在线编译时，TBE 算子编译由一个进程池执行，默认 8 进程/rank（CANN 包 `te_fusion/parallel_compilation.py` 读取进程环境变量 `TE_PARALLEL_COMPILER`，合法范围 1~32）。扩大进程池可缩短算子编译窗口。

**实现（config 样例，写入 `[ascend_context]` 段）**：

```ini
[ascend_context]
TE_PARALLEL_COMPILER=16
```

mslite 在创建 GE session 前把 `[ascend_context]` 段中**白名单键**（当前仅 `TE_PARALLEL_COMPILER`，定义于 mslite 代码 `IsCompileEnvKey`；其余键一律忽略；不覆盖用户已 export 的值）导出为进程环境变量。**两种设置方式等价，任选其一**：

- **config 文件**：`[ascend_context]` 段写 `TE_PARALLEL_COMPILER=16`——部署配置随 config 走，进程内生效（即使进程已启动也能在首次编译前生效），适合由编排系统拉起、不便改启动脚本/镜像的场景。config 注入同时覆盖两处编译点：运行时在线 GE 编译（GeGraphExecutor）与 **converter ACL provider 的进程内离线编译**（`aclgrphBuildModel`，含 `popen("aoe")` 子进程继承）；
- **启动脚本 export**：`export TE_PARALLEL_COMPILER=16`——需要控制进程启动环境；若与 config 同时设置，**用户已 export 的值优先**（config 注入不会覆盖）。

**约束**：
- 合法范围 1~32，越界静默回退 8；
- 多 rank 并发编译时，总编译进程数 = rank 数 × N，必须 ≤ CPU 核数 × 80%；
- 必须在**第一次算子编译发生前**生效（session 创建前；mslite 的注入点满足）；
- 旧写法 `[ge_session_options]` 段 `ge.env.<NAME>=...` 无效（不是合法 GE 选项，会被 GE 忽略）：请写在 `[ascend_context]` 段（不带 `ge.env.` 前缀）。

**风险**：设置过大导致 CPU 过载、整机卡死（共享环境尤其注意）。

### 2. 双 rank 并行 build

**原理**：`build_from_file` 阶段只做图加载与基础编译（不含算子编译），两 rank 之间无依赖，串行执行纯属等待。

**实现（脚本样例）**：两 rank 的 worker 进程**同时** spawn（不要等 rank1 build 完再启动 rank0）：

```python
procs = []
for rank, dev in enumerate([2, 3]):          # 两 rank 同时启动
    p = mp.Process(target=worker, args=(rank, dev, cfg_pf[rank], cfg_dc[rank], ...))
    p.start()
    procs.append(p)
for _ in range(2):
    built_q.get()                            # 等两 rank 都 build 完
for _ in range(2):
    go_q.put(1)                              # barrier 释放，同时进入预热
```

**约束**：
- 两 rank 的 config 必须使用**独立生成的 rank_table_file 与独立编译目录后缀**（避免 TBE 编译缓存竞态导致 E40021 全崩）；
- 预热开始前必须 barrier 对齐（HCCL 集合通信要求两 rank 同时参与）。

**风险**：无（build 阶段不涉及集合通信）。

### 3. 双图并发首推编译（AllowMultiGraphParallelCompile）

**原理**：GE 编译是同步函数（在调用线程内逐阶段执行）。同一 rank 进程内有 prefill/decode 两个 Model，`RunGraph`（首推编译实际发生处）**无锁**——用两个线程分别并发预热两个 Model，两图的编译（pass 链 + 算子编译）即重叠执行。`ge.AllowMultiGraphParallelCompile=1` 的作用是解除并发编译的冲突源：强制关闭 VariableOpPass（该 pass 存在跨图共享状态），并启用 thread-local 编译上下文隔离（GE 开源源码 `ge_local_context.cc`）。

**实现**：

config（`[ge_session_options]` 段加一行）：

```ini
ge.AllowMultiGraphParallelCompile=1
```

脚本样例（同 rank 进程内，双线程并发预热；PF 线程跑 prefill、主线程跑 decode）：

```python
th_pf = threading.Thread(target=warm_pf)   # warm_pf 内循环 prefill 两个档位
th_pf.start()
warm_dc()                                   # 主线程循环 decode 两个档位
th_pf.join()
```

**约束**：
- **必须与 `ge.AllowMultiGraphParallelCompile=1` 同时使用**（否则并发编译竞争变量优化共享状态，结果不可信）；
- **与 `ge.exec.variable_acc=True` 互斥**（同时设置会被 GE 拒绝）；
- 两 rank 的预热必须在 barrier 内同步开始（HCCL 集合通信对齐）；
- 开启后 VariableOpPass 被关闭——权重不再做格式加速优化，需确认首 token 精度与图内无新增 TransData（Qwen3-8B 实测无影响）。

**风险（必须处理）**：**EI0005 竞态**——两 rank 间 prefill/decode 的集合通信调用顺序在线程调度下可能错位（实测出现过一次）。**防护：预热失败（EI0005/EI0006）时自动回退串行预热重试一次**；或不开启并发预热（串行是安全默认形态）。

### 4. 图级编译缓存（graph_compiler_cache_dir）

**原理**：GE 支持把整图编译结果序列化落盘；下次相同条件的 build 直接加载，跳过全部编译（pass 链 + 算子编译）。命中 key 是 mslite 传入的 graphKey——本仓库已把**可自动判定的条件**（图结构、config 全部键值、SoC）编进该 key：这些条件漂移 → key 变化 → 自动 miss → 回退在线编译，不会误用旧缓存；CANN 版本/架构与 mslite 构建版本不在 key 中，需按下方约束人工保持一致。

**实现（config 样例，`[ge_session_options]` 段）**：

```ini
ge.graph_compiler_cache_dir=/部署目录/cache/graph
```

（**cache 目录需预先创建**。目录不存在 / 是普通文件 / 不可写时，新版不会替你创建，而是把缓存**降级为"不启用"**并继续运行——功能正常，但每次启动都全量重编，等于没开缓存。只配 `model_cache_mode`、默认目录 `./model_build_cache_<rank>` 不可写时同样降级为不启用（WARNING），不再让 build 失败）

**约束（除标注"人工"的项外，命中需全部满足，不满足会自动 miss 回退在线编译）**：
- MindIR 图结构与算子 attrs 完全一致；
- config 所有键值完全一致（precision_mode/dynamicDims/环境注入等；两个 `*_cache_dir` 路径键除外，不参与指纹）；
- SoC 型号一致；
- **CANN 版本/架构需人工保持一致**（CANN 安装信息已不参与缓存键）：升级或更换 CANN 后不会自动 miss，需手工清理缓存目录；
- **key 不含 rank**（同一张图在两个 rank 上的 key 完全相同）——因此**多卡必须按 rank 分缓存目录**：
  `model_cache_dir=<dir>/rank{r}`、`ge.graph_compiler_cache_dir=<dir>_rank{r}`，或直接用 mem_opt 默认的
  `./model_build_cache_<rank>/`；**两 rank 共用一个目录会互相覆盖，不报错但数值全错**
  （实测：首对 P(yes) 0.1322 → 0.0168，40 对相关系数 0.17）；
- **mslite 构建版本建议保持一致**（构建版本不在缓存键中，升级后旧缓存**不会**自动失效；需要按新版本重建时，请手工清理缓存目录）。

**指纹盲区（唯一需要人工干预的场景）**：
- **自定义算子 vendor 二进制（opp/vendors）不在指纹内**——重装/更新算子实现后，图结构、config、SoC 均不变（CANN 版本同样不在键中），图级缓存会静默命中旧算子产物；算子级缓存（方案 6）的 key 同样不含算子代码，行为相同。**重装 vendor 后必须手动清空两个缓存目录**（见通用红线），除此之外缓存全自动判定、无误用风险。

**风险**：
- 缓存目录磁盘占用大（本例 ~16GB/轮）——放在容量充足的文件系统并定期清理；
- 首次部署（缓存空）无收益且多一次写盘；
- 不要手工移动/篡改缓存内文件后仍指向原路径（指纹外部分）。

### 5. 构建期预热 + 缓存制品化（生产环境推荐）

**原理**：把"首次全量编译"从生产机挪到构建机：构建机用**生产同版本** mslite + 同 config 跑一次预热（开启图级缓存）→ 缓存目录随部署制品分发到生产机 → 生产机首次启动即命中。

**实现**：
1. 构建机：按用例 1 执行一次完整预热（cache 目录落在构建机指定路径）；
2. 将整个 cache 目录打包进部署制品；
3. 生产机：解包到 config 指定的 cache 路径（路径可与构建机不同——cache_dir 路径不参与指纹）；
4. 正常启动服务。

**约束**：
- 构建机与生产机：同 SoC、同 config 内容（指纹自动校验，不一致自动 miss 回退全量编译）；CANN 版本/架构与 mslite 构建版本需人工保持一致（均不在缓存键中，跨版本分发前先清理缓存）；
- cache 目录中除产物外还有权重等附属文件，需整体分发。

**风险**：分发不完整（缺 `.rdcpkt/.idx` 附属文件）会导致命中失败——整体目录分发即可。

### 6. 算子级编译缓存（op_compiler_cache）

**原理**：图级缓存粒度是"整图"（任一条件变化全 miss）。算子级缓存按 **单个算子** 存储编译产物（key = sha256(算子类型+输入输出 desc+attrs+编译选项)，**显式抹除图上下文**——图名/节点名/device_id 不参与），图结构部分变化时，未变化的算子仍可命中，只重编变化的算子。编译成功后自动落盘，下次编译自动查询（含 sha256 完整性校验）。

**实现（config 样例，`[ge_session_options]` 段）**：

```ini
ge.op_compiler_cache_dir=/部署目录/cache/op
ge.op_compiler_cache_mode=enable
```

**约束**：
- **`ge.opDebugLevel` 必须为 0（或不设置）**——非 0 时算子级缓存被强制禁用（两者互斥）；
- **`mode=force` 禁止在多实例/共享目录使用**——会在初始化时清空整个 cache 目录；
- 自定义（tikcpp）算子同样支持（InnerPFA 静态特化 kernel 实证落盘并跨进程命中）；
- **`op_cache.ini` 必须随缓存目录一起分发**——只有 `<soc>/` 子目录没有 ini（或 ini 版本与生产机 CANN 不一致）时，GE 首次编译初始化会**清空整个 `<soc>` 子目录**（te_cache_space_manager.cc），预置缓存全部作废；
- **缓存目录必须可写**——命中路径刷新文件 atime（老化机制），只读目录上 utime 失败会被判 miss（系统性全量重编）；真只读部署必须 `export ASCEND_MAX_OP_CACHE_SIZE=-1` 关闭老化（防 500MB 默认限额修剪）；
- 解析优先级：`ge.op_compiler_cache_dir`(config) > `ASCEND_CACHE_PATH`(env) > `$HOME/atc_data`（默认即开启）。**config 键进 G1 图缓存指纹、env 不进**——预热机与生产机缓存路径不同时用 env 保持 G1 命中；
- 图结构/档位变化会使对应算子签名变化 → 对应条目 miss（细粒度的预期行为，其余算子仍命中）；
- **key 不含算子实现代码内容**——重装/更新自定义算子 vendor 后对应条目不会自动失效，必须清空缓存目录（见通用红线）。

**制品化（首次冷部署消除算子在线编译，opt_ge.md §16 NPU 实证闭环）**：预热捕获（全新 cache 目录 + env 驱动）→ 打包校验 → 随部署制品分发 → 生产机可写恢复 + 环境变量注入。kernel 与在线特化编译产物**逐位一致**（推理性能零损失）。实测（Qwen3-8B TP=2）：捕获 119 条目/2 条 InnerPFA 静态特化，制品 0.3MB，恢复后冷启动零重编译。工具（qwen3_8b 目录，pack/deploy 通用）：`opcache_warmup_capture.py`（预热捕获，`--cache-dir` 验证模式带命中判定）/ `opcache_pack.py`（sha256 抽样校验+manifest）/ `opcache_deploy.py`（版本核验+恢复+环境变量输出）。

**风险**：
- 缓存目录磁盘增长（500MB 默认上限与 atime 老化，`op_cache.ini`/`ASCEND_MAX_OP_CACHE_SIZE` 调整；预置制品目录建议关闭老化或留余量）；
- 与 `ge.opDebugLevel=1`（保留编译产物调试）互斥——需要调试产物时接受 op cache 失效。

### 7. 串行预热编排（首次部署的预热标准形态）

**原理**：GE 在线图的算子编译懒触发在第一次 RunGraph。用 dummy 输入按档位顺序跑一遍 predict，即可完成全部编译；之后真实请求零编译。两 rank 必须同步推进（集合通信对齐）。

**实现（脚本样例，worker 内，barrier 释放后）**：

```python
# 1) prefill 两个档位（顺序两 rank 一致）
for bucket, kv in [([256,256,256], 768), ([512,512,512], 1024)]:
    feeds = [mslite.Tensor(np.ones((1, bucket[0]), np.int32)),          # 注意 INT32
             mslite.Tensor(np.ones((1, bucket[0]), np.int32)),
             mslite.Tensor(np.arange(bucket[0], dtype=np.int32).reshape(1, -1))]
    pf_outs = [mslite.Tensor(np.zeros((151936,), np.float32)),          # 物理 dtype：logits fp32
               mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16)),# KV fp16
               mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16))]
    pf.predict(feeds, pf_outs)                                          # 必须显式传 outputs
# 2) decode 两个档位
for bucket, kv in [([768,768,768], 768), ([1024,1024,1024], 1024)]:
    dfeeds = [mslite.Tensor(np.zeros((1, 1), np.int32)),
              mslite.Tensor(np.ones((1, kv), np.int32)),
              mslite.Tensor(np.zeros((1, 1), np.int32)),
              mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16)),
              mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16))]
    dc_outs = [mslite.Tensor(np.zeros((1, 1, 151936), np.float32)),
               mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16)),
               mslite.Tensor(np.zeros((36, 1, 4, kv, 128), np.float16))]
    dc.predict(dfeeds, dc_outs)
```

**预热 4 条硬规则（违反即失败）**：
1. **必须显式传 outputs**——GE 在线图 `get_outputs()` 返回空 shape，缺 outputs 直接 E19999；
2. **输出 buffer 用物理 dtype**——声明 fp32、GE 实际产出 fp16（logits fp32 / KV fp16），按声明构造会 size 2× mismatch；
3. **输入是 INT32**（ONNX int64 被转换器降型为 int32）；
4. **TP 两 rank 必须在 barrier 内同步预热**，且预热顺序（图顺序、档位顺序）两 rank 一致——单 rank 独自预热会 HCCL 死锁，顺序错位会 EI0005。

**约束**：预热档位集合必须覆盖业务实际会用到的全部档位（GE 一次预热覆盖 config 的全部 dynamicDims 档位，之后切档零编译）。

**风险**：预热档位集合与真实请求档位不一致时（超出 dynamicDims 范围），运行时报错——dynamicDims 必须覆盖业务 shape 域。

## 生成后的 per-rank config 完整样例（prefill；decode 同构）

部署脚本生成（合并 rank_table 后）的运行时 config 最终形态：

```ini
[ascend_context]
rank_table_file=/部署工作目录/tp_run_deploy/rank_table.json
plugin_custom_ops=All
TE_PARALLEL_COMPILER=16

[ge_session_options]
ge.graph_compiler_cache_dir=/部署目录/cache/graph
ge.op_compiler_cache_dir=/部署目录/cache/op
ge.op_compiler_cache_mode=enable
ge.AllowMultiGraphParallelCompile=1
ge.constLifecycle=session
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.precision_mode=force_fp16
ge.externalWeight=2

[ge_graph_options]
ge.inputShape=input_ids:1,-1;attention_mask:1,-1;position_ids:1,-1
ge.dynamicDims=256,256,256;512,512,512
ge.dynamicNodeType=1
```

decode 图差异：`ge.inputShape=input_ids:1,1;attention_mask:1,-1;position_ids:1,1;past_key_cache:36,1,4,-1,128;past_value_cache:36,1,4,-1,128`，`ge.dynamicDims=768,768,768;1024,1024,1024`。

说明：
- `inputShape` 使用 **ONNX 原名**（新版本 mslite 仅接受模型输入名，写 U 名会直接报 `Invalid input name 'U292' ...` 并列出合法输入名，**无位置回退**；仅当旧版本 mslite 或模型未记录张量名时，才需要改用每次导出都会变化的图参数名）；
- `rank_table_file` 每次部署由脚本生成（TP 场景必须包含，否则 HCCL 初始化失败）；
- 缓存两个目录需预先创建。

## 部署脚本完整用例

```python
#!/usr/bin/env python3
# 完整可运行参考：qwen3_8b/e21_fullstack.py（去掉实验参数即为生产形态）
import multiprocessing as mp
import os, sys, threading, time
import numpy as np
import mindspore_lite as mslite

VOCAB = 151936
CACHE = "/部署目录/cache"          # graph/ 与 op/ 两个子目录，预创建

def worker(rank, device_id, cfg_pf, cfg_dc, built_q, go_q):
    ctx = mslite.Context()
    ctx.target = ["ascend"]
    ctx.ascend.device_id = device_id
    ctx.ascend.rank_id = rank
    ctx.ascend.provider = "ge"
    # 阶段1：build（两 rank 同时进行）
    pf = mslite.Model()
    pf.build_from_file(f".../rank{rank}/prefill/..._graph.mindir",
                       mslite.ModelType.MINDIR, ctx, cfg_pf)
    dc = mslite.Model()
    dc.build_from_file(f".../rank{rank}/decode/..._graph.mindir",
                       mslite.ModelType.MINDIR, ctx, cfg_dc)
    built_q.put(rank)
    go_q.get()                                   # barrier：两 rank 都 build 完
    # 阶段2：串行预热（见"串行预热编排"样例；首次部署在此发生全部编译）
    ...
    # 此后进入服务循环：predict 时延与预热完成时一致

if __name__ == "__main__":
    built_q, go_q = mp.Queue(), mp.Queue()
    procs = [mp.Process(target=worker, args=(r, d, cfg_pf[r], cfg_dc[r], built_q, go_q))
             for r, d in enumerate([2, 3])]
    for p in procs: p.start()                    # 两 rank 同时启动（并行 build）
    for _ in range(2): built_q.get()
    for _ in range(2): go_q.put(1)               # barrier 释放 → 预热
    ...
```

per-rank config 生成：TP 场景必须由脚本把 `rank_table_file` 合并进源 config（TP 场景 config 不含 rank_table_file 会在 HCCL 初始化时报 `hccl_graph_optimizer Subgraph pre proc failed`）。

## 实战补充：GE 缓存的实际使用方式（Qwen3-Reranker-8B CP2 实测）

> 实测环境：**Atlas 300I Duo ×2 + CANN 9.1.0 + mslite 2.11.0（wheel 2026-10-21，commit 3c0217cb）**，
> 场景为 **CP2 双 rank：prefix 静态 + suffix 4 档动态**。结论可类推到其它「每 rank 图不同」的多卡推理。
>
> 注意有两处行为在本仓库 `!1395 fix ge cache dir issue`（2026-11-04）之后发生了变化，
> 下文凡涉及处都标注了【旧版】/【新版】，落地前先确认运行时包的版本。

### 两类缓存与最小配置

```ini
[ascend_context]
model_cache_mode=mem_opt            ; 打开图级缓存；未配置目录时默认 ./model_build_cache_<rank>
; model_cache_dir=<dir>/rank{r}     ; 可选：显式目录（必须按 rank 分开，见下文）

[ge_session_options]
ge.op_compiler_cache_dir=<dir>/op   ; 算子级缓存（可选）
ge.op_compiler_cache_mode=enable
```

- 图级缓存：命中则**整图跳过编译**，落盘 `<dir>/<key>.idx | .om | .rdcpkt` + `weight/`；
- 算子级缓存：落盘 `kernel_cache/op_cache.ini` 与 `kernel_cache/<SoC>/te_<算子名>_<sha256>.json`；
- 目录来源优先级（实测）：
  1. `[ascend_context] model_cache_dir`（会被映射成 `ge.graph_compiler_cache_dir`）**优先**：两者都配且都有效时，
     日志为 `Update session attr ge.graph_compiler_cache_dir to <model_cache_dir>`，产物只写该目录，
     `ge.graph_compiler_cache_dir` 目录保持为空；
  2. `model_cache_dir` 无效（不存在 / 是文件 / 不可写）而 `[ge_session_options] ge.graph_compiler_cache_dir` 有效时
     **回退**到后者，日志为 `Invalid model_cache_dir '<dir>', keep using ge.graph_compiler_cache_dir '<dir2>'`；
  3. 两者都未配置时才用默认的 `./model_build_cache_<rank>`（需 `model_cache_mode=mem_opt`）。

### 指纹（缓存 key）规则

| 参与指纹 | 不参与指纹 |
|---|---|
| 图结构/算子 attrs、**config 全部键值**、SoC、图序号 | **rank**、CANN 版本与架构、mslite 构建版本、`*_cache_dir` 路径键、非 config 来源的 session 参数（如经 `AscendDeviceInfo` 设置的 `precision_mode`） |

实测要点：

1. config 里**任何键值变化都会 miss**：加一行 `ge.AllowMultiGraphParallelCompile=1`，或删掉
   op cache 的两行，都会换 key 重编（本用例冷编 447~609s，热跑 285~295s）；
2. **`*_cache_dir` 路径不入指纹**：把整份缓存目录复制到新路径、只改路径键，仍然命中
   （实测 285s ≈ 热跑 295s，且副本目录**无新增 `.om`**）→ 制品化可换盘/换机（需同 SoC、同 CANN、同 mslite 构建）；
3. session 参数一变就要**手工清缓存**，否则复用旧产物。

### ⚠ 最关键：缓存 key 不含 rank → 两个 rank 必须分开目录

- 证据：rank0/rank1 各自目录里的条目名**完全相同**（`1_U0_1_39_15028296068670656355.idx`），
  但 `.om` 大小不同（45974714 vs 45975740 字节）→ **key 不区分 rank，产物却不同**；
- 后果：两 rank 共用一个缓存目录 → 互相覆盖/复用对方的图，**不报任何错，但数值全错**
  （实测 GE 与 eager 的分数相关从 **0.9997 掉到 0.13~0.17**，成对排序从 **16/20 掉到 4/20 / 9/20**）；
- 正确做法（二选一）：

  1. **每 rank 独立目录**：用 mem_opt 默认的 `./model_build_cache_<rank>/`，或
     `model_cache_dir=<dir>/rank{r}`、`ge.graph_compiler_cache_dir=<dir>_rank{r}`；
  2. **每 rank 独立 graph_key**：`ge.graph_key=<tag>_r{r}`，且**必须写在 `[ge_graph_options]` 段**
     （写在 `[ge_session_options]` 会被 mslite 的自动 key 覆盖，实测无效）。共用目录下实测条目名
     变成 `suffix_r0.*` / `suffix_r1.*`，两份互不覆盖、数值正确（与基线逐位一致）。

- **前缀图不需要 per-rank key**：两张 per-rank 前缀图完全相同（full-head、无集合通信），
  共用目录下只有一个条目、不会冲突；只有「每 rank 图不同」的 suffix 图需要隔离。

### ⚠ 缓存目录必须预先创建（行为随版本不同）

| 版本 | 目录不存在 / 不可用时 |
|---|---|
| 【旧版】≤ 2.11.0-3c0217cb（实测） | build 阶段**不报错**，第一次 `predict` 才报 `File_Operation_Error_Invalid_Path(E13026) ... The cache directory does not exist`；rank worker 抛 `RuntimeError: predict failed!` 退出后，**编排父进程会永久阻塞**（实测挂住 914s 后才人工 kill） |
| 【过渡版】`f8e9c866`（2026-11-02 实测） | 启动时用 `IsUsableCompileCacheDir()` 校验（存在 / 是目录 / 可写）；不满足时打 **`ERROR`**，随后**把缓存降级为"不启用"并继续跑**，不再让 build 失败 |
| 【新版】含 `!1395`（`8d42c12a` 实测） | 同上，但该行日志级别为 **`WARNING`**（`Compile cache dir '<dir>' does not exist` → `Invalid model_cache_dir '…', GE compile cache is disabled for this model`）；若 `[ge_session_options]` 里另有有效目录，则保留那个（见上文"目录来源优先级"） |

不论版本：**目录一定要预建**（否则缓存白配、等于没开），预热/服务脚本仍应做子进程死亡与超时检测
（旧版本会挂死，新版本虽不挂但也失去了缓存收益）。

### 收益与观测

- 收益（同一 config）：冷编 **609s** → 热跑 **295s**（−52%）；冷启动整程约 9 分钟，命中缓存约 5 分钟；
- 是否命中：看 `<dir>` 下 `.idx` / `.om` 的**文件名与 mtime**——`.om` 名字里带编译时间戳，
  出现新的 `<key>_<时间戳>.om` 就是重编了：

  ```bash
  ls -la <cache_dir> | grep -E '\.(idx|om)$'
  find <op_cache_dir>/kernel_cache -type f | wc -l   # 图缓存命中后不再增长
  ```

- 算子级缓存：`ge.opDebugLevel != 0` 时该目录**不会有任何产物**（算子缓存被禁用）；
- 观测编译进程：本机 CANN 下 `ps -ef | grep bisheng` **恒为 0**，真正的 TBE 并行编译池是
  `python3.11 -c from multiprocessing...`（按它计数才与 `TE_PARALLEL_COMPILER` 对得上）。

### 清缓存的时机（任一变化都要清）

重装/更新自定义算子 vendor（指纹不含算子代码）、改 `precision_mode` / `externalWeight` 等
session 参数、换 SoC / CANN 大版本 / mslite 构建、改 config 想强制重编、把别的模型的缓存目录
挪给本模型用。

### 副作用

- 【旧版】`model_cache_mode=mem_opt` 即使已被显式目录覆盖，**仍会创建**空的
  `./model_build_cache_<rank>/`；【新版】配置了目录就**不再创建**（避免残留空目录、
  也避免 CWD 不可写时 build 失败）；
- GE 会在**当前工作目录**留下 `kernel_meta/`、`fusion_result.json` 与 16GB 级缓存目录 →
  用独立工作目录跑，避免污染示例/仓库目录。

## 缓存使用限制汇总

| 缓存 | 命中条件（全部满足） | 自动失效条件（任一出现） | 禁止事项 |
|---|---|---|---|
| 图级（方案 4） | 图结构、config 全部键值（`*_cache_dir` 路径键除外）、SoC 完全一致；CANN 版本/架构与 mslite 构建版本需人工保持一致（均不在键中） | 上述自动判定条件任一变化（自动 miss 回退在线编译）；CANN 版本/架构、mslite 构建版本变化**不会**自动失效 | 不要把 A 条件的缓存目录指给 B 条件的模型当"加速"（指纹不同不会命中，不会误用）；重装自定义算子 vendor 后不清缓存目录（指纹不含算子代码，会静默误用旧产物）；升级 CANN / mslite 后不清缓存（可能复用旧产物） |
| 算子级（方案 6） | 单算子的 (类型+输入输出 desc+attrs+编译选项) 一致（key 抹除图名/节点名/device_id，跨图跨部署有效） | 对应算子签名变化（仅该算子重编） | `ge.opDebugLevel≠0`（禁用 op cache）；`mode=force` 共用目录（清空他人缓存）；把 cache 目录放 /tmp（容量/清理风险） |

两个缓存可以同时开启：图级命中 = 全部跳过（最优）；图级 miss（条件变化）时算子级缓存仍可复用未变化算子的 kernel。

## 部署时延实测参考（Atlas 300I Duo + Qwen3-8B TP=2）

以下为上述方案全部叠加后的实测数据，供容量规划参考（详细过程见仓库根 opt_ge.md）：

| 场景 | 方案组合 | 部署总等待 |
|---|---|---|
| 首次部署（缓存空，全量编译，串行预热） | 方案 1+2+3+4(空)+6(空)+7+8 | **449.4s**（build 81.6s 并行 + 串行预热 367.8s） |
| 重部署（G1+G3 命中，串行预热即缓存加载） | 同上，缓存已就位 | **92.7s**（build 71.4s + 预热 21.4s） |
| 首请求（预热后第一个真实请求） | 预热完成 | **0.15s** |
| 稳态推理 | — | prefill 152~156ms / decode 与基线一致，零回归 |

可选叠加：方案 3 的双线程并发预热可把首次部署的预热段 367.8s 压缩到 ~227s（两图编译重叠），但存在 EI0005 竞态风险，必须实现失败自动重试（回退串行）后才可用于生产。

## 验证与观测说明

以下方法不依赖先验知识，按步骤即可确认每个优化是否生效：

> **日志级别**：缓存相关的 mslite 消息（`skip creating the default build cache dir`、`Update session attr ge.graph_compiler_cache_dir to …`、`Invalid model_cache_dir … keep using …`、`… is not writable`）都是 **INFO**，默认不输出到 stdout；想直接看到它们需要 `GLOG_v=1`（mslite）+ `ASCEND_GLOBAL_LOG_LEVEL=1` + `ASCEND_SLOG_PRINT_TO_STDOUT=1`（GE）。不打开时只能靠本节的文件/耗时方法判断。

1. **确认并行编译池生效**：编译期间另开终端执行 `ps -ef | grep bisheng | wc -l`，进程数 ≈ 设定的 TE_PARALLEL_COMPILER 值即生效。
2. **确认环境变量注入生效**（config 注入与 export 两种方式均可用此法验证）：部署脚本运行后，在脚本内执行 `python3 -c "import ctypes; print(ctypes.CDLL(None).getenv(b'TE_PARALLEL_COMPILER'))"`。注意：`os.environ` 和 `/proc/self/environ` 都是进程启动时的快照，看不到运行期 C 层 `setenv` 的结果，必须用 ctypes 方式验证。
3. **确认双图并发预热生效**：预热期间观察 prefill 与 decode 两个线程是否同时处于编译状态（串行时 prefill 完成前 decode 不会开始）；或比较并发与串行两次预热的总时长（并发应接近两者中较大值而非之和）。
4. **确认图级缓存命中**：cache 目录下出现 `.om/.rdcpkt/.idx` 文件；第二次部署预热时长从分钟级降到秒级，且这些文件集合不变化。未命中时不会报错，只是重新走全量编译——通过预热时长即可区分。
5. **确认算子级缓存命中**：`<op_cache_dir>/kernel_cache/<SoC型号>/` 下出现按"算子名+hash"命名的 `.o/.json` 文件；第二次部署文件数量不增加（纯复用）。注意 `ge.opDebugLevel≠0` 时该目录不会有任何产物（算子级缓存被禁用）。
6. **确认输入名匹配生效**：config 使用 ONNX 原名且 build 成功、日志无 `Invalid input name` 报错；若 config 仍写 U 名，会直接报 `Invalid input name 'U...'` 并列出全部合法输入名，按提示改成 ONNX 原名即可（**没有位置回退/兼容**）。
7. **时延测量口径**：部署总等待 = 进程启动 → 最后一个预热完成的时刻（`time.perf_counter()` 差值）；预热窗口 = barrier 释放 → 预热完成。
8. **对照纪律**：任何 A/B 时延比较必须在同一批次内交替进行（A/B/A/B），因为连续编译后机器性能会漂移（实测同条件三轮 404s→1322s）；跨批次的绝对数值不可直接比较。
9. **精度判据**：预热与推理输出的 logits 需满足 max/min/finite 与基线一致（Qwen3-8B 参考值 max≈19.453、min≈-6.098、finite=True）；任何优化方案不允许改变该数值。
