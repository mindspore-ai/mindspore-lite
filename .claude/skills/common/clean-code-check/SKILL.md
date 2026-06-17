---
name: clean-code-check
description: 改完或 review MindSpore Lite 的 C++/Python/Shell/CMake 代码时用。先跑 pre-push（8 个 lint 全自动），再按 30 秒快速清单逐项过——三类硬指标（CCN≤19 / NLOC≤50 / 入参≤5）+ 指针/边界/逻辑/异常/接口契约等 lint 抓不到的语义问题 + 涉及外部数据时的安全编码。深度参考 clean-code-guidelines.md。
---

# MindSpore Lite Clean Code Skill

## 30 秒快速 review 清单

按顺序过，前 5 项最常被遗漏：

1. **跑 pre-push**——`bash scripts/pre_commit/githooks/pre-push`（8 个 lint 全自动）
2. **三类硬指标**——CCN ≤ 19 / NLOC ≤ 50 / 入参 ≤ 5，超阈值必须重构或加白名单
3. **指针校验**——`cast<>` 后必查 null；`delete` 后置 `nullptr`；优先 RAII
4. **边界 / off-by-one**——`>` 改 `>=`；`==0` 改 `<=0`（防御负数）；广播用 shape 比较非元素数
5. **逻辑反了**——`strcmp` 匹配返回 0；error 路径返回 `RET_ERROR`；循环变量对应循环
6. **错误码 vs 异常**——默认 `MS_CHECK_*` 返回码；`MS_EXCEPTION_*` 仅构造/运算符/不可恢复时
7. **接口契约**——`override` / `explicit` / `const` / Lambda 显式捕获
8. **返回值不 cast void**——安全函数、系统调用、文件操作的返回值必查
9. **外部数据校验**（涉及 IPC/文件/网络/用户输入时）——注入、整数溢出、缓冲区溢出
10. **UT 覆盖**——新分支、新边界条件必须有对应用例

下面是每项的展开。深度示例、CI 配置、PR 案例库见 [clean-code-guidelines.md](clean-code-guidelines.md)。

## 第一步：pre-push 抓机械问题

```bash
bash scripts/pre_commit/githooks/pre-push     # 8 个 lint：clang-format/cmakelint/codespell/cpplint/lizard/pylint/shellcheck/tab/codespell
```

工具缺失按提示装：`cd scripts/pre_commit && bash install_tools.sh`。机械问题（格式、命名、拼写、复杂度阈值）由 lint 自动抓，**本 skill 不重复**。

## 第二步：lint 抓不到的语义问题

### 指针与内存（最高频 PR 问题）

```
✗ auto out = node->cast<CNodePtr>(); out->func_graph();         // cast 结果未校验
✗ delete ptr;                                                   // 悬空指针（ptr 仍可被读到）
✗ new T[n]; if (fail) { clear(); return; }                      // 已分配项泄漏

✓ MS_EXCEPTION_IF_NULL(node);                                   // 使用前校验
✓ auto out = node->cast<CNodePtr>(); MS_EXCEPTION_IF_NULL(out); // cast 后也校验
✓ delete ptr; ptr = nullptr;                                    // delete 后置空
✓ for (j = 0; j < i; j++) { delete arr[j]; arr[j] = nullptr; } // 部分失败时逐项清理 + 置空
✓ auto p = std::make_unique<T>();                               // RAII 优先于裸 new/delete
```

所有权模型：独占 `unique_ptr` / 共享 `shared_ptr`（仅必要时）/ 观察裸指针或引用 / 转移 `std::move`。

### 边界与类型安全

```
✗ if (scope_idx > MAX) { ... }       // 应为 >=（off-by-one）
✗ if (rank_size == 0) { ... }        // rank_size 为 int32_t 时应为 <= 0（防御负数）
✗ broadcast_ = a.ElementsNum() != b.ElementsNum();  // 应比较 shape，元素数相同但 shape 不同也会广播
✗ size_t idx = width - 1 - bit_pos;  // bit_pos > width 时无符号下溢
✗ int32_t r = int64_val;             // 隐式截断，应先判断范围或用 Narrow 操作

✓ if (scope_idx >= MAX) { ... }
✓ if (rank_size <= 0) { ... }        // 注意：若 rank_size 是 size_t，<= 0 与 == 0 等价，规则失效
✓ broadcast_ = (a.shape() != b.shape());
✓ MS_CHECK_GT(vec.size(), 0, RET_ERROR);
✓ if (SIZE_MUL_OVERFLOW(a, b)) return ERR; a *= b;  // 乘法前先查溢出
```

`SIZE_MUL_OVERFLOW` / `INT_MUL_OVERFLOW` 定义在 `src/litert/kernel/cpu/nnacl_c/op_base.h`。

### 逻辑正确性（Critical — 编译器不会报错，但行为完全错误）

```
✗ if (strcmp(name, target)) { return; }     // strcmp 匹配返回 0（false），这里逻辑反了
✗ for (j = 0; i < N; ++j) { ... }           // 循环变量写错（i 应为 j）
✗ return RET_OK;                            // error 路径上返回了 OK
✗ if (!empty) { LOG(ERROR) << "is empty"; } // 条件与日志消息自相矛盾
✗ if (status = OK) { ... }                  // 赋值不是比较（== 写成 =）

✓ if (strcmp(name, target) == 0) { return; }
✓ for (j = 0; j < N; ++j)
✓ return RET_ERROR;
✓ if (empty) { LOG(ERROR) << "is empty"; }
✓ if (status == OK) { ... }
✓ switch(x) { case A: ... case B: ... default: LOG(WARNING) << "unexpected"; }  // 别忘了 default
```

### 错误处理决策（返回码 vs 异常）

项目规范**禁用 C++ 异常机制，错误用返回码传递**。lite 全树实测：`MS_CHECK_*` 返回码 **7527 处** vs `MS_EXCEPTION_*` 抛异常 **1444 处**（~5:1）——**默认返回码**。

| 场景 | 选什么 | 头文件 |
|------|--------|--------|
| 默认（含致命/不变量违反） | `MS_CHECK_TRUE_MSG(cond, errcode, msg)` / `MS_CHECK_TRUE_RET(...)` | `src/common/log_util.h`（lite 自带） |
| 可恢复 + 打日志 | `MS_CHECK_TRUE_MSG` / `MS_CHECK_FALSE_MSG` | 同上 |
| 可恢复 + 静默返回 | `MS_CHECK_TRUE_RET` / `MS_CHECK_FALSE_RET` | 同上 |
| 数值边界 | `MS_CHECK_GT(a, b, errcode)` / `MS_CHECK_LT` / `MS_CHECK_LE` | 同上 |
| 仅当返回码无法传递（构造函数、运算符、不可恢复） | `MS_EXCEPTION_IF_NULL(ptr)` / `MS_EXCEPTION(...)` | `utils/log_adapter.h`（mindspore core） |

注意：

- `MS_EXCEPTION_*` **抛异常偏离项目规范**。新代码默认 `MS_CHECK_*`；仅返回码无法传递（构造/运算符）或不可恢复致命错误时用 `MS_EXCEPTION_*`，PR 须标注偏离。端侧 C API 性能敏感路径慎用。
- `MS_EXCEPTION_IF_NULL` 在 `utils/log_adapter.h`（不在 lite 自家头）；`assert`/`MS_ASSERT` release 下被移除，不能用于外部输入校验或 error path 返回。

### C++ 接口约定（编译器只强制一部分，但属契约）

- 虚函数重写必须加 `override`（编译器只在写了 override 时才校验签名匹配，漏写 = 静默隐藏父类方法）
- 单参数构造函数必须加 `explicit`（否则允许隐式转换，常引发意外的临时对象构造）
- Lambda 禁止默认捕获 `[=]` / `[&]`——显式列出捕获变量（lite src 实测 0 处使用 `[=]`/`[&]`）
- 不修改对象状态的成员方法加 `const`（const 正确性一旦破坏会传染，改起来代价大）
- 成员变量全部加类内默认初始化器 `int size_{0};`，构造函数初始化列表顺序与声明顺序一致

### 代码结构与三类硬指标

| 指标 | 阈值 | 工具 | 语言 |
|------|------|------|------|
| 圈复杂度 CCN | ≤ 19 | lizard | C++/Python/Java |
| 函数长度 NLOC | ≤ 50 | lizard + pylint `max-statements=50` | 全部 |
| 函数入参个数 | ≤ 5 | pylint `max-args=5`（Python 强制）；C++/Shell/CMake 推荐同阈值 | 全部 |

入参超 5 个**用参数对象**（不是把第 6 个参数硬塞）。CCN/NLOC 超阈值的处理见 [guidelines 白名单管理](clean-code-guidelines.md#附录复杂度白名单管理)。常用降复杂度模式：

```
嵌套 3 层 → 合并条件:
  if (a) { if (b) { if (c) { ... } } }   →   if (a && b && c) { ... }

深层 else → 反转条件提前返回:
  if (ok) { 30 行 } else { throw; }      →   if (!ok) { throw; return; } 30 行

重复代码 → 提取函数:
  30 行 if-else 出现 2 次              →   void PrintByType(Type t, void* d, size_t n) { ... }

超大函数 → 按职责拆分:
  FindProviderKernel(60 行)            →   FindCustomKernel() + FindGeneralKernel()
```

**什么时候不要硬拆**：数据驱动的 switch / map 查找（30 个算子注册、opcode 派发表）CCN 高是业务本质复杂，硬拆成 30 个函数更难读。两种处理方式：

- 重构为表驱动 + 通用 handler（仍是 1 个函数，CCN 大幅下降）
- 实在无法表驱动时，加入 `.jenkins/check/config/whitelizard.txt`，并在 PR 说明中解释为何加入

### 宏与头文件约定

- 宏定义末尾**不**加分号，分号在调用处（否则展开后多一个空语句）
- 对应 `.h` 头文件作为第一个 `#include`（保证头文件自洽）
- 不跨 `.so` 边界的函数**不**加 `FRONTEND_EXPORT`（污染符号表）
- 头文件保护用 `#pragma once`（lite 主流，无命名冲突风险）
- 能用前置声明 `class Foo;` 就不要 `#include "foo.h"`（降低编译依赖、加快构建）

### 魔法数字与字面量

```
✗ if (mode == 3) { ... }                       // 3 是什么模式？要翻文档/协议
✗ buf[7] = 0;                                  // 7 是偏移？长度？协议字段？
✗ auto v = std::get<2>(tuple);                 // 2 是哪个字段？
✗ constexpr int kMaxDim = 4;  // ... 分散在各函数体内，改一处漏一处

✓ constexpr int kModeStrict = 3; if (mode == kModeStrict) { ... }
✓ constexpr size_t kHeaderLen = 7; buf[kHeaderLen] = 0;
✓ constexpr size_t kIdxValueField = 2; auto v = std::get<kIdxValueField>(tuple);
✓ // 命名常量集中在匿名 namespace / class 静态成员里
```

例外（裸数字可接受）：`0` / `1`（循环起点、初值、布尔语义）、循环步长、单位已在变量名（`timeout_ms = 1000`）、测试 / 示例代码。C++ 用 `constexpr` 不用 `#define`（类型/作用域/调试友好），Python 用模块级 `UPPER_SNAKE_CASE`。

## 第三步：安全编码（涉及外部数据时）

只在以下场景需要查：文件（含配置）、网络、环境变量、命令行、用户输入、IPC（管道/消息/共享内存/socket/RPC）、API 函数参数、跨线程全局变量。深度示例、冲突取舍见 [clean-code-guidelines.md 安全编码章](clean-code-guidelines.md#安全编码)。

### 注入类（最高危）

```cpp
✗ system(("ls " + user_dir).c_str());        // 命令分隔符注入
✗ popen(user_input.c_str(), "r");
✗ dlopen(user_path.c_str(), RTLD_NOW);       // 加载攻击者预制模块
✗ sql = "SELECT ... WHERE name='" + name + "'";  // SQL 注入
✓ execv("/bin/ls", argv);                    // 数组参数不经 shell
✓ if (!InWhitelist(cmd)) return RET_ERROR;   // 白名单
✓ stmt = "...WHERE name=?"; bind(name);      // 参数化查询
```

### 文件操作

```cpp
✗ open(path, O_CREAT | O_WRONLY);                        // 缺权限位，umask 默认可能过宽
✗ if (access(path, W_OK)==0) fopen(path,"w+");           // TOCTOU 竞态
✓ open(path, O_CREAT | O_WRONLY, S_IRUSR|S_IWUSR);
✓ fd = open(path, O_CREAT|O_EXCL|O_WRONLY, mode);        // 原子创建防竞态
```

（realpath 校验、临时文件位置等见 guidelines 安全章。）

### 整数与索引（外部输入）

```cpp
✗ int a = data >> 24;                        // data 有符号，符号位扩展未定义
✗ uintptr_t n = (unsigned int)ptr;           // 指针转 int 丢高位
✗ for (i=0; i<ext_count; ++i){...}           // ext_count 来自报文，死循环/溢出
✗ memcpy(dst, src, ext_len);                 // ext_len 外部，缓冲区溢出
✓ unsigned a = (unsigned)data >> 24;
✓ if (ext_count==0 || ext_count>MAX) return RET_ERROR;
✓ if (off<0 || off>=size) return RET_ERROR;
```

（乘法溢出 `SIZE_MUL_OVERFLOW` 见上「边界与类型安全」节；realpath / 临时文件位置见 guidelines 安全章。）

### 内存申请

```cpp
✗ char *p = malloc(ext_size);                // ext_size 未校验（0 或巨大）
✗ int *r = malloc(n*sizeof(int)); r[0]+=..;  // 未初始化就读
✓ if (ext_size==0 || ext_size>MAX) return RET_ERROR;
✓ p = malloc(ext_size); if(!p) return RET_ERROR;
✓ // calloc / memset_s 清零后再读，或 std::vector
```

### 内存操作用安全函数

```cpp
✗ memcpy/strcpy/sprintf/memset（裸函数，无 destMax、不查返回值）
✓ if (memcpy_s(dst, dstMax, src, len) != EOK) return RET_ERROR;  // securec.h
```

裸函数例外（固定长度数组初始化、堆分配后赋初值、等大复制、静态字符串常量）见 guidelines 安全章。

### 敏感信息

```cpp
✗ memset(pwd, 0, len);                       // 可能被编译器优化掉
✗ srand(time(nullptr)); token = rand();      // 伪随机可预测
✗ string pwd = GetPassword();                // 散落内存无法可靠清零
✓ (void)memset_s(pwd, len, 0, len);          // 安全函数不被优化
✓ // /dev/urandom 或加密随机源
✓ char pwd[MAX]; ... (void)memset_s(pwd, sizeof(pwd), 0, sizeof(pwd));
```

### 禁用机制

```
✗ setjmp / longjmp          // 跨函数跳转，资源不清理
✗ 信号处理例程调 fprintf/malloc 等非异步安全函数
✗ realloc(ptr, size)        // malloc/free/realloc 三合一行为二义
✗ alloca(n)                 // 栈分配可溢出栈边界
```

错误处理（异常）见上「错误处理决策」——项目规范：错误用返回码传递，禁抛异常。

## PR 软审查清单

30 秒清单的补充项——非机械、靠经验：

1. **SOLID / 迪米特法则**——一个类不该直接伸手进另一个类的内部。反例：`tensor->GetCNode()->GetPrimitive()->GetAttr("name")` 一条链扒穿 3 层。正例：让 `Tensor` 暴露 `GetName()`，调用方只跟直接对象对话。
2. **外部接口变更**——API 签名、参数含义、返回格式变了，须同步改文档。
3. **行为变更**——即使接口不变，行为变了（如默认值、错误码含义）也要更新文档。

## 详细参考

需要更多代码示例、CI 工具配置、安全编码深度内容时，查：

- [clean-code-guidelines.md](clean-code-guidelines.md) -- CI 工具配置/阈值、设计原则、错误处理决策、安全编码、按主题 interleaved 的规则与 PR 案例库（10 类、100+ PR 模式）
