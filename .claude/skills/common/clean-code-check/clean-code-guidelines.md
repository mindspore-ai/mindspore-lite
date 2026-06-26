# MindSpore Lite Clean Code 代码规范

本文档是 clean-code-check skill 的深度参考：CI 工具配置、设计原则、错误处理决策、安全编码，以及按主题 interleaved 的规则与 PR 案例。

## 文档导航

- 想快速过规则、做 code review → [SKILL.md](SKILL.md)
- CI 工具配置、阈值、白名单 → [本文 CI 工具配置](#ci-工具配置)
- 安全编码→ [本文安全编码章](#安全编码)
- 某主题的规则 + 具体 ✗/✓ PR 案例 → [本文语言规范与 PR 案例](#语言规范与-pr-案例)
- 贡献流程 → 仓库根 [CONTRIBUTING_CN.md](/CONTRIBUTING_CN.md)

## 目录

- [CI 工具配置](#ci-工具配置)
- [设计原则](#设计原则)
- [错误处理决策](#错误处理决策)
- [安全编码](#安全编码)
- [语言规范与 PR 案例](#语言规范与-pr-案例)
  - 指针与内存安全 / 边界与类型安全 / 逻辑正确性 / 代码结构与复杂度 / C++ 风格与现代特性 / 布尔与条件简化 / 死代码与冗余清理 / Python 专项规范 / 构建、日志与格式化
- [附录：复杂度白名单管理](#附录复杂度白名单管理)

## PR 代码审查清单

每个 PR 需通过以下审查项：

1. **返回值校验** -- 禁止将安全函数的返回值强制转换为 `void`。
2. **SOLID 原则 / 迪米特法则** -- 一个类不该直接伸手进另一个类的内部。反例：`tensor->GetCNode()->GetPrimitive()->GetAttr("name")` 一条链扒穿 3 层。正例：让 `Tensor` 暴露 `GetName()`，调用方只跟直接对象对话。
3. **单元测试覆盖** -- 每次变更必须包含有效的测试用例。
4. **外部接口变更** -- 必须有文档记录。
5. **文档更新** -- 行为变更时需同步更新官方文档。

## CI 工具配置

代码质量通过 CI 流水线（Jenkins "code check" 门禁）和本地 `scripts/pre_commit/githooks/pre-push` 钩子强制执行。CI 按顺序执行 8 项检查：

| 序号 | 工具 | 语言 | 用途 |
|------|------|------|------|
| 1 | clang-format | C/C++ | 代码格式化 |
| 2 | cmakelint | CMake | CMake 脚本风格 |
| 3 | codespell | 所有语言 | 拼写错误检测 |
| 4 | cpplint | C/C++ | Google C++ 风格检查 |
| 5 | lizard | C++、Python、Java | 圈复杂度和函数长度 |
| 6 | pylint | Python | PEP 8 风格和质量检查 |
| 7 | shellcheck | Shell | Shell 脚本静态检查 |
| 8 | tab 检查 | 所有语言 | 禁止使用 Tab 字符 |

### 通用规则（行宽/Tab/排除目录）

#### 行长度

- **最大行长度：120 字符**，适用于所有语言（C++、Python、CMake）。
- cpplint（`--linelength=120`）、pylint（`max-line-length=120`）、cmakelint（`--linelength=120`）统一执行此限制。

#### Tab 字符

- **所有源文件中禁止使用 Tab 字符**，统一使用空格缩进。
- 例外：`Makefile` 文件（语法要求必须使用 Tab）在检查中被排除。

#### 复杂度阈值（Lizard）

函数不得超过以下阈值：

| 指标 | 阈值 | 说明 |
|------|------|------|
| CCN（圈复杂度） | <= 19 | 代码中独立执行路径的数量 |
| NLOC（非注释代码行数） | <= 50 | 不含注释和空行的代码行数 |

收到 Lizard 告警时，**必须**在合入前重构代码。仅在极少数情况下允许加入白名单，详见[附录：复杂度白名单管理](#附录复杂度白名单管理)。

#### 排除目录

以下目录不参与代码检查：
- `third_party/` -- 第三方依赖
- `tests/` 和 `test/` -- Lizard 分析中排除

### clang-format

所有 C/C++ 代码必须按照项目根目录下的 `.clang-format` 文件（基于 Google 风格）格式化。关键配置：

| 配置项 | 值 |
|--------|-----|
| 列宽限制 | 120 |
| 缩进宽度 | 2 个空格 |
| 构造函数初始化列表缩进 | 4 个空格 |
| 指针对齐方式 | 右对齐（`int *p`） |
| 大括号风格 | 附加风格（K&R 风格） |
| 排序 include | 禁用 |
| 最大连续空行 | 1 |
| Tab 使用 | 禁止 |

本地执行命令：
```bash
# 检查所有文件
bash scripts/check_clang_format.sh -a

# 仅检查最近一次提交中修改的文件
bash scripts/check_clang_format.sh -l
```

### cpplint

基于 Google C++ 编码规范，以下规则**全局禁用**：

| 禁用规则 | 原因 |
|----------|------|
| `build/header_guard` | 项目使用自定义头文件保护宏 |
| `build/c++11` | 允许使用 C++11 及更高版本特性 |
| `build/include_what_you_use` | 允许前向声明 |
| `whitespace/indent_namespace` | 命名空间内容不缩进 |
| `whitespace/newline` | 不强制文件末尾空行 |
| `readability/casting` | C 代码（nnacl、OpenCL）需要 C 风格类型转换 |

**按文件豁免**记录在 `.jenkins/check/config/filter_cpplint.txt`，会随代码演进。需要查最新豁免列表时直接看该文件。

### CppCheck

cpplint 之外的静态分析工具。豁免记录在 `.jenkins/check/config/filter_cppcheck.txt`，会随代码演进，以仓库为准。常见豁免场景包括：

- `useStlAlgorithm` -- 为性能保留手写循环
- `knownConditionTrueFalse` -- 卷积 kernel 中条件路径优化
- `shadowVariable` -- kernel 实现中内层循环变量遮蔽
- `nullPointerRedundantCheck` -- Fusion pass 中的显式空指针检查
- 其他详见上述文件

### pylint

配置文件：`.jenkins/rules/pylint/pylintrc`

**关键阈值：**

| 参数 | 值 |
|------|-----|
| 最大行长度 | 120 |
| 最大参数个数 | 5 |
| 最大局部变量数 | 15 |
| 最大返回语句数 | 6 |
| 最大分支数 | 12 |
| 最大语句数 | 50 |
| 最大父类数 | 7 |
| 最大属性数 | 7 |
| 最少公共方法数 | 2 |
| 最大公共方法数 | 20 |
| 缩进字符串 | 4 个空格 |

**全局禁用检查：**

| 禁用检查项 | 原因 |
|------------|------|
| `design` | 设计指标由 lizard 处理 |
| `similarities` / `duplicate-code` | 模型定义中容易误报 |
| `no-self-use` | API 封装类中常见 |
| `no-member` / `no-name-in-module` | C++ 绑定模块对 pylint 不可见 |
| `import-error` | Pybind11 模块无法静态导入 |
| `consider-using-f-string` | 向后兼容考虑 |
| `import-outside-toplevel` | 为性能使用延迟导入 |
| `broad-exception-raised` | 错误处理惯例 |

**按文件豁免**记录在 `.jenkins/check/config/filter_pylint.txt`，会随代码演进，以仓库为准。

### shellcheck

Shell 脚本在 **warning** 严重级别进行检查，以下检查项已排除：

| 排除项 | 编码 | 原因 |
|--------|------|------|
| 双引号防止通配和分词 | SC2086 | 构建脚本中有意使用分词 |
| 无法跟踪非常量 source | SC1090 | CI 脚本中的动态加载 |
| 未跟踪被 source 的文件 | SC1091 | CI 环境文件在本地不可用 |
| 声明和赋值应分开 | SC2155 | 构建脚本中的常见模式 |
| 使用 `cd ... \|\| exit` | SC2164 | 错误处理在其他地方完成 |
| 转义单引号 | SC1003 | 模式中 `'` 的误报 |

**常见规范：**

- 所有脚本使用 `bash` shebang（`#!/bin/bash`）。
- 引用变量展开，除非有意使用分词。
- 关键脚本使用 `set -e` 或显式错误检查。

### cmakelint

配置：`--spaces=2 --linelength=120`

**启用的规则：**

| 规则 | 说明 |
|------|------|
| `convention/filename` | CMake 文件命名规范 |
| `linelength` | 120 字符行宽限制 |
| `package/consistency` | `find_package` 使用一致性 |
| `readability/logic` | 逻辑结构 |
| `whitespace/eol` | 行尾空白 |
| `whitespace/extra` | 多余空白 |
| `whitespace/indent` | 缩进（2 个空格） |
| `whitespace/mismatch` | 空白不一致 |
| `whitespace/newline` | 文件末尾换行 |
| `whitespace/tabs` | 禁止 Tab |
| `syntax` | CMake 语法错误 |

**禁用的规则：**

| 规则 | 原因 |
|------|------|
| `readability/mixedcase` | 允许 CMake 变量名使用混合大小写 |
| `readability/wonkycase` | 允许非标准大小写 |

### MarkdownLint

基于默认 MarkdownLint 规则，以下为项目特定覆盖：

| 规则 | 覆盖值 | 说明 |
|------|--------|------|
| MD007 | `indent=4` | 无序列表缩进：每级 4 个空格 |
| MD009 | `br_spaces=2` | 行尾空白：允许 0 或 2 个空格 |
| MD029 | `style=ordered` | 有序列表编号必须递增（1, 2, 3...） |

### codespell

检查源代码和注释中的常见拼写错误。自定义允许词列表定义在 `.jenkins/rules/codespell/codespell.allow`，会随代码演进，以仓库为准。

另有 `sensitive.allow` 文件包含因敏感性原因排除的词汇。

## 设计原则

这些原则源自 Google C++ Style Guide、C++ Core Guidelines、CERT C++、MISRA C++、PEP 8 和 OWASP 安全编码实践。它们通过代码评审而非自动化工具来执行。

### RAII -- 资源获取即初始化

所有资源（内存、文件句柄、锁、设备内存）必须通过 RAII 包装器管理。禁止依赖手动 `new`/`delete` 或显式 `close()`。

```cpp
// 错误：手动所有权管理，容易泄漏
auto *tensor = new Tensor();
MS_EXCEPTION_IF_NULL(tensor);
process(tensor);
// ... 必须在每条退出路径上记得 delete

// 正确：通过智能指针实现 RAII
auto tensor = std::make_unique<Tensor>();
process(tensor.get());
// 自动清理，不可能泄漏
```

```python
# 错误：无保证的清理
f = open(path, "r")
data = f.read()
# 如果 read() 抛异常，文件句柄泄漏

# 正确：上下文管理器
with open(path, "r") as f:
    data = f.read()
# 保证清理
```

**智能指针所有权模型：**

| 所有权 | 使用方式 | 示例 |
|--------|----------|------|
| 独占 | `std::unique_ptr` | `auto p = std::make_unique<T>(args)` |
| 共享 | `std::shared_ptr`（仅真正需要时） | `auto p = std::make_shared<T>(args)` |
| 非持有观察者 | 裸指针或引用 | `T *p = ptr.get()` |
| 转移所有权 | `std::move(ptr)` | `sink(std::move(src))` |

不要仅仅为了传递对象而使用 `std::shared_ptr`。通过 `std::unique_ptr` 转移所有权，通过裸指针或引用观察。

### 虚函数重写标注 `override` 和 `final`

重写虚函数时必须标注 `override`。不再允许进一步重写的类或方法使用 `final`。

```cpp
// 错误：无 override -- 基类签名改了这里静默失效
class Derived : public Base {
  void Process(int x) { ... }  // 笔误：基类是 Process(long x)
};

// 正确：编译器捕获签名不匹配
class Derived : public Base {
  void Process(int x) override { ... }  // 编译错误：无匹配基类方法
};
```

### 单参数构造函数加 `explicit`

所有单参数构造函数和转换运算符必须标注 `explicit`，防止隐式转换。

```cpp
// 错误：隐式转换
class Size { Size(int s) : val(s) {} };
void process(Size s);
process(42);  // 静默创建 Size(42)

// 正确：显式
class Size { explicit Size(int s) : val(s) {} };
process(Size(42));  // 必须显式构造
```

### 异常安全

1. **绝不全吞异常。** 每个 `catch` 块必须做可见操作（记录日志、重新抛出或转换异常）。
2. **用 RAII 而非 try/catch 做资源清理。** 使用析构函数和智能指针，而非 catch 块中的手动清理。
3. **析构函数绝不抛异常。** 析构函数必须是 `noexcept`。
4. **使用特定异常类型。** 除非立即重新抛出，否则禁止 `catch (...)`。

```cpp
// 错误：静默吞掉异常
try {
  DoSomething();
} catch (...) {
  // 什么都不做 -- bug 不可见
}

// 正确：至少记录日志
try {
  DoSomething();
} catch (const std::exception &e) {
  MS_LOG(ERROR) << "DoSomething failed: " << e.what();
  throw;  // 如果调用方应处理则重新抛出
}
```

```python
# 错误：裸 except，静默吞掉
try:
    process(data)
except:
    pass  # 掩盖所有错误包括 SystemExit

# 正确：特定异常，可见操作
try:
    process(data)
except ValueError as e:
    logger.error("Invalid data: %s", e)
    raise
```

### `volatile` 不用于线程通信

`volatile` 不提供原子性、内存序或同步。跨线程数据必须使用 `std::atomic`。

```cpp
// 错误：volatile 不保证线程安全
volatile int counter_;

// 正确：正确的原子操作
std::atomic<int> counter_;
```

### Lambda 捕获规则

禁止使用默认捕获模式（`[=]`、`[&]`）。按名称显式捕获变量，使所有权和生命周期清晰。

```cpp
// 错误：默认捕获 -- 不清楚捕获了什么
auto callback = [=, this]() { use(x, y); };

// 正确：显式捕获
auto callback = [this, &x, y]() { use(x, y); };
```

### `switch` 必须有 `default`

每个 `switch` 语句必须以 `default` 标签结束，即使看起来已覆盖所有枚举值。这能捕获未来新增的值。

```cpp
// 错误：无 default
switch (type) {
  case Type::INT: return "int";
  case Type::FLOAT: return "float";
}

// 正确：default 处理意外值
switch (type) {
  case Type::INT: return "int";
  case Type::FLOAT: return "float";
  default:
    MS_LOG(WARNING) << "Unexpected type: " << static_cast<int>(type);
    return "unknown";
}
```

### 构造/析构函数中不调用虚函数

构造或析构期间虚分派不起作用。虚表指向当前类而非派生类。

```cpp
// 错误：构造函数中的虚调用
class Base {
  Base() { Init(); }  // 调用 Base::Init，永远不会调到 Derived::Init
  virtual void Init() { ... }
};

// 正确：使用显式两阶段初始化或非虚辅助函数
class Base {
  Base() { InitImpl(); }  // 非虚函数
  void InitImpl() { ... }
  virtual void OnInit() { ... }  // 构造完成后调用
};
```

## 错误处理决策

什么时候抛异常、什么时候返回错误码？**禁用 C++ 异常机制，错误用返回码传递**。lite 全树实测：`MS_CHECK_*` 返回码 **7527 处** vs `MS_EXCEPTION_*` 抛异常 **1444 处**（~5:1），返回码已是主流。

### 决策表

| 场景 | 选什么 | 头文件 |
|------|--------|--------|
| 默认（含致命/不变量违反） | `MS_CHECK_TRUE_MSG(cond, errcode, msg)` / `MS_CHECK_TRUE_RET(...)` | `src/common/log_util.h`（lite 自带） |
| 可恢复 + 打日志 | `MS_CHECK_TRUE_MSG` / `MS_CHECK_FALSE_MSG` | `src/common/log_util.h` |
| 可恢复 + 静默返回 | `MS_CHECK_TRUE_RET` / `MS_CHECK_FALSE_RET` | `src/common/log_util.h` |
| 数值边界 | `MS_CHECK_GT(a, b, errcode)` / `MS_CHECK_LT` / `MS_CHECK_LE` | `src/common/log_util.h` |
| 仅当返回码无法传递（构造函数、运算符重载、不可恢复） | `MS_EXCEPTION_IF_NULL(ptr)` / `MS_EXCEPTION(...)` | `utils/log_adapter.h`（mindspore core） |

### 注意事项

- **禁异常**：`MS_EXCEPTION_*` 抛异常，是偏离项目规范的做法。新代码默认用 `MS_CHECK_*` 返回码；仅当返回码无法传递（构造函数内、运算符重载）或属不可恢复致命错误时用 `MS_EXCEPTION_*`，并在 PR 说明中标注为偏离。
- `MS_EXCEPTION_*` 异常展开有开销，端侧 C API（libmindspore-lite）性能敏感路径慎用；嵌入式场景可能禁用异常。
- `MS_EXCEPTION_IF_NULL` 不在 lite 自家头里，使用前确认已 `#include "utils/log_adapter.h"`。
- `assert` / `MS_ASSERT` 在 release 下被移除（`src/common/log.h:157-159`，非 Debug 展开为 `((void)0)`），不能用于外部输入校验或 error path 返回——见 [安全编码·ASSERT](#assert)。

### 示例

```cpp
// 默认：可恢复错误用 MS_CHECK_* 返回码
MS_CHECK_TRUE_MSG(tensor != nullptr, RET_ERROR, "tensor is null");
MS_CHECK_GT(tensor->ElementsNum(), 0, RET_ERROR);
MS_CHECK_FALSE_MSG(SIZE_MUL_OVERFLOW(size, count), RET_ERROR, "mul overflow");

// 仅当返回码无法传递（如构造函数内、运算符内）才用 MS_EXCEPTION_*
MS_EXCEPTION_IF_NULL(node);
MS_EXCEPTION(ValueError) << "shape mismatch";
```

## 安全编码

本章是 [SKILL.md 第三步](SKILL.md#第三步安全编码外部数据--攻击者思维) 的深度配套。### 攻击者思维与外部数据

核心假设：**程序处理的所有外部数据都是不可信的攻击数据**。以下来源一律视为外部数据，使用前必须严格校验：

- 文件（含配置文件）、注册表、网络、环境变量、命令行、用户输入
- 用户态数据（对内核程序）、进程间通信（管道、消息、共享内存、socket、RPC）
- 函数参数（对 API 而言）
- 全局变量（在本函数内，其他线程会修改）

安全编码基本思想：(1) 处理外部数据必须经严格合法性校验，不对外部数据做任何符合预期的假设；(2) 尽量减小代码攻击面，避免与外部环境多余数据交互；(3) 防御性编码弥补疏忽（变量赋初值、谨慎全局变量、禁用易错函数/机制、严格错误处理、合理用 ASSERT）。

### 安全规则总览

| 主题 | 子项 | 处理章节 |
|------|------|---------|
| 主题 | 子项 | 处理章节 |
|------|------|---------|
| 变量 | 初值 / 释放后置新值 / 成员初值 | [指针与内存安全](#指针与内存安全) / [C++ 风格](#c-风格与现代特性) |
| 变量 | 全局变量跨线程竞争 / 局部变量空间不过大 | [变量补充](#变量补充) |
| 断言 | 宏定义 / 运行时错误禁断言 / 禁改环境 / 单条件单断言 | [ASSERT](#assert) |
| 函数 | 数组配长度 / API 禁 ASSERT / const 指针 / 不可重入 / NULL 检查 / 入参 ≤ 5 | [函数](#函数) |
| 循环 | 必须有退出条件 | [循环退出条件](#循环退出条件) |
| 异常 | 禁用 C++ 异常机制（偏离待收敛） | [错误处理决策](#错误处理决策) |
| 类 | 有构造必有析构 / 构造函数限制 / 禁 delete this / 公共接口返回私有地址 const / 避免 public 成员 | [内存](#内存) |
| 安全退出 | 禁 atexit / kill / pthread_exit / exit / abort | [禁用机制与安全退出](#禁用机制与安全退出) |
| 字符串/数组 | 存储空间 / '\0' 结束符 / 索引校验 / 复制长度校验 | [字符串与数组](#字符串与数组) / [整数与索引](#整数与索引) |
| 格式化 | format 参数禁外部可控 / 类型个数一致 | [整数与索引](#整数与索引) |
| 整数 | 溢出/反转/除0 / 类型提升 / 禁位运算有符号 / 禁 int↔指针 / 禁指针位运算 / 外部循环次数 | [边界与类型安全](#边界与类型安全) / [整数与索引](#整数与索引) |
| 内存 | 申请前校验 / 分配判空 / 禁未初始化 / 释放后置新值 / 禁 realloc / 禁 alloca / 禁 sizeof 指针 | [内存](#内存) / [指针与内存安全](#指针与内存安全) |
| 安全函数 | destMax 准确 / 查返回值 / 禁封装重命名自定义 | [安全函数](#安全函数) |
| 不安全函数 | 命令/模块/SQL 注入 / 信号异步安全 / setjmp/longjmp / 危险内存函数改用 _s | [注入类](#注入类命令模块sql) / [禁用机制与安全退出](#禁用机制与安全退出) / [安全函数](#安全函数) |
| 文件 | 显式权限 / 路径规范化 / 临时文件 / TOCTOU | [文件操作](#文件操作) |
| 敏感信息 | 禁 rand 安全随机 / 清零防优化 / 禁 std::string | [敏感信息](#敏感信息) |

> 原三处历史冲突（异常 / `_s` / `std::string`）全部已收敛到项目规范一侧。lite 现状数据：`MS_CHECK_*` 7527 vs `MS_EXCEPTION_*` 1444（返回码主流）；`_s` 函数现有 337 处（securec.h 可用）。


> 原三处历史冲突（异常 / `_s` / std::string）全部已收敛到项目规范一侧。lite 现状数据：`MS_CHECK_*` 7527 vs `MS_EXCEPTION_*` 1444（返回码主流）；`_s` 函数现有 337 处（securec.h 可用）。

### 注入类（命令/模块/SQL）

#### 命令注入

禁外部可控数据作 `system` / `popen` / `execl` / `execlp` / `execle` / `execv` / `execvp` / `CreateProcess` 等进程启动函数参数。命令分隔符（`;` `|` `&` `` ` `` `$` `(``)`）即便拼接仍可注入。

```cpp
✗ system(user_input.c_str());                       // 直接注入
✗ sprintf(cmd, "ls %s", user_dir); system(cmd);     // 拼接仍有分隔符注入
✓ execv("/bin/ls", const_cast<char* const*>(argv)); // 数组参数不经 shell
✓ if (!InWhitelist(cmd)) return RET_ERROR;          // 白名单
```

Linux/Unix 建议用 `execv` 系列，且 `path`/`file` 禁用命令解析器（`/bin/sh`）。Windows `CreateProcess` 注意 `lpApplicationName` 空格（8dot3 或转义），禁 `cmd`/`powershell` 解析器。命令注入特殊字符见仓库根目录的《C&C++ 安全编程规范》附录 B。

#### 模块加载注入

禁外部可控数据作 `dlopen` / `LoadLibrary` 参数，防加载攻击者预制模块。用白名单或签名校验。

```cpp
✗ dlopen(user_path.c_str(), RTLD_NOW);
✓ if (!InWhitelist(path)) return RET_ERROR;
✓ dlopen(path.c_str(), RTLD_NOW);
```

例外：使用密钥/签名机制保护的动态模块，完整性有保证时可例外。

#### SQL 注入

禁外部数据拼接 SQL。优先参数化查询（预处理语句）；其次白名单校验；或转义 SQL 特殊字符。

```cpp
✗ sql = "SELECT ... WHERE name='" + name + "'";
✓ stmt = "SELECT ... WHERE name=?";
✓ mysql_stmt_bind_param(stmt, params);  // 参数化
```

### 文件操作

#### 文件创建显式权限

创建文件须显式指定访问权限，否则用 umask 默认可能过宽，让未授权用户可访问。

```cpp
✗ int fd = open(path, O_CREAT | O_WRONLY);                    // 缺权限位
✓ int fd = open(path, O_CREAT | O_WRONLY, S_IRUSR | S_IWUSR);
```

#### 路径规范化（已覆盖）

外部路径须 `realpath`（Linux）/ `PathCanonicalize`（Windows）规范化后再校验合法性，防 `../../../etc/passwd` 穿越访问。精简规则见 [SKILL.md 第三步](SKILL.md#第三步安全编码外部数据--攻击者思维)；C++ 用 `FileUtils::GetRealPath`，Python 用 `os.path.realpath`，Java 用 `File.getCanonicalFile`。例外：命令行手工输入路径的控制台程序可例外。

#### 临时文件不进共享目录

不要在共享目录（如 `/tmp` 多用户可写）创建临时文件，防符号链接攻击与竞争。

#### 避免 TOCTOU

避免 `access()` + `open()` 的时间差竞态（access 与 open 之间攻击者可改文件属性）。

```cpp
✗ if (access(path, W_OK) == 0) { fp = fopen(path, "w+"); }   // 竞态窗口
✓ int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, mode);    // 原子创建
```

### 字符串与数组

#### 确保有足够存储空间

字符串/数组存储操作前须确保目标缓冲区有足够空间，避免缓冲区溢出。优先用 `_s` 安全函数（带 destMax）。

```cpp
✗ char buf[8]; strcpy(buf, user_input);                    // 未校验长度
✓ char buf[8];
✓ if (strlen(user_input) >= sizeof(buf)) return RET_ERROR;
✓ if (strcpy_s(buf, sizeof(buf), user_input) != EOK) return RET_ERROR;
```

#### 字符串须有 '\0' 结束符

对字符串做存储操作前须确保字符串有 `'\0'` 结束符。外部数据可能是定长字符数组（无 `'\0'`），直接 `strcpy` 会越界读取。

```cpp
✗ char buf[N]; read(fd, buf, N); strcpy(dst, buf);        // buf 无 '\0'，strcpy 越界读
✓ char buf[N];
✓ ssize_t n = read(fd, buf, N - 1); buf[n >= 0 ? n : 0] = '\0';
✓ if (strcpy_s(dst, dst_max, buf) != EOK) return RET_ERROR;
```

详细字符串安全函数（`strcpy_s` / `strncpy_s` / `sprintf_s`）见 [安全函数](#安全函数)。

### 整数与索引

外部数据作数组索引、内存复制长度、循环次数、位运算操作数时必须校验。整数运算防溢出/反转/除0。

#### 外部索引与复制长度

```cpp
✗ int off = ReadInt(); BYTE c = buf[off];      // off 未校验越界
✓ if (off < 0 || off >= size) return RET_ERROR;

✗ memcpy(dst, src, ext_len);                    // ext_len 外部可控
✓ if (ext_len > dst_size) return RET_ERROR;
✓ if (memcpy_s(dst, dst_size, src, ext_len) != EOK) return RET_ERROR;
```

#### 格式化函数

`format` 参数禁外部可控（`printf(msg)` 是格式化漏洞，用 `printf("%s", msg)`）；`format` 中参数类型与个数须与实际参数一致。

#### 溢出/反转/除0（部分已覆盖）

乘法溢出见 [边界与类型安全·操作前检查溢出](#边界与类型安全)（`SIZE_MUL_OVERFLOW`）。补除0：

```cpp
✗ size_t b = 1000 / a;          // a 可能为 0
✓ if (a == 0) return RET_ERROR;
✓ size_t b = 1000 / a;
```

#### 更大类型前先提升求值

整型表达式赋值/比较更大类型前，须先用更大类型求值，否则中间结果按原类型溢出。

```cpp
✗ unsigned long long b = a * 0xab;             // a 为 unsigned int，先按 int 乘再提升，溢出
✓ unsigned long long b = (unsigned long long)a * 0xab;
```

#### 禁有符号整数位运算

位运算（`~` `>>` `<<` `&` `^` `|`）只用于无符号操作数。

```cpp
✗ int a = data >> 24;          // data 有符号，符号位扩展行为未定义
✓ unsigned int a = (unsigned int)data >> 24;
```

#### 禁 int↔指针转换 / 禁指针位运算

指针大小随平台不同，整数与指针互转丢高位信息；对指针做逻辑/位运算改变指针性质，可致非法访问。

```cpp
✗ unsigned int n = (unsigned int)ptr;          // 丢高位
✓ uintptr_t n = (uintptr_t)ptr;

✗ if (nameA) ...                               // 指针做逻辑运算
✓ if (nameA != nullptr) ...
```

例外：地址对齐检查的位运算可例外。

#### 外部控制循环次数

受外部数据控制的循环须校验合法（非0、有界、增量非0），防死循环或缓冲区溢出。

```cpp
✗ for (i = 0; i < ext_count; ++i) { ... }      // ext_count 来自报文
✓ if (ext_count == 0 || ext_count > MAX_COUNT) return RET_ERROR;
✓ for (i = 0; i < ext_count; ++i) { ... }
```

### 内存

#### 禁 sizeof 指针

编码人员常误把指针当数组 `sizeof`，得到指针大小而非缓冲区大小。判断指针类型大小用 `sizeof(char *)`。

```cpp
✗ char *buf = malloc(size); memset(buf, 0, sizeof(buf));   // sizeof(buf) = 指针大小
✓ memset(buf, 0, size);                                    // 传实际长度
```

#### 构造函数限制 / 禁 delete this

- 构造函数没有返回值，不能做可能失败的操作（`open` / `new` / `ConnectServer` 等不放构造函数）。
- 严禁在构造函数中创建线程（构造函数仅初始化成员变量）。
- 严禁 `delete this`：资源申请与释放须在同一逻辑层，谁申请谁释放。

#### 有构造函数必有析构函数

定义了构造函数的类应显式声明析构函数（即便为 `=default`）。隐式生成的析构函数在成员未来扩展时易遗漏清理逻辑。

```cpp
✗ class Foo { public: Foo() {...} };                          // 无显式析构
✓ class Foo { public: Foo() {...}; ~Foo() = default; };       // 显式声明
```

#### 公共接口返回私有数据地址必须加 const

公共接口返回类私有数据地址（指针/引用）必须加 `const`，否则外部可绕过封装直接修改内部状态。

```cpp
✗ class Config {
    const char *name_;
   public:
    char *GetName() { return const_cast<char *>(name_); }     // 暴露内部可改
  };

✓ class Config {
    const char *name_;
   public:
    const char *GetName() const { return name_; }
  };
```

#### 尽量避免 public 成员

数据成员应 `private`/`protected`，通过 getter/setter 暴露。public 成员破坏封装，类不变量可被外部绕过。与 SOLID 中的封装原则一致。

#### 申请前校验大小

内存申请大小可能来自外部数据，须校验合法，不能申请 0 长度内存。

```cpp
✗ char *p = malloc(ext_size);                  // ext_size 未校验，可能 0 或巨大
✓ if (ext_size == 0 || ext_size > MAX_SIZE) return RET_ERROR;
✓ char *p = malloc(ext_size);
```

#### 禁读未初始化内存

`malloc` / `new` 分配的内存未初始化为 0，引用前须初始化。

```cpp
✗ int *r = malloc(n * sizeof(int)); r[0] += m[0];      // 未初始化就读
✓ int *r = malloc(n * sizeof(int));
✓ (void)memset_s(r, n * sizeof(int), 0, n * sizeof(int));  // 或用 calloc / std::vector
```

#### 判空与释放后置空（已覆盖）

分配后判空、释放后置新值见 [指针与内存安全](#指针与内存安全)。

#### 禁 realloc / 禁 alloca

- `realloc(ptr, size)` 行为二义（ptr/size 不同组合等同 malloc/free/realloc 三合一），易引入 bug，禁用。
- `alloca(n)` 申请栈内存，可超过栈边界导致 stack overflow，且 POSIX/C99 未定义，禁用。改用 `malloc` / `new` 从堆分配。

### 安全函数

#### 危险内存函数 → _s

C 标准的内存操作函数未将目标缓冲区大小作参数，未考虑内存重叠/非法指针，易引入缓冲区溢出。下列函数禁用，改用对应 `_s` 安全函数（`securec.h`，lite 已链接）：内存拷贝 `memcpy`/`memmove` → `memcpy_s`/`memmove_s`；内存初始化 `memset` → `memset_s`；字符串拷贝 `strcpy`/`strncpy` → `strcpy_s`/`strncpy_s`；字符串拼接 `strcat`/`strncat` → `strcat_s`/`strncat_s`；格式化输出 `sprintf`/`snprintf` → `sprintf_s`/`snprintf_s`；格式化输入 `scanf`/`sscanf`/`fscanf` → `_s` 变体；`gets` → `gets_s`。

```cpp
✗ memcpy(dst, src, len);
✓ if (memcpy_s(dst, dstMax, src, len) != EOK) return RET_ERROR;  // 带 destMax + 查返回值
```

**例外**（未涉及外部数据处理，内存操作在本函数内完成，可留用裸函数）：
1. 对固定长度数组初始化：`memset(arr, 0, sizeof(arr))`（arr 是 `BYTE g_array[N]` 或局部 `char buf[N]`）。
2. 函数参数中有表示内存的参数，对该内存初始化：`memset(buff, 0, len)`（buff 与 len 都是参数）。
3. 从堆分配后赋初值：`char *s = malloc(len); memset(s, 0, len);`。
4. 根据源内存大小进行等大复制：按 `srcSize` 分配等大 `dst`，`memcpy(dst, src, srcSize)`。
5. 源内存全部是静态字符串常量（编码时检查目标足够）：`strcpy(buf, "hello")`。

#### destMax 准确

安全函数的 `destMax` 须为目标缓冲区实际大小，准确、有效。

#### 禁封装/重命名/自定义安全函数

- 禁以宏/函数封装安全函数时忽略 `destMax` 或用 `count` 直接代替 `destMax`。
- 禁用宏重命名安全函数（不利静态扫描、易误解误用）。
- 禁自定义安全函数（与 C11 标准 实现混淆，引入风险）。

#### 查返回值

使用安全函数须查返回值。返回值 `!= EOK` 时本函数应立即返回，不能继续执行；可记日志 / 返回错误 / `abort`。例外：规则 6.6 例外场景对应代码若用了安全函数可不查返回值；错误处理代码内再次调用安全函数（如记日志的 `sprintf_s`）可不查。

### 敏感信息

#### 禁 rand 做安全随机

C 标准库 `rand()` 生成伪随机数，可预测，禁用于安全用途（如生成 token/密钥/会话 ID）。用 `/dev/urandom` 或加密随机源。

```cpp
✗ srand(time(nullptr)); token = rand();
✓ // 读 /dev/urandom 或用加密库的随机源
```

#### 敏感信息清零（防编译器优化）

口令、密钥等敏感信息使用完毕后立即清 0。普通 `memset` 可能被编译器优化掉（编译器判定无副作用而删除），须用 `memset_s`（不被优化）、Windows `SecureZeroMemory`、或 `#pragma optimize` 关闭优化。

```cpp
✗ memset(pwd, 0, sizeof(pwd));                          // 可能被编译器优化掉
✓ (void)memset_s(pwd, sizeof(pwd), 0, sizeof(pwd));     // 安全函数不被优化
```

#### 禁 std::string 存敏感信息

`std::string` 内部短字符串优化、拷贝、传参会将敏感信息散落到内存各处，且无法可靠清 0。敏感信息（口令/密钥/token）用 `char[]` + `memset_s`。

```cpp
✗ std::string pwd = GetPassword();                      // 散落内存，无法可靠清零
✓ char pwd[MAX_PWD_LEN] = {0};
✓ GetPassword(pwd, sizeof(pwd));
✓ // ... 使用 ...
✓ (void)memset_s(pwd, sizeof(pwd), 0, sizeof(pwd));
```

非敏感数据用 `std::string` 不受此规则限制。

### 禁用机制与安全退出

#### 安全退出

- **1.7.1 禁 `atexit`**：资源用后即主动清理，不靠程序退出时被动注册。例外：维测监控、定位异常退出原因的模块可例外。
- **1.7.2 严禁 `kill` / `TerminateProcess` 终止他进程**：会导致他进程资源不清理。进程间通信应先发停止命令，等待超时后才 `kill`。
- **1.7.3 禁 `pthread_exit` / `ExitThread`**：线程函数执行完毕后自动安全退出，禁主动终止自身线程（降低复用性、易资源泄漏）。
- **建议 1.7.1 禁 `exit` / `ExitProcess`**（`main` 除外）：直接退出进程降低复用性、资源不清理，应用错误值传递机制。
- **建议 1.7.2 禁 `abort`**：例外——仅发生致命错误、程序无法继续时在错误处理函数中使用。

#### 信号处理异步安全

信号处理例程应尽可能简化，只调用异步安全函数（见仓库根目录的《C&C++ 安全编程规范》附录 D）。禁在信号处理例程中调用 `fprintf` / `malloc` 等非异步安全函数（可能死锁或状态不一致）。

#### 禁 setjmp / longjmp

`setjmp` / `longjmp` 允许跨函数跳转，使程序复杂、资源得不到清理、不可重入、多线程不安全。禁用。

### 变量补充

#### 全局变量跨线程竞争

涉及多个线程访问的全局变量，须考虑竞争条件。普通 `int g_count;` 在多线程下 `++g_count` 是读-改-写三步，非原子。改用 `std::atomic<int>` 或加锁。

```cpp
✗ int g_request_count;          // 多线程 ++g_request_count 竞争
✓ std::atomic<int> g_request_count{0};  // 原子自增
```

与 [设计原则·`volatile` 不用于线程通信](#volatile-不用于线程通信) 同源——`volatile` 不保证原子性。

#### 同函数局部变量空间不要过大

同一函数内局部变量占用空间不要过大，避免栈溢出（嵌入式/端侧尤其敏感）。大缓冲用堆分配。

```cpp
✗ void Process() { char buf[1024*1024]; ... }   // 1MB 栈分配，可能溢出
✓ void Process() { auto buf = std::make_unique<char[]>(1024*1024); ... }   // 堆分配
```

### ASSERT

断言是除错机制，验证代码是否符合编码人员预期，只在调试版有效，发布版必须移除（`MS_ASSERT` 非 Debug 展开为 `((void)0)`，见 `src/common/log.h:157-159`）。

#### 断言须用宏定义

断言须用宏定义，通过编译选项控制仅在 Debug 生效。禁直接调用系统 `assert()`（无法按编译选项移除）。

#### 运行时可能错误禁断言

文件打开失败、内存分配失败、外部数据不符预期等运行时可能发生的错误，禁用断言——应用错误处理（返回码/异常）。

```cpp
✗ FILE *fp = fopen(path, "r"); ASSERT(fp != NULL);     // 文件可能打开失败
✗ char *s = malloc(len); ASSERT(s != NULL);            // 内存可能分配失败
✓ FILE *fp = fopen(path, "r");
✓ if (fp == NULL) { return RET_ERROR; }                // 错误处理
```

#### 禁断言内改环境

发布版断言被移除，为确保 Debug/release 功能一致，断言内禁任何赋值、修改变量、资源操作、内存申请。

```cpp
✗ ASSERT(p1 = p2);          // p1 被修改
✗ ASSERT(i++ > 1000);       // i 被修改
✗ ASSERT(close(fd) == 0);   // fd 被关闭
```

#### 单条件单断言

每条断言只校验一个条件，触发时能准确定位是哪个条件失败。

```cpp
✗ ASSERT(arr != NULL && size > 0 && size < MAX);
✓ ASSERT(arr != NULL);
✓ ASSERT(size > 0);
✓ ASSERT(size < MAX);
```

> lite 现状：`MS_ASSERT` 仅 Debug 生效，release 消失。外部输入校验、内存安全不变量保护、error path 返回，全部用会保留的宏（`MS_EXCEPTION_*` 或 `MS_CHECK_*`），不用 `MS_ASSERT`/`assert`。详见 [错误处理决策](#错误处理决策)。

### 函数

函数设计规则在 lite 普遍适用，但 [SKILL.md](SKILL.md) 主文档未独立列出。补在安全编码章。

#### 数组参数必须配长度

数组作为函数参数时必须同时传长度——函数内 `sizeof(arr)` 是指针大小（与 [禁 sizeof 指针](#禁-sizeof-指针) 同源），拿不到真实长度。

```cpp
✗ void Process(int arr[]) { for (size_t i=0; i<sizeof(arr)/sizeof(int); ++i) ... }  // sizeof(arr) = 指针大小
✓ void Process(const int *arr, size_t n) { for (size_t i=0; i<n; ++i) ... }
```

#### API 参数禁用 ASSERT

公共接口（导出 API、跨模块对外接口）的参数不能用 `ASSERT`/`MS_ASSERT` 校验——release 版 ASSERT 被移除，外部输入校验失效。改用 `MS_CHECK_*` 返回码（保留到 release）。仅函数内部辅助函数可用 ASSERT（建议 1.3.3）。

```cpp
✗ API int Run(const Tensor *t) { MS_ASSERT(t != nullptr); ... }   // release 下 t=nullptr 解引用
✓ API int Run(const Tensor *t) { MS_CHECK_TRUE_RET(t != nullptr, RET_ERROR); ... }
```

#### 不修改的指针参数加 const

不修改内容的指针参数声明为 `const T *`，接口契约清晰、便于 const 正确性传染。

```cpp
✗ void Print(char *str);          // 调用方担心被改
✓ void Print(const char *str);    // 明确不改
```

#### 谨慎使用不可重入函数

`strtok` / `gmtime` / `rand` 等不可重入函数有内部静态状态，多线程下行为未定义。改用可重入变体（`strtok_r` / `gmtime_r` / `rand_r`）。

#### 字符串/指针参数检查 NULL

字符串/指针作为函数参数应在入口校验 NULL，与 [指针与内存安全](#指针与内存安全) 同源。

#### 入参个数 ≤ 5（lint 强制 + 推荐）

pylint `max-args=5`（Python）强制；C++/Shell/CMake 推荐同阈值。超 5 个**用参数对象**，不是硬塞第 6 个参数：

```cpp
✗ void ParseKernel(const Node &n, vector<int> *k, vector<int> *s, vector<int> *p, int *mode, bool *is3d);  // 6 个参数

✓ struct PoolAttrs { vector<int> kernels, strides, pads; int round_mode; bool is_3d; };
✓ void ParseKernel(const Node &n, PoolAttrs *attrs);  // 2 个参数
```

详见 [SKILL.md 代码结构章](SKILL.md#代码结构让-lizard-不报警) 三类硬指标表。

### 循环退出条件

每个循环必须有明确且可达的退出条件。

```cpp
✗ for (size_t i = 0; i != N; ++i) { ... }     // i 溢出后绕过 N，死循环
✗ while (true) { if (cond) break; ... }       // 须 100% 保证 cond 能达成，否则加上限保护

✓ for (size_t i = 0; i < N; ++i) { ... }      // 用 < 不用 !=
✓ size_t max_iter = 1000;
  while (!done && max_iter-- > 0) { ... }     // 加最大迭代上限兜底
```

外部控制的循环次数还须先校验非 0、有界。

### 错误处理（异常）

#### 禁 C++ 异常机制

严禁 C++ 异常机制：异常打乱执行流程、资源可能不清理、降低复用性、依赖编译器/OS/处理器致性能降低、二进制层面增加攻击面。所有错误应通过错误值在函数间传递并判断。例外：接管 C++ 语言本身抛出的异常（如 `new` 失败、STL）、第三方库抛出的异常时，可用 `try`/`catch`。

lite 收敛方向见 [错误处理决策](#错误处理决策)：默认 `MS_CHECK_*` 返回码（全树 7527 处），`MS_EXCEPTION_*` 抛异常（1444 处）仅作返回码无法传递时的偏离，新代码避免。

## 语言规范与 PR 案例

每个主题 = 规则 + 来自历史 PR 的 ✗/✓ 具体修复 + 来源 PR 号。主题内的规则若已在 SKILL.md 精简出现，此处给深度示例；若无重复，直接看案例。

### 指针与内存安全

**规则：** 每个指针使用前校验。`delete` 后置 `nullptr`。数组分配失败时释放已分配项。安全关键函数返回值绝不能 cast 为 void。

```cpp
// --- 空指针校验 ---
// 错误：未校验
auto output = node->cast<CNodePtr>();
output->func_graph();

// 正确：逐步校验
MS_EXCEPTION_IF_NULL(node);
auto output = node->cast<CNodePtr>();
MS_EXCEPTION_IF_NULL(output);
output->func_graph();
```

```cpp
// --- 方法返回值链式调用前校验 ---
// 错误
auto pos = GetValue<int64_t>(prim->GetPrimalAttr(RING_ATTENTION_POS));

// 正确
MS_EXCEPTION_IF_NULL(prim);
auto attr = prim->GetPrimalAttr(RING_ATTENTION_POS);
MS_EXCEPTION_IF_NULL(attr);
auto pos = GetValue<int64_t>(attr);
```

```cpp
// --- 预防 Use-After-Free ---
// 错误：delete 后指针悬空
void Destroy(MSTensor *tensor) noexcept {
  if (tensor != nullptr) { delete tensor; }
}

// 正确：delete 后置空
void Destroy(MSTensor *tensor) noexcept {
  if (tensor != nullptr) { delete tensor; tensor = nullptr; }
}
```

```cpp
// --- 数组分配时错误路径清理 ---
// 错误：部分失败时内存泄漏
for (size_t i = 0; i < inputs.size(); i++) {
  inputs_[i] = new (std::nothrow) MSTensor(inputs[i].impl());
  if (inputs_[i] == nullptr) {
    inputs_.clear();  // 已分配的 inputs_[0..i-1] 泄漏了！
    return nullptr;
  }
}

// 正确：返回前释放已分配的项，并逐位置空（与"delete 后置 nullptr"规则一致）
for (size_t i = 0; i < inputs.size(); i++) {
  inputs_[i] = new (std::nothrow) MSTensor(inputs[i].impl());
  if (inputs_[i] == nullptr) {
    for (size_t j = 0; j < i; j++) { delete inputs_[j]; inputs_[j] = nullptr; }
    inputs_.clear();
    return nullptr;
  }
}
```

```cpp
// --- 析构函数内存泄漏 ---
// 错误：调用 Terminate 未传递 actor_mgr
virtual ~MindrtExecutor() { MindrtTerminate(); }

// 正确：传递并做空指针检查
virtual ~MindrtExecutor() { MindrtTerminate(actor_mgr_); }
void MindrtTerminate(const std::shared_ptr<ActorMgr> &actor_mgr) {
  if (actor_mgr != nullptr) { actor_mgr->TerminateAll(); }
}
```

**来源 PR：** !5, !70, !582, !85871, !86513, !86358, !63, !85, !704, !812, !542, !226

整体所有权模型参见 [设计原则·RAII](#raii----资源获取即初始化)。优先使用 `std::unique_ptr` / `std::make_unique` 而非手动 `new`/`delete` + 空指针检查。

### 边界与类型安全

**规则：** 正确比较运算符（`>=` 非 `>`）。有符号/无符号不混用。乘法前查溢出。用形状比较（非元素数）判断广播。

```cpp
// --- 边界条件正确性 ---
// 错误：允许边界值本身
if (scope_idx > MEM_SCOPE_BULK) { MS_EXCEPTION(...) << "should be less"; }
if (rank_size == 0) { MS_EXCEPTION(...) << "can not be zero"; }  // 仅当 rank_size 是 int32_t 时，== 0 才会放过负数

// 正确：使用 >= 和 <=
if (scope_idx >= MEM_SCOPE_BULK) { MS_EXCEPTION(...) << "should be less"; }
if (rank_size <= 0) { MS_EXCEPTION(...) << "must be larger than zero"; }
// 注意：若 rank_size 是 size_t（无符号），<= 0 与 == 0 等价，"防御负数"规则失效。
// 此时 == 0 已足够，不必硬改成 <= 0。
```

```cpp
// --- 有符号/无符号类型安全 ---
// 错误：无符号下溢（可能导致无限循环）
size_t index = bit_width_ - 1 - bit_pos;

// 正确：显式转换
size_t index = bit_width_ - 1 - static_cast<size_t>(bit_pos);
```

```cpp
// --- 操作前检查溢出 ---
if (SIZE_MUL_OVERFLOW(index_size, index_shape[i])) {
  return NNACL_ERRCODE_MUL_OVERFLOW;
}
index_size *= index_shape[i];

if (in_tensors_.at(0)->ElementsNum() > INT_MAX) {
  MS_LOG(ERROR) << "Data size exceeds INT32_MAX";
  return RET_NOT_SUPPORT;
}
```

```cpp
// --- 数组越界安全 ---
// 错误：未检查长度直接索引
for (; i < length - C4NUM; i += C4NUM) { vld1q_f32(src + i); ... }

// 正确：先检查长度
if (length > C4NUM) {
  for (; i < length - C4NUM; i += C4NUM) { vld1q_f32(src + i); ... }
}

// 错误：未校验容器非空
auto out_shape = out_tensors_.at(0)->shape();
for (size_t i = 0; i < out_shape.size() - 1; ++i) { row *= out_shape.at(i); }

// 正确：先校验大小
MS_CHECK_GT(out_tensors_.size(), 0, RET_ERROR);
auto out_shape = out_tensors_.at(0)->shape();
MS_CHECK_GT(out_shape.size(), 0, RET_ERROR);
```

```cpp
// --- 广播：比较形状而非元素数 ---
// 错误：元素数相同但形状不同 → 结果错误
broadcast_ = input0->ElementsNum() != input1->ElementsNum();

// 正确：直接比较形状
broadcast_ = (in_tensors_.at(0)->shape() != in_tensors_.at(1)->shape());
```

`SIZE_MUL_OVERFLOW` / `INT_MUL_OVERFLOW` 定义在 `src/litert/kernel/cpu/nnacl_c/op_base.h`。

**来源 PR：** !713, !91449, !23701, !87111, !296, !420, !303, !302, !667, !707

### 逻辑正确性

**规则：** 条件准确反映意图。`strcmp` 匹配返回 0。循环变量对应其循环。错误路径返回错误码。错误消息匹配条件。

```cpp
// --- strcmp 匹配时返回 0（逻辑 false）---
// 错误：名字匹配时条件为 false，永远进不来
if (strcmp(tensor->name, target_name)) { return tensor; }

// 正确：显式 == 0
if (strcmp(tensor->name, target_name) == 0) { return tensor; }
```

```cpp
// --- 复制粘贴 / 循环变量错误 ---
// 错误：使用了外层变量 `i` 而非 `j`
for (size_t j = 0; i < data_size; ++j) { data[j] = 0; }

// 正确
for (size_t j = 0; j < data_size; ++j) { data[j] = 0; }
```

```cpp
// --- 字符串解析索引笔误 ---
// 错误：hex_str[p] 应为 hex_str[p+1]
} else if (hex_str[p] >= 'A' && hex_str[p] <= 'F') {

// 正确
} else if (hex_str[p + 1] >= 'A' && hex_str[p + 1] <= 'F') {
```

```cpp
// --- 错误路径返回码错误 ---
// 错误：失败时返回成功码
MS_LOG(ERROR) << "convert failed";
return RET_OK;

// 正确
MS_LOG(ERROR) << "convert failed";
return RET_ERROR;
```

```cpp
// --- 条件与消息矛盾 ---
// 错误：条件说"非空"但日志说"为空"
if (!templates.empty()) {
  MS_LOG(ERROR) << "templates is empty";
  return;
}

// 正确：条件匹配消息
if (templates.empty()) {
  MS_LOG(ERROR) << "templates is empty";
  return;
}
```

```cpp
// --- 迭代器安全 ---
// 错误：迭代器失效
for (auto it = vec.begin(); it < vec.end(); it++) {
  if (should_remove(*it)) { it = vec.erase(it); }
}

// 正确：构建新容器
std::vector<T> new_vec;
for (auto &item : vec) {
  if (!should_remove(item)) { new_vec.push_back(item); }
}
vec = std::move(new_vec);
```

每个 `switch` 必须有 `default` 标签（MISRA 0-3-2），详见 [设计原则·switch](#switch-必须有-default)。

**来源 PR：** !338, !23701, model_c.cc 修复, 模型解析器修复, !711

### 代码结构与复杂度

**规则：** 合并条件、提前返回、提前 continue 减少嵌套。拆分大函数。提取公共逻辑为辅助函数。用参数对象替代多个输出参数。

```cpp
// --- 合并条件减少嵌套 ---
// 错误：3 层嵌套
if (plugin != nullptr) {
  if (op_def->is_graph_view_) {
    if (build_info != nullptr) { /* 核心逻辑 */ }
  }
}

// 正确：2 层
if (plugin != nullptr && op_def->is_graph_view_) {
  if (build_info != nullptr) { /* 核心逻辑 */ }
}
```

```cpp
// --- 提前 continue 减少嵌套 ---
// 错误
for (auto ifa = list; ifa != nullptr; ifa = ifa->next) {
  if (ifa->addr != nullptr && ifa->family == AF_INET6) {
    // 深层嵌套逻辑
  }
}

// 正确
for (auto ifa = list; ifa != nullptr; ifa = ifa->next) {
  if (ifa->addr == nullptr || ifa->family != AF_INET6) { continue; }
  // 浅层逻辑
}
```

```cpp
// --- 反转条件 + 提前返回 ---
// 错误：错误在 else 分支，主逻辑深层嵌套
if (result > 0) {
  // 30+ 行处理
} else {
  MS_LOG(EXCEPTION) << "Invalid format";
}

// 正确：错误先处理，提前返回
if (result <= 0) {
  MS_LOG(EXCEPTION) << "Invalid format";
  return;
}
// 正常处理，减少嵌套
```

```cpp
// --- 函数拆分 ---
// 错误：一个函数处理 Custom 和 General 查找（60+ 行）
int FindProviderKernel(...) {
  if (prim_type == Custom) { /* 30 行自定义逻辑 */ }
  // 20 行通用逻辑
}

// 正确：拆分为独立函数
int FindCustomKernel(...) { /* 仅自定义 */ }
int FindGeneralKernel(...) { /* 仅通用 */ }
int FindProviderKernel(...) {
  if (prim_type == Custom) return FindCustomKernel(...);
  return FindGeneralKernel(...);
}
```

```cpp
// --- 提取重复代码块 ---
// 错误：30 行 if-else 链在 2 处调用点重复
if (data_type == kFloat32) { PrintBuffer<float>(...); }
else if (data_type == kFloat64) { PrintBuffer<double>(...); }
// ... 还有 10 种类型

// 正确：提取辅助函数
void PrintByDataType(DataType type, const void *data, size_t n) {
  if (type == kFloat32) { PrintBuffer<float>(data, n); }
  // ... 所有类型
}
// 调用处：一行
PrintByDataType(data_type, data, n);
```

```cpp
// --- 参数对象模式 ---
// 错误：5 个独立输出参数
void ParseKernel(const Node &node, vector<int> *k, vector<int> *s, vector<int> *p, int *mode, bool *is3d);

// 正确：单个结构体
struct PoolAttrs { vector<int> kernels, strides, pads; int round_mode; bool is_3d; };
void ParseKernel(const Node &node, PoolAttrs *attrs);
```

**来源 PR：** !89495, !88411, !92072, !891, !901, !580, !92651, !23, !24692

Lizard 阈值与白名单管理见 [CI 工具配置·通用规则](#通用规则行宽tab排除目录) 与 [附录：复杂度白名单管理](#附录复杂度白名单管理)。

### C++ 风格与现代特性

**规则：** `static_cast` 不用 C 风格转换。`nullptr` 不用 `NULL`。`constexpr` 不用 `#define`。Include 顺序：对应头 → 系统 → C++ 标准库 → 其他库 → 项目头。魔法数字与元组索引用命名常量。成员变量用类内初始化器。

#### 命名规范

遵循 Google C++ 编码规范命名：

| 实体 | 风格 | 示例 |
|------|------|------|
| 类名 | PascalCase | `OpenCLExecutor` |
| 函数名 | PascalCase | `RunKernel()` |
| 变量名 | snake_case | `kernel_size` |
| 常量名 | kCamelCase | `kMaxDimension` |
| 成员变量 | snake_case 加尾部 `_` | `input_tensor_` |
| 命名空间 | snake_case | `mindspore::lite` |
| 宏 | UPPER_SNAKE_CASE | `MAX_STACK_SIZE` |

#### Include 顺序

Include 排序已禁用（`SortIncludes: false`），但 `.clang-format` 文件定义了优先级类别：

1. 系统头文件（`<*.h>`）
2. 扩展头文件（`<ext/*.h>`）
3. 其他 include（`<*>`）
4. 项目头文件（`"*.h"`）

#### 现代特性示例

```cpp
// --- C 风格 → 现代 C++ ---
// 错误                          → 正确
(int8_t)temp                   → static_cast<int8_t>(temp)
time(NULL)                     → time(nullptr)
#define NUM_OF_CLASSES 10       → constexpr int kNumClasses = 10;
isnan(x) || isinf(x)           → std::isnan(x) || std::isinf(x)
std::get<3>(res)               → std::get<kIndex3>(res)
if (total_count == 2)          → if (total_count == kSecondStep)
```

```cpp
// --- Include 排序（Google 风格）---
// 错误：项目头文件在系统头文件之前
#include "pybind_api/backward_node_py.h"
#include <memory>
#include "include/common/utils/exception.h"

// 正确：系统头文件在前，项目头文件在后，按字母排序
#include <memory>
#include "include/common/utils/exception.h"
#include "include/common/utils/pyobj_manager.h"
#include "pybind_api/backward_node_py.h"
```

```cpp
// --- 头文件使用前置声明 ---
// 错误：头文件不必要地包含完整定义
// message.h
#include "full/enum_definition.h"  // 把所有内容拉入头文件

// 正确：头文件前置声明，.cc 文件包含完整定义
// message.h: 仅前置声明
// message.cc: #include "full/enum_definition.h"
```

```cpp
// --- 移除不必要的导出宏 ---
// 错误：不跨 so 边界的函数加了导出宏
FRONTEND_EXPORT ValuePtr ConvertSlice(const py::object &obj);

// 正确：不跨 so 边界时移除
ValuePtr ConvertSlice(const py::object &obj);
```

```cpp
// --- 成员变量初始化和排序 ---
// 错误：未初始化，初始化列表顺序与声明顺序不一致
class Kernel {
  int64_t size_;           // 无初始化器
  bool flag_;
  Kernel(...) : flag_(f), size_(s) {}  // 顺序错误
};

// 正确：类内初始化器，列表顺序与声明一致
class Kernel {
  int64_t size_{0};
  bool flag_{false};
  Kernel(...) : size_(s), flag_(f) {}
};
```

```cpp
// --- 错误检查宏统一 ---
// 错误：4 行冗长检查
if (aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
  MS_LOG(ERROR) << "Malloc failed";
  return false;
}

// 正确：单行宏
MS_CHECK_TRUE_MSG(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS,
                  false, "Malloc failed.");
```

```cpp
// --- 宏分号约定 ---
// 错误：宏定义中有尾部分号
#define TYPEID_TRAIT(typeid, prototype) \
  struct TypeIdTrait<prototype> { static const TypeId type_id = typeid; };  // ← 无分号

// 正确：分号只在调用处
#define TYPEID_TRAIT(typeid, prototype) \
  struct TypeIdTrait<prototype> { static const TypeId type_id = typeid; }
TYPEID_TRAIT(kNumberTypeInt8, int8_t);  // ← 分号在此
```

错误检查宏（`MS_EXCEPTION_*` vs `MS_CHECK_*_MSG` vs `MS_CHECK_*_RET`）的选用决策见 [错误处理决策](#错误处理决策)。

**来源 PR：** !23, !638, !89483, !89393, !91492, !90276, !89406, !88411, !87111, !582, !768, !89522, !86817

错误检查宏选用见 [错误处理决策](#错误处理决策)。

### 布尔与条件简化

**规则：** 直接用布尔变量（不加 `== true`/`!= false`）。简化字符串检查。反转深层 else 为提前返回。

```cpp
// --- 布尔简化 ---
// 错误                          → 正确
if (trans_a != false)          → if (trans_a)
if (trans_b != true)           → if (!trans_b)
if (enabled == true)           → if (enabled)
if (str != "")                 → if (!str.empty())
if (str == "true" ? true : false) → bool enabled = (str == "true");
```

注：运算符优先级澄清不在此处展开。位运算混入 `<<` 链时（如 `x << y + z`）才真正需要加括号澄清，普通算术加减（如 `graphs.size() - 1`）的优先级本身就高于 `<<`，无需额外括号。单行 if 缺大括号属于格式问题，由 clang-format/cpplint 处理。

**来源 PR：** !713, !768, !85855, !89673

### 死代码与冗余清理

**规则：** 完全删除未使用代码、导入、头文件、注释掉的代码。移除 CMake 重复源文件。移除未使用函数声明与导出宏。

```cpp
// --- 未使用的函数声明 ---
// 错误：声明但从未定义/调用
void MakeProperNameToFuncGraph(const FuncGraphPtr &func_graph, std::string name);
bool ConvertPrepareAdapt(const ResourcePtr &resource);

// 正确：完全删除
```

```cpp
// --- 注释掉的代码 ---
// 错误：60+ 行注释掉的测试代码
// TEST_F(LiteMindRtTest, HQueueTest) {
//   HQueue<int *> hq;
//   ...

// 正确：完全删除；需要时从 git 历史恢复
```

```python
# --- 未使用的 Python 导入 ---
# 错误
from mindspore.ops._utils.utils import get_broadcast_shape  # 从未使用

# 正确：移除
```

```cmake
# --- CMake 重复源文件 ---
# 错误：同一文件编译进两个目标
set(SRC_A ... file_utils.cc)
set(SRC_B ... file_utils.cc)

# 正确：只编译一次
set(SRC_A ... file_utils.cc)
set(SRC_B ...)  # 移除重复
```

**来源 PR：** !494, !296, !314, !90715, !5, !638, !88181, !90276, !88525, !21, !23

### Python 专项规范

**规则：** PEP 8 命名（局部 lower_snake_case，模块常量 UPPER）。Import 顺序：标准库 → 第三方 → 本地。特定异常。不用 CPython 内部 API。不用 `self` 时用 `@staticmethod`。

#### 命名规范

pylintrc 定义了以下正则模式：

| 实体 | 模式 | 示例 |
|------|------|------|
| 模块 | `([a-z_][a-z0-9_]*)` 或 `([A-Z][a-zA-Z0-9]+)` | `model_parser`、`OpenCL` |
| 类 | `_?[A-Z][a-zA-Z0-9]*` | `ModelImpl`、`_InternalHelper` |
| 函数 | `_?[A-Z][a-zA-Z0-9]*` 或 `_?[a-z][a-z0-9_]*` | `RunInference`、`parse_config` |
| 方法 | `__[a-z0-9_]+__` 或 `_{0,2}[A-Z][a-zA-Z0-9]*` 或 `_{0,2}[a-z][a-z0-9_]*` | `__init__`、`_Run`、`get_output` |
| 变量 | `[a-z][a-z0-9_]*` | `input_tensor` |
| 参数 | `[a-z][a-z0-9_]*` | `batch_size` |
| 常量 | `_?[A-Z][A-Z0-9_]*` 或 `__[a-z0-9_]+__` 或 `_?[a-z][a-z0-9_]*` | `MAX_SIZE`、`__all__` |
| 属性 | `_{0,2}[a-z][a-z0-9_]*` | `_internal`、`output_data` |
| 允许的名称 | `main`、`_` | -- |

#### Clean Code 常见问题

1. **遮蔽 Python 内置名称** -- 禁止将 `id`、`str`、`list`、`dict`、`type`、`input`、`format`、`hash`、`filter`、`map` 用作变量名或函数名。在 PR !90715 中被标记为"代码坏味道"。

2. **将 `len()` 用作条件判断** -- 使用 `if items:` 代替 `if len(items) > 0:`。使用 `if not items:` 代替 `if len(items) == 0:`。

3. **捕获过于宽泛的异常** -- 生产代码中避免使用 `except Exception:`，应捕获具体异常。在封装 C++ 绑定时，如必须使用宽泛捕获，需在代码中说明原因。

4. **缺少文档字符串** -- 超过 10 行的函数和类应添加 docstring。

5. **未使用的导入** -- 必须删除所有未使用的 import 语句。

6. **访问保护成员** -- 避免访问外部类的 `_protected` 成员。在封装 C++ 绑定时有时不可避免，但需在代码中说明。

#### PR 案例

```python
# --- 局部变量命名 ---
# 错误：局部变量使用 UPPER_SNAKE_CASE
PATH_WHITE_LIST_REGEX = re.compile(r"...")

# 正确：局部变量使用 lower_snake_case
path_white_list_regex = re.compile(r"...")
```

```python
# --- @staticmethod ---
# 错误
def check_path(self, path):
    ...  # 从未使用 self

# 正确
@staticmethod
def check_path(path):
    ...
```

```python
# --- 特定异常 ---
# 错误
except Exception as e:
    print(f"Error: {e}")

# 正确
except RuntimeError as e:
    print(f"Error: {e}")
```

```python
# --- 不用内部 API ---
# 错误：CPython 内部 API，无兼容性保证
vlog_print("1", "ME", __file__, sys._getframe().f_lineno, msg)

# 正确：标准库 inspect；返回的 frame 对象可能为 None，需要判空
frame = inspect.currentframe()
line_no = frame.f_lineno if frame is not None else 0
del frame  # 避免引用循环
vlog_print("1", "ME", __file__, line_no, msg)
```

```python
# --- Import 排序（PEP 8）---
# 错误：__all__ 在 import 之后
from module import func
__all__ = ['func']

# 正确：__all__ 在前
__all__ = ['func']
from module import func
```

```python
# --- 布尔简化 ---
# 错误
if world_size != -1 and world_size != group_size:

# 正确：使用集合成员检查
if world_size not in {-1, group_size}:
```

```python
# --- 列表推导简化 ---
# 错误：不必要的推导
out = [i for i in range(x)]

# 正确：直接用 list()
out = list(range(x))
```

```python
# --- 不变量移到 try 外 ---
# 错误：常量在 try 内
try:
    prefix = "tcp://"
    if not url.startswith(prefix): raise ValueError(...)

# 正确
prefix = "tcp://"
try:
    if not url.startswith(prefix): raise ValueError(...)
```

完整命名正则详见 `.jenkins/rules/pylint/pylintrc`，pylint 配置变更以仓库为准。

**来源 PR：** !89498, !582, !744, !754, !768, !92113, !91821

### 构建、日志与格式化

**规则：** 正确日志级别（DEBUG < INFO < WARNING < ERROR）。移除硬编码敏感默认值。注释格式须闭合前有空格。

```cpp
// --- 日志级别适当性 ---
// 错误：提示性信息用 WARNING，预期条件用 ERROR
MS_LOG(WARNING) << "experimental feature advisory...";
MS_LOG(ERROR) << "node has no user";    // 预期条件

// 正确：使用适当级别
MS_LOG(INFO) << "experimental feature advisory...";
MS_LOG(INFO) << "node has no user";
```

```cpp
// --- 注释格式 ---
// 错误
/* CPU GPU NPU support*/

// 正确
/* CPU GPU NPU support */
```

```cpp
// --- 敏感默认值 ---
// 错误：硬编码 localhost IP
constexpr char kDefaultCacheHost[] = "127.0.0.1";

// 正确：空默认值
constexpr char kDefaultCacheHost[] = "";
```

**来源 PR：** !91936, !91938, !91940, !87419, !23, !62

## 附录：复杂度白名单管理

### 何时加入白名单

Lizard 复杂度白名单（`.jenkins/check/config/whitelizard.txt`）仅适用于以下场景：

1. **自动生成的代码** -- HPC GEMM kernel（AVX512、FMA 变体）、SIMD 内联函数。
2. **性能关键的算子 kernel** -- 手动优化的卷积、池化、Winograd 变换实现，拆分函数会影响性能。
3. **历史遗留代码** -- 现有函数重构风险高、回归测试成本大。

**不应加入白名单的情况：**
- 新代码 -- 应重构。
- 可以在不影响性能的前提下拆分的代码。
- 复杂度来源于结构不佳而非算法固有复杂性的代码。

### 如何添加白名单

在 `.jenkins/check/config/whitelizard.txt` 中添加条目，使用以下格式：

```
# 格式 1：函数名（适用于所有重载）
function_name1, function_name2

# 格式 2：指定文件
file_path:function_name1, function_name2
```

示例：
```
mindspore-lite/mindspore-lite/tools/converter/graphdef_transform.cc:mindspore::lite::GraphDefTransform::Transform
```

### 当前白名单规模

白名单会随代码演进增减，**不要在本文档里写死数字**。运行以下命令查看当前规模与构成：

```bash
wc -l .jenkins/check/config/whitelizard.txt
grep -E "converter|parser" .jenkins/check/config/whitelizard.txt | wc -l  # 示例：parser/converter 相关条目数
```

### 重构指南

当 Lizard 对你的代码触发告警时，优先使用以下重构策略：

1. **提取辅助函数** -- 将复杂条件逻辑移入命名清晰的辅助函数。
2. **使用提前返回** -- 通过在错误条件处提前返回减少嵌套层级。
3. **使用查找表** -- 用 map 分发替代冗长的 `switch`/`if-else` 链。
4. **简化布尔逻辑** -- 合并条件、使用德摩根定律、提取有意义的布尔变量名。

数据驱动的 `switch` / map 查找（30 个算子注册、opcode 派发表）CCN 高是业务本质复杂，硬拆成 30 个函数更难读。这种情况：

- 重构为表驱动 + 通用 handler（仍是 1 个函数，CCN 大幅下降）
- 实在无法表驱动时，加入白名单并在 PR 说明中解释
