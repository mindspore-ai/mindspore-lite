---
description: 用 clean-code-check skill 对当前改动做 PR 级 review
argument-hint: [可选: 文件路径或 commit 范围，默认 review 工作树相对 master 的所有改动]
---

Invoke the `clean-code-check` skill, then perform a PR-style review.

**Scope:** `$ARGUMENTS` if provided (file path or `<base>..<head>` commit range); otherwise the diff between `origin/master` (or `master`) and `HEAD`, plus any unstaged working tree changes.

**Process:**

1. Get the diff: `git diff --stat <base>..<head>` + `git diff <base>..<head>` (and `git diff` for unstaged).
2. Walk the skill's three steps:
   - **Step 1 (mechanical):** spot format/spelling/complexity issues that the 8 lint tools (clang-format/cmakelint/codespell/cpplint/pylint/shellcheck/tab/lizard) would catch — don't run them, just flag obvious violations.
   - **Step 2 (semantic):** pointers/memory, boundary/type, logic correctness, error-handling macro choice (MS_CHECK_* default vs MS_EXCEPTION_* as documented deviation).
   - **Step 3 (security):** injection / file ops / integer / memory / secure-function / sensitive / forbidden-mechanism per the skill's secure-coding chapter.
3. Cross-reference the 5-item PR checklist (返回值校验 / SOLID 迪米特 / UT 覆盖 / 外部接口变更 / 文档更新).
4. Output grouped by severity: **Critical / Important / Minor**, each with `file:line` and the specific fix.

Skip sections that have no findings. End with a one-line verdict: APPROVED / NEEDS FIXES.
