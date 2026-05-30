---
name: mslite-model-support-table-update
description: Update MindSpore Lite README_CN.md/README.md supported model tables from mindspore-lite/examples/base_models. Invoke when base model folders change or when support list links/&#9989; need syncing.
---

# MindSpore Lite 模型支持表格更新指南

## 目标

把 `mindspore-lite/examples/base_models/` 目录中**已存在的模型**同步到：

- `README_CN.md` 的“云侧推理模型支持列表”表格
- `README.md` 的 “Supported models for cloud-side inference” 表格

同步内容包括：

- 表格中已存在的模型：补齐超链接，并在同一单元格追加 `&#9989;`
- 表格中不存在的模型：按列分类插入（优先复用现有行的空单元格；仅当该列所有行都已填满才新增行）
- 中英文表格：结构与链接保持一致（模型名文本可按表格既有写法保留）

## 输入与输出

- 输入
  - 目录：`mindspore-lite/examples/base_models/`
  - 文档：`README_CN.md`、`README.md`
- 输出
  - 两个 README 的表格都更新到最新目录清单，且链接/`&#9989;` 完整一致

## 操作步骤（推荐顺序）

1. 盘点模型目录
   - 列出 `mindspore-lite/examples/base_models/` 下**每个子目录**（必要时包含二级子目录作为具体模型，例如 `yolov10/yolov10-X`）。
2. 对照表格现有条目
   - 在两份 README 的表格中搜索每个模型：
     - 找到：仅做“补链接 + `&#9989;`”（不改模型名文本，除非表格明显不一致）
     - 找不到：进入“新增条目”流程
3. 生成链接
   - 统一使用 AtomGit tree 链接，路径为 base_models 下相对路径：
     - `https://atomgit.com/mindspore/mindspore-lite/tree/master/mindspore-lite/examples/base_models/<REL_PATH>`
4. 新增条目（核心规则）
   - 先确定该模型应该归入哪一列（见“列分类规则”）。
   - **不要新增行，除非该列每一行都已填满**：
     - 把该模型填入该列**从上到下第一个空单元格**（等价于“追加到该列当前列表的末尾”，但通过复用已有行完成）。
     - 只填目标列这一格，其它列保持原值不动。
   - 当目标列所有行都非空时，才在表格末尾新增一行：
     - 新行其它列留空，目标列写入新模型的“链接 + `&#9989;`”。
5. 中英文同步
   - 在 `README_CN.md` 完成更新后，把**同样的结构变更**同步到 `README.md`：
     - 相同模型的链接必须一致
     - 新增规则一致（同一列的空单元格复用逻辑一致）
6. 自检
   - 每一行的 `|` 列数一致（5 列）
   - 新增内容没有引入多余的空白行或多余的表格行
   - 链接路径与目录实际存在的相对路径一致

## 列分类规则（可扩展）

优先用“表格已有语义”保持一致；当新增模型时按以下启发式分类：

- 音频模型（ASR/TTS）
  - 目录名/模型名含：`asr`、`tts`、`cosyvoice`、`wenet`
- 信息检索/向量嵌入/CNN/其他
  - 目录名/模型名含：`reranker`、`embedding`、`vit`、`yolo`、`bevdet`、`bert`
- 视觉语言模型（VLM）
  - 模型名含 `VL` 且语义为 VLM（如 `...VL...Instruct`）
  - 注意：如果表格历史上把某些 `VL-Embedding` / `VL-Reranker` 放在“其他”，则新增时保持一致放“其他”
- 大语言模型（LLM）
  - `qwen` 系列且不属于 `vl/asr/tts/reranker/embedding`
- 图像/视频生成模型
  - `stable-diffusion`、`wan`、`kandinsky`、`flux`、`qwen-image` 等生成类

## 常见目录名 ↔ 展示名（建议）

新增条目时，若表格中没有该模型名，可参考以下转换生成展示名（也可直接用目录名作为展示名）：

- `qwen3.5_4b` → `Qwen3.5-4B`
- `qwen3_5_0.8b` → `Qwen3.5-0.8B`
- `qwen3_1.7b` → `Qwen3-1.7B`
- `qwen3_0.6b` → `Qwen3-0.6B`
- `qwen2.5_0.5b` → `Qwen2.5-0.5B`
- `qwen2_0.5b` → `Qwen2-0.5B`
- `qwen3_asr_1.7b` → `Qwen3-ASR-1.7B`
- `qwen3_reranker_0.6b` → `Qwen3-Reranker-0.6B`
- `qwen3_vl_embedding_2b` → `Qwen3-VL-Embedding-2B`
- `qwen3_vl_reranker_2b` → `Qwen3-VL-Reranker-2B`
- `vit_base_patch16_224` → `ViT-Base-Patch16-224`
- `bevdet` → `BEVDet`
- `bert_base_chinese` → `bert_base_chinese`
- `yolov10/yolov10-X` → `yolov10-X`

## 单元格写法规范

- 已存在文本补链接：`[展示名](URL) &#9989;`
- 只加勾：不要单独放在其他列或新增行；始终跟在该模型单元格文本后
- 同一单元格内只放一个模型（避免多模型用逗号拼在一起）
