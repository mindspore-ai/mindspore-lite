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
   - 推荐命令（自动排除 `configs`、`utils`、`upstream` 等组织性子目录）：
     ```bash
     find mindspore-lite/examples/base_models \
       -mindepth 1 -maxdepth 2 -type d \
       -not -name configs -not -name utils -not -name upstream \
       | sort
     ```
   - 跳过**空目录/占位目录**：若子目录内既无 `README*` 也无 `*.py` / `*.onnx` / 权重文件，视为未落地，不计入表格（如 `tcp/upstream/TCP/` 这种只有空嵌套的）。
2. 对照表格现有条目
   - 在两份 README 的表格中搜索每个模型：
     - 找到：仅做“补链接 + `&#9989;`”（不改模型名文本，除非触发下方“明显不一致判定”）
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
   - 一次新增多个条目到**同一列**时，按“**子类聚类 + 字母序**”决定填入空单元格的先后：
     - 例：同列已有 `yolov10x`、`vit-*`、`bert-*`，新增 `yolov8`、`gliner-*`、`grounding-dino-*` 时，优先把 `yolov8` 放到离 `yolov10x` 最近的空单元格；其余按目录名字母序自上而下填充。
     - 找不到子类聚类关系时，统一按目录名字母序填充。
     - 不必为"聚类"强行插入新行或挪动已有条目。
5. 中英文同步
   - 在 `README_CN.md` 完成更新后，把**同样的结构变更**同步到 `README.md`：
     - 相同模型的链接必须一致
     - 新增规则一致（同一列的空单元格复用逻辑一致）
6. 自检
   - 每一行的 `|` 列数一致（**6 列**）。一行 awk 即可核对（无输出即正确）：
     ```bash
     awk -F'|' '/云侧推理模型支持列表/,/^### API与文档/' README_CN.md \
       | awk -F'|' 'NF>1 && NF-2!=6 {print NR": "NF-2" cols — "$0}'
     ```
     EN 版把章节标题换成 `Supported models for cloud-side` / `^### API and documentation`。
   - 新增内容没有引入多余的空白行或多余的表格行。
   - 链接路径与目录实际存在的相对路径一致。批量校验所有链接：
     ```bash
     grep -oE 'tree/master/mindspore-lite/examples/base_models/[^)]+' README_CN.md README.md \
       | sed 's|.*tree/master/||' | sort -u \
       | while read -r p; do [ -e "$p" ] || echo "MISSING: $p"; done
     ```

## 列分类规则（可扩展）

优先用“表格已有语义”保持一致；当新增模型时按以下启发式分类：

- 音频模型（ASR/TTS）
  - 目录名/模型名含：`asr`、`tts`、`cosyvoice`、`wenet`
- 信息检索/向量嵌入/CNN/其他
  - 目录名/模型名含：`reranker`、`embedding`、`vit`、`yolo`、`bevdet`、`bert`
- 视觉语言模型（VLM）
  - 满足**任一**即归入 VLM：
    - 模型名含 `VL` 且语义为 VLM（如 `...VL...Instruct`）；
    - 任务语义为视觉-语言：OCR（`ocr`）、文本条件检测/grounding（`grounding_dino`）、image-caption、VQA 等。
  - 注意：如果表格历史上把某些 `VL-Embedding` / `VL-Reranker` 放在“其他”，则新增时保持一致放“其他”。
  - VLM 列已满（需要新增行）而“其他”列仍有空单元格时，可酌情把**非对话型 VLM**（OCR、grounding、CLIP 类视觉编码器）放到“其他”，避免新增行；对话型 VLM（`*-Instruct` / `*-Thinking`）仍优先放 VLM 列。
- 大语言模型（LLM）
  - `qwen` 系列且不属于 `vl/asr/tts/reranker/embedding`
- 图像/视频生成模型
  - `stable-diffusion`、`wan`、`kandinsky`、`flux`、`qwen-image` 等生成类

## 常见目录名 ↔ 展示名（建议）

新增条目时，若表格中没有该模型名，按以下**主规则**生成展示名：

- **主规则（默认）**：沿用目录名，仅把 `_` 换成 `-`，保持原大小写。
  - 例：`bert_base_chinese` → `bert-base-chinese`、`yolov8` → `yolov8`、`gliner_large-v2.5` → `gliner-large-v2.5`、`glm_ocr` → `glm-ocr`。
- **例外 1（有广为人知的官方品牌大小写时启用）**：`ViT-*`、`BEVDet`、`YOLOv*`、`GLiNER-*`、`GLM-*`、`Grounding-DINO-*` 等可改用官方大小写。
- **例外 2（已有同族模型时跟随）**：表格里已存在 `yolov10x`（小写）→ 新增 `yolov8` 也用小写，不要写成 `YOLOv8`；已存在 `Qwen3-*`（Title Case）→ 新增同族 qwen 模型也用 Title Case。
- 拿不准时回退主规则（沿用目录名最安全）。

历史样例（仅供回溯参考，新条目以上述主规则为准）：

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

## 明显不一致判定（决定"是否改显示文本"）

仅当满足以下**任一**才视为"明显不一致"，可改表格显示文本；否则一律保留原文本只补链接：

- 显示名中的**关键修饰词**在目录名里完全找不到。例：显示 `Qwen3-VL-Reranker-8B`、目录却是 `qwen3_reranker_8b`（无 `vl`）→ 可考虑去掉 `VL`。
- 显示名与目录名指向**不同尺寸/版本**。例：显示 `Qwen3-VL-4B-Instruct`、目录却是 `qwen3_vl_4b_thinking`。
- 显示名有明显拼写错误。例：`Kand0-T2V0` 这类历史笔误（属"约定俗成"的可不动）。

边界模糊时**保守处理**：只补链接，并在交付总结里单列疑点让用户复核，不擅自改文本。

## 反向不一致（表格条目无对应目录）

同步是单向"目录 → 表格"。遇到反向不一致时：

1. **表格有条目、目录里没有**：
   - **不要自行删除**（可能在其他仓库维护、或尚未开源、或表格为路线图）。
   - 保留原条目不动，**不补** `&#9989;`。
   - 在交付总结里单列"无目录的表格条目"，让用户决定下架/保留。
2. **表格有条目、目录对得上但有疑点**（如显示 `Qwen3-VL-Reranker-8B`、目录是 `qwen3_reranker_8b`）：
   - 按"明显不一致判定"处理；判定不通过则只补链接、不改文本，并单列疑点。

## 单元格写法规范

- 已存在文本补链接：`[展示名](URL) &#9989;`
- 只加勾：不要单独放在其他列或新增行；始终跟在该模型单元格文本后
- 同一单元格内只放一个模型（避免多模型用逗号拼在一起）
