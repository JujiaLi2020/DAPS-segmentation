# DAPS Think-Aloud Segmentation Toolkit

这个项目包含两个独立界面：

- **DAPS Excel Segmenter**：上传 Excel/CSV，运行 DAPS segmentation，输出 `segments` 和 `boundaries`。
- **DAPS Calibration Lab**：从 segmentation workbook 中抽样 boundary context，生成多人标注模板，并评估 annotator agreement。

## 1. 环境准备

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果虚拟环境还不存在，先创建：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如需启用 parser-assisted boundary legality filter，安装 spaCy 和英文模型：

```powershell
.\.venv\Scripts\python.exe -m pip install spacy
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

如果没有安装 spaCy，程序会自动退回到 rule-based legality filter，pipeline 仍然可以运行。

## 2. 启动 DAPS Excel Segmenter

用途：正式切分 transcript。

```powershell
.\.venv\Scripts\python.exe code\daps_excel_ui.py
```

打开浏览器访问：

```text
http://127.0.0.1:7861
```

### 操作流程

1. 上传 Excel 或 CSV。
2. 点击 `Load Columns`。
3. 选择 ID column 和 transcript column。
4. 根据需要调整参数：
   - `Context width (w)`：候选边界左右窗口大小，建议 `12`。
   - `Local radius (r)`：局部阈值半径，建议 `6`。
   - `Sensitivity (tau)`：边界敏感度，建议 `0.55`。
   - `Minimum segment tokens (L_min)`：最短 segment，建议 `12`。
   - `Maximum segment tokens (L_max)`：最长 segment，建议 `60`。
   - `Maximum segments per record (K_max)`：单条 transcript 最多 segment，建议 `80`，设为 `0` 表示关闭。
   - `NMS radius (rho)`：相邻 boundary 抑制窗口，建议 `6`。
5. 在 `Signal vocabularies` 中可编辑 `C_t / M_t / A_t / R_t` 的 vocabulary。
6. 点击 `Run DAPS Segmentation`。
7. 下载输出 workbook。

输出 workbook 包含：

- `segments`：最终切分结果。
- `boundaries`：每个 token gap 的 signal evidence。

`boundaries` 中还包含 boundary legality 信息：

- `boundary_legality`：`legal`、`semi_legal` 或 `illegal`。
- `legality_reason`：为什么该 gap 被认为非法或半合法。
- `rule_constraint_reason`：rule-based filter 的判断。
- `spacy_left_pos / spacy_right_pos`：spaCy POS tags。
- `spacy_left_dep / spacy_right_dep`：spaCy dependency labels。
- `raw_semantic_gravity` 与 `semantic_gravity`：前者是原始 DAPS 分数，后者包含 legality penalty/boost。

`segments` 中包含自动预标注列：

- `Segment_Type`
- `Segment_Type_Confidence`
- `Mean_C_t`
- `Mean_M_t`
- `Mean_A_t`
- `Mean_R_t`

这些是 algorithmic pre-label，不是最终人工标签。

## 3. 命令行切分

也可以不用界面，直接运行：

```powershell
.\.venv\Scripts\python.exe code\daps_excel_segmenter.py data\Think_aloud_All_documents.csv -o data\daps_segments.xlsx --text-columns Interviewee_Response --embedding-model lexical
```

常用参数：

```powershell
--id-column ID
--text-columns Interviewee_Response
--embedding-model lexical
--min-segment-tokens 12
--max-segment-tokens 60
--max-segments-per-record 80
--nms-radius 6
--sensitivity 0.55
--legality-mode hybrid
--spacy-model en_core_web_sm
--semi-legal-penalty 0.18
--sentence-boundary-boost 0.08
```

## 4. 启动 DAPS Calibration Lab

用途：构建更可靠的 signal vocabulary，支持多用户人工标注。

```powershell
.\.venv\Scripts\python.exe code\daps_calibration_lab.py
```

打开浏览器访问：

```text
http://127.0.0.1:8000
```

Railway deployment uses:

```text
web: python code/daps_calibration_lab.py
```

The app reads Railway's `PORT` environment variable and binds to `0.0.0.0`. Research data, SQLite files, generated workbooks, manuscripts, and proposal files are excluded from GitHub by `.gitignore`; upload raw data through the app after deployment.

如果需要指定其他端口：

```powershell
$env:GRADIO_SERVER_PORT='8501'
.\.venv\Scripts\python.exe code\daps_calibration_lab.py
```

## 5. Calibration Lab 操作流程

Calibration Lab 现在按 method section 分成六个步骤。每个步骤标题后都有一个圆形 `?`，鼠标悬浮会显示该步骤的目的、建议用法和注意事项。每次点击主要按钮都会记录到 SQLite：

```text
data\calibration_outputs\daps_calibration.sqlite3
```

### Step 0. Segment Raw Transcript Data

用途：当你手里是原始 CSV/Excel，而不是 DAPS workbook 时，从这里开始。

对于当前 `data\input.csv`，默认结构是：

```text
source_file
interview_id
turn_number
speaker
text
```

当前 `data\input.csv` 已预清理为 `speaker = Interviewee`，原始含 interviewer 的版本保存在 `data\input_original_with_interviewer.csv`。

推荐设置：

```text
ID column: interview_id
Item/task column: turn_number
Transcript column(s): text
Clean encoding artifacts / mojibake: checked
Semantic model: lexical
```

`Clean encoding artifacts / mojibake` 也会清理规则稳定的 speaker label，例如 `Interviewer:`、`Interviewee:`、`Participant:`、`Child:`、`Speaker A:`。这一步不需要 LLM，因为这些标签可以用可复现的规则删除，避免 LLM 改写 transcript 内容。
Step 0 默认使用 `hybrid` legality filter：rule-based constraints 先阻断明显非法切点；如果本机安装了 spaCy `en_core_web_sm`，还会使用 POS、dependency、noun chunk 和 sentence boundary 特征来判断 `legal / semi_legal / illegal`。

点击 `Run Initial DAPS Segmentation` 后会输出一个新的 DAPS workbook，例如：

```text
data\calibration_outputs\input_initial_daps_segments.xlsx
```

该 workbook 包含：

- `segments`
- `boundaries`
- `cleaned_input`
- `run_metadata`

这个输出文件就是 Step 1 的输入。
Lab 会自动把 Step 0 生成的 workbook 同步填入 Step 1 和 Step 2 的 workbook 输入框；如果你只是继续使用同一份结果，不需要重复上传。
刷新浏览器或重新打开 Lab 时，系统会自动从 SQLite 读取最近一次 Step 0/Step 1 的结果并恢复到界面，不会重新执行 segmentation。也可以点击 `Load Latest SQLite State` 手动恢复。

### Step 1. Inspect Segmentation Workbook

用途：在抽样标注前做 pre-validation，确认 segmentation workbook 适合进入 calibration。

1. 打开 `Step 1 Inspect` tab。
2. 上传 DAPS segmentation workbook，例如：

```text
data\Think_aloud_All_documents_daps_segments_typed.xlsx
```

3. 点击 `Run Pre-Validation Inspection`。

输出包括：

- segment 数量、boundary 数量、record 数量。
- median/mean segment length。
- short segment rate。
- long segment count。
- event/speaker leakage。
- selected vs non-selected semantic gravity / transition pressure。
- signal saturation summary。

Step 1 还会给 `Suggested criterion` 对应的检查结果加上 `Status` 和 `Status_Note`：

- `PASS`：当前结果满足建议标准。
- `WARNING`：结果可用但需要检查或调参，例如片段偏短、signal 饱和。
- `FAIL`：建议先修复再进入人工标注，例如 event token、speaker label 或异常标点泄漏。
- `INFO`：仅用于记录，没有通用 pass/fail 阈值。

界面中只显示彩色状态表；每个 metric/signal 后的小圆形 `?` 可以悬浮查看含义。例如 `Speaker label leakage` 指 `Interviewer:`、`Interviewee:`、`Participant:`、`Child:`、`Speaker A:` 这类说话人标签残留在 segment 文本中。完整数值表会保存到 SQLite 的 `step1_prevalidation_summary` 和 `step1_signal_saturation_summary`。

### Step 2. Generate Multi-Annotator Template

用途：生成多人 boundary-context annotation template。

1. 打开 `Step 2 Template` tab。
2. 上传 DAPS segmentation workbook。如果刚刚完成 Step 0，这里会自动使用同一个 workbook。
3. 选择 `Calibration round`：
   - `Pilot annotation`：建议先抽 50 条，用于检查 codebook 是否清楚。
   - `Formal calibration`：建议抽 400-500 条，用于正式 vocabulary calibration。
4. 设置 `Unique boundary items`。
5. 设置 `Annotator IDs`，例如：

```text
Annotator_A, Annotator_B
```

6. 点击 `Generate Template`。
7. 下载生成的 annotation template。

生成的模板会把同一批 sampled boundary items 复制给每个 annotator。每一行包含：

- `Annotation_ID`
- `Annotation_Item_ID`
- `Annotator_ID`
- `Boundary_Strength_0_3`
- `Human_C_t`
- `Human_M_t`
- `Human_A_t`
- `Human_R_t`
- `Human_Primary_Type`
- `Cue_Span`
- `Counterexample_or_Exclusion`
- `Codebook_Issue`
- `Notes`
- `Previous_Segment`
- `Left_Segment`
- `Right_Segment`
- `Boundary_Context`
- algorithm signal scores

`Boundary_Context` 中的 `[[CANDIDATE_BOUNDARY]]` 只是候选边界标记，不是原始 transcript 文本。它表示 annotator 正在判断这个位置是否应该切分。Step 2 会排除已经被 phrase constraint 判定为明显不合理的候选点，例如 `this / one`、`a / triangle` 这类 phrase-internal split。

导出的 workbook 有四个 sheet：

- `annotation_items`：人工标注主表。
- `instructions`：逐步标注说明。
- `codebook`：每个变量的填写规则、允许值和例子。
- `short_labels`：界面短名和导出列名的对应关系。

人工标注时，`Boundary_Strength_0_3` 使用一个 0-3 判断：

- `0` = merge / not a boundary
- `1` = weak boundary
- `2` = moderate boundary
- `3` = strong boundary

`Human_*` signal 列使用 `0/1`：

- `1` = 该 signal 存在
- `0` = 该 signal 不存在

四个 signal 是多标签，不是单选。一个 boundary 可以同时有 `C_t` 和 `R_t`。

### Step 2b. In-App Human Review

用途：不在 Excel 宽表里逐格填写，而是在 Lab 界面中逐条 review。

推荐流程：

1. 先完成 Step 2，生成 annotation template。
2. 打开 `Step 2 Review` tab。
3. 点击 `Load Latest Template From SQLite`。
4. 在 `Annotation item` 下拉框选择一条待标注记录。
5. 左侧阅读 `Boundary_Context`、`Previous Segment`、`Left Segment`、`Right Segment`。
6. 右侧填写：
   - `Boundary`
   - `C_t`
   - `M_t`
   - `A_t`
   - `R_t`
   - `Primary`
   - `Cue`
   - `Reject` optional
   - `Issue` optional
   - `Notes` optional
7. 点击 `Save Current` 保存当前条，或点击 `Save and Next` 保存并自动进入下一条。
8. 全部完成后点击 `Export Reviewed Workbook`。
9. 导出的 reviewed workbook 会自动同步到 Step 3 的 completed annotation workbook 输入框。

Step 2 Review 的每次保存都会更新 SQLite 的 `step2_annotation_items` 当前表，不需要每次下载 Excel。Excel 主要用于备份、共享给其他 annotator 或进入 Step 3。

只填写这些人工列，不要修改 ID、context 或 algorithm score 列：

- `Boundary` (`Boundary_Strength_0_3`)：required。`0` = merge/not a boundary；`1` = weak；`2` = moderate；`3` = strong。
- `C_t` (`Human_C_t`)：required。`1` 表示有 cognitive transition，例如对象、操作、策略或 reasoning state 发生变化。
- `M_t` (`Human_M_t`)：required。`1` 表示有 metacognitive reset/monitoring，例如 `wait`、`actually`、`I don't know`、重新检查或修正。
- `A_t` (`Human_A_t`)：required。`1` 表示有 affective friction，例如困难、挫败、困惑、努力或情绪阻力。
- `R_t` (`Human_R_t`)：required。`1` 表示有 rhetorical/structural break，例如 speaker turn、pause/event、`then/okay` 这类结构性组织。
- `Primary` (`Human_Primary_Type`)：recommended。填写最主要类型：`cognitive`、`metacognitive`、`affective`、`structural`、`mixed`、`low_signal` 或 `unclear`。
这三个字段可以理解为同一个 `Evidence Log` 的三个盒子：

- `Cue` (`Cue_Span`) = `Evidence Log 1: Supporting cue`。recommended if any signal = `1`。你为什么把某个 signal 打成 `1`？复制原文证据。例：`wait, no`、`actually`、`flip it over`、`then`、`[Pause]`。
- `Reject` (`Counterexample_or_Exclusion`) = `Evidence Log 2: Rejected cue`。optional。有什么词看起来像 cue，但你决定不算？没有就留空。例：`looks like = visual comparison, not A_t`；`no = answer choice label, not M_t`。
- `Issue` (`Codebook_Issue`) = `Evidence Log 3: Rule question`。optional。这个例子是否暴露出规则/vocabulary 需要讨论？没有就留空。例：`wait is ambiguous: M_t or R_t?`；`boundary valid but no signal definition fits`。

多数普通行只需要填第一个盒子 `Cue_Span`；第二和第三个盒子只在有排除项或规则问题时填写。
- `Notes`：optional。其他说明。

### Step 3. Evaluate Multi-Annotator Labels

用途：评估多人标注一致性，检查 signal 定义是否清楚。

1. 打开 `Step 3 Agreement` tab。
2. 上传完成标注的 Excel 或 CSV。
3. 点击 `Evaluate Multi-Annotator Labels`。

输出包括：

- 每个 signal 的 completed label count。
- positive rate。
- multi-annotated item count。
- exact agreement。
- 两个 annotator 时的 Cohen's kappa。
- item-level disagreement preview。
- downloadable agreement report workbook。

这些结果用于判断 vocabulary 是否可靠，以及哪些 signal 需要重新定义或补充 pattern。

### Step 4. Analyze Cue Lexicon

用途：从人工标注中计算候选 cue 的 support、precision、recall 和 lift，并导出推荐 vocabulary。

1. 打开 `Step 4 Cue Lexicon` tab。
2. 上传完成标注的 Excel 或 CSV。
3. 设置筛选阈值：
   - `Minimum cue support`：建议初始值 `5`。
   - `Minimum precision`：建议初始值 `0.65`。
   - `Minimum lift`：建议初始值 `1.5`。
4. 点击 `Analyze Cues and Export Vocabulary`。

输出包括：

- cue analysis workbook。
- recommended vocabulary JSON。
- recommended cue preview。

### Step 5. SQLite History

用途：审计 calibration 过程，复现每一步操作。

1. 打开 `Step 5 History` tab。
2. 设置显示行数。
3. 点击 `Refresh SQLite History`。

SQLite 记录包括：

- timestamp
- operation
- input path
- output path
- parameters JSON
- status
- message

除了 append-only history，Lab 还会把每一步最新生成的数据保存为 SQLite 固定表。下一次运行同一步时，这些表会被替换：

```text
step0_segments
step0_boundaries
step0_cleaned_input
step0_run_metadata
step1_prevalidation_summary
step1_signal_saturation_summary
step2_annotation_items
step2_codebook
step3_completed_labels
step3_signal_agreement
step3_item_agreement
step3_disagreement_items
step4_labeled_rows_for_cue_analysis
step4_recommended_cues
step4_vocab_summary
current_artifacts
```

其中 `current_artifacts` 保存每一步当前最新文件路径、参数和状态；`calibration_events` 保留完整历史，不会被替换。

刷新页面时，Lab 会优先读取这些 SQLite current tables 来恢复界面状态；只有点击 `Run Initial DAPS Segmentation` 才会重新 segment 并替换 Step 0 当前表。

## 6. 推荐研究流程

建议不要直接靠直觉定 vocabulary。更稳的流程是：

1. 用当前 DAPS 生成 preliminary segmentation。
2. 在 Calibration Lab 中抽样 300-500 个 boundary context。
3. 至少两个 annotator 标注 `C_t / M_t / A_t / R_t`。
4. 计算 agreement 和 kappa。
5. 检查 disagreement items，修订 codebook。
6. 基于人工标签计算 cue precision / recall / lift。
7. 更新 vocabulary。
8. 用新 vocabulary 重跑 DAPS。

## 7. 当前建议 vocabulary 原则

优先使用 high-precision cue，避免过宽词。

建议从 `M_t` 中谨慎使用或移除：

```text
know, no
```

建议从 `A_t` 中谨慎使用或移除：

```text
like, good, bad
```

更可靠的做法是把歧义词做成 contextual pattern，例如：

```text
I don't know
now I know
no, wait
no, actually
looks like  -> not affective
I like      -> weak affective
```

## 8. 常见输出位置

主切分输出通常在：

```text
data\
data\ui_outputs\
```

Calibration 输出通常在：

```text
data\calibration_outputs\
```
