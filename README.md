# DAPS Think-Aloud Segmentation Toolkit

## Brief Introduction

DAPS is a lightweight toolkit for segmenting think-aloud transcripts into interpretable process units. It combines local semantic-continuity scoring, theory-guided process-shift signals, parser-assisted boundary legality filtering, and human-in-the-loop calibration. The toolkit is designed for research workflows where automatic segmentation should remain inspectable, auditable, and adjustable before being used for formal analysis.

The project includes two Gradio interfaces:

- **DAPS Excel Segmenter**: uploads CSV/Excel transcript files and exports `segments` and `boundaries` workbooks.
- **DAPS Calibration Lab**: runs the full calibration workflow, including initial segmentation, pre-validation, multi-annotator review, agreement evaluation, cue-vocabulary analysis, and SQLite logging.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The parser-assisted legality filter uses spaCy. The deployment requirements include the English spaCy model. If installing manually, use:

```powershell
.\.venv\Scripts\python.exe -m spacy download en_core_web_sm
```

If spaCy or the model is unavailable, DAPS falls back to the rule-based legality filter.

## Run Locally

### Calibration Lab

```powershell
.\.venv\Scripts\python.exe code\daps_calibration_lab.py
```

Open:

```text
http://127.0.0.1:8000
```

To use a different local port:

```powershell
$env:GRADIO_SERVER_PORT='8501'
.\.venv\Scripts\python.exe code\daps_calibration_lab.py
```

### Excel Segmenter

```powershell
.\.venv\Scripts\python.exe code\daps_excel_ui.py
```

Open:

```text
http://127.0.0.1:7861
```

## Railway Deployment

The repository includes Railway-compatible startup files:

```text
Procfile
railway.json
```

Railway starts the app with:

```text
python code/daps_calibration_lab.py
```

The app automatically reads Railway's `PORT` environment variable and binds to `0.0.0.0`.

Research data, SQLite databases, generated workbooks, manuscripts, proposal files, and local virtual environments are excluded from GitHub through `.gitignore`. Upload transcript files through the web app after deployment.

### Persistent SQLite Storage on Railway

To keep the Calibration Lab SQLite database after redeployments, attach a Railway Volume to the app service.

Recommended Railway volume setting:

```text
Mount path: /data
```

When the app detects Railway and the `/data` volume, it stores the current SQLite database at:

```text
/data/daps_calibration.sqlite3
```

Local runs still use:

```text
data/calibration_outputs/daps_calibration.sqlite3
```

You can also override the storage directory manually with:

```text
DAPS_OUTPUT_DIR=/data
```

## Expected Input Format

The Calibration Lab accepts CSV or Excel transcript files. Recommended columns are:

```text
interview_id
turn_number
text
```

Optional columns such as `source_file` or `speaker` can be retained as metadata. If transcripts contain interviewer turns, clean or filter them before segmentation unless they are analytically relevant.

Recommended Step 0 settings:

```text
ID column: interview_id
Item/task column: turn_number
Transcript column(s): text
Semantic continuity model: lexical
Clean encoding artifacts / mojibake: enabled
```

## DAPS Output

The segmentation workbook contains:

- `segments`: final process-unit segments.
- `boundaries`: evidence for each candidate token gap.
- `cleaned_input`: cleaned source rows used in the run.
- `run_metadata`: parameters and run summary.

The `segments` sheet includes:

- `Segment_Text`
- `Token_Start`
- `Token_End`
- `Token_Count`
- `Segment_Type`
- `Segment_Type_Confidence`
- `Mean_C_t`
- `Mean_M_t`
- `Mean_A_t`
- `Mean_R_t`

The `boundaries` sheet includes DAPS evidence and legality fields such as:

- `semantic_continuity`
- `transition_pressure`
- `raw_semantic_gravity`
- `semantic_gravity`
- `boundary_legality`
- `legality_reason`
- `rule_constraint_reason`
- `spacy_left_pos`
- `spacy_right_pos`
- `spacy_left_dep`
- `spacy_right_dep`

## Main Segmentation Parameters

- `Context width (w)`: number of tokens on each side of a candidate boundary. Suggested: `12`.
- `Local radius (r)`: neighborhood radius for adaptive local valley detection. Suggested: `6`.
- `Sensitivity (tau)`: higher values require deeper local valleys and usually produce fewer boundaries. Suggested: `0.55`.
- `Minimum segment tokens (L_min)`: minimum segment length. Suggested: `12`.
- `Maximum segment tokens (L_max)`: preferred maximum segment length before forced legal splitting. Suggested: `60`.
- `Maximum segments per record (K_max)`: record-level cap for extreme oversegmentation. Suggested: `80`.
- `NMS radius (rho)`: non-maximum suppression radius for nearby candidate boundaries. Suggested: `6`.

## Calibration Lab Workflow

### Step 0. Segment Raw Transcript Data

Upload a raw CSV/Excel file and run initial DAPS segmentation. The output workbook is automatically passed to Step 1 and Step 2.

### Step 1. Inspect Segmentation Workbook

Run pre-validation before annotation. The app reports:

- segment and boundary counts
- median and mean segment length
- short/long segment rates
- event or speaker-label leakage
- selected versus non-selected semantic gravity
- selected versus non-selected transition pressure
- signal saturation summaries
- number and rate of segments assigned to each signal group
- boundary legality diagnostics

Status values are shown as `PASS`, `WARNING`, `FAIL`, or `INFO`.

### Step 2. Generate Multi-Annotator Template

Generate a sampled boundary-context annotation template. The same sampled items are duplicated for each annotator listed in `Annotator IDs`.

The main human-label columns are:

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

`Boundary_Strength_0_3` is the single boundary-quality variable:

```text
0 = merge / not a boundary
1 = weak boundary
2 = moderate boundary
3 = strong boundary
```

The four signal labels are independent binary labels:

```text
Human_C_t: cognitive transition
Human_M_t: metacognitive reset
Human_A_t: affective friction
Human_R_t: rhetorical/structural break
```

### Step 2b. In-App Human Review

Use the guided review page instead of editing a wide spreadsheet. Work is separated by annotator:

- choose an `Annotator`
- review only that annotator's assigned rows
- monitor that annotator's completion table
- save each row to SQLite with `Save Current` or `Save and Next`

The app stores progress in:

```text
Railway: /data/daps_calibration.sqlite3
Local: data/calibration_outputs/daps_calibration.sqlite3
```

Rows are considered complete when these fields are filled:

```text
Boundary_Strength_0_3
Human_C_t
Human_M_t
Human_A_t
Human_R_t
```

### Step 3. Evaluate Multi-Annotator Labels

Upload a completed reviewed workbook and compute agreement summaries for:

- `Boundary_Strength_0_3`
- `C_t`
- `M_t`
- `A_t`
- `R_t`

The report includes completed label counts, positive rates, exact agreement, Cohen's kappa when there are two annotators, and item-level disagreement previews.

### Step 4. Analyze Cue Lexicon

Use completed annotations to identify candidate cue vocabulary using:

- support
- precision
- recall
- lift

The app exports a cue-analysis workbook and a recommended vocabulary JSON file.

### Step 5. SQLite History

The Lab logs each major operation to SQLite for reproducibility. It stores both append-only event history and current-step tables such as:

- `step0_segments`
- `step0_boundaries`
- `step1_prevalidation_summary`
- `step1_signal_saturation_summary`
- `step2_annotation_items`
- `step3_signal_agreement`
- `step4_recommended_cues`

## Command-Line Segmentation

You can also run DAPS without the UI:

```powershell
.\.venv\Scripts\python.exe code\daps_excel_segmenter.py data\input.csv -o data\daps_segments.xlsx --id-column interview_id --text-columns text --embedding-model lexical
```

Common options:

```text
--id-column interview_id
--text-columns text
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

## Notes

- `lexical` is the recommended semantic-continuity baseline for calibration and debugging.
- `all-mpnet-base-v2` can be used later for comparison after human labels exist.
- Parser-assisted legality filtering is used to reduce phrase-internal boundary errors; it does not replace DAPS scoring.
- Algorithmic segment labels are pre-labels only and should not be treated as final human-coded categories.
