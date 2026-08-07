r"""
Gradio UI for DAPS Excel segmentation.

Launch from the project root:
    .\.venv\Scripts\python.exe code\daps_excel_ui.py
"""

from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import pandas as pd

from daps_excel_segmenter import (
    DAPSConfig,
    DAPSSegmenter,
    SimilarityModel,
    SignalVocabulary,
    clean_dataframe_text,
    infer_id_column,
    infer_text_columns,
    normalize_transcript,
    read_table,
)


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "ui_outputs"
DEFAULT_VOCABULARY = SignalVocabulary()

APP_CSS = """
.parameter-row {
    align-items: center;
    gap: 12px;
    margin: 4px 0;
}
.parameter-label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    font-size: 13px;
    margin: 0;
    min-width: 205px;
    white-space: nowrap;
}
.parameter-symbol {
    color: var(--body-text-color-subdued);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
}
.help-dot {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border: 1px solid var(--border-color-primary);
    border-radius: 50%;
    color: var(--body-text-color-subdued);
    font-size: 11px;
    line-height: 1;
    cursor: help;
}
.help-dot:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    z-index: 100;
    left: 50%;
    bottom: 130%;
    transform: translateX(-50%);
    width: 320px;
    padding: 10px 12px;
    border: 1px solid var(--border-color-primary);
    border-radius: 6px;
    background: var(--background-fill-primary);
    color: var(--body-text-color);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
    font-size: 12px;
    font-weight: 400;
    line-height: 1.35;
    white-space: normal;
}
.help-dot:hover::before {
    content: "";
    position: absolute;
    left: 50%;
    bottom: 108%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--border-color-primary);
}
"""

PARAMETER_HELP = {
    "Context width": ("w", "Formula symbol: w. Number of tokens on each side used to compute S_t, D_t, and process-shift signals around a candidate boundary. Suggested: w = 12."),
    "Local radius": ("r", "Formula symbol: r. Neighborhood radius for the adaptive local threshold around semantic-gravity valleys. Suggested: r = 6."),
    "Sensitivity": ("tau", "Formula symbol: tau. Multiplier in the local valley threshold; higher tau requires a deeper valley and usually produces fewer boundaries. Suggested: tau = 0.55."),
    "Minimum segment tokens": ("L_min", "Formula symbol: L_min. Minimum legal segment length during constrained decoding. Suggested: L_min = 12."),
    "Maximum segment tokens": ("L_max", "Formula symbol: L_max. Maximum preferred segment length; longer spans receive an additional legal split. Suggested: L_max = 60."),
    "Maximum segments per record": ("K_max", "Formula symbol: K_max. Record-level cap that merges adjacent short segments in extreme oversegmentation cases. Suggested: K_max = 80; use 0 to disable."),
    "NMS radius": ("rho", "Formula symbol: rho. Non-maximum suppression radius for competing nearby boundary candidates. Suggested: rho = 6."),
}

COGNITIVE_VOCAB_HELP = (
    "Used for C_t as left/right cue shift, not simple word presence. "
    "Best for action or strategy-change cues such as flip, rotate, compare, switch, instead, wait, actually. "
    "Avoid broad object words such as triangle, square, shape."
)


def parameter_label(name: str) -> gr.HTML:
    symbol, help_text = PARAMETER_HELP[name]
    tooltip = html.escape(help_text, quote=True)
    return gr.HTML(
        f'<div class="parameter-label">{html.escape(name)} '
        f'<span class="parameter-symbol">({html.escape(symbol)})</span>'
        f'<span class="help-dot" data-tooltip="{tooltip}">?</span></div>'
    )


def parameter_slider(name: str, minimum: float, maximum: float, value: float, step: float) -> gr.Slider:
    with gr.Row(elem_classes=["parameter-row"]):
        parameter_label(name)
        slider = gr.Slider(minimum, maximum, value=value, step=step, show_label=False, scale=1)
    return slider


def _path_from_upload(uploaded_file: str | Path | None) -> Path | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, dict):
        for key in ("path", "name", "orig_name"):
            value = uploaded_file.get(key)
            if value:
                return Path(value)
    if hasattr(uploaded_file, "path"):
        return Path(uploaded_file.path)
    if hasattr(uploaded_file, "name"):
        return Path(uploaded_file.name)
    return Path(uploaded_file)


def load_columns(uploaded_file: str | None, sheet_name: str):
    path = _path_from_upload(uploaded_file)
    if path is None:
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            "Upload an Excel or CSV file first.",
        )

    try:
        sheet = sheet_name.strip() or None
        df = read_table(path, sheet)
        id_column = infer_id_column(df, None)
        text_columns = infer_text_columns(df, None, id_column)
        preview = df.head(10)
        status = f"Loaded {path.name}: {len(df)} rows, {len(df.columns)} columns."
        return (
            preview,
            gr.update(choices=list(df.columns), value=id_column),
            gr.update(choices=list(df.columns), value=text_columns),
            status,
        )
    except Exception as exc:
        return (
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            f"Could not read file: {exc}",
        )


def run_segmentation(
    uploaded_file: str | None,
    sheet_name: str,
    id_column: str | None,
    text_columns: list[str] | None,
    embedding_model: str,
    context_width: int,
    local_radius: int,
    sensitivity: float,
    min_segment_tokens: int,
    max_segment_tokens: int,
    max_segments_per_record: int,
    nms_radius: int,
    task_vocab: str,
    cognitive_vocab: str,
    metacognitive_vocab: str,
    affective_vocab: str,
    structural_vocab: str,
    clean_text: bool,
):
    path = _path_from_upload(uploaded_file)
    if path is None:
        return None, pd.DataFrame(), pd.DataFrame(), "Upload an Excel or CSV file first."

    try:
        sheet = sheet_name.strip() or None
        df = read_table(path, sheet)
        artifact_count = 0
        if clean_text:
            df, artifact_count = clean_dataframe_text(df)
        selected_id = id_column or infer_id_column(df, None)
        selected_text_columns = text_columns or infer_text_columns(df, None, selected_id)
        if isinstance(selected_text_columns, str):
            selected_text_columns = [selected_text_columns]

        segmenter = DAPSSegmenter(
            DAPSConfig(
                context_width=int(context_width),
                local_radius=int(local_radius),
                sensitivity=float(sensitivity),
                min_segment_tokens=int(min_segment_tokens),
                max_segment_tokens=int(max_segment_tokens),
                max_segments_per_record=int(max_segments_per_record),
                nms_radius=int(nms_radius),
                cognitive_drop_floor=0.35,
                cognitive_drop_ceiling=0.80,
                cognitive_semantic_weight=0.45,
                cognitive_cue_shift_weight=0.55,
                cognitive_semantic_gate_floor=0.25,
                legality_mode="hybrid",
                spacy_model="en_core_web_sm",
                semi_legal_penalty=0.18,
                sentence_boundary_boost=0.08,
            ),
            SimilarityModel(embedding_model.strip() or "lexical"),
            SignalVocabulary.from_texts(
                task=task_vocab,
                cognitive=cognitive_vocab,
                metacognitive=metacognitive_vocab,
                affective=affective_vocab,
                structural=structural_vocab,
            ),
        )

        all_segments = []
        all_boundaries = []
        for row_index, row in df.iterrows():
            record_id = row[selected_id] if selected_id else row_index + 1
            for source_column in selected_text_columns:
                text = normalize_transcript(row[source_column], clean_text=clean_text)
                if not text:
                    continue
                segments, boundaries = segmenter.segment(text)
                for segment in segments:
                    all_segments.append(
                        {
                            "Record_ID": record_id,
                            "Source_Row": row_index + 2,
                            "Source_Column": source_column,
                            **segment,
                        }
                    )
                for boundary in boundaries:
                    all_boundaries.append(
                        {
                            "Record_ID": record_id,
                            "Source_Row": row_index + 2,
                            "Source_Column": source_column,
                            **boundary,
                        }
                    )

        segments_df = pd.DataFrame(all_segments)
        boundaries_df = pd.DataFrame(all_boundaries)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / f"{path.stem}_daps_segments.xlsx"
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            segments_df.to_excel(writer, sheet_name="segments", index=False)
            boundaries_df.to_excel(writer, sheet_name="boundaries", index=False)

        status = (
            f"Done. Generated {len(segments_df)} segments and "
            f"{len(boundaries_df)} boundary evidence rows. "
            f"Cleaned {artifact_count} encoding artifact(s). "
            f"Vocabulary sizes: C={len(segmenter.vocabulary.cognitive)}, "
            f"M={len(segmenter.vocabulary.metacognitive)}, "
            f"A={len(segmenter.vocabulary.affective)}, "
            f"R={len(segmenter.vocabulary.structural)}, "
            f"Task={len(segmenter.vocabulary.task)}."
        )
        return str(output_path), segments_df.head(30), boundaries_df.head(30), status
    except Exception as exc:
        return None, pd.DataFrame(), pd.DataFrame(), f"Segmentation failed: {exc}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="DAPS Excel Segmenter") as app:
        gr.Markdown("# DAPS Excel Segmenter")
        gr.Markdown("Upload a transcript spreadsheet, choose columns, and export segmented process units.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Input Excel or CSV",
                    file_types=[".xlsx", ".xlsm", ".xls", ".csv"],
                    type="filepath",
                )
                sheet_name = gr.Textbox(label="Sheet name", placeholder="Blank = first sheet")
                load_button = gr.Button("Load Columns")
                id_dropdown = gr.Dropdown(label="ID column", choices=[], value=None, allow_custom_value=True)
                text_dropdown = gr.Dropdown(
                    label="Transcript column(s)",
                    choices=[],
                    value=[],
                    multiselect=True,
                    allow_custom_value=True,
                )
                embedding_model = gr.Radio(
                    label="Semantic model",
                    choices=["lexical", "all-mpnet-base-v2"],
                    value="lexical",
                )
                clean_text = gr.Checkbox(label="Clean encoding artifacts", value=True)

            with gr.Column(scale=1):
                context_width = parameter_slider("Context width", 4, 30, 12, 1)
                local_radius = parameter_slider("Local radius", 2, 20, 6, 1)
                sensitivity = parameter_slider("Sensitivity", 0.05, 1.5, 0.55, 0.05)
                min_segment_tokens = parameter_slider("Minimum segment tokens", 3, 40, 12, 1)
                max_segment_tokens = parameter_slider("Maximum segment tokens", 30, 120, 60, 5)
                max_segments_per_record = parameter_slider("Maximum segments per record", 0, 200, 80, 5)
                nms_radius = parameter_slider("NMS radius", 1, 15, 6, 1)
                run_button = gr.Button("Run DAPS Segmentation", variant="primary")

        with gr.Accordion("Signal vocabularies", open=False):
            gr.Markdown("Edit words separated by commas, spaces, or new lines. These vocabularies are used at run time.")
            task_vocab = gr.Textbox(
                label="Task-density vocabulary",
                value=DEFAULT_VOCABULARY.as_text("task"),
                lines=4,
            )
            with gr.Row():
                cognitive_vocab = gr.Textbox(
                    label="Cognitive transition vocabulary (C_t)",
                    value=DEFAULT_VOCABULARY.as_text("cognitive"),
                    info=COGNITIVE_VOCAB_HELP,
                    lines=5,
                )
                metacognitive_vocab = gr.Textbox(
                    label="Metacognitive reset vocabulary (M_t)",
                    value=DEFAULT_VOCABULARY.as_text("metacognitive"),
                    lines=5,
                )
            with gr.Row():
                affective_vocab = gr.Textbox(
                    label="Affective friction vocabulary (A_t)",
                    value=DEFAULT_VOCABULARY.as_text("affective"),
                    lines=5,
                )
                structural_vocab = gr.Textbox(
                    label="Rhetorical/structural break vocabulary (R_t)",
                    value=DEFAULT_VOCABULARY.as_text("structural"),
                    lines=5,
                )

        status = gr.Textbox(label="Status", interactive=False)
        preview = gr.Dataframe(label="Input preview", interactive=False)
        output_file = gr.File(label="Download output workbook")

        with gr.Tab("Segments"):
            segments_preview = gr.Dataframe(label="Segment preview", interactive=False)
        with gr.Tab("Boundary Evidence"):
            boundaries_preview = gr.Dataframe(label="Boundary preview", interactive=False)

        load_button.click(
            load_columns,
            inputs=[file_input, sheet_name],
            outputs=[preview, id_dropdown, text_dropdown, status],
        )
        file_input.change(
            load_columns,
            inputs=[file_input, sheet_name],
            outputs=[preview, id_dropdown, text_dropdown, status],
        )
        run_button.click(
            run_segmentation,
            inputs=[
                file_input,
                sheet_name,
                id_dropdown,
                text_dropdown,
                embedding_model,
                context_width,
                local_radius,
                sensitivity,
                min_segment_tokens,
                max_segment_tokens,
                max_segments_per_record,
                nms_radius,
                task_vocab,
                cognitive_vocab,
                metacognitive_vocab,
                affective_vocab,
                structural_vocab,
                clean_text,
            ],
            outputs=[output_file, segments_preview, boundaries_preview, status],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7861, css=APP_CSS)
