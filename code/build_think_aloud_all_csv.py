"""
Build clean CSVs from raw text exported from data/Think-aloud-All documents.

Expected input:
    data/Think_aloud_All_raw_text.jsonl

Outputs:
    data/Think_aloud_All_documents.csv
    data/Think_aloud_All_turns.csv
    data/Think_aloud_All_events.csv
    data/Think_aloud_All_extraction_report.md
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_JSONL = DATA / "Think_aloud_All_raw_text.jsonl"
DOCS_CSV = DATA / "Think_aloud_All_documents.csv"
TURNS_CSV = DATA / "Think_aloud_All_turns.csv"
EVENTS_CSV = DATA / "Think_aloud_All_events.csv"
REPORT = DATA / "Think_aloud_All_extraction_report.md"

SPEAKER_RE = re.compile(r"\b(Interviewer|Interviewee)(?:\s+\d+)?\s*:", re.IGNORECASE)
EVENT_RE = re.compile(
    r"\[(?:Pause|Drawing|End of Audio|Puzzle|Unintelligible|Laughter|Sighing|Rustling|[^\]]*paper[^\]]*|[^\]]*background[^\]]*)[^\]]*\]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
MOJIBAKE_RE = re.compile(r"(?:\u00ef\u00bf\u00bd)+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def mentioned_id(text: str) -> str:
    match = re.search(r"\b(ET[E]?\d{2}[_-]?\d{3,5}|ET[_-]?\d{3,5})\b", text, re.IGNORECASE)
    return match.group(1).replace("-", "_").upper() if match else ""


def source_id(stem: str) -> str:
    return stem.replace("-", "_").upper()


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\u00a0", " ")
    text = CONTROL_RE.sub("\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def artifact_count(text: str) -> int:
    text = text or ""
    return len(MOJIBAKE_RE.findall(text)) + len(CONTROL_RE.findall(text)) + text.count("\ufffd")


def strip_events(text: str) -> tuple[str, list[dict]]:
    events = []

    def replace(match: re.Match) -> str:
        event_text = match.group(0).strip()
        event_type_match = re.match(r"\[([A-Za-z ]+)", event_text)
        event_type = event_type_match.group(1).strip().title() if event_type_match else "Event"
        events.append(
            {
                "Event_Index": len(events) + 1,
                "Event_Type": event_type,
                "Event_Text": event_text,
                "Char_Start": match.start(),
                "Char_End": match.end(),
            }
        )
        return " <EVENT> "

    cleaned = EVENT_RE.sub(replace, text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"(?:\s*<EVENT>\s*)+", " <EVENT> ", cleaned).strip()
    return cleaned, events


def parse_turns(text: str) -> list[dict]:
    matches = list(SPEAKER_RE.finditer(text))
    turns = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        speaker = match.group(1).capitalize()
        raw_turn_text = normalize_text(text[start:end])
        turn_text, events = strip_events(raw_turn_text)
        if not turn_text:
            continue
        turns.append(
            {
                "Turn_Index": len(turns) + 1,
                "Speaker": speaker,
                "Turn_Text": turn_text,
                "Raw_Turn_Text": raw_turn_text,
                "Word_Count": word_count(turn_text),
                "Event_Count": len(events),
                "Unintelligible_Count": sum(1 for event in events if "unintelligible" in event["Event_Text"].lower()),
                "Timestamp_Count": len(re.findall(r"\b\d{1,2}:\d{2}\b", turn_text)),
                "Events": events,
            }
        )
    return turns


def load_raw_records() -> list[dict]:
    records = []
    with RAW_JSONL.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    raw_records = load_raw_records()
    doc_rows = []
    turn_rows = []
    event_rows = []

    for rec in raw_records:
        text = normalize_text(rec.get("text", ""))
        participant_id = source_id(rec.get("stem", ""))
        text_id = mentioned_id(text)
        turns = parse_turns(text)
        interviewee_turns = [t for t in turns if t["Speaker"] == "Interviewee"]
        interviewer_turns = [t for t in turns if t["Speaker"] == "Interviewer"]
        combined_interviewee = " | ".join(t["Turn_Text"] for t in interviewee_turns)

        doc_rows.append(
            {
                "ID": participant_id,
                "Mentioned_ID": text_id,
                "Source_File": rec.get("file_name", ""),
                "Source_Path": rec.get("source_path", ""),
                "Status": rec.get("status", ""),
                "Error": rec.get("error", ""),
                "Text_Chars": len(text),
                "Text_Words": word_count(text),
                "Turn_Count": len(turns),
                "Interviewee_Turns": len(interviewee_turns),
                "Interviewer_Turns": len(interviewer_turns),
                "Interviewee_Words": sum(t["Word_Count"] for t in interviewee_turns),
                "Unintelligible_Count": len(re.findall(r"unintelligible|\[uninte", text, re.IGNORECASE)),
                "Timestamp_Count": len(re.findall(r"\b\d{1,2}:\d{2}\b", text)),
                "Artifact_Count": artifact_count(text),
                "Interviewee_Response": combined_interviewee,
                "Full_Text": text,
            }
        )

        for turn in turns:
            turn_events = turn.pop("Events")
            turn_rows.append(
                {
                    "ID": participant_id,
                    "Mentioned_ID": text_id,
                    "Source_File": rec.get("file_name", ""),
                    **turn,
                }
            )
            for event in turn_events:
                event_rows.append(
                    {
                        "ID": participant_id,
                        "Mentioned_ID": text_id,
                        "Source_File": rec.get("file_name", ""),
                        "Turn_Index": turn["Turn_Index"],
                        "Speaker": turn["Speaker"],
                        **event,
                    }
                )

    docs_df = pd.DataFrame(doc_rows)
    turns_df = pd.DataFrame(turn_rows)
    events_df = pd.DataFrame(event_rows)

    docs_df.to_csv(DOCS_CSV, index=False, encoding="utf-8-sig")
    turns_df.to_csv(TURNS_CSV, index=False, encoding="utf-8-sig")
    events_df.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")

    report = [
        "# Think-aloud-All Extraction Report",
        "",
        f"- Input JSONL: `{RAW_JSONL.relative_to(ROOT)}`",
        f"- Source documents: {len(raw_records)}",
        f"- Successful documents: {int((docs_df.Status == 'ok').sum())}",
        f"- Error documents: {int((docs_df.Status != 'ok').sum())}",
        f"- Document-level CSV: `{DOCS_CSV.relative_to(ROOT)}`",
        f"- Turn-level CSV: `{TURNS_CSV.relative_to(ROOT)}`",
        f"- Event-level CSV: `{EVENTS_CSV.relative_to(ROOT)}`",
        "",
        "## Summary",
        "",
        f"- Total parsed speaker turns: {len(turns_df)}",
        f"- Extracted event markers: {len(events_df)}",
        f"- Interviewee turns: {int((turns_df.Speaker == 'Interviewee').sum()) if not turns_df.empty else 0}",
        f"- Interviewer turns: {int((turns_df.Speaker == 'Interviewer').sum()) if not turns_df.empty else 0}",
        f"- Total interviewee words: {int(docs_df.Interviewee_Words.sum())}",
        f"- Documents with zero interviewee turns: {int((docs_df.Interviewee_Turns == 0).sum())}",
        f"- Documents with artifacts/control chars: {int((docs_df.Artifact_Count > 0).sum())}",
        f"- Total artifact/control count: {int(docs_df.Artifact_Count.sum())}",
        f"- Documents with unintelligible markers: {int((docs_df.Unintelligible_Count > 0).sum())}",
        f"- Total unintelligible markers: {int(docs_df.Unintelligible_Count.sum())}",
        "",
        "## Notes",
        "",
        "- `Interviewee_Response` joins all interviewee turns with ` | ` for compatibility with the existing segmenter.",
        "- Event markers such as `[Pause ...]`, `[Drawing ...]`, and `[End of Audio]` are removed from turn text and stored in the event CSV. A `<EVENT>` marker is retained as a segmentation boundary cue.",
        "- `Think_aloud_All_turns.csv` preserves one row per speaker turn and is the safer input for later speaker-aware processing.",
        "- Files were read through Microsoft Word COM, which preserves old `.doc` text much better than the earlier broken CSV export.",
    ]
    REPORT.write_text("\n".join(report), encoding="utf-8")

    print(REPORT)
    print(DOCS_CSV)
    print(TURNS_CSV)
    print(EVENTS_CSV)
    print(docs_df[["ID", "Mentioned_ID", "Source_File", "Text_Words", "Interviewee_Turns", "Interviewee_Words", "Artifact_Count"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
