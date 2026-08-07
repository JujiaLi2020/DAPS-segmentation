# DAPS Excel Segmentation

Run the minimal DAPS spreadsheet segmenter from the project root:

```powershell
.\.venv\Scripts\python.exe code\daps_excel_segmenter.py path\to\input.xlsx -o data\daps_segments.xlsx
```

By default the script reads the first worksheet, uses `ID` as the record ID when
available, and auto-detects transcript columns named `Interviewee_Response`,
`Transcript`, `Text`, `Content`, or `Item 1`, `Item 2`, etc.

If your Excel file uses different column names:

```powershell
.\.venv\Scripts\python.exe code\daps_excel_segmenter.py path\to\input.xlsx -o data\daps_segments.xlsx --id-column StudentID --text-columns Transcript
```

The output workbook contains:

- `segments`: final DAPS process units.
- `boundaries`: dense boundary-level evidence for every token gap.

The script automatically cleans common transcript encoding artifacts by default,
including sequences such as `ï¿½ï¿½ï¿½`, stray control-code quote markers, and
mis-decoded smart quotes. To disable this behavior:

```powershell
.\.venv\Scripts\python.exe code\daps_excel_segmenter.py path\to\input.csv -o data\daps_segments.xlsx --no-clean-text
```

Use `--embedding-model lexical` for a fast dependency-light smoke test. Use the
default `all-mpnet-base-v2` for embedding-based semantic continuity.
