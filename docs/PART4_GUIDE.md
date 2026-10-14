# Part 4 Guide — Situation Report Generator

Builds on Parts 1–3. The report stage runs automatically at the end of `analyze_flood()`
and the dashboard's **Report** tab shows the result.

```bash
python scripts/make_report.py data/sample/mock_run --print    # regenerate from an existing run, in seconds
```
Sample output (clearly labelled MOCK): `sample_situation_report_MOCK.pdf`, `sample_situation_report_ne_MOCK.txt`.

---

## 1. What it produces (per run folder)

| File | Language | Contract status |
|---|---|---|
| `situation_report.txt` | English | **required** output of stage `report` |
| `situation_report.pdf` | English, one A4 page | optional |
| `situation_report.html` | English, printable, self-contained | optional |
| `situation_report_ne.txt` | Nepali | optional (config `report.languages`) |
| `situation_report_ne.html` | Nepali, printable | optional |

Report content (one page): title and disclaimer, MOCK banner if relevant, area / date / imagery /
OSM snapshot / model / time, five key figures, settlements likely cut off (max 10 + "and N more"),
**how to read this**, **data quality** (model confidence + processing notes), **limitations**,
and the three required attribution lines. If `figures/flood_map.png` exists it is embedded.

## 2. Architecture

```
summary.json --fmt()--> build_model() --> ReportModel --+--> render_text()  -> .txt
 (the only source        (ALL wording                   +--> render_html()  -> .html (en, ne)
  of numbers)             decisions)                    +--> render_pdf()   -> .pdf  (en)
                              ^
                     reports/strings.py  (all sentences, per language)
```

| File | Purpose |
|---|---|
| `reports/strings.py` | every sentence, English + Nepali. **Edit wording here, not in code.** |
| `reports/format.py` | `fmt()` number formatting (shared with the dashboard), exact attribution text, Nepali digits |
| `reports/generator.py` | `build_model`, `render_text/html/pdf`, `write_reports`, `run_report` (the stage entry point) |
| `reports/guard.py` | number guard: proves no number in a text was invented |
| `scripts/make_report.py` | regenerate a run's report without re-running the pipeline |

## 3. Function reference (`reports/generator.py`)

| Name | What it does |
|---|---|
| `build_model(summary, lang, nepali_reviewed, image)` | decides what is said: headings, KPI lines, capped lists, warnings, hedged wording |
| `render_text(model)` | wrapped plain text (88 columns) |
| `render_html(model)` | self-contained printable page; all text HTML-escaped; image embedded as base64 |
| `render_pdf(model, path)` | one-page A4 PDF with reportlab; shrinks content to fit |
| `generate_report(summary, lang)` | shortcut: summary → text |
| `write_reports(run, summary, cfg)` | writes every enabled file atomically |
| `run_report(run, aoi, date, cfg)` | the pipeline stage |
| `pdf_safe(text)` | replaces characters the built-in PDF font cannot print |

`reports/guard.py`: `unexpected_numbers(text, summary)` returns the numbers that neither the
summary nor the fixed wording explains (empty list = clean).

## 4. Clever tricks (not obvious at first glance)

1. **One `ReportModel`, three renderers.** Every decision (what to print when a value is missing, how many
   names fit, which warnings to show) is made once. The text, HTML and PDF only draw, so they cannot disagree.
2. **The "no invented numbers" promise is a test, not a hope.** `unexpected_numbers` extracts every number
   from a report (both digit systems, thousands separators removed) and checks each one against the summary,
   the fixed wording, or the derived "and N more" count. The tests run it on 5 summaries × 2 languages, and
   also prove it catches an invented "8,400 people". The same function can guard a future language-model rewrite.
3. **`None` and `0` read differently.** `None` → "not available" and never gets a unit ("not available km²" cannot
   happen); `0` cut-off settlements → a positive sentence ("No settlement without a road connection was found").
4. **Guaranteed one page.** The PDF content is wrapped in reportlab's `KeepInFrame(mode="shrink", vAlign="TOP")`,
   so too much content is scaled down instead of spilling to page 2. Lists are capped first (10 names, 6 notes,
   names shortened to 60 characters), so the shrink stays legible. A worst-case test (40 long names, 12 long
   notes, a map image) still gives exactly one page.
5. **PDF text safety.** The built-in Helvetica covers Windows-1252 only, so `pdf_safe` replaces other characters
   (such as Devanagari place names) with `?` and adds a one-line note. Text is also XML-escaped, because
   reportlab's `Paragraph` parses mini-XML (a name like `A & B <tag> ]]>` is tested).
6. **No Nepali PDF, on purpose.** Reportlab cannot shape Devanagari (conjuncts would render wrongly). The Nepali
   report is therefore text + printable HTML; the HTML uses a font stack with **Nirmala UI**, which ships with
   Windows 10, and the browser's Print → Save as PDF gives a correct Nepali PDF.
7. **Nepali digits, but legal text untouched.** Numbers are converted to Devanagari digits (२०२६-०८-२६), while the
   three attribution lines stay exactly as the brief requires. A test checks that no ASCII digit is left in the
   Nepali body.
8. **Translation honesty switch.** Until `report.nepali_reviewed: true` in `config/default.yaml`, the Nepali
   report prints "Translation not yet reviewed by a native speaker". Processing notes stay in English (they are
   written by the pipeline), and the report says so.
9. **Wording is policed by tests.** A test fails if "destroyed", "confirmed", "victims", "people affected", etc.
   ever appear in the English wording; both languages must define the same keys and the same `{placeholders}`.
10. **The report stage is never cached** (`ALWAYS_RERUN` in `pipeline/analyze.py`). Its inputs include settings that
    are not files (languages, the reviewed flag), so a cached copy could silently be out of date; it takes
    milliseconds anyway.
11. **Safe formatting.** Sentences use `str.format` on *our* templates only; values are inserted, never re-parsed,
    so a settlement named `{x}` cannot break anything. A test asserts no `{` or `}` is left in any output.
12. **Atomic writes** for every file including the PDF (temp file, then replace); a test checks no `.tmp` is left.
13. **MOCK data is labelled everywhere**: red banner in the PDF/HTML, a line in the text, and never duplicated.
14. **Readable timestamp**: `2026-10-07T10:00:00+00:00` → `2026-10-07 10:00 UTC`.
15. **Fast polishing loop**: edit `reports/strings.py` → `python scripts/make_report.py <run_dir>` → open the PDF.

## 5. How to improve the wording

Open `reports/strings.py`. Rules at the top of the file: never type a number into a sentence (use `{placeholders}`),
never claim destruction, casualties or population impact. After editing run
`python -m unittest tests.test_report_generator`; the guard and wording tests tell you if you broke a rule.

## 6. Agreements with teammates

- **T3:** `cutoff_names` in `connectivity_summary.json` should be **Latin-script** (English name or a
  transliteration). Non-Latin names print as `?` in the PDF (they are fine in the text/HTML versions).
  Counts must be real numbers (not strings), names a list.
- **T2:** `mean_confidence` is defined as the mean model probability over the pixels the mask calls flooded.
- **Nepali:** find a native speaker to review `STRINGS["ne"]` before the demo; then set `nepali_reviewed: true`.
- **Figures (not written yet):** if a `figures/flood_map.png` is produced it is embedded automatically.

## 7. Tested and NOT tested

- **Tested here (225 tests pass in the repo, 6 skipped):** formatting; every English/Nepali paragraph path;
  `None` vs `0`; list caps; the number guard (including catching an invented number); banned wording; HTML
  escaping; PDF generation verified with **pypdf** (1 page, numbers, attribution present, special characters,
  worst case, embedded image, no temp files); the stage through the pipeline (files pass the contract and
  numbers equal `summary.json`); the dashboard Report tab; the `make_report.py` script. I also looked at the
  rendered PDF as an image.
- **Not tested:** (1) how the Nepali HTML looks in a real browser (no Devanagari font in my sandbox; the Unicode
  text itself is correct); (2) the Nepali sentences have **not** been reviewed by a native speaker (my draft);
  (3) the PDF with a real figure from the (still unwritten) figures stage, only with a synthetic PNG;
  (4) Windows-specific behaviour (the code uses only portable calls).
- A 5-minute manual check: open `situation_report.pdf` from a run, confirm the key figures equal the dashboard
  cards, then open `situation_report_ne.html` in Chrome and check the Nepali text displays correctly.

## 8. Changes outside Part 4

- `pipeline/contracts.py`: four optional report artifacts (`situation_report.html`, `_ne.txt`, `_ne.html`; PDF
  already existed) in the `report` stage. `CONTRACT_VERSION` stays `1.0` (only additions).
- `config/default.yaml`: new `report:` section (`languages`, `nepali_reviewed`); not in the cache fingerprint.
- `pipeline/analyze.py`: `ALWAYS_RERUN = {"report"}`; the Part 2 tests were updated (the report now really runs).
- `dashboard/components.py`: Report tab has a language switch, PDF/HTML/text downloads and the Nepali print hint.
