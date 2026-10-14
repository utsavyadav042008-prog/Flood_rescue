# Written report: template (maximum 6 pages)

Save the final PDF as `docs/report/report.pdf` (the audit script looks there). About 400 to 450 words per page
with figures. **Never retype a number**: run `python scripts/report_facts.py data/sample/trishuli_run` and paste.

| Page | Owner | Content | Must include |
|---|---|---|---|
| 1 | T4 | Problem and our answer to the three questions; architecture in one diagram | the `docs/PART2_GUIDE.md` pipeline picture, simplified; what is new compared with a plain classifier |
| 2 | T1 | Data and preprocessing | Sentinel-1 same-orbit rule and why; dates/orbit of the Trishuli pair; DEM; OSM snapshot date and how it is guaranteed pre-event |
| 3 | T2 | The AI model | training data (cite Bountos et al., 2024), architecture, loss, metrics on **Himalayan scenes the model never saw**, failure examples |
| 4 | T3 + T4 | Damage and cut-off method; dashboard and situation report | how overlap and "affected" are defined; graph rule for "cut off" and the `unconnected_before` class; 1 dashboard screenshot |
| 5 | T4 + T2/T3 | Case study vs the official reference (check-only) | table from `report_facts.py`; `docs/figures/case_study_agreement.png`; where and **why** we disagree; where we are worse |
| 6 | T4 | Limitations, responsible use, attribution | content of `docs/limitations.md`; the three required attribution lines; dataset citations; reference credit |

## Writing rules (graded under "honesty about limitations")
- State results with their uncertainty ("IoU 0.xx on N held-out scenes"), not just the best case.
- Say what the system **cannot** do before a judge asks (revisit time, radar shadow, OSM gaps, no damage verification).
- The reference maps were **check-only**; say so once, clearly.
- No images of victims; respectful tone about a recent disaster.
- Figures need captions with data source and date. Add the attribution line under any figure that shows Sentinel or OSM data.

## Figures (suggested)
1. Pipeline diagram. 2. Before/after radar crop with our flood mask. 3. Dashboard screenshot with cut-off settlements.
4. Agreement map vs the reference (green agree, red only ours, blue only reference). 5. One failure case with explanation.

## Before exporting
Page count ≤ 6; `python scripts/audit_submission.py --report docs/report/report.pdf` shows PASS for the report check.
