# Final audit (Day 13)

Automatic part: `python scripts/audit_submission.py` (add `--report docs/report/report.pdf`). Fix every FAIL.
Read every WARN and decide. Then do the manual part below, in a **fresh clone**.

## Timeline
| Day | Do |
|---|---|
| 11 | Case study run (`python -m evaluation.case_study ...`); export the demo run (`python scripts/export_demo_run.py`); write the report from `scripts/report_facts.py` |
| 12 | Judging rehearsal (`docs/JUDGING_REHEARSAL.md`); record the video (`docs/DEMO_SCRIPT.md`); freeze features |
| 13 | Fresh clone, install from README only; audit; tag `git tag v1.0-submission`; push |

## Manual checklist
- [ ] Fresh clone installs and shows the dashboard using only the README
- [ ] "Load cached demo" works with the network switched off
- [ ] Dashboard cards, situation report and report tables show the same numbers
- [ ] The demo run is real (no MOCK banner)
- [ ] The report is at most 6 pages and has a Limitations section
- [ ] The video is under 3:00 and has no images of victims
- [ ] Attribution lines and citations are in README, report, dashboard and situation report
- [ ] The reference-map credit appears wherever its comparison is shown
- [ ] No secrets, raw data, checkpoints or files over 5 MB in git (`python scripts/check_repo_hygiene.py --all`)
- [ ] Nepali report either reviewed by a native speaker (`report.nepali_reviewed: true`) or clearly marked unreviewed
- [ ] After today: **only critical bug fixes**
