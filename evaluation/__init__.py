"""Evaluation and case-study tooling (owner: T4, with T2/T3 inputs).

IMPORTANT DATA RULE: this is the ONLY package allowed to read check-only
reference data (published damage maps, post-event OSM edits).  Nothing in the
pipeline, the models, the GIS code, the dashboard or the report may import
``evaluation`` (``tests/test_evaluation.py`` enforces this).  Results leave
this package only as plain numbers in ``evaluation/case_study_results.json``.
"""
