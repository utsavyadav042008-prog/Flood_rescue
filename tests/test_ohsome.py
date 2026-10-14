"""Runs T1's offline ohsome tests (they live inside acquisition/ohsome.py) with the main suite."""

from acquisition.ohsome import TestOsm  # noqa: F401  (unittest discovers the imported TestCase)
