"""Runs T3's offline self-tests (they live inside gis/validation.py) with the main suite."""

import unittest

try:
    from gis.validation import TestPart05  # noqa: F401  (unittest discovers the imported TestCase)
except ImportError as exc:                 # geopandas/shapely missing on this machine
    raise unittest.SkipTest(f"gis.validation needs the geo stack: {exc}")
