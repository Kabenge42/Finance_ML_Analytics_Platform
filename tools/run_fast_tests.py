"""
Run fast unit tests for helper modules with a reliable sys.path setup.

This script is designed for quick verification during development (Priority 6 —
light tests and verification). It avoids heavy ML training and focuses on small
utility modules added in recent optimization work:

- Conformal uncertainty intervals
- Robust outlier safety helpers
- Sector-specific feature engineering
- Sector-level prediction calibration

Usage (Windows PowerShell):
  python tools\run_fast_tests.py

Exit code is non-zero if any test fails.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def _ensure_project_root_on_sys_path() -> None:
    # This file lives under tools/. Project root is parent of this file's parent
    tools_dir = Path(__file__).resolve().parent
    project_root = tools_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def main() -> int:
    """Run fast unit tests for utility modules."""
    _ensure_project_root_on_sys_path()
    # Explicit test modules to keep runtime minimal
    test_modules = [
        "tests.unit.regression.test_uncertainty_conformal",
        "tests.unit.preprocessing.test_robust_outlier_safety",
        "tests.unit.features.test_sector_specific_features",
        "tests.unit.regression.test_sector_calibration",
    ]
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for mod in test_modules:
        try:
            suite.addTests(loader.loadTestsFromName(mod))
        except Exception as e:
            print(f"Error loading {mod}: {e}")
            return 2

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
