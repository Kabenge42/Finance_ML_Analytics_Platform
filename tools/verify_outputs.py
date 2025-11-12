"""
Verify existence of expected output artifacts produced by the notebook/script.

Checks for the following files (Priority 7 — manual validation):
- outputs/regression/regression_predictions.csv
- outputs/regression/regression_metrics_by_sector.csv
- outputs/regression/quantile_predictions.csv
- outputs/regression/feature_importance.csv
- outputs/evaluation/tscv_metrics.csv

Usage (after running ml_finance_model_main.ipynb or ml_finance_model_main.py):
  python tools\verify_outputs.py

Exit code 0 if all required files exist and are non-empty; 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path


REQUIRED_FILES = {
    "Predictions": Path("outputs/regression/regression_predictions.csv"),
    "Sector Metrics": Path("outputs/regression/regression_metrics_by_sector.csv"),
    "Quantile Predictions": Path("outputs/regression/quantile_predictions.csv"),
    "Feature Importance": Path("outputs/regression/feature_importance.csv"),
    "TimeSeries CV Metrics": Path("outputs/evaluation/tscv_metrics.csv"),
}


def main() -> int:
    print("\n📁 Output Files Status:")
    all_ok = True
    for name, path in REQUIRED_FILES.items():
        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > 0:
                print(f"  ✓ {name}: {path} ({size} bytes)")
            else:
                print(f"  ✗ {name}: {path} exists but is empty (0 bytes)")
                all_ok = False
        else:
            print(f"  ✗ {name}: {path} (NOT FOUND)")
            all_ok = False

    if all_ok:
        print("\n✓ All expected output files are present and non-empty.")
        return 0
    else:
        print("\n⚠ Some expected output files are missing or empty. Rerun the notebook/script or check logs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
