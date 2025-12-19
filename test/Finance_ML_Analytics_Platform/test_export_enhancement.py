"""
Quick test to verify export_predictions_to_csv works with export_all_columns parameter
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from finance_ml.ml_workflow.analytics.eval import export_predictions_to_csv


def test_export_all_columns():
    """Test that export_all_columns=True exports all dataframe columns"""
    # Create test dataframe with many columns
    test_data = {
        "ticker": ["AAPL", "GOOGL", "MSFT"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "last_price": [150.0, 2800.0, 300.0],
        "price_target": [170.0, 3000.0, 350.0],
        "predicted_price_target": [165.0, 2950.0, 340.0],
        "market_cap": [2500000, 1800000, 2200000],
        "mispricing_score": [0.10, 0.05, 0.13],
        "mispricing_pct": [10.0, 5.0, 13.0],
        # Additional columns that should be exported
        "prediction_error": [5.0, 50.0, 10.0],
        "prediction_error_pct": [2.9, 1.7, 2.9],
        "model_analyst_diff_pct": [2.9, -1.7, -2.9],
        "p_e": [25.5, 28.3, 30.1],
        "p_b": [10.2, 5.5, 12.0],
        "roe": [0.35, 0.28, 0.40],
        "ev_ebitda": [15.5, 18.2, 20.1],
    }

    df = pd.DataFrame(test_data)

    # Test 1: Default behavior (limited columns)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)

    try:
        export_predictions_to_csv(df, csv_path, export_all_columns=False)
        df_limited = pd.read_csv(csv_path)
        print(f"✓ Test 1 (export_all_columns=False): {len(df_limited.columns)} columns")
        print(f"  Columns: {list(df_limited.columns)}")
        assert len(df_limited.columns) <= 10, "Limited export should have <= 10 columns"
    finally:
        csv_path.unlink(missing_ok=True)

    # Test 2: Export all columns
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path = Path(f.name)

    try:
        export_predictions_to_csv(df, csv_path, export_all_columns=True)
        df_all = pd.read_csv(csv_path)
        print(f"✓ Test 2 (export_all_columns=True): {len(df_all.columns)} columns")
        print(f"  Columns: {list(df_all.columns)}")
        assert len(df_all.columns) == len(
            df.columns
        ), f"Expected {len(df.columns)} columns, got {len(df_all.columns)}"

        # Verify key analytical columns are present
        required_cols = [
            "prediction_error",
            "prediction_error_pct",
            "model_analyst_diff_pct",
            "p_e",
            "roe",
        ]
        for col in required_cols:
            assert col in df_all.columns, f"Column '{col}' should be in exported CSV"
        print(f"✓ All required analytical columns present")

    finally:
        csv_path.unlink(missing_ok=True)

    print("\n✅ All export tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_export_all_columns()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
