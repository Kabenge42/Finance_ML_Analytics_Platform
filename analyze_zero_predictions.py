"""
Analyze prediction outputs to identify zero prediction issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_predictions(csv_path):
    """Analyze a predictions CSV file for zero predictions."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(csv_path).name}")
    print(f"{'='*80}")

    df = pd.read_csv(csv_path)

    print(f"\nDataset Overview:")
    print(f"  Total predictions: {len(df)}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")

    # Identify prediction column
    pred_cols = [c for c in df.columns if "pred" in c.lower() and "y_pred" in c.lower()]
    if not pred_cols:
        pred_cols = [c for c in df.columns if "pred" in c.lower()]

    if not pred_cols:
        print("  WARNING: No prediction column found!")
        return

    pred_col = pred_cols[0] if "y_pred" in pred_cols[0] else pred_cols[0]
    print(f"  Prediction column: {pred_col}")

    # Analyze zeros
    n_zeros = np.sum(df[pred_col] == 0)
    pct_zeros = 100 * n_zeros / len(df)

    print(f"\nZero Predictions:")
    print(f"  Count: {n_zeros} ({pct_zeros:.2f}%)")

    if n_zeros > 0:
        zero_mask = df[pred_col] == 0

        # Actual values for zeros
        if "y_true" in df.columns:
            y_true_zeros = df.loc[zero_mask, "y_true"]
            print(f"\nActual values (y_true) for zero predictions:")
            print(f"  Min: ${y_true_zeros.min():.2f}")
            print(f"  Max: ${y_true_zeros.max():.2f}")
            print(f"  Median: ${y_true_zeros.median():.2f}")
            print(f"  Mean: ${y_true_zeros.mean():.2f}")

        # Sector distribution
        if "sector" in df.columns:
            print(f"\nSector distribution of zero predictions:")
            sector_counts = df.loc[zero_mask, "sector"].value_counts()
            for sector, count in sector_counts.head(10).items():
                pct = 100 * count / n_zeros
                print(f"  {sector}: {count} ({pct:.1f}%)")

        # Region distribution
        if "region" in df.columns:
            print(f"\nRegion distribution of zero predictions:")
            region_counts = df.loc[zero_mask, "region"].value_counts()
            for region, count in region_counts.items():
                pct = 100 * count / n_zeros
                print(f"  {region}: {count} ({pct:.1f}%)")

        # Show examples
        print(f"\nExample zero predictions (first 10):")
        display_cols = ["ticker", "y_true", pred_col]
        if "sector" in df.columns:
            display_cols.append("sector")
        available_cols = [c for c in display_cols if c in df.columns]
        print(df.loc[zero_mask, available_cols].head(10).to_string(index=False))

    # Near-zero analysis
    n_near_zero = np.sum(df[pred_col] < 1)
    pct_near_zero = 100 * n_near_zero / len(df)
    print(f"\nNear-Zero Predictions (<$1):")
    print(f"  Count: {n_near_zero} ({pct_near_zero:.2f}%)")

    # Prediction range
    print(f"\nPrediction Range:")
    print(f"  Min: ${df[pred_col].min():.2f}")
    print(f"  Max: ${df[pred_col].max():.2f}")
    print(f"  Mean: ${df[pred_col].mean():.2f}")
    print(f"  Median: ${df[pred_col].median():.2f}")


def main():
    print("=" * 80)
    print("ZERO PREDICTIONS ANALYSIS - ACTUAL OUTPUT FILES")
    print("=" * 80)

    # Analyze each output file
    files = [
        "outputs/regression/regression_predictions_detailed.csv",
        "outputs/regression/regression_predictions.csv",
        "outputs/regression/quantile_predictions.csv",
    ]

    for file_path in files:
        if Path(file_path).exists():
            try:
                analyze_predictions(file_path)
            except Exception as e:
                print(f"\nERROR analyzing {file_path}: {e}")
        else:
            print(f"\nWARNING: {file_path} not found")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
