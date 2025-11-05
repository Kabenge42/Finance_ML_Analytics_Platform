"""
Quick test to verify KNN imputation fix with real-like data
"""

import pandas as pd
import numpy as np
from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step

# Create sample data that mimics the real scenario
np.random.seed(42)
n_samples = 200

# Create data with multiple sectors including Energy
data = {
    "ticker": [f"STOCK{i}" for i in range(n_samples)],
    "sector": (
        ["Energy"] * 40
        + ["Technology"] * 50
        + ["Healthcare"] * 30
        + ["Financials"] * 40
        + [np.nan] * 40
    ),  # Some missing sectors
    "last_price": np.random.uniform(50, 500, n_samples),
    # KNN imputation columns with missing values
    "market_cap": np.where(
        np.random.rand(n_samples) > 0.7, np.nan, np.random.uniform(1e9, 1e12, n_samples)
    ),
    "enterprise_value": np.where(
        np.random.rand(n_samples) > 0.6, np.nan, np.random.uniform(1e9, 1e12, n_samples)
    ),
    "ebitda_ltm": np.where(
        np.random.rand(n_samples) > 0.65, np.nan, np.random.uniform(1e8, 1e10, n_samples)
    ),
    "revenue_ltm": np.where(
        np.random.rand(n_samples) > 0.7, np.nan, np.random.uniform(1e9, 1e11, n_samples)
    ),
    "total_assets_ltm": np.where(
        np.random.rand(n_samples) > 0.75, np.nan, np.random.uniform(1e9, 1e12, n_samples)
    ),
    # Zero imputation columns
    "impairment_of_goodwill_fq": np.where(
        np.random.rand(n_samples) > 0.95, np.random.uniform(0, 1e8, n_samples), np.nan
    ),
    "restructuring_charges_ltm": np.where(
        np.random.rand(n_samples) > 0.92, np.random.uniform(0, 5e7, n_samples), np.nan
    ),
    # Price target columns
    "price_target": np.where(
        np.random.rand(n_samples) > 0.5, np.nan, np.random.uniform(60, 600, n_samples)
    ),
}

df = pd.DataFrame(data)

print("=" * 80)
print("Testing KNN Imputation Fix with Real-like Data")
print("=" * 80)
print(f"\nInitial data shape: {df.shape}")
print(f"Missing values before: {df.select_dtypes(include=[np.number]).isna().sum().sum()}")
print(f"\nSector distribution:")
print(df["sector"].value_counts(dropna=False))

print("\n" + "=" * 80)
print("Applying Enhanced 4-Step Imputation Strategy...")
print("=" * 80)

try:
    df_imputed = apply_enhanced_imputation_strategy_4step(
        df, sector_column="sector", n_neighbors=5, price_column="last_price"
    )

    missing_after = df_imputed.select_dtypes(include=[np.number]).isna().sum().sum()

    print("\n" + "=" * 80)
    print("✓ SUCCESS - Imputation completed without errors!")
    print("=" * 80)
    print(f"Missing values after: {missing_after}")
    print(f"Reduction: {df.select_dtypes(include=[np.number]).isna().sum().sum() - missing_after}")

    if missing_after == 0:
        print("\n✓✓✓ Perfect! Zero missing values remaining!")
    else:
        print(
            f"\n⚠ Note: {missing_after} missing values still remain (expected for all-NaN columns)"
        )

except Exception as e:
    print("\n" + "=" * 80)
    print(f"✗ ERROR: {type(e).__name__}")
    print("=" * 80)
    print(f"Message: {e}")
    import traceback

    traceback.print_exc()
