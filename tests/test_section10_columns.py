"""Test script to verify Section 10 column references work with schema."""

import pandas as pd
import numpy as np
from finance_ml.ml_workflow.analytics.ml_returns import create_ml_return_features

# Create sample data with schema-aligned column names
np.random.seed(42)
n_stocks = 50
n_days = 100

# Schema columns: "1-Day %" → 1_day_pct, "Last Price" → last_price
sample_data = pd.DataFrame(
    {
        "ticker": [f"STOCK{i}" for i in range(n_stocks)],
        "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_stocks),
        "1_day_pct": np.random.normal(0.001, 0.02, n_stocks),  # Schema: "1-Day %"
        "last_price": np.random.uniform(50, 500, n_stocks),  # Schema: "Last Price"
        "market_cap": np.random.uniform(1e9, 1e12, n_stocks),
        "expected_return": np.random.normal(0.10, 0.05, n_stocks),
    }
)

print("=" * 80)
print("TEST: Section 10 Column References with Schema-Aligned Names")
print("=" * 80)

print("\n1. Test ml_returns.create_ml_return_features() with schema columns:")
print(f"   Input columns: {list(sample_data.columns)}")
print(f"   Expected return column: '1_day_pct' (from schema '1-Day %')")
print(f"   Expected price column: 'last_price' (from schema 'Last Price')")

try:
    # Test auto-detection (should find 1_day_pct and last_price)
    ml_features_df = create_ml_return_features(
        sample_data, lags=[1, 3, 6], technical_indicators=["momentum", "volatility"]
    )
    print(f"\n✅ SUCCESS: Created {ml_features_df.shape[1]} ML features")
    print(f"   Output shape: {ml_features_df.shape}")

    # Check for expected feature columns
    expected_features = [
        "return_lag_1",
        "return_lag_3",
        "return_lag_6",
        "momentum_10",
        "volatility_20",
    ]
    found_features = [col for col in expected_features if col in ml_features_df.columns]
    print(f"   Features found: {found_features}")

    if len(found_features) == len(expected_features):
        print("✅ All expected features created successfully")
    else:
        print(f"⚠️  Only {len(found_features)}/{len(expected_features)} features found")

except KeyError as e:
    print(f"\n❌ FAILED with KeyError: {e}")
    print("   The auto-detection did not work as expected")
except Exception as e:
    print(f"\n❌ FAILED with error: {type(e).__name__}: {e}")

print("\n2. Test with legacy column names (backward compatibility):")
sample_data_legacy = sample_data.copy()
sample_data_legacy["return_1d"] = sample_data_legacy["1_day_pct"]

try:
    ml_features_legacy = create_ml_return_features(
        sample_data_legacy, lags=[1, 3], technical_indicators=["momentum"]
    )
    print(f"✅ SUCCESS: Legacy columns work - {ml_features_legacy.shape[1]} features")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")

print("\n3. Test explicit column specification:")
try:
    ml_features_explicit = create_ml_return_features(
        sample_data,
        lags=[1],
        technical_indicators=["momentum"],
        return_col="1_day_pct",
        price_col="last_price",
    )
    print(f"✅ SUCCESS: Explicit columns work - {ml_features_explicit.shape[1]} features")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("If all three tests show ✅ SUCCESS, the KeyError is resolved")
print("and Section 10 of the notebook should work with schema-aligned columns.")
