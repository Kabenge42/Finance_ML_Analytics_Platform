"""
Test script to verify the filter_stocks_by_criteria fix for Step 3.5

This script tests the corrected parameters:
- min_market_cap=1 with cap_unit="B" (1 billion minimum)
- min_mispricing=-10.0 (allow up to 10% overvalued)
- max_mispricing=None (no upper limit)
"""

import pandas as pd
import numpy as np
from finance_ml.ml_workflow.analytics.eval import filter_stocks_by_criteria


def create_test_data():
    """Create synthetic test data with various market caps and mispricing values"""
    np.random.seed(42)

    data = {
        "ticker": [f"TICK{i}" for i in range(100)],
        "sector": np.random.choice(["Technology", "Healthcare", "Finance", "Energy"], 100),
        "region": np.random.choice(["US", "EU", "APAC", "ROTW"], 100),
        "market_cap": np.random.uniform(100e6, 500e9, 100),  # 100M to 500B
        "mispricing_pct": np.random.uniform(-50, 100, 100),  # -50% to +100%
        "last_price": np.random.uniform(10, 500, 100),
    }

    return pd.DataFrame(data)


def test_original_bug():
    """Test the original bug scenario: unit mismatch"""
    print("\n" + "=" * 70)
    print("TEST 1: Original Bug - Unit Mismatch (10M instead of 10B)")
    print("=" * 70)

    df = create_test_data()

    # Original buggy call (as described in issue)
    # min_market_cap=10 with cap_unit="M" means 10 million (too small)
    filtered = filter_stocks_by_criteria(
        df,
        min_market_cap=10,
        cap_unit="M",  # Bug: should be "B" for billions
        min_mispricing=0.0,  # Only undervalued
    )

    print(f"Input stocks: {len(df)}")
    print(f"Filtered stocks (10M cap, undervalued only): {len(filtered)}")

    # This would match almost all stocks on market cap (since most > 10M)
    # but would fail if mispricing_pct column has many negative values

    if len(filtered) > 0:
        print(
            f"Market cap range: ${filtered['market_cap'].min()/1e9:.2f}B to ${filtered['market_cap'].max()/1e9:.2f}B"
        )
        print(
            f"Mispricing range: {filtered['mispricing_pct'].min():.1f}% to {filtered['mispricing_pct'].max():.1f}%"
        )


def test_fixed_version():
    """Test the fixed version with correct parameters"""
    print("\n" + "=" * 70)
    print("TEST 2: Fixed Version - 1B minimum, allow some overvalued")
    print("=" * 70)

    df = create_test_data()

    # Fixed version (as recommended in issue)
    filtered = filter_stocks_by_criteria(
        df,
        sectors=None,
        regions=["US", "EU"],
        min_market_cap=1,  # 1 billion minimum
        cap_unit="B",  # Correct: billions
        min_mispricing=-10.0,  # Allow up to 10% overvalued
        max_mispricing=None,
    )

    print(f"Input stocks: {len(df)}")
    print(f"Filtered stocks (1B cap, -10% to +∞ mispricing, US/EU): {len(filtered)}")

    if len(filtered) > 0:
        print(
            f"Market cap range: ${filtered['market_cap'].min()/1e9:.2f}B to ${filtered['market_cap'].max()/1e9:.2f}B"
        )
        print(
            f"Mispricing range: {filtered['mispricing_pct'].min():.1f}% to {filtered['mispricing_pct'].max():.1f}%"
        )
        print(f"Regions: {filtered['region'].unique()}")

        # Verify all conditions are met
        assert (filtered["market_cap"] >= 1e9).all(), "Market cap filter failed"
        assert (filtered["mispricing_pct"] >= -10.0).all(), "Mispricing filter failed"
        assert filtered["region"].isin(["US", "EU"]).all(), "Region filter failed"
        print("✅ All filter conditions validated!")
    else:
        print("⚠️  Warning: No stocks matched criteria")


def test_missing_market_cap_column():
    """Test behavior when market_cap column is missing"""
    print("\n" + "=" * 70)
    print("TEST 3: Missing market_cap Column")
    print("=" * 70)

    df = create_test_data()
    df_no_cap = df.drop(columns=["market_cap"])

    # Should silently skip market cap filtering
    filtered = filter_stocks_by_criteria(
        df_no_cap,
        min_market_cap=1,
        cap_unit="B",
        min_mispricing=-10.0,
    )

    print(f"Input stocks (no market_cap column): {len(df_no_cap)}")
    print(f"Filtered stocks (mispricing only): {len(filtered)}")

    if len(filtered) > 0:
        print(
            f"Mispricing range: {filtered['mispricing_pct'].min():.1f}% to {filtered['mispricing_pct'].max():.1f}%"
        )
        assert (filtered["mispricing_pct"] >= -10.0).all(), "Mispricing filter failed"
        print("✅ Correctly skipped market cap filter when column missing")


def test_diagnostic_logging():
    """Test the diagnostic logging code from the notebook"""
    print("\n" + "=" * 70)
    print("TEST 4: Diagnostic Logging (as added to notebook)")
    print("=" * 70)

    df = create_test_data()

    # Simulate the diagnostic code from the notebook
    print("\n📊 Pre-filter diagnostics:")
    print(f"  Total stocks: {len(df):,}")

    if "market_cap" in df.columns:
        mc = df["market_cap"].dropna()
        print(f"  Market cap available: {len(mc):,} stocks")
        if len(mc) > 0:
            print(f"    Range: ${mc.min()/1e9:.2f}B to ${mc.max()/1e9:.2f}B")
            print(f"    Median: ${mc.median()/1e9:.2f}B")
    else:
        print("  ⚠️  WARNING: 'market_cap' column not found!")

    if "mispricing_pct" in df.columns:
        mp = df["mispricing_pct"].dropna()
        print(f"  Mispricing available: {len(mp):,} stocks")
        if len(mp) > 0:
            print(f"    Range: {mp.min():.1f}% to {mp.max():.1f}%")
            print(f"    Median: {mp.median():.1f}%")
            print(
                f"    Undervalued (>0%): {(mp > 0).sum():,} stocks ({(mp > 0).sum()/len(mp)*100:.1f}%)"
            )
    else:
        print("  ⚠️  WARNING: 'mispricing_pct' column not found!")


def test_cap_unit_scaling():
    """Test that different cap_unit values scale correctly"""
    print("\n" + "=" * 70)
    print("TEST 5: Cap Unit Scaling (B, M, K)")
    print("=" * 70)

    df = create_test_data()

    # Test with billions
    filtered_b = filter_stocks_by_criteria(df, min_market_cap=1, cap_unit="B")
    print(f"min_market_cap=1, cap_unit='B' → {len(filtered_b)} stocks (>= $1B)")

    # Test with millions
    filtered_m = filter_stocks_by_criteria(df, min_market_cap=1000, cap_unit="M")
    print(f"min_market_cap=1000, cap_unit='M' → {len(filtered_m)} stocks (>= $1B)")

    # Both should produce same result (1B = 1000M)
    assert len(filtered_b) == len(filtered_m), "Billion and million scaling mismatch!"
    print("✅ Scaling works correctly: 1B == 1000M")

    # Test with thousands
    filtered_k = filter_stocks_by_criteria(df, min_market_cap=1000000, cap_unit="K")
    print(f"min_market_cap=1000000, cap_unit='K' → {len(filtered_k)} stocks (>= $1B)")
    assert len(filtered_k) == len(filtered_b), "Thousand scaling mismatch!"
    print("✅ Scaling works correctly: 1B == 1000M == 1,000,000K")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FILTER_STOCKS_BY_CRITERIA FIX VALIDATION")
    print("Testing the corrected Step 3.5 parameters")
    print("=" * 70)

    try:
        test_original_bug()
        test_fixed_version()
        test_missing_market_cap_column()
        test_diagnostic_logging()
        test_cap_unit_scaling()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nSummary of fix:")
        print("  - Added min_market_cap=1 with cap_unit='B' (1 billion minimum)")
        print("  - Set min_mispricing=-10.0 (allow up to 10% overvalued)")
        print("  - Added diagnostic logging to show data distribution")
        print("  - Function correctly handles missing columns with silent fallback")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
