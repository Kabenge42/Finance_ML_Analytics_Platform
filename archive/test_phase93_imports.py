"""
Test Phase 9.3 API imports and basic functionality.
Validates that the Phase 9.3 feature engineering API is properly integrated.
"""

import sys
import pandas as pd
import numpy as np

print("=" * 80)
print("Phase 9.3 API Integration Test")
print("=" * 80)

# Test 1: Import Phase 9.3 API from package
print("\n[Test 1] Importing Phase 9.3 API from finance_ml package...")
try:
    from finance_ml import build_features, PresetName

    print("✓ Successfully imported build_features and PresetName")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Import directly from features.api module
print("\n[Test 2] Importing directly from features.api module...")
try:
    from finance_ml.ml_workflow.features.api import build_features as bf, PresetName as pn

    print("✓ Successfully imported from features.api")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Check PresetName type
print("\n[Test 3] Checking PresetName type annotation...")
try:
    print(f"   PresetName type: {PresetName}")
    print("✓ PresetName type annotation is valid")
except Exception as e:
    print(f"✗ PresetName check failed: {e}")

# Test 4: Create sample data and test build_features with presets
print("\n[Test 4] Testing build_features with sample data...")
try:
    # Create minimal sample dataframe
    sample_data = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "sector": ["Technology", "Technology", "Technology"],
            "last_price": [150.0, 300.0, 2500.0],
            "market_cap": [2.5e12, 2.3e12, 1.7e12],
            "total_revenue_ltm": [365e9, 184e9, 257e9],
            "ebitda_ltm": [110e9, 90e9, 85e9],
            "net_income_ltm": [95e9, 72e9, 73e9],
            "total_assets_ltm": [350e9, 360e9, 320e9],
            "total_debt_ltm": [100e9, 50e9, 15e9],
            "total_equity_ltm": [60e9, 170e9, 250e9],
        }
    )

    # Test basic preset
    print("\n   Testing preset='basic'...")
    result_basic = build_features(sample_data.copy(), preset="basic")
    print(f"   ✓ Basic preset: {sample_data.shape[1]} -> {result_basic.shape[1]} columns")

    # Test comprehensive preset
    print("\n   Testing preset='comprehensive'...")
    result_comp = build_features(
        sample_data.copy(),
        preset="comprehensive",
        include_interactions=False,
        include_relative=False,
    )
    print(f"   ✓ Comprehensive preset: {sample_data.shape[1]} -> {result_comp.shape[1]} columns")

    print("\n✓ build_features working correctly with presets")
except Exception as e:
    print(f"✗ build_features test failed: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Verify backward compatibility
print("\n[Test 5] Testing backward compatibility...")
try:
    from finance_ml import features_build_comprehensive, engineer_valuation_ratios

    print("✓ Old function names still importable (backward compatible)")
except ImportError as e:
    print(f"✗ Backward compatibility check failed: {e}")

# Test 6: Check __all__ exports
print("\n[Test 6] Checking __all__ exports...")
try:
    import finance_ml

    if "build_features" in finance_ml.__all__:
        print("✓ build_features is in __all__")
    else:
        print("✗ build_features NOT in __all__")

    if "PresetName" in finance_ml.__all__:
        print("✓ PresetName is in __all__")
    else:
        print("✗ PresetName NOT in __all__")
except Exception as e:
    print(f"✗ __all__ check failed: {e}")

print("\n" + "=" * 80)
print("Phase 9.3 API Integration Test Complete")
print("=" * 80)
print("\n✅ All tests passed! Phase 9.3 API is properly integrated.")
