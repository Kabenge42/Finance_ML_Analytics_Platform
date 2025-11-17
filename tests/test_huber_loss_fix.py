"""
Test script to verify Huber loss ValueError fix in realistic scenario.
Simulates the notebook Phase 9.5.1 execution that was failing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from finance_ml.models import train_and_evaluate_regression


def test_huber_loss_with_real_data_scenario():
    """Test Huber loss with realistic data including missing values."""
    print("=" * 80)
    print("Testing Huber Loss with Realistic Data Scenario")
    print("=" * 80)

    # Create realistic dataset similar to all_stocks_phase95
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:04d}" for i in range(n_samples)],
            "sector": np.random.choice(
                ["Technology", "Finance", "Healthcare", "Energy", "Industrials"], n_samples
            ),
            "market_cap": np.random.lognormal(20, 2, n_samples),
            "last_price": np.random.uniform(10, 500, n_samples),
            "price_target": np.random.uniform(10, 600, n_samples),
            # Features with realistic missing values
            "revenue": np.random.lognormal(18, 2, n_samples),
            "ebitda": np.random.lognormal(17, 2, n_samples),
            "net_income": np.random.lognormal(16, 2, n_samples),
            "total_assets": np.random.lognormal(20, 2, n_samples),
            "pe_ratio": np.random.uniform(5, 50, n_samples),
            "pb_ratio": np.random.uniform(0.5, 10, n_samples),
        }
    )

    # Introduce missing values (realistic scenario that caused the issue)
    missing_rate = 0.15  # 15% missing values
    for col in ["revenue", "ebitda", "net_income", "total_assets", "pe_ratio", "pb_ratio"]:
        mask = np.random.random(n_samples) < missing_rate
        df.loc[mask, col] = np.nan

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Features with NaN:")
    for col in df.columns:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"      {col}: {nan_count} ({nan_count/len(df)*100:.1f}%)")

    # Create output directory
    out_dir = Path("outputs/regression")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔧 Training regression model with Huber loss...")
    print(f"   This previously failed with: ValueError: Input X contains NaN")

    try:
        result = train_and_evaluate_regression(
            df=df, out_dir=out_dir, n_jobs=1, loss="huber"  # This was causing the ValueError
        )

        if result:
            print(f"\n✅ SUCCESS! Training completed without ValueError")
            print(f"\n📈 Regression Metrics (Huber Loss):")
            print(f"   MAE:  {result['mae']:.2f}")
            print(f"   RMSE: {result['rmse']:.2f}")
            print(f"   R²:   {result['r2']:.4f}")

            # Check predictions
            if "predictions" in result:
                preds_df = result["predictions"]
                print(f"\n✓ Predictions generated: {len(preds_df)} rows")
                print(f"   Columns: {list(preds_df.columns)}")

            print(f"\n✅ FIX VERIFIED: NaN imputation working correctly")
            return True
        else:
            print(f"\n⚠️  Training returned None (insufficient data or dry_run)")
            return False

    except ValueError as e:
        if "Input X contains NaN" in str(e):
            print(f"\n❌ FAILED: ValueError still occurs")
            print(f"   Error: {e}")
            return False
        else:
            raise


if __name__ == "__main__":
    success = test_huber_loss_with_real_data_scenario()
    print("\n" + "=" * 80)
    if success:
        print("✅ All tests passed - Huber loss ValueError fix working correctly")
        sys.exit(0)
    else:
        print("❌ Tests failed - Issue not fully resolved")
        sys.exit(1)
