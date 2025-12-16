"""Test script for dashboard enhancements."""

from pathlib import Path

import numpy as np
import pandas as pd


# Test data creation
def create_test_data():
    """Create test DataFrame mimicking equities_dash_df structure."""
    np.random.seed(42)
    n_stocks = 100

    data = {
        "ticker": [f"TICK{i:03d}" for i in range(n_stocks)],
        "name": [f"Company {i}" for i in range(n_stocks)],
        "sector": np.random.choice(
            ["Technology", "Healthcare", "Financials", "Energy", "Consumer"], n_stocks
        ),
        "region": np.random.choice(
            ["United States and Canada", "Europe", "Asia / Pacific"], n_stocks
        ),
        "country": np.random.choice(["US", "GB", "JP", "DE", "FR"], n_stocks),
        "trading_country": np.random.choice(["US", "GB", "JP"], n_stocks),
        "industry": np.random.choice(["Software", "Banks", "Oil & Gas"], n_stocks),
        "exchange": np.random.choice(["NYSE", "NASDAQ", "LSE"], n_stocks),
        "style_class": np.random.choice(["Growth", "Value", "Core"], n_stocks),
        "size_class": np.random.choice(["Large Cap", "Mid Cap", "Small Cap"], n_stocks),
        "last_price": np.random.uniform(10, 500, n_stocks),
        "price_target": np.random.uniform(15, 550, n_stocks),
        "market_cap": np.random.uniform(1e8, 1e12, n_stocks),
        "next_earnings": pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=30),
            end=pd.Timestamp.now() + pd.Timedelta(days=30),
            periods=n_stocks,
        ),
        "eps_adj_ltm": np.random.uniform(-5, 20, n_stocks),
        "eps_norm_est_avg_ntm": np.random.uniform(-3, 22, n_stocks),
    }

    return pd.DataFrame(data)


def test_export_function():
    """Test CSV export functionality."""
    from finance_ml.dashboards.equities_dashboard_app import export_equities_data

    df = create_test_data()
    test_output = Path("outputs/dashboards/equities_dashboard/test_export.csv")
    test_metadata = Path("outputs/dashboards/equities_dashboard/test_metadata.json")

    try:
        metadata = export_equities_data(df, test_output, test_metadata)
        print("[PASS] CSV export successful")
        print(f"  - Exported {metadata['row_count']} rows")
        print(f"  - File size: {metadata['file_size_mb']:.2f} MB")
        return True
    except Exception as e:
        print(f"[FAIL] CSV export failed: {e}")
        return False


def test_visualization_functions():
    """Test log-scaled visualization functions."""
    from finance_ml.dashboards.equities_dashboard_app import (
        _market_cap_distribution,
        _target_vs_price_scatter,
        create_earnings_events_chart,
    )

    df = create_test_data()

    try:
        # Test scatter plot
        fig1 = _target_vs_price_scatter(df, use_log_scale=True)
        assert fig1.data, "Scatter plot has no data"
        print("[PASS] Log-scaled scatter plot created")

        # Test market cap distribution
        fig2 = _market_cap_distribution(df)
        assert fig2.data, "Market cap distribution has no data"
        print("[PASS] Market cap distribution created")

        # Test earnings events chart
        fig3 = create_earnings_events_chart(df, days_window=30)
        print("[PASS] Earnings events chart created")

        return True
    except Exception as e:
        print(f"[FAIL] Visualization functions failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_artifact_generation():
    """Test artifact generation (may take longer)."""
    from finance_ml.dashboards.equities_dashboard_app import (
        generate_dashboard_artifacts,
    )

    df = create_test_data()
    test_dir = Path("outputs/dashboards/equities_dashboard/test_artifacts")
    test_metadata = Path(
        "outputs/dashboards/equities_dashboard/test_artifacts_metadata.json"
    )

    try:
        print("Generating artifacts (this may take a minute)...")
        metadata = generate_dashboard_artifacts(df, test_dir, test_metadata)
        artifacts_count = len(metadata.get("artifacts", {}))
        print(f"[PASS] Artifact generation successful")
        print(f"  - Generated {artifacts_count} artifacts")
        return True
    except Exception as e:
        print(f"[FAIL] Artifact generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Dashboard Enhancements")
    print("=" * 60)

    results = []

    print("\n1. Testing CSV Export...")
    results.append(test_export_function())

    print("\n2. Testing Visualization Functions...")
    results.append(test_visualization_functions())

    print("\n3. Testing Artifact Generation...")
    results.append(test_artifact_generation())

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
