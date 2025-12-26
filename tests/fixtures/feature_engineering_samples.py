"""
Test fixtures: Sample DataFrames for feature engineering tests (Phase 9.3 - Week 1 Infra)

Provides compact, deterministic samples with edge cases (NaN, inf, zeros, negatives, outliers)
covering common columns used by finance_ml.features.advanced functions.

Usage:
    from tests.fixtures.feature_engineering_samples import (
        make_minimal_sample,
        make_edge_case_sample,
        make_large_sample,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_minimal_sample() -> pd.DataFrame:
    """A tiny sample with key columns present and clean values."""
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Tech", "Energy", "Financials"],
            "last_price": [100.0, 50.0, 25.0],
            "eps": [5.0, 2.5, 1.25],
            "book_value_per_share": [20.0, 10.0, 5.0],
            "revenue": [1000.0, 500.0, 250.0],
            "shares_outstanding": [100.0, 100.0, 100.0],
            "enterprise_value": [1000.0, 500.0, 250.0],
            "ebitda": [100.0, 50.0, 25.0],
            "earnings_growth_pct": [10.0, 10.0, 10.0],
            "dividend_per_share": [2.0, 1.0, 0.5],
            "net_income": [100.0, 50.0, 25.0],
            "total_equity": [1000.0, 500.0, 250.0],
            "total_assets": [2000.0, 1000.0, 500.0],
            "gross_profit": [400.0, 200.0, 100.0],
            "operating_income": [200.0, 100.0, 50.0],
            "total_debt": [500.0, 250.0, 125.0],
            "net_debt": [300.0, 150.0, 75.0],
            "ebit": [120.0, 60.0, 30.0],
            "interest_expense": [12.0, 6.0, 3.0],
        }
    )


def make_edge_case_sample() -> pd.DataFrame:
    """Include NaN, inf, zeros, negatives, and outliers to probe robustness."""
    df = pd.DataFrame(
        {
            "ticker": ["X", "Y", "Z", "W"],
            "sector": ["Tech", "Tech", "Energy", "Energy"],
            "last_price": [np.nan, np.inf, 0.0, -10.0],
            "eps": [0.0, -5.0, 1.0, np.nan],
            "book_value_per_share": [0.0, 20.0, np.nan, 5.0],
            "revenue": [0.0, 1e9, 500.0, np.nan],
            "shares_outstanding": [0.0, 1e6, 100.0, 0.0],
            "enterprise_value": [np.nan, 1e12, 250.0, -100.0],
            "ebitda": [0.0, 1e3, np.nan, -50.0],
            "earnings_growth_pct": [0.0, -10.0, 5.0, np.nan],
            "dividend_per_share": [0.0, 10.0, np.nan, -1.0],
            "net_income": [np.nan, -1e6, 25.0, 0.0],
            "total_equity": [0.0, 5e6, 250.0, np.nan],
            "total_assets": [0.0, 1e7, np.nan, 500.0],
            "gross_profit": [0.0, 2e6, 100.0, np.nan],
            "operating_income": [np.nan, 1e6, 50.0, 0.0],
            "total_debt": [0.0, 3e6, 125.0, 0.0],
            "net_debt": [0.0, 2e6, 75.0, np.nan],
            "ebit": [0.0, 8e5, 30.0, -10.0],
            "interest_expense": [0.0, 8e4, 3.0, 0.0],
        }
    )
    return df


def make_large_sample(n_rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a larger sample (>=1000 rows) for validation/perf tests.

    Values are constructed to avoid divide-by-zero catastrophes but still include
    controlled zeros and NaNs at a small rate.
    """
    rng = np.random.default_rng(seed)
    sectors = np.array(["Tech", "Energy", "Financials", "Healthcare", "Industrials"])  # 5 sectors
    df = pd.DataFrame(
        {
            "ticker": [f"T{i:05d}" for i in range(n_rows)],
            "sector": rng.choice(sectors, size=n_rows),
            "last_price": rng.lognormal(mean=4.5, sigma=0.4, size=n_rows),
            "eps": rng.normal(loc=5.0, scale=2.0, size=n_rows),
            "book_value_per_share": rng.lognormal(mean=3.0, sigma=0.5, size=n_rows),
            "revenue": rng.lognormal(mean=10.0, sigma=0.6, size=n_rows),
            "shares_outstanding": rng.integers(50e6, 500e6, size=n_rows),
            "enterprise_value": rng.lognormal(mean=12.0, sigma=0.7, size=n_rows),
            "ebitda": rng.lognormal(mean=9.0, sigma=0.7, size=n_rows),
            "earnings_growth_pct": rng.normal(loc=10.0, scale=5.0, size=n_rows),
            "dividend_per_share": np.maximum(0.0, rng.normal(loc=2.0, scale=0.5, size=n_rows)),
            "net_income": rng.normal(loc=1e9, scale=3e8, size=n_rows),
            "total_equity": rng.lognormal(mean=12.0, sigma=0.5, size=n_rows),
            "total_assets": rng.lognormal(mean=12.5, sigma=0.5, size=n_rows),
            "gross_profit": rng.lognormal(mean=11.0, sigma=0.6, size=n_rows),
            "operating_income": rng.lognormal(mean=10.5, sigma=0.6, size=n_rows),
            "total_debt": rng.lognormal(mean=12.0, sigma=0.6, size=n_rows),
            "net_debt": rng.lognormal(mean=11.5, sigma=0.6, size=n_rows),
            "ebit": rng.lognormal(mean=10.0, sigma=0.6, size=n_rows),
            "interest_expense": rng.lognormal(mean=8.0, sigma=0.6, size=n_rows),
        }
    )

    # Inject controlled NaNs and zeros
    mask = rng.random(n_rows) < 0.02  # 2%
    df.loc[mask, "eps"] = 0.0
    df.loc[rng.random(n_rows) < 0.01, "earnings_growth_pct"] = 0.0
    df.loc[rng.random(n_rows) < 0.01, "ebitda"] = 0.0
    df.loc[rng.random(n_rows) < 0.01, "total_equity"] = 0.0

    # Sprinkle NaNs
    for col in ("book_value_per_share", "revenue", "dividend_per_share"):
        df.loc[rng.random(n_rows) < 0.01, col] = np.nan

    return df
