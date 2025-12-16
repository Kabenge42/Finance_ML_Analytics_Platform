"""Scheduled earnings monitoring runner.

Generates earnings analytics dashboards + rule-based alerts without opening notebooks.

Intended usage (Windows Task Scheduler friendly):
  python tools\\run_earnings_monitor.py --data-source csv --out-dir outputs\\eda\\earnings_analytics
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from finance_ml.dashboards.earnings_widgets import (
    EarningsAlertConfig,
    create_analyst_recommendation_heatmap,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
    generate_earnings_quality_alerts,
)
from finance_ml.ml_workflow.preprocessing.etl import etl_with_features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate earnings monitoring artifacts (dashboards + alerts)."
    )

    parser.add_argument(
        "--data-source",
        choices=["auto", "csv", "db"],
        default="auto",
        help="Data source (auto=DB if DB_URL/--db-url provided, else CSV).",
    )
    parser.add_argument(
        "--data-dir", default="data", help="CSV directory (when using --data-source csv)."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DB_URL"),
        help="SQLAlchemy DB URL (when using --data-source db).",
    )
    parser.add_argument(
        "--feature-preset",
        default="comprehensive",
        help="ETL feature preset: basic|momentum|quality|standard|comprehensive.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path("outputs") / "eda" / "earnings_analytics"),
        help="Output directory for HTML/JSON artifacts.",
    )

    # Dashboard params
    parser.add_argument("--surprise-top-n", type=int, default=100)
    parser.add_argument("--heatmap-top-n-sectors", type=int, default=12)
    parser.add_argument("--movers-lookback-days", type=int, default=7)
    parser.add_argument("--movers-top-n", type=int, default=20)
    parser.add_argument("--price-target-top-n-sectors", type=int, default=12)

    # Alert thresholds
    parser.add_argument("--eps-miss-threshold-pct", type=float, default=20.0)
    parser.add_argument("--analyst-downgrade-threshold-pct", type=float, default=5.0)
    parser.add_argument("--analyst-downgrade-min-periods", type=int, default=2)
    parser.add_argument("--target-spread-threshold-pct", type=float, default=30.0)
    parser.add_argument("--pre-earnings-window-days", type=int, default=7)
    parser.add_argument("--pre-earnings-vol-quantile", type=float, default=0.75)
    parser.add_argument("--max-tickers-per-alert", type=int, default=10)

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.data_source == "auto":
        source = "db" if args.db_url else "csv"
    else:
        source = args.data_source

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_date = pd.Timestamp.now()

    # Load enriched dataset
    df = etl_with_features(
        source=source,
        data_dir=Path(args.data_dir),
        db_url=args.db_url,
        feature_preset=args.feature_preset,
        return_metrics=False,
    )

    # Dashboards
    create_earnings_surprise_dashboard(
        df,
        reference_date=reference_date,
        top_n=args.surprise_top_n,
        output_path=out_dir / "earnings_surprise_dashboard.html",
    )
    create_analyst_recommendation_heatmap(
        df,
        top_n_sectors=args.heatmap_top_n_sectors,
        output_path=out_dir / "analyst_recommendation_heatmap.html",
    )
    create_market_movers_dashboard(
        df,
        reference_date=reference_date,
        lookback_days=args.movers_lookback_days,
        top_n=args.movers_top_n,
        output_path=out_dir / "market_movers_dashboard.html",
    )
    create_price_target_analytics(
        df,
        top_n_sectors=args.price_target_top_n_sectors,
        output_path=out_dir / "price_target_analytics.html",
    )

    # Alerts
    alert_config = EarningsAlertConfig(
        eps_surprise_miss_threshold_pct=args.eps_miss_threshold_pct,
        analyst_downgrade_threshold_pct=args.analyst_downgrade_threshold_pct,
        analyst_downgrade_min_periods=args.analyst_downgrade_min_periods,
        target_spread_threshold_pct=args.target_spread_threshold_pct,
        pre_earnings_window_days=args.pre_earnings_window_days,
        pre_earnings_volatility_quantile=args.pre_earnings_vol_quantile,
        max_tickers_per_alert=args.max_tickers_per_alert,
    )
    generate_earnings_quality_alerts(
        df,
        config=alert_config,
        reference_date=reference_date,
        output_path=out_dir / "earnings_quality_alerts.json",
    )

    print("Earnings monitoring artifacts generated:")
    print(f"  Output dir: {out_dir}")
    print("  - earnings_surprise_dashboard.html")
    print("  - analyst_recommendation_heatmap.html")
    print("  - market_movers_dashboard.html")
    print("  - price_target_analytics.html")
    print("  - earnings_quality_alerts.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
