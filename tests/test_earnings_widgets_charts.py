import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from finance_ml.dashboards.widgets import (
    create_analyst_recommendation_heatmap,
    create_category_comparison_chart,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_market_movers_dashboard,
    create_price_target_analytics,
)


def _df_for_dashboards(n_per_sector: int = 6) -> pd.DataFrame:
    now = pd.Timestamp.now().normalize()
    rows = []
    for sector in ["Tech", "Banks"]:
        for i in range(n_per_sector):
            last_price = 50.0 + i * 10.0 + (0.5 if sector == "Tech" else 1.0)
            rows.append(
                {
                    "ticker": f"{sector[:2].upper()}{i:02d}",
                    "name": f"{sector} {i}",
                    "sector": sector,
                    "region": "US" if sector == "Tech" else "EU",
                    "market_cap": 1_000 + i * 100 + (500 if sector == "Tech" else 0),
                    "next_earnings": now + timedelta(days=(i % 5) - 2),
                    # Surprise dashboards
                    "total_revenues_ltm": 100.0 + i * 5.0,
                    "revenues_est_avg_ntm": 95.0 + i * 4.5,
                    "ebitda_ltm": 30.0 + i * 2.0,
                    "ebitda_est_avg_fy1e": 28.0 + i * 1.8,
                    "ebit_ltm": 20.0 + i * 1.5,
                    "ebit_est_med_ntm": 19.0 + i * 1.4,
                    "net_income_is_ltm": 10.0 + i * 1.0,
                    "net_income_adj_1fy": 9.5 + i * 0.9,
                    "eps_adj_ltm": 5.0 + i * 0.2,
                    "eps_norm_est_avg_ntm": 4.8 + i * 0.18,
                    # Analyst ratings
                    "num_strong_buys_ratings": 10 - i,
                    "num_buys_ratings": 12 - i,
                    "num_hold_ratings": 5 + (i % 3),
                    "num_sell_ratings": i % 2,
                    "num_strong_sell_ratings": 0,
                    # Market movers
                    "last_price": last_price,
                    "price_momentum_1m": 0.10 - i * 0.02,
                    "volatility_1m": 0.15 + i * 0.01,
                    "rel_volume": 0.8 + i * 0.1,
                    "one_day_pct": i * 0.5,
                    "price_chg_pct_1m": i * 2.0,
                    "price_chg_pct_3m": i * 5.0,
                    # Price target analytics
                    "price_target": last_price * (1.05 + (i % 3) * 0.05),
                    "price_target_high": last_price * 1.25,
                    "price_target_low": last_price * 0.85,
                }
            )
    return pd.DataFrame(rows)


class TestEarningsWidgetsCharts(unittest.TestCase):
    def test_full_charts_write_html_and_have_traces(self):
        df = _df_for_dashboards(n_per_sector=6)
        now = pd.Timestamp.now().normalize()

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)

            fig_metrics = create_earnings_metrics_chart(
                df,
                metric_category="profitability",
                reference_date=now,
                top_n=10,
                output_path=out_dir / "metrics.html",
            )
            self.assertTrue((out_dir / "metrics.html").exists())
            self.assertGreater(len(fig_metrics.data), 0)

            fig_compare = create_category_comparison_chart(
                df,
                categories=["profitability", "growth"],
                reference_date=now,
                top_n=10,
                output_path=out_dir / "compare.html",
            )
            self.assertTrue((out_dir / "compare.html").exists())
            self.assertGreater(len(fig_compare.data), 0)

            fig_surprise = create_earnings_surprise_dashboard(
                df,
                reference_date=now,
                top_n=20,
                output_path=out_dir / "surprise.html",
            )
            self.assertTrue((out_dir / "surprise.html").exists())
            self.assertGreater(len(fig_surprise.data), 0)

            fig_heatmap = create_analyst_recommendation_heatmap(
                df,
                top_n_sectors=5,
                output_path=out_dir / "heatmap.html",
            )
            self.assertTrue((out_dir / "heatmap.html").exists())
            self.assertGreater(len(fig_heatmap.data), 0)

            fig_movers = create_market_movers_dashboard(
                df,
                reference_date=now,
                lookback_days=7,
                top_n=10,
                output_path=out_dir / "movers.html",
            )
            self.assertTrue((out_dir / "movers.html").exists())
            self.assertGreater(len(fig_movers.data), 0)

            fig_targets = create_price_target_analytics(
                df,
                top_n_sectors=2,
                output_path=out_dir / "targets.html",
            )
            self.assertTrue((out_dir / "targets.html").exists())
            self.assertGreater(len(fig_targets.data), 0)


if __name__ == "__main__":
    unittest.main()
