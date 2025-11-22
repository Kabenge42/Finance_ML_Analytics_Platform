"""Portfolio Reporting Wrappers for Section 10 (Enhancement Plan).

This module provides thin, fast wrappers that generate portfolio
reporting artifacts under an output directory. The functions are
designed to be lightweight and deterministic for unit tests; heavy
optimization, plotting, or interactive UIs are intentionally avoided
here and delegated to specialized modules when needed.

Artifacts created adhere to the directory map outlined in the plan:
outputs/portfolio/*.json|.html|.csv

Notes:
- All functions return a manifest dict with a list of files created.
- HTML artifacts are simple placeholders containing minimal content,
  suitable for tests and manual inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


# ------------------------------ helpers ---------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_html(path: Path, title: str, body: str = "") -> None:
    html = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title></head>
    <style>body{{font-family:Arial,Helvetica,sans-serif;margin:16px}}</style>
    <body><h2>{title}</h2><div>{body}</div></body></html>"""
    path.write_text(html, encoding="utf-8")


def _manifest(files: Iterable[Path]) -> Dict[str, List[str]]:
    return {"files": [str(f.name) for f in files]}


# ------------------------------ 10.2 ------------------------------------


def universe_summary(df: pd.DataFrame, out_dir: Path | str) -> Dict[str, List[str]]:
    """Create selection diagnostics and simple explorer placeholder.

    Writes:
    - portfolio_universe_summary.json
    - portfolio_universe_summary.html
    - portfolio_filter_explorer.html (placeholder)
    """

    out = _ensure_dir(Path(out_dir))

    # Group summaries (robust to missing columns)
    by_sector = (
        df.groupby("sector").size().to_dict() if "sector" in df.columns else {}
    )
    by_region = (
        df.groupby("region").size().to_dict() if "region" in df.columns else {}
    )
    # Market cap buckets (if available)
    market_cap_buckets: Dict[str, int] = {}
    if "market_cap" in df.columns:
        try:
            bins = pd.qcut(df["market_cap"], q=3, labels=["small", "mid", "large"])
            market_cap_buckets = bins.value_counts().sort_index().to_dict()  # type: ignore[assignment]
        except Exception:
            # Fallback: simple size thresholds
            s = df["market_cap"].fillna(0)
            market_cap_buckets = {
                "small": int((s < s.median()).sum()),
                "large": int((s >= s.median()).sum()),
            }

    summary = {
        "n": int(len(df)),
        "by_sector": by_sector,
        "by_region": by_region,
        "market_cap_buckets": market_cap_buckets,
    }

    json_path = out / "portfolio_universe_summary.json"
    html_path = out / "portfolio_universe_summary.html"
    filter_html = out / "portfolio_filter_explorer.html"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html(html_path, "Portfolio Universe Summary", "Auto-generated summary table placeholder")
    _write_html(filter_html, "Portfolio Filter Explorer", "Interactive filter placeholder")

    return _manifest([json_path, html_path, filter_html])


# ------------------------------ 10.3 ------------------------------------


def returns_risk_diagnostics(mu: pd.Series, cov: np.ndarray, out_dir: Path | str) -> Dict[str, List[str]]:
    """Create expected-returns distribution and risk inputs QA artifacts.

    Writes:
    - expected_returns_diagnostics.json
    - expected_returns_distribution.html
    - risk_correlation_heatmap.html
    - risk_drift_dashboard.html
    """

    out = _ensure_dir(Path(out_dir))

    mu = mu.dropna()
    diag = {
        "num_assets": int(mu.shape[0]),
        "mean_return": float(mu.mean()) if not mu.empty else 0.0,
        "std_return": float(mu.std()) if mu.shape[0] > 1 else 0.0,
    }
    # Convert covariance to correlation for diagnostics (safe for tests)
    try:
        std = np.sqrt(np.diag(cov))
        corr = cov / (std[:, None] * std[None, :] + 1e-12)
        diag["avg_correlation"] = float(np.nanmean(corr))
    except Exception:
        diag["avg_correlation"] = None

    json_path = out / "expected_returns_diagnostics.json"
    dist_html = out / "expected_returns_distribution.html"
    heatmap_html = out / "risk_correlation_heatmap.html"
    drift_html = out / "risk_drift_dashboard.html"

    json_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    _write_html(dist_html, "Expected Returns Distribution", "Histogram placeholder")
    _write_html(heatmap_html, "Risk Correlation Heatmap", "Heatmap placeholder")
    _write_html(drift_html, "Risk Drift Dashboard", "Volatility/correlation drift placeholder")

    return _manifest([json_path, dist_html, heatmap_html, drift_html])


# ------------------------------ 10.4 ------------------------------------


def frontier_and_constraints(
    mu: pd.Series,
    cov: np.ndarray,
    constraints: Optional[Mapping[str, Iterable[float]]],
    out_dir: Path | str,
) -> Dict[str, List[str]]:
    """Create efficient frontier placeholder and constraint sensitivity artifacts.

    Writes:
    - efficient_frontier.html
    - constraints_sensitivity.html
    - constraints_scenarios.csv
    - transaction_cost_impact.html
    - transaction_cost_summary.json
    """

    out = _ensure_dir(Path(out_dir))

    eff_html = out / "efficient_frontier.html"
    cons_html = out / "constraints_sensitivity.html"
    cons_csv = out / "constraints_scenarios.csv"
    tc_html = out / "transaction_cost_impact.html"
    tc_json = out / "transaction_cost_summary.json"

    # Scenario grid from constraints mapping (single-parameter for tests)
    df_scen = pd.DataFrame()
    if constraints:
        for k, vals in constraints.items():
            df_scen[k] = list(vals)
    else:
        df_scen["max_weight"] = [1.0]
    df_scen.to_csv(cons_csv, index=False)

    _write_html(eff_html, "Efficient Frontier", "Mean-variance frontier placeholder")
    _write_html(cons_html, "Constraints Sensitivity", "Sliders and scenarios placeholder")
    _write_html(tc_html, "Transaction Cost Impact", "TC impact curves placeholder")
    tc_json.write_text(json.dumps({"assumed_tc_bps": 10, "impact": "placeholder"}, indent=2), encoding="utf-8")

    return _manifest([eff_html, cons_html, cons_csv, tc_html, tc_json])


# ------------------------------ 10.5 ------------------------------------


def risk_decomposition_dashboard(
    weights: pd.Series,
    exposures: pd.DataFrame,
    out_dir: Path | str,
) -> Dict[str, List[str]]:
    """Create holdings and risk decomposition placeholders.

    Writes:
    - portfolio_holdings_detailed.csv
    - portfolio_exposures.html
    - risk_decomposition.html
    - stress_tests_dashboard.html
    """

    out = _ensure_dir(Path(out_dir))

    hold_csv = out / "portfolio_holdings_detailed.csv"
    exp_html = out / "portfolio_exposures.html"
    decomp_html = out / "risk_decomposition.html"
    stress_html = out / "stress_tests_dashboard.html"

    weights = weights.fillna(0.0)
    holdings = pd.DataFrame({"ticker": weights.index, "weight": weights.values})
    holdings.to_csv(hold_csv, index=False)

    # Summaries for exposures
    by_sector = (
        exposures.groupby("sector").size().to_dict() if "sector" in exposures.columns else {}
    )
    by_region = (
        exposures.groupby("region").size().to_dict() if "region" in exposures.columns else {}
    )
    _write_html(exp_html, "Portfolio Exposures", f"Sectors: {by_sector} Regions: {by_region}")
    _write_html(decomp_html, "Risk Decomposition", "Contribution-to-risk waterfall placeholder")
    _write_html(stress_html, "Stress Tests", "Scenarios placeholder")

    return _manifest([hold_csv, exp_html, decomp_html, stress_html])


# ------------------------------ 10.6 ------------------------------------


def backtest_and_attribution(
    prices: pd.DataFrame,
    weights: pd.Series,
    out_dir: Path | str,
) -> Dict[str, List[str]]:
    """Create backtest and attribution placeholders.

    Writes:
    - backtest_performance.html
    - performance_attribution.html
    - attribution_breakdown.csv
    """

    out = _ensure_dir(Path(out_dir))

    backtest_html = out / "backtest_performance.html"
    attrib_html = out / "performance_attribution.html"
    attrib_csv = out / "attribution_breakdown.csv"

    # Minimal attribution: proportional to weights (placeholder)
    w = weights.fillna(0.0)
    contrib = (w / (w.sum() if w.sum() else 1.0)).rename("contribution")
    pd.DataFrame({"ticker": contrib.index, "contribution": contrib.values}).to_csv(
        attrib_csv, index=False
    )

    _write_html(backtest_html, "Backtest Performance", "Cumulative returns placeholder")
    _write_html(attrib_html, "Performance Attribution", "Attribution bars placeholder")

    return _manifest([backtest_html, attrib_html, attrib_csv])


# ------------------------------ 10.7 ------------------------------------


def risk_management_dashboard(
    weights: pd.Series,
    cov: np.ndarray,
    out_dir: Path | str,
) -> Dict[str, List[str]]:
    """Create risk management dashboard placeholders.

    Writes:
    - risk_management_dashboard.html
    - portfolio_rebalancing_widget.html
    """

    out = _ensure_dir(Path(out_dir))

    rmd_html = out / "risk_management_dashboard.html"
    widget_html = out / "portfolio_rebalancing_widget.html"

    # Quick metrics for display (not used by tests, but informative)
    try:
        w = weights.values.reshape(-1, 1)
        vol = float(np.sqrt(np.maximum(0.0, (w.T @ cov @ w).item())))
    except Exception:
        vol = None

    _write_html(rmd_html, "Risk Management Dashboard", f"Portfolio volatility: {vol}")
    _write_html(widget_html, "Rebalancing Widget Snapshot", "Static snapshot placeholder")

    return _manifest([rmd_html, widget_html])


# ------------------------------ 10.8 ------------------------------------


def portfolio_summary(kpis: Mapping[str, float], out_dir: Path | str) -> Dict[str, List[str]]:
    """Create portfolio analytics summary and ensure comparison artifact exists.

    Writes:
    - portfolio_summary.json
    - portfolio_multi_period_comparison.html (placeholder)
    """

    out = _ensure_dir(Path(out_dir))
    summary_json = out / "portfolio_summary.json"
    multi_html = out / "portfolio_multi_period_comparison.html"

    payload = {"kpis": dict(kpis), "notes": "Auto-generated summary placeholder"}
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_html(multi_html, "Portfolio Multi-Period Comparison", "Link-out placeholder")

    return _manifest([summary_json, multi_html])


__all__ = [
    "universe_summary",
    "returns_risk_diagnostics",
    "frontier_and_constraints",
    "risk_decomposition_dashboard",
    "backtest_and_attribution",
    "risk_management_dashboard",
    "portfolio_summary",
]
