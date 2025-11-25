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

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ------------------------------ helpers ---------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_html(path: Path, title: str, body: str = "") -> None:
    html = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title></head>
    <style>body{{font-family:Arial,Helvetica,sans-serif;margin:16px}}</style>
    <body><h2>{title}</h2><div>{body}</div></body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_plotly_html(fig: "go.Figure", path: Path) -> None:
    """Write a Plotly figure to an HTML file."""
    if PLOTLY_AVAILABLE:
        fig.write_html(str(path), include_plotlyjs="cdn")
    else:
        # Fallback to placeholder if Plotly not available
        _write_html(path, "Visualization", "Plotly not available - install with: pip install plotly")


def _manifest(files: Iterable[Path]) -> Dict[str, List[str]]:
    return {"files": [str(f.name) for f in files]}


def _create_portfolio_composition_pie(weights: pd.Series, title: str = "Portfolio Composition") -> "go.Figure":
    """Create adaptive portfolio composition pie chart with dynamic sizing.
    
    Features:
    - Dynamic chart sizing based on asset count (8/15/25/25+ thresholds)
    - Adaptive text display (percent+label, auto, percent only)
    - Intelligent grouping for large portfolios (>15 assets)
    - Long asset name truncation for readability
    - Pull effect for small slices (<5%)
    """
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    # Sort by weight descending
    weights = weights.sort_values(ascending=False)
    n_assets = len(weights)
    
    # Dynamic sizing based on asset count
    if n_assets <= 8:
        width, height = 700, 600
        textposition = "inside"
        textinfo = "percent+label"
        textfont_size = 12
        hole_size = 0.3
    elif n_assets <= 15:
        width, height = 900, 800
        textposition = "auto"
        textinfo = "percent+label"
        textfont_size = 11
        hole_size = 0.35
    elif n_assets <= 25:
        width, height = 1100, 1000
        textposition = "outside"
        textinfo = "percent"
        textfont_size = 10
        hole_size = 0.4
    else:
        width, height = 1300, 1200
        textposition = "outside"
        textinfo = "percent"
        textfont_size = 9
        hole_size = 0.45
    
    # Intelligent grouping for large portfolios
    labels = weights.index.tolist()
    values = weights.values.tolist()
    
    if n_assets > 15:
        # Group small positions (<2% for 16-25, <3% for >25)
        threshold = 0.02 if n_assets <= 25 else 0.03
        min_small_count = 3 if n_assets <= 25 else 5
        
        small_mask = weights < threshold
        if small_mask.sum() > min_small_count:
            large_weights = weights[~small_mask]
            small_sum = weights[small_mask].sum()
            
            labels = large_weights.index.tolist() + ["Others"]
            values = large_weights.values.tolist() + [small_sum]
    
    # Truncate long names for portfolios >15 holdings
    if n_assets > 15:
        labels = [str(lbl)[:40] + "..." if len(str(lbl)) > 40 else str(lbl) for lbl in labels]
    
    # Pull effect for small slices (<5%)
    pull_values = [0.05 if v / sum(values) < 0.05 else 0 for v in values]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        textposition=textposition,
        textinfo=textinfo,
        textfont_size=textfont_size,
        hole=hole_size,
        pull=pull_values,
        marker=dict(line=dict(color='#303030', width=2)),  # Dark border
        hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<br>Value: %{value:.4f}<extra></extra>"
    )])
    
    # Dynamic title with actual holding count
    fig.update_layout(
        title=dict(
            text=f"{title} ({n_assets} Holdings)",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            font=dict(size=textfont_size)
        ),
        margin=dict(l=20, r=200, t=80, b=20),
        template="plotly_dark"
    )
    
    return fig


def _create_returns_distribution_histogram(mu: pd.Series) -> "go.Figure":
    """Create histogram of expected returns distribution."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    fig = go.Figure(data=[go.Histogram(
        x=mu.values,
        nbinsx=30,
        marker=dict(color='#375a7f', line=dict(color='#303030', width=1)),  # Primary
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Expected Returns Distribution",
        xaxis_title="Expected Return",
        yaxis_title="Count",
        width=900,
        height=600,
        showlegend=False,
        template="plotly_dark"
    )
    
    return fig


def _create_correlation_heatmap(cov: np.ndarray, asset_names: List[str]) -> "go.Figure":
    """Create correlation heatmap from covariance matrix."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    # Convert covariance to correlation
    std = np.sqrt(np.diag(cov))
    corr = cov / (std[:, None] * std[None, :] + 1e-12)
    
    # Limit to top 50 assets for readability
    if len(asset_names) > 50:
        top_idx = np.argsort(std)[-50:]
        corr = corr[top_idx][:, top_idx]
        asset_names = [asset_names[i] for i in top_idx]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=asset_names,
        y=asset_names,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-1,
        zmax=1,
        hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.2f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Correlation Heatmap (Top {len(asset_names)} Assets)",
        width=1000,
        height=900,
        xaxis=dict(tickangle=-45),
        template="plotly_dark"
    )
    
    return fig


def _create_efficient_frontier_chart(returns: List[float], risks: List[float], 
                                     max_sharpe_idx: int = -1) -> "go.Figure":
    """Create efficient frontier scatter plot."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    fig = go.Figure()
    
    # Frontier points
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='#375a7f', width=3),  # Primary
        marker=dict(size=6),
        hovertemplate="Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
    ))
    
    # Highlight max Sharpe ratio point
    if max_sharpe_idx >= 0 and max_sharpe_idx < len(returns):
        fig.add_trace(go.Scatter(
            x=[risks[max_sharpe_idx]],
            y=[returns[max_sharpe_idx]],
            mode='markers',
            name='Max Sharpe',
            marker=dict(size=15, color='#e74c3c', symbol='star'),  # Danger/Highlight
            hovertemplate="Max Sharpe<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Mean-Variance Efficient Frontier",
        xaxis_title="Portfolio Risk (Volatility)",
        yaxis_title="Portfolio Return",
        width=900,
        height=600,
        template="plotly_dark"
    )
    
    return fig


def _create_backtest_performance_chart(dates: pd.DatetimeIndex, cumulative_returns: pd.Series) -> "go.Figure":
    """Create cumulative returns line chart for backtesting."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_returns.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#375a7f', width=2),  # Primary
        fill='tozeroy',
        fillcolor='rgba(55, 90, 127, 0.2)',  # Primary with opacity
        hovertemplate="Date: %{x}<br>Cumulative Return: %{y:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Backtest Cumulative Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        width=1000,
        height=600,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def _create_attribution_bar_chart(tickers: List[str], contributions: List[float]) -> "go.Figure":
    """Create horizontal bar chart for performance attribution."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    # Sort by contribution
    df = pd.DataFrame({'ticker': tickers, 'contribution': contributions})
    df = df.sort_values('contribution', ascending=True)
    
    # Color coding: green for positive, red for negative
    colors = ['#00bc8c' if c >= 0 else '#e74c3c' for c in df['contribution']]  # Success/Danger
    
    fig = go.Figure(data=[go.Bar(
        y=df['ticker'],
        x=df['contribution'],
        orientation='h',
        marker=dict(color=colors),
        hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2%}<extra></extra>"
    )])
    
    fig.update_layout(
        title="Performance Attribution by Asset",
        xaxis_title="Contribution to Return",
        yaxis_title="Asset",
        width=900,
        height=max(500, len(tickers) * 20),  # Dynamic height
        template="plotly_dark",
        showlegend=False
    )
    
    return fig


def _create_risk_decomposition_waterfall(components: Dict[str, float]) -> "go.Figure":
    """Create waterfall chart for risk decomposition."""
    if not PLOTLY_AVAILABLE:
        return go.Figure()
    
    labels = list(components.keys())
    values = list(components.values())
    
    fig = go.Figure(go.Waterfall(
        name="Risk",
        orientation="v",
        measure=["relative"] * (len(labels) - 1) + ["total"],
        x=labels,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#e74c3c"}},  # Danger
        increasing={"marker": {"color": "#00bc8c"}},  # Success
        totals={"marker": {"color": "#375a7f"}},  # Primary
        hovertemplate="%{x}<br>Contribution: %{y:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Risk Decomposition Waterfall",
        xaxis_title="Component",
        yaxis_title="Risk Contribution",
        width=1000,
        height=600,
        template="plotly_dark"
    )
    
    return fig


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
    
    # Generate interactive pie charts for sector and region distribution
    if PLOTLY_AVAILABLE and "sector" in df.columns and len(by_sector) > 0:
        sector_series = pd.Series(by_sector)
        fig = _create_portfolio_composition_pie(sector_series, "Universe by Sector")
        _write_plotly_html(fig, html_path)
    else:
        _write_html(html_path, "Portfolio Universe Summary", "Auto-generated summary table placeholder")
    
    # Filter explorer remains placeholder for now
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
    
    # Generate interactive histogram for returns distribution
    if PLOTLY_AVAILABLE and not mu.empty:
        fig_dist = _create_returns_distribution_histogram(mu)
        _write_plotly_html(fig_dist, dist_html)
    else:
        _write_html(dist_html, "Expected Returns Distribution", "Histogram placeholder")
    
    # Generate interactive correlation heatmap
    if PLOTLY_AVAILABLE and cov.shape[0] > 0:
        asset_names = mu.index.tolist()
        fig_heatmap = _create_correlation_heatmap(cov, asset_names)
        _write_plotly_html(fig_heatmap, heatmap_html)
    else:
        _write_html(heatmap_html, "Risk Correlation Heatmap", "Heatmap placeholder")
    
    # Drift dashboard remains placeholder for now
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
        df_scen["max_weight"] = [0.1, 0.15, 0.2, 0.25, 0.3]
    df_scen.to_csv(cons_csv, index=False)
    
    # Generate simple efficient frontier (equal-weighted portfolios with varying number of assets)
    if PLOTLY_AVAILABLE and not mu.empty and cov.shape[0] > 1:
        n_points = 15
        returns_list = []
        risks_list = []
        
        # Simple approach: vary concentration from equal-weight to concentrated
        for i in range(n_points):
            # Create weights that concentrate gradually
            alpha = i / (n_points - 1)  # 0 to 1
            n_assets = max(2, int(len(mu) * (1 - alpha * 0.8)))  # Use 100% to 20% of assets
            
            # Select top assets by return
            top_assets = mu.nlargest(n_assets).index
            w = np.zeros(len(mu))
            indices = [mu.index.get_loc(asset) for asset in top_assets]
            w[indices] = 1.0 / n_assets
            
            # Calculate portfolio return and risk
            port_return = float(np.dot(w, mu.values))
            port_risk = float(np.sqrt(np.maximum(0.0, w.T @ cov @ w)))
            
            returns_list.append(port_return)
            risks_list.append(port_risk)
        
        # Find max Sharpe (simple: max return/risk)
        sharpe_ratios = [r / (v + 1e-9) for r, v in zip(returns_list, risks_list)]
        max_sharpe_idx = int(np.argmax(sharpe_ratios))
        
        fig_frontier = _create_efficient_frontier_chart(returns_list, risks_list, max_sharpe_idx)
        _write_plotly_html(fig_frontier, eff_html)
    else:
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
    
    # Generate portfolio composition pie chart with actual holdings
    if PLOTLY_AVAILABLE and len(weights) > 0:
        fig_composition = _create_portfolio_composition_pie(weights, "Optimized Portfolio Composition")
        _write_plotly_html(fig_composition, exp_html)
    else:
        _write_html(exp_html, "Portfolio Exposures", f"Sectors: {by_sector} Regions: {by_region}")
    
    # Generate risk decomposition waterfall if we have sector data
    if PLOTLY_AVAILABLE and "sector" in exposures.columns and len(by_sector) > 0:
        # Simple risk decomposition: weight contribution by sector
        sector_weights = {}
        for ticker in weights.index:
            if ticker in exposures.index:
                sector = exposures.loc[ticker, "sector"]
                sector_weights[sector] = sector_weights.get(sector, 0.0) + weights[ticker]
        
        sector_weights["Total"] = sum(sector_weights.values())
        fig_decomp = _create_risk_decomposition_waterfall(sector_weights)
        _write_plotly_html(fig_decomp, decomp_html)
    else:
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
    
    # Generate interactive attribution bar chart
    if PLOTLY_AVAILABLE and len(contrib) > 0:
        tickers = contrib.index.tolist()
        contributions = contrib.values.tolist()
        fig_attrib = _create_attribution_bar_chart(tickers, contributions)
        _write_plotly_html(fig_attrib, attrib_html)
    else:
        _write_html(attrib_html, "Performance Attribution", "Attribution bars placeholder")
    
    # Generate backtest performance chart if prices has time-series data
    if PLOTLY_AVAILABLE and isinstance(prices.index, pd.DatetimeIndex) and len(prices) > 1:
        # Compute simple portfolio returns (weighted average)
        aligned_weights = w.reindex(prices.columns, fill_value=0.0)
        returns = prices.pct_change().fillna(0.0)
        portfolio_returns = (returns * aligned_weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        fig_backtest = _create_backtest_performance_chart(prices.index, cumulative_returns)
        _write_plotly_html(fig_backtest, backtest_html)
    else:
        _write_html(backtest_html, "Backtest Performance", "Cumulative returns placeholder")

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
