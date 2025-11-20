"""Portfolio dashboard widgets and visualisation helpers (Phase 6).

This module implements lightweight, test-friendly helpers for the
interactive portfolio dashboards described in
docs/improvement_plan/portfolio_optimization_enhancement_plan.md:

* :class:`PortfolioRebalanceWidget` – compute rebalance trades from
  current holdings to target weights.
* :func:`create_multi_period_comparison` – Plotly figure comparing
  portfolio vs. benchmark performance across multiple periods.
* :func:`create_factor_exposure_dashboard` – radar / spider chart of
  portfolio factor exposures.

The implementations are intentionally minimal and rely only on
``pandas``, ``numpy`` and ``plotly`` to keep tests fast and
dependencies manageable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:  # Plotly is an optional dependency but required for dashboard tests
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - handled defensively for environments without plotly
    go = None  # type: ignore


@dataclass
class PortfolioRebalanceWidget:
    """Simple helper to compute rebalance trades.

    Parameters
    ----------
    current_holdings:
        DataFrame with at least ``ticker``, ``shares`` and ``price``
        columns representing the current portfolio.
    target_weights:
        Series whose index are tickers and values are desired
        portfolio weights that should sum to 1 (within tolerance).
    cash_buffer:
        Optional fraction of portfolio value to keep in cash. For the
        current tests this is unused and defaults to 0.0.
    """

    current_holdings: pd.DataFrame
    target_weights: pd.Series
    cash_buffer: float = 0.0

    def get_rebalance_trades(self) -> pd.DataFrame:
        """Return a DataFrame of trades required to reach target weights.

        The returned frame contains the following columns:

        - ``ticker`` – asset identifier
        - ``action`` – ``"BUY"`` or ``"SELL"``
        - ``shares`` – signed share quantity (positive for buys)
        - ``estimated_cost`` – trade notional in the same units as price
        """

        holdings = self.current_holdings.set_index("ticker").copy()
        if "shares" not in holdings.columns or "price" not in holdings.columns:
            raise KeyError("current_holdings must contain 'shares' and 'price' columns")

        # Ensure target weights are aligned to holdings index; missing
        # entries default to zero weight.
        target_w = self.target_weights.reindex(holdings.index).fillna(0.0).astype(float)

        # Normalise target weights to sum to 1 after optional cash buffer.
        total_w = float(target_w.sum())
        if total_w <= 0:
            raise ValueError("target_weights must sum to a positive value")

        effective_total = 1.0 - float(self.cash_buffer)
        target_w = target_w / total_w * effective_total

        prices = holdings["price"].astype(float)
        current_shares = holdings["shares"].astype(float)
        current_values = current_shares * prices
        portfolio_value = float(current_values.sum())

        # Desired values and shares under target weights
        target_values = target_w * portfolio_value
        desired_shares = target_values / prices

        share_diff = desired_shares - current_shares

        trades = pd.DataFrame(
            {
                "ticker": holdings.index,
                "shares": share_diff,
            }
        )
        trades["action"] = np.where(trades["shares"] >= 0, "BUY", "SELL")
        trades["estimated_cost"] = trades["shares"] * prices.values

        # Filter out zero-share trades to keep output concise
        trades = trades[trades["shares"].abs() > 0].reset_index(drop=True)
        return trades


def _compute_period_return(returns: pd.Series, window: int | None) -> float:
    """Helper to convert daily returns to a cumulative return over a window.

    If ``window`` is None, the full series is used (ITD). The function
    assumes ``returns`` are simple returns (not log returns).
    """

    if window is None or window >= len(returns):
        window_slice = returns
    else:
        window_slice = returns.iloc[-window:]

    cumulative = float((1.0 + window_slice).prod() - 1.0)
    return cumulative


_PERIOD_WINDOWS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "YTD": None,  # treated same as ITD for synthetic tests
    "ITD": None,
}


def create_multi_period_comparison(
    portfolio_returns: pd.Series,
    periods: Sequence[str],
    benchmark_returns: pd.Series | None = None,
):
    """Create a multi-period performance comparison figure.

    The figure shows portfolio (and optional benchmark) cumulative
    returns over a list of period labels such as ``["1M", "3M", "1Y"]``.
    """

    if go is None:  # pragma: no cover - guard for missing plotly
        raise RuntimeError("plotly is required for create_multi_period_comparison")

    x_labels: list[str] = []
    port_vals: list[float] = []
    bench_vals: list[float] = []

    for label in periods:
        window = _PERIOD_WINDOWS.get(label, None)
        x_labels.append(label)
        port_vals.append(_compute_period_return(portfolio_returns, window))
        if benchmark_returns is not None:
            bench_vals.append(_compute_period_return(benchmark_returns, window))

    fig = go.Figure()
    fig.add_bar(name="Portfolio", x=x_labels, y=port_vals)

    if benchmark_returns is not None:
        fig.add_bar(name="Benchmark", x=x_labels, y=bench_vals)

    fig.update_layout(
        barmode="group",
        title="Multi-Period Performance Comparison",
        xaxis_title="Period",
        yaxis_title="Return",
        legend_title="Series",
    )
    return fig


def create_factor_exposure_dashboard(
    portfolio_weights: pd.Series,
    factor_loadings: pd.DataFrame,
    factors: Sequence[str],
):
    """Create a radar / spider chart of portfolio factor exposures.

    Factor exposure for a given factor is computed as the weighted sum
    of asset-level loadings using ``portfolio_weights``.
    """

    if go is None:  # pragma: no cover - guard for missing plotly
        raise RuntimeError("plotly is required for create_factor_exposure_dashboard")

    # Align weights to factor loading index
    w = portfolio_weights.reindex(factor_loadings.index).fillna(0.0).astype(float)
    norm = float(w.sum())
    if norm <= 0:
        raise ValueError("portfolio_weights must sum to a positive value")
    w = w / norm

    exposures: list[float] = []
    labels: list[str] = []
    for factor in factors:
        if factor not in factor_loadings.columns:
            continue
        labels.append(factor)
        col = factor_loadings[factor].astype(float)
        exposures.append(float(np.dot(w.values, col.values)))

    # Close the radar chart loop by repeating first element
    if labels:
        labels.append(labels[0])
        exposures.append(exposures[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=exposures,
            theta=labels,
            fill="toself",
            name="Factor Exposure",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        title="Factor Exposure Dashboard",
    )
    return fig
