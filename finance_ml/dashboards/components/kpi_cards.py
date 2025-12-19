from __future__ import annotations

from typing import Any, List

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html

from .constants import FONT_FAMILY, FONT_SIZES
from .data_utils import DEFAULT_ALERTS_PATH, load_alerts_payload


def _monitoring_kpi_cards(df: pd.DataFrame) -> List[Any]:
    """Generate monitoring KPI cards.

    Styling aligned with code_guidelines.md Section 17.4.
    """
    cards = []

    def card(title: str, value: str, color: str = "primary") -> dbc.Card:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        title,
                        className="kpi-title",
                        style={
                            "fontSize": f"{FONT_SIZES['caption']}px",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                    html.Div(
                        value,
                        className="kpi-value",
                        style={
                            "fontSize": f"{FONT_SIZES['h3']}px",
                            "fontWeight": "bold",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                ]
            ),
            color=color,
            inverse=True,
            className="kpi-card",
            style={"minWidth": "150px"},
        )

    # 1. % Positive Revenue Growth
    if "total_revenues_cagr_5y_fy" in df.columns:
        growth = pd.to_numeric(df["total_revenues_cagr_5y_fy"], errors="coerce")
        pct_positive = (growth > 0).sum() / len(growth) * 100 if len(growth) > 0 else 0
        cards.append(
            card(
                "% Positive Rev Growth",
                f"{pct_positive:.1f}%",
                "success" if pct_positive > 50 else "warning",
            )
        )

    # 2. Median Net Margin
    if "net_income_margin_pct_ltm" in df.columns:
        margin = pd.to_numeric(df["net_income_margin_pct_ltm"], errors="coerce")
        median_margin = margin.median() if margin.notna().any() else 0
        cards.append(card("Median Net Margin", f"{median_margin:.1f}%", "info"))

    # 3. % Flagged by Alerts
    payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
    alert_tickers = set()
    for a in payload.get("alerts", []):
        alert_tickers.update(a.get("tickers", []))
    if "ticker" in df.columns and len(df) > 0:
        pct_flagged = len(alert_tickers & set(df["ticker"])) / len(df) * 100
        cards.append(
            card(
                "% With Alerts",
                f"{pct_flagged:.1f}%",
                "danger" if pct_flagged > 20 else "secondary",
            )
        )

    # 4. Median EPS Revision (if available)
    rev_cols = [c for c in df.columns if "eps_est_avg_rev_pct" in c.lower()]
    if rev_cols:
        rev = pd.to_numeric(df[rev_cols[0]], errors="coerce")
        median_rev = rev.median() if rev.notna().any() else 0
        cards.append(
            card(
                "Median EPS Revision",
                f"{median_rev:+.1f}%",
                "success" if median_rev > 0 else "danger",
            )
        )

    return cards


def _kpi_cards(df: pd.DataFrame) -> List[Any]:
    """Generate overview KPI cards.

    Styling aligned with code_guidelines.md Section 17.4.
    """

    def _num(series: pd.Series) -> float:
        return float(pd.to_numeric(series, errors="coerce").dropna().mean())

    total = int(len(df))
    tickers = int(df["ticker"].nunique()) if "ticker" in df.columns else 0
    mean_upside = None
    if "price_target" in df.columns and "last_price" in df.columns:
        pt = pd.to_numeric(df["price_target"], errors="coerce")
        lp = pd.to_numeric(df["last_price"], errors="coerce")
        valid = pt.notna() & lp.notna() & (lp > 0)
        if valid.any():
            mean_upside = float((((pt[valid] - lp[valid]) / lp[valid]) * 100).mean())

    market_cap_mean = _num(df["market_cap"]) if "market_cap" in df.columns else None

    def card(title: str, value: str) -> dbc.Card:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.Div(
                        title,
                        className="kpi-title",
                        style={
                            "fontSize": f"{FONT_SIZES['caption']}px",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                    html.Div(
                        value,
                        className="kpi-value",
                        style={
                            "fontSize": f"{FONT_SIZES['h3']}px",
                            "fontWeight": "bold",
                            "fontFamily": FONT_FAMILY,
                        },
                    ),
                ]
            ),
            className="kpi-card",
        )

    cards = [
        card("Rows", f"{total:,}"),
        card("Tickers", f"{tickers:,}"),
    ]
    if mean_upside is not None:
        cards.append(card("Mean Upside", f"{mean_upside:,.1f}%"))
    if market_cap_mean is not None and market_cap_mean == market_cap_mean:
        cards.append(card("Mean Market Cap", f"${market_cap_mean:,.0f}"))
    return cards
