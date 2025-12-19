from __future__ import annotations
import pandas as pd
import numpy as np
import logging
from dash import Input, Output, State, html, dcc, callback
import dash_bootstrap_components as dbc
from finance_ml.dashboards.components import (
    apply_filters,
    _coerce_list,
    load_alerts_payload,
    create_empty_state_figure,
    create_earnings_events_chart,
)
from finance_ml.dashboards.components.data_utils import (
    DEFAULT_ALERTS_PATH,
    PROJECT_ROOT,
    ARTIFACTS_DIR,
)
from finance_ml.dashboards.components.constants import (
    COLOR_PALETTE,
    FONT_FAMILY,
)
from finance_ml.dashboards.earnings_widgets import (
    create_earnings_surprise_dashboard,
    create_analyst_recommendation_heatmap,
    create_market_movers_dashboard,
    create_price_target_analytics,
    create_earnings_calendar_dashboard,
    create_earnings_metrics_chart,
)
from finance_ml.dashboards.components.temporal_utils import (
    get_reference_date,
    compute_days_to_earnings,
)

logger = logging.getLogger(__name__)


def register_earnings_callbacks(app, initial_df):
    @app.callback(
        Output("earnings-alert-summary", "children"),
        Input("equities-data-store", "data"),
    )
    def _update_alert_summary(data_json):
        """Render compact alert summary panel."""
        payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
        alerts = payload.get("alerts", [])

        if not alerts:
            return html.Div(
                "No alerts available. Click 'Generate Alerts' in the Alerts tab.",
                style={
                    "color": COLOR_PALETTE["neutral"],
                    "padding": "10px",
                    "fontFamily": FONT_FAMILY,
                },
            )

        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for a in alerts:
            sev = a.get("severity", "low").lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        cards = []
        for sev, count in severity_counts.items():
            if count > 0:
                color = {"high": "danger", "medium": "warning", "low": "info"}.get(
                    sev, "secondary"
                )
                cards.append(
                    dbc.Badge(f"{sev.upper()}: {count}", color=color, className="me-2")
                )

        return html.Div(
            [html.Span("Alerts: ", style={"fontWeight": "bold"})] + cards,
            style={
                "padding": "10px",
                "backgroundColor": "#1a1a1a",
                "borderRadius": "5px",
                "marginBottom": "10px",
            },
        )

    @app.callback(
        Output("earnings-events-timeline", "figure"),
        Output("earnings-surprise-fig", "figure"),
        Output("analyst-heatmap-fig", "figure"),
        Output("market-movers-fig", "figure"),
        Output("price-target-analytics-fig", "figure"),
        Input("equities-data-store", "data"),
        Input("earnings-alert-filter-dropdown", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
    )
    def _update_earnings_figs(
        data_json,
        alert_filter,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
    ):
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            empty = create_empty_state_figure("Earnings Analytics", "No data loaded")
            return empty, empty, empty, empty, empty

        df = apply_filters(
            df,
            sectors=_coerce_list(sectors),
            regions=_coerce_list(regions),
            countries=_coerce_list(countries),
            trading_countries=_coerce_list(trading_countries),
            industries=_coerce_list(industries),
            exchanges=_coerce_list(exchanges),
            style_classes=_coerce_list(style_classes),
            size_classes=_coerce_list(size_classes),
        )

        if df.empty:
            empty = create_empty_state_figure(
                "Earnings Analytics", "No data matching filters"
            )
            return empty, empty, empty, empty, empty

        if alert_filter == "alerts_only":
            payload = load_alerts_payload(DEFAULT_ALERTS_PATH)
            alert_tickers = set()
            for a in payload.get("alerts", []):
                alert_tickers.update(a.get("tickers", []))
            if alert_tickers and "ticker" in df.columns:
                df = df[df["ticker"].isin(alert_tickers)]

        if df.empty:
            empty = create_empty_state_figure(
                "Earnings Analytics", "No data matching alerts"
            )
            return empty, empty, empty, empty, empty

        return (
            create_earnings_events_chart(df),
            create_earnings_surprise_dashboard(df),
            create_analyst_recommendation_heatmap(df),
            create_market_movers_dashboard(df),
            create_price_target_analytics(df),
        )

    @app.callback(
        Output("earnings-artifacts-status", "children"),
        Input("generate-earnings-artifacts-btn", "n_clicks"),
        State("equities-data-store", "data"),
        prevent_initial_call=True,
    )
    def _generate_earnings_artifacts(_n, data_json):
        """Generate earnings analytics artifacts (Task 5)."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return "No data available"

        try:
            # Generate the 4 main earnings artifacts
            artifacts_generated = []

            for name, func in [
                (
                    "earnings_surprise_dashboard.html",
                    create_earnings_surprise_dashboard,
                ),
                (
                    "analyst_recommendation_heatmap.html",
                    create_analyst_recommendation_heatmap,
                ),
                ("market_movers_dashboard.html", create_market_movers_dashboard),
                ("price_target_analytics.html", create_price_target_analytics),
            ]:
                output_path = ARTIFACTS_DIR / name
                func(df, output_path=output_path)
                artifacts_generated.append(name)

            return f"✓ Generated {len(artifacts_generated)} artifacts"
        except Exception as e:
            return f"Error: {e}"

    @app.callback(
        Output("earnings-calendar-table", "columns"),
        Output("earnings-calendar-table", "data"),
        Output("earnings-calendar-status", "children"),
        Input("equities-data-store", "data"),
        Input("earnings-calendar-mode", "value"),
        Input("earnings-calendar-days", "value"),
        Input("earnings-calendar-top-n", "value"),
        Input("earnings-calendar-apply-filters", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("trading-country-dropdown", "value"),
        Input("industry-dropdown", "value"),
        Input("exchange-dropdown", "value"),
        Input("style-class-dropdown", "value"),
        Input("size-class-dropdown", "value"),
        prevent_initial_call=False,
    )
    def _update_earnings_calendar(
        data_json,
        mode,
        days_window,
        top_n,
        should_apply_filters,
        sectors,
        regions,
        countries,
        trading_countries,
        industries,
        exchanges,
        style_classes,
        size_classes,
    ):
        """Update the interactive earnings calendar DataTable."""
        try:
            df = pd.read_json(data_json, orient="split") if data_json else initial_df
        except Exception:
            df = initial_df

        if df is None or df.empty:
            return [], [], "No data available"

        # Apply global filters if checkbox is checked
        if should_apply_filters:
            df = apply_filters(
                df,
                sectors=_coerce_list(sectors),
                regions=_coerce_list(regions),
                countries=_coerce_list(countries),
                trading_countries=_coerce_list(trading_countries),
                industries=_coerce_list(industries),
                exchanges=_coerce_list(exchanges),
                style_classes=_coerce_list(style_classes),
                size_classes=_coerce_list(size_classes),
            )

        if df.empty:
            return [], [], "No data after applying filters"

        # These widgets handle their own column selection and data transformation
        # based on the mode provided.
        # This callback delegates to create_earnings_calendar_dashboard from earnings_widgets.py

        # We need a slightly modified version of create_earnings_calendar_dashboard
        # that returns (columns, data, status) instead of a full layout
        # Or we use it as is if it supports this.

        # In equities_dashboard_app.py, this callback implementation is quite long.
        # I should extract the implementation logic as well.
        # For now, I'll just keep the original implementation but in this file.

        # Use consistent reference_date for all temporal calculations
        # Per code_guidelines.md Section 9.3.0 Temporal Calculation Standards
        reference_date = get_reference_date()

        # Compute days_to_earnings using the utility function for consistency
        df["days_to_earnings"] = compute_days_to_earnings(df, reference_date)

        # Filter by days window
        days_window = int(days_window) if days_window else 10
        mask = df["days_to_earnings"].notna() & (
            df["days_to_earnings"].abs() <= days_window
        )
        filtered_df = df[mask].copy()

        if filtered_df.empty:
            return [], [], f"No earnings within ±{days_window} days"

        # Use create_earnings_calendar_dashboard for consistent column selection
        try:
            calendar_df = create_earnings_calendar_dashboard(
                filtered_df,
                reference_date=reference_date,
                top_n=int(top_n) if top_n else 50,
                mode=mode or "all",
            )
        except Exception as e:
            logger.warning(f"Calendar dashboard creation failed: {e}")
            # Fallback to basic columns
            basic_cols = [
                "ticker",
                "sector",
                "region",
                "next_earnings",
                "days_to_earnings",
            ]
            available_cols = [c for c in basic_cols if c in filtered_df.columns]
            if "market_cap" in filtered_df.columns:
                available_cols.append("market_cap")
            calendar_df = filtered_df[available_cols].head(int(top_n) if top_n else 50)

        if calendar_df.empty:
            return [], [], "No data to display"

        # Ensure days_to_earnings is in the output
        if (
            "days_to_earnings" not in calendar_df.columns
            and "next_earnings" in calendar_df.columns
        ):
            calendar_df["days_to_earnings"] = (
                pd.to_datetime(calendar_df["next_earnings"], errors="coerce")
                - reference_date
            ).dt.days

        # Format columns for DataTable (code_guidelines.md Section 17.3)
        columns = []
        for col in calendar_df.columns:
            col_name = col.replace("_", " ").strip().capitalize()
            # Headers: Bold, sentence case (handled via css/DataTable props)
            col_def = {"name": col_name, "id": col, "selectable": True}

            # Apply numeric formatting based on column role/type
            if any(
                x in col
                for x in [
                    "price",
                    "market_cap",
                    "enterprise_value",
                    "ebitda",
                    "ebit",
                    "income",
                    "revenue",
                ]
            ):
                col_def.update({"type": "numeric", "format": {"specifier": "$,.2f"}})
            elif "pct" in col or "margin" in col or "growth" in col or "yield" in col:
                col_def.update({"type": "numeric", "format": {"specifier": ".2%"}})
            elif col in calendar_df.columns and calendar_df[col].dtype in [
                np.float64,
                np.float32,
            ]:
                col_def.update({"type": "numeric", "format": {"specifier": ".2f"}})

            columns.append(col_def)

        # Convert to records, handling dates and rounding
        display_df = calendar_df.copy()
        for col in display_df.columns:
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                display_df[col] = display_df[col].dt.strftime("%Y-%m-%d")
            elif "days_to_earnings" in col:
                # Special handling for days display as seen in images (+0, -1)
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x):+d}" if pd.notnull(x) else ""
                )

        data = display_df.to_dict("records")

        # Status message
        mode_display = (mode or "all").replace("_", " ").title()
        status = f"Showing {len(data)} companies | Mode: {mode_display} | Window: ±{days_window} days"
        if should_apply_filters:
            status += " | Global filters applied"

        return columns, data, status
