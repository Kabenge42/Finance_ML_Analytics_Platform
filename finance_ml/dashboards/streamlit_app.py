"""
Interactive Streamlit Dashboard for Finance ML Analytics
Run: streamlit run finance_ml/dashboards/streamlit_app.py
"""

from pathlib import Path
import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from finance_ml.dashboards.artifact_registry import ARTIFACTS
except ImportError:
    # Fallback for when running as standalone
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from finance_ml.dashboards.artifact_registry import ARTIFACTS

try:
    from finance_ml.ml_workflow.analytics.eval import (
        prepare_plotly_dashboard_data,
        calculate_mispricing_score,
        rank_stocks_by_sector,
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
    )
except ImportError:
    # Fallback for when running as standalone
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from finance_ml.ml_workflow.analytics.eval import (
        prepare_plotly_dashboard_data,
        calculate_mispricing_score,
        rank_stocks_by_sector,
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
    )


def get_file_age(filepath: Path) -> str:
    """Get formatted age of a file."""
    if filepath.exists():
        mtime = os.path.getmtime(filepath)
        age = datetime.now() - datetime.fromtimestamp(mtime)
        if age.days == 0:
            return f"Updated today ({age.seconds//3600}h ago)"
        return f"Updated {age.days} days ago"
    return "Not generated"


def render_streamlit_artifact(category, key, height=600):
    """Render HTML artifact in Streamlit."""
    if category in ARTIFACTS and key in ARTIFACTS[category]:
        item = ARTIFACTS[category][key]
        st.subheader(item["title"])

        # Resolve path
        path = Path(__file__).parent.parent.parent / "outputs" / category / item["file"]
        if path.exists():
            age = get_file_age(path)
            st.caption(f"🕒 {age}")
            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=height, scrolling=True)
        else:
            st.info(f"⚠️ Artifact not available. Run Section {item['section']}.")
    else:
        st.error(f"Configuration missing for {category}/{key}")


st.set_page_config(page_title="Finance ML Analytics", layout="wide", page_icon="📊")

# Sidebar filters
st.sidebar.title("🔍 Filters")

# Quick Access
st.sidebar.markdown("### 🔗 Quick Access")
if (Path(__file__).parent.parent.parent / "outputs/governance/model_card_v9_10.md").exists():
    st.sidebar.markdown("- 📄 [Model Card](?tab=governance)")
st.sidebar.markdown("- 📊 [Portfolio Analytics](?tab=portfolio)")
st.sidebar.markdown("- 🔬 [Uncertainty Analysis](?tab=uncertainty)")
st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader("Upload predictions CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize columns for robustness
    df.columns = df.columns.str.strip().str.lower()

    # Compute mispricing_score if missing
    if "mispricing_score" not in df.columns and {"predicted_price_target", "last_price"}.issubset(
        df.columns
    ):
        with pd.option_context("mode.chained_assignment", None):
            denom = pd.to_numeric(df["last_price"], errors="coerce").replace({0: pd.NA})
            df["mispricing_score"] = (
                pd.to_numeric(df["predicted_price_target"], errors="coerce")
                - pd.to_numeric(df["last_price"], errors="coerce")
            ) / denom

    # Multi-select filters
    sectors = st.sidebar.multiselect(
        "Sector", sorted(df["sector"].unique()) if "sector" in df.columns else []
    )
    regions = st.sidebar.multiselect(
        "Region", sorted(df["region"].unique()) if "region" in df.columns else []
    )
    countries = st.sidebar.multiselect(
        "Country", sorted(df["country"].unique()) if "country" in df.columns else []
    )
    trading_countries = st.sidebar.multiselect(
        "Trading Country",
        sorted(df["trading_country"].unique()) if "trading_country" in df.columns else [],
    )
    industries = st.sidebar.multiselect(
        "Industry", sorted(df["industry"].unique()) if "industry" in df.columns else []
    )
    style_classes = st.sidebar.multiselect(
        "Style Class", sorted(df["style_class"].unique()) if "style_class" in df.columns else []
    )
    size_classes = st.sidebar.multiselect(
        "Size Class", sorted(df["size_class"].unique()) if "size_class" in df.columns else []
    )
    earnings_statuses = st.sidebar.multiselect(
        "Next Earnings",
        sorted(df["next_earnings_status"].unique()) if "next_earnings_status" in df.columns else [],
    )
    exchanges = st.sidebar.multiselect(
        "Exchange", sorted(df["exchange"].unique()) if "exchange" in df.columns else []
    )
    units = st.sidebar.multiselect(
        "Unit", sorted(df["unit"].unique()) if "unit" in df.columns else []
    )

    model_version = st.sidebar.selectbox("Model Version", ["v9_10"])

    # Market cap range slider
    if "market_cap" in df.columns:
        min_cap, max_cap = st.sidebar.slider(
            "Market Cap Range (Millions)",
            float(df["market_cap"].min()),
            float(df["market_cap"].max()),
            (float(df["market_cap"].min()), float(df["market_cap"].max())),
        )
        df = df[(df["market_cap"] >= min_cap) & (df["market_cap"] <= max_cap)]

    # Apply filters
    if sectors:
        df = df[df["sector"].isin(sectors)]
    if regions:
        df = df[df["region"].isin(regions)]
    if countries:
        df = df[df["country"].isin(countries)]
    if trading_countries:
        df = df[df["trading_country"].isin(trading_countries)]
    if industries:
        df = df[df["industry"].isin(industries)]
    if style_classes:
        df = df[df["style_class"].isin(style_classes)]
    if size_classes:
        df = df[df["size_class"].isin(size_classes)]
    if earnings_statuses:
        df = df[df["next_earnings_status"].isin(earnings_statuses)]
    if exchanges:
        df = df[df["exchange"].isin(exchanges)]
    if units:
        df = df[df["unit"].isin(units)]

    # Main tabs
    tab_summary, tab1, tab2, tab3, tab_uncertainty, tab_safety, tab4, tab5, tab_gov, tab6 = st.tabs(
        [
            "📋 Executive Summary",
            "📈 Overview",
            "🎯 Stock Ranking",
            "📊 Sector Analysis",
            "🔬 Uncertainty & Calibration",
            "🛡️ Safety Rails",
            "🔍 Data Quality",
            "🤖 Model Performance",
            "🏛️ Model Governance",
            "💼 Portfolio & Risk Metrics",
        ]
    )

    with tab_summary:
        st.title("📋 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stocks", len(df))
        col2.metric("Sectors", df["sector"].nunique() if "sector" in df.columns else 0)
        col3.metric("Regions", df["region"].nunique() if "region" in df.columns else 0)
        if "mispricing_score" in df.columns:
            col4.metric("Avg Mispricing", f"{df['mispricing_score'].mean():.2%}")

        st.info("Use the tabs above to navigate through the comprehensive analytics platform.")

    with tab1:
        st.title("📊 Financial Analytics Overview")

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stocks", len(df))
        col2.metric("Sectors", df["sector"].nunique() if "sector" in df.columns else 0)
        col3.metric("Regions", df["region"].nunique() if "region" in df.columns else 0)
        if "mispricing_score" in df.columns:
            avg_mispricing = df["mispricing_score"].mean()
            col4.metric("Avg Mispricing", f"{avg_mispricing:.2%}")

            # Add gauge chart for overall mispricing score
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_mispricing * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Average Mispricing Score (%)"},
                    delta={"reference": 0},
                    gauge={
                        "axis": {"range": [-50, 50]},
                        "bar": {"color": "#375a7f"},  # Primary
                        "steps": [
                            {"range": [-50, -10], "color": "#e74c3c"},  # Danger
                            {"range": [-10, 10], "color": "#adb5bd"},  # Neutral
                            {"range": [10, 50], "color": "#00bc8c"},  # Success
                        ],
                        "threshold": {
                            "line": {"color": "#e74c3c", "width": 4},  # Danger
                            "thickness": 0.75,
                            "value": avg_mispricing * 100,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=300, template="plotly_dark", font_family="Arial")
            st.plotly_chart(fig_gauge, width="stretch")

        # Interactive scatter plot
        if "mispricing_score" in df.columns and "market_cap" in df.columns:
            fig = px.scatter(
                df,
                x="market_cap",
                y="mispricing_score",
                color="sector" if "sector" in df.columns else None,
                size="last_price" if "last_price" in df.columns else None,
                hover_data=["ticker"] if "ticker" in df.columns else None,
                title="Mispricing Score vs Market Cap",
                labels={"mispricing_score": "Mispricing Score", "market_cap": "Market Cap"},
                template="plotly_dark",
            )
            fig.update_layout(font_family="Arial")
            st.plotly_chart(fig, width="stretch")

    with tab2:
        st.title("🎯 Stock Rankings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟢 Most Undervalued")
            if "mispricing_score" in df.columns:
                undervalued = df.nlargest(10, "mispricing_score")[
                    ["ticker", "sector", "mispricing_score", "last_price"]
                ]
                st.dataframe(undervalued, width="stretch")

        with col2:
            st.subheader("🔴 Most Overvalued")
            if "mispricing_score" in df.columns:
                overvalued = df.nsmallest(10, "mispricing_score")[
                    ["ticker", "sector", "mispricing_score", "last_price"]
                ]
                st.dataframe(overvalued, width="stretch")

        # Sector-specific rankings
        if "sector" in df.columns:
            st.subheader("📊 Top Opportunities by Sector")
            selected_sector = st.selectbox("Select Sector", df["sector"].unique())
            sector_df = df[df["sector"] == selected_sector]
            if "mispricing_score" in sector_df.columns:
                top_sector = sector_df.nlargest(5, "mispricing_score")
                st.dataframe(top_sector, width="stretch")

    with tab3:
        st.title("📊 Sector Analysis")

        # Financial metrics dashboard by sector
        dashboard = calculate_financial_metrics_dashboard(df, group_by="sector")

        # Convert to DataFrame for display
        if "valuation" in dashboard:
            st.subheader("💰 Valuation Metrics by Sector")
            val_df = pd.DataFrame(dashboard["valuation"]).T
            st.dataframe(val_df, width="stretch")

        # Sector performance heatmap
        if "sector" in df.columns and "region" in df.columns and "mispricing_score" in df.columns:
            st.subheader("🌡️ Sector-Region Performance Heatmap")
            pivot = df.pivot_table(
                values="mispricing_score", index="sector", columns="region", aggfunc="mean"
            )
            fig = px.imshow(
                pivot,
                text_auto=".2f",
                aspect="auto",
                title="Average Mispricing Score by Sector and Region",
                template="plotly_dark",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(font_family="Arial")
            st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("📊 Phase 9.3: Advanced Feature Analysis")
        col1, col2 = st.columns(2)
        with col1:
            render_streamlit_artifact("eda", "correlation", height=500)
        with col2:
            render_streamlit_artifact("eda", "distributions", height=500)

        render_streamlit_artifact("eda", "sector_bubbles", height=600)
        render_streamlit_artifact("eda", "sector_heatmap", height=600)
        render_streamlit_artifact("eda", "regional_radar", height=600)

    with tab_uncertainty:
        st.title("🔬 Uncertainty & Calibration")
        st.subheader("📊 Prediction Interval Analysis")
        col1, col2 = st.columns(2)
        with col1:
            render_streamlit_artifact("uncertainty", "interval_width", height=500)
        with col2:
            render_streamlit_artifact("uncertainty", "coverage_heatmap", height=500)

        st.divider()
        st.subheader("🎯 Calibration Quality")
        render_streamlit_artifact("uncertainty", "reliability_diagram", height=550)

        st.divider()
        st.subheader("⚖️ Sector Bias Calibration")
        render_streamlit_artifact("calibration", "sector_bias", height=700)

    with tab_safety:
        st.title("🛡️ Safety Rails")
        st.subheader("Safety Rails Monitoring")
        render_streamlit_artifact("safety_rails", "winsorization")
        render_streamlit_artifact("safety_rails", "violations")
        render_streamlit_artifact("safety_rails", "sensitivity")

    with tab4:
        st.title("🔍 Data Quality Monitoring")

        # Real-time quality alerts
        alerts = generate_data_quality_alerts(df)

        if alerts:
            for alert in alerts:
                alert_type = alert.get("severity", "info")
                if alert_type == "critical":
                    st.error(f"🚨 {alert['message']}")
                elif alert_type == "warning":
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No data quality issues detected!")

        # Missing value heatmap
        st.subheader("📉 Missing Values Analysis")
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_df = pd.DataFrame({"Column": missing_pct.index, "Missing %": missing_pct.values})
        fig = px.bar(
            missing_df.head(20),
            x="Missing %",
            y="Column",
            orientation="h",
            title="Top 20 Columns by Missing Data %",
            template="plotly_dark",
            color_discrete_sequence=["#375a7f"],  # Primary
        )
        fig.update_layout(font_family="Arial")
        st.plotly_chart(fig, width="stretch")

        st.divider()
        st.subheader("Comprehensive Data Quality Dashboard")
        render_streamlit_artifact("eda", "data_quality", height=800)
        render_streamlit_artifact("eda", "outliers", height=600)

    with tab5:
        st.title("🤖 Model Performance Analytics")

        # Prediction accuracy metrics
        if all(
            col in df.columns for col in ["predicted_price_target", "price_target", "last_price"]
        ):
            st.subheader("📊 Prediction vs Analyst Comparison")

            # Calculate errors if not already present
            if "prediction_error_pct" not in df.columns:
                pred_error = (
                    abs(df["predicted_price_target"] - df["price_target"]) / df["price_target"]
                )
            else:
                pred_error = df["prediction_error_pct"] / 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{pred_error.mean():.2%}")
            col2.metric("Median Error", f"{pred_error.median():.2%}")
            col3.metric("RMSE", f"{(pred_error**2).mean()**0.5:.2%}")

            # Error distribution
            fig = px.histogram(
                pred_error,
                nbins=50,
                title="Prediction Error Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#375a7f"],  # Primary
            )
            fig.update_layout(font_family="Arial")
            st.plotly_chart(fig, width="stretch")

            # Residual plot using graph_objects for more control
            residuals = df["predicted_price_target"] - df["price_target"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df["price_target"],
                    y=residuals,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=residuals,
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Residual"),
                    ),
                    text=df["ticker"] if "ticker" in df.columns else None,
                    hovertemplate="<b>%{text}</b><br>Target: %{x}<br>Residual: %{y}<extra></extra>",
                )
            )
            fig.add_hline(
                y=0, line_dash="dash", line_color="#e74c3c", annotation_text="Zero Error"
            )  # Danger
            fig.update_layout(
                title="Residual Plot: Predicted vs Actual Target",
                xaxis_title="Actual Target",
                yaxis_title="Residual",
                hovermode="closest",
                template="plotly_dark",
                font_family="Arial",
            )
            st.plotly_chart(fig, width="stretch")

            # Model vs Analyst disagreement analysis
            if "model_analyst_diff_pct" in df.columns:
                st.subheader("🎯 Model-Analyst Disagreement Analysis")

                disagreement_fig = px.histogram(
                    df,
                    x="model_analyst_diff_pct",
                    nbins=50,
                    title="Distribution of Model-Analyst Disagreement",
                    labels={"model_analyst_diff_pct": "Difference (%)"},
                    template="plotly_dark",
                    color_discrete_sequence=["#375a7f"],  # Primary
                )
                disagreement_fig.update_layout(font_family="Arial")
                disagreement_fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text="Perfect Agreement",  # Danger
                )
                st.plotly_chart(disagreement_fig, width="stretch")

                # High-conviction disagreements
                high_disagreement = df[abs(df["model_analyst_diff_pct"]) > 10].nlargest(
                    10, "model_analyst_diff_pct"
                )
                if len(high_disagreement) > 0:
                    st.subheader("📌 High-Conviction Disagreements (>10% difference)")
                    display_cols = [
                        c
                        for c in [
                            "ticker",
                            "sector",
                            "price_target",
                            "predicted_price_target",
                            "model_analyst_diff_pct",
                        ]
                        if c in high_disagreement.columns
                    ]
                    st.dataframe(high_disagreement[display_cols], width="stretch")

            # Error by sector analysis
            if "prediction_error_pct" in df.columns and "sector" in df.columns:
                st.subheader("📊 Error Analysis by Sector")
                sector_errors = (
                    df.groupby("sector")["prediction_error_pct"]
                    .agg(["mean", "median", "std"])
                    .round(2)
                )

                fig = px.bar(
                    sector_errors.reset_index(),
                    x="sector",
                    y="mean",
                    error_y="std",
                    title="Mean Prediction Error by Sector (with Std Dev)",
                    labels={"mean": "Mean Error %", "sector": "Sector"},
                    template="plotly_dark",
                    color_discrete_sequence=["#375a7f"],  # Primary
                )
                fig.update_layout(xaxis_tickangle=-45, font_family="Arial")
                st.plotly_chart(fig, width="stretch")

                st.dataframe(sector_errors, width="stretch")

            # Financial metrics display
            if any(col in df.columns for col in ["p_e", "p_b", "roe", "ev_ebitda"]):
                st.subheader("💰 Key Financial Metrics")
                financial_cols = [
                    c
                    for c in [
                        "ticker",
                        "sector",
                        "p_e",
                        "p_b",
                        "roe",
                        "roa",
                        "ev_ebitda",
                        "operating_margin",
                        "debt_to_equity",
                    ]
                    if c in df.columns
                ]
                if financial_cols:
                    st.dataframe(df[financial_cols].head(20), width="stretch")

    with tab_gov:
        st.title("🏛️ Model Governance & Lineage")
        render_streamlit_artifact("governance", "error_map")

        # Model Card
        governance_dir = Path(__file__).parent.parent.parent / "outputs" / "governance"
        if governance_dir.exists():
            model_cards = list(governance_dir.glob("model_card_*.md"))
            if model_cards:
                latest_card = sorted(model_cards)[-1]
                with open(latest_card, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content)

                # Download button
                st.download_button(
                    label="Download Model Card",
                    data=content,
                    file_name=latest_card.name,
                    mime="text/markdown",
                )
            else:
                st.info("⚠️ Model Card not available. Run Section 9.8.")
        else:
            st.info("⚠️ Model Governance outputs not available.")

    with tab6:
        st.title("💼 Portfolio Optimization & Risk Metrics")

        st.markdown(
            """
        This section displays portfolio optimization results and risk metrics analysis.
        These visualizations are generated from Section 10 of the ml_finance_model_main.ipynb notebook.
        
        **Section 10 artifacts** are now located in `outputs/portfolio/` following the enhanced reporting structure.
        """
        )

        # Portfolio artifacts path (Section 10 structure)
        portfolio_path = Path("outputs/portfolio")

        # Section 10.2: Universe & Filters Diagnostics
        st.subheader("📊 Universe & Filters Diagnostics")
        universe_summary_file = portfolio_path / "portfolio_universe_summary.html"
        if universe_summary_file.exists():
            try:
                with open(universe_summary_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=400, scrolling=True)
            except Exception as e:
                st.warning(f"Could not load universe summary: {e}")
        else:
            st.info(
                "Run Section 10.2 of ml_finance_model_main.ipynb to generate universe diagnostics."
            )

        st.divider()

        # Section 10.3: Expected Returns & Risk Inputs QA
        st.subheader("📈 Expected Returns & Risk Inputs")
        col1, col2 = st.columns(2)

        with col1:
            returns_dist_file = portfolio_path / "expected_returns_distribution.html"
            if returns_dist_file.exists():
                try:
                    with open(returns_dist_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=400, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load returns distribution: {e}")
            else:
                st.info("Returns distribution not found (Section 10.3)")

        with col2:
            corr_heatmap_file = portfolio_path / "risk_correlation_heatmap.html"
            if corr_heatmap_file.exists():
                try:
                    with open(corr_heatmap_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=400, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load correlation heatmap: {e}")
            else:
                st.info("Correlation heatmap not found (Section 10.3)")

        st.divider()

        # Section 10.4: Efficient Frontier
        st.subheader("🎯 Efficient Frontier & Constraints")
        efficient_frontier_file = portfolio_path / "efficient_frontier.html"
        if efficient_frontier_file.exists():
            try:
                with open(efficient_frontier_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=650, scrolling=True)
            except Exception as e:
                st.warning(f"Could not load efficient frontier visualization: {e}")
        else:
            st.warning("⚠️ Efficient frontier visualization not found.")
            st.info(
                "Run Section 10.4 of ml_finance_model_main.ipynb to generate optimization visualizations."
            )

        st.divider()

        # Section 10.5: Risk Decomposition
        st.subheader("🔍 Portfolio Breakdown & Risk Decomposition")
        risk_decomp_file = portfolio_path / "risk_decomposition.html"
        if risk_decomp_file.exists():
            try:
                with open(risk_decomp_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600, scrolling=True)
            except Exception as e:
                st.warning(f"Could not load risk decomposition: {e}")
        else:
            st.info("Risk decomposition not found. Run Section 10.5 to generate.")

        # Stress tests
        stress_tests_file = portfolio_path / "stress_tests_dashboard.html"
        if stress_tests_file.exists():
            with st.expander("⚡ Stress Tests", expanded=False):
                try:
                    with open(stress_tests_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load stress tests: {e}")

        st.divider()

        # Section 10.6: Backtesting
        st.subheader("📉 Backtesting & Performance Attribution")
        backtest_file = portfolio_path / "backtest_performance.html"
        if backtest_file.exists():
            try:
                with open(backtest_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=650, scrolling=True)
            except Exception as e:
                st.warning(f"Could not load backtest performance: {e}")
        else:
            st.info("Backtest performance not found. Run Section 10.6 to generate.")

        # Attribution
        attribution_file = portfolio_path / "performance_attribution.html"
        if attribution_file.exists():
            with st.expander("📊 Performance Attribution", expanded=False):
                try:
                    with open(attribution_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load attribution: {e}")

        st.divider()

        # Section 10.7: Risk Management Dashboard
        st.subheader("🛡️ Risk Management Dashboard")
        risk_mgmt_file = portfolio_path / "risk_management_dashboard.html"
        if risk_mgmt_file.exists():
            try:
                with open(risk_mgmt_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=700, scrolling=True)
            except Exception as e:
                st.warning(f"Could not load risk management dashboard: {e}")
        else:
            st.warning("⚠️ Risk management dashboard not found.")
            st.info(
                "Run Section 10.7 of ml_finance_model_main.ipynb to generate risk management visuals."
            )

        st.divider()

        # Section 10.8: Multi-Period Comparison & Summary
        st.subheader("🧭 Portfolio Summary & Multi-Period Comparison")

        # Multi-period comparison (now in portfolio/)
        multi_period_file = portfolio_path / "portfolio_multi_period_comparison.html"
        with st.expander("Multi-Period Performance Comparison", expanded=False):
            if multi_period_file.exists():
                try:
                    with open(multi_period_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load multi-period comparison: {e}")
            else:
                st.info("Multi-period comparison not found. Run Section 10.8 to generate.")

        # Factor exposure dashboard (legacy analytics/ path for backward compatibility)
        analytics_path = Path("outputs/analytics")
        factor_exposure_file = analytics_path / "portfolio_factor_exposure_dashboard.html"
        with st.expander("Factor Exposure Dashboard", expanded=False):
            if factor_exposure_file.exists():
                try:
                    with open(factor_exposure_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load factor exposure dashboard: {e}")
            else:
                st.info(
                    "Factor exposure HTML not found. Run Section 10.6 in the notebook to "
                    "generate `portfolio_factor_exposure_dashboard.html`."
                )

        # Rebalancing widget snapshot
        rebalance_file = analytics_path / "portfolio_rebalance_widget.html"
        with st.expander("Rebalancing Suggestions (Static Example)", expanded=False):
            if rebalance_file.exists():
                try:
                    with open(rebalance_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
                except Exception as e:
                    st.warning(f"Could not load rebalance widget snapshot: {e}")
            else:
                st.info(
                    "Rebalance widget HTML not found. Run Section 10.6 in the notebook to "
                    "generate `portfolio_rebalance_widget.html`."
                )

        # Summary of portfolio optimization features
        st.subheader("📋 Portfolio Optimization Features")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Optimization Methods:**
            - ✅ Maximum Sharpe Ratio
            - ✅ Minimum Volatility
            - ✅ Target Return Optimization
            - ✅ Efficient Frontier Generation
            """
            )

        with col2:
            st.markdown(
                """
            **Risk Metrics:**
            - 📊 Value at Risk (VaR)
            - 📊 Conditional VaR (CVaR)
            - 📊 Sharpe Ratio
            - 📊 Sortino Ratio
            - 📊 Maximum Drawdown
            """
            )

        st.info(
            "💡 To update these visualizations, run ml_finance_model_main.ipynb Section 10 with your latest data."
        )

else:
    st.info("👆 Upload a predictions CSV file to start analysis")
