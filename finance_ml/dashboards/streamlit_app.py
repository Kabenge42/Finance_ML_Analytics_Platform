"""
Interactive Streamlit Dashboard for Finance ML Analytics
Run: streamlit run finance_ml/dashboards/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

try:
    from finance_ml.eval import (
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
    from finance_ml.eval import (
        prepare_plotly_dashboard_data,
        calculate_mispricing_score,
        rank_stocks_by_sector,
        calculate_financial_metrics_dashboard,
        generate_data_quality_alerts,
    )

st.set_page_config(page_title="Finance ML Analytics", layout="wide", page_icon="📊")

# Sidebar filters
st.sidebar.title("🔍 Filters")
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
        "Sector", df["sector"].unique() if "sector" in df.columns else []
    )
    regions = st.sidebar.multiselect(
        "Region", df["region"].unique() if "region" in df.columns else []
    )

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

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📈 Overview",
            "🎯 Stock Ranking",
            "📊 Sector Analysis",
            "🔍 Data Quality",
            "🤖 Model Performance",
        ]
    )

    with tab1:
        st.title("📊 Financial Analytics Overview")

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stocks", len(df))
        col2.metric("Sectors", df["sector"].nunique() if "sector" in df.columns else 0)
        col3.metric("Regions", df["region"].nunique() if "region" in df.columns else 0)
        if "mispricing_score" in df.columns:
            col4.metric("Avg Mispricing", f"{df['mispricing_score'].mean():.2%}")

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
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.title("🎯 Stock Rankings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟢 Most Undervalued")
            if "mispricing_score" in df.columns:
                undervalued = df.nlargest(10, "mispricing_score")[
                    ["ticker", "sector", "mispricing_score", "last_price"]
                ]
                st.dataframe(undervalued, use_container_width=True)

        with col2:
            st.subheader("🔴 Most Overvalued")
            if "mispricing_score" in df.columns:
                overvalued = df.nsmallest(10, "mispricing_score")[
                    ["ticker", "sector", "mispricing_score", "last_price"]
                ]
                st.dataframe(overvalued, use_container_width=True)

        # Sector-specific rankings
        if "sector" in df.columns:
            st.subheader("📊 Top Opportunities by Sector")
            selected_sector = st.selectbox("Select Sector", df["sector"].unique())
            sector_df = df[df["sector"] == selected_sector]
            if "mispricing_score" in sector_df.columns:
                top_sector = sector_df.nlargest(5, "first")
                st.dataframe(top_sector, use_container_width=True)

    with tab3:
        st.title("📊 Sector Analysis")

        # Financial metrics dashboard by sector
        dashboard = calculate_financial_metrics_dashboard(df, group_by="sector")

        # Convert to DataFrame for display
        if "valuation" in dashboard:
            st.subheader("💰 Valuation Metrics by Sector")
            val_df = pd.DataFrame(dashboard["valuation"]).T
            st.dataframe(val_df, use_container_width=True)

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
            )
            st.plotly_chart(fig, use_container_width=True)

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
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.title("🤖 Model Performance Analytics")

        # Prediction accuracy metrics
        if all(
            col in df.columns for col in ["predicted_price_target", "price_target", "last_price"]
        ):
            st.subheader("📊 Prediction vs Analyst Comparison")

            # Calculate errors
            pred_error = abs(df["predicted_price_target"] - df["price_target"]) / df["price_target"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{pred_error.mean():.2%}")
            col2.metric("Median Error", f"{pred_error.median():.2%}")
            col3.metric("RMSE", f"{(pred_error**2).mean()**0.5:.2%}")

            # Error distribution
            fig = px.histogram(pred_error, nbins=50, title="Prediction Error Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            residuals = df["predicted_price_target"] - df["price_target"]
            fig = px.scatter(
                x=df["price_target"],
                y=residuals,
                title="Residual Plot: Predicted vs Actual Target",
                labels={"x": "Actual Target", "y": "Residual"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload a predictions CSV file to start analysis")
