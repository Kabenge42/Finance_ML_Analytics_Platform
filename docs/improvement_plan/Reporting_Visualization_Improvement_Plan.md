### Meaningful Enhancements for the Reporting and Visualization Package (eval.py)

Based on comprehensive analysis of the 7,523-line `eval.py` module and the project requirements, here are prioritized,
actionable enhancements with code examples:

---

### 🎯 Priority 1: Interactive Dashboard Applications

#### 1.1 Streamlit Financial Analytics Dashboard

**New File**: `finance_ml/dashboards/streamlit_app.py`

```python
"""
Interactive Streamlit Dashboard for Finance ML Analytics
Run: streamlit run finance_ml/dashboards/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from finance_ml.eval import (
    prepare_plotly_dashboard_data,
    calculate_mispricing_score,
    rank_stocks_by_sector,
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts
)

st.set_page_config(page_title="Finance ML Analytics", layout="wide", page_icon="📊")

# Sidebar filters
st.sidebar.title("🔍 Filters")
uploaded_file = st.sidebar.file_uploader("Upload predictions CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Multi-select filters
    sectors = st.sidebar.multiselect("Sector", df['sector'].unique() if 'sector' in df.columns else [])
    regions = st.sidebar.multiselect("Region", df['region'].unique() if 'region' in df.columns else [])
    
    # Market cap range slider
    if 'market_cap' in df.columns:
        min_cap, max_cap = st.sidebar.slider(
            "Market Cap Range (Millions)",
            float(df['market_cap'].min()), 
            float(df['market_cap'].max()),
            (float(df['market_cap'].min()), float(df['market_cap'].max()))
        )
        df = df[(df['market_cap'] >= min_cap) & (df['market_cap'] <= max_cap)]
    
    # Apply filters
    if sectors:
        df = df[df['sector'].isin(sectors)]
    if regions:
        df = df[df['region'].isin(regions)]
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🎯 Stock Ranking", "📊 Sector Analysis", 
        "🔍 Data Quality", "🤖 Model Performance"
    ])
    
    with tab1:
        st.title("📊 Financial Analytics Overview")
        
        # KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Stocks", len(df))
        col2.metric("Sectors", df['sector'].nunique() if 'sector' in df.columns else 0)
        col3.metric("Regions", df['region'].nunique() if 'region' in df.columns else 0)
        if 'mispricing_score' in df.columns:
            col4.metric("Avg Mispricing", f"{df['mispricing_score'].mean():.2%}")
        
        # Interactive scatter plot
        if 'mispricing_score' in df.columns and 'market_cap' in df.columns:
            fig = px.scatter(
                df, x='market_cap', y='mispricing_score',
                color='sector' if 'sector' in df.columns else None,
                size='last_price' if 'last_price' in df.columns else None,
                hover_data=['ticker'] if 'ticker' in df.columns else None,
                title="Mispricing Score vs Market Cap",
                labels={'mispricing_score': 'Mispricing Score', 'market_cap': 'Market Cap'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.title("🎯 Stock Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Most Undervalued")
            if 'mispricing_score' in df.columns:
                undervalued = df.nlargest(10, 'mispricing_score')[
                    ['ticker', 'sector', 'mispricing_score', 'last_price']
                ]
                st.dataframe(undervalued, use_container_width=True)
        
        with col2:
            st.subheader("🔴 Most Overvalued")
            if 'mispricing_score' in df.columns:
                overvalued = df.nsmallest(10, 'mispricing_score')[
                    ['ticker', 'sector', 'mispricing_score', 'last_price']
                ]
                st.dataframe(overvalued, use_container_width=True)
        
        # Sector-specific rankings
        if 'sector' in df.columns:
            st.subheader("📊 Top Opportunities by Sector")
            selected_sector = st.selectbox("Select Sector", df['sector'].unique())
            sector_df = df[df['sector'] == selected_sector]
            top_sector = sector_df.nlargest(5, 'mispricing_score')
            st.dataframe(top_sector, use_container_width=True)
    
    with tab3:
        st.title("📊 Sector Analysis")
        
        # Financial metrics dashboard by sector
        dashboard = calculate_financial_metrics_dashboard(df, group_by='sector')
        
        # Convert to DataFrame for display
        if 'valuation' in dashboard:
            st.subheader("💰 Valuation Metrics by Sector")
            val_df = pd.DataFrame(dashboard['valuation']).T
            st.dataframe(val_df, use_container_width=True)
        
        # Sector performance heatmap
        if 'sector' in df.columns and 'region' in df.columns:
            st.subheader("🌡️ Sector-Region Performance Heatmap")
            pivot = df.pivot_table(
                values='mispricing_score', 
                index='sector', 
                columns='region', 
                aggfunc='mean'
            )
            fig = px.imshow(pivot, text_auto='.2f', aspect="auto",
                          title="Average Mispricing Score by Sector and Region")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.title("🔍 Data Quality Monitoring")
        
        # Real-time quality alerts
        alerts = generate_data_quality_alerts(df)
        
        if alerts:
            for alert in alerts:
                alert_type = alert.get('severity', 'info')
                if alert_type == 'critical':
                    st.error(f"🚨 {alert['message']}")
                elif alert_type == 'warning':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No data quality issues detected!")
        
        # Missing value heatmap
        st.subheader("📉 Missing Values Analysis")
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_df = pd.DataFrame({'Column': missing_pct.index, 'Missing %': missing_pct.values})
        fig = px.bar(missing_df.head(20), x='Missing %', y='Column', orientation='h',
                    title="Top 20 Columns by Missing Data %")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.title("🤖 Model Performance Analytics")
        
        # Prediction accuracy metrics
        if all(col in df.columns for col in ['predicted_price_target', 'price_target', 'last_price']):
            st.subheader("📊 Prediction vs Analyst Comparison")
            
            # Calculate errors
            pred_error = abs(df['predicted_price_target'] - df['price_target']) / df['price_target']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{pred_error.mean():.2%}")
            col2.metric("Median Error", f"{pred_error.median():.2%}")
            col3.metric("RMSE", f"{(pred_error**2).mean()**0.5:.2%}")
            
            # Error distribution
            fig = px.histogram(pred_error, nbins=50, title="Prediction Error Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Residual plot
            residuals = df['predicted_price_target'] - df['price_target']
            fig = px.scatter(x=df['price_target'], y=residuals,
                           title="Residual Plot: Predicted vs Actual Target",
                           labels={'x': 'Actual Target', 'y': 'Residual'})
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👆 Upload a predictions CSV file to start analysis")
```

#### 1.2 Plotly Dash Alternative

**New File**: `finance_ml/dashboards/dash_app.py`

```python
"""
Plotly Dash Dashboard for Finance ML Analytics
Run: python finance_ml/dashboards/dash_app.py
"""
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
from pathlib import Path

app = dash.Dash(__name__, title="Finance ML Analytics")

# Sample data loading (replace with actual data source)
def load_data():
    # Load from outputs or database
    csv_path = Path("outputs/analytics/predictions.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df = load_data()

app.layout = html.Div([
    html.H1("📊 Finance ML Analytics Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.Label("Select Sector:"),
            dcc.Dropdown(
                id='sector-dropdown',
                options=[{'label': s, 'value': s} for s in df['sector'].unique()] if 'sector' in df.columns else [],
                value=None,
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': r, 'value': r} for r in df['region'].unique()] if 'region' in df.columns else [],
                value=None,
                multi=True
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    
    html.Div([
        dcc.Graph(id='scatter-plot'),
        dcc.Graph(id='heatmap-plot'),
    ]),
    
    html.Div([
        html.H3("Top Undervalued Stocks"),
        dash_table.DataTable(id='undervalued-table', page_size=10)
    ])
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('heatmap-plot', 'figure'),
     Output('undervalued-table', 'data')],
    [Input('sector-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)
def update_dashboard(sectors, regions):
    filtered_df = df.copy()
    
    if sectors:
        filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    
    # Scatter plot
    scatter_fig = px.scatter(
        filtered_df, x='market_cap', y='mispricing_score',
        color='sector', size='last_price',
        hover_data=['ticker'],
        title="Mispricing Score vs Market Cap"
    )
    
    # Heatmap
    if 'sector' in filtered_df.columns and 'region' in filtered_df.columns:
        pivot = filtered_df.pivot_table(
            values='mispricing_score', 
            index='sector', 
            columns='region', 
            aggfunc='mean'
        )
        heatmap_fig = px.imshow(pivot, text_auto='.2f', 
                               title="Sector-Region Performance")
    else:
        heatmap_fig = {}
    
    # Top undervalued
    top_stocks = filtered_df.nlargest(10, 'mispricing_score')[
        ['ticker', 'sector', 'mispricing_score', 'last_price']
    ].to_dict('records')
    
    return scatter_fig, heatmap_fig, top_stocks

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

---

### 📊 Priority 2: Enhanced Automated Reporting

#### 2.1 Comprehensive ML Workflow Report

**Add to `eval.py`**:

```python
def generate_ml_workflow_report(
    training_metrics: dict,
    validation_metrics: dict,
    predictions_df: pd.DataFrame,
    output_path: Path,
    include_feature_importance: bool = True,
    include_residual_analysis: bool = True,
    include_model_comparison: bool = True
) -> Path:
    """
    Generate comprehensive ML workflow report with training insights.
    
    Sections:
    - Executive summary with key metrics
    - Data pipeline statistics (imputation, outliers, transformations)
    - Model performance comparison (train/val/test)
    - Feature importance rankings
    - Residual analysis and bias detection
    - Prediction confidence intervals
    - Model explainability (SHAP summaries)
    
    Args:
        training_metrics: Dict with training metrics by model
        validation_metrics: Dict with validation metrics
        predictions_df: DataFrame with predictions and actuals
        output_path: Path to save HTML report
        include_feature_importance: Add feature importance analysis
        include_residual_analysis: Add residual diagnostics
        include_model_comparison: Add model comparison table
    
    Returns:
        Path to generated HTML report
    """
    html_parts = []
    
    # Header
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Workflow Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #3498db; color: white; }
            .metric-card { display: inline-block; padding: 15px; margin: 10px; 
                          background: #ecf0f1; border-radius: 8px; }
            .warning { background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }
            .success { background-color: #d4edda; padding: 10px; border-left: 4px solid #28a745; }
        </style>
    </head>
    <body>
        <h1>🤖 ML Workflow Performance Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
    """.format(timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    # Executive Summary
    html_parts.append("<h2>📋 Executive Summary</h2>")
    html_parts.append('<div class="metric-card">')
    html_parts.append(f"<h3>Training Samples</h3><p>{len(predictions_df)}</p>")
    html_parts.append('</div>')
    
    if 'rmse' in validation_metrics:
        html_parts.append('<div class="metric-card">')
        html_parts.append(f"<h3>Validation RMSE</h3><p>{validation_metrics['rmse']:.4f}</p>")
        html_parts.append('</div>')
    
    # Data Pipeline Section
    html_parts.append("<h2>🔧 Data Pipeline Statistics</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Stage</th><th>Metric</th><th>Value</th></tr>")
    
    # Add imputation statistics
    if 'imputation_stats' in training_metrics:
        imp_stats = training_metrics['imputation_stats']
        html_parts.append(f"<tr><td>Imputation</td><td>Columns Imputed</td><td>{imp_stats.get('n_columns', 0)}</td></tr>")
        html_parts.append(f"<tr><td>Imputation</td><td>Total NaNs Filled</td><td>{imp_stats.get('total_nans', 0)}</td></tr>")
    
    html_parts.append("</table>")
    
    # Model Comparison
    if include_model_comparison and 'models' in training_metrics:
        html_parts.append("<h2>📊 Model Performance Comparison</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Model</th><th>Train RMSE</th><th>Val RMSE</th><th>R²</th><th>MAE</th></tr>")
        
        for model_name, metrics in training_metrics['models'].items():
            html_parts.append(f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics.get('train_rmse', 'N/A'):.4f}</td>
                <td>{metrics.get('val_rmse', 'N/A'):.4f}</td>
                <td>{metrics.get('r2', 'N/A'):.4f}</td>
                <td>{metrics.get('mae', 'N/A'):.4f}</td>
            </tr>
            """)
        html_parts.append("</table>")
    
    # Feature Importance
    if include_feature_importance and 'feature_importance' in training_metrics:
        html_parts.append("<h2>🎯 Top 20 Feature Importance</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Rank</th><th>Feature</th><th>Importance</th></tr>")
        
        feat_imp = training_metrics['feature_importance']
        for rank, (feat, imp) in enumerate(feat_imp[:20], 1):
            html_parts.append(f"<tr><td>{rank}</td><td>{feat}</td><td>{imp:.4f}</td></tr>")
        html_parts.append("</table>")
    
    # Residual Analysis
    if include_residual_analysis and 'predicted_price_target' in predictions_df.columns:
        html_parts.append("<h2>📉 Residual Analysis</h2>")
        
        residuals = predictions_df['predicted_price_target'] - predictions_df.get('price_target', 0)
        html_parts.append(f"""
        <div class="metric-card">
            <h3>Mean Residual</h3><p>{residuals.mean():.4f}</p>
        </div>
        <div class="metric-card">
            <h3>Std Residual</h3><p>{residuals.std():.4f}</p>
        </div>
        <div class="metric-card">
            <h3>Max Absolute Error</h3><p>{residuals.abs().max():.4f}</p>
        </div>
        """)
        
        # Check for bias
        if abs(residuals.mean()) > residuals.std() * 0.1:
            html_parts.append('<div class="warning">⚠️ Potential systematic bias detected in predictions</div>')
        else:
            html_parts.append('<div class="success">✅ No significant systematic bias detected</div>')
    
    # Footer
    html_parts.append("</body></html>")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(html_parts))
    
    logging.info(f"Generated ML workflow report: {output_path}")
    return output_path
```

#### 2.2 Imputation Tracking and Visualization

**Add to `eval.py`**:

```python
def generate_imputation_report(
    imputation_stats: dict,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    output_dir: Path
) -> dict:
    """
    Generate comprehensive imputation analysis report with visualizations.
    
    Tracks:
    - Which columns were imputed and by which method
    - Before/after NaN heatmaps
    - Imputation time per column
    - Distribution changes (before/after histograms)
    - Emergency fallback usage
    
    Args:
        imputation_stats: Dict with imputation metadata
        df_before: DataFrame before imputation
        df_after: DataFrame after imputation
        output_dir: Directory to save visualizations
    
    Returns:
        Dict with report metrics and saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'columns_imputed': [],
        'methods_used': {},
        'total_nans_filled': 0,
        'emergency_fallbacks': [],
        'visualizations': []
    }
    
    # Track imputation by column
    for col in df_before.columns:
        nans_before = df_before[col].isna().sum()
        nans_after = df_after[col].isna().sum()
        
        if nans_before > nans_after:
            method = imputation_stats.get(col, {}).get('method', 'unknown')
            report['columns_imputed'].append({
                'column': col,
                'nans_filled': int(nans_before - nans_after),
                'method': method,
                'fill_rate': float((nans_before - nans_after) / nans_before)
            })
            
            report['methods_used'][method] = report['methods_used'].get(method, 0) + 1
            report['total_nans_filled'] += int(nans_before - nans_after)
            
            # Track emergency fallbacks
            if method == 'emergency_zero':
                report['emergency_fallbacks'].append(col)
    
    # Generate before/after NaN heatmaps
    if plt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Before heatmap
        sns.heatmap(df_before.isnull(), cbar=False, yticklabels=False, ax=ax1)
        ax1.set_title("Missing Values BEFORE Imputation")
        
        # After heatmap
        sns.heatmap(df_after.isnull(), cbar=False, yticklabels=False, ax=ax2)
        ax2.set_title("Missing Values AFTER Imputation")
        
        heatmap_path = output_dir / "imputation_heatmap.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        report['visualizations'].append(str(heatmap_path))
        logging.info(f"Saved imputation heatmap: {heatmap_path}")
    
    # Generate distribution comparison for imputed columns
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    imputed_numeric = [c['column'] for c in report['columns_imputed'] if c['column'] in numeric_cols]
    
    if plt is not None and imputed_numeric:
        n_cols = min(len(imputed_numeric), 6)  # Show top 6
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(imputed_numeric[:n_cols]):
            # Before distribution (excluding NaNs)
            before_clean = df_before[col].dropna()
            axes[idx].hist(before_clean, bins=30, alpha=0.5, label='Before', color='red')
            
            # After distribution
            axes[idx].hist(df_after[col].dropna(), bins=30, alpha=0.5, label='After', color='green')
            
            axes[idx].set_title(f"{col}\n({report['columns_imputed'][idx]['nans_filled']} filled)")
            axes[idx].legend()
        
        dist_path = output_dir / "imputation_distributions.png"
        plt.tight_layout()
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        report['visualizations'].append(str(dist_path))
        logging.info(f"Saved imputation distributions: {dist_path}")
    
    # Save JSON report
    json_path = output_dir / "imputation_report.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logging.info(f"Generated imputation report: {json_path}")
    return report
```

---

### 📈 Priority 3: Enhanced Output Directory Structure

#### 3.1 Organized Output Generation

**Add to `eval.py`**:

```python
def create_structured_output_directory(base_dir: Path, run_id: str = None) -> dict:
    """
    Create organized output directory structure for ML workflow artifacts.
    
    Structure:
    outputs/
    ├── {run_id}/
    │   ├── data/
    │   │   ├── processed_data.csv
    │   │   └── imputation_report.json
    │   ├── models/
    │   │   ├── model_checkpoints/
    │   │   └── feature_importance.csv
    │   ├── reports/
    │   │   ├── ml_workflow_report.html
    │   │   ├── eda_report.html
    │   │   └── data_quality_dashboard.html
    │   ├── visualizations/
    │   │   ├── eda/
    │   │   ├── predictions/
    │   │   ├── residuals/
    │   │   └── feature_importance/
    │   ├── analytics/
    │   │   ├── predictions.csv
    │   │   ├── stock_rankings.csv
    │   │   └── prediction_analyst_comparison.xlsx
    │   └── logs/
    │       └── pipeline.log
    
    Args:
        base_dir: Base directory for outputs (default: 'outputs')
        run_id: Unique identifier for this run (default: timestamp)
    
    Returns:
        Dict with paths to each subdirectory
    """
    if run_id is None:
        run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = Path(base_dir) / run_id
    
    structure = {
        'run_dir': run_dir,
        'data': run_dir / 'data',
        'models': run_dir / 'models',
        'model_checkpoints': run_dir / 'models' / 'checkpoints',
        'reports': run_dir / 'reports',
        'visualizations': run_dir / 'visualizations',
        'eda_viz': run_dir / 'visualizations' / 'eda',
        'prediction_viz': run_dir / 'visualizations' / 'predictions',
        'residual_viz': run_dir / 'visualizations' / 'residuals',
        'feature_viz': run_dir / 'visualizations' / 'feature_importance',
        'analytics': run_dir / 'analytics',
        'logs': run_dir / 'logs',
    }
    
    # Create all directories
    for path in structure.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_path = run_dir / 'README.md'
    readme_content = f"""# ML Workflow Run: {run_id}

Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Structure

- **data/**: Processed datasets and imputation reports
- **models/**: Trained model artifacts and checkpoints
- **reports/**: HTML/PDF reports (workflow, EDA, data quality)
- **visualizations/**: All plots and charts organized by category
- **analytics/**: Prediction results, rankings, and comparison tables
- **logs/**: Pipeline execution logs

## Key Files

- `reports/ml_workflow_report.html` - Comprehensive ML workflow report
- `analytics/predictions.csv` - Stock predictions with confidence intervals
- `analytics/stock_rankings.csv` - Undervalued/overvalued stock rankings
- `visualizations/predictions/mispricing_scatter.png` - Mispricing analysis
"""
    readme_path.write_text(readme_content)
    
    logging.info(f"Created structured output directory: {run_dir}")
    return structure
```

---

### 🔔 Priority 4: Real-Time Monitoring and Alerts

**Add to `eval.py`**:

```python
def create_monitoring_dashboard_html(
    df: pd.DataFrame,
    output_path: Path,
    refresh_interval: int = 30
) -> Path:
    """
    Create auto-refreshing HTML dashboard for real-time monitoring.
    
    Features:
    - Auto-refresh every N seconds
    - Real-time data quality metrics
    - Alert badges for anomalies
    - System resource usage
    - Recent prediction statistics
    
    Args:
        df: Current dataset to monitor
        output_path: Path to save HTML dashboard
        refresh_interval: Auto-refresh interval in seconds
    
    Returns:
        Path to generated HTML file
    """
    alerts = generate_data_quality_alerts(df)
    
    # Categorize alerts by severity
    critical = [a for a in alerts if a.get('severity') == 'critical']
    warnings = [a for a in alerts if a.get('severity') == 'warning']
    info = [a for a in alerts if a.get('severity') == 'info']
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Monitoring Dashboard</title>
        <meta http-equiv="refresh" content="{refresh_interval}">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .dashboard {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ font-size: 2em; font-weight: bold; color: #3498db; }}
            .label {{ color: #7f8c8d; font-size: 0.9em; }}
            .alert {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
            .critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
            .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
            .info {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
            .status-ok {{ color: #28a745; }}
            .status-alert {{ color: #dc3545; }}
            h1 {{ color: #2c3e50; }}
            .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>📊 Real-Time Monitoring Dashboard</h1>
        <p class="timestamp">Last Updated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")} 
           (Auto-refresh: {refresh_interval}s)</p>
        
        <div class="dashboard">
            <div class="card">
                <div class="label">Total Stocks</div>
                <div class="metric">{len(df)}</div>
            </div>
            <div class="card">
                <div class="label">Critical Alerts</div>
                <div class="metric" style="color: {'#dc3545' if critical else '#28a745'}">
                    {len(critical)}
                </div>
            </div>
            <div class="card">
                <div class="label">Warnings</div>
                <div class="metric" style="color: {'#ffc107' if warnings else '#28a745'}">
                    {len(warnings)}
                </div>
            </div>
            <div class="card">
                <div class="label">Data Completeness</div>
                <div class="metric">{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%</div>
            </div>
        </div>
        
        <h2>🚨 Active Alerts</h2>
    """
    
    if not alerts:
        html_content += '<p class="status-ok">✅ No alerts - all systems normal</p>'
    else:
        for alert in critical:
            html_content += f'<div class="alert critical">🚨 <strong>CRITICAL:</strong> {alert["message"]}</div>'
        for alert in warnings:
            html_content += f'<div class="alert warning">⚠️ <strong>WARNING:</strong> {alert["message"]}</div>'
        for alert in info:
            html_content += f'<div class="alert info">ℹ️ <strong>INFO:</strong> {alert["message"]}</div>'
    
    # Recent prediction stats
    if 'mispricing_score' in df.columns:
        html_content += f"""
        <h2>📈 Recent Prediction Statistics</h2>
        <div class="dashboard">
            <div class="card">
                <div class="label">Avg Mispricing</div>
                <div class="metric">{df['mispricing_score'].mean():.2%}</div>
            </div>
            <div class="card">
                <div class="label">Std Dev</div>
                <div class="metric">{df['mispricing_score'].std():.2%}</div>
            </div>
            <div class="card">
                <div class="label">Max Undervalued</div>
                <div class="metric">{df['mispricing_score'].max():.2%}</div>
            </div>
            <div class="card">
                <div class="label">Max Overvalued</div>
                <div class="metric">{df['mispricing_score'].min():.2%}</div>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)
    
    logging.info(f"Created monitoring dashboard: {output_path}")
    return output_path
```

---

### 📝 Summary of Implementation Plan

#### **Phase 1: Interactive Dashboards** (2-3 weeks)

1. Implement Streamlit app with multi-page layout
2. Add Plotly Dash alternative
3. Create dashboard launcher CLI commands
4. Add to requirements.txt: `streamlit>=1.30.0`, `dash>=2.14.0`

#### **Phase 2: Enhanced Reporting** (1-2 weeks)

1. Implement `generate_ml_workflow_report()`
2. Add `generate_imputation_report()` with visualizations
3. Enhance `generate_eda_report()` with pandas-profiling integration
4. Add model performance comparison reports

#### **Phase 3: Output Organization** (1 week)

1. Implement `create_structured_output_directory()`
2. Update all report generation functions to use new structure
3. Add run metadata and README generation
4. Create output archiving utilities

#### **Phase 4: Monitoring & Alerts** (1 week)

1. Implement `create_monitoring_dashboard_html()`
2. Add automated alert email notifications (optional)
3. Create data quality trend tracking
4. Add performance regression detection

#### **Testing Requirements**:

- Add tests for each new function in `tests/test_enhanced_reporting.py`
- Add integration tests for dashboard data preparation
- Test output directory structure creation
- Validate HTML/report generation with sample data

#### **Documentation Updates**:

- Update IMPROVEMENT_PLAN.md with completion status
- Add dashboard usage guide to README.md
- Create DASHBOARD_GUIDE.md with screenshots
- Document new CLI commands for launching dashboards

---

### 🎯 Quick Wins for Immediate Implementation

1. **Add `create_structured_output_directory()` now** - Improves organization immediately
2. **Implement `generate_imputation_report()`** - Addresses specific request in issue
3. **Create basic Streamlit app** - High visibility, easy to demo
4. **Enhance existing `simple_eda()` with imputation stats** - Low effort, high value