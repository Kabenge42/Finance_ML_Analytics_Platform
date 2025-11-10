"""
Enhance ml_finance_model_main.ipynb with interactive visualizations in sections 4-10
"""

import json
from pathlib import Path
import copy

notebook_path = Path("ml_finance_model_main.ipynb")
output_path = Path("ml_finance_model_main.ipynb")

# Load notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def create_code_cell(code_lines):
    """Create a new code cell"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines,
    }


def create_markdown_cell(text_lines):
    """Create a new markdown cell"""
    return {"cell_type": "markdown", "metadata": {}, "source": text_lines}


# Find section indices
section_indices = {}
for idx, cell in enumerate(cells):
    if cell["cell_type"] == "markdown":
        source = "".join(cell.get("source", []))
        if "## 4." in source or source.startswith("## 4 "):
            section_indices[4] = idx
        elif "## 5." in source or source.startswith("## 5 "):
            section_indices[5] = idx
        elif "## 6." in source or source.startswith("## 6 "):
            section_indices[6] = idx
        elif "## 7." in source or source.startswith("## 7 "):
            section_indices[7] = idx
        elif "## 8." in source or source.startswith("## 8 "):
            section_indices[8] = idx
        elif "## 9." in source or source.startswith("## 9 "):
            section_indices[9] = idx
        elif "## 10." in source or source.startswith("## 10 "):
            section_indices[10] = idx

print("Found sections:", section_indices)

# Track where to insert cells (working backwards to maintain indices)
insertions = []

# Section 4: Feature Engineering Visualizations
sec4_viz = [
    "# 📊 Section 4 Enhanced Visualizations - Feature Engineering\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE FEATURE ENGINEERING VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "# Feature importance visualization (if available from feature engineering)\n",
    "if 'X_featured' in dir() and X_featured is not None:\n",
    "    print('\\n📈 Feature Importance Analysis...')\n",
    "    \n",
    "    # Calculate feature correlations\n",
    "    import plotly.express as px\n",
    "    import plotly.graph_objects as go\n",
    "    \n",
    "    numeric_features = X_featured.select_dtypes(include=[np.number]).columns[:20]  # Top 20\n",
    "    corr_matrix = X_featured[numeric_features].corr()\n",
    "    \n",
    "    # Interactive correlation heatmap\n",
    "    fig = px.imshow(corr_matrix, \n",
    "                    text_auto='.2f',\n",
    "                    aspect='auto',\n",
    "                    color_continuous_scale='RdBu_r',\n",
    "                    title='Feature Correlation Heatmap (Top 20 Features)')\n",
    "    fig.update_layout(width=900, height=800)\n",
    "    fig.show()\n",
    "    \n",
    "    # Feature distribution comparison by sector\n",
    "    if 'sector' in all_stocks_featured.columns:\n",
    "        print('\\n📊 Feature Distributions by Sector...')\n",
    "        key_features = ['market_cap', 'last_price', 'pe_ratio'] if 'pe_ratio' in all_stocks_featured.columns else ['market_cap', 'last_price']\n",
    "        for feature in key_features:\n",
    "            if feature in all_stocks_featured.columns:\n",
    "                fig = px.box(all_stocks_featured, \n",
    "                            x='sector', \n",
    "                            y=feature,\n",
    "                            color='sector',\n",
    '                            title=f\'{feature.replace("_", " ").title()} Distribution by Sector\',\n',
    "                            points='outliers')\n",
    "                fig.update_layout(showlegend=False, xaxis_tickangle=-45)\n",
    "                fig.show()\n",
    "                break  # Show just one example\n",
    "    \n",
    "    print('✓ Feature engineering visualizations complete')\n",
]

# Section 5: Classification Visualizations
sec5_viz = [
    "# 📊 Section 5 Enhanced Visualizations - Classification Models\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE CLASSIFICATION VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "# Confusion matrix and classification metrics\n",
    "if 'y_test_class' in dir() and 'y_pred_class' in dir():\n",
    "    from finance_ml.ml_workflow.classification.evaluation import (\n",
    "        plot_confusion_matrices,\n",
    "        analyze_calibration\n",
    "    )\n",
    "    import plotly.figure_factory as ff\n",
    "    from sklearn.metrics import confusion_matrix, classification_report\n",
    "    \n",
    "    print('\\n📈 Confusion Matrix Visualization...')\n",
    "    \n",
    "    # Create confusion matrix\n",
    "    cm = confusion_matrix(y_test_class, y_pred_class)\n",
    "    class_names = ['Negative', 'Neutral', 'Positive'] if cm.shape[0] == 3 else [f'Class {i}' for i in range(cm.shape[0])]\n",
    "    \n",
    "    # Interactive confusion matrix heatmap\n",
    "    fig = ff.create_annotated_heatmap(\n",
    "        z=cm,\n",
    "        x=class_names,\n",
    "        y=class_names,\n",
    "        colorscale='Blues',\n",
    "        showscale=True\n",
    "    )\n",
    "    fig.update_layout(\n",
    "        title='Classification Confusion Matrix',\n",
    "        xaxis_title='Predicted',\n",
    "        yaxis_title='Actual',\n",
    "        width=600,\n",
    "        height=600\n",
    "    )\n",
    "    fig.show()\n",
    "    \n",
    "    # Classification report\n",
    "    print('\\n📊 Classification Report:')\n",
    "    print(classification_report(y_test_class, y_pred_class, target_names=class_names))\n",
    "    \n",
    "    # Class distribution\n",
    "    import pandas as pd\n",
    "    class_dist = pd.Series(y_pred_class).value_counts().sort_index()\n",
    "    fig = px.bar(x=class_names, y=class_dist.values,\n",
    "                title='Predicted Class Distribution',\n",
    "                labels={'x': 'Class', 'y': 'Count'},\n",
    "                color=class_names)\n",
    "    fig.update_layout(showlegend=False)\n",
    "    fig.show()\n",
    "    \n",
    "    print('✓ Classification visualizations complete')\n",
]

# Section 6: Regression Model Visualizations
sec6_viz = [
    "# 📊 Section 6 Enhanced Visualizations - Regression Models\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE REGRESSION MODEL VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "# Regression predictions and residuals\n",
    "if 'y_val' in dir() and 'y_val_pred' in dir():\n",
    "    import plotly.express as px\n",
    "    import plotly.graph_objects as go\n",
    "    \n",
    "    print('\\n📈 Prediction vs Actual Scatter Plot...')\n",
    "    \n",
    "    # Predicted vs Actual\n",
    "    fig = go.Figure()\n",
    "    fig.add_trace(go.Scatter(\n",
    "        x=y_val,\n",
    "        y=y_val_pred,\n",
    "        mode='markers',\n",
    "        marker=dict(size=6, opacity=0.6, color='blue'),\n",
    "        name='Predictions'\n",
    "    ))\n",
    "    \n",
    "    # Perfect prediction line\n",
    "    min_val, max_val = y_val.min(), y_val.max()\n",
    "    fig.add_trace(go.Scatter(\n",
    "        x=[min_val, max_val],\n",
    "        y=[min_val, max_val],\n",
    "        mode='lines',\n",
    "        line=dict(color='red', dash='dash'),\n",
    "        name='Perfect Prediction'\n",
    "    ))\n",
    "    \n",
    "    fig.update_layout(\n",
    "        title='Predicted vs Actual Price Targets',\n",
    "        xaxis_title='Actual Price Target',\n",
    "        yaxis_title='Predicted Price Target',\n",
    "        width=800,\n",
    "        height=600\n",
    "    )\n",
    "    fig.show()\n",
    "    \n",
    "    # Residual plot\n",
    "    print('\\n📉 Residual Analysis...')\n",
    "    residuals = y_val_pred - y_val\n",
    "    \n",
    "    fig = go.Figure()\n",
    "    fig.add_trace(go.Scatter(\n",
    "        x=y_val_pred,\n",
    "        y=residuals,\n",
    "        mode='markers',\n",
    "        marker=dict(size=6, opacity=0.6, color='purple'),\n",
    "        name='Residuals'\n",
    "    ))\n",
    "    \n",
    "    # Zero line\n",
    "    fig.add_hline(y=0, line_dash='dash', line_color='red', annotation_text='Zero Error')\n",
    "    \n",
    "    fig.update_layout(\n",
    "        title='Residual Plot - Model Error Analysis',\n",
    "        xaxis_title='Predicted Price Target',\n",
    "        yaxis_title='Residual (Predicted - Actual)',\n",
    "        width=800,\n",
    "        height=600\n",
    "    )\n",
    "    fig.show()\n",
    "    \n",
    "    # Residual distribution\n",
    "    fig = px.histogram(residuals, nbins=50, \n",
    "                      title='Residual Distribution',\n",
    "                      labels={'value': 'Residual', 'count': 'Frequency'})\n",
    "    fig.add_vline(x=0, line_dash='dash', line_color='red')\n",
    "    fig.show()\n",
    "    \n",
    "    print('✓ Regression model visualizations complete')\n",
]

# Section 7: Model Evaluation Visualizations
sec7_viz = [
    "# 📊 Section 7 Enhanced Visualizations - Model Evaluation & Error Analysis\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE MODEL EVALUATION VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "# Comprehensive error analysis\n",
    "if 'all_stocks_featured' in dir() and 'predicted_price_target' in all_stocks_featured.columns:\n",
    "    from finance_ml.ml_workflow.analytics.eval import (\n",
    "        create_region_sector_heatmap,\n",
    "        compute_sector_region_metrics\n",
    "    )\n",
    "    import plotly.express as px\n",
    "    \n",
    "    print('\\n📊 Error Analysis by Sector and Region...')\n",
    "    \n",
    "    # Calculate errors\n",
    "    if 'price_target' in all_stocks_featured.columns:\n",
    "        all_stocks_featured['prediction_error'] = abs(\n",
    "            all_stocks_featured['predicted_price_target'] - all_stocks_featured['price_target']\n",
    "        )\n",
    "        all_stocks_featured['prediction_error_pct'] = (\n",
    "            all_stocks_featured['prediction_error'] / all_stocks_featured['price_target'] * 100\n",
    "        )\n",
    "        \n",
    "        # Error by sector\n",
    "        if 'sector' in all_stocks_featured.columns:\n",
    "            sector_errors = all_stocks_featured.groupby('sector')['prediction_error_pct'].agg(['mean', 'median', 'std']).round(2)\n",
    "            \n",
    "            fig = px.bar(sector_errors.reset_index(), \n",
    "                        x='sector', \n",
    "                        y='mean',\n",
    "                        error_y='std',\n",
    "                        title='Mean Prediction Error by Sector (with Std Dev)',\n",
    "                        labels={'mean': 'Mean Error %', 'sector': 'Sector'})\n",
    "            fig.update_layout(xaxis_tickangle=-45)\n",
    "            fig.show()\n",
    "            \n",
    "            print('\\n📈 Sector Error Statistics:')\n",
    "            print(sector_errors)\n",
    "        \n",
    "        # Error by region and sector (heatmap)\n",
    "        if 'sector' in all_stocks_featured.columns and 'region' in all_stocks_featured.columns:\n",
    "            pivot_errors = all_stocks_featured.pivot_table(\n",
    "                values='prediction_error_pct',\n",
    "                index='sector',\n",
    "                columns='region',\n",
    "                aggfunc='mean'\n",
    "            )\n",
    "            \n",
    "            fig = px.imshow(pivot_errors,\n",
    "                           text_auto='.1f',\n",
    "                           aspect='auto',\n",
    "                           color_continuous_scale='Reds',\n",
    "                           title='Mean Prediction Error % by Sector and Region')\n",
    "            fig.update_layout(width=900, height=600)\n",
    "            fig.show()\n",
    "    \n",
    "    print('✓ Model evaluation visualizations complete')\n",
]

# Section 8: Enhanced Stock Valuation Visualizations
sec8_viz = [
    "# 📊 Section 8 Additional Enhanced Visualizations - Stock Valuation\n",
    "print('\\n' + '='*80)\n",
    "print('📊 ADDITIONAL INTERACTIVE VALUATION VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "if 'all_stocks_featured' in dir() and 'mispricing_score' in all_stocks_featured.columns:\n",
    "    from finance_ml.ml_workflow.analytics.eval import create_valuation_scatter_plot\n",
    "    import plotly.express as px\n",
    "    \n",
    "    print('\\n📈 Mispricing Score Analysis...')\n",
    "    \n",
    "    # Mispricing distribution by sector\n",
    "    if 'sector' in all_stocks_featured.columns:\n",
    "        fig = px.violin(all_stocks_featured, \n",
    "                       x='sector', \n",
    "                       y='mispricing_score',\n",
    "                       color='sector',\n",
    "                       box=True,\n",
    "                       title='Mispricing Score Distribution by Sector',\n",
    "                       points='outliers')\n",
    "        fig.update_layout(showlegend=False, xaxis_tickangle=-45, height=600)\n",
    "        fig.show()\n",
    "    \n",
    "    # Top undervalued opportunities\n",
    "    print('\\n🎯 Top 10 Undervalued Stocks:')\n",
    "    top_undervalued = all_stocks_featured.nlargest(10, 'mispricing_score')\n",
    "    display_cols = ['ticker', 'sector', 'last_price', 'predicted_price_target', 'mispricing_score']\n",
    "    display_cols = [c for c in display_cols if c in top_undervalued.columns]\n",
    "    print(top_undervalued[display_cols].to_string(index=False))\n",
    "    \n",
    "    # Sector-region performance matrix\n",
    "    if 'sector' in all_stocks_featured.columns and 'region' in all_stocks_featured.columns:\n",
    "        print('\\n🌍 Sector-Region Performance Matrix...')\n",
    "        pivot_mispricing = all_stocks_featured.pivot_table(\n",
    "            values='mispricing_score',\n",
    "            index='sector',\n",
    "            columns='region',\n",
    "            aggfunc='mean'\n",
    "        )\n",
    "        \n",
    "        fig = px.imshow(pivot_mispricing,\n",
    "                       text_auto='.2%',\n",
    "                       aspect='auto',\n",
    "                       color_continuous_scale='RdYlGn',\n",
    "                       title='Average Mispricing Score by Sector and Region')\n",
    "        fig.update_layout(width=900, height=600)\n",
    "        fig.show()\n",
    "    \n",
    "    print('✓ Enhanced valuation visualizations complete')\n",
]

# Section 9: Analyst Comparison Visualizations
sec9_viz = [
    "# 📊 Section 9 Enhanced Visualizations - Prediction vs Analyst Analytics\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE PREDICTION VS ANALYST VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "if 'all_stocks_featured' in dir():\n",
    "    required_cols = ['predicted_price_target', 'price_target', 'last_price']\n",
    "    if all(col in all_stocks_featured.columns for col in required_cols):\n",
    "        import plotly.express as px\n",
    "        import plotly.graph_objects as go\n",
    "        \n",
    "        print('\\n📊 Model vs Analyst Target Comparison...')\n",
    "        \n",
    "        # Scatter plot: Model vs Analyst predictions\n",
    "        fig = go.Figure()\n",
    "        \n",
    "        fig.add_trace(go.Scatter(\n",
    "            x=all_stocks_featured['price_target'],\n",
    "            y=all_stocks_featured['predicted_price_target'],\n",
    "            mode='markers',\n",
    "            marker=dict(size=8, opacity=0.6, \n",
    "                       color=all_stocks_featured.get('sector', None),\n",
    "                       colorscale='Viridis'),\n",
    "            text=all_stocks_featured.get('ticker', None),\n",
    "            name='Stocks'\n",
    "        ))\n",
    "        \n",
    "        # Perfect agreement line\n",
    "        min_val = min(all_stocks_featured['price_target'].min(), \n",
    "                     all_stocks_featured['predicted_price_target'].min())\n",
    "        max_val = max(all_stocks_featured['price_target'].max(), \n",
    "                     all_stocks_featured['predicted_price_target'].max())\n",
    "        \n",
    "        fig.add_trace(go.Scatter(\n",
    "            x=[min_val, max_val],\n",
    "            y=[min_val, max_val],\n",
    "            mode='lines',\n",
    "            line=dict(color='red', dash='dash'),\n",
    "            name='Perfect Agreement'\n",
    "        ))\n",
    "        \n",
    "        fig.update_layout(\n",
    "            title='Model Predictions vs Analyst Consensus Targets',\n",
    "            xaxis_title='Analyst Target Price',\n",
    "            yaxis_title='Model Predicted Price',\n",
    "            width=900,\n",
    "            height=700\n",
    "        )\n",
    "        fig.show()\n",
    "        \n",
    "        # Disagreement analysis\n",
    "        print('\\n🎯 Disagreement Analysis...')\n",
    "        all_stocks_featured['model_analyst_diff_pct'] = (\n",
    "            (all_stocks_featured['predicted_price_target'] - all_stocks_featured['price_target']) /\n",
    "            all_stocks_featured['price_target'] * 100\n",
    "        )\n",
    "        \n",
    "        # Histogram of disagreement\n",
    "        fig = px.histogram(all_stocks_featured, \n",
    "                          x='model_analyst_diff_pct',\n",
    "                          nbins=50,\n",
    "                          title='Distribution of Model-Analyst Disagreement',\n",
    "                          labels={'model_analyst_diff_pct': 'Difference (%)'})\n",
    "        fig.add_vline(x=0, line_dash='dash', line_color='red', annotation_text='Perfect Agreement')\n",
    "        fig.show()\n",
    "        \n",
    "        # High-conviction disagreements\n",
    "        high_disagreement = all_stocks_featured[\n",
    "            abs(all_stocks_featured['model_analyst_diff_pct']) > 10\n",
    "        ].nlargest(10, 'model_analyst_diff_pct', keep='all')\n",
    "        \n",
    "        if len(high_disagreement) > 0:\n",
    "            print(f'\\n📌 Top High-Conviction Disagreements (>10% difference):')\n",
    "            display_cols = ['ticker', 'sector', 'price_target', 'predicted_price_target', 'model_analyst_diff_pct']\n",
    "            display_cols = [c for c in display_cols if c in high_disagreement.columns]\n",
    "            print(high_disagreement[display_cols].head(10).to_string(index=False))\n",
    "        \n",
    "        print('✓ Analyst comparison visualizations complete')\n",
]

# Section 10: Portfolio Optimization Visualizations
sec10_viz = [
    "# 📊 Section 10 Enhanced Visualizations - Portfolio Optimization\n",
    "print('\\n' + '='*80)\n",
    "print('📊 INTERACTIVE PORTFOLIO OPTIMIZATION VISUALIZATIONS')\n",
    "print('='*80)\n",
    "\n",
    "if 'portfolio_results' in dir() or 'optimized_weights' in dir():\n",
    "    import plotly.graph_objects as go\n",
    "    import plotly.express as px\n",
    "    \n",
    "    print('\\n📊 Portfolio Composition Visualization...')\n",
    "    \n",
    "    # If we have optimized weights, visualize them\n",
    "    if 'optimized_weights' in dir() and isinstance(optimized_weights, dict):\n",
    "        # Portfolio composition pie chart\n",
    "        weights_df = pd.DataFrame([\n",
    "            {'Asset': k, 'Weight': v} \n",
    "            for k, v in optimized_weights.items() if v > 0.001\n",
    "        ]).sort_values('Weight', ascending=False)\n",
    "        \n",
    "        fig = px.pie(weights_df, \n",
    "                    values='Weight', \n",
    "                    names='Asset',\n",
    "                    title='Optimized Portfolio Composition',\n",
    "                    hole=0.3)\n",
    "        fig.update_traces(textposition='inside', textinfo='percent+label')\n",
    "        fig.show()\n",
    "        \n",
    "        # Top holdings bar chart\n",
    "        top_holdings = weights_df.head(10)\n",
    "        fig = px.bar(top_holdings, \n",
    "                    x='Weight', \n",
    "                    y='Asset',\n",
    "                    orientation='h',\n",
    "                    title='Top 10 Portfolio Holdings',\n",
    "                    labels={'Weight': 'Portfolio Weight', 'Asset': 'Stock'})\n",
    "        fig.update_layout(yaxis={'categoryorder': 'total ascending'})\n",
    "        fig.show()\n",
    "    \n",
    "    # Risk-Return scatter if we have portfolio metrics\n",
    "    if 'portfolio_results' in dir() and isinstance(portfolio_results, dict):\n",
    "        print('\\n📈 Risk-Return Analysis...')\n",
    "        \n",
    "        metrics_to_show = {\n",
    "            'Expected Return': portfolio_results.get('expected_return', 'N/A'),\n",
    "            'Portfolio Risk (Std)': portfolio_results.get('portfolio_risk', 'N/A'),\n",
    "            'Sharpe Ratio': portfolio_results.get('sharpe_ratio', 'N/A'),\n",
    "            'Max Drawdown': portfolio_results.get('max_drawdown', 'N/A')\n",
    "        }\n",
    "        \n",
    "        print('\\n📊 Portfolio Metrics:')\n",
    "        for metric, value in metrics_to_show.items():\n",
    "            if value != 'N/A':\n",
    "                print(f'  {metric}: {value:.4f}' if isinstance(value, (int, float)) else f'  {metric}: {value}')\n",
    "    \n",
    "    print('✓ Portfolio optimization visualizations complete')\n",
    "else:\n",
    "    print('⚠️  Portfolio results not available for visualization')\n",
]

# Find insertion points (right before the next section)
insertion_points = []

# For each section, find where it ends (before next section starts)
for sec_num in [4, 5, 6, 7, 8, 9, 10]:
    if sec_num in section_indices:
        # Find the start of the next section
        next_sec = sec_num + 1
        if next_sec in section_indices:
            insert_before = section_indices[next_sec]
        else:
            # If no next section, insert at end
            insert_before = len(cells)

        # Choose the visualization code for this section
        if sec_num == 4:
            viz_code = sec4_viz
        elif sec_num == 5:
            viz_code = sec5_viz
        elif sec_num == 6:
            viz_code = sec6_viz
        elif sec_num == 7:
            viz_code = sec7_viz
        elif sec_num == 8:
            viz_code = sec8_viz
        elif sec_num == 9:
            viz_code = sec9_viz
        elif sec_num == 10:
            viz_code = sec10_viz

        insertion_points.append((insert_before, viz_code, sec_num))

# Insert cells in reverse order to maintain indices
insertion_points.sort(reverse=True)

for insert_idx, viz_code, sec_num in insertion_points:
    print(f"Inserting visualization for Section {sec_num} at cell {insert_idx}")
    new_cell = create_code_cell(viz_code)
    cells.insert(insert_idx, new_cell)

# Save enhanced notebook
nb["cells"] = cells

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Enhanced notebook saved to: {output_path}")
print(f"✓ Added {len(insertion_points)} visualization cells")
