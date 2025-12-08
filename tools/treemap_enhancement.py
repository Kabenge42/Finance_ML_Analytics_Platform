# Enhanced Treemap code with top 5 metrics and average values
# Copy the treemap section below and replace the existing treemap code in Cell 9

treemap_code = """
# Treemap visualization with top 5 metrics average values
treemap_data = coverage_df[coverage_df['Present'] > 0].copy()
if not treemap_data.empty:
    # Calculate top 5 metrics with average values for each category
    category_metrics_summary = []
    for category in treemap_data['Category']:
        # Get features for this category
        category_features = PHASE93_FEATURE_CATEGORIES.get(category, [])
        available_features = [f for f in category_features if f in all_stocks_features.columns]

        if available_features:
            # Get top 5 features with highest data availability
            feature_availability = {f: all_stocks_features[f].notna().sum() for f in available_features}
            top_5_features = sorted(feature_availability.items(), key=lambda x: x[1], reverse=True)[:5]

            # Calculate average values for top 5 features
            metrics_text = []
            for feat, count in top_5_features:
                avg_val = all_stocks_features[feat].mean()
                # Format the metric name and value
                feat_name = feat.replace('_', ' ').title()[:25]  # Truncate long names
                metrics_text.append(f"{feat_name}: {avg_val:.2f}")

            category_metrics_summary.append('<br>'.join(metrics_text))
        else:
            category_metrics_summary.append('No metrics available')

    # Add metrics summary to treemap data
    treemap_data['Top5Metrics'] = category_metrics_summary

    fig_treemap = px.treemap(
            treemap_data,
            path=['Category'],
            values='Present',
            title='Phase 9.3 Feature Coverage by Category<br><sub>Branch values show top 5 metrics with averages</sub>',
            template=PLOTLY_TEMPLATE,
            color='Coverage %',
            color_continuous_scale='Viridis',
            hover_data=['Description', 'Total', 'Top5Metrics']
            )
    fig_treemap.update_traces(
            textinfo='label+value',
            textposition='middle center',
            hovertemplate='<b>%{label}</b><br>Present: %{value}<br>Coverage: %{color:.1f}%<br><br><b>Top 5 Metrics (Avg):</b><br>%{customdata[2]}<extra></extra>'
            )
    fig_treemap.update_layout(
            height=800,  # Increased height for better visibility
            font=dict(size=11)
            )
    fig_treemap.show()
"""

print("=" * 80)
print("ENHANCED TREEMAP CODE")
print("=" * 80)
print("\nReplace the existing treemap section in Cell 9 with this code:")
print("=" * 80)
print(treemap_code)
print("=" * 80)
print("\nKey Enhancements:")
print("1. Calculates top 5 metrics per category by data availability")
print("2. Shows average values for each metric")
print("3. Displays metrics in hover tooltip")
print("4. Increased chart height to 800px for better readability")
print("5. Formats metric names by replacing underscores and title casing")
print("\n✓ Code ready to copy into Cell 9 of etl_data_explorer.ipynb")
