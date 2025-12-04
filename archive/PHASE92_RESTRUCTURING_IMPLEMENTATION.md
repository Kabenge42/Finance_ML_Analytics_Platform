# Phase 9.2 Enhanced EDA Restructuring - Implementation Guide

## Status: Ready for Implementation

### Completed Steps ✓

1. Added missing Phase 9.2 imports to Cell 4 (calculate_financial_metrics_dashboard, generate_data_quality_alerts,
   perform_comprehensive_hypothesis_tests)
2. Updated Cell 20 markdown header with revised objectives and outputs

### Remaining Implementation Steps

## Step 3: Create New Phase 9.2 Code Cells

### Cell 21: EDA Report + Data Quality + Metrics Dashboard

```python
# Phase 9.2 Cell 21: Comprehensive EDA Report, Data Quality, and Metrics Dashboard
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 1: EDA REPORT + DATA QUALITY + METRICS DASHBOARD")
print("=" * 80)

# Initialize output directory
eda_output_dir = Path("outputs/eda")
eda_output_dir.mkdir(parents=True, exist_ok=True)

# 1. Generate comprehensive EDA HTML report
print("\n📊 Step 1/3: Generating comprehensive EDA report...")
eda_report_path = generate_eda_report(
        all_stocks_scaled,
        output_dir=eda_output_dir,
        sector_col="sector"
        )
print(f"  ✓ EDA report complete: {eda_report_path}")

# 2. Generate data quality alerts
print("\n📊 Step 2/3: Analyzing data quality and detecting anomalies...")
quality_alerts = generate_data_quality_alerts(
        all_stocks_scaled,
        outlier_threshold=3.0
        )

# Save quality alerts to JSON
quality_alerts_path = eda_output_dir / "data_quality_alerts.json"
with open(quality_alerts_path, 'w') as f:
    json.dump(quality_alerts, f, indent=2, default=str)

print(f"  ✓ Data quality analysis complete")
print(f"  ✓ Total alerts: {len(quality_alerts)}")
print(f"  ✓ Critical: {sum(1 for a in quality_alerts if a.get('severity') == 'high')}")
print(f"  ✓ Warnings: {sum(1 for a in quality_alerts if a.get('severity') == 'medium')}")
print(f"  ✓ Output: {quality_alerts_path}")

# Print top 5 critical alerts
critical_alerts = [a for a in quality_alerts if a.get('severity') == 'high'][:5]
if critical_alerts:
    print("\n  Top 5 Critical Data Quality Issues:")
    for i, alert in enumerate(critical_alerts, 1):
        print(f"    {i}. {alert.get('message', 'N/A')}")

# 3. Calculate financial metrics dashboard
print("\n📊 Step 3/3: Calculating financial metrics dashboard...")
metrics_dashboard = calculate_financial_metrics_dashboard(
        all_stocks_scaled,
        group_by="sector"
        )

# Save metrics dashboard to JSON
metrics_dashboard_path = eda_output_dir / "metrics_dashboard.json"
with open(metrics_dashboard_path, 'w') as f:
    json.dump(metrics_dashboard, f, indent=2, default=str)

print(f"  ✓ Metrics dashboard complete")
print(f"  ✓ Categories: Valuation, Profitability, Growth, Leverage")
print(f"  ✓ Sectors analyzed: {len(metrics_dashboard.get('by_group', {}))}")
print(f"  ✓ Output: {metrics_dashboard_path}")

print("\n✅ Cell 21 Complete: Generated 3 reports")
print(f"   • EDA Report: {eda_report_path}")
print(f"   • Data Quality: {quality_alerts_path}")
print(f"   • Metrics Dashboard: {metrics_dashboard_path}")
```

### Cell 22: Statistical Hypothesis Testing (NEW)

```python
# Phase 9.2 Cell 22: Statistical Hypothesis Testing
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 2: STATISTICAL HYPOTHESIS TESTING")
print("=" * 80)

# Key metrics for hypothesis testing
test_metrics = [
    'p_e', 'p_b', 'p_s', 'ev_ebitda',  # Valuation
    'roe', 'roa', 'roic', 'net_margin', 'operating_margin',  # Profitability
    'revenue_growth', 'earnings_growth',  # Growth
    'debt_to_equity', 'current_ratio'  # Leverage & Liquidity
    ]

# Filter to available metrics
available_test_metrics = [m for m in test_metrics if m in all_stocks_scaled.columns]

print(f"\n📊 Performing hypothesis tests on {len(available_test_metrics)} metrics...")
print(f"   Tests: ANOVA (parametric) and Kruskal-Wallis (non-parametric)")
print(f"   Grouping: By sector")
print(f"   Significance level: α = 0.05")

# Perform comprehensive hypothesis tests
hypothesis_results = perform_comprehensive_hypothesis_tests(
        all_stocks_scaled,
        group_column="sector",
        metrics=available_test_metrics,
        alpha=0.05
        )

# Save results
hypothesis_test_path = eda_output_dir / "hypothesis_tests.json"
with open(hypothesis_test_path, 'w') as f:
    json.dump(hypothesis_results, f, indent=2, default=str)

print(f"\n✓ Hypothesis testing complete")
print(f"✓ Output: {hypothesis_test_path}")

# Print significant findings
print("\n📊 Significant Findings (p < 0.05):")
significant_count = 0
for metric, results in hypothesis_results.items():
    if isinstance(results, dict):
        # Check ANOVA p-value
        anova_p = results.get('anova', {}).get('p_value', 1.0)
        kruskal_p = results.get('kruskal', {}).get('p_value', 1.0)

        if anova_p < 0.05 or kruskal_p < 0.05:
            significant_count += 1
            test_used = "ANOVA" if anova_p < 0.05 else "Kruskal-Wallis"
            p_val = anova_p if anova_p < 0.05 else kruskal_p
            print(f"  • {metric}: {test_used} p={p_val:.4f} - Sectors differ significantly")

if significant_count == 0:
    print("  No significant differences detected across sectors")
else:
    print(f"\n  Total: {significant_count}/{len(available_test_metrics)} metrics show significant sector differences")

print("\n✅ Cell 22 Complete: Statistical hypothesis testing performed")
```

### Cell 23: Interactive Visualizations (Merged 29+32)

```python
# Phase 9.2 Cell 23: Interactive Visualizations - Distributions & Correlations
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 3: INTERACTIVE VISUALIZATIONS")
print("=" * 80)

print("\n📊 Creating 4 interactive visualizations...")

# Select key financial metrics for visualization
key_metrics = [
    'market_cap', 'enterprise_value', 'last_price',
    'p_e', 'p_b', 'p_s', 'ev_ebitda', 'peg_ratio',
    'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin',
    'roe', 'roa', 'roic', 'roce',
    'revenue', 'revenue_growth', 'earnings_growth', 'ebitda',
    'debt_to_equity', 'total_debt_ratio', 'current_ratio', 'quick_ratio',
    'free_cash_flow', 'operating_cash_flow',
    'dividend_yield', 'payout_ratio', 'analyst_target_price'
    ]

# Filter to available metrics
viz_metrics = [m for m in key_metrics if m in all_stocks_scaled.columns][:30]

# 1. Correlation Heatmap (top 30 metrics, clustered)
print("\n  1/4: Correlation heatmap...")
corr_matrix = all_stocks_scaled[viz_metrics].corr()

fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Phase 9.2: Financial Metrics Correlation Matrix (Top 30 Metrics)'
        )
fig_corr.update_layout(height=800, width=1000)
corr_path = eda_output_dir / "correlation_heatmap.html"
fig_corr.write_html(corr_path)
print(f"  ✓ Saved: {corr_path}")

# 2. Distribution Histograms (by sector)
print("\n  2/4: Distribution histograms...")
dist_metrics = ['p_e', 'p_b', 'net_margin', 'roe']
available_dist = [m for m in dist_metrics if m in all_stocks_scaled.columns]

fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=[m.upper().replace('_', ' ') for m in available_dist[:4]]
        )

for idx, metric in enumerate(available_dist[:4], 1):
    row = (idx - 1) // 2 + 1
    col = (idx - 1) % 2 + 1

    for sector in all_stocks_scaled['sector'].unique()[:8]:  # Limit sectors for clarity
        sector_data = all_stocks_scaled[all_stocks_scaled['sector'] == sector][metric].dropna()
        fig_dist.add_trace(
                go.Histogram(x=sector_data, name=sector, showlegend=(idx == 1)),
                row=row, col=col
                )

fig_dist.update_layout(
        height=700,
        title_text="Phase 9.2: Key Metric Distributions by Sector",
        showlegend=True
        )
dist_path = eda_output_dir / "distributions.html"
fig_dist.write_html(dist_path)
print(f"  ✓ Saved: {dist_path}")

# 3. Missing Values Heatmap
print("\n  3/4: Missing values heatmap...")
missing_pct = (all_stocks_scaled[viz_metrics].isnull().sum() / len(all_stocks_scaled) * 100).sort_values(
    ascending=False)
missing_df = pd.DataFrame({
    'Metric': missing_pct.index,
    'Missing %': missing_pct.values
    })

fig_missing = px.bar(
        missing_df,
        x='Metric',
        y='Missing %',
        title='Phase 9.2: Data Completeness Analysis (Top 30 Metrics)',
        labels={'Missing %': 'Missing Percentage (%)'},
        color='Missing %',
        color_continuous_scale='Reds'
        )
fig_missing.update_layout(height=500, xaxis_tickangle=-45)
missing_path = eda_output_dir / "missing_values.html"
fig_missing.write_html(missing_path)
print(f"  ✓ Saved: {missing_path}")

# 4. 3D Valuation Scatter
print("\n  4/4: 3D valuation scatter...")
if all(['market_cap' in all_stocks_scaled.columns, 'p_e' in all_stocks_scaled.columns,
        'gross_margin' in all_stocks_scaled.columns, 'sector' in all_stocks_scaled.columns]):

    viz_df = all_stocks_scaled[['market_cap', 'p_e', 'gross_margin', 'sector', 'ticker']].dropna()

    fig_3d = px.scatter_3d(
            viz_df,
            x='market_cap',
            y='p_e',
            z='gross_margin',
            color='sector',
            hover_data=['ticker'],
            title='Phase 9.2: 3D Valuation Analysis (Market Cap × P/E × Gross Margin)',
            labels={
                'market_cap': 'Market Cap',
                'p_e': 'P/E Ratio',
                'gross_margin': 'Gross Margin'
                },
            height=700
            )
    valuation_3d_path = eda_output_dir / "valuation_3d.html"
    fig_3d.write_html(valuation_3d_path)
    print(f"  ✓ Saved: {valuation_3d_path}")
else:
    print(f"  ⚠ Skipped: Required columns not available")

print("\n✅ Cell 23 Complete: Created 4 interactive visualizations")
```

### Cell 24: Sector & Regional Benchmarking

```python
# Phase 9.2 Cell 24: Sector & Regional Benchmarking
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 4: SECTOR & REGIONAL BENCHMARKING")
print("=" * 80)

# Select top metrics for benchmarking
benchmark_metrics = [
    'p_e', 'p_b', 'ev_ebitda',  # Valuation
    'roe', 'roa', 'net_margin',  # Profitability
    'revenue_growth', 'market_cap'  # Growth & Size
    ]
available_benchmark_metrics = [m for m in benchmark_metrics if m in all_stocks_scaled.columns]

print(f"\n📊 Step 1/2: Generating benchmarking report ({len(available_benchmark_metrics)} metrics)...")

# Generate benchmarking report
benchmark_report = generate_benchmarking_report(
        all_stocks_scaled,
        metrics=available_benchmark_metrics,
        sector_column='sector',
        region_column='region' if 'region' in all_stocks_scaled.columns else None,
        include_statistical_tests=True
        )

# Save benchmarking report
benchmark_path = eda_output_dir / "benchmarking_report.json"
with open(benchmark_path, 'w') as f:
    json.dump(benchmark_report, f, indent=2, default=str)

print(f"  ✓ Benchmarking complete: {benchmark_path}")

# Generate sector distribution summary
print("\n📊 Step 2/2: Creating sector distribution visualizations...")
sector_dist_metrics = ['market_cap', 'p_e', 'roe', 'net_margin']
available_sector_metrics = [m for m in sector_dist_metrics if m in all_stocks_scaled.columns]

sector_summaries = sector_distribution_summary(
        all_stocks_scaled,
        sector_column='sector',
        metrics=available_sector_metrics
        )

# Create sector box plots
if available_sector_metrics and 'sector' in all_stocks_scaled.columns:
    # 1. Region-Sector Heatmap
    if 'region' in all_stocks_scaled.columns and 'market_cap' in all_stocks_scaled.columns:
        print("\n  1/3: Region-sector heatmap...")
        region_sector = all_stocks_scaled.groupby(['region', 'sector'])['market_cap'].agg(
                ['mean', 'count']).reset_index()
        region_sector_pivot = region_sector.pivot(index='sector', columns='region', values='mean')

        fig_region_sector = px.imshow(
                region_sector_pivot,
                labels=dict(color="Avg Market Cap"),
                title='Phase 9.2: Average Market Cap by Region and Sector',
                aspect="auto",
                color_continuous_scale='Viridis'
                )
        region_sector_path = eda_output_dir / "region_sector_heatmap.html"
        fig_region_sector.write_html(region_sector_path)
        print(f"  ✓ Saved: {region_sector_path}")

    # 2. Sector Box Plots
    print("\n  2/3: Sector box plots...")
    fig_box = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.upper().replace('_', ' ') for m in available_sector_metrics[:4]]
            )

    for idx, metric in enumerate(available_sector_metrics[:4], 1):
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        for sector in all_stocks_scaled['sector'].unique()[:8]:
            sector_data = all_stocks_scaled[all_stocks_scaled['sector'] == sector][metric].dropna()
            fig_box.add_trace(
                    go.Box(y=sector_data, name=sector, showlegend=(idx == 1)),
                    row=row, col=col
                    )

    fig_box.update_layout(
            height=700,
            title_text="Phase 9.2: Sector Distribution Box Plots",
            showlegend=True
            )
    box_plot_path = eda_output_dir / "sector_boxplots.html"
    fig_box.write_html(box_plot_path)
    print(f"  ✓ Saved: {box_plot_path}")

    # 3. Regional Comparison Bar Charts
    if 'region' in all_stocks_scaled.columns:
        print("\n  3/3: Regional comparison bar charts...")
        regional_metrics = ['p_e', 'roe']
        available_regional = [m for m in regional_metrics if m in all_stocks_scaled.columns]

        if available_regional:
            regional_summary = all_stocks_scaled.groupby('region')[available_regional].median().reset_index()

            fig_regional = make_subplots(
                    rows=1, cols=len(available_regional),
                    subplot_titles=[m.upper().replace('_', ' ') for m in available_regional]
                    )

            for idx, metric in enumerate(available_regional, 1):
                fig_regional.add_trace(
                        go.Bar(x=regional_summary['region'], y=regional_summary[metric], name=metric),
                        row=1, col=idx
                        )

            fig_regional.update_layout(
                    height=400,
                    title_text="Phase 9.2: Regional Comparison (Median Values)",
                    showlegend=False
                    )
            regional_path = eda_output_dir / "regional_comparison.html"
            fig_regional.write_html(regional_path)
            print(f"  ✓ Saved: {regional_path}")

print("\n✅ Cell 24 Complete: Sector and regional benchmarking performed")
```

### Cell 25: EDA Summary Dashboard (NEW)

```python
# Phase 9.2 Cell 25: EDA Summary Dashboard
print("\n" + "=" * 80)
print("PHASE 9.2 CELL 5: EDA SUMMARY DASHBOARD")
print("=" * 80)

print("\n📊 Compiling Phase 9.2 Summary...")

# Collect all Phase 9.2 outputs
phase92_outputs = {
    "json_reports": [
        "eda_summary.json",
        "data_quality_alerts.json",
        "metrics_dashboard.json",
        "hypothesis_tests.json"
        ],
    "html_visualizations": [
        "correlation_heatmap.html",
        "distributions.html",
        "missing_values.html",
        "valuation_3d.html",
        "region_sector_heatmap.html",
        "sector_boxplots.html",
        "regional_comparison.html"
        ]
    }

# Count existing files
existing_json = sum(1 for f in phase92_outputs["json_reports"] if (eda_output_dir / f).exists())
existing_html = sum(1 for f in phase92_outputs["html_visualizations"] if (eda_output_dir / f).exists())

print(f"\n✅ Phase 9.2 Enhanced EDA Complete!")
print(f"\n📁 Output Directory: {eda_output_dir}")
print(f"\n📄 JSON Reports ({existing_json}/{len(phase92_outputs['json_reports'])}):")
for report in phase92_outputs["json_reports"]:
    status = "✓" if (eda_output_dir / report).exists() else "✗"
    print(f"   {status} {report}")

print(f"\n🌐 Interactive Visualizations ({existing_html}/{len(phase92_outputs['html_visualizations'])}):")
for viz in phase92_outputs["html_visualizations"]:
    status = "✓" if (eda_output_dir / viz).exists() else "✗"
    print(f"   {status} {viz}")

# Print key findings summary
print(f"\n📊 Key Findings Summary:")
print(f"   • Dataset: {all_stocks_scaled.shape[0]} stocks × {all_stocks_scaled.shape[1]} features")
print(f"   • Sectors: {all_stocks_scaled['sector'].nunique() if 'sector' in all_stocks_scaled.columns else 'N/A'}")
print(f"   • Regions: {all_stocks_scaled['region'].nunique() if 'region' in all_stocks_scaled.columns else 'N/A'}")

# Calculate data completeness
if viz_metrics:
    completeness = (1 - all_stocks_scaled[viz_metrics].isnull().sum().mean() / len(all_stocks_scaled)) * 100
    print(f"   • Data Completeness: {completeness:.1f}%")

# Top correlations
if len(viz_metrics) >= 2:
    corr_pairs = []
    for i in range(len(viz_metrics)):
        for j in range(i + 1, len(viz_metrics)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                corr_pairs.append((viz_metrics[i], viz_metrics[j], corr_val))

    if corr_pairs:
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n   Top 3 Correlated Metric Pairs:")
        for m1, m2, corr in corr_pairs[:3]:
            print(f"     • {m1} ↔ {m2}: r={corr:.3f}")

# Data quality summary
if len(quality_alerts) > 0:
    high_severity = sum(1 for a in quality_alerts if a.get('severity') == 'high')
    if high_severity > 0:
        print(f"\n   ⚠ Data Quality: {high_severity} critical issues detected")
    else:
        print(f"\n   ✓ Data Quality: No critical issues detected")

print("\n" + "=" * 80)
print("PHASE 9.2 COMPLETE - Proceed to Phase 9.3 Feature Engineering")
print("=" * 80)
```

## Step 4: Delete Old Redundant Cells

Delete the following cells after creating the new ones:

- **Cells 22-28**: Phase 9.3 category analysis (will be moved to Phase 9.3 section)
- **Cell 29**: Old interactive visualizations (merged into new Cell 23)
- **Cell 30**: Redundant markdown header
- **Cell 32**: Redundant correlation heatmap (merged into new Cell 23)

Keep **Cell 31** (eda_summary call) and **Cell 33** (sector_distribution_summary) as they provide programmatic access to
the analysis functions.

## Step 5: Move Cells 22-28 to Phase 9.3

Move the saved category analysis cells (22-28) to Phase 9.3 section after Cell 43 (Schema 1.3 summary).

## Validation

After implementation, verify:

- [ ] All 4 JSON reports generated in outputs/eda/
- [ ] All 7 HTML visualizations created
- [ ] No import errors when running cells
- [ ] Cell execution order is correct (20 → 21 → 22 → 23 → 24 → 25)
- [ ] Phase 9.3 section has the moved category analysis cells
- [ ] Total Phase 9.2 cells reduced from 14 to 6

## Notes

- The new structure eliminates redundancy while increasing analytical depth
- All Phase 9.2 functions from the library are now utilized
- Output format is standardized across all cells
- Console output is concise (<50 lines per cell)
