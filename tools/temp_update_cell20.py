#!/usr/bin/env python3
"""Update Cell 20 markdown header for Phase 9.2."""

import json
from pathlib import Path

notebook_path = Path("ml_finance_model_main.ipynb")

# Read notebook
with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

# New markdown content for Cell 20
new_markdown = """## Phase 9.2: Enhanced Exploratory Data Analysis of Financial Metrics

### Business Goal
Understand data distributions, quality issues, correlations, and sector/regional patterns to inform feature engineering and validate data integrity.

### Key Objectives
1. Generate comprehensive statistical summaries and data quality reports
2. Analyze correlations and multicollinearity
3. Perform hypothesis testing across sectors and regions
4. Create interactive visualizations for distributions and relationships
5. Generate benchmarking reports comparing sector and regional performance

### Inputs
- `all_stocks_scaled`: Scaled and preprocessed data from Phase 9.1

### Outputs
**JSON Reports** (4 files):
- `eda_summary.json` - Comprehensive EDA statistics
- `data_quality_alerts.json` - Data quality issues and outliers
- `metrics_dashboard.json` - Financial KPIs by sector
- `hypothesis_tests.json` - Statistical test results (ANOVA/Kruskal-Wallis)

**Interactive Visualizations** (7 HTML files):
- `correlation_heatmap.html` - Top 30 metric correlations (clustered)
- `distributions.html` - Distribution histograms by sector
- `missing_values.html` - Data completeness heatmap
- `valuation_3d.html` - 3D scatter (Market Cap × P/E × Margin)
- `region_sector_heatmap.html` - Regional market cap distribution
- `sector_boxplots.html` - Valuation metrics by sector
- `regional_comparison.html` - Median metrics by region

### Key Functions Used
- `generate_eda_report()` - HTML EDA report orchestrator
- `calculate_financial_metrics_dashboard()` - KPI summary by sector/region
- `generate_data_quality_alerts()` - Outlier and anomaly detection
- `perform_comprehensive_hypothesis_tests()` - ANOVA/Kruskal-Wallis tests
- `generate_benchmarking_report()` - Sector/region comparisons
- `eda_summary()` - Statistical summary dictionary
- `sector_distribution_summary()` - Sector-wise distributions

### Validation Checkpoints
- [ ] All 4 JSON reports generated
- [ ] All 7 interactive visualizations created
- [ ] No critical data quality alerts
- [ ] Statistical tests identify significant sector differences (p < 0.05)
- [ ] Key correlations documented for feature engineering

### Analysis Coverage
This phase provides comprehensive statistical analysis including:
- **Distribution Analysis**: Histograms, box plots, outlier detection
- **Correlation Analysis**: Pearson correlations with clustering
- **Data Quality**: Missing values, outliers, invalid data detection
- **Hypothesis Testing**: ANOVA/Kruskal-Wallis for sector/region comparisons
- **Benchmarking**: Sector and regional performance metrics
"""

# Update Cell 20
cell20 = nb["cells"][20]
cell20["source"] = [new_markdown]

print(f"[OK] Updated Cell 20 markdown header")
print(f"  New length: {len(new_markdown)} characters")

# Write back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"[OK] Updated {notebook_path}")
