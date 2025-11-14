#!/usr/bin/env python3
"""
Automated notebook refactoring script for ml_finance_model_main.ipynb.

This script implements Option C (Hybrid Approach) from NOTEBOOK_REFACTORING_SUMMARY.md:
- Programmatically updates section headers to Phase 9.1-9.8 nomenclature
- Inserts phase description templates from notebook_refactoring_plan.md
- Updates Table of Contents

Usage:
    python tools/refactor_notebook_phases.py
"""
import json
import sys
from pathlib import Path


# Phase header mappings from notebook_refactoring_plan.md
PHASE_HEADERS = {
    "## 2. Loading and Preprocessing": {
        "new_header": "## Phase 9.1: Loading and Preprocessing with 6-Step Imputation Strategy",
        "description": """
### Business Goal
Load multi-region equity data and apply comprehensive preprocessing to ensure high-quality inputs for downstream modeling.

### Key Objectives
1. Load data from PostgreSQL/SQLite or CSV fallback
2. Apply 6-step imputation strategy (zero-fill, KNN, price-based, median)
3. Detect and handle outliers (IQR, z-score, isolation forest)
4. Apply sector-wise winsorization
5. Validate data quality and completeness

### Inputs
- Raw data: CSV files or database tables (US, EU, APAC, ROTW regions)

### Outputs
- `all_stocks_preprocessed`: Fully preprocessed DataFrame
- `outputs/preprocessing/`: Data quality reports, imputation stats
- `outputs/catalog/`: Data catalog metadata

### v1.2 Standards Applied
- ✅ 6-step imputation strategy
- ✅ Outlier safety rails (winsorization at [1st, 99th] percentiles)
- ✅ Data quality validation

### Validation Checkpoint
- Zero missing values after imputation
- Outliers capped within acceptable ranges
- All required columns present
""",
    },
    "## 3. Exploratory Data Analysis": {
        "new_header": "## Phase 9.2: Enhanced Exploratory Data Analysis with Statistical Testing",
        "description": """
### Business Goal
Understand data distributions, relationships, and sector-specific patterns to inform feature engineering and modeling strategies.

### Key Objectives
1. Generate comprehensive statistical summaries
2. Analyze correlations and multicollinearity
3. Perform sector-wise comparisons with statistical tests
4. Create interactive visualizations
5. Generate benchmarking reports

### Inputs
- `all_stocks_preprocessed`: Preprocessed data from Phase 9.1

### Outputs
- `outputs/eda/`: EDA summary reports, correlation matrices, sector distributions
- Interactive visualizations (Plotly HTML files)

### Key Functions Used
- `generate_eda_report()` - Phase 9.2 EDA orchestrator
- `generate_benchmarking_report()` - Sector/region comparisons
- `compare_sector_distributions()` - Statistical testing

### Validation Checkpoint
- EDA report generated successfully
- Key metrics identified for feature engineering
- Sector patterns documented
""",
    },
    "## 4. Advanced Feature Engineering": {
        "new_header": "## Phase 9.3: Advanced Feature Engineering with Sector-Specific Optimizations",
        "description": """
### Business Goal
Engineer comprehensive financial features including valuation ratios, profitability metrics, quality indicators, and sector-specific features to maximize model predictive power.

### Key Objectives
1. Engineer valuation ratios (P/E, P/B, EV/EBITDA, PEG)
2. Engineer profitability features (margins, ROE, ROA, ROIC)
3. Create momentum and technical indicators
4. Engineer analyst quality features
5. Create accounting quality scores (Altman Z, Piotroski F)
6. Build sector-relative features
7. Create interaction features

### Inputs
- `all_stocks_preprocessed`: Preprocessed data from Phase 9.1

### Outputs
- `all_stocks_features`: Data with 400+ engineered features
- `outputs/features/`: Feature importance reports, correlation analysis

### Phase 9.3 API
```python
from finance_ml.ml_workflow.features.api import build_features
all_stocks_features = build_features(
    all_stocks_preprocessed, 
    preset="comprehensive",
    include_interactions=True,
    include_relative=True
)
```

### Validation Checkpoint
- 400+ features engineered
- No infinite values (replaced with NaN)
- Feature importance calculated
- Top features identified
""",
    },
    "## 5. Multi-Class Classification": {
        "new_header": "## Phase 9.4: Multi-Class Event Classification",
        "description": """
### Business Goal
Classify stocks into financial event categories (Positive, Neutral, Negative) to provide sentiment signals that enhance regression model accuracy.

### Key Objectives
1. Create enhanced event labels using multiple methods
2. Prepare classification data with Phase 9.3 features
3. Train and optimize classifiers (XGBoost, LightGBM, CatBoost)
4. Evaluate classification performance
5. Extract classification probabilities as meta-features

### Inputs
- `all_stocks_features`: Feature-engineered data from Phase 9.3

### Outputs
- `clf_result`: Classification model result dict with probabilities
- `outputs/classification/`: Model artifacts, evaluation metrics, confusion matrices
- Event probability features for Phase 9.5

### Standardized Return Format (v1.2)
```python
clf_result = {
    'model': fitted_classifier,
    'metrics': {'accuracy': float, 'f1_macro': float, ...},
    'y_pred': np.ndarray,
    'y_proba': np.ndarray,  # 3-class probabilities
    'artifacts': {'feature_importance': pd.DataFrame, ...}
}
```

### Validation Checkpoint
- Classification accuracy > 60%
- All 3 classes represented in predictions
- Probabilities sum to 1.0
- Feature importance extracted
""",
    },
    "## 6. Sector-Optimized Regression": {
        "new_header": "## Phase 9.5: Sector-Optimized Regression Models with Quantile Predictions",
        "description": """
### Business Goal
Predict stock price targets using regression models enhanced with classification meta-features, with uncertainty quantification via quantile regression.

### Key Objectives
1. Integrate classification probabilities as meta-features
2. Train multiple regression models (XGBoost, LightGBM, CatBoost, etc.)
3. Build stacking ensemble for robust predictions
4. Train quantile models for prediction intervals (p10, p50, p90)
5. Apply non-negative constraints (prices must be ≥ 0)
6. Perform time-series cross-validation

### Inputs
- `all_stocks_features`: From Phase 9.3
- `clf_result['y_proba']`: Classification probabilities from Phase 9.4

### Outputs
- `reg_result`: Regression result dict
- `outputs/regression/`: Model artifacts, quantile predictions
- `outputs/regression/regression_predictions_detailed.csv`: Standardized predictions
- `outputs/models/regression_metrics_by_sector.csv`: Per-sector metrics

### v1.2 Standards Applied
- ✅ Quantile regression (p10, p50, p90) with conformal calibration
- ✅ Monotonicity enforcement (p10 ≤ p50 ≤ p90)
- ✅ Non-negativity constraints
- ✅ Data split policy (TimeSeriesSplit → GroupKFold → Stratified)
- ✅ Standardized predictions schema
- ✅ Outlier safety rails (Huber loss, post-prediction clipping)

### Standardized Predictions Schema
Required columns:
- ticker, isin, sector, region, last_price, snapshot_date
- y_true, y_pred, y_pred_calibrated
- pred_p10, pred_p50, pred_p90, interval_width
- abs_error, pct_error
- model_version

### Validation Checkpoint
- MAE < 50% on validation set
- R² > 0.3
- Zero predictions < 1% (non-negativity enforced)
- Quantile monotonicity verified
- Prediction intervals coverage: 80% ± 5%
""",
    },
    "## 7. Model Evaluation": {
        "new_header": "## Phase 9.6: Model Evaluation and Comprehensive Error Analysis",
        "description": """
### Business Goal
Thoroughly evaluate regression model performance with comprehensive metrics, residual analysis, and segment-wise breakdowns.

### Key Objectives
1. Calculate comprehensive regression metrics (MAE, RMSE, R², MAPE)
2. Perform segment analysis (by sector, region, market cap)
3. Generate residual plots and error distributions
4. Identify systematic biases
5. Analyze prediction errors by magnitude

### Inputs
- `reg_result`: Regression results from Phase 9.5
- `all_stocks_features`: Full dataset with predictions

### Outputs
- `outputs/evaluation/`: Comprehensive metrics, residual plots
- `outputs/evaluation/tscv_metrics.csv`: Time-series CV results
- Sector-wise performance analysis

### Key Metrics
- Overall: MAE, RMSE, R², MAPE, Median AE
- By sector: Per-sector performance comparison
- By region: Geographic performance patterns
- Error distribution: Histogram, percentiles

### Validation Checkpoint
- Comprehensive metrics calculated
- Residuals analyzed
- Sector biases identified
- Error patterns documented
""",
    },
    "## 8. Identification of Under/Overvalued": {
        "new_header": "## Phase 9.7: Stock Ranking, Analytics, and Analyst Comparison",
        "description": """
### Business Goal
Identify investment opportunities through mispricing scores, stock rankings, analyst comparison, and portfolio optimization.

### Key Objectives
1. Calculate mispricing scores: (predicted_target - last_price) / last_price
2. Rank stocks by sector and region
3. Compare predictions vs analyst targets
4. Perform portfolio optimization (max Sharpe, min volatility)
5. Calculate risk metrics (VaR, CVaR, drawdown)
6. Generate investment recommendations

### Inputs
- `all_stocks_features`: Full dataset with predictions from Phase 9.5
- Analyst price targets

### Outputs
- `outputs/analytics/`: Mispricing rankings, analyst comparison reports
- `outputs/analytics/portfolio_optimization.csv`: Optimal portfolios
- `outputs/analytics/risk_metrics.csv`: Risk analysis
- Top undervalued/overvalued stocks by sector

### Key Functions
- `calculate_mispricing_score()` - Identify mispricing
- `rank_undervalued_stocks()` - Top opportunities
- `compare_prediction_vs_analyst_targets()` - Analyst agreement analysis
- `optimize_max_sharpe_portfolio()` - Portfolio optimization
- `calculate_portfolio_risk_metrics()` - Risk quantification

### Validation Checkpoint
- Mispricing scores calculated
- Top 20 undervalued stocks identified
- Analyst comparison complete
- Portfolio optimization converged
- Risk metrics within acceptable ranges
""",
    },
}

# Phases 9.8 will consolidate sections 9 and 10
PHASE_98_HEADER = "## Phase 9.8: Comprehensive Reporting and Dashboard Data"
PHASE_98_DESCRIPTION = """
### Business Goal
Generate comprehensive reports, dashboards, and export final results for stakeholders and downstream applications.

### Key Objectives
1. Calculate financial metrics dashboard
2. Generate data quality alerts
3. Export predictions with standardized schema
4. Create interactive visualizations
5. Generate Excel/PDF reports

### Inputs
- All outputs from Phases 9.1-9.7

### Outputs
- `outputs/reporting/`: Final reports, dashboards
- `outputs/regression/regression_predictions_detailed.csv`: Final predictions export
- Excel reports with formatted tables and charts

### Key Functions
- `calculate_financial_metrics_dashboard()` - KPI reporting
- `generate_data_quality_alerts()` - Validation alerts
- `export_predictions()` - Standardized export
- `generate_prediction_analyst_excel_report()` - Excel report generation

### Validation Checkpoint
- All reports generated successfully
- Predictions exported with complete schema
- Dashboard data prepared
- Final artifacts persisted
"""


def load_notebook(path):
    """Load notebook JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(notebook, path):
    """Save notebook JSON with proper formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)


def find_cell_with_header(cells, header_prefix):
    """Find cell index containing specific header."""
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if header_prefix in source:
                return i
    return None


def update_phase_header(cells, old_header, new_header, description):
    """Update a phase header and insert description."""
    idx = find_cell_with_header(cells, old_header)
    if idx is None:
        print(f"Warning: Header '{old_header}' not found")
        return False

    # Get current cell content
    cell = cells[idx]
    source = "".join(cell.get("source", []))

    # Replace header
    new_source = source.replace(old_header, new_header, 1)

    # Insert description after header (after first line)
    lines = new_source.split("\n")
    if len(lines) > 0:
        # Insert description after header line
        lines.insert(1, description.rstrip())
        new_source = "\n".join(lines)

    # Update cell - convert to list of lines with proper newlines
    source_lines = new_source.split("\n")
    cells[idx]["source"] = [
        line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)
    ]

    print(f"✓ Updated: {old_header} → {new_header[:60]}...")
    return True


def consolidate_phases_97_98(cells):
    """Consolidate sections 8, 9, 10 into phases 9.7 and 9.8."""
    # Update section 8 to Phase 9.7
    idx8 = find_cell_with_header(cells, "## 8. Identification of Under/Overvalued")
    if idx8 is not None:
        update_phase_header(
            cells,
            "## 8. Identification of Under/Overvalued",
            PHASE_HEADERS["## 8. Identification of Under/Overvalued"]["new_header"],
            PHASE_HEADERS["## 8. Identification of Under/Overvalued"]["description"],
        )

    # Update section 9 header only (content stays under 9.7)
    idx9 = find_cell_with_header(cells, "## 9. Comprehensive Analytics")
    if idx9 is not None:
        cell = cells[idx9]
        source = "".join(cell.get("source", []))
        # Change to subsection of 9.7
        new_source = source.replace(
            "## 9. Comprehensive Analytics", "### Analyst Comparison and Advanced Analytics", 1
        )
        source_lines = new_source.split("\n")
        cells[idx9]["source"] = [
            line + "\n" if i < len(source_lines) - 1 else line
            for i, line in enumerate(source_lines)
        ]
        print("✓ Updated: Section 9 → Subsection of Phase 9.7")

    # Update section 10 to Phase 9.8
    idx10 = find_cell_with_header(cells, "## 10. Portfolio Optimization")
    if idx10 is not None:
        cell = cells[idx10]
        source = "".join(cell.get("source", []))
        new_source = source.replace("## 10. Portfolio Optimization", PHASE_98_HEADER, 1)
        lines = new_source.split("\n")
        lines.insert(1, PHASE_98_DESCRIPTION.rstrip())
        new_source = "\n".join(lines)
        source_lines = new_source.split("\n")
        cells[idx10]["source"] = [
            line + "\n" if i < len(source_lines) - 1 else line
            for i, line in enumerate(source_lines)
        ]
        print(f"✓ Updated: Section 10 → Phase 9.8")


def update_table_of_contents(cells):
    """Update Table of Contents with Phase 9.1-9.8 structure."""
    # Find ToC cell (usually in first 10 cells)
    toc_idx = None
    for i, cell in enumerate(cells[:10]):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if "Table of Contents" in source or "## Contents" in source:
                toc_idx = i
                break

    if toc_idx is None:
        print("Warning: Table of Contents not found")
        return False

    # Get current ToC
    source = "".join(cells[toc_idx].get("source", []))

    # Update section references to phase references
    replacements = [
        (
            "2. Loading and Preprocessing",
            "Phase 9.1: Loading and Preprocessing with 6-Step Imputation",
        ),
        ("3. Exploratory Data Analysis", "Phase 9.2: Enhanced Exploratory Data Analysis"),
        ("4. Advanced Feature Engineering", "Phase 9.3: Advanced Feature Engineering"),
        ("5. Multi-Class Classification", "Phase 9.4: Multi-Class Event Classification"),
        ("6. Sector-Optimized Regression", "Phase 9.5: Sector-Optimized Regression Models"),
        ("7. Model Evaluation", "Phase 9.6: Model Evaluation and Error Analysis"),
        (
            "8. Identification of Under/Overvalued",
            "Phase 9.7: Stock Ranking, Analytics, and Analyst Comparison",
        ),
        ("9. Comprehensive Analytics", ""),  # Remove, now subsection
        ("10. Portfolio Optimization", "Phase 9.8: Comprehensive Reporting and Dashboard Data"),
    ]

    new_source = source
    for old, new in replacements:
        if old in new_source:
            if new:  # Replace
                new_source = new_source.replace(old, new)
            else:  # Remove line
                lines = new_source.split("\n")
                lines = [line for line in lines if old not in line]
                new_source = "\n".join(lines)

    source_lines = new_source.split("\n")
    cells[toc_idx]["source"] = [
        line + "\n" if i < len(source_lines) - 1 else line for i, line in enumerate(source_lines)
    ]

    print("✓ Updated Table of Contents")
    return True


def main():
    """Main refactoring workflow."""
    notebook_path = Path("ml_finance_model_main.ipynb")

    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        sys.exit(1)

    print("=" * 80)
    print("NOTEBOOK REFACTORING: Phase 9.1-9.8 Architecture")
    print("=" * 80)
    print()

    # Load notebook
    print("Loading notebook...")
    notebook = load_notebook(notebook_path)
    cells = notebook.get("cells", [])
    print(f"Loaded {len(cells)} cells")
    print()

    # Update phase headers 2-7
    print("Updating phase headers 2-7...")
    for old_header, config in PHASE_HEADERS.items():
        if old_header not in ["## 8. Identification of Under/Overvalued"]:  # Handle 8-10 separately
            update_phase_header(cells, old_header, config["new_header"], config["description"])
    print()

    # Consolidate phases 9.7 and 9.8
    print("Consolidating Sections 8-10 into Phases 9.7-9.8...")
    consolidate_phases_97_98(cells)
    print()

    # Update Table of Contents
    print("Updating Table of Contents...")
    update_table_of_contents(cells)
    print()

    # Save notebook
    print("Saving refactored notebook...")
    save_notebook(notebook, notebook_path)
    print(f"✓ Saved to {notebook_path}")
    print()

    print("=" * 80)
    print("REFACTORING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Run tests: python -m unittest tests.test_notebook_refactoring -v")
    print("2. Open notebook and verify changes")
    print("3. Execute notebook end-to-end to verify functionality")


if __name__ == "__main__":
    main()
