"""
Fix ml_finance_model_main2_0.ipynb to align with actual ETL Data Explorer output.
Target: 6,957 stocks x 585 columns (363 preprocessed + 222 engineered features)
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("ml_finance_model_main2_0.ipynb")
TARGET_STOCKS = 6957
TARGET_PREPROCESSED_COLS = 363
TARGET_FEATURES_COLS = 582
TARGET_ENHANCED_COLS = 585

print("=" * 80)
print("FIXING ML_FINANCE_MODEL_MAIN2_0.IPYNB")
print("=" * 80)

# Load notebook
print(f"\nLoading notebook: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"+ Loaded {len(nb['cells'])} cells")

# ============================================================================
# Phase 1: Update ETL Pipeline Configuration (Cell 8)
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 1: Update ETL Pipeline Configuration (Cell 8)")
print("=" * 80)

etl_cell_source = """# Phase 9.1: Consolidated ETL Pipeline with Financial Metrics (code_guidelines.md Section 8.2)
# Unified data loading using etl_with_financial_metrics() - RECOMMENDED approach
# Replaces: load_from_db/csv → normalize_columns → validate_schema → imputation → financial_metrics
# Note: Winsorization and scaling are done in downstream cells per code_guidelines.md
# Reference: code_guidelines.md Section 8.2, finance_ml/ml_workflow/preprocessing/etl.py

# IMPORTANT: ETL Data Explorer execution shows CSV as reliable source
# Database connection fails consistently - prioritize CSV fallback
DB_URL = os.getenv('DB_URL', 'postgresql+psycopg2://postgres:@localhost:5432/postgres')
FINANCIAL_METRICS_OUTPUT_DIR = Path("outputs/eda/financial_metrics")
FINANCIAL_METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Expected output from ETL Data Explorer execution:
# - Stocks: 6,957
# - Preprocessed columns: 363 (from ETL pipeline)
# - Feature engineered columns: 582 (after build_features)
# - Enhanced columns: 585 (with additional financial metrics)
# - Phase 9.3 coverage: 182/196 features (92.9%)

print("\\n" + "=" * 70)
print("ETL PIPELINE EXECUTION (Phase 9.1)")
print("=" * 70)
print(f"Target: {TARGET_STOCKS:,} stocks x {TARGET_PREPROCESSED_COLS} columns")
print(f"Expected imputation: 6-step strategy, 0 missing values")
print("=" * 70)

try:
    # Primary: Load from CSV (reliable per ETL Data Explorer)
    print("\\n📂 Loading from CSV (primary source)...")
    all_stocks_preprocessed, etl_metrics = etl_with_financial_metrics(
            source='csv',
            data_dir=Path("data"),
            compute_all_metrics=True,  # Valuation, Profitability, Growth, Leverage metrics
            output_dir=FINANCIAL_METRICS_OUTPUT_DIR,  # Quality alerts and dashboard JSON
            return_metrics=True
            )
    print(f"+ Loaded from CSV: {len(all_stocks_preprocessed):,} stocks")

except Exception as csv_error:
    print(f"\! CSV load failed: {csv_error}")
    print("\\nAttempting database fallback...")
    try:
        # Fallback: Try database
        all_stocks_preprocessed, etl_metrics = etl_with_financial_metrics(
                source='all_stocks',
                db_url=DB_URL,
                compute_all_metrics=True,
                output_dir=FINANCIAL_METRICS_OUTPUT_DIR,
                return_metrics=True
                )
        print(f"+ Loaded from database: {len(all_stocks_preprocessed):,} stocks")
    except Exception as db_error:
        print(f"X Both CSV and database loading failed!")
        print(f"   CSV error: {csv_error}")
        print(f"   DB error: {db_error}")
        raise RuntimeError("ETL pipeline failed on all data sources")

# Display comprehensive ETL metrics
print("\\n" + "=" * 70)
print("ETL PIPELINE METRICS (with Financial Metrics)")
print("=" * 70)
print(etl_metrics.summary())
print("=" * 70)

# Critical validation: Imputation completeness (code_guidelines.md Section 8.5.2)
assert etl_metrics.imputation_completeness, "X Imputation incomplete!"

print(f"\\n+ ETL Pipeline Complete (Phase 9.1 - Consolidated ETL with Financial Metrics):")
print(f"  - Final shape: {all_stocks_preprocessed.shape}")
print(f"  - Expected: ({TARGET_STOCKS:,}, {TARGET_PREPROCESSED_COLS})")

# Validate shape
actual_rows, actual_cols = all_stocks_preprocessed.shape
if actual_rows != TARGET_STOCKS:
    print(f"  \! Row count mismatch: got {actual_rows:,}, expected {TARGET_STOCKS:,}")
if actual_cols < TARGET_PREPROCESSED_COLS - 10:  # Allow small variation
    print(f"  \! Column count significantly lower: got {actual_cols}, expected ~{TARGET_PREPROCESSED_COLS}")
elif actual_cols > TARGET_PREPROCESSED_COLS + 10:
    print(f"  ℹ Column count higher than expected: got {actual_cols}, expected ~{TARGET_PREPROCESSED_COLS}")

# Display financial metrics status
print(f"  - Date columns ready: +" if etl_metrics.date_columns_ready else "  - Date columns ready: ✗")
print(f"  - Processing time: {etl_metrics.total_duration:.2f}s")
print(f"  - Quality score: {etl_metrics.quality_score:.3f}")
print(f"  - Validation score: {etl_metrics.validation_score:.3f}")
print(f"  - Imputation: {etl_metrics.imputation_strategy} strategy")
print(f"  - Missing before: {etl_metrics.missing_values_before_imputation:,}")
print(f"  - Missing after: {etl_metrics.missing_values_after_imputation:,}")

# Validate zero missing values
current_missing = all_stocks_preprocessed.isnull().sum().sum()
if current_missing > 0:
    print(f"  \! WARNING: {current_missing:,} missing values detected post-ETL!")
else:
    print(f"  + Zero missing values confirmed")

print(f"\\n📝 Notes:")
print(f"  - ETL output is BEFORE feature engineering (Phase 9.3)")
print(f"  - Feature engineering will add ~220 columns in next stage")
print(f"  - Financial metrics (valuation, profitability, etc.) included in ETL")
print(f"  - This separation allows proper preprocessing order per code_guidelines.md")

# Stage naming per code_guidelines.md Section 8.2 (4-stage pipeline)
# Stage 1: all_stocks_preprocessed - ETL output with financial metrics
all_stocks_normalized = all_stocks_preprocessed.copy()  # Alias for backward compatibility
print(f"\\n+ Stage 1 (preprocessed): {all_stocks_preprocessed.shape}")
"""

nb["cells"][8]["source"] = etl_cell_source.split("\n")
print("+ Updated Cell 8: ETL Pipeline Configuration")
print(f"  - CSV as primary source")
print(f"  - Database as fallback")
print(f"  - Added shape validation: {TARGET_STOCKS} x {TARGET_PREPROCESSED_COLS}")
print(f"  - Added missing value checks")

# ============================================================================
# Phase 2: Update Financial Metrics Validation (Cell 9)
# ============================================================================
print("\\n" + "=" * 80)
print("PHASE 2: Update Financial Metrics Validation (Cell 9)")
print("=" * 80)

validation_cell_source = """# Phase 9.1.1: Financial Metrics ETL Verification
# Validate that etl_with_financial_metrics() properly computed all metrics
# Expected metrics based on ETL Data Explorer execution:
# - Valuation: 4 metrics
# - Profitability: 5 metrics
# - Growth: 3 metrics
# - Leverage: 2 metrics
# - Target vs Price: 2 metrics
# Total: 16 core financial metrics

print("\\n" + "=" * 70)
print("FINANCIAL METRICS VERIFICATION")
print("=" * 70)

# Define expected metrics by category (from ETL Data Explorer)
EXPECTED_VALUATION_METRICS = ['p_e_ratio', 'p_s_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio']
EXPECTED_PROFITABILITY_METRICS = ['roe', 'roa', 'gross_margin_pct', 'operating_margin_pct', 'net_margin_pct']
EXPECTED_GROWTH_METRICS = ['revenue_growth', 'ebitda_growth', 'earnings_growth']
EXPECTED_LEVERAGE_METRICS = ['debt_to_equity', 'debt_to_assets']
EXPECTED_TARGET_METRICS = ['target_vs_price', 'target_vs_price_median']

# Define critical price columns that must be preserved (Section 8.5.3)
CRITICAL_PRICE_COLS = [
    'last_price', 'price_target', 'price_target_median', 'price_target_low', 'price_target_high',
    'price_target_ytd_ago', '52w_high_adj', '52w_low_adj',
    'ema_20d', 'ema_50d', 'ema_100d', 'ema_250d',
    'price_1m_ago', 'price_3m_ago', 'price_6m_ago', 'price_1y_ago',
    'price_5d_ago', 'price_1w_ago', 'price_qtd_ago',
    'price_3y_ago', 'price_5y_ago'
]

# Check available metrics in preprocessed data
available_cols = set(all_stocks_preprocessed.columns)

print("\\nFinancial Metrics Coverage:")
print("-" * 50)

# Valuation metrics
val_present = [m for m in EXPECTED_VALUATION_METRICS if m in available_cols]
print(f"Valuation metrics:     {len(val_present)}/{len(EXPECTED_VALUATION_METRICS)}")
for m in val_present:
    non_null = all_stocks_preprocessed[m].notna().sum()
    print(f"  + {m}: {non_null:,} non-null")

# Profitability metrics
prof_present = [m for m in EXPECTED_PROFITABILITY_METRICS if m in available_cols]
print(f"\\nProfitability metrics: {len(prof_present)}/{len(EXPECTED_PROFITABILITY_METRICS)}")
for m in prof_present:
    non_null = all_stocks_preprocessed[m].notna().sum()
    print(f"  + {m}: {non_null:,} non-null")

# Growth metrics
growth_present = [m for m in EXPECTED_GROWTH_METRICS if m in available_cols]
print(f"\\nGrowth metrics:        {len(growth_present)}/{len(EXPECTED_GROWTH_METRICS)}")
for m in growth_present:
    non_null = all_stocks_preprocessed[m].notna().sum()
    print(f"  + {m}: {non_null:,} non-null")

# Leverage metrics
lev_present = [m for m in EXPECTED_LEVERAGE_METRICS if m in available_cols]
print(f"\\nLeverage metrics:      {len(lev_present)}/{len(EXPECTED_LEVERAGE_METRICS)}")
for m in lev_present:
    non_null = all_stocks_preprocessed[m].notna().sum()
    print(f"  + {m}: {non_null:,} non-null")

# Target vs Price metrics
target_present = [m for m in EXPECTED_TARGET_METRICS if m in available_cols]
print(f"\\nTarget vs Price:       {len(target_present)}/{len(EXPECTED_TARGET_METRICS)}")
for m in target_present:
    non_null = all_stocks_preprocessed[m].notna().sum()
    print(f"  + {m}: {non_null:,} non-null")

# Summary
total_expected = (len(EXPECTED_VALUATION_METRICS) + len(EXPECTED_PROFITABILITY_METRICS) +
                  len(EXPECTED_GROWTH_METRICS) + len(EXPECTED_LEVERAGE_METRICS) + len(EXPECTED_TARGET_METRICS))
total_present = len(val_present) + len(prof_present) + len(growth_present) + len(lev_present) + len(target_present)

print(f"\\n" + "=" * 50)
print(f"Total Financial Metrics: {total_present}/{total_expected} ({total_present/total_expected*100:.1f}%)")
print("=" * 50)

# Verify critical price columns are preserved
price_cols_present = [c for c in CRITICAL_PRICE_COLS if c in available_cols]
print(f"\\nCritical Price Columns Preserved: {len(price_cols_present)}/{len(CRITICAL_PRICE_COLS)}")
for col in price_cols_present[:10]:  # Show first 10
    non_null = all_stocks_preprocessed[col].notna().sum()
    print(f"    {col}: {non_null:,} non-null values")
if len(price_cols_present) > 10:
    print(f"    ... and {len(price_cols_present) - 10} more")

# Verify output files exist
financial_metrics_files = [
    'valuation_opportunities.json',
    'multi_dimensional_valuation_analysis.json',
    'financial_metrics_dashboard.json'
]
print(f"\\nOutput Files:")
for fname in financial_metrics_files:
    fpath = FINANCIAL_METRICS_OUTPUT_DIR / fname
    if fpath.exists():
        print(f"  - {fname}")
    else:
        print(f"  ! {fname} (not found)")

print(f"\\n+ Financial Metrics ETL Verification Complete")
"""

nb["cells"][9]["source"] = validation_cell_source.split("\n")
print("+ Updated Cell 9: Financial Metrics Validation")
print(f"  - Validates 16 core financial metrics")
print(f"  - Checks 21 critical price columns")
print(f"  - Verifies output file generation")

# ============================================================================
# Save updated notebook
# ============================================================================
print("\\n" + "=" * 80)
print("SAVING UPDATED NOTEBOOK")
print("=" * 80)

# Create backup
backup_path = NOTEBOOK_PATH.parent / f"{NOTEBOOK_PATH.stem}_backup.ipynb"
import shutil

shutil.copy2(NOTEBOOK_PATH, backup_path)
print(f"+ Backup created: {backup_path}")

# Save updated notebook
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"+ Updated notebook saved: {NOTEBOOK_PATH}")

print("\\n" + "=" * 80)
print("SUMMARY OF CHANGES")
print("=" * 80)
print("+ Cell 8: ETL Pipeline Configuration")
print("  - CSV prioritized as primary source")
print("  - Database as fallback only")
print("  - Shape validation: 6,957 x 363")
print("  - Missing value assertions added")
print("\\n+ Cell 9: Financial Metrics Validation")
print("  - Validates 16 financial metrics")
print("  - Checks 21 price columns preserved")
print("  - Verifies JSON output files")
print("\\n+ Backup: ml_finance_model_main2_0_backup.ipynb")
print("=" * 80)
print("\\nNOTEBOOK FIX COMPLETE!")
print("=" * 80)
