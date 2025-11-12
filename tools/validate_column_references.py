"""
Validation script to check column references in advanced.py and labels.py.

This script:
1. Loads available columns from preprocessed_stocks_metadata.json
2. Extracts all column references from advanced.py and labels.py
3. Identifies mismatches and missing columns
4. Reports findings with line numbers for easy fixing

Usage:
    python validate_column_references.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent

# File paths
METADATA_FILE = PROJECT_ROOT / "outputs" / "catalog" / "preprocessed_stocks_metadata.json"
ADVANCED_PY = PROJECT_ROOT / "finance_ml" / "ml_workflow" / "features" / "advanced.py"
LABELS_PY = PROJECT_ROOT / "finance_ml" / "ml_workflow" / "classification" / "labels.py"
DATA_PY = PROJECT_ROOT / "finance_ml" / "ml_workflow" / "data.py"


def load_available_columns() -> Set[str]:
    """Load available columns from preprocessed_stocks_metadata.json."""
    if not METADATA_FILE.exists():
        print(f"⚠️  Warning: {METADATA_FILE} not found")
        return set()
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    columns = set(metadata.get("columns", []))
    print(f"✓ Loaded {len(columns)} columns from preprocessed_stocks_metadata.json")
    return columns


def extract_column_aliases_from_data_py() -> Dict[str, str]:
    """Extract column aliases from data.py normalize_columns() function."""
    # Sync with data.py column_aliases dict (lines 353-421)
    aliases = {
        # Existing aliases
        "p_e": "p_e_ltm",
        "revenue": "total_revenues_ltm",
        "ebitda": "ebitda_ltm",
        "net_income": "net_income_is_ltm",
        "p_b": "p_b_ltm",
        "gross_margin": "gross_profit_margin_pct_ltm",
        
        # Phase 9.3 Enhancement: Additional commonly used aliases
        "eps": "eps_adj_ltm",
        "total_equity": "total_equity_ltm",
        "total_assets": "total_assets_ltm",
        "total_debt": "total_debt_ltm",
        "inventory": "inventory_ltm",
        "capex": "capital_expenditure_ltm",
        "cash_and_equivalents": "cash_and_equivalents_ltm",
        "current_assets": "total_current_assets_ltm",
        "current_liabilities": "total_current_liabilities_ltm",
        "working_capital": "working_capital_ltm",
        "retained_earnings": "retained_earnings_ltm",
        "cfo": "cfo_ltm",
        "cfi": "cfi_ltm",
        "cff": "cff_ltm",
        "fcf": "fcf_ltm",
        "ebit": "ebit_ltm",
        "gross_profit": "gross_profit_ltm",
        "operating_income": "operating_income_ltm",
        "interest_expense": "interest_expense_total_ltm",
        "r_d_expenses": "r_d_expenses_ltm",
        "goodwill": "goodwill_ltm",
        "intangible_assets": "gross_intangible_assets_ltm",
        "dividend_per_share": "dividend_per_share_ltm",
        "employees": "avg_employees_ltm",
        "shares_outstanding": "shrs_out",
        "operating_expenses": "total_operating_expenses_ltm",
        "operating_cash_flow": "cfo_ltm",
        "price_target_number": "price_target_count",
        "net_income_ltm": "net_income_is_ltm",
        "volatility_1y_pct": "volatility_1y",
        
        # Previous year columns (for YoY growth calculations)
        "revenue_previous_year": "total_revenues_1fy",
        "eps_previous_year": "eps_adj_1fy",
        "ebitda_previous_year": "ebitda_1fy",
        "total_equity_previous_year": "total_equity_fy",
        "total_assets_previous_year": "total_assets_fy",
        "gross_profit_previous_year": "gross_profit_fy",
        "revenue_fy": "total_revenues_fy",
        "working_capital_1fy": "working_capital_fy",
        "roa_previous_year": "return_on_assets_roa_pct_fy",
        "current_ratio_previous_year": "current_ratio_fy",
        "gross_margin_pct_previous_year": "gross_profit_margin_pct_fy",
        "asset_turnover_previous_year": "asset_turnover_fy",
    }
    
    print(f"✓ Loaded {len(aliases)} column aliases from data.py")
    return aliases


def extract_column_references(file_path: Path) -> List[Tuple[str, int, str]]:
    """
    Extract column references from a Python file.
    
    Returns:
        List of (column_name, line_number, context_line)
    """
    if not file_path.exists():
        print(f"⚠️  Warning: {file_path} not found")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    references = []
    
    # Patterns to match column references
    patterns = [
        r'df\["([a-z_0-9]+)"\]',  # df["column_name"]
        r"df\['([a-z_0-9]+)'\]",  # df['column_name']
        r'in df\.columns',  # "column" in df.columns (need to look at previous tokens)
        r'"([a-z_0-9]+)" in df',  # "column" in df.columns
        r"'([a-z_0-9]+)' in df",  # 'column' in df.columns
    ]
    
    for line_num, line in enumerate(lines, start=1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        
        for pattern in patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                # Filter out obvious non-column references
                if match in ['str', 'int', 'float', 'bool', 'object', 'datetime64', 'category']:
                    continue
                if match.startswith('_'):  # Internal functions
                    continue
                
                references.append((match, line_num, line.strip()))
    
    return references


def validate_column_references(
    file_path: Path,
    available_columns: Set[str],
    aliases: Dict[str, str]
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """
    Validate column references in a file.
    
    Returns:
        (valid_references, invalid_references)
    """
    references = extract_column_references(file_path)
    
    valid = []
    invalid = []
    
    # Expanded available columns including aliases
    expanded_available = available_columns.copy()
    expanded_available.update(aliases.keys())
    expanded_available.update(aliases.values())
    
    # Add engineered feature patterns that are dynamically created
    engineered_patterns = [
        # These are created by the feature engineering functions themselves
        'roe', 'roa', 'roic',  # Profitability ratios
        'debt_to_equity', 'debt_to_assets', 'net_debt_to_ebitda',  # Leverage
        'current_ratio', 'quick_ratio', 'cash_ratio',  # Liquidity
        'asset_turnover', 'inventory_turnover',  # Efficiency
        'revenue_growth', 'earnings_growth', 'ebitda_growth',  # Growth
        'piotroski_f_score', 'altman_z_score', 'beneish_m_score',  # Composite scores
        'price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m',  # Momentum
        'rsi_14d', 'rsi_30d',  # RSI
        'gross_margin_pct', 'operating_margin_pct', 'net_margin_pct',  # Margins
        'ev_ebitda_ratio', 'peg_ratio', 'p_s_ratio', 'p_e_ratio', 'p_b_ratio',  # Valuation
        'upside_potential', 'analyst_bullish_pct',  # Analyst
        'distress_risk_score', 'accounting_quality_score',  # Quality
        'book_value_per_share',  # Calculated: total_equity / shares_outstanding
        'earnings_growth_pct',  # Calculated growth percentage
        'net_debt',  # Calculated: total_debt - cash_and_equivalents
        'revenue_growth_yoy',  # Year-over-year revenue growth
        'ebitda_adjustment_ratio',  # EBITDA adjustment quality metric
        'ebit_adjustment_ratio',  # EBIT adjustment quality metric
        'return_stability_score',  # Return consistency metric
        'debt_to_equity_previous_year',  # Calculated from previous year data
        'analyst_rating_change',  # Change in analyst ratings
        'in df.columns',  # Pattern matching artifact (ignore)
    ]
    expanded_available.update(engineered_patterns)
    
    # Add common variations and conditionally-checked columns
    common_variations = [
        'shares_outstanding', 'shrs_out',  # Known alias
        'cogs', 'cost_of_goods_sold',
        'accounts_receivable', 'receivables',
        'total_liabilities',
        'working_capital',
        'retained_earnings',
        # Columns that are conditionally checked (may not exist in all datasets)
        'sga_expenses',  # SG&A expenses (not always available)
        'marketing_expenses',  # Marketing expenses (not always available)
        'depreciation_amortization',  # Depreciation & amortization (calculated if available)
        'depreciation_amortization_ltm',  # Depreciation & amortization LTM
        'dividends_paid',  # Dividends paid (calculated from cash flow)
        'dividends_paid_ltm',  # Dividends paid LTM
        'share_repurchases_ltm',  # Share repurchases LTM (buybacks)
        'shares_outstanding_previous_year',  # Previous year shares outstanding
        'accounts_receivable_previous_year',  # Previous year receivables (for Beneish M-Score)
    ]
    expanded_available.update(common_variations)
    
    seen = set()
    for col, line_num, context in references:
        if col in seen:
            continue  # Skip duplicates
        seen.add(col)
        
        if col in expanded_available:
            valid.append((col, line_num, context))
        else:
            invalid.append((col, line_num, context))
    
    return valid, invalid


def main():
    """Main validation routine."""
    print("=" * 80)
    print("COLUMN REFERENCE VALIDATION")
    print("=" * 80)
    
    # Load available columns
    available_columns = load_available_columns()
    aliases = extract_column_aliases_from_data_py()
    
    # Validate advanced.py
    print("\n" + "=" * 80)
    print(f"VALIDATING: {ADVANCED_PY}")
    print("=" * 80)
    
    valid_adv, invalid_adv = validate_column_references(ADVANCED_PY, available_columns, aliases)
    
    print(f"\n✓ Valid column references: {len(valid_adv)}")
    if invalid_adv:
        print(f"❌ Invalid column references: {len(invalid_adv)}")
        print("\nInvalid References:")
        for col, line_num, context in sorted(invalid_adv, key=lambda x: x[1]):
            print(f"  Line {line_num:4d}: '{col}'")
            print(f"             {context[:100]}")
    else:
        print("✓ No invalid column references found!")
    
    # Validate labels.py
    print("\n" + "=" * 80)
    print(f"VALIDATING: {LABELS_PY}")
    print("=" * 80)
    
    valid_lbl, invalid_lbl = validate_column_references(LABELS_PY, available_columns, aliases)
    
    print(f"\n✓ Valid column references: {len(valid_lbl)}")
    if invalid_lbl:
        print(f"❌ Invalid column references: {len(invalid_lbl)}")
        print("\nInvalid References:")
        for col, line_num, context in sorted(invalid_lbl, key=lambda x: x[1]):
            print(f"  Line {line_num:4d}: '{col}'")
            print(f"             {context[:100]}")
    else:
        print("✓ No invalid column references found!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_invalid = len(invalid_adv) + len(invalid_lbl)
    
    if total_invalid == 0:
        print("\n🎉 SUCCESS! All column references are valid!")
        return 0
    else:
        print(f"\n⚠️  Found {total_invalid} invalid column references that need attention:")
        print(f"   - advanced.py: {len(invalid_adv)} issues")
        print(f"   - labels.py: {len(invalid_lbl)} issues")
        print("\nRecommended actions:")
        print("1. Check if these columns should be in preprocessed_stocks_metadata.json")
        print("2. Verify column names match data.py normalize_columns() mapping")
        print("3. Update feature engineering functions to use correct column names")
        print("4. Add missing columns to the validation whitelist if they are engineered features")
        return 1


if __name__ == "__main__":
    sys.exit(main())
