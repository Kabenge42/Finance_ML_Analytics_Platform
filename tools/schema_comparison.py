"""
Schema Comparison Tool

Compares SQL schema columns, CSV headers, and Python COLUMN_SCHEMA
to identify gaps and misalignments.
"""

import sys

sys.path.insert(0, ".")

from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA, normalize_column_name


def main():
    # Read CSV header
    with open("data/screening_us.csv", "r", encoding="utf-8") as f:
        header = f.readline().strip()
    csv_columns = header.split(",")

    print("=" * 80)
    print("SCHEMA COMPARISON ANALYSIS")
    print("=" * 80)
    print(f"\nTotal CSV columns: {len(csv_columns)}")
    print(f"Total COLUMN_SCHEMA entries: {len(COLUMN_SCHEMA)}")
    print()

    # Apply normalization to each CSV column and check against COLUMN_SCHEMA
    missing_from_schema = []
    found_in_schema = []
    normalization_examples = []

    for col in csv_columns:
        norm = normalize_column_name(col)
        normalization_examples.append((col, norm))
        if norm in COLUMN_SCHEMA:
            found_in_schema.append((col, norm, COLUMN_SCHEMA[norm]))
        else:
            missing_from_schema.append((col, norm))

    print("=" * 80)
    print("NORMALIZATION EXAMPLES (showing special character handling)")
    print("=" * 80)

    # Show examples with special characters
    special_chars = ["#", "%", "&", "/", "-", "(", ")"]
    special_examples = [
        (orig, norm)
        for orig, norm in normalization_examples
        if any(c in orig for c in special_chars)
    ][:30]

    for orig, norm in special_examples:
        print(f"  {orig:55} -> {norm}")

    print()
    print("=" * 80)
    print(f"COLUMNS FOUND IN SCHEMA: {len(found_in_schema)}/{len(csv_columns)}")
    print("=" * 80)

    print()
    print("=" * 80)
    print(f"COLUMNS MISSING FROM SCHEMA: {len(missing_from_schema)}")
    print("=" * 80)

    if missing_from_schema:
        for orig, norm in missing_from_schema:
            print(f"  CSV: {orig:50} -> Normalized: {norm}")
    else:
        print("  None - All CSV columns have corresponding COLUMN_SCHEMA entries!")

    # Check for extra entries in COLUMN_SCHEMA not from CSV
    csv_normalized = set(normalize_column_name(col) for col in csv_columns)
    schema_keys = set(COLUMN_SCHEMA.keys())

    extra_in_schema = schema_keys - csv_normalized

    print()
    print("=" * 80)
    print(f"EXTRA ENTRIES IN COLUMN_SCHEMA (not in CSV): {len(extra_in_schema)}")
    print("=" * 80)

    # Categorize extra entries
    derived_cols = [
        k
        for k in extra_in_schema
        if k.startswith("log_")
        or "_previous_year" in k
        or k
        in [
            "p_e_ratio",
            "p_s_ratio",
            "ev_ebitda_ratio",
            "ev_sales_ratio",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "roe",
            "roa",
            "revenue_growth",
            "ebitda_growth",
            "earnings_growth",
            "debt_to_equity",
            "debt_to_assets",
            "target_vs_price",
            "target_vs_price_median",
            "peg_ratio",
            "dividend_yield",
            "roic",
        ]
    ]

    alias_cols = [k for k in extra_in_schema if COLUMN_SCHEMA.get(k, {}).get("role") == "auxiliary"]

    generic_cols = [
        k
        for k in extra_in_schema
        if k
        in [
            "p_e",
            "p_b",
            "revenue",
            "ebitda",
            "ebit",
            "net_income",
            "gross_margin",
            "eps",
            "total_equity",
            "total_assets",
            "total_debt",
            "inventory",
            "capex",
            "cash_and_equivalents",
            "current_assets",
            "current_liabilities",
            "working_capital",
            "retained_earnings",
            "cfo",
            "cfi",
            "cff",
            "fcf",
            "gross_profit",
            "operating_income",
            "interest_expense",
            "goodwill",
            "dividend_per_share",
            "operating_expenses",
            "operating_cash_flow",
            "dividends_paid",
            "dividends_paid_ltm",
            "r_d_expenses",
            "intangible_assets",
            "marketing_expenses",
            "employees",
            "net_income_ltm",
        ]
    ]

    conditional_cols = [
        k
        for k in extra_in_schema
        if "_applicable" in k
        or "cash_burn" in k
        or "per_employee" in k
        or "employee_growth" in k
        or "workforce_" in k
        or "hiring_intensity" in k
    ]

    other_extra = (
        extra_in_schema
        - set(derived_cols)
        - set(alias_cols)
        - set(generic_cols)
        - set(conditional_cols)
    )

    print(f"\n  Derived/Computed columns (log_, ratios, etc.): {len(derived_cols)}")
    print(f"  Legacy aliases (role=auxiliary): {len(alias_cols)}")
    print(f"  Generic base columns (no time suffix): {len(generic_cols)}")
    print(f"  Conditional metrics (with _applicable): {len(conditional_cols)}")
    print(f"  Other extra: {len(other_extra)}")

    if other_extra:
        print("\n  Other extra columns:")
        for k in sorted(other_extra):
            meta = COLUMN_SCHEMA.get(k, {})
            print(f"    {k}: dtype={meta.get('dtype')}, role={meta.get('role')}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  CSV columns: {len(csv_columns)}")
    print(f"  COLUMN_SCHEMA entries: {len(COLUMN_SCHEMA)}")
    print(f"  CSV columns found in schema: {len(found_in_schema)}")
    print(f"  CSV columns missing from schema: {len(missing_from_schema)}")
    print(f"  Extra schema entries (derived/aliases/computed): {len(extra_in_schema)}")

    if len(missing_from_schema) == 0:
        print("\n  ✓ ALL CSV COLUMNS ARE PROPERLY MAPPED IN COLUMN_SCHEMA")
    else:
        print(f"\n  ✗ {len(missing_from_schema)} CSV columns need to be added to COLUMN_SCHEMA")

    print()


if __name__ == "__main__":
    main()
