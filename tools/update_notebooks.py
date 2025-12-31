import json
import re
from pathlib import Path

notebooks = [
    "etl_data_explorer.ipynb",
    "stock_price_target_prediction.ipynb",
    "stock_analytics.ipynb",
    "portfolio_optimization_risk_management.ipynb",
    "ml_finance_model_main.ipynb",
]

replacements = {
    "merger_restructuring_charges_ltm": "merger_and_restructuring_charges_ltm",
    "merger_restructuring_charges_fq": "merger_and_restructuring_charges_fq",
    "merger_restructuring_charges_fy": "merger_and_restructuring_charges_fy",
    "merger_restructuring_charges_5yavgfq": "merger_and_restructuring_charges_5yavgfq",
    "r_d_expenses_ltm": "randd_expenses_ltm",
    "selling_general_admin_expenses_total_fq": "selling_general_and_admin_expenses_total_fq",
    "selling_general_admin_expenses_total_fy": "selling_general_and_admin_expenses_total_fy",
    "selling_general_admin_expenses_total_1fy": "selling_general_and_admin_expenses_total_1fy",
    "selling_general_admin_expenses_total_5yavgfq": "selling_general_and_admin_expenses_total_5yavgfq",
    "strong_sell_ratings": "num_strong_sell_ratings",
    "strong_buys_ratings": "num_strong_buys_ratings",
    "hold_ratings": "num_hold_ratings",
    "buys_ratings": "num_buys_ratings",
    "sell_ratings": "num_sell_ratings",
    "price_target_number": "price_target_count",
    "sga_expenses_fq": "selling_general_and_admin_expenses_total_fq",
    "sga_expenses_fy": "selling_general_and_admin_expenses_total_fy",
    "sga_expenses_1fy": "selling_general_and_admin_expenses_total_1fy",
    "sga_expenses_5yavgfq": "selling_general_and_admin_expenses_total_5yavgfq",
    "accounts_receivable_fy": "accounts_receivable_total_fy",
    "accounts_receivable_1fy": "accounts_receivable_total_1fy",
    "accounts_receivable_5yavgfq": "accounts_receivable_total_5yavgfq",
    # sga_expenses without suffix? Mapped to fy in legacy, but maybe just replace with fy variant or keep distinct if it was exact match?
    # Schema had "sga_expenses" -> "auxiliary". data.py mapped it to "sga_expenses_fy".
    # I should likely replace "sga_expenses" with "selling_general_and_admin_expenses_total_fy".
    "sga_expenses": "selling_general_and_admin_expenses_total_fy",
    # accounts_receivable without suffix -> total_fy
    "accounts_receivable": "accounts_receivable_total_fy",
}


def update_notebook(path):
    print(f"Processing {path}...")
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        return

    try:
        with open(p, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return

    changes = 0

    def process_text(text):
        nonlocal changes
        new_text = text
        for old, new in replacements.items():
            # Use word boundary to avoid replacing partial substrings if keys overlap (e.g. sga_expenses vs sga_expenses_fq)
            # But regex replacements on source code strings can be tricky.
            # However, these are specific variable names.
            # I'll use simple string replacement but ordered by length (longest first) to avoid partial replacement issues.
            pass

        # Sort keys by length descending
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)

        for old in sorted_keys:
            new = replacements[old]
            # Simple replace might be safer than regex if we assume they are identifiers.
            # But if "sga_expenses" is inside "sga_expenses_fq", we must process longest first.
            if old in new_text:
                # Check if it's already part of the new name? No, legacy names shouldn't be part of canonical names usually.
                # Canonical: selling_general_and_admin... vs sga_expenses... completely different.
                # But "merger_restructuring_charges_ltm" vs "merger_and_restructuring_charges_ltm".
                # If I replace, I count changes.
                if old in new_text:
                    new_text = new_text.replace(old, new)
                    changes += 1
        return new_text

    for cell in nb.get("cells", []):
        if "source" in cell:
            source = cell["source"]
            new_source = []
            if isinstance(source, list):
                for line in source:
                    new_source.append(process_text(line))
            elif isinstance(source, str):
                new_source = process_text(source)
            cell["source"] = new_source

    if changes > 0:
        print(f"  Made {changes} replacements.")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                nb, f, indent=1
            )  # Notebooks usually have indent 1 or 2. I'll use 1 to minimize diff noise or just dump.
    else:
        print("  No changes made.")


if __name__ == "__main__":
    for nb in notebooks:
        update_notebook(nb)
